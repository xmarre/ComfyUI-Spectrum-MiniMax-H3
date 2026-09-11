from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat


def _measure_nodes():
    required = os.getenv("SPECTRUM_REQUIRE_REVIEWED_BSA_MEASURE_FIXTURE") == "1"
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        if required:
            pytest.fail(f"required measure-capable core BSA fixture failed to import: {exc}")
        pytest.skip(f"measure-capable core BSA fixture unavailable: {exc}")
    blob = core_bsa_compat._module_blob_sha(nodes)
    if blob not in core_bsa_compat.MEASURE_CAPABLE_BSA_GIT_BLOBS:
        if required:
            pytest.fail(f"required measure-capable core BSA blob not loaded: {blob}")
        pytest.skip("this ComfyUI fixture is not the reviewed measure-capable BSA source")
    return nodes


class _Attention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 128
        self.qkv_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16, device="cpu")
        )

    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x


class _Block:
    def __init__(self):
        self.attn = _Attention()


class _Model:
    def __init__(self, count=2):
        self.blocks = [_Block() for _ in range(count)]
        self.dtype = torch.bfloat16


def _layout():
    return SimpleNamespace(
        seq_len=192,
        signature=(64, 2, 16, 16, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, 192, "video"),
        ],
    )


def _request(*, source_grid=(4, 8), prefix_grid=(8, 8)):
    source_rows = source_grid[0] * source_grid[1]
    prefix_rows = prefix_grid[0] * prefix_grid[1]
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
        "q_rows": 192,
        "kv_rows": 192,
        "video_start": 96,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": list(source_grid),
        "prefix_grid": list(prefix_grid),
        "segments": [
            {"start": 0, "stop": 96, "mass_num": 1, "mass_den": 1},
            {
                "start": 96,
                "stop": 96 + prefix_rows,
                "mass_num": source_rows,
                "mass_den": prefix_rows,
            },
            {
                "start": 96 + prefix_rows,
                "stop": 192,
                "mass_num": 1,
                "mass_den": 1,
            },
        ],
    }


def _installation(monkeypatch, *, count=2):
    nodes = _measure_nodes()
    model = _Model(count)
    patch = nodes.SparseAttnPatch(
        tau=1.0,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=0,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {
            "dit": {
                ("double_block", index): nodes.make_h3_block_patch(block, index, patch)
                for index, block in enumerate(model.blocks)
            }
        },
        "sigmas": torch.tensor([0.5]),
        "uuids": ("positive",),
        "minimax_h3_layout": _layout(),
        core_bsa_compat.ATTENTION_MEASURE_KEY: _request(),
    }
    nodes.measure.register(patch, options)
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    monkeypatch.setattr(nodes.measure, "supports_key_bias", lambda _provider: True)
    return nodes, model, patch, options


def test_measure_fixture_is_source_pinned_when_required():
    nodes = _measure_nodes()
    assert core_bsa_compat._module_blob_sha(nodes) in core_bsa_compat.MEASURE_CAPABLE_BSA_GIT_BLOBS
    assert core_bsa_compat._module_blob_sha(nodes.measure) in core_bsa_compat.AUDITED_BSA_MEASURE_GIT_BLOBS
    import comfy.attention_measure as core_measure
    assert core_bsa_compat._module_blob_sha(core_measure) in core_bsa_compat.AUDITED_ATTENTION_MEASURE_GIT_BLOBS


def test_weighted_measure_changes_exact_sink_and_uses_measure_bound_pool(monkeypatch):
    _nodes, model, patch, options = _installation(monkeypatch)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None
    assert audit is not None and audit.safe and audit.measure is not None
    assert audit.measure.semantic_digest
    assert audit.measure.exact_k_block_range == (0, 3)
    assert {spec[0] for spec in audit.route_specs} == {"h3_chunked_sparse_cold"}
    assert {spec[1] for spec in audit.route_specs} == {(0, 3)}
    assert len(audit.expected_receipts[0]) == 9
    measure_receipt = audit.expected_receipts[0][8]
    assert measure_receipt[0] == core_bsa_compat.ATTENTION_MEASURE_KEY
    assert measure_receipt[2] == 0
    assert measure_receipt[5] == core_bsa_compat.SPARSE_MEASURE_PROFILE
    assert measure_receipt[6] == core_bsa_compat.SPARSE_MEASURE_ROUTE
    assert measure_receipt[-1] is True

    shape = (model.blocks[0].attn.heads, model.blocks[0].attn.head_dim)
    for index in range(len(model.blocks)):
        key = audit.measure.pool_keys[index]
        assert key != (index, audit.seq_len, audit.uuids)
        patch.pooled[key] = (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    primed, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and primed is not None and primed.safe
    assert {spec[0] for spec in primed.route_specs} == {"h3_chunked_sparse_primed"}


def test_measure_digest_and_pool_key_change_for_anisotropic_grid_identity(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch)
    first, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and first is not None and first.measure is not None

    changed = dict(options)
    changed[core_bsa_compat.ATTENTION_MEASURE_KEY] = _request(
        source_grid=(8, 4), prefix_grid=(8, 8)
    )
    second, reason = core_bsa_compat.probe(changed, _layout(), model)
    assert reason is None and second is not None and second.measure is not None
    assert first.measure.semantic_digest != second.measure.semantic_digest
    assert first.measure.pool_keys != second.measure.pool_keys
    assert first.identity != second.identity


def test_foreign_measure_capability_fails_closed(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch)
    registry = dict(options[core_bsa_compat.ATTENTION_MEASURE_CAPABILITIES_KEY])
    registry[core_bsa_compat.CORE_BSA_MEASURE_PROVIDER] = object()
    options[core_bsa_compat.ATTENTION_MEASURE_CAPABILITIES_KEY] = registry
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "measure_capability_unproven"


def test_missing_chunked_key_bias_is_actual_only_for_sparse_route(monkeypatch):
    nodes, model, _patch, options = _installation(monkeypatch)
    monkeypatch.setattr(nodes.measure, "supports_key_bias", lambda _provider: False)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None
    assert not audit.safe
    assert {spec[0] for spec in audit.route_specs} == {"h3_sparse_unproven"}


def test_measure_disappearance_between_preflight_and_actual_is_rejected(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch, count=1)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is not None
    call_options = dict(options)
    call_options.pop(core_bsa_compat.ATTENTION_MEASURE_KEY)
    assert not core_bsa_compat._measure_runtime_matches(
        audit,
        0,
        call_options,
        sparse_selected=True,
    )


def test_unweighted_fixture_keeps_legacy_eight_field_receipt(monkeypatch):
    nodes = _measure_nodes()
    model = _Model(1)
    patch = nodes.SparseAttnPatch(
        tau=1.0,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=0,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {
            "dit": {("double_block", 0): nodes.make_h3_block_patch(model.blocks[0], 0, patch)}
        },
        "sigmas": torch.tensor([0.5]),
        "uuids": ("positive",),
    }
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is None
    assert len(audit.expected_receipts[0]) == 8
