from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat, vdn_measure_compat


def _measure_nodes():
    required = os.getenv("SPECTRUM_REQUIRE_REVIEWED_BSA_MEASURE_FIXTURE") == "1"
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        if required:
            pytest.fail(f"required measure-capable core BSA fixture failed to import: {exc}")
        pytest.skip(f"measure-capable core BSA fixture unavailable: {exc}")
    if core_bsa_compat._module_blob_sha(nodes) not in core_bsa_compat.MEASURE_CAPABLE_BSA_GIT_BLOBS:
        if required:
            pytest.fail("required measure-capable core BSA source is not loaded")
        pytest.skip("this ComfyUI fixture is not the reviewed measure-capable BSA source")
    if core_bsa_compat._module_blob_sha(nodes.measure) not in core_bsa_compat.AUDITED_BSA_MEASURE_GIT_BLOBS:
        if required:
            pytest.fail("required optional-VDN-correct measure adapter is not loaded")
        pytest.skip("this ComfyUI fixture predates the optional-VDN measure adapter")
    return nodes


def _vdn_modules():
    required = os.getenv("SPECTRUM_REQUIRE_REVIEWED_BSA_MEASURE_FIXTURE") == "1"
    try:
        from vdn_h3 import hybrid
        import vdn_h3.mixed_measure_epilogue as epilogue
    except Exception as exc:  # noqa: BLE001
        if required:
            pytest.fail(f"required reviewed VDN fixture failed to import: {exc}")
        pytest.skip(f"reviewed VDN fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(hybrid)
        not in vdn_measure_compat.AUDITED_VDN_HYBRID_GIT_BLOBS
        or core_bsa_compat._module_blob_sha(epilogue)
        not in vdn_measure_compat.AUDITED_VDN_EPILOGUE_GIT_BLOBS
    ):
        if required:
            pytest.fail("required reviewed VDN epilogue source is not loaded")
        pytest.skip("this VDN fixture is not the reviewed epilogue source")
    return hybrid, epilogue


class _Attention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 128
        self.qkv_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16, device="cpu")
        )
        self.out_proj = torch.nn.Identity()

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


def _request():
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
        "source_grid": [4, 8],
        "prefix_grid": [8, 8],
        "segments": [
            {"start": 0, "stop": 96, "mass_num": 1, "mass_den": 1},
            {"start": 96, "stop": 160, "mass_num": 1, "mass_den": 2},
            {"start": 160, "stop": 192, "mass_num": 1, "mass_den": 1},
        ],
    }


def _external():
    return {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 160,
        "sequence_rows": 192,
        "video_start": 96,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 32,
        "prefix_rows_per_frame": 64,
    }


def _install(monkeypatch, *, count=2, with_vdn=False):
    nodes = _measure_nodes()
    model = _Model(count)
    state = None
    if with_vdn:
        hybrid, _epilogue = _vdn_modules()
        branches = [SimpleNamespace() for _ in range(count)]
        state = SimpleNamespace(
            name="spectrum-vdn-fixture",
            cfg={"enable_softmax_gate": True},
            branches=branches,
            managed_weights=None,
            layout=None,
        )
        for index, block in enumerate(model.blocks):
            block.attn.forward = hybrid.make_vdn_forward(block.attn, state, index)

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
        "vdn_h3_external_sequence_v1": _external(),
    }
    nodes.measure.register(patch, options)
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    monkeypatch.setattr(nodes.measure, "supports_key_bias", lambda _provider: True)
    return model, state, patch, options


def test_api2_mixed_geometry_without_vdn_remains_forecast_safe(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, with_vdn=False)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)

    assert reason is None
    assert audit is not None and audit.safe and audit.measure is not None
    assert audit.measure.vdn.active is False
    assert vdn_measure_compat.identity(audit.measure.vdn) == ("absent",)
    assert all(
        vdn_measure_compat.VDN_EPILOGUE_KEY not in receipt[8]
        for receipt in audit.expected_receipts
    )

    prepared = core_bsa_compat.instrument_actual_options(
        {**options, "attention_backend_receipts_v1": []},
        audit,
        "attention_backend_receipts_v1",
    )
    assert vdn_measure_compat.VDN_EPILOGUE_RECEIPTS_KEY not in prepared


def test_reviewed_vdn_owner_enters_identity_and_sparse_receipt(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, with_vdn=True)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)

    assert reason is None
    assert audit is not None and audit.safe and audit.measure is not None
    vdn = audit.measure.vdn
    assert vdn.active is True
    assert vdn.hybrid_blob in vdn_measure_compat.AUDITED_VDN_HYBRID_GIT_BLOBS
    assert vdn.epilogue_blob in vdn_measure_compat.AUDITED_VDN_EPILOGUE_GIT_BLOBS
    assert all(vdn_measure_compat.runtime_matches(vdn, index) for index in range(2))
    assert any(item[0] == "vdn_epilogue" for item in core_bsa_compat._measure_identity(audit.measure))

    measure_receipt = audit.expected_receipts[0][8]
    vdn_receipt = measure_receipt[-2]
    assert vdn_receipt[0] == vdn_measure_compat.VDN_EPILOGUE_KEY
    assert dict(vdn_receipt[1:])["vdn_completed"] is True
    assert measure_receipt[-1] is True

    prepared = core_bsa_compat.instrument_actual_options(
        {**options, "attention_backend_receipts_v1": []},
        audit,
        "attention_backend_receipts_v1",
    )
    sink = prepared[vdn_measure_compat.VDN_EPILOGUE_RECEIPTS_KEY]
    assert sink is audit.vdn_receipts
    assert sink == []

    for index, (route, _sink, _sink_q) in enumerate(audit.route_specs):
        before = core_bsa_compat._vdn_receipt_before(audit)
        expected = vdn_measure_compat.expected_receipt(vdn, index, route)
        assert expected is not None
        sink.append((index, expected))
        ok, fields = core_bsa_compat._vdn_receipt_after(audit, index, route, before)
        assert ok and fields == expected
    receipts = tuple(
        core_bsa_compat._receipt(
            audit,
            index,
            route,
            completed=True,
            vdn_fields=vdn_measure_compat.expected_receipt(vdn, index, route),
        )
        for index, (route, _sink, _sink_q) in enumerate(audit.route_specs)
    )
    assert core_bsa_compat.accepts_actual(audit, receipts)


def test_vdn_config_change_after_preflight_invalidates_runtime_owner(monkeypatch):
    model, state, _patch, options = _install(monkeypatch, count=1, with_vdn=True)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is not None

    state.cfg["enable_softmax_gate"] = False
    assert not vdn_measure_compat.runtime_matches(audit.measure.vdn, 0)
    assert not core_bsa_compat._measure_runtime_matches(
        audit, 0, options, sparse_selected=True
    )


def test_partial_vdn_owner_is_fail_closed(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, count=2, with_vdn=True)
    model.blocks[1].attn.forward = _Attention().forward
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "vdn_owner_incomplete"


def test_vdn_receipt_sink_conflict_is_fail_closed(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, count=1, with_vdn=True)
    options[vdn_measure_compat.VDN_EPILOGUE_RECEIPTS_KEY] = []
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "vdn_receipt_ownership_conflict"


def test_missing_duplicate_and_wrong_vdn_completion_receipts_are_rejected(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, count=1, with_vdn=True)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is not None
    vdn = audit.measure.vdn
    route = audit.route_specs[0][0]
    expected = vdn_measure_compat.expected_receipt(vdn, 0, route)
    assert expected is not None

    provider_receipt = core_bsa_compat._receipt(
        audit, 0, route, completed=True, vdn_fields=expected
    )
    audit.vdn_receipts = []
    assert not core_bsa_compat.accepts_actual(audit, (provider_receipt,))

    audit.vdn_receipts[:] = [(0, expected), (0, expected)]
    assert not core_bsa_compat.accepts_actual(audit, (provider_receipt,))

    tampered = tuple(
        (name, False if name == "vdn_completed" else value)
        for name, value in expected
    )
    audit.vdn_receipts[:] = [(0, tampered)]
    assert not core_bsa_compat.accepts_actual(audit, (provider_receipt,))


def test_dense_vdn_route_requires_no_external_epilogue_receipt(monkeypatch):
    model, _state, _patch, options = _install(monkeypatch, count=1, with_vdn=True)
    options["sigmas"] = torch.tensor([1.0])
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is not None
    assert audit.route_specs[0][0] == "h3_dense"
    assert vdn_measure_compat.expected_receipt(
        audit.measure.vdn, 0, "h3_dense"
    ) is None

    prepared = core_bsa_compat.instrument_actual_options(
        {**options, "attention_backend_receipts_v1": []},
        audit,
        "attention_backend_receipts_v1",
    )
    assert prepared[vdn_measure_compat.VDN_EPILOGUE_RECEIPTS_KEY] == []
    assert vdn_measure_compat.validate_final_receipts(
        audit.measure.vdn, audit.route_specs, audit.vdn_receipts
    )
