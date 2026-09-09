from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat, core_bsa_preprocess_compat
from comfyui_spectrum_h3.backend_history import RECEIPTS, preflight


class _FakeAttention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 128
        self.qkv_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16, device="cpu")
        )

    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttention()


class _FakeModel:
    def __init__(self, count=2):
        self.blocks = [_FakeBlock() for _ in range(count)]
        self.dtype = torch.bfloat16


def _audited_nodes():
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core BSA is unavailable in this reviewed ComfyUI fixture: {exc}")
    if core_bsa_compat._module_blob_sha(nodes) not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        pytest.skip("this ComfyUI fixture is not the reviewed core BSA source")
    return nodes


def _layout(seq_len=128):
    return SimpleNamespace(
        seq_len=seq_len,
        signature=(64, 1, 8, 8, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, seq_len, "video"),
        ],
    )


def _installation(*, count=2, sigma=1.0):
    nodes = _audited_nodes()
    model = _FakeModel(count)
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
        "sigmas": torch.tensor([sigma]),
        "uuids": ("positive",),
    }
    return model, patch, override, options


def _make_recreated_preprocess(previous):
    def preprocess(q, k, v, heads, **kwargs):
        return q, k, v

    def override(original, q, k, v, heads, *args, **kwargs):
        q, k, v = preprocess(q, k, v, heads, **kwargs)
        return previous(original, q, k, v, heads, *args, **kwargs)

    override.attention_preprocess_v1 = (preprocess, previous)
    return override


def _actual_args(options, layout, seq_len=128):
    call_options = dict(options)
    call_options["minimax_h3_layout"] = layout
    return {
        "img": torch.zeros(seq_len, 4, dtype=torch.bfloat16),
        "rope_freqs": torch.zeros(seq_len, 1),
        "transformer_options": call_options,
    }


def test_recreated_preprocess_wrapper_keeps_stable_backend_identity():
    model, _patch, bsa_override, options = _installation()
    first_options = dict(options)
    first_outer = _make_recreated_preprocess(bsa_override)
    first_options["optimized_attention_override"] = first_outer
    first, reason = core_bsa_preprocess_compat.probe(first_options, _layout(), model)
    assert reason is None and first is not None and first.safe
    assert first.current_override is first_outer

    second_options = dict(options)
    second_outer = _make_recreated_preprocess(bsa_override)
    assert second_outer is not first_outer
    second_options["optimized_attention_override"] = second_outer
    second, reason = core_bsa_preprocess_compat.probe(second_options, _layout(), model)
    assert reason is None and second is not None and second.safe
    assert second.current_override is second_outer
    assert first.identity == second.identity


def test_backend_history_uses_preprocess_aware_core_bsa_probe():
    model, _patch, bsa_override, options = _installation()
    options["optimized_attention_override"] = _make_recreated_preprocess(bsa_override)
    identity, safe = preflight(options, _layout(), model)
    assert safe
    assert identity[0] == core_bsa_compat.ADAPTER_KEY
    assert identity[-1][0] == "outer_preprocess"


def test_uncontracted_outer_attention_stays_fail_closed():
    model, _patch, bsa_override, options = _installation()

    def outer(*args, **kwargs):
        return bsa_override(*args, **kwargs)

    options["optimized_attention_override"] = outer
    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "outer_attention_uncontracted"


def test_preprocess_with_unstable_closure_stays_fail_closed():
    model, _patch, bsa_override, options = _installation()
    mutable_owner = object()

    def preprocess(q, k, v, heads, **kwargs):
        if mutable_owner is None:
            raise AssertionError
        return q, k, v

    def outer(original, q, k, v, heads, *args, **kwargs):
        q, k, v = preprocess(q, k, v, heads, **kwargs)
        return bsa_override(original, q, k, v, heads, *args, **kwargs)

    outer.attention_preprocess_v1 = (preprocess, bsa_override)
    options["optimized_attention_override"] = outer
    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "outer_preprocess_identity_unproven"


def test_dense_actual_receipt_accepts_real_outer_preprocess_owner():
    model, _patch, bsa_override, options = _installation(count=1)
    outer = _make_recreated_preprocess(bsa_override)
    options["optimized_attention_override"] = outer
    layout = _layout()
    audit, reason = core_bsa_preprocess_compat.probe(options, layout, model)
    assert reason is None and audit is not None and audit.safe
    assert audit.route_specs[0][0] == "h3_dense"

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    args = _actual_args(prepared, layout)
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    output = wrapped(
        args,
        {"original_block": lambda call_args: {"img": call_args["img"]}},
    )
    assert "img" in output
    assert audit.failure is None
    assert core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))


def test_midforward_outer_preprocess_swap_invalidates_receipts():
    model, _patch, bsa_override, options = _installation(count=2)
    outer = _make_recreated_preprocess(bsa_override)
    options["optimized_attention_override"] = outer
    layout = _layout()
    audit, reason = core_bsa_preprocess_compat.probe(options, layout, model)
    assert reason is None and audit is not None

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    args = _actual_args(prepared, layout)
    context = {"original_block": lambda call_args: {"img": call_args["img"]}}

    first = prepared["patches_replace"]["dit"][("double_block", 0)]
    first(args, context)
    assert audit.failure is None

    args["transformer_options"]["optimized_attention_override"] = _make_recreated_preprocess(
        bsa_override
    )
    second = prepared["patches_replace"]["dit"][("double_block", 1)]
    second(args, context)
    assert audit.failure == "actual_route_mismatch"
    assert not core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))
