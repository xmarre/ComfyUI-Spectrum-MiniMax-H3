import sys
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat, core_bsa_preprocess_compat
from comfyui_spectrum_h3.backend_history import (
    BackendHistory,
    RECEIPTS,
    observe,
    preflight,
    prepare,
)


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


class _EmptyForecaster:
    history_length = 0


class _BackendRuntime:
    def __init__(self):
        self.config = SimpleNamespace(debug=False)
        self._backend_history = BackendHistory()
        self._history_topology = None
        self._history_labels = None
        self._primary_forecaster = _EmptyForecaster()
        self._stage_forecasters = {}
        self._step = None
        self._offline_archive = None
        self._offline_smoother = None
        self.prepared = None
        self.observed = None

    def prepare_backend_history(self, run_id, step_id, identity, safe):
        self.prepared = (run_id, step_id, identity, bool(safe))

    def observe_backend_history(self, run_id, step_id, identity, receipts, safe):
        self.observed = (run_id, step_id, identity, receipts, bool(safe))


def _audited_nodes():
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core BSA is unavailable in this reviewed ComfyUI fixture: {exc}")
    if core_bsa_compat._module_blob_sha(nodes) not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        pytest.skip("this ComfyUI fixture is not the reviewed core BSA source")
    return nodes


def _untwist_factory():
    try:
        from flux_untwist import patches
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Untwist fixture is unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(patches)
        not in core_bsa_preprocess_compat.AUDITED_UNTWIST_GIT_BLOBS
    ):
        pytest.skip("this Untwist fixture is not the reviewed v0.2.4 source")
    return patches.make_minimax_h3_attention_override


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


def _with_untwist(
    options,
    bsa_override,
    *,
    progress=0.5,
    instance_id="untwist-h3-1",
    active=True,
):
    outer = _untwist_factory()(bsa_override)
    out = dict(options)
    out["optimized_attention_override"] = outer
    out["minimax_h3_untwist_rope"] = {"enabled": bool(active), "progress": float(progress)}
    out["spectrum_h3_visual_reference_patch_runtime"] = (
        {
            "schema_version": 2,
            "provider": "comfyui-flux2-untwisting-rope",
            "instance_id": instance_id,
            "schedule_progress": float(progress),
            "active": bool(active),
        },
    )
    return out, outer


def _actual_args(options, layout, seq_len=128):
    call_options = dict(options)
    call_options["minimax_h3_layout"] = layout
    return {
        "img": torch.zeros(seq_len, 4, dtype=torch.bfloat16),
        "rope_freqs": torch.zeros(seq_len, 1),
        "transformer_options": call_options,
    }


def _run_prepared_dense_block(prepared, layout):
    args = _actual_args(prepared, layout)
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    wrapped(
        args,
        {"original_block": lambda call_args: {"img": call_args["img"]}},
    )
    return args


def test_reviewed_untwist_wrapper_keeps_stable_backend_identity():
    model, _patch, bsa_override, options = _installation()
    first_options, first_outer = _with_untwist(options, bsa_override, progress=0.25)
    first, reason = core_bsa_preprocess_compat.probe(first_options, _layout(), model)
    assert reason is None and first is not None and first.safe
    assert first.current_override is first_outer

    second_options, second_outer = _with_untwist(options, bsa_override, progress=0.75)
    assert second_outer is not first_outer
    second, reason = core_bsa_preprocess_compat.probe(second_options, _layout(), model)
    assert reason is None and second is not None and second.safe
    assert second.current_override is second_outer
    # Smooth Untwist schedule progress is handled by Spectrum's external-patch
    # runtime contract, not by forcing a backend-history reset on every call.
    assert first.identity == second.identity


def test_comfy_loader_namespaced_untwist_module_is_accepted(monkeypatch):
    model, _patch, bsa_override, options = _installation()
    options, outer = _with_untwist(options, bsa_override)
    transform, _previous = outer.attention_preprocess_v1
    original_module_name = transform.__module__
    module = sys.modules[original_module_name]
    namespaced_module_name = (
        "/home/toor/ComfyUI/custom_nodes/comfyui-untwisting-rope"
        ".flux_untwist.patches"
    )
    monkeypatch.setitem(sys.modules, namespaced_module_name, module)
    monkeypatch.setattr(transform, "__module__", namespaced_module_name)

    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.safe
    assert audit.current_override is outer


def test_backend_history_uses_reviewed_untwist_aware_core_bsa_probe():
    model, _patch, bsa_override, options = _installation()
    options, _outer = _with_untwist(options, bsa_override)
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
    assert reason == "ownership_unproven"


def test_unknown_preprocess_contract_stays_fail_closed():
    model, _patch, bsa_override, options = _installation()

    def preprocess(q, k, v, heads, **kwargs):
        return q, k, v

    def outer(original, q, k, v, heads, *args, **kwargs):
        return bsa_override(original, q, k, v, heads, *args, **kwargs)

    outer.attention_preprocess_v1 = (preprocess, bsa_override)
    options["optimized_attention_override"] = outer
    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "untwist_preprocess_unreviewed"


def test_reviewed_untwist_requires_spectrum_runtime_descriptor():
    model, _patch, bsa_override, options = _installation()
    outer = _untwist_factory()(bsa_override)
    options["optimized_attention_override"] = outer
    options["minimax_h3_untwist_rope"] = {"enabled": True, "progress": 0.5}
    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "untwist_runtime_unproven"


def test_inactive_untwist_runtime_stays_fail_closed_if_wrapper_is_present():
    model, _patch, bsa_override, options = _installation()
    options, _outer = _with_untwist(options, bsa_override, active=False)
    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "untwist_runtime_unproven"


def test_untwist_instance_change_changes_backend_identity():
    model, _patch, bsa_override, options = _installation()
    first_options, _outer = _with_untwist(
        options, bsa_override, instance_id="untwist-h3-1"
    )
    first, reason = core_bsa_preprocess_compat.probe(first_options, _layout(), model)
    assert reason is None and first is not None

    second_options, _outer = _with_untwist(
        options, bsa_override, instance_id="untwist-h3-2"
    )
    second, reason = core_bsa_preprocess_compat.probe(second_options, _layout(), model)
    assert reason is None and second is not None
    assert first.identity != second.identity


def test_dense_actual_receipt_accepts_real_untwist_owner():
    model, _patch, bsa_override, options = _installation(count=1)
    options, outer = _with_untwist(options, bsa_override)
    layout = _layout()
    audit, reason = core_bsa_preprocess_compat.probe(options, layout, model)
    assert reason is None and audit is not None and audit.safe
    assert audit.current_override is outer
    assert audit.route_specs[0][0] == "h3_dense"

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    output_args = _run_prepared_dense_block(prepared, layout)
    assert output_args["img"].shape[0] == layout.seq_len
    assert audit.failure is None
    assert core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))


def test_midforward_untwist_owner_swap_invalidates_receipts():
    model, _patch, bsa_override, options = _installation(count=2)
    options, _outer = _with_untwist(options, bsa_override)
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

    replacement_options, replacement_outer = _with_untwist(options, bsa_override)
    assert replacement_outer is not audit.current_override
    args["transformer_options"]["optimized_attention_override"] = replacement_options[
        "optimized_attention_override"
    ]
    second = prepared["patches_replace"]["dit"][("double_block", 1)]
    second(args, context)
    assert audit.failure == "actual_route_mismatch"
    assert not core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))


def test_untwist_introspection_failure_is_actual_only(monkeypatch):
    def fail(_options):
        raise RuntimeError("introspection exploded")

    monkeypatch.setattr(core_bsa_preprocess_compat, "_unwrap_reviewed_untwist", fail)
    audit, reason = core_bsa_preprocess_compat.probe({}, None, None)
    assert audit is None
    assert reason == "untwist_introspection_failed"


def test_untwist_introspection_oom_propagates(monkeypatch):
    def fail(_options):
        raise torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(core_bsa_preprocess_compat, "_unwrap_reviewed_untwist", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        core_bsa_preprocess_compat.probe({}, None, None)


def test_backend_prepare_observe_round_trip_accepts_untwist_bsa():
    model, _patch, bsa_override, options = _installation(count=1)
    options, _outer = _with_untwist(options, bsa_override)
    layout = _layout()
    runtime = _BackendRuntime()

    prepared, pending = prepare(runtime, 7, 3, options, layout, model)
    assert pending is not None
    assert core_bsa_compat.PRIVATE_AUDIT_KEY in prepared
    _run_prepared_dense_block(prepared, layout)
    observe(runtime, 7, 3, prepared, pending)

    assert runtime.observed is not None
    assert runtime.observed[-1] is True
    assert len(runtime.observed[-2]) == 1


def test_backend_prepare_observe_round_trip_rejects_owner_change():
    model, _patch, bsa_override, options = _installation(count=1)
    options, _outer = _with_untwist(options, bsa_override)
    layout = _layout()
    runtime = _BackendRuntime()

    prepared, pending = prepare(runtime, 7, 4, options, layout, model)
    assert pending is not None
    args = _actual_args(prepared, layout)
    args["transformer_options"]["optimized_attention_override"] = object()
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    wrapped(
        args,
        {"original_block": lambda call_args: {"img": call_args["img"]}},
    )
    observe(runtime, 7, 4, prepared, pending)

    assert runtime.observed is not None
    assert runtime.observed[-1] is False
