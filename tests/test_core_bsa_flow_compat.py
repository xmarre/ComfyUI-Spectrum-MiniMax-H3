from types import SimpleNamespace
import types

import pytest
import torch

from comfyui_spectrum_h3 import (
    core_bsa_compat,
    core_bsa_flow_compat,
    core_bsa_preprocess_compat,
)
from comfyui_spectrum_h3.backend_history import BackendHistory, prepare, observe


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
        pytest.skip(f"core BSA fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(nodes)
        not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS
    ):
        pytest.skip("ComfyUI fixture is not the reviewed core BSA source")
    return nodes


def _flow_modules():
    try:
        from h3_flow_regenerate import attention, mixed_grid
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Flow fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(attention)
        not in core_bsa_flow_compat.AUDITED_FLOW_ATTENTION_GIT_BLOBS
    ):
        pytest.skip("Flow attention fixture is not the reviewed source")
    if (
        core_bsa_compat._module_blob_sha(mixed_grid)
        not in core_bsa_flow_compat.AUDITED_FLOW_MIXED_GRID_GIT_BLOBS
    ):
        pytest.skip("Flow mixed-grid fixture is not the reviewed source")
    return attention, mixed_grid


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


def _mixed_layout():
    return SimpleNamespace(
        seq_len=160,
        signature=("h3_flow_mixed_grid_v1", 64, 3, 8, 8, 16, 1, 8, 8),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, 160, "video"),
        ],
    )


def _installation(*, count=2, sigma=1.0):
    nodes = _audited_nodes()
    model = _FakeModel(count)
    patch = nodes.SparseAttnPatch(
        tau=1.3,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=64,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    replacements = {
        ("double_block", index): nodes.make_h3_block_patch(block, index, patch)
        for index, block in enumerate(model.blocks)
    }
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {"dit": replacements},
        "sigmas": torch.tensor([sigma]),
        "uuids": ("positive",),
    }
    return model, patch, override, options


def _wrap_layout(options, *, index=0):
    attention, _mixed = _flow_modules()
    from h3_flow_regenerate.metrics import H3FlowMetrics

    out = dict(options)
    patches = dict(options["patches_replace"])
    dit = dict(patches["dit"])
    previous = dit[("double_block", index)]
    metrics = H3FlowMetrics()
    wrapper = attention.make_layout_block_wrapper(index, metrics, previous)
    wrapper = attention.mark_layout_wrapper(
        wrapper,
        metrics=metrics,
        previous=previous,
        scope="layout",
    )
    dit[("double_block", index)] = wrapper
    patches["dit"] = dit
    out["patches_replace"] = patches
    return out, wrapper


def _cell(value):
    def close():
        return value

    return close.__closure__[0]


def _mixed_wrapper(previous, index, carrier, mixed, model):
    _attention, mixed_module = _flow_modules()
    from h3_flow_regenerate.metrics import H3FlowMetrics

    plan = mixed_module.MixedGridPlan(
        prefix=torch.zeros(1, 24, 1, 8, 8),
        temporal=3,
        source_h=4,
        source_w=4,
        prefix_noise=torch.zeros(1, 24, 1, 8, 8),
        attention_measure=False,
    )
    code = core_bsa_compat._nested_code(mixed_module.mixed_diffusion_wrapper, "call")
    assert code is not None
    values = {name: None for name in code.co_freevars}
    values.update(
        cached={},
        inner=model,
        layer=index,
        layout=carrier,
        measure_contract=None,
        metrics=H3FlowMetrics(),
        mixed_layout=mixed,
        native=None,
        old_prefix=8,
        plan=plan,
        positions=torch.empty(0),
        previous=previous,
        va=96,
        vb=128,
    )
    wrapper = types.FunctionType(
        code,
        mixed_module.__dict__,
        name=code.co_name,
        closure=tuple(_cell(values[name]) for name in code.co_freevars),
    )
    return wrapper, plan


def _wrap_mixed(options, model, *, include_all=True):
    carrier = _layout()
    mixed = _mixed_layout()
    out = dict(options)
    patches = dict(options["patches_replace"])
    dit = dict(patches["dit"])
    plan = None
    limit = len(model.blocks) if include_all else len(model.blocks) - 1
    for index in range(limit):
        wrapper, current_plan = _mixed_wrapper(
            dit[("double_block", index)], index, carrier, mixed, model
        )
        if plan is None:
            plan = current_plan
        else:
            # Real mixed_diffusion_wrapper shares one plan object across every
            # generated layer closure. Rebind the synthetic closure accordingly.
            closure = core_bsa_compat._closure_values(wrapper)
            values = dict(closure)
            values["plan"] = plan
            wrapper = types.FunctionType(
                wrapper.__code__,
                wrapper.__globals__,
                name=wrapper.__name__,
                closure=tuple(_cell(values[name]) for name in wrapper.__code__.co_freevars),
            )
        dit[("double_block", index)] = wrapper
    patches["dit"] = dit
    out["patches_replace"] = patches
    return out, carrier, mixed


def _actual_args(options, layout, seq_len=128):
    call_options = dict(options)
    call_options["minimax_h3_layout"] = layout
    return {
        "img": torch.zeros(seq_len, 4, dtype=torch.bfloat16),
        "rope_freqs": torch.zeros(seq_len, 1),
        "transformer_options": call_options,
    }


def test_reviewed_flow_layout_wrapper_is_unwrapped_to_bsa():
    model, _patch, _override, options = _installation()
    options, wrapper = _wrap_layout(options)
    audit, reason = core_bsa_flow_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.safe
    assert audit.flow_mixed is False
    assert audit.flow_outer_replacements[0] is wrapper
    assert audit.route_specs[0][0] == "h3_dense"


def test_unknown_outer_block_wrapper_stays_fail_closed():
    model, _patch, _override, options = _installation()
    previous = options["patches_replace"]["dit"][("double_block", 0)]

    def foreign(args, extra):
        return previous(args, extra)

    options["patches_replace"]["dit"][("double_block", 0)] = foreign
    audit, reason = core_bsa_flow_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "flow_wrapper_unreviewed"


def test_reviewed_mixed_grid_wrapper_uses_effective_sequence_and_zero_sinks():
    model, _patch, _override, options = _installation()
    options, carrier, _mixed = _wrap_mixed(options, model)
    audit, reason = core_bsa_flow_compat.probe(options, carrier, model)
    assert reason is None and audit is not None and audit.safe
    assert audit.flow_mixed is True
    assert audit.seq_len == 160
    assert audit.flow_carrier_seq_len == 128
    assert audit.flow_outer_seq_lens == (128, 160)
    assert all(spec == ("h3_dense", (0, 0), (0, 0)) for spec in audit.route_specs)


def test_mixed_grid_wrapper_must_cover_every_main_block():
    model, _patch, _override, options = _installation()
    options, carrier, _mixed = _wrap_mixed(options, model, include_all=False)
    audit, reason = core_bsa_flow_compat.probe(options, carrier, model)
    assert audit is None
    assert reason == "flow_mixed_wrapper_incomplete"


def test_pool_version_classifier_distinguishes_dense_and_primed():
    kmean = torch.zeros(2, 128)
    vscale = torch.zeros(2, 128)
    before = ("present", kmean, vscale, kmean._version, vscale._version)
    assert core_bsa_flow_compat._classify_snapshot(before, before) == "h3_dense"
    kmean.copy_(torch.ones_like(kmean))
    vscale.copy_(torch.ones_like(vscale))
    after = ("present", kmean, vscale, kmean._version, vscale._version)
    assert (
        core_bsa_flow_compat._classify_snapshot(before, after)
        == "h3_chunked_sparse_primed"
    )


def test_flow_prepare_observe_round_trip_accepts_dense_route():
    model, _patch, _override, options = _installation(count=1)
    options, _wrapper = _wrap_layout(options)
    layout = _layout()
    runtime = _BackendRuntime()

    prepared, pending = prepare(runtime, 9, 0, options, layout, model)
    assert pending is not None
    args = _actual_args(prepared, layout)
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    wrapped(
        args,
        {"original_block": lambda call_args: {"img": call_args["img"]}},
    )
    observe(runtime, 9, 0, prepared, pending)

    assert runtime.observed is not None
    assert runtime.observed[-1] is True
    receipts = runtime.observed[-2]
    assert len(receipts) == 1
    assert receipts[0][4] == "h3_dense"


def test_untwist_plus_flow_wrapper_reaches_core_bsa_audit():
    model, _patch, bsa_override, options = _installation()
    options, _wrapper = _wrap_layout(options)
    try:
        from flux_untwist import patches as untwist_patches
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Untwist fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(untwist_patches)
        not in core_bsa_preprocess_compat.AUDITED_UNTWIST_GIT_BLOBS
    ):
        pytest.skip("Untwist fixture is not the reviewed source")
    outer = untwist_patches.make_minimax_h3_attention_override(bsa_override)
    options["optimized_attention_override"] = outer
    options["minimax_h3_untwist_rope"] = {"enabled": True, "progress": 0.5}
    options["spectrum_h3_visual_reference_patch_runtime"] = (
        {
            "schema_version": 2,
            "provider": "comfyui-flux2-untwisting-rope",
            "instance_id": "untwist-h3-1",
            "schedule_progress": 0.5,
            "active": True,
        },
    )

    audit, reason = core_bsa_preprocess_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.safe
    assert audit.current_override is outer
    assert hasattr(audit, "flow_wrapper_specs")
