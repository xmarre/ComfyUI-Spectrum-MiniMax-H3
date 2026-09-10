"""Source-gated Flow wrapper composition around ComfyUI core H3 BSA.

Flow-Aligned Regenerate legitimately wraps MiniMax-H3 ``double_block`` replacements.
Core BSA can therefore be numerically active without remaining the top-level block
replacement. This module recognizes only reviewed Flow wrapper code, unwraps it
for BSA preflight, and verifies the actual route around the real outer wrapper
chain without depending on inference-tensor version counters.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch

from . import core_bsa_compat, source_code_audit

AUDITED_FLOW_ATTENTION_GIT_BLOBS = frozenset(
    {"c58c652f7b7d6030a2311f4805c3443615e432eb"}
)
AUDITED_FLOW_MIXED_GRID_GIT_BLOBS = frozenset(
    {
        "8fc0f753ff2cd21fae898a4dd3c9ab1025f98443",  # released v0.3.3
        "66c59f26ad41154fcce7e1ce5250fa233a96dc3b",  # PR #26 canonical layout propagation
    }
)
_FLOW_MIXED_LAYOUT_PROPAGATED_BLOBS = frozenset(
    {"66c59f26ad41154fcce7e1ce5250fa233a96dc3b"}
)
_FLOW_ATTENTION_TOPLEVEL = "h3_flow_regenerate.attention"
_FLOW_ATTENTION_SUFFIX = ".h3_flow_regenerate.attention"
_FLOW_MIXED_TOPLEVEL = "h3_flow_regenerate.mixed_grid"
_FLOW_MIXED_SUFFIX = ".h3_flow_regenerate.mixed_grid"
_LAYOUT_QUALNAME = "make_layout_block_wrapper.<locals>.wrapper"
_MIXED_QUALNAME = "mixed_diffusion_wrapper.<locals>.wrap.<locals>.call"
_LAYOUT_CLOSURE_SCHEMA = frozenset({"layer", "metrics", "previous", "record_layout"})
_MIXED_CLOSURE_SCHEMA = frozenset(
    {
        "cached",
        "inner",
        "layer",
        "layout",
        "measure_contract",
        "metrics",
        "mixed_layout",
        "native",
        "old_prefix",
        "plan",
        "positions",
        "previous",
        "va",
        "vb",
    }
)


def _loaded_source_module(
    base: Any,
    *,
    top_level: str,
    suffix: str,
    parent: str,
    filename: str,
    blobs: frozenset[str],
) -> tuple[Any, str] | None:
    module_name = getattr(base, "__module__", None)
    if not isinstance(module_name, str):
        return None
    if module_name != top_level and not module_name.endswith(suffix):
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    source = getattr(module, "__file__", None)
    code = getattr(base, "__code__", None)
    if not isinstance(source, str) or code is None:
        return None
    try:
        source_path = Path(source).resolve()
        code_path = Path(code.co_filename).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if (
        source_path != code_path
        or source_path.name != filename
        or source_path.parent.name != parent
    ):
        return None
    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in blobs:
        return None
    return module, blob


def _source_matches(base: Any, module: Any, lexical_path: tuple[str, ...]) -> bool:
    source = getattr(module, "__file__", None)
    return isinstance(source, str) and source_code_audit.matches_nested_source_code(
        base,
        Path(source),
        lexical_path,
    )


def _audited_layout_wrapper(
    wrapper: Any, index: int
) -> tuple[Any, tuple[Any, ...]] | None:
    base = getattr(wrapper, "__func__", wrapper)
    if getattr(base, "__qualname__", None) != _LAYOUT_QUALNAME:
        return None
    loaded = _loaded_source_module(
        base,
        top_level=_FLOW_ATTENTION_TOPLEVEL,
        suffix=_FLOW_ATTENTION_SUFFIX,
        parent="h3_flow_regenerate",
        filename="attention.py",
        blobs=AUDITED_FLOW_ATTENTION_GIT_BLOBS,
    )
    if loaded is None:
        return None
    module, blob = loaded
    if not _source_matches(base, module, ("make_layout_block_wrapper", "wrapper")):
        return None
    if getattr(base, "__defaults__", None) is not None or getattr(base, "__kwdefaults__", None):
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure is None or frozenset(closure) != _LAYOUT_CLOSURE_SCHEMA:
        return None
    if type(closure["layer"]) is not int or closure["layer"] != index:
        return None
    if type(closure["record_layout"]) is not bool:
        return None
    previous = closure["previous"]
    if getattr(wrapper, "_h3_flow_layout_wrapper", False) is not True:
        return None
    if getattr(wrapper, "_h3_flow_previous", None) is not previous:
        return None
    scope = getattr(wrapper, "_h3_flow_layout_scope", None)
    if scope not in {"layout", "attention"}:
        return None
    metrics = closure["metrics"]
    if getattr(wrapper, "_h3_flow_metrics", None) is not metrics:
        return None
    if not callable(getattr(metrics, "increment", None)) or not callable(
        getattr(metrics, "event", None)
    ):
        return None
    return previous, (
        "flow_layout_wrapper",
        blob,
        scope,
        index,
        core_bsa_compat._callable_identity(wrapper),
        core_bsa_compat._lifetime_generation(metrics),
    )


def _plan_geometry(plan: Any) -> dict[str, Any] | None:
    try:
        prefix = plan.prefix
        prefix_noise = plan.prefix_noise
        temporal = plan.temporal
        source_h = plan.source_h
        source_w = plan.source_w
        attention_measure = plan.attention_measure
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if not torch.is_tensor(prefix) or prefix.ndim != 5:
        return None
    if not torch.is_tensor(prefix_noise) or tuple(prefix_noise.shape) != tuple(prefix.shape):
        return None
    if (
        type(temporal) is not int
        or type(source_h) is not int
        or type(source_w) is not int
        or type(attention_measure) is not bool
    ):
        return None
    prefix_t = int(prefix.shape[2])
    target_h, target_w = map(int, prefix.shape[-2:])
    if (
        temporal <= prefix_t
        or prefix_t <= 0
        or min(source_h, source_w, target_h, target_w) < 2
        or any(value % 2 for value in (source_h, source_w, target_h, target_w))
    ):
        return None
    source_rows = source_h * source_w // 4
    target_rows = target_h * target_w // 4
    if source_rows <= 0 or target_rows <= source_rows:
        return None
    mixed_rows = prefix_t * target_rows + (temporal - prefix_t) * source_rows
    return {
        "temporal": temporal,
        "prefix_t": prefix_t,
        "source_h": source_h,
        "source_w": source_w,
        "target_h": target_h,
        "target_w": target_w,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "mixed_rows": mixed_rows,
        "attention_measure": attention_measure,
    }


def _expected_measure_contract(
    geometry: dict[str, Any], *, video_start: int, sequence_rows: int
) -> dict[str, Any] | None:
    if not geometry["attention_measure"]:
        return None
    return {
        "api": 1,
        "mode": "prefix_kv_stratified_subsample",
        "video_start": video_start,
        "sequence_rows": sequence_rows,
        "temporal": geometry["temporal"],
        "prefix_t": geometry["prefix_t"],
        "source_grid_h": geometry["source_h"] // 2,
        "source_grid_w": geometry["source_w"] // 2,
        "prefix_grid_h": geometry["target_h"] // 2,
        "prefix_grid_w": geometry["target_w"] // 2,
        "source_rows_per_frame": geometry["source_rows"],
        "prefix_rows_per_frame": geometry["target_rows"],
        "expected_kv_rows": video_start
        + geometry["temporal"] * geometry["source_rows"],
        "exact_prefix_queries_preserved": True,
        "suffix_kv_unchanged": True,
    }


def _audited_mixed_wrapper(
    wrapper: Any,
    index: int,
    carrier_layout: Any,
    model: Any,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]] | None:
    base = getattr(wrapper, "__func__", wrapper)
    if getattr(base, "__qualname__", None) != _MIXED_QUALNAME:
        return None
    loaded = _loaded_source_module(
        base,
        top_level=_FLOW_MIXED_TOPLEVEL,
        suffix=_FLOW_MIXED_SUFFIX,
        parent="h3_flow_regenerate",
        filename="mixed_grid.py",
        blobs=AUDITED_FLOW_MIXED_GRID_GIT_BLOBS,
    )
    if loaded is None:
        return None
    module, blob = loaded
    if not _source_matches(
        base,
        module,
        ("mixed_diffusion_wrapper", "wrap", "call"),
    ):
        return None
    if getattr(base, "__defaults__", None) is not None or getattr(base, "__kwdefaults__", None):
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure is None or frozenset(closure) != _MIXED_CLOSURE_SCHEMA:
        return None
    if type(closure["layer"]) is not int or closure["layer"] != index:
        return None
    if closure["inner"] is not model:
        return None
    native = closure["native"]
    if (
        getattr(native, "__name__", None) != "comfy.ldm.minimax.model"
        or sys.modules.get("comfy.ldm.minimax.model") is not native
    ):
        return None
    cached = closure["cached"]
    if type(cached) is not dict or cached:
        return None
    previous = closure["previous"]
    mixed_layout = closure["mixed_layout"]
    plan = closure["plan"]
    closed_carrier = closure["layout"]
    if type(plan) is not getattr(module, "MixedGridPlan", object):
        return None
    geometry = _plan_geometry(plan)
    if geometry is None:
        return None
    normalized_carrier = core_bsa_compat._normalize_layout(carrier_layout)
    normalized_closed_carrier = core_bsa_compat._normalize_layout(closed_carrier)
    normalized_mixed = core_bsa_compat._normalize_layout(mixed_layout)
    if (
        normalized_carrier is None
        or normalized_closed_carrier != normalized_carrier
        or normalized_mixed is None
    ):
        return None
    carrier_seq, carrier_identity = normalized_carrier
    mixed_seq, mixed_identity = normalized_mixed
    if mixed_seq <= carrier_seq:
        return None
    carrier_segments = dict(carrier_identity).get("segments")
    mixed_segments = dict(mixed_identity).get("segments")
    if not isinstance(carrier_segments, tuple) or not isinstance(mixed_segments, tuple):
        return None
    carrier_video = [item for item in carrier_segments if item[2] == "video"]
    mixed_video = [item for item in mixed_segments if item[2] == "video"]
    if len(carrier_video) != 1 or len(mixed_video) != 1:
        return None
    va, vb, _kind = carrier_video[0]
    mixed_va, mixed_vb, _mixed_kind = mixed_video[0]
    if (
        va != mixed_va
        or vb != carrier_seq
        or mixed_vb != mixed_seq
        or mixed_seq != va + geometry["mixed_rows"]
    ):
        return None
    if tuple(item for item in carrier_segments if item[2] != "video") != tuple(
        item for item in mixed_segments if item[2] != "video"
    ):
        return None
    if closure["va"] != va or closure["vb"] != vb:
        return None
    if closure["old_prefix"] != geometry["prefix_t"] * geometry["source_rows"]:
        return None
    positions = closure["positions"]
    if not torch.is_tensor(positions) or getattr(mixed_layout, "position_ids", None) is not positions:
        return None
    if int(positions.shape[0]) != mixed_seq:
        return None
    expected_measure = _expected_measure_contract(
        geometry,
        video_start=va,
        sequence_rows=mixed_seq,
    )
    if closure["measure_contract"] != expected_measure:
        return None
    metrics = closure["metrics"]
    if not callable(getattr(metrics, "increment", None)) or not callable(
        getattr(metrics, "event", None)
    ):
        return None
    plan_generation = core_bsa_compat._lifetime_generation(plan)
    mixed_generation = core_bsa_compat._lifetime_generation(mixed_layout)
    identity = (
        "flow_mixed_grid_wrapper",
        blob,
        index,
        core_bsa_compat._callable_identity(wrapper),
        plan_generation,
        mixed_generation,
        carrier_identity,
        mixed_identity,
    )
    return previous, identity, {
        "source_blob": blob,
        "plan": plan,
        "plan_generation": plan_generation,
        "mixed_layout": mixed_layout,
        "mixed_generation": mixed_generation,
        "mixed_seq": mixed_seq,
        "mixed_identity": mixed_identity,
        "carrier_seq": carrier_seq,
        "carrier_identity": carrier_identity,
        "cached": cached,
        "metrics": metrics,
        "positions": positions,
        "native": native,
        "inner": model,
        "measure_contract": expected_measure,
        "old_prefix": closure["old_prefix"],
        "va": va,
        "vb": vb,
    }


def _unwrap_replacement(
    replacement: Any,
    index: int,
    carrier_layout: Any,
    model: Any,
) -> tuple[Any | None, tuple[Any, ...], dict[str, Any] | None, str | None]:
    current = replacement
    identities: list[Any] = []
    mixed: dict[str, Any] | None = None
    seen: set[int] = set()
    for _depth in range(8):
        if core_bsa_compat._looks_like_core_bsa_callable(
            current, "make_h3_block_patch", "block_patch"
        ):
            return current, tuple(identities), mixed, None
        if current is None or id(current) in seen:
            return None, tuple(identities), mixed, "flow_wrapper_chain_invalid"
        seen.add(id(current))

        layout = _audited_layout_wrapper(current, index)
        if layout is not None:
            current, identity = layout
            identities.append(identity)
            continue

        mixed_result = _audited_mixed_wrapper(current, index, carrier_layout, model)
        if mixed_result is not None:
            if mixed is not None:
                return None, tuple(identities), mixed, "flow_mixed_wrapper_stacked"
            current, identity, mixed = mixed_result
            identities.append(identity)
            continue

        return None, tuple(identities), mixed, "flow_wrapper_unreviewed"
    return None, tuple(identities), mixed, "flow_wrapper_chain_too_deep"


def has_core_bsa_evidence(options: dict[str, Any]) -> bool:
    if core_bsa_compat.has_core_bsa_evidence(options):
        return True
    patches = options.get("patches_replace")
    if not isinstance(patches, dict):
        return False
    dit = patches.get("dit")
    if not isinstance(dit, dict):
        return False
    for replacement in dit.values():
        current = replacement
        seen: set[int] = set()
        for _depth in range(8):
            if core_bsa_compat._looks_like_core_bsa_callable(
                current, "make_h3_block_patch", "block_patch"
            ):
                return True
            if current is None or id(current) in seen:
                break
            seen.add(id(current))
            if getattr(current, "_h3_flow_layout_wrapper", False) is True:
                current = getattr(current, "_h3_flow_previous", None)
                continue
            closure = core_bsa_compat._closure_values(current)
            if (
                getattr(current, "__qualname__", None) == _MIXED_QUALNAME
                and isinstance(closure, dict)
            ):
                current = closure.get("previous")
                continue
            break
    return False


def _effective_mixed_layout(mixed: dict[str, Any], carrier_layout: Any) -> Any:
    """Return the exact layout core BSA sees for the reviewed Flow source.

    Released Flow v0.3.3 passes ``mixed_layout`` as the direct block argument but
    leaves ``transformer_options['minimax_h3_layout']`` on the carrier layout.
    Core BSA therefore sees a token/layout mismatch and uses zero conditioning
    sinks. PR #26 fixes the producer-side contract and publishes ``mixed_layout``
    through the canonical transformer metadata as well. The audit preserves both
    numerical semantics instead of pretending the released and fixed sources are
    equivalent.
    """
    source_blob = mixed.get("source_blob")
    if source_blob in _FLOW_MIXED_LAYOUT_PROPAGATED_BLOBS:
        layout = mixed.get("mixed_layout")
        normalized = core_bsa_compat._normalize_layout(layout)
        if normalized is None or normalized[0] != int(mixed["mixed_seq"]):
            raise ValueError("propagated mixed layout is invalid")
        return layout

    carrier = core_bsa_compat._normalize_layout(carrier_layout)
    if carrier is None:
        raise ValueError("carrier layout is invalid")
    mixed_seq = int(mixed["mixed_seq"])
    return SimpleNamespace(
        seq_len=mixed_seq,
        signature=("flow_mixed_bsa_effective_v1", carrier[1], mixed["mixed_identity"]),
        segments=[(0, mixed_seq, "flow_mixed_effective")],
    )


def probe(options: dict[str, Any], layout: Any, model: Any):
    """Prove reviewed Flow wrappers and then delegate the underlying BSA audit."""
    try:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, (tuple, list, torch.nn.ModuleList)) or not blocks:
            return None, "flow_model_shape_unproven"
        patches = options.get("patches_replace")
        if not isinstance(patches, dict):
            return core_bsa_compat.probe(options, layout, model)
        dit = patches.get("dit")
        if not isinstance(dit, dict):
            return core_bsa_compat.probe(options, layout, model)

        normalized_dit = dict(dit)
        wrapper_specs: list[Any] = []
        mixed_specs: list[dict[str, Any] | None] = []
        saw_flow = False
        for index in range(len(blocks)):
            replacement = dit.get(("double_block", index))
            if replacement is None:
                return None, "main_h3_replacement_missing"
            if core_bsa_compat._looks_like_core_bsa_callable(
                replacement, "make_h3_block_patch", "block_patch"
            ):
                normalized_dit[("double_block", index)] = replacement
                wrapper_specs.append(tuple())
                mixed_specs.append(None)
                continue
            underlying, identities, mixed, reason = _unwrap_replacement(
                replacement, index, layout, model
            )
            if underlying is None:
                return None, reason or "flow_wrapper_unreviewed"
            saw_flow = True
            normalized_dit[("double_block", index)] = underlying
            wrapper_specs.append(identities)
            mixed_specs.append(mixed)

        if not saw_flow:
            return core_bsa_compat.probe(options, layout, model)

        mixed_present = [item is not None for item in mixed_specs]
        if any(mixed_present) and not all(mixed_present):
            return None, "flow_mixed_wrapper_incomplete"
        mixed = mixed_specs[0] if all(mixed_present) else None
        if mixed is not None:
            for item in mixed_specs[1:]:
                if (
                    item is None
                    or item["plan"] is not mixed["plan"]
                    or item["mixed_layout"] is not mixed["mixed_layout"]
                    or item["source_blob"] != mixed["source_blob"]
                    or item["carrier_identity"] != mixed["carrier_identity"]
                    or item["cached"] is not mixed["cached"]
                    or item["metrics"] is not mixed["metrics"]
                    or item["positions"] is not mixed["positions"]
                    or item["native"] is not mixed["native"]
                    or item["inner"] is not mixed["inner"]
                    or item["measure_contract"] != mixed["measure_contract"]
                    or item["old_prefix"] != mixed["old_prefix"]
                    or item["va"] != mixed["va"]
                    or item["vb"] != mixed["vb"]
                ):
                    return None, "flow_mixed_wrapper_inconsistent"

        normalized = dict(options)
        normalized_patches = dict(patches)
        normalized_patches["dit"] = normalized_dit
        normalized["patches_replace"] = normalized_patches
        audit_layout = _effective_mixed_layout(mixed, layout) if mixed is not None else layout
        audit, reason = core_bsa_compat.probe(normalized, audit_layout, model)
        if audit is None:
            return None, reason

        carrier = core_bsa_compat._normalize_layout(layout)
        if carrier is None:
            return None, "flow_carrier_layout_unproven"
        carrier_seq, carrier_identity = carrier
        if mixed is None:
            outer_seq_lens = tuple(carrier_seq for _ in blocks)
            flow_mode = "layout_wrapped"
        else:
            outer_seq_lens = (carrier_seq, *(mixed["mixed_seq"] for _ in blocks[1:]))
            flow_mode = (
                "mixed_grid_layout_propagated_v1"
                if mixed["source_blob"] in _FLOW_MIXED_LAYOUT_PROPAGATED_BLOBS
                else "mixed_grid_v0.3.3"
            )

        flow_identity = (
            "reviewed_flow_h3_wrapper_chain_v1",
            flow_mode,
            tuple(wrapper_specs),
            ("carrier_layout", carrier_identity),
            (
                "mixed_layout",
                None
                if mixed is None
                else (
                    mixed["plan_generation"],
                    mixed["mixed_generation"],
                    mixed["mixed_identity"],
                ),
            ),
        )
        audit.identity = (*audit.identity, ("outer_block_wrappers", flow_identity))
        audit.flow_wrapper_specs = tuple(wrapper_specs)
        audit.flow_outer_replacements = tuple(
            dit[("double_block", index)] for index in range(len(blocks))
        )
        audit.flow_bsa_replacements = tuple(
            normalized_dit[("double_block", index)] for index in range(len(blocks))
        )
        audit.flow_outer_seq_lens = outer_seq_lens
        audit.flow_carrier_seq_len = carrier_seq
        audit.flow_carrier_layout_identity = carrier_identity
        audit.flow_mixed = mixed is not None
        audit.flow_mixed_layout_propagated = bool(
            mixed is not None
            and mixed["source_blob"] in _FLOW_MIXED_LAYOUT_PROPAGATED_BLOBS
        )
        audit.flow_identity = flow_identity
        audit.flow_model = model
        return audit, None
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - optional Flow introspection remains actual-only
        return None, "flow_introspection_failed"


def _pool_entry(
    audit: core_bsa_compat.CoreBSAAudit, index: int
) -> tuple[Any, ...]:
    return core_bsa_compat._pool_entry(
        audit.patch,
        (index, audit.seq_len, audit.uuids),
        audit.pool_specs[index],
    )


def _current_metadata_matches(
    audit: core_bsa_compat.CoreBSAAudit,
    index: int,
    args: dict[str, Any],
) -> bool:
    try:
        hidden = args["img"]
        call_options = args["transformer_options"]
    except (KeyError, TypeError):
        return False
    if (
        not torch.is_tensor(hidden)
        or hidden.ndim < 1
        or int(hidden.shape[0]) != int(audit.flow_outer_seq_lens[index])
    ):
        return False
    if core_bsa_compat._normalize_uuids(call_options) != audit.uuids:
        return False
    if core_bsa_compat._settings_identity(audit.patch) != audit.settings_identity:
        return False
    if call_options.get("optimized_attention_override") is not audit.current_override:
        return False
    actual_layout = core_bsa_compat._normalize_layout(
        call_options.get("minimax_h3_layout")
    )
    if (
        actual_layout is None
        or actual_layout[0] != audit.flow_carrier_seq_len
        or actual_layout[1] != audit.flow_carrier_layout_identity
    ):
        return False
    sigma = core_bsa_compat._sigma_value(call_options)
    if sigma is None:
        return False
    sigma_start = audit.settings_identity[4]
    sigma_end = audit.settings_identity[5]
    min_tokens = audit.settings_identity[6]
    dense_blocks = audit.settings_identity[7]
    dense = (
        not (sigma_end <= sigma <= sigma_start)
        or audit.seq_len < min_tokens
        or index in dense_blocks
    )
    expected = audit.route_specs[index][0]
    if expected == "h3_dense":
        return dense
    if expected not in {"h3_chunked_sparse_cold", "h3_chunked_sparse_primed"}:
        return False
    return not dense


def _make_actual_wrapper(
    audit: core_bsa_compat.CoreBSAAudit,
    index: int,
    replacement: Any,
    bsa_replacement: Any,
    receipts: list[Any],
):
    bsa_closure = core_bsa_compat._closure_values(bsa_replacement)
    expected_attention = (
        bsa_closure.get("attention") if isinstance(bsa_closure, dict) else None
    )

    def audited_replacement(args, replacement_context):
        try:
            metadata_ok = _current_metadata_matches(audit, index, args)
            before = _pool_entry(audit, index)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - observation remains fail-closed
            metadata_ok = False
            before = ("invalid",)
            audit.failure = "actual_metadata_failed"

        sparse_selected = False
        original_block_calls = 0
        try:
            original_block = replacement_context.get("original_block")
        except Exception:  # noqa: BLE001 - malformed observation context fails closed
            original_block = None
        if not callable(expected_attention) or not callable(original_block):
            metadata_ok = False
            output = replacement(args, replacement_context)
        else:
            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)

            observed_context = dict(replacement_context)
            observed_context["original_block"] = audited_original_block
            output = replacement(args, observed_context)
            if original_block_calls != 1:
                metadata_ok = False

        try:
            after = _pool_entry(audit, index)
            observed = core_bsa_compat._classify_pool_transition(
                before,
                after,
                sparse_selected=sparse_selected,
            )
            expected_route, expected_sink, expected_sink_q = audit.route_specs[index]
            if not metadata_ok or observed != expected_route:
                audit.failure = "actual_route_mismatch"
            receipts.append(
                (
                    core_bsa_compat.ADAPTER_KEY,
                    core_bsa_compat.ADAPTER_VERSION,
                    audit.patch_generation,
                    index,
                    observed,
                    audit.seq_len,
                    expected_sink,
                    expected_sink_q,
                )
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - observation failure must not abort sampling
            audit.failure = "actual_observation_failed"
        return output

    return audited_replacement


def instrument_actual_options(
    options: dict[str, Any],
    audit: core_bsa_compat.CoreBSAAudit,
    receipts_key: str,
) -> dict[str, Any]:
    if not hasattr(audit, "flow_wrapper_specs"):
        return core_bsa_compat.instrument_actual_options(options, audit, receipts_key)
    prepared = dict(options)
    receipts = prepared.get(receipts_key)
    patches = prepared.get("patches_replace")
    if not isinstance(receipts, list) or not isinstance(patches, dict):
        audit.failure = "flow_receipt_or_replacement_table_missing"
        return prepared
    dit = patches.get("dit")
    if not isinstance(dit, dict):
        audit.failure = "dit_replacement_table_missing"
        return prepared
    bsa_replacements = getattr(audit, "flow_bsa_replacements", None)
    if not isinstance(bsa_replacements, tuple) or len(bsa_replacements) != audit.block_count:
        audit.failure = "flow_bsa_replacement_evidence_missing"
        return prepared
    flow_model = getattr(audit, "flow_model", None)
    if flow_model is None:
        audit.failure = "flow_model_evidence_missing"
        return prepared

    local_patches = dict(patches)
    local_dit = dict(dit)
    for index in range(audit.block_count):
        key = ("double_block", index)
        replacement = local_dit.get(key)
        if replacement is None or replacement is not audit.flow_outer_replacements[index]:
            audit.failure = "flow_outer_replacement_changed"
            return prepared
        underlying, identities, mixed, reason = _unwrap_replacement(
            replacement,
            index,
            SimpleNamespace(
                seq_len=audit.flow_carrier_seq_len,
                signature=dict(audit.flow_carrier_layout_identity).get("signature"),
                segments=list(dict(audit.flow_carrier_layout_identity).get("segments", ())),
            ),
            flow_model,
        )
        if (
            underlying is None
            or underlying is not bsa_replacements[index]
            or identities != audit.flow_wrapper_specs[index]
        ):
            audit.failure = reason or "flow_wrapper_chain_changed"
            return prepared
        if bool(mixed is not None) != bool(audit.flow_mixed):
            audit.failure = "flow_wrapper_mode_changed"
            return prepared
        if (
            mixed is not None
            and bool(mixed["source_blob"] in _FLOW_MIXED_LAYOUT_PROPAGATED_BLOBS)
            != bool(getattr(audit, "flow_mixed_layout_propagated", False))
        ):
            audit.failure = "flow_mixed_layout_mode_changed"
            return prepared
        local_dit[key] = _make_actual_wrapper(
            audit,
            index,
            replacement,
            underlying,
            receipts,
        )
    local_patches["dit"] = local_dit
    prepared["patches_replace"] = local_patches
    prepared[core_bsa_compat.PRIVATE_AUDIT_KEY] = audit
    return prepared
