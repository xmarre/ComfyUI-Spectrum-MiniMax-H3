"""Source-gated Flow wrapper composition around ComfyUI core H3 BSA.

Flow-Aligned Regenerate legitimately wraps MiniMax-H3 ``double_block`` replacements.
Core BSA can therefore be numerically active without remaining the top-level block
replacement. This module recognizes only the reviewed Flow v0.3.3 wrapper code,
unwraps it for BSA preflight, and verifies the actual route from BSA pooled-state
transitions around the real outer wrapper chain.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import torch

from . import core_bsa_compat

AUDITED_FLOW_ATTENTION_GIT_BLOBS = frozenset(
    {"c58c652f7b7d6030a2311f4805c3443615e432eb"}
)
AUDITED_FLOW_MIXED_GRID_GIT_BLOBS = frozenset(
    {"8fc0f753ff2cd21fae898a4dd3c9ab1025f98443"}
)
_FLOW_ATTENTION_TOPLEVEL = "h3_flow_regenerate.attention"
_FLOW_ATTENTION_SUFFIX = ".h3_flow_regenerate.attention"
_FLOW_MIXED_TOPLEVEL = "h3_flow_regenerate.mixed_grid"
_FLOW_MIXED_SUFFIX = ".h3_flow_regenerate.mixed_grid"
_LAYOUT_QUALNAME = "make_layout_block_wrapper.<locals>.wrapper"
_MIXED_QUALNAME = "mixed_diffusion_wrapper.<locals>.wrap.<locals>.call"


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
    expected = core_bsa_compat._nested_code(
        getattr(module, "make_layout_block_wrapper", None), "wrapper"
    )
    if expected is None or getattr(base, "__code__", None) is not expected:
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure is None or closure.get("layer") != index:
        return None
    previous = closure.get("previous")
    if getattr(wrapper, "_h3_flow_layout_wrapper", False) is not True:
        return None
    if getattr(wrapper, "_h3_flow_previous", None) is not previous:
        return None
    scope = getattr(wrapper, "_h3_flow_layout_scope", None)
    if scope not in {"layout", "attention"}:
        return None
    metrics = closure.get("metrics")
    if getattr(wrapper, "_h3_flow_metrics", None) is not metrics:
        return None
    return previous, (
        "flow_layout_wrapper",
        blob,
        scope,
        index,
        core_bsa_compat._callable_identity(wrapper),
        core_bsa_compat._lifetime_generation(metrics),
    )


def _audited_mixed_wrapper(
    wrapper: Any,
    index: int,
    carrier_layout: Any,
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
    expected = core_bsa_compat._nested_code(
        getattr(module, "mixed_diffusion_wrapper", None), "call"
    )
    if expected is None or getattr(base, "__code__", None) is not expected:
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure is None or closure.get("layer") != index:
        return None
    previous = closure.get("previous")
    mixed_layout = closure.get("mixed_layout")
    plan = closure.get("plan")
    closed_carrier = closure.get("layout")
    if type(plan) is not getattr(module, "MixedGridPlan", object):
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
    if carrier_video[0][0] != mixed_video[0][0]:
        return None
    if tuple(item for item in carrier_segments if item[2] != "video") != tuple(
        item for item in mixed_segments if item[2] != "video"
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
        "plan": plan,
        "plan_generation": plan_generation,
        "mixed_layout": mixed_layout,
        "mixed_generation": mixed_generation,
        "mixed_seq": mixed_seq,
        "mixed_identity": mixed_identity,
        "carrier_seq": carrier_seq,
        "carrier_identity": carrier_identity,
    }


def _unwrap_replacement(
    replacement: Any,
    index: int,
    carrier_layout: Any,
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

        mixed_result = _audited_mixed_wrapper(current, index, carrier_layout)
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
    """Mirror reviewed v0.3.3 BSA semantics inside Mixed-Grid.

    Flow v0.3.3 passes the mixed layout as a direct block argument but leaves
    ``transformer_options['minimax_h3_layout']`` on the carrier layout. Core BSA
    therefore sees a token/layout length mismatch and uses zero conditioning
    sinks. The opaque effective layout below reproduces that exact numerical
    route for preflight; both real carrier and mixed identities are retained
    separately in the policy identity.
    """
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
                replacement, index, layout
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
                    or item["carrier_identity"] != mixed["carrier_identity"]
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
            flow_mode = "mixed_grid_v0.3.3"

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
        audit.flow_outer_seq_lens = outer_seq_lens
        audit.flow_carrier_seq_len = carrier_seq
        audit.flow_carrier_layout_identity = carrier_identity
        audit.flow_mixed = mixed is not None
        audit.flow_identity = flow_identity
        return audit, None
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - optional Flow introspection remains actual-only
        return None, "flow_introspection_failed"


def _pool_snapshot(
    audit: core_bsa_compat.CoreBSAAudit, index: int
) -> tuple[Any, ...]:
    key = (index, audit.seq_len, audit.uuids)
    state = core_bsa_compat._pool_entry(
        audit.patch, key, audit.pool_specs[index]
    )
    if state[0] != "present":
        return state
    return (
        "present",
        state[1],
        state[2],
        int(state[1]._version),
        int(state[2]._version),
    )


def _classify_snapshot(before: tuple[Any, ...], after: tuple[Any, ...]) -> str:
    if before[0] == "missing":
        if after[0] == "missing":
            return "h3_dense"
        if after[0] == "present":
            return "h3_chunked_sparse_cold"
        return "unknown"
    if before[0] != "present" or after[0] != "present":
        return "unknown"
    if before[1] is not after[1] or before[2] is not after[2]:
        return "unknown"
    k_delta = int(after[3]) - int(before[3])
    v_delta = int(after[4]) - int(before[4])
    if k_delta == 0 and v_delta == 0:
        return "h3_dense"
    if k_delta > 0 and v_delta > 0:
        return "h3_chunked_sparse_primed"
    return "unknown"


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
    receipts: list[Any],
):
    def audited_replacement(args, replacement_context):
        try:
            metadata_ok = _current_metadata_matches(audit, index, args)
            before = _pool_snapshot(audit, index)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - observation remains fail-closed
            metadata_ok = False
            before = ("invalid",)
            audit.failure = "actual_metadata_failed"

        output = replacement(args, replacement_context)

        try:
            after = _pool_snapshot(audit, index)
            observed = _classify_snapshot(before, after)
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
        )
        if underlying is None or identities != audit.flow_wrapper_specs[index]:
            audit.failure = reason or "flow_wrapper_chain_changed"
            return prepared
        if bool(mixed is not None) != bool(audit.flow_mixed):
            audit.failure = "flow_wrapper_mode_changed"
            return prepared
        local_dit[key] = _make_actual_wrapper(audit, index, replacement, receipts)
    local_patches["dit"] = local_dit
    prepared["patches_replace"] = local_patches
    prepared[core_bsa_compat.PRIVATE_AUDIT_KEY] = audit
    return prepared
