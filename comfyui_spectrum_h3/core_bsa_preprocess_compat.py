"""Bridge the reviewed Untwist attention preprocessor around core H3 BSA.

Untwist v0.2.4 deliberately composes with an existing MiniMax-H3 attention
provider through ``attention_preprocess_v1``. In the production stack this puts
Untwist above core BlockSparseAttention at model-call time while BSA can itself
sit below reviewed Flow block wrappers. The auditor therefore proves both
wrapper layers before accepting backend history.

This module does not accept arbitrary preprocess contracts. It recognizes only
the reviewed Untwist source and requires Untwist's Spectrum runtime descriptor,
then temporarily exposes the underlying BSA override to the source-gated
BSA/Flow audit. Unknown preprocessors remain actual-only.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

from . import core_bsa_compat, core_bsa_flow_compat, source_code_audit

AUDITED_UNTWIST_GIT_BLOBS = frozenset(
    {"49a6eeda841a9dffe52974dceb3bce78bf02f25d"}
)
_UNTWIST_TOPLEVEL_MODULE = "flux_untwist.patches"
_UNTWIST_MODULE_SUFFIX = ".flux_untwist.patches"
_UNTWIST_FACTORY = "make_minimax_h3_attention_override"
_UNTWIST_PREPROCESS_QUALNAME = (
    "make_minimax_h3_attention_override.<locals>.preprocess"
)
_UNTWIST_PROVIDER = "comfyui-flux2-untwisting-rope"
_UNTWIST_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
_UNTWIST_CONFIG_KEY = "minimax_h3_untwist_rope"
_MISSING = object()


def _loaded_untwist_module(base: Any) -> Any | None:
    """Resolve the reviewed source module under either test or ComfyUI package naming."""
    module_name = getattr(base, "__module__", None)
    if not isinstance(module_name, str):
        return None
    if module_name != _UNTWIST_TOPLEVEL_MODULE and not module_name.endswith(
        _UNTWIST_MODULE_SUFFIX
    ):
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
        or source_path.name != "patches.py"
        or source_path.parent.name != "flux_untwist"
    ):
        return None
    return module


def _audited_untwist_preprocess(transform: Any) -> tuple[Any, ...] | None:
    """Return a stable identity only for the exact reviewed Untwist preprocessor."""
    base = getattr(transform, "__func__", transform)
    if getattr(base, "__qualname__", None) != _UNTWIST_PREPROCESS_QUALNAME:
        return None
    module = _loaded_untwist_module(base)
    if module is None:
        return None
    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in AUDITED_UNTWIST_GIT_BLOBS:
        return None
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source_code_audit.matches_nested_source_code(
        base,
        Path(source),
        (_UNTWIST_FACTORY, "preprocess"),
    ):
        return None
    if getattr(base, "__defaults__", None) is not None:
        return None
    if getattr(base, "__kwdefaults__", None):
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure != {}:
        return None
    return ("untwist_h3_attention_preprocess_v1", blob)


def _untwist_runtime_identity(options: dict[str, Any]) -> tuple[Any, ...] | None:
    """Require the runtime descriptor Spectrum already uses to track Untwist."""
    raw = options.get(_UNTWIST_RUNTIME_KEY)
    if not isinstance(raw, (tuple, list)):
        return None
    active = []
    for value in raw:
        if not isinstance(value, dict) or value.get("provider") != _UNTWIST_PROVIDER:
            continue
        if value.get("active") is not True:
            continue
        schema = value.get("schema_version")
        instance_id = value.get("instance_id")
        progress = value.get("schedule_progress")
        if (
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema not in {1, 2}
            or not isinstance(instance_id, str)
            or not instance_id
            or isinstance(progress, bool)
            or not isinstance(progress, (int, float))
            or not 0.0 <= float(progress) <= 1.0
        ):
            return None
        active.append((schema, instance_id))
    if len(active) != 1:
        return None

    cfg = options.get(_UNTWIST_CONFIG_KEY)
    if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
        return None
    # Progress-dependent scaling is already represented by Spectrum's dedicated
    # external-patch runtime contract. Keeping only the stable instance here avoids
    # turning every smooth schedule point into a backend-history discontinuity.
    return (_UNTWIST_PROVIDER, active[0][0], active[0][1])


def _unwrap_reviewed_untwist(
    options: dict[str, Any],
) -> tuple[Any | None, tuple[Any, ...] | None, str | None]:
    current = options.get("optimized_attention_override")
    if core_bsa_compat._looks_like_core_bsa_callable(
        current, "make_attention_override", "override"
    ):
        return current, None, None
    if current is None:
        return None, None, "ownership_unproven"

    contract = getattr(current, "attention_preprocess_v1", _MISSING)
    if not isinstance(contract, tuple) or len(contract) != 2:
        return None, None, "ownership_unproven"
    transform, previous = contract
    preprocess_identity = _audited_untwist_preprocess(transform)
    if preprocess_identity is None:
        return None, None, "untwist_preprocess_unreviewed"
    runtime_identity = _untwist_runtime_identity(options)
    if runtime_identity is None:
        return None, None, "untwist_runtime_unproven"
    if not core_bsa_compat._looks_like_core_bsa_callable(
        previous, "make_attention_override", "override"
    ):
        return None, None, "ownership_unproven"
    return previous, (preprocess_identity, runtime_identity), None


def _stabilize_flow_audit_identity(audit: Any) -> bool:
    """Remove per-call object generations from reviewed dynamic Flow wrappers.

    Mixed-Grid creates fresh wrapper functions, layout objects and plan objects on
    each actual model invocation. Those addresses/generations prove the current
    call's ownership but are not numerical backend semantics. Retaining them in
    backend history would force a reset on every model call and suppress every
    forecast. Exact source/code/closure proof remains in ``flow_wrapper_specs``
    for the immediate receipt instrumentation; the retained policy uses only the
    reviewed source, wrapper role/layer and structural carrier/mixed layouts.
    """
    raw = getattr(audit, "flow_identity", None)
    if raw is None:
        return True
    if not isinstance(raw, tuple) or len(raw) != 5:
        return False
    if raw[0] != "reviewed_flow_h3_wrapper_chain_v1":
        return False
    mode, wrapper_specs, carrier_entry, mixed_entry = raw[1:]
    if not isinstance(wrapper_specs, tuple):
        return False

    stable_blocks = []
    for block in wrapper_specs:
        if not isinstance(block, tuple):
            return False
        stable_chain = []
        for item in block:
            if not isinstance(item, tuple) or not item:
                return False
            if item[0] == "flow_layout_wrapper" and len(item) == 6:
                stable_chain.append(item[:4])
            elif item[0] == "flow_mixed_grid_wrapper" and len(item) == 8:
                stable_chain.append((item[0], item[1], item[2], item[6], item[7]))
            else:
                return False
        stable_blocks.append(tuple(stable_chain))

    if (
        not isinstance(carrier_entry, tuple)
        or len(carrier_entry) != 2
        or carrier_entry[0] != "carrier_layout"
        or not isinstance(mixed_entry, tuple)
        or len(mixed_entry) != 2
        or mixed_entry[0] != "mixed_layout"
    ):
        return False
    if mixed_entry[1] is None:
        stable_mixed = mixed_entry
    else:
        payload = mixed_entry[1]
        if not isinstance(payload, tuple) or len(payload) != 3:
            return False
        stable_mixed = ("mixed_layout", payload[2])

    stable = (
        raw[0],
        mode,
        tuple(stable_blocks),
        carrier_entry,
        stable_mixed,
    )
    expected_tail = ("outer_block_wrappers", raw)
    identity = getattr(audit, "identity", None)
    if not isinstance(identity, tuple) or not identity or identity[-1] != expected_tail:
        return False
    audit.identity = (*identity[:-1], ("outer_block_wrappers", stable))
    audit.flow_identity = stable
    return True


def _probe_flow(options: dict[str, Any], layout: Any, model: Any):
    audit, reason = core_bsa_flow_compat.probe(options, layout, model)
    if audit is None:
        return None, reason
    if not _stabilize_flow_audit_identity(audit):
        return None, "flow_identity_unproven"
    return audit, None


def probe(options: dict[str, Any], layout: Any, model: Any):
    """Prove core BSA through reviewed Untwist and Flow composition."""
    try:
        real_override = options.get("optimized_attention_override")
        bsa_override, preprocess_identity, reason = _unwrap_reviewed_untwist(options)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - optional-provider introspection fails closed
        return None, "untwist_introspection_failed"
    if bsa_override is None:
        return None, reason
    if preprocess_identity is None:
        return _probe_flow(options, layout, model)

    normalized = dict(options)
    normalized["optimized_attention_override"] = bsa_override
    audit, reason = _probe_flow(normalized, layout, model)
    if audit is None:
        return None, reason

    # Underlying BSA/Flow ownership and numerical route are now proven. Add only
    # the stable reviewed Untwist owner, then restore the real call-time provider
    # so receipt instrumentation still rejects a mid-forward owner change.
    audit.identity = (*audit.identity, ("outer_preprocess", preprocess_identity))
    audit.current_override = real_override
    return audit, None
