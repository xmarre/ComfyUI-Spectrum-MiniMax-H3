"""Bridge the reviewed Untwist attention preprocessor around core H3 BSA.

Untwist v0.2.4 deliberately composes with an existing MiniMax-H3 attention
provider through ``attention_preprocess_v1``. In the production stack this puts
Untwist above core BlockSparseAttention at model-call time while BSA still owns
the H3 block replacements. The core-BSA auditor therefore cannot require BSA to
remain the top-level ``optimized_attention_override``.

This module does not accept arbitrary preprocess contracts. It recognizes only
the reviewed Untwist source and requires Untwist's Spectrum runtime descriptor,
then temporarily exposes the underlying BSA override to the existing exact
source/closure audit. Unknown preprocessors remain actual-only.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

from . import core_bsa_compat

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
    factory = getattr(module, _UNTWIST_FACTORY, None)
    expected_code = core_bsa_compat._nested_code(factory, "preprocess")
    if expected_code is None or getattr(base, "__code__", None) is not expected_code:
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


def probe(options: dict[str, Any], layout: Any, model: Any):
    """Prove core BSA ownership through one reviewed active Untwist wrapper."""
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
        return core_bsa_compat.probe(options, layout, model)

    normalized = dict(options)
    normalized["optimized_attention_override"] = bsa_override
    audit, reason = core_bsa_compat.probe(normalized, layout, model)
    if audit is None:
        return None, reason

    # The underlying BSA source, block replacements, patch owner and route are now
    # proven by the existing auditor. Add only the stable reviewed Untwist owner to
    # backend history, then restore the actual call-time provider so every actual
    # block receipt still fails closed if ownership changes during the forward.
    audit.identity = (*audit.identity, ("outer_preprocess", preprocess_identity))
    audit.current_override = real_override
    return audit, None
