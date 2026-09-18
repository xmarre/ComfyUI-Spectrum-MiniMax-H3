"""Source-gated Spectrum identity for canonical Keyless H3 Untwist routing.

Canonical Keyless H3 exposes Untwist through the public routing-preprocessor chain,
not through ``optimized_attention_override``. The Untwist producer constructs a
fresh immutable preprocessor object for every model call because its snapshot binds
the current denoising progress. Generic Spectrum callable identity would therefore
see a new owner generation on every call and reset forecast history continuously.

Progress itself is already tracked by Spectrum's visual-reference external-patch
runtime transaction. This adapter recognizes only the exact reviewed Untwist source,
proves that its immutable snapshot and declared digest agree with the active runtime
descriptor, and removes only the smooth per-call progress coordinate from backend-
history identity. All static numerical semantics (ranges, scales, RoPE geometry,
schedule window, scope, etc.) remain identity-bearing. Unknown or inconsistent
preprocessors fall back to the generic Keyless identity path and remain conservative.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from . import core_bsa_compat


_AUDITED_UNTWIST_KEYLESS_SOURCE_CONTRACTS = {
    # Untwist #11: full-domain Keyless routing preprocessor.
    "53079e70b7996b1a872d9ff1e5f219cc1f110c3e": (
        "minimax_h3_untwist_keyless_route_v1:",
        "minimax_h3_untwist_keyless_routing_preprocessor_v1",
        1,
        False,
    ),
    # Untwist #12: additive domain-aware routing-position remapping.
    "1de8f77d0c439ec073daad1a5a0928f2ad9a343e": (
        "minimax_h3_untwist_keyless_route_v2:",
        "minimax_h3_untwist_keyless_routing_preprocessor_v2",
        2,
        True,
    ),
}
AUDITED_UNTWIST_KEYLESS_GIT_BLOBS = frozenset(
    _AUDITED_UNTWIST_KEYLESS_SOURCE_CONTRACTS
)
_UNTWIST_TOPLEVEL_MODULE = "flux_untwist.keyless_h3"
_UNTWIST_MODULE_SUFFIX = ".flux_untwist.keyless_h3"
_UNTWIST_CLASS = "KeylessUntwistRoutingPreprocessor"
_UNTWIST_PROVIDER = "comfyui-flux2-untwisting-rope"
_UNTWIST_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
_REFERENCE_SCOPES = frozenset(
    {"image_only", "image_and_video", "all_visual_including_continuum"}
)


def _loaded_reviewed_module(value: Any) -> tuple[Any, str] | None:
    cls = type(value)
    module_name = getattr(cls, "__module__", None)
    if not isinstance(module_name, str):
        return None
    if module_name != _UNTWIST_TOPLEVEL_MODULE and not module_name.endswith(
        _UNTWIST_MODULE_SUFFIX
    ):
        return None
    module = sys.modules.get(module_name)
    if module is None or getattr(module, _UNTWIST_CLASS, None) is not cls:
        return None
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        return None
    try:
        source_path = Path(source).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if source_path.name != "keyless_h3.py" or source_path.parent.name != "flux_untwist":
        return None
    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in AUDITED_UNTWIST_KEYLESS_GIT_BLOBS:
        return None
    return module, blob


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _runtime_identity(
    options: dict[str, Any],
    *,
    instance_id: str,
    progress: float,
) -> tuple[Any, ...] | None:
    raw = options.get(_UNTWIST_RUNTIME_KEY)
    if not isinstance(raw, (tuple, list)):
        return None
    matches = []
    for value in raw:
        if not isinstance(value, dict):
            continue
        if value.get("provider") != _UNTWIST_PROVIDER:
            continue
        if value.get("instance_id") != instance_id:
            continue
        schema = value.get("schema_version")
        active = value.get("active")
        runtime_progress = _finite(value.get("schedule_progress"))
        if (
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema not in {1, 2}
            or active is not True
            or runtime_progress is None
            or not 0.0 <= runtime_progress <= 1.0
            or runtime_progress != progress
        ):
            return None
        matches.append((schema, instance_id))
    if len(matches) != 1:
        return None
    return (_UNTWIST_PROVIDER, matches[0][0], matches[0][1])


def _snapshot_semantics(
    snapshot: Any,
    *,
    identity_schema: str,
    identity_version: int,
) -> tuple[str, float, tuple[Any, ...], dict[str, Any]] | None:
    required = (
        "instance_id",
        "expected_rows",
        "reference_ranges",
        "reference_scope",
        "rope_axis_count",
        "rope_freqs_per_axis",
        "high_scale_start",
        "high_scale_end",
        "low_scale_start",
        "low_scale_end",
        "beta",
        "start_percent",
        "end_percent",
        "progress",
        "scale_temporal_axis",
    )
    if any(not hasattr(snapshot, name) for name in required):
        return None

    instance_id = snapshot.instance_id
    scope = snapshot.reference_scope
    if not isinstance(instance_id, str) or not instance_id or scope not in _REFERENCE_SCOPES:
        return None
    try:
        expected_rows = int(snapshot.expected_rows)
        axis_count = int(snapshot.rope_axis_count)
        freq_count = int(snapshot.rope_freqs_per_axis)
    except (TypeError, ValueError):
        return None
    if expected_rows <= 0 or axis_count <= 0 or freq_count <= 0:
        return None

    raw_ranges = snapshot.reference_ranges
    if not isinstance(raw_ranges, tuple) or not raw_ranges:
        return None
    ranges: list[tuple[int, int]] = []
    for item in raw_ranges:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        try:
            start, stop = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or stop <= start or stop > expected_rows:
            return None
        ranges.append((start, stop))

    numeric_names = (
        "high_scale_start",
        "high_scale_end",
        "low_scale_start",
        "low_scale_end",
        "beta",
        "start_percent",
        "end_percent",
        "progress",
    )
    numeric: dict[str, float] = {}
    for name in numeric_names:
        value = _finite(getattr(snapshot, name))
        if value is None:
            return None
        numeric[name] = value
    if numeric["beta"] <= 0.0:
        return None
    if not (
        0.0 <= numeric["start_percent"] <= numeric["end_percent"] <= 1.0
        and 0.0 <= numeric["progress"] <= 1.0
    ):
        return None
    temporal = snapshot.scale_temporal_axis
    if not isinstance(temporal, bool):
        return None

    static = (
        expected_rows,
        tuple(ranges),
        scope,
        axis_count,
        freq_count,
        numeric["high_scale_start"],
        numeric["high_scale_end"],
        numeric["low_scale_start"],
        numeric["low_scale_end"],
        numeric["beta"],
        numeric["start_percent"],
        numeric["end_percent"],
        temporal,
    )
    identity_payload = {
        "schema": identity_schema,
        "version": identity_version,
        "instance_id": instance_id,
        "expected_rows": expected_rows,
        "reference_ranges": [list(item) for item in ranges],
        "reference_scope": scope,
        "rope_axis_count": axis_count,
        "rope_freqs_per_axis": freq_count,
        "high_scale_start": numeric["high_scale_start"],
        "high_scale_end": numeric["high_scale_end"],
        "low_scale_start": numeric["low_scale_start"],
        "low_scale_end": numeric["low_scale_end"],
        "beta": numeric["beta"],
        "start_percent": numeric["start_percent"],
        "end_percent": numeric["end_percent"],
        "progress": numeric["progress"],
        "scale_temporal_axis": temporal,
    }
    return instance_id, numeric["progress"], static, identity_payload


def _expected_declared_identity(
    identity_payload: dict[str, Any],
    *,
    identity_prefix: str,
) -> str | None:
    try:
        encoded = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return identity_prefix + hashlib.sha256(encoded).hexdigest()


def reviewed_keyless_untwist_identity(
    value: Any,
    options: dict[str, Any],
) -> tuple[Any, ...] | None:
    """Return stable numerical identity only for the exact reviewed route preprocessor."""
    reviewed = _loaded_reviewed_module(value)
    if reviewed is None:
        return None
    _module, blob = reviewed
    source_contract = _AUDITED_UNTWIST_KEYLESS_SOURCE_CONTRACTS.get(blob)
    if source_contract is None:
        return None
    identity_prefix, identity_schema, identity_version, requires_domain_fn = source_contract
    if requires_domain_fn and not callable(getattr(value, "apply_domain", None)):
        return None

    snapshot = getattr(value, "_snapshot", None)
    parsed = _snapshot_semantics(
        snapshot,
        identity_schema=identity_schema,
        identity_version=identity_version,
    )
    if parsed is None:
        return None
    instance_id, progress, static, identity_payload = parsed

    declared = getattr(value, "identity", None)
    expected_declared = _expected_declared_identity(
        identity_payload,
        identity_prefix=identity_prefix,
    )
    if declared != expected_declared:
        return None

    runtime = _runtime_identity(
        options,
        instance_id=instance_id,
        progress=progress,
    )
    if runtime is None:
        return None

    return (
        f"reviewed_keyless_untwist_route_v{identity_version}",
        blob,
        runtime,
        static,
    )


__all__ = [
    "AUDITED_UNTWIST_KEYLESS_GIT_BLOBS",
    "reviewed_keyless_untwist_identity",
]
