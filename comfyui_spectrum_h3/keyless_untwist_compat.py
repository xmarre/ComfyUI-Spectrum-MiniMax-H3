"""Source-gated Spectrum identity for canonical Keyless H3 Untwist routing.

Canonical Keyless H3 exposes Untwist through the public routing-preprocessor chain,
not through ``optimized_attention_override``. The Untwist producer constructs a
fresh immutable preprocessor object for every model call because its snapshot binds
the current denoising progress. Generic Spectrum callable identity would therefore
see a new owner generation on every call and reset forecast history continuously.

Progress itself is already tracked by Spectrum's visual-reference external-patch
runtime transaction. This adapter recognizes only the exact reviewed Untwist source,
proves that its immutable snapshot matches the active runtime descriptor, and removes
only the smooth per-call progress coordinate from backend-history identity. All
static numerical semantics (ranges, scales, RoPE geometry, schedule window, scope,
etc.) remain identity-bearing. Unknown or inconsistent preprocessors fall back to the
generic Keyless identity path and therefore remain conservative.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any

from . import core_bsa_compat


AUDITED_UNTWIST_KEYLESS_GIT_BLOBS = frozenset(
    {"53079e70b7996b1a872d9ff1e5f219cc1f110c3e"}
)
_UNTWIST_TOPLEVEL_MODULE = "flux_untwist.keyless_h3"
_UNTWIST_MODULE_SUFFIX = ".flux_untwist.keyless_h3"
_UNTWIST_CLASS = "KeylessUntwistRoutingPreprocessor"
_UNTWIST_PROVIDER = "comfyui-flux2-untwisting-rope"
_UNTWIST_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
_UNTWIST_IDENTITY_PREFIX = "minimax_h3_untwist_keyless_route_v1:"
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


def _snapshot_static_identity(snapshot: Any) -> tuple[Any, ...] | None:
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

    instance_id = getattr(snapshot, "instance_id")
    scope = getattr(snapshot, "reference_scope")
    if not isinstance(instance_id, str) or not instance_id or scope not in _REFERENCE_SCOPES:
        return None
    try:
        expected_rows = int(getattr(snapshot, "expected_rows"))
        axis_count = int(getattr(snapshot, "rope_axis_count"))
        freq_count = int(getattr(snapshot, "rope_freqs_per_axis"))
    except (TypeError, ValueError):
        return None
    if expected_rows <= 0 or axis_count <= 0 or freq_count <= 0:
        return None

    raw_ranges = getattr(snapshot, "reference_ranges")
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
    temporal = getattr(snapshot, "scale_temporal_axis")
    if not isinstance(temporal, bool):
        return None

    # Progress is intentionally returned separately. Spectrum's visual-reference
    # external-patch transaction already identities and guards this coordinate.
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
    return instance_id, numeric["progress"], static


def reviewed_keyless_untwist_identity(
    value: Any,
    options: dict[str, Any],
) -> tuple[Any, ...] | None:
    """Return stable numerical identity only for the exact reviewed route preprocessor."""
    reviewed = _loaded_reviewed_module(value)
    if reviewed is None:
        return None
    _module, blob = reviewed

    declared = getattr(value, "identity", None)
    if (
        not isinstance(declared, str)
        or not declared.startswith(_UNTWIST_IDENTITY_PREFIX)
        or len(declared) != len(_UNTWIST_IDENTITY_PREFIX) + 64
    ):
        return None
    try:
        int(declared[len(_UNTWIST_IDENTITY_PREFIX) :], 16)
    except ValueError:
        return None

    snapshot = getattr(value, "_snapshot", None)
    parsed = _snapshot_static_identity(snapshot)
    if parsed is None:
        return None
    instance_id, progress, static = parsed
    runtime = _runtime_identity(
        options,
        instance_id=instance_id,
        progress=progress,
    )
    if runtime is None:
        return None

    return (
        "reviewed_keyless_untwist_route_v1",
        blob,
        runtime,
        static,
    )


__all__ = [
    "AUDITED_UNTWIST_KEYLESS_GIT_BLOBS",
    "reviewed_keyless_untwist_identity",
]
