from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from . import replay_component_shadow as _component
from . import replay_spectral_mixture_shadow as _spectral
from . import replay_trust_shadow as _replay
from . import trust_probe as _trust
from .experiments import OfflineSmoother
from .runtime import SpectrumH3Runtime

_SCHEMA_VERSION = 1
_SOURCE_SCHEMA_REVISION = "pr45-replay-calibration-v1"
_LOG_PREFIX = "SPECTRUM_REPLAY_CALIBRATION_JSON="
_ARCHIVE_STATE_ATTR = "_spectrum_replay_calibration_state"
_PACKAGE_NAME = "comfyui-spectrum-minimax-h3"
_FALLBACK_PACKAGE_VERSION = "0.2.12"
_PARITY_TOLERANCE = 2e-5
_MAX_SERIALIZED_BYTES = 96 * 1024
_FIXED_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

_ORIGINAL_BEGIN_OFFLINE_CAPTURE: Callable[..., Any] | None = None
_ORIGINAL_COMPLETE_OFFLINE_CAPTURE: Callable[..., Any] | None = None
_ORIGINAL_SPECTRAL_VALIDATOR: Callable[..., Any] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


@dataclass(slots=True)
class _CalibrationState:
    enabled: bool
    config_snapshot: dict[str, Any]
    run_id: int | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    failures: int = 0
    validated: bool = False
    emitted: bool = False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _package_version() -> str:
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return _FALLBACK_PACKAGE_VERSION


def _source_revision() -> tuple[str | None, str]:
    value = os.environ.get("SPECTRUM_H3_SOURCE_REVISION")
    if value:
        return value, "SPECTRUM_H3_SOURCE_REVISION"
    return None, "unavailable_without_external_annotation"


def _topology_map(archive: Any) -> dict[str, Any]:
    return {
        str(entry[0]): entry[1]
        for entry in (getattr(archive, "topology", None) or ())
        if isinstance(entry, tuple) and len(entry) == 2
    }


def _shape_text(value: Any) -> str | None:
    if not isinstance(value, (tuple, list)):
        return None
    try:
        return "x".join(str(int(item)) for item in value)
    except (TypeError, ValueError):
        return None


def _normalize_sampler_name(value: Any) -> str:
    name = getattr(value, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(value).__name__


def _storage_name(value: Any) -> str:
    if isinstance(value, torch.device):
        return str(value)
    if value is None:
        return "none"
    return str(value)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _iter_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_iter_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_iter_tensor_bytes(item) for item in value)
    return 0


def _scalar_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().to(device="cpu", dtype=torch.float32).item()
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _archive_schedule(archive: Any) -> tuple[float, ...]:
    raw = getattr(archive, "schedule", None)
    if raw is None:
        return ()
    result: list[float] = []
    for value in raw:
        scalar = _scalar_float(value)
        if scalar is None:
            return ()
        result.append(scalar)
    return tuple(result)


def _config_snapshot(runtime: SpectrumH3Runtime) -> dict[str, Any]:
    config = runtime.config
    keys = (
        "enabled",
        "blend_weight",
        "audio_blend_weight",
        "degree",
        "ridge_lambda",
        "window_size",
        "flex_window",
        "warmup_steps",
        "tail_actual_steps",
        "max_history",
        "history_storage",
        "offline_archive_storage",
        "bootstrap_first_forecast",
        "offline_smoothing_replay",
        "model_aware_mode",
        "model_aware_risk_threshold",
        "model_aware_trust_shrinkage",
        "model_aware_replay_generic_correction",
        "generic_correction_mode",
        "generic_correction_attenuation",
        "generic_correction_limiter",
        "generic_correction_limit",
        "anchor_residual_feedback",
        "selective_rollback_correction",
    )
    snapshot: dict[str, Any] = {}
    for key in keys:
        value = getattr(config, key, None)
        if isinstance(value, (str, bool, int)):
            snapshot[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            snapshot[key] = value
    return snapshot


def _get_state(runtime: SpectrumH3Runtime) -> _CalibrationState | None:
    return getattr(runtime, _ARCHIVE_STATE_ATTR, None)


def _set_state(runtime: SpectrumH3Runtime, value: _CalibrationState | None) -> None:
    setattr(runtime, _ARCHIVE_STATE_ATTR, value)


def _reset_state(runtime: SpectrumH3Runtime) -> None:
    _set_state(runtime, None)


def _get_or_create_state(runtime: SpectrumH3Runtime) -> _CalibrationState:
    existing = _get_state(runtime)
    if existing is not None:
        return existing
    state = _CalibrationState(
        enabled=bool(runtime.config.offline_smoothing_replay),
        config_snapshot=_config_snapshot(runtime),
    )
    _set_state(runtime, state)
    return state


def _archive_summary(runtime: SpectrumH3Runtime, archive: Any) -> dict[str, Any]:
    topology = _topology_map(archive)
    schedule = _archive_schedule(archive)
    actual_steps = tuple(int(value) for value in getattr(archive, "actual_steps", ()) or ())
    record_count = len(getattr(archive, "records", ()) or ())
    retained_bytes = int(getattr(archive, "retained_bytes", 0) or 0)
    storage = _storage_name(getattr(archive, "device", None))
    return {
        "sampler": _normalize_sampler_name(getattr(runtime._run, "sampler_fn", None)),
        "schedule": list(schedule),
        "step_count": max(0, len(schedule) - 1),
        "actual_steps": list(actual_steps),
        "anchor_count": len(actual_steps),
        "record_count": int(record_count),
        "retained_bytes": retained_bytes,
        "storage": storage,
        "video_shape": _shape_text(topology.get("video_shape")),
        "audio_shape": _shape_text(topology.get("audio_shape")),
        "target_audio_rows": topology.get("target_audio_rows"),
        "target_video_rows": topology.get("target_video_rows"),
        "hidden_width": topology.get("hidden_width"),
        "sigma_shifts": topology.get("sigma_shifts"),
    }


def _base_payload(runtime: SpectrumH3Runtime, archive: Any) -> dict[str, Any]:
    source_revision, source_revision_source = _source_revision()
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_schema_revision": _SOURCE_SCHEMA_REVISION,
        "package_name": _PACKAGE_NAME,
        "package_version": _package_version(),
        "source_revision": source_revision,
        "source_revision_source": source_revision_source,
        "config": _config_snapshot(runtime),
        "archive": _archive_summary(runtime, archive),
    }
    payload["provenance_hash"] = _sha256_json(payload)
    return payload


def _tensor_to_cpu_f32(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _tensor_rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    work = value.detach().to(dtype=torch.float32)
    return float(torch.sqrt(torch.mean(work * work)).item())


def _relative_rms_error(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    diff = actual - predicted
    denom = max(_tensor_rms(actual), 1e-6)
    return _tensor_rms(diff) / denom


def _cosine_similarity(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    left = actual.reshape(-1).to(dtype=torch.float32)
    right = predicted.reshape(-1).to(dtype=torch.float32)
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    denom_value = float(denom.item())
    if denom_value <= 1e-12 or not math.isfinite(denom_value):
        return 1.0 if torch.equal(left, right) else 0.0
    value = float(torch.dot(left, right).item()) / denom_value
    return max(-1.0, min(1.0, value))


def _prediction_metrics(actual: torch.Tensor, predicted: torch.Tensor) -> dict[str, float]:
    return {
        "rel_rms_error": _relative_rms_error(actual, predicted),
        "cosine_similarity": _cosine_similarity(actual, predicted),
    }


def _split_modalities(archive: Any, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    audio_rows = int(getattr(archive, "target_audio_rows", 0) or 0)
    return value[:audio_rows], value[audio_rows:]


def _collect_prediction_row(
    archive: Any,
    smoother: OfflineSmoother,
    *,
    step: int,
    actual: torch.Tensor,
) -> dict[str, Any] | None:
    try:
        baseline = smoother.predict(step, blend_weight=0.0, audio_blend_weight=0.0)
        full = smoother.predict(step, blend_weight=1.0, audio_blend_weight=1.0)
    except Exception:
        return None
    actual_cpu = _tensor_to_cpu_f32(actual)
    baseline_cpu = _tensor_to_cpu_f32(baseline)
    full_cpu = _tensor_to_cpu_f32(full)
    if actual_cpu.shape != baseline_cpu.shape or actual_cpu.shape != full_cpu.shape:
        return None
    actual_audio, actual_video = _split_modalities(archive, actual_cpu)
    baseline_audio, baseline_video = _split_modalities(archive, baseline_cpu)
    full_audio, full_video = _split_modalities(archive, full_cpu)
    row: dict[str, Any] = {
        "step": int(step),
        "baseline": {
            "all": _prediction_metrics(actual_cpu, baseline_cpu),
            "audio": _prediction_metrics(actual_audio, baseline_audio),
            "video": _prediction_metrics(actual_video, baseline_video),
        },
        "full": {
            "all": _prediction_metrics(actual_cpu, full_cpu),
            "audio": _prediction_metrics(actual_audio, full_audio),
            "video": _prediction_metrics(actual_video, full_video),
        },
        "weights": {},
    }
    delta = full_cpu - baseline_cpu
    audio_rows = int(getattr(archive, "target_audio_rows", 0) or 0)
    for weight in _FIXED_WEIGHTS:
        candidate = baseline_cpu + delta * float(weight)
        candidate_audio = candidate[:audio_rows]
        candidate_video = candidate[audio_rows:]
        row["weights"][f"{weight:.2f}"] = {
            "all": _prediction_metrics(actual_cpu, candidate),
            "audio": _prediction_metrics(actual_audio, candidate_audio),
            "video": _prediction_metrics(actual_video, candidate_video),
        }
    return row


def _validate_archive(runtime: SpectrumH3Runtime, archive: Any) -> dict[str, Any]:
    payload = _base_payload(runtime, archive)
    actual_steps = tuple(int(value) for value in getattr(archive, "actual_steps", ()) or ())
    records = getattr(archive, "records", None)
    if not actual_steps or not isinstance(records, dict):
        payload["validation"] = {"rows": [], "status": "no_actual_records"}
        return payload
    try:
        smoother = OfflineSmoother(archive)
    except Exception as exc:
        payload["validation"] = {
            "rows": [],
            "status": "smoother_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return payload
    rows: list[dict[str, Any]] = []
    for step in actual_steps:
        record = records.get(step)
        if record is None:
            continue
        actual = getattr(record, "feature", None)
        if not isinstance(actual, torch.Tensor):
            continue
        row = _collect_prediction_row(archive, smoother, step=step, actual=actual)
        if row is not None:
            rows.append(row)
    payload["validation"] = {
        "rows": rows,
        "status": "ok" if rows else "no_valid_rows",
    }
    return payload


def _emit_payload(payload: dict[str, Any]) -> None:
    encoded = _canonical_json(payload)
    if len(encoded.encode("utf-8")) > _MAX_SERIALIZED_BYTES:
        compact = dict(payload)
        validation = compact.get("validation")
        if isinstance(validation, dict):
            rows = validation.get("rows")
            if isinstance(rows, list):
                validation = dict(validation)
                validation["rows"] = rows[:4]
                validation["truncated_rows"] = max(0, len(rows) - 4)
                compact["validation"] = validation
        encoded = _canonical_json(compact)
    print(f"{_LOG_PREFIX}{encoded}")


def _begin_offline_capture(runtime: SpectrumH3Runtime, *args: Any, **kwargs: Any) -> Any:
    _reset_state(runtime)
    result = _ORIGINAL_BEGIN_OFFLINE_CAPTURE(runtime, *args, **kwargs)
    state = _get_or_create_state(runtime)
    state.run_id = getattr(runtime._run, "run_id", None)
    return result


def _complete_offline_capture(runtime: SpectrumH3Runtime, *args: Any, **kwargs: Any) -> Any:
    result = _ORIGINAL_COMPLETE_OFFLINE_CAPTURE(runtime, *args, **kwargs)
    state = _get_or_create_state(runtime)
    if not state.enabled or state.emitted:
        return result
    archive = getattr(runtime, "_offline_archive", None)
    if archive is None:
        return result
    try:
        payload = _validate_archive(runtime, archive)
        _emit_payload(payload)
        state.validated = True
        state.emitted = True
    except Exception:
        state.failures += 1
    return result


def _spectral_validator(*args: Any, **kwargs: Any) -> Any:
    result = _ORIGINAL_SPECTRAL_VALIDATOR(*args, **kwargs)
    return result


def _runtime_debug_summary(runtime: SpectrumH3Runtime) -> str:
    base = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(runtime)
    state = _get_state(runtime)
    if state is None:
        return base
    return (
        f"{base} replay_calibration_validated={state.validated}"
        f" replay_calibration_failures={state.failures}"
    )


def install_replay_calibration() -> None:
    global _ORIGINAL_BEGIN_OFFLINE_CAPTURE
    global _ORIGINAL_COMPLETE_OFFLINE_CAPTURE
    global _ORIGINAL_SPECTRAL_VALIDATOR
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if _ORIGINAL_BEGIN_OFFLINE_CAPTURE is not None:
        return
    _ORIGINAL_BEGIN_OFFLINE_CAPTURE = SpectrumH3Runtime.begin_offline_capture
    _ORIGINAL_COMPLETE_OFFLINE_CAPTURE = SpectrumH3Runtime.complete_offline_capture
    _ORIGINAL_SPECTRAL_VALIDATOR = _spectral.validate_candidate
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    SpectrumH3Runtime.begin_offline_capture = _begin_offline_capture
    SpectrumH3Runtime.complete_offline_capture = _complete_offline_capture
    _spectral.validate_candidate = _spectral_validator
    SpectrumH3Runtime.debug_summary = _runtime_debug_summary
