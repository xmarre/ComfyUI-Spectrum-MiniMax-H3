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
_FALLBACK_PACKAGE_VERSION = "0.2.9"
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


def _scalar_topology_metadata(archive: Any) -> dict[str, Any]:
    topology = _topology_map(archive)
    feature_shape = getattr(archive, "feature_shape", None)
    patch_size = topology.get("patch_size")
    sigma_shifts = topology.get("sigma_shifts")
    return {
        "feature_shape": _shape_text(feature_shape),
        "video_shape": _shape_text(topology.get("video_shape")),
        "video_padded": _shape_text(topology.get("video_padded")),
        "audio_shape": _shape_text(topology.get("audio_shape")),
        "text_length": topology.get("text_length") if isinstance(topology.get("text_length"), int) else None,
        "hidden_width": topology.get("hidden_width") if isinstance(topology.get("hidden_width"), int) else None,
        "target_audio_rows": topology.get("target_audio_rows") if isinstance(topology.get("target_audio_rows"), int) else None,
        "target_video_rows": topology.get("target_video_rows") if isinstance(topology.get("target_video_rows"), int) else None,
        "patch_size": _shape_text(patch_size),
        "sigma_shifts": (
            ",".join(f"{float(item):.17g}" for item in sigma_shifts)
            if isinstance(sigma_shifts, (tuple, list))
            else None
        ),
        "adaln_curves": topology.get("adaln_curves") if isinstance(topology.get("adaln_curves"), bool) else None,
    }


def _topology_fingerprint(archive: Any) -> str:
    topology = _topology_map(archive)
    stable = {
        "feature_shape": _shape_text(getattr(archive, "feature_shape", None)),
        "video_shape": _shape_text(topology.get("video_shape")),
        "video_padded": _shape_text(topology.get("video_padded")),
        "audio_shape": _shape_text(topology.get("audio_shape")),
        "text_length": topology.get("text_length"),
        "hidden_width": topology.get("hidden_width"),
        "target_audio_rows": topology.get("target_audio_rows"),
        "target_video_rows": topology.get("target_video_rows"),
        "patch_size": _shape_text(topology.get("patch_size")),
        "sigma_shifts": (
            [float(item) for item in topology.get("sigma_shifts")]
            if isinstance(topology.get("sigma_shifts"), (tuple, list))
            else None
        ),
        "adaln_curves": topology.get("adaln_curves"),
        "segments": repr(topology.get("segments")),
        "refs": repr(topology.get("refs")),
        "keyframes": repr(topology.get("keyframes")),
    }
    return _sha256_json(stable)


def _schedule_fingerprint(archive: Any) -> str:
    schedule = [
        {
            "step_id": int(step.step_id),
            "coordinate": float(step.coordinate),
            "actual": bool(step.actual),
        }
        for step in getattr(archive, "steps", ())
    ]
    return _sha256_json(schedule)


def _state(archive: Any) -> _CalibrationState | None:
    state = getattr(archive, _ARCHIVE_STATE_ATTR, None)
    return state if isinstance(state, _CalibrationState) else None


def _weight_suffix(weight: float) -> str:
    return f"{float(weight):.2f}".replace(".", "p")


def ratio_from_moments(row: dict[str, Any], weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    a = float(row["local_error_sq_mean"])
    b = float(row["local_error_dot_spectral_delta_mean"])
    c = float(row["spectral_delta_sq_mean"])
    denominator = float(row["ratio_denominator_rms"])
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("calibration ratio denominator must be finite and positive")
    mse = a + 2.0 * w * b + w * w * c
    if mse < 0.0 and abs(mse) <= 1e-12 * max(1.0, abs(a), abs(b), abs(c)):
        mse = 0.0
    if mse < 0.0 or not math.isfinite(mse):
        raise ValueError("calibration quadratic produced an invalid MSE")
    return math.sqrt(mse) / denominator


def oracle_weight_from_moments(row: dict[str, Any]) -> float:
    b = float(row["local_error_dot_spectral_delta_mean"])
    c = float(row["spectral_delta_sq_mean"])
    epsilon = float(row["ratio_epsilon"])
    denominator = max(c, epsilon * epsilon)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return 0.0
    return max(0.0, min(1.0, -b / denominator))


def _predictor_snapshot(
    *,
    record: _trust._ReplayShadowRecord,
    candidates: _component._ReplayCandidates,
    effective_blends: torch.Tensor,
) -> dict[str, torch.Tensor]:
    current_weight = _spectral._weight_projection(
        candidates.local,
        candidates.blend_uncorrected,
        candidates.spectral,
    )
    validation_penalty = (
        float(record.blend_weight) / effective_blends
    ).mean()
    spectral_gap = _trust._tensor_rms(
        candidates.spectral - candidates.local
    ) / _trust._tensor_rms(candidates.local).clamp_min(_trust._EPS)
    return {
        "current_weight": current_weight,
        "causal_disagreement": current_weight.new_tensor(float(record.disagreement)),
        "validation_penalty": validation_penalty,
        "spectral_gap": spectral_gap,
        "coordinate": current_weight.new_tensor(float(record.coordinate)),
    }


def _calibration_row(
    smoother: OfflineSmoother,
    record: _trust._ReplayShadowRecord,
    samples: torch.Tensor,
    anchor_ids: list[int],
    *,
    run_id: int | None,
) -> dict[str, Any] | None:
    if record.stream_name != "video" or record.blend_weight <= 1e-12:
        return None

    candidates = _component._construct_candidates(
        smoother,
        record,
        samples,
        anchor_ids,
    )
    if candidates is None:
        return None

    target_index = anchor_ids.index(record.step_id)
    retained = [index for index in range(len(anchor_ids)) if index != target_index]
    retained_ids = [anchor_ids[index] for index in retained]
    left_id = anchor_ids[target_index - 1]
    right_id = anchor_ids[target_index + 1]
    left_position = retained_ids.index(left_id)
    right_position = retained_ids.index(right_id)
    effective_blends = _replay._effective_blends_for_withheld_target(
        smoother,
        record,
        samples,
        anchor_ids,
        retained,
        left_position,
        right_position,
    )
    if torch.any(effective_blends <= 0):
        raise RuntimeError("replay calibration received nonpositive video blend")

    # Predictor/deployable values are frozen before the withheld target is read.
    predictors = _predictor_snapshot(
        record=record,
        candidates=candidates,
        effective_blends=effective_blends,
    )

    left = smoother.archive.anchors[target_index - 1]
    right = smoother.archive.anchors[target_index + 1]
    spacing = float(right.coordinate - left.coordinate)
    if abs(spacing) <= 1e-12:
        raise RuntimeError("replay calibration bracket has duplicate coordinates")
    bracket_fraction = (float(record.coordinate) - float(left.coordinate)) / spacing

    # Post-target scoring starts here. None of these values may feed a deployable
    # predictor or a production replay path.
    actual = samples[target_index]
    e = (candidates.local - actual).to(torch.float32)
    d = (candidates.spectral - candidates.local).to(torch.float32)
    hold_error = (actual - candidates.hold).to(torch.float32)
    actual_rms = _trust._tensor_rms(actual)
    ratio_epsilon = actual_rms.mul(1e-6).clamp_min(_trust._EPS)
    hold_error_sq_mean = torch.mean(hold_error.square())
    ratio_denominator_rms = torch.sqrt(hold_error_sq_mean).clamp_min(ratio_epsilon)
    local_error_sq_mean = torch.mean(e.square())
    spectral_delta_sq_mean = torch.mean(d.square())
    local_error_dot_spectral_delta_mean = torch.mean(e * d)

    local_ratio = _component._ratio(actual, candidates.local, ratio_denominator_rms)
    current_ratio = _component._ratio(
        actual,
        candidates.blend_uncorrected,
        ratio_denominator_rms,
    )
    full_spectral_ratio = _component._ratio(
        actual,
        candidates.spectral,
        ratio_denominator_rms,
    )
    oracle_ratio, oracle_weight = _component._axis_score(
        actual,
        candidates.local,
        candidates.spectral,
        ratio_denominator_rms,
    )
    fixed_ratios = {
        weight: _component._ratio(
            actual,
            candidates.local
            + float(weight) * (candidates.spectral - candidates.local),
            ratio_denominator_rms,
        )
        for weight in _FIXED_WEIGHTS
    }

    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("local_error_sq_mean", local_error_sq_mean),
        ("spectral_delta_sq_mean", spectral_delta_sq_mean),
        (
            "local_error_dot_spectral_delta_mean",
            local_error_dot_spectral_delta_mean,
        ),
        ("hold_error_sq_mean", hold_error_sq_mean),
        ("ratio_epsilon", ratio_epsilon),
        ("ratio_denominator_rms", ratio_denominator_rms),
        ("local_ratio", local_ratio),
        ("current_ratio", current_ratio),
        ("full_spectral_ratio", full_spectral_ratio),
        ("oracle_ratio", oracle_ratio),
        ("oracle_weight", oracle_weight),
        *[(name, tensor) for name, tensor in predictors.items()],
        *[
            (f"fixed_{_weight_suffix(weight)}_ratio", fixed_ratios[weight])
            for weight in _FIXED_WEIGHTS
        ],
    ]
    resolved_values = (
        torch.stack([tensor for _, tensor in named_tensors])
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .tolist()
    )
    resolved = {
        name: float(value)
        for (name, _), value in zip(named_tensors, resolved_values, strict=True)
    }

    topology = _topology_map(smoother.archive)
    row: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "target_step_id": int(record.step_id),
        "target_anchor_index": int(target_index),
        "coordinate": resolved["coordinate"],
        "left_anchor_step_id": int(left_id),
        "left_anchor_index": int(target_index - 1),
        "right_anchor_step_id": int(right_id),
        "right_anchor_index": int(target_index + 1),
        "bracket_coordinate_spacing": spacing,
        "bracket_fraction": float(bracket_fraction),
        "target_video_rows": (
            int(topology["target_video_rows"])
            if isinstance(topology.get("target_video_rows"), int)
            else None
        ),
        "scoring_sample_count": int(actual.numel()),
        "local_error_sq_mean": resolved["local_error_sq_mean"],
        "spectral_delta_sq_mean": resolved["spectral_delta_sq_mean"],
        "local_error_dot_spectral_delta_mean": resolved[
            "local_error_dot_spectral_delta_mean"
        ],
        "hold_error_sq_mean": resolved["hold_error_sq_mean"],
        "ratio_epsilon": resolved["ratio_epsilon"],
        "ratio_denominator_rms": resolved["ratio_denominator_rms"],
        "current_weight": resolved["current_weight"],
        "causal_disagreement": resolved["causal_disagreement"],
        "validation_penalty": resolved["validation_penalty"],
        "spectral_gap": resolved["spectral_gap"],
        "oracle_weight": resolved["oracle_weight"],
        "required_adjustment": (
            resolved["oracle_weight"] - resolved["current_weight"]
        ),
        "local_ratio": resolved["local_ratio"],
        "current_ratio": resolved["current_ratio"],
        "full_spectral_ratio": resolved["full_spectral_ratio"],
        "oracle_ratio": resolved["oracle_ratio"],
    }
    for weight in _FIXED_WEIGHTS:
        suffix = _weight_suffix(weight)
        row[f"fixed_{suffix}_ratio"] = resolved[f"fixed_{suffix}_ratio"]

    current_axis_ratio = ratio_from_moments(row, row["current_weight"])
    moment_oracle_weight = oracle_weight_from_moments(row)
    moment_oracle_ratio = ratio_from_moments(row, moment_oracle_weight)
    parity_errors = [
        abs(ratio_from_moments(row, 0.0) - row["local_ratio"]),
        abs(ratio_from_moments(row, 1.0) - row["full_spectral_ratio"]),
        abs(current_axis_ratio - row["current_ratio"]),
        abs(moment_oracle_weight - row["oracle_weight"]),
        abs(moment_oracle_ratio - row["oracle_ratio"]),
    ]
    for weight in _FIXED_WEIGHTS:
        suffix = _weight_suffix(weight)
        parity_errors.append(
            abs(
                ratio_from_moments(row, weight)
                - row[f"fixed_{suffix}_ratio"]
            )
        )
    row["current_axis_ratio"] = current_axis_ratio
    row["moment_oracle_weight"] = moment_oracle_weight
    row["moment_oracle_ratio"] = moment_oracle_ratio
    row["max_parity_abs_error"] = max(parity_errors, default=0.0)
    row["row_compatible"] = row["max_parity_abs_error"] <= _PARITY_TOLERANCE
    return row


def _validate_calibration(
    smoother: OfflineSmoother,
    _trust_aggregate: _trust._TrustAggregate,
) -> None:
    state = _state(smoother.archive)
    if state is None or not state.enabled or state.validated:
        return
    state.rows.clear()
    try:
        try:
            ranges = {
                name: (start, end)
                for name, start, end in smoother._stream_ranges
            }
            video_range = ranges.get("video")
            if video_range is None:
                state.validated = True
                return
            samples = _trust._sample_archive_stream(
                smoother,
                video_range[0],
                video_range[1],
            )
            anchor_ids = list(smoother._anchor_ids)
            records = getattr(
                smoother.archive,
                "_model_aware_trust_replay_shadow_records",
                None,
            )
            if not isinstance(records, list):
                state.validated = True
                return
        except torch.cuda.OutOfMemoryError:
            raise
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
        ):
            state.failures += 1
            state.validated = True
            return

        for record in records:
            if (
                not isinstance(record, _trust._ReplayShadowRecord)
                or record.stream_name != "video"
            ):
                continue
            try:
                row = _calibration_row(
                    smoother,
                    record,
                    samples,
                    anchor_ids,
                    run_id=state.run_id,
                )
                if row is not None:
                    state.rows.append(row)
            except torch.cuda.OutOfMemoryError:
                raise
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                KeyError,
                IndexError,
            ):
                state.failures += 1
        state.rows.sort(key=lambda row: int(row["target_step_id"]))
        state.validated = True
    except torch.cuda.OutOfMemoryError:
        raise


def _validate_spectral_with_calibration(
    smoother: OfflineSmoother,
    aggregate: _trust._TrustAggregate,
) -> None:
    if _ORIGINAL_SPECTRAL_VALIDATOR is None:
        raise RuntimeError("replay calibration was not installed correctly")
    failures_before = aggregate.replay_shadow_failures
    _ORIGINAL_SPECTRAL_VALIDATOR(smoother, aggregate)
    if aggregate.replay_shadow_failures != failures_before:
        return
    _validate_calibration(smoother, aggregate)


def _begin_offline_capture_with_calibration(
    self: SpectrumH3Runtime,
    *,
    total_steps: int,
    sampler_name: str,
) -> None:
    if _ORIGINAL_BEGIN_OFFLINE_CAPTURE is None:
        raise RuntimeError("replay calibration was not installed correctly")
    _ORIGINAL_BEGIN_OFFLINE_CAPTURE(
        self,
        total_steps=total_steps,
        sampler_name=sampler_name,
    )
    archive = getattr(self, "_offline_archive", None)
    if archive is None:
        return
    enabled = bool(
        self.config.debug
        and self.config.offline_smoothing_replay
        and self.config.model_aware_mode == "full"
    )
    setattr(
        archive,
        _ARCHIVE_STATE_ATTR,
        _CalibrationState(
            enabled=enabled,
            config_snapshot=asdict(self.config),
        ),
    )


def _complete_offline_capture_with_calibration(self: SpectrumH3Runtime) -> bool:
    if _ORIGINAL_COMPLETE_OFFLINE_CAPTURE is None:
        raise RuntimeError("replay calibration was not installed correctly")
    archive = getattr(self, "_offline_archive", None)
    state = None if archive is None else _state(archive)
    if state is not None:
        state.run_id = int(self.stats.run_id)
    return bool(_ORIGINAL_COMPLETE_OFFLINE_CAPTURE(self))


def _build_block(runtime: SpectrumH3Runtime, state: _CalibrationState) -> dict[str, Any]:
    archive = runtime.offline_archive
    if archive is None:
        raise ValueError("replay calibration export requires an offline archive")

    package_version = _package_version()
    source_revision, source_revision_source = _source_revision()
    config_hash = _sha256_json(state.config_snapshot)
    topology_fingerprint = _topology_fingerprint(archive)
    schedule_fingerprint = _schedule_fingerprint(archive)
    target_signature = [
        {
            "target_step_id": int(row["target_step_id"]),
            "coordinate": float(row["coordinate"]),
            "left_anchor_step_id": int(row["left_anchor_step_id"]),
            "right_anchor_step_id": int(row["right_anchor_step_id"]),
            "current_weight": float(row["current_weight"]),
            "causal_disagreement": float(row["causal_disagreement"]),
            "validation_penalty": float(row["validation_penalty"]),
            "spectral_gap": float(row["spectral_gap"]),
        }
        for row in state.rows
    ]
    trace_fingerprint = _sha256_json(
        {
            "schema_version": _SCHEMA_VERSION,
            "source_schema_revision": _SOURCE_SCHEMA_REVISION,
            "package_version": package_version,
            "source_revision": source_revision,
            "config_hash": config_hash,
            "sampler": archive.sampler_name,
            "steps": int(archive.total_steps),
            "schedule_fingerprint": schedule_fingerprint,
            "topology_fingerprint": topology_fingerprint,
            "target_signature": target_signature,
        }
    )
    rows = []
    for original in state.rows:
        row = dict(original)
        row["trace_fingerprint"] = trace_fingerprint
        rows.append(row)

    topology = _scalar_topology_metadata(archive)
    compatible = bool(rows) and state.failures == 0 and all(
        bool(row["row_compatible"]) for row in rows
    )
    actual_step_ids = ",".join(
        str(int(step.step_id))
        for step in archive.steps
        if bool(step.actual)
    )
    block = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "spectrum_h3_replay_calibration",
        "provenance": {
            "package_name": _PACKAGE_NAME,
            "package_version": package_version,
            "source_schema_revision": _SOURCE_SCHEMA_REVISION,
            "source_revision": source_revision,
            "source_revision_source": source_revision_source,
            "seed": None,
            "label": None,
            "config_hash": config_hash,
            "schedule_fingerprint": schedule_fingerprint,
            "topology_fingerprint": topology_fingerprint,
            "trace_fingerprint": trace_fingerprint,
        },
        "config": dict(state.config_snapshot),
        "metadata": {
            "run_id": state.run_id,
            "sampler": str(archive.sampler_name),
            "steps": int(archive.total_steps),
            "scheduler": None,
            "actual_step_ids": actual_step_ids,
            "ratio_definition": "sqrt(A+2*w*B+w^2*C)/ratio_denominator_rms; mean across targets",
            "moment_sign_convention": "e=local-withheld_target; d=spectral-local; B=mean(e*d)",
            "scoring_subset": "replay_shadow_deterministic_video_sample",
            "scoring_sample_cap_total": int(_trust._REPLAY_SHADOW_SAMPLE_ELEMENTS),
            "parity_tolerance": _PARITY_TOLERANCE,
            "predictor_fields": "current_weight,causal_disagreement,validation_penalty,spectral_gap,coordinate",
            "post_target_fields": "quadratic_moments,oracle_weight,required_adjustment,ratios,parity",
            "calibration_failures": int(state.failures),
            "compatible": compatible,
            **topology,
        },
        "target_rows": rows,
    }
    return block


def _debug_summary_with_calibration(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("replay calibration was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    archive = getattr(self, "_offline_archive", None)
    state = None if archive is None else _state(archive)
    if (
        state is None
        or not state.enabled
        or state.emitted
        or self.offline_phase != "first_pass"
        or not state.validated
        or not state.rows
    ):
        return summary
    try:
        block = _build_block(self, state)
        payload = _canonical_json(block)
        size = len(payload.encode("utf-8"))
        if size > _MAX_SERIALIZED_BYTES:
            state.failures += 1
            return (
                f"{summary} replay_calibration_export=skipped_payload_too_large "
                f"replay_calibration_payload_bytes={size}"
            )
        state.emitted = True
        return (
            f"{summary} replay_calibration_schema={_SCHEMA_VERSION} "
            f"replay_calibration_rows={len(state.rows)} "
            f"replay_calibration_payload_bytes={size} "
            f"{_LOG_PREFIX}{payload}"
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        OverflowError,
    ):
        state.failures += 1
        return f"{summary} replay_calibration_export=failed"


def install_replay_calibration() -> None:
    """Install debug-only exact replay calibration export and parity diagnostics."""
    global _ORIGINAL_BEGIN_OFFLINE_CAPTURE
    global _ORIGINAL_COMPLETE_OFFLINE_CAPTURE
    global _ORIGINAL_SPECTRAL_VALIDATOR
    global _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_replay_calibration_installed", False):
        return
    if not getattr(SpectrumH3Runtime, "_replay_spectral_alpha_shadow_installed", False):
        raise RuntimeError("install replay spectral alpha shadow before replay calibration")
    if not getattr(SpectrumH3Runtime, "_replay_generic_correction_gate_installed", False):
        raise RuntimeError("install replay generic-correction gate before replay calibration")

    _ORIGINAL_BEGIN_OFFLINE_CAPTURE = SpectrumH3Runtime.begin_offline_capture
    _ORIGINAL_COMPLETE_OFFLINE_CAPTURE = SpectrumH3Runtime.complete_offline_capture
    _ORIGINAL_SPECTRAL_VALIDATOR = _spectral._validate_spectral_mixture_shadow
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary

    SpectrumH3Runtime.begin_offline_capture = _begin_offline_capture_with_calibration
    SpectrumH3Runtime.complete_offline_capture = _complete_offline_capture_with_calibration
    _spectral._validate_spectral_mixture_shadow = _validate_spectral_with_calibration
    SpectrumH3Runtime.debug_summary = _debug_summary_with_calibration
    SpectrumH3Runtime._replay_calibration_installed = True


__all__ = [
    "install_replay_calibration",
    "oracle_weight_from_moments",
    "ratio_from_moments",
]
