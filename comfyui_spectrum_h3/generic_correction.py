from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import Any

import torch

from . import model_aware as _model_aware
from .forecast import HistoryWeightForecaster
from .model_aware import (
    AnchorEvidence,
    AnchorEvidenceTiming,
    CorrectionGainTelemetry,
    ModelAwareController,
    ModelAwareForecastDecision,
    StreamAnchorEvidence,
    SubspaceCorrectionTelemetry,
)
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)
_EPS = torch.finfo(torch.float32).eps


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


def _compact_head_metric(_weight: Any, _hidden_size: int):
    """Feature 1/2 need scalar stream sensitivity, not retained head operators."""
    return None, None, 0


def _generic_telemetry(source: CorrectionGainTelemetry) -> CorrectionGainTelemetry:
    """Collapse the rejected exact/diagonal branches onto the surviving generic gain."""
    raw = float(source.raw_generic_gain)
    gain = float(source.generic_gain)
    active = bool(source.generic_bound_active)
    return CorrectionGainTelemetry(
        residual_projection=float(source.residual_projection),
        diagonal_projection=float(source.residual_projection),
        model_projection=float(source.residual_projection),
        raw_generic_gain=raw,
        raw_diagonal_gain=raw,
        raw_model_gain=raw,
        generic_gain=gain,
        diagonal_candidate_gain=gain,
        model_candidate_gain=gain,
        model_gain=gain,
        model_trust=0.0,
        generic_bound_active=active,
        diagonal_bound_active=active,
        model_bound_active=active,
    )


_ORIGINAL_DECISION = ModelAwareController.decision


def _generic_decision(self: ModelAwareController, **kwargs) -> ModelAwareForecastDecision:
    decision = _ORIGINAL_DECISION(self, **kwargs)
    if self.mode != "full":
        return decision
    audio = _generic_telemetry(decision.audio_correction_telemetry)
    video = _generic_telemetry(decision.video_correction_telemetry)
    return replace(
        decision,
        audio_correction_gain=audio.generic_gain,
        video_correction_gain=video.generic_gain,
        audio_correction_telemetry=audio,
        video_correction_telemetry=video,
        audio_subspace_telemetry=SubspaceCorrectionTelemetry(),
        video_subspace_telemetry=SubspaceCorrectionTelemetry(),
        correction_anchor_ids=(),
    )


_ORIGINAL_FORECASTER_UPDATE = HistoryWeightForecaster.update


def _forecaster_update(
    self: HistoryWeightForecaster,
    coordinate: float,
    feature: torch.Tensor,
    **kwargs,
) -> None:
    mode = getattr(self, "_generic_correction_capture_mode", None)
    kwargs["exact_head_weights"] = None
    if mode == "schedule":
        kwargs["evidence_segments"] = None
    return _ORIGINAL_FORECASTER_UPDATE(self, coordinate, feature, **kwargs)


def _risk_anchor_evidence(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or forecaster.feature_shape is None:
        return None
    if tuple(combined.shape) != forecaster.feature_shape:
        raise ValueError("actual feature shape changed during model-aware risk evidence sampling")
    if len(forecaster._evidence_history) != forecaster.history_length:
        raise RuntimeError("device-local model-aware risk evidence history is not aligned")

    stream_evidence: dict[str, StreamAnchorEvidence] = {}
    weight_fit_seconds = sample_index_seconds = reduction_seconds = 0.0
    scalar_transfer_seconds = 0.0
    for name, start, end in runtime._stream_ranges(step.calls[0]):
        if name == "audio":
            blend = decision.audio_blend_weight
        elif name == "video":
            blend = decision.video_blend_weight
        else:
            if not math.isclose(
                decision.audio_blend_weight,
                decision.video_blend_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("packed risk evidence requires audio/video row metadata")
            blend = decision.video_blend_weight

        started = time.perf_counter()
        raw_weights = forecaster.model_aware_weights(
            step.coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=0.0,
        )
        weight_fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        actual = forecaster._sample_segment_device(combined, start, end)
        sample_index_seconds += time.perf_counter() - started
        if any(sample.device != actual.device for sample in history_samples):
            raise RuntimeError("model-aware risk evidence device changed during actual history")

        started = time.perf_counter()
        predicted = torch.zeros_like(actual)
        for weight_value, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight_value != 0.0:
                predicted.add_(sample, alpha=float(weight_value))
        latest = history_samples[-1]
        previous = history_samples[-2]
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_EPS)
        forecast_ratio = _tensor_rms(actual - predicted) / _tensor_rms(
            actual - latest
        ).clamp_min(epsilon)
        if len(history_samples) >= 3:
            curvature = latest - 2.0 * previous + history_samples[-3]
            curvature_ratio = _tensor_rms(curvature) / _tensor_rms(
                latest - previous
            ).clamp_min(epsilon)
        else:
            curvature_ratio = torch.zeros((), dtype=torch.float32, device=actual.device)
        values = torch.stack((forecast_ratio, curvature_ratio))
        reduction_seconds += time.perf_counter() - started
        started = time.perf_counter()
        forecast_value, curvature_value = values.detach().to(device="cpu").tolist()
        scalar_transfer_seconds += time.perf_counter() - started
        stream_evidence[name] = StreamAnchorEvidence(
            forecast_ratio=float(forecast_value),
            curvature_ratio=float(curvature_value),
        )

    if "packed" in stream_evidence:
        audio = video = stream_evidence["packed"]
    else:
        audio = stream_evidence.get("audio", StreamAnchorEvidence())
        video = stream_evidence.get("video", StreamAnchorEvidence())
    started = time.perf_counter()
    fit_condition = forecaster.fit_condition(degree=decision.degree)
    fit_condition_seconds = time.perf_counter() - started
    return AnchorEvidence(
        forecast_ratio=max(audio.forecast_ratio, video.forecast_ratio),
        curvature_ratio=max(audio.curvature_ratio, video.curvature_ratio),
        fit_condition=fit_condition,
        audio_projection=0.0,
        video_projection=0.0,
        model_corrected_ratio=0.0,
        generic_corrected_ratio=0.0,
        audio=audio,
        video=video,
        subspace_workspace_bytes=0,
        timing=AnchorEvidenceTiming(
            weight_fit_seconds=weight_fit_seconds,
            sample_index_seconds=sample_index_seconds,
            device_transfer_seconds=scalar_transfer_seconds,
            scalar_transfer_seconds=scalar_transfer_seconds,
            reduction_seconds=reduction_seconds,
            exact_head_projection_seconds=0.0,
            fit_condition_seconds=fit_condition_seconds,
            subspace_gram_seconds=0.0,
            subspace_solve_seconds=0.0,
        ),
    )


def _generic_anchor_evidence(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or forecaster.feature_shape is None:
        return None
    if tuple(combined.shape) != forecaster.feature_shape:
        raise ValueError("actual feature shape changed during generic correction evidence sampling")
    if len(forecaster._evidence_history) != forecaster.history_length:
        raise RuntimeError("device-local generic correction evidence history is not aligned")

    stream_evidence: dict[str, StreamAnchorEvidence] = {}
    weight_fit_seconds = sample_index_seconds = reduction_seconds = 0.0
    scalar_transfer_seconds = 0.0
    for name, start, end in runtime._stream_ranges(step.calls[0]):
        if name == "audio":
            blend = decision.audio_blend_weight
            gain = decision.audio_correction_gain
        elif name == "video":
            blend = decision.video_blend_weight
            gain = decision.video_correction_gain
        else:
            if not math.isclose(
                decision.audio_blend_weight,
                decision.video_blend_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("packed generic correction requires audio/video row metadata")
            blend = decision.video_blend_weight
            gain = 0.5 * (
                decision.audio_correction_gain + decision.video_correction_gain
            )

        started = time.perf_counter()
        raw_weights = forecaster.model_aware_weights(
            step.coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=0.0,
        )
        weight_fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        actual = forecaster._sample_segment_device(combined, start, end)
        sample_index_seconds += time.perf_counter() - started

        started = time.perf_counter()
        predicted = torch.zeros_like(actual)
        for weight_value, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight_value != 0.0:
                predicted.add_(sample, alpha=float(weight_value))
        latest = history_samples[-1]
        previous = history_samples[-2]
        delta = latest - previous
        residual = actual - predicted
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        dot_epsilon = epsilon.square() * max(1, int(delta.numel()))
        projection = torch.dot(residual, delta) / torch.dot(delta, delta).clamp_min(
            dot_epsilon
        )
        forecast_ratio = _tensor_rms(residual) / hold_rms
        corrected_ratio = _tensor_rms(
            actual - (predicted + float(gain) * delta)
        ) / hold_rms
        if len(history_samples) >= 3:
            curvature = latest - 2.0 * previous + history_samples[-3]
            curvature_ratio = _tensor_rms(curvature) / _tensor_rms(delta).clamp_min(
                epsilon
            )
        else:
            curvature_ratio = torch.zeros((), dtype=torch.float32, device=actual.device)
        values = torch.stack(
            (forecast_ratio, curvature_ratio, projection, corrected_ratio)
        )
        reduction_seconds += time.perf_counter() - started
        started = time.perf_counter()
        forecast_value, curvature_value, projection_value, corrected_value = (
            values.detach().to(device="cpu").tolist()
        )
        scalar_transfer_seconds += time.perf_counter() - started
        stream_evidence[name] = StreamAnchorEvidence(
            forecast_ratio=float(forecast_value),
            curvature_ratio=float(curvature_value),
            residual_projection=float(projection_value),
            diagonal_projection=float(projection_value),
            model_projection=float(projection_value),
            model_corrected_ratio=float(corrected_value),
            generic_corrected_ratio=float(corrected_value),
            diagonal_candidate_ratio=float(corrected_value),
            model_candidate_ratio=float(corrected_value),
            model_corrected_head_ratio=float(corrected_value),
            generic_corrected_head_ratio=float(corrected_value),
            diagonal_candidate_head_ratio=float(corrected_value),
            model_candidate_head_ratio=float(corrected_value),
        )

    if "packed" in stream_evidence:
        audio = video = stream_evidence["packed"]
    else:
        audio = stream_evidence.get("audio", StreamAnchorEvidence())
        video = stream_evidence.get("video", StreamAnchorEvidence())
    started = time.perf_counter()
    fit_condition = forecaster.fit_condition(degree=decision.degree)
    fit_condition_seconds = time.perf_counter() - started
    return AnchorEvidence(
        forecast_ratio=max(audio.forecast_ratio, video.forecast_ratio),
        curvature_ratio=max(audio.curvature_ratio, video.curvature_ratio),
        fit_condition=fit_condition,
        audio_projection=audio.residual_projection,
        video_projection=video.residual_projection,
        model_corrected_ratio=max(
            audio.generic_corrected_ratio,
            video.generic_corrected_ratio,
        ),
        generic_corrected_ratio=max(
            audio.generic_corrected_ratio,
            video.generic_corrected_ratio,
        ),
        audio=audio,
        video=video,
        subspace_workspace_bytes=0,
        timing=AnchorEvidenceTiming(
            weight_fit_seconds=weight_fit_seconds,
            sample_index_seconds=sample_index_seconds,
            device_transfer_seconds=scalar_transfer_seconds,
            scalar_transfer_seconds=scalar_transfer_seconds,
            reduction_seconds=reduction_seconds,
            exact_head_projection_seconds=0.0,
            fit_condition_seconds=fit_condition_seconds,
            subspace_gram_seconds=0.0,
            subspace_solve_seconds=0.0,
        ),
    )


_ORIGINAL_RUNTIME_WEIGHT_SEGMENTS = SpectrumH3Runtime._model_aware_weight_segments
_ORIGINAL_RUNTIME_START = SpectrumH3Runtime.start_run
_ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary


def _head_metrics_disabled(
    self: SpectrumH3Runtime,
    _device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    self.stats.model_aware_head_materialization_seconds = 0.0
    self.stats.model_aware_head_materialized_bytes = 0
    return {}, {}


def _weight_segments(
    self: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
):
    fit_before = self.forecaster.model_aware_fit_seconds
    correction_before = self.forecaster.model_aware_correction_seconds
    weighted = []
    for name, start, end in self._stream_ranges(call):
        if name == "audio":
            blend = decision.audio_blend_weight
            correction = decision.audio_correction_gain
        elif name == "video":
            blend = decision.video_blend_weight
            correction = decision.video_correction_gain
        else:
            if not math.isclose(
                decision.audio_blend_weight,
                decision.video_blend_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "packed H3 topology does not expose audio/video correction boundary"
                )
            blend = decision.video_blend_weight
            correction = 0.5 * (
                decision.audio_correction_gain + decision.video_correction_gain
            )
        weights = self.forecaster.model_aware_weights(
            coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=correction,
            correction_coefficients=(),
            correction_anchor_ids=(),
        )
        weighted.append((start, end, weights))
    fit_elapsed = max(0.0, self.forecaster.model_aware_fit_seconds - fit_before)
    correction_elapsed = max(
        0.0,
        self.forecaster.model_aware_correction_seconds - correction_before,
    )
    self.stats.model_aware_fit_seconds += fit_elapsed
    self.stats.model_aware_causal_correction_seconds += correction_elapsed
    self.stats.model_aware_correction_seconds = (
        self.stats.model_aware_causal_correction_seconds
        + self.stats.model_aware_offline_correction_seconds
    )
    self.stats.model_aware_overhead_seconds += fit_elapsed + correction_elapsed
    return tuple(weighted)


def _observe_anchor(
    self: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    _exact_head_weights: dict[str, torch.Tensor],
    _stream_diagonals: dict[str, torch.Tensor],
) -> None:
    if not self._model_aware_enabled():
        return
    mode = self.config.model_aware_mode
    if mode == "schedule":
        return
    started = time.perf_counter()
    try:
        decision = self.model_aware.decision(
            forecast_horizon=1.0,
            history_length=self.forecaster.history_length,
            configured_degree=self.config.degree,
            configured_ridge_lambda=self.config.ridge_lambda,
            configured_audio_blend=self.config.audio_blend_weight,
            configured_video_blend=self.config.blend_weight,
        )
        evidence = (
            _risk_anchor_evidence(self, step, combined, decision)
            if mode == "schedule_confidence"
            else _generic_anchor_evidence(self, step, combined, decision)
        )
        if evidence is not None:
            self.model_aware.observe_anchor(evidence, decision)
            self.stats.model_aware_anchor_updates += 1
            self.stats.model_aware_evidence_weight_fit_seconds += (
                evidence.timing.weight_fit_seconds
            )
            self.stats.model_aware_evidence_sample_index_seconds += (
                evidence.timing.sample_index_seconds
            )
            self.stats.model_aware_evidence_device_transfer_seconds += (
                evidence.timing.device_transfer_seconds
            )
            self.stats.model_aware_evidence_scalar_transfer_seconds += (
                evidence.timing.scalar_transfer_seconds
            )
            self.stats.model_aware_evidence_reduction_seconds += (
                evidence.timing.reduction_seconds
            )
            self.stats.model_aware_evidence_fit_condition_seconds += (
                evidence.timing.fit_condition_seconds
            )
            self.stats.model_aware_evidence_exact_head_projection_seconds = 0.0
            self.stats.model_aware_exact_head_projection_seconds = 0.0
            self.stats.model_aware_exact_head_projection_calls = 0
            self.stats.model_aware_exact_head_workspace_bytes = 0
            self.stats.model_aware_subspace_gram_seconds = 0.0
            self.stats.model_aware_subspace_solve_seconds = 0.0
            self.stats.model_aware_subspace_workspace_bytes = 0
            self.stats.model_aware_model_corrected_ratio_mean = (
                self.model_aware.generic_corrected_ratio_mean
            )
            self.stats.model_aware_generic_corrected_ratio_mean = (
                self.model_aware.generic_corrected_ratio_mean
            )
            if self.config.debug and mode == "full":
                LOG.warning(
                    "Spectrum H3 generic scalar correction anchor step=%s "
                    "audio_raw=%.6f audio_generic=%.6f "
                    "video_raw=%.6f video_generic=%.6f",
                    step.step_id,
                    evidence.audio.forecast_ratio,
                    evidence.audio.generic_corrected_ratio,
                    evidence.video.forecast_ratio,
                    evidence.video.generic_corrected_ratio,
                )
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        self._model_aware_disabled_reason = f"model-aware evidence failed: {exc}"
        self.stats.model_aware_failures += 1
        LOG.warning(
            "Spectrum H3 model-aware evidence disabled for this run: %s",
            self._model_aware_disabled_reason,
        )
    finally:
        elapsed = time.perf_counter() - started
        self.stats.model_aware_evidence_seconds += elapsed
        self.stats.model_aware_overhead_seconds += elapsed


def _start_run(self: SpectrumH3Runtime, *args, **kwargs):
    run_id = _ORIGINAL_RUNTIME_START(self, *args, **kwargs)
    self.forecaster._generic_correction_capture_mode = self.config.model_aware_mode
    return run_id


def _debug_summary(self: SpectrumH3Runtime) -> str:
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    replacements = {
        "model_aware_correction_subspace=two_causal_actual_deltas": (
            "model_aware_correction_subspace=retired_k2_inactive"
        ),
        "model_aware_subspace_model_comparison=exact_2d_vs_generic_2d_head_rms": (
            "model_aware_subspace_model_comparison=retired_k2_inactive"
        ),
        "model_aware_subspace_bound=radial_rational_softsign_0.25": (
            "model_aware_subspace_bound=retired_k2_inactive"
        ),
    }
    for old, new in replacements.items():
        summary = summary.replace(old, new)
    return (
        f"{summary} "
        "feature3_model_informed_correction=retired_no_material_gain "
        "feature3_applied_correction=generic_scalar_latest_delta "
        "feature3_k2_runtime=retired "
        "feature3_transformed_trajectory_runtime=retired "
        "feature3_previous_error_runtime=retired "
        "feature3_direction_evidence_bytes=0 "
        "feature3_direction_workspace_bytes=0 "
        "feature3_error_evidence_bytes=0 "
        "feature3_error_workspace_bytes=0 "
        "feature3_extra_transformer_nfe=0"
    )


def install_generic_residual_correction() -> None:
    """Install the final post-experiment architecture once per interpreter."""
    if getattr(SpectrumH3Runtime, "_generic_residual_correction_installed", False):
        return

    # Feature 1/2 derive compact scalar sensitivities during profile construction.
    # Full output-head copies and Gram diagonals were only needed by rejected
    # Feature-3 experiments, so prevent them from entering the process LRU.
    _model_aware._head_metric = _compact_head_metric

    ModelAwareController.decision = _generic_decision
    ModelAwareController.materialized_head_bytes = property(lambda _self: 0)
    ModelAwareController.head_metric_tensors = (
        lambda _self, _stream, _device: (None, None, 0.0)
    )
    ModelAwareController.channel_sensitivity = (
        lambda _self, _stream, _device: (None, 0.0)
    )

    HistoryWeightForecaster.update = _forecaster_update
    SpectrumH3Runtime._model_aware_head_metrics = _head_metrics_disabled
    SpectrumH3Runtime._model_aware_weight_segments = _weight_segments
    SpectrumH3Runtime._observe_model_aware_anchor = _observe_anchor
    SpectrumH3Runtime.start_run = _start_run
    SpectrumH3Runtime.debug_summary = _debug_summary
    SpectrumH3Runtime._generic_residual_correction_installed = True


__all__ = ["install_generic_residual_correction"]
