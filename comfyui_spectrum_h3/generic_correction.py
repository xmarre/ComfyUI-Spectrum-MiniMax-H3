from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import replace
from typing import Any

import torch

from . import model_aware as _model_aware
from .forecast import HistoryWeightForecaster
from .generic_correction_calibration import (
    GenericCalibrationState,
)
from .generic_correction_calibration import (
    create_state as create_calibration_state,
)
from .generic_correction_calibration import (
    emit_block as emit_calibration_block,
)
from .generic_correction_research import persist_and_analyze
from .generic_correction_controller import GenericCorrectionController
from .generic_correction_core import (
    GainApplication,
    coordinate_transport_scale,
    limit_gain,
    resolve_attenuation_policy,
)
from .generic_correction_runtime import (
    _advanced_weight_segments,
    _exact_anchor_analysis,
)
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

_RETIRED_SUMMARY_FIELDS = (
    "model_aware_evidence_exact_head_projection_s",
    "model_aware_subspace_gram_s",
    "model_aware_subspace_solve_s",
    "model_aware_subspace_workspace_bytes",
    "model_aware_head_materialization_s",
    "model_aware_head_materialized_bytes",
    "model_aware_exact_head_projection_s",
    "model_aware_exact_head_projection_calls",
    "model_aware_exact_head_workspace_bytes",
    "model_aware_correction_subspace",
    "model_aware_subspace_model_comparison",
    "model_aware_diagonal_ablation_metric",
    "model_aware_model_comparison_metric",
    "model_aware_head_metric_available",
    "model_aware_subspace_bound",
    "model_aware_correction_bound",
    "model_corrected_ratio_mean",
    "model_aware_exact_head_evidence_bytes",
)
_RETIRED_SUMMARY_PATTERNS = (
    r"model_aware_(?:audio|video)_model_corrected_ratio_mean",
    r"model_aware_(?:audio|video)_diagonal\S*",
    r"model_aware_(?:audio|video)_exact\S*",
    r"model_aware_(?:audio|video)_generic_head_ratio_mean",
    r"model_aware_(?:audio|video)_gain_delta\S*",
    r"model_aware_\S*2d\S*",
    r"model_aware_subspace_\S*",
)


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


def _compact_head_metric(_weight: Any, _hidden_size: int):
    """Feature 1/2 need scalar stream sensitivity, not retained head operators."""
    return None, None, 0


def _generic_telemetry(
    source: CorrectionGainTelemetry,
    controller: GenericCorrectionController | None = None,
    *,
    stream: str = "audio",
    confidence: float = 1.0,
) -> CorrectionGainTelemetry:
    """Expose only the surviving generic gain; retired candidate fields stay zero."""
    raw_gain = float(source.raw_generic_gain)
    gain = float(source.generic_gain)
    bound_active = bool(source.generic_bound_active)
    projection = float(source.residual_projection)
    if controller is not None:
        if (
            controller.mode == "legacy"
            and controller.limiter == "rational"
            and math.isclose(controller.limit, 0.25, rel_tol=0.0, abs_tol=1e-12)
        ):
            pass
        elif controller.mode == "legacy":
            gain = limit_gain(raw_gain, controller.limiter, controller.limit)
            bound_active = abs(gain - raw_gain) > 1e-12
        else:
            application = controller.application(
                stream,
                general_confidence=confidence,
            )
            projection = application.raw_gain
            raw_gain = application.scaled_gain
            gain = application.bounded_gain
            bound_active = application.bound_active
    return CorrectionGainTelemetry(
        residual_projection=projection,
        raw_generic_gain=raw_gain,
        generic_gain=gain,
        generic_bound_active=bound_active,
    )


def _application_telemetry(application: GainApplication) -> CorrectionGainTelemetry:
    return CorrectionGainTelemetry(
        residual_projection=application.raw_gain,
        raw_generic_gain=application.scaled_gain,
        generic_gain=application.bounded_gain,
        generic_bound_active=application.bound_active,
    )


_ORIGINAL_DECISION = ModelAwareController.decision


def _generic_decision(self: ModelAwareController, **kwargs) -> ModelAwareForecastDecision:
    decision = _ORIGINAL_DECISION(self, **kwargs)
    if self.mode != "full":
        return decision
    controller = getattr(self, "_generic_correction_controller", None)
    if isinstance(controller, GenericCorrectionController) and controller.mode != "legacy":
        audio = _application_telemetry(
            controller.application(
                "audio",
                general_confidence=decision.confidence,
            )
        )
        video = _application_telemetry(
            controller.application(
                "video",
                general_confidence=decision.confidence,
            )
        )
    else:
        audio = _generic_telemetry(
            decision.audio_correction_telemetry,
            controller if isinstance(controller, GenericCorrectionController) else None,
            stream="audio",
            confidence=decision.confidence,
        )
        video = _generic_telemetry(
            decision.video_correction_telemetry,
            controller if isinstance(controller, GenericCorrectionController) else None,
            stream="video",
            confidence=decision.confidence,
        )
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
) -> tuple[AnchorEvidence | None, dict[str, torch.Tensor]]:
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or forecaster.feature_shape is None:
        return None, {}
    if tuple(combined.shape) != forecaster.feature_shape:
        raise ValueError("actual feature shape changed during generic correction evidence sampling")
    if len(forecaster._evidence_history) != forecaster.history_length:
        raise RuntimeError("device-local generic correction evidence history is not aligned")

    stream_evidence: dict[str, StreamAnchorEvidence] = {}
    weight_fit_seconds = sample_index_seconds = reduction_seconds = 0.0
    scalar_transfer_seconds = 0.0
    base_weights: dict[str, torch.Tensor] = {}
    controller = getattr(runtime, "_generic_correction_controller", None)
    advanced = (
        isinstance(controller, GenericCorrectionController)
        and controller.mode != "legacy"
    )
    transport_scale = 1.0
    if advanced and len(forecaster._history) >= 2:
        transport_scale, _ = coordinate_transport_scale(
            forecaster._history[-2].coordinate,
            forecaster._history[-1].coordinate,
            step.coordinate,
        )
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
        base_weights[name] = raw_weights
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
        applied_delta_gain = float(gain) * transport_scale if advanced else float(gain)
        corrected_ratio = _tensor_rms(
            actual - (predicted + applied_delta_gain * delta)
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
            generic_corrected_ratio=float(corrected_value),
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
        model_corrected_ratio=0.0,
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
    ), base_weights


_ORIGINAL_RUNTIME_WEIGHT_SEGMENTS = SpectrumH3Runtime._model_aware_weight_segments
_ORIGINAL_RUNTIME_START = SpectrumH3Runtime.start_run
_ORIGINAL_RUNTIME_END = SpectrumH3Runtime.end_run
_ORIGINAL_RUNTIME_RESTORE = SpectrumH3Runtime.restore_rollback_snapshot
_ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
_ORIGINAL_CONTROLLER_SNAPSHOT = ModelAwareController.snapshot
_ORIGINAL_CONTROLLER_RESTORE = ModelAwareController.restore


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
    controller = getattr(self, "_generic_correction_controller", None)
    if (
        isinstance(controller, GenericCorrectionController)
        and controller.mode != "legacy"
    ):
        self._generic_fit_marker = self.forecaster.model_aware_fit_seconds
        return _advanced_weight_segments(
            self,
            call,
            decision,
            coordinate=coordinate,
            controller=controller,
        )
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
        base_weights: dict[str, torch.Tensor] = {}
        legacy_decision = decision
        if mode == "schedule_confidence":
            evidence = _risk_anchor_evidence(self, step, combined, decision)
        else:
            legacy_decision = _ORIGINAL_DECISION(
                self.model_aware,
                forecast_horizon=1.0,
                history_length=self.forecaster.history_length,
                configured_degree=self.config.degree,
                configured_ridge_lambda=self.config.ridge_lambda,
                configured_audio_blend=self.config.audio_blend_weight,
                configured_video_blend=self.config.blend_weight,
            )
            evidence, base_weights = _generic_anchor_evidence(
                self,
                step,
                combined,
                decision,
            )
        if evidence is not None:
            controller = getattr(self, "_generic_correction_controller", None)
            calibration = getattr(self, "_generic_correction_calibration", None)
            exact_required = bool(
                mode == "full"
                and self._offline_phase is None
                and (
                    (
                        isinstance(controller, GenericCorrectionController)
                        and controller.mode != "legacy"
                    )
                    or (
                        isinstance(calibration, GenericCalibrationState)
                        and calibration.enabled
                    )
                )
            )
            if exact_required:
                exact_started = time.perf_counter()
                _exact_anchor_analysis(
                    self,
                    step,
                    combined,
                    decision,
                    legacy_decision,
                    base_weights,
                )
                if isinstance(controller, GenericCorrectionController):
                    controller.overhead_seconds += (
                        time.perf_counter() - exact_started
                    )
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
    if self.config.model_aware_mode == "off":
        self.forecaster._generic_correction_capture_mode = None
        self._generic_correction_controller = None
        self._generic_correction_calibration = None
        self.model_aware._generic_correction_controller = None
        self._generic_fit_marker = 0.0
        return run_id
    self.forecaster._generic_correction_capture_mode = self.config.model_aware_mode
    controller = GenericCorrectionController(
        mode=self.config.generic_correction_mode,
        limiter=self.config.generic_correction_limiter,
        limit=self.config.generic_correction_limit,
        attenuation=self.config.generic_correction_attenuation,
    )
    self._generic_correction_controller = controller
    self.model_aware._generic_correction_controller = controller
    sigmas = args[0] if args else kwargs.get("sigmas")
    if sigmas is None:
        raise RuntimeError("generic correction run setup did not receive a sigma schedule")
    calibration_enabled = bool(
        self.config.debug
        and self.config.model_aware_mode == "full"
        and not self.config.offline_smoothing_replay
        and self._offline_phase is None
    )
    self._generic_correction_calibration = (
        create_calibration_state(self, sigmas, enabled=True)
        if calibration_enabled
        else None
    )
    self._generic_fit_marker = 0.0
    return run_id


def _end_run(self: SpectrumH3Runtime, run_id: int) -> None:
    active = getattr(self, "_run", None)
    if active is None or active.run_id != int(run_id):
        _ORIGINAL_RUNTIME_END(self, run_id)
        return
    calibration = getattr(self, "_generic_correction_calibration", None)
    block = None
    if isinstance(calibration, GenericCalibrationState):
        try:
            block = emit_calibration_block(self, calibration)
        except (TypeError, ValueError) as exc:
            calibration.failures += 1
            LOG.warning(
                "Spectrum H3 generic-correction calibration export failed: %s",
                exc,
            )
    _ORIGINAL_RUNTIME_END(self, run_id)
    self._generic_correction_controller = None
    self._generic_correction_calibration = None
    self.model_aware._generic_correction_controller = None
    if block is not None and block.get("compatible"):
        try:
            result = persist_and_analyze(block)
            duplicate_note = " (duplicate ignored)" if result.duplicate else ""
            LOG.warning("\n%s%s", result.console_summary, duplicate_note)
            LOG.warning(
                "Spectrum H3 generic-correction post-run analysis completed in %.3f s",
                result.elapsed_seconds,
            )
        # Research is strictly post-run and must never invalidate a completed
        # generation, including failures from an unexpected report bug.
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "Spectrum H3 generic-correction research persistence/evaluation failed; "
                "the completed generation remains valid: %s",
                exc,
            )


def _controller_snapshot(self: ModelAwareController) -> dict[str, Any]:
    state = _ORIGINAL_CONTROLLER_SNAPSHOT(self)
    controller = getattr(self, "_generic_correction_controller", None)
    if isinstance(controller, GenericCorrectionController):
        state["generic_correction_controller"] = controller.snapshot()
    return state


def _controller_restore(self: ModelAwareController, state: dict[str, Any]) -> None:
    _ORIGINAL_CONTROLLER_RESTORE(self, state)
    generic_state = state.get("generic_correction_controller")
    if isinstance(generic_state, dict):
        self._generic_correction_controller = (
            GenericCorrectionController.from_snapshot(generic_state)
        )


def _restore_rollback_snapshot(
    self: SpectrumH3Runtime,
    snapshot: Any,
) -> None:
    _ORIGINAL_RUNTIME_RESTORE(self, snapshot)
    controller = getattr(self.model_aware, "_generic_correction_controller", None)
    if isinstance(controller, GenericCorrectionController):
        self._generic_correction_controller = controller


def _remove_summary_field(summary: str, field_pattern: str) -> str:
    return re.sub(
        rf"(?<!\S){field_pattern}=\S+\s*",
        "",
        summary,
    )


def _debug_summary(self: SpectrumH3Runtime) -> str:
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    summary = summary.replace(
        "model_aware_correction_metric=final_layer_exact_linear_head_space",
        "model_aware_correction_metric=generic_latest_delta_hidden_residual_projection",
    )
    for field in _RETIRED_SUMMARY_FIELDS:
        summary = _remove_summary_field(summary, re.escape(field))
    for pattern in _RETIRED_SUMMARY_PATTERNS:
        summary = _remove_summary_field(summary, pattern)
    summary = " ".join(summary.split())
    base = (
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
    controller = getattr(self, "_generic_correction_controller", None)
    if not isinstance(controller, GenericCorrectionController):
        return (
            f"{base} generic_correction_mode={self.config.generic_correction_mode!r} "
            f"generic_correction_attenuation={self.config.generic_correction_attenuation!r} "
            f"generic_correction_attenuation_used={resolve_attenuation_policy(self.config.generic_correction_mode, self.config.generic_correction_attenuation)!r} "
            f"generic_correction_limiter={self.config.generic_correction_limiter!r} "
            f"generic_correction_limit={self.config.generic_correction_limit:.6f} "
            "generic_correction_estimator=legacy_ewma generic_correction_regions=0 "
            "generic_correction_extra_transformer_nfes=0"
        )
    aggregate_count = controller.audio_aggregate.count + controller.video_aggregate.count
    reliability_sum = (
        controller.audio_aggregate.reliability_sum
        + controller.video_aggregate.reliability_sum
    )
    reliability_mean = reliability_sum / aggregate_count if aggregate_count else 0.0
    minima = [
        item.resolved_reliability_min
        for item in (controller.audio_aggregate, controller.video_aggregate)
        if item.count
    ]
    maxima = [
        item.reliability_max
        for item in (controller.audio_aggregate, controller.video_aggregate)
        if item.count
    ]
    estimator = "legacy_ewma" if controller.mode == "legacy" else "ewls_rls_lambda_0.90"
    geometry = (
        "legacy_latest_delta"
        if controller.mode == "legacy"
        else "coordinate_transported_latest_delta"
    )
    return (
        f"{base} "
        f"generic_correction_mode={controller.mode!r} "
        f"generic_correction_attenuation={controller.attenuation!r} "
        f"generic_correction_attenuation_used={controller.resolved_attenuation!r} "
        f"generic_correction_geometry={geometry} "
        f"generic_correction_estimator={estimator} "
        f"generic_correction_coordinate_active={controller.coordinate_active_count} "
        f"generic_correction_coordinate_fallbacks={controller.coordinate_fallback_count} "
        f"generic_correction_reliability_mean={reliability_mean:.6f} "
        f"generic_correction_reliability_min={min(minima, default=0.0):.6f} "
        f"generic_correction_reliability_max={max(maxima, default=0.0):.6f} "
        f"generic_correction_limiter={controller.limiter!r} "
        f"generic_correction_limit={controller.limit:.6f} "
        f"generic_correction_scope={'regional' if controller.mode == 'regional' else 'global'} "
        f"generic_correction_regions={len(controller.regions)} "
        f"generic_correction_regional_active={controller.regional_active_count} "
        f"generic_correction_regional_fallbacks={controller.regional_fallback_count} "
        f"generic_correction_audio_raw_gain_mean={controller.audio_aggregate.raw_mean:.6f} "
        f"generic_correction_audio_applied_gain_mean={controller.audio_aggregate.applied_mean:.6f} "
        f"generic_correction_audio_estimator_support={controller.audio.rls.support:.6f} "
        f"generic_correction_audio_estimator_energy={controller.audio.rls.c_acc:.6e} "
        f"generic_correction_audio_estimator_age={controller.audio.rls.effective_age:.6f} "
        f"generic_correction_video_raw_gain_mean={controller.video_aggregate.raw_mean:.6f} "
        f"generic_correction_video_applied_gain_mean={controller.video_aggregate.applied_mean:.6f} "
        f"generic_correction_video_estimator_support={controller.video.rls.support:.6f} "
        f"generic_correction_video_estimator_energy={controller.video.rls.c_acc:.6e} "
        f"generic_correction_video_estimator_age={controller.video.rls.effective_age:.6f} "
        f"generic_correction_overhead_s={controller.overhead_seconds:.6f} "
        "generic_correction_extra_transformer_nfes=0"
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
    SpectrumH3Runtime.end_run = _end_run
    SpectrumH3Runtime.restore_rollback_snapshot = _restore_rollback_snapshot
    SpectrumH3Runtime.debug_summary = _debug_summary
    ModelAwareController.snapshot = _controller_snapshot
    ModelAwareController.restore = _controller_restore
    SpectrumH3Runtime._generic_residual_correction_installed = True


__all__ = ["install_generic_residual_correction"]
