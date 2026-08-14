from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .config import SpectrumH3Config
from .experiments import (
    OfflineFeatureArchive,
    OfflineSmoother,
    StreamResidualScore,
    measure_stream_residual,
    tensor_all_finite,
)
from .forecast import ForecasterSnapshot, HistoryWeightForecaster
from .model_aware import (
    ModelAwareController,
    ModelAwareForecastDecision,
    ModelForecastabilityProfile,
    ProfileLookup,
)

LOG = logging.getLogger(__name__)

_FEEDBACK_SCORE_THRESHOLD = 1.5
_FEEDBACK_MAX_REFRESHES = 3
_ROLLBACK_SCORE_THRESHOLD = 1.5
_ROLLBACK_MAX_CORRECTIONS = 3


def _as_cpu_float64_vector(value: Any) -> torch.Tensor:
    """Detach a tensor-like value, move it to CPU, then cast to float64."""
    return (
        torch.as_tensor(value)
        .detach()
        .to(device="cpu")
        .to(dtype=torch.float64)
        .reshape(-1)
    )


class ForecastRetryActual(RuntimeError):
    """Internal signal used to discard a partial forecast attempt transactionally."""


class OfflineReplayAbort(RuntimeError):
    """Internal signal used to return the valid first-pass result after replay failure."""


@dataclass(slots=True)
class RuntimeStats:
    run_id: int = 0
    sampler_name: str = "unknown"
    total_steps: int = 0
    actual_steps: int = 0
    forecast_steps: int = 0
    actual_transformer_calls: int = 0
    forecast_model_calls: int = 0
    forecast_fallbacks: int = 0
    bypassed_steps: int = 0
    causal_video_blend_weight: float = 0.0
    causal_audio_blend_weight: float = 0.0
    history_archive_seconds: float = 0.0
    history_update_seconds: float = 0.0
    forecast_prediction_seconds: float = 0.0
    direct_history_updates: int = 0
    current_window: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None
    residual_measure_seconds: float = 0.0
    residual_output_head_seconds: float = 0.0
    residual_anchors: int = 0
    residual_failures: int = 0
    residual_skipped_terminal_probes: int = 0
    residual_max_score: float = 0.0
    residual_max_video_score: float = 0.0
    residual_max_audio_score: float = 0.0
    residual_policy_max_score: float = 0.0
    feedback_refreshes: int = 0
    feedback_suppressed_threshold: int = 0
    feedback_suppressed_budget: int = 0
    speculative_forecast_calls: int = 0
    discarded_actual_calls: int = 0
    rollback_count: int = 0
    rollback_suppressed_threshold: int = 0
    rollback_suppressed_budget: int = 0
    replayed_transformer_calls: int = 0
    offline_archive_bytes: int = 0
    offline_estimated_archive_bytes: int = 0
    offline_archive_seconds: float = 0.0
    offline_smoother_build_seconds: float = 0.0
    offline_replay_steps: int = 0
    offline_replay_model_calls: int = 0
    offline_replay_anchor_steps: int = 0
    offline_replay_smoothed_steps: int = 0
    offline_validation_samples_per_branch: int = 0
    offline_validation_anchors: int = 0
    offline_validation_streams: int = 0
    offline_validation_seconds: float = 0.0
    offline_validation_audio_max: float = 0.0
    offline_validation_video_max: float = 0.0
    offline_validation_packed_max: float = 0.0
    offline_attenuated_predictions: int = 0
    offline_local_only_predictions: int = 0
    offline_effective_blend_min: float = 0.0
    offline_effective_blend_mean: float = 0.0
    offline_effective_blend_max: float = 0.0
    offline_effective_audio_blend_min: float = 0.0
    offline_effective_audio_blend_mean: float = 0.0
    offline_effective_audio_blend_max: float = 0.0
    offline_effective_video_blend_min: float = 0.0
    offline_effective_video_blend_mean: float = 0.0
    offline_effective_video_blend_max: float = 0.0
    offline_attenuated_audio_predictions: int = 0
    offline_attenuated_video_predictions: int = 0
    offline_local_only_audio_predictions: int = 0
    offline_local_only_video_predictions: int = 0
    adaptive_extra_nfes: int = 0
    model_profile_cache_hit: bool = False
    model_profile_build_seconds: float = 0.0
    model_profile_lookup_seconds: float = 0.0
    model_profile_bytes: int = 0
    model_profile_workspace_bytes: int = 0
    model_profile_patch_count: int = 0
    model_profile_unknown_patch_count: int = 0
    model_profile_sensitivity: float = 0.0
    model_profile_patch_perturbation: float = 0.0
    model_aware_forecasts: int = 0
    model_aware_anchor_updates: int = 0
    model_aware_failures: int = 0
    model_aware_risk_max: float = 0.0
    model_aware_confidence_min: float = 1.0
    model_aware_correction_max: float = 0.0
    model_aware_overhead_seconds: float = 0.0
    model_aware_decision_seconds: float = 0.0
    model_aware_evidence_seconds: float = 0.0
    model_aware_evidence_weight_fit_seconds: float = 0.0
    model_aware_evidence_sample_index_seconds: float = 0.0
    model_aware_evidence_device_transfer_seconds: float = 0.0
    model_aware_evidence_sensitivity_transfer_seconds: float = 0.0
    model_aware_evidence_scalar_transfer_seconds: float = 0.0
    model_aware_evidence_reduction_seconds: float = 0.0
    model_aware_evidence_exact_head_projection_seconds: float = 0.0
    model_aware_evidence_fit_condition_seconds: float = 0.0
    model_aware_subspace_gram_seconds: float = 0.0
    model_aware_subspace_solve_seconds: float = 0.0
    model_aware_subspace_workspace_bytes: int = 0
    model_aware_head_materialization_seconds: float = 0.0
    model_aware_head_materialized_bytes: int = 0
    model_aware_exact_head_projection_seconds: float = 0.0
    model_aware_exact_head_projection_calls: int = 0
    model_aware_exact_head_workspace_bytes: int = 0
    model_aware_fit_seconds: float = 0.0
    model_aware_correction_seconds: float = 0.0
    model_aware_causal_correction_seconds: float = 0.0
    model_aware_offline_correction_seconds: float = 0.0
    model_aware_offline_correction_applications: int = 0
    model_aware_model_corrected_ratio_mean: float = 0.0
    model_aware_generic_corrected_ratio_mean: float = 0.0


@dataclass(slots=True)
class _ActualRecord:
    feature: torch.Tensor
    labels: tuple[Any, ...] | None


@dataclass(slots=True)
class ResidualProbe:
    shadow: torch.Tensor
    hold: torch.Tensor


@dataclass(slots=True)
class _ResidualRecord:
    labels: tuple[Any, ...]
    video_score: StreamResidualScore
    audio_score: StreamResidualScore


@dataclass(slots=True)
class _AggregatedResidual:
    policy_score: float
    video_score: float
    audio_score: float


@dataclass(slots=True)
class _CallState:
    topology: tuple[Any, ...]
    labels: tuple[Any, ...] | None
    expected_shape: tuple[int, ...]
    observed_actual: bool = False
    used_forecast: bool = False


@dataclass(slots=True)
class _StepState:
    step_id: int
    coordinate: float
    adaptive_recompute: bool
    mode: str
    reason: str
    bootstrap_forecast: bool = False
    calls: list[_CallState] = field(default_factory=list)
    actual_records: list[_ActualRecord] = field(default_factory=list)
    used_history_rows: set[int] = field(default_factory=set)
    fallback: bool = False
    residual_expected: bool = False
    residual_records: list[_ResidualRecord] = field(default_factory=list)
    rollback_replay: bool = False
    consumes_feedback_refresh: bool = False
    residual_skip_reason: str | None = None
    model_aware_decision: ModelAwareForecastDecision | None = None
    model_aware_forced_actual: bool = False


@dataclass(slots=True)
class _RunState:
    run_id: int
    sampler_name: str
    total_steps: int
    sigma_min: float
    sigma_max: float
    supported_sampler: bool
    max_consecutive_forecasts: int | None
    min_actual_steps_after_forecast: int
    min_tail_actual_steps: int
    min_actual_prefix_steps: int
    next_step_id: int = 0


@dataclass(slots=True)
class RuntimeRollbackSnapshot:
    next_step_id: int
    forecaster: ForecasterSnapshot
    history_topology: tuple[Any, ...] | None
    history_labels: tuple[Any, ...] | None
    current_window: float
    consecutive_forecasts: int
    required_actual_refreshes: int
    required_feedback_actuals: int
    disabled: bool
    disable_reason: str | None
    experiment_disabled: bool
    experiment_disable_reason: str | None
    last_completed_mode: str | None
    last_completed_step_id: int | None
    model_aware_state: dict[str, float | int]
    stats: RuntimeStats


class SpectrumH3Runtime:
    def __init__(self, config: SpectrumH3Config):
        self.config = config.validate()
        self.forecaster = HistoryWeightForecaster(
            degree=self.config.degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.max_history,
            history_storage=self.config.history_storage,
        )
        self.model_aware = ModelAwareController(
            self.config.model_aware_mode,
            self.config.model_aware_risk_threshold,
        )

        self.stats = RuntimeStats(current_window=self.config.window_size)
        self._run_counter = 0
        self._run: _RunState | None = None
        self._step: _StepState | None = None
        self._history_topology: tuple[Any, ...] | None = None
        self._history_labels: tuple[Any, ...] | None = None
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._disabled = False
        self._disable_reason: str | None = None
        self._experiment_disabled = False
        self._experiment_disable_reason: str | None = None
        self._last_completed_mode: str | None = None
        self._last_completed_step_id: int | None = None
        self._rollback_requested = False
        self._forced_actual_reason: str | None = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        self._offline_phase: str | None = None
        self._offline_archive: OfflineFeatureArchive | None = None
        self._offline_smoother: OfflineSmoother | None = None
        self._offline_archive_seconds_total = 0.0
        self._offline_smoother_build_seconds_total = 0.0
        self._model_profile: ModelForecastabilityProfile | None = None
        self._model_profile_cache_hit = False
        self._model_profile_lookup_seconds = 0.0
        self._model_aware_disabled_reason: str | None = None

    @property
    def active_run_id(self) -> int | None:
        return None if self._run is None else self._run.run_id

    @property
    def active_step_id(self) -> int | None:
        return None if self._step is None else self._step.step_id

    @property
    def active_model_aware_decision(self) -> ModelAwareForecastDecision | None:
        return None if self._step is None else self._step.model_aware_decision

    @property
    def supported_sampler(self) -> bool:
        return self._run is not None and self._run.supported_sampler

    @property
    def disabled_reason(self) -> str | None:
        return self._disable_reason

    @property
    def history_labels(self) -> tuple[Any, ...] | None:
        return self._history_labels

    @property
    def last_completed_mode(self) -> str | None:
        return self._last_completed_mode

    @property
    def last_completed_step_id(self) -> int | None:
        return self._last_completed_step_id

    @property
    def experiment_disabled_reason(self) -> str | None:
        return self._experiment_disable_reason

    @property
    def offline_archive(self) -> OfflineFeatureArchive | None:
        return self._offline_archive

    @property
    def offline_phase(self) -> str | None:
        return self._offline_phase

    @property
    def prediction_history_length(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_length
        return self.forecaster.history_length

    @property
    def prediction_history_device(self) -> torch.device | None:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_device
        return self.forecaster.history_device

    @property
    def prediction_history_tensor_bytes(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_tensor_bytes
        return self.forecaster.history_tensor_bytes

    @property
    def last_prediction_chunk_count(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.last_prediction_chunk_count
        return self.forecaster.last_prediction_chunk_count

    def record_residual_output_head_seconds(self, elapsed: float) -> None:
        self.stats.residual_output_head_seconds += max(0.0, float(elapsed))

    def set_model_profile(self, lookup: ProfileLookup) -> None:
        if self._run is not None:
            raise RuntimeError("cannot replace the model forecastability profile during a run")
        self._model_profile = lookup.profile
        self._model_profile_cache_hit = bool(lookup.cache_hit)
        self._model_profile_lookup_seconds = max(0.0, float(lookup.lookup_seconds))
        self._model_aware_disabled_reason = None
        self.model_aware.set_profile(lookup.profile)

    def disable_model_aware(self, reason: str) -> None:
        self._model_aware_disabled_reason = str(reason)
        self._model_profile = None
        self._model_profile_cache_hit = False
        self._model_profile_lookup_seconds = 0.0
        self.model_aware.set_profile(None)

    @property
    def model_profile(self) -> ModelForecastabilityProfile | None:
        return self._model_profile

    def _model_aware_enabled(self) -> bool:
        return bool(
            self.config.model_aware_mode != "off"
            and self._model_aware_disabled_reason is None
            and self._model_profile is not None
        )

    def _record_offline_smoother_stats(self) -> None:
        smoother = self._offline_smoother
        if smoother is None:
            return
        self.stats.offline_validation_samples_per_branch = smoother.validation_samples_per_branch
        self.stats.offline_validation_anchors = smoother.validation_anchor_count
        self.stats.offline_validation_streams = smoother.validation_stream_count
        self.stats.offline_validation_seconds = smoother.validation_seconds
        self.stats.offline_validation_audio_max = smoother.validation_stream_max_scores.get("audio", 0.0)
        self.stats.offline_validation_video_max = smoother.validation_stream_max_scores.get("video", 0.0)
        self.stats.offline_validation_packed_max = smoother.validation_stream_max_scores.get("packed", 0.0)
        self.stats.offline_attenuated_predictions = smoother.attenuated_prediction_count
        self.stats.offline_local_only_predictions = smoother.local_only_prediction_count
        self.stats.offline_effective_blend_min = smoother.effective_blend_min
        self.stats.offline_effective_blend_mean = smoother.effective_blend_mean
        self.stats.offline_effective_blend_max = smoother.effective_blend_max
        audio_blends = smoother.effective_blend_stream_stats.get("audio", (0.0, 0.0, 0.0))
        video_blends = smoother.effective_blend_stream_stats.get("video", (0.0, 0.0, 0.0))
        (
            self.stats.offline_effective_audio_blend_min,
            self.stats.offline_effective_audio_blend_mean,
            self.stats.offline_effective_audio_blend_max,
        ) = audio_blends
        (
            self.stats.offline_effective_video_blend_min,
            self.stats.offline_effective_video_blend_mean,
            self.stats.offline_effective_video_blend_max,
        ) = video_blends
        self.stats.offline_attenuated_audio_predictions = smoother.attenuated_prediction_counts.get("audio", 0)
        self.stats.offline_attenuated_video_predictions = smoother.attenuated_prediction_counts.get("video", 0)
        self.stats.offline_local_only_audio_predictions = smoother.local_only_prediction_counts.get("audio", 0)
        self.stats.offline_local_only_video_predictions = smoother.local_only_prediction_counts.get("video", 0)
        previous_model_aware_fit = self.stats.model_aware_fit_seconds
        self.stats.model_aware_fit_seconds = smoother.model_aware_fit_seconds
        previous_offline_correction = self.stats.model_aware_offline_correction_seconds
        self.stats.model_aware_offline_correction_seconds = (
            smoother.model_aware_offline_correction_seconds
        )
        self.stats.model_aware_offline_correction_applications = (
            smoother.model_aware_offline_correction_applications
        )
        self.stats.model_aware_correction_seconds = (
            self.stats.model_aware_causal_correction_seconds
            + self.stats.model_aware_offline_correction_seconds
        )
        self.stats.model_aware_overhead_seconds += max(
            0.0,
            self.stats.model_aware_fit_seconds - previous_model_aware_fit,
        ) + max(
            0.0,
            self.stats.model_aware_offline_correction_seconds
            - previous_offline_correction,
        )

    def _stream_ranges(self, call: _CallState) -> tuple[tuple[str, int, int], ...]:
        topology = {
            str(entry[0]): entry[1]
            for entry in call.topology
            if isinstance(entry, tuple) and len(entry) == 2
        }
        audio_rows = topology.get("target_audio_rows")
        video_rows = topology.get("target_video_rows")
        target_rows = call.expected_shape[1]
        if (
            isinstance(audio_rows, int)
            and isinstance(video_rows, int)
            and audio_rows > 0
            and video_rows > 0
            and audio_rows + video_rows == target_rows
        ):
            return (
                ("audio", 0, audio_rows),
                ("video", audio_rows, target_rows),
            )
        return (("packed", 0, target_rows),)

    def _model_aware_head_metrics(
        self,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.config.model_aware_mode != "full":
            return {}, {}
        weights: dict[str, torch.Tensor] = {}
        diagonals: dict[str, torch.Tensor] = {}
        materialization_seconds = 0.0
        for stream in ("audio", "video"):
            weight, diagonal, elapsed = self.model_aware.head_metric_tensors(
                stream,
                device,
            )
            materialization_seconds += elapsed
            if weight is not None and diagonal is not None:
                weights[stream] = weight
                diagonals[stream] = diagonal
        self.stats.model_aware_head_materialization_seconds += materialization_seconds
        self.stats.model_aware_head_materialized_bytes = (
            self.model_aware.materialized_head_bytes
        )
        self.stats.model_aware_overhead_seconds += materialization_seconds
        return weights, diagonals

    def _prediction_segments(self, call: _CallState) -> tuple[tuple[int, int, float], ...]:
        audio_blend_weight, video_blend_weight = self._causal_prediction_blends()
        ranges = self._stream_ranges(call)
        if len(ranges) == 2:
            return (
                (ranges[0][1], ranges[0][2], audio_blend_weight),
                (ranges[1][1], ranges[1][2], video_blend_weight),
            )
        if math.isclose(
            audio_blend_weight,
            video_blend_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return ((ranges[0][1], ranges[0][2], video_blend_weight),)
        raise ValueError("packed H3 topology does not expose the target audio/video boundary")

    def _model_aware_weight_segments(
        self,
        call: _CallState,
        decision: ModelAwareForecastDecision,
        *,
        coordinate: float,
    ) -> tuple[tuple[int, int, torch.Tensor], ...]:
        fit_before = self.forecaster.model_aware_fit_seconds
        correction_before = self.forecaster.model_aware_correction_seconds
        weighted = []
        for name, start, end in self._stream_ranges(call):
            if name == "audio":
                blend = decision.audio_blend_weight
                subspace = decision.audio_subspace_telemetry
                correction = (
                    0.0 if subspace.eligible else decision.audio_correction_gain
                )
                coefficients = (
                    subspace.applied_coefficients if subspace.eligible else ()
                )
            elif name == "video":
                blend = decision.video_blend_weight
                subspace = decision.video_subspace_telemetry
                correction = (
                    0.0 if subspace.eligible else decision.video_correction_gain
                )
                coefficients = (
                    subspace.applied_coefficients if subspace.eligible else ()
                )
            else:
                if not math.isclose(
                    decision.audio_blend_weight,
                    decision.video_blend_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "model-aware packed prediction requires target audio/video row metadata"
                    )
                blend = decision.video_blend_weight
                if (
                    decision.audio_subspace_telemetry.eligible
                    and decision.video_subspace_telemetry.eligible
                ):
                    correction = 0.0
                    coefficients = tuple(
                        0.5
                        * (
                            decision.audio_subspace_telemetry.applied_coefficients[index]
                            + decision.video_subspace_telemetry.applied_coefficients[index]
                        )
                        for index in range(2)
                    )
                else:
                    correction = max(
                        -0.25,
                        min(
                            0.25,
                            0.5
                            * (
                                decision.audio_correction_gain
                                + decision.video_correction_gain
                            ),
                        ),
                    )
                    coefficients = ()
            weights = self.forecaster.model_aware_weights(
                coordinate,
                blend,
                degree=decision.degree,
                ridge_lambda=decision.ridge_lambda,
                correction_gain=correction,
                correction_coefficients=coefficients,
                correction_anchor_ids=(
                    decision.correction_anchor_ids if coefficients else ()
                ),
            )
            weighted.append((start, end, weights))
        fit_elapsed = self.forecaster.model_aware_fit_seconds - fit_before
        correction_elapsed = (
            self.forecaster.model_aware_correction_seconds - correction_before
        )
        self.stats.model_aware_fit_seconds += max(0.0, fit_elapsed)
        correction_elapsed = max(0.0, correction_elapsed)
        self.stats.model_aware_causal_correction_seconds += correction_elapsed
        self.stats.model_aware_correction_seconds = (
            self.stats.model_aware_causal_correction_seconds
            + self.stats.model_aware_offline_correction_seconds
        )
        self.stats.model_aware_overhead_seconds += max(
            0.0,
            fit_elapsed + correction_elapsed,
        )
        return tuple(weighted)

    def _causal_prediction_blends(self) -> tuple[float, float]:
        if self._offline_phase in {"first_pass", "replay"}:
            return 0.0, 0.0
        return self.config.audio_blend_weight, self.config.blend_weight

    def _residual_experiment_enabled(self) -> bool:
        return bool(
            not self._experiment_disabled
            and (
                self.config.anchor_residual_feedback
                or self.config.selective_rollback_correction
            )
        )

    def disable_experiment(self, reason: str) -> bool:
        newly_disabled = not self._experiment_disabled
        self._experiment_disabled = True
        self._experiment_disable_reason = str(reason)
        self._rollback_requested = False
        self._required_feedback_actuals = 0
        if newly_disabled:
            self.stats.residual_failures += 1
            LOG.warning("Spectrum H3 experimental mode disabled for this run: %s", reason)
        return newly_disabled

    def start_run(
        self,
        sigmas: torch.Tensor,
        sampler_name: str,
        *,
        supported_sampler: bool,
        max_consecutive_forecasts: int | None = None,
        min_actual_steps_after_forecast: int = 0,
        min_tail_actual_steps: int = 0,
        min_actual_prefix_steps: int = 0,
    ) -> int:
        if self._run is not None:
            raise RuntimeError("Spectrum H3 runtime already has an active run")
        if max_consecutive_forecasts is not None and (
            isinstance(max_consecutive_forecasts, bool)
            or not isinstance(max_consecutive_forecasts, int)
            or max_consecutive_forecasts < 1
        ):
            raise ValueError("max_consecutive_forecasts must be None or an integer >= 1")
        for name, value in (
            ("min_actual_steps_after_forecast", min_actual_steps_after_forecast),
            ("min_tail_actual_steps", min_tail_actual_steps),
            ("min_actual_prefix_steps", min_actual_prefix_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        sigma_values = _as_cpu_float64_vector(sigmas)
        total_steps = max(0, sigma_values.numel() - 1)
        evaluated = sigma_values[:-1]
        finite_schedule = bool(evaluated.numel()) and bool(torch.isfinite(evaluated).all().item())
        sigma_min = float(evaluated.min().item()) if finite_schedule else 0.0
        sigma_max = float(evaluated.max().item()) if finite_schedule else 0.0
        schedule_valid = finite_schedule and math.isfinite(sigma_min) and math.isfinite(sigma_max) and sigma_max > sigma_min

        self._run_counter += 1
        effective_supported = bool(
            self.config.enabled and supported_sampler and schedule_valid and total_steps > 0
        )
        self._run = _RunState(
            run_id=self._run_counter,
            sampler_name=str(sampler_name),
            total_steps=total_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            supported_sampler=effective_supported,
            max_consecutive_forecasts=max_consecutive_forecasts,
            min_actual_steps_after_forecast=min_actual_steps_after_forecast,
            min_tail_actual_steps=min_tail_actual_steps,
            min_actual_prefix_steps=min(min_actual_prefix_steps, total_steps),
        )
        self._step = None
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._disabled = not effective_supported
        self._experiment_disabled = False
        self._experiment_disable_reason = None
        self._last_completed_mode = None
        self._last_completed_step_id = None
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        self.model_aware.reset()
        self.model_aware.set_profile(self._model_profile)
        if not self.config.enabled:
            self._disable_reason = "forecasting disabled by configuration"
        elif not supported_sampler:
            self._disable_reason = f"sampler {sampler_name!r} is not allowlisted for one-call solver-step tracking"
        elif not schedule_valid:
            self._disable_reason = "supplied sigma schedule is empty, nonfinite, or has no usable range"
        elif total_steps <= 0:
            self._disable_reason = "supplied sigma schedule has no solver steps"
        else:
            self._disable_reason = None
        causal_audio_blend, causal_video_blend = self._causal_prediction_blends()
        self.stats = RuntimeStats(
            run_id=self._run.run_id,
            sampler_name=self._run.sampler_name,
            total_steps=total_steps,
            current_window=self._current_window,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
            causal_video_blend_weight=causal_video_blend,
            causal_audio_blend_weight=causal_audio_blend,
        )
        if self._model_profile is not None:
            self.stats.model_profile_cache_hit = self._model_profile_cache_hit
            self.stats.model_profile_build_seconds = (
                0.0 if self._model_profile_cache_hit else self._model_profile.build_seconds
            )
            self.stats.model_profile_lookup_seconds = self._model_profile_lookup_seconds
            self.stats.model_profile_bytes = self._model_profile.estimated_bytes
            self.stats.model_profile_workspace_bytes = (
                self._model_profile.transient_workspace_bytes
            )
            self.stats.model_profile_patch_count = self._model_profile.active_patch_count
            self.stats.model_profile_unknown_patch_count = self._model_profile.unknown_patch_count
            self.stats.model_profile_sensitivity = self._model_profile.aggregate_sensitivity
            self.stats.model_profile_patch_perturbation = self._model_profile.patch_perturbation
        if self._offline_phase == "replay" and self._offline_archive is not None:
            self.stats.offline_archive_bytes = self._offline_archive.tensor_bytes
            self.stats.offline_estimated_archive_bytes = self._offline_archive.estimated_tensor_bytes
            self.stats.offline_archive_seconds = self._offline_archive_seconds_total
            self.stats.offline_smoother_build_seconds = self._offline_smoother_build_seconds_total
            self._record_offline_smoother_stats()
        return self._run.run_id

    def end_run(self, run_id: int) -> None:
        if self._run is None:
            return
        if self._run.run_id != int(run_id):
            raise RuntimeError("attempted to end a stale Spectrum H3 run")
        self._step = None
        self._run = None
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        self._last_completed_mode = None
        self._last_completed_step_id = None

    def begin_offline_capture(self, *, total_steps: int, sampler_name: str) -> None:
        self.release_offline_archive()
        self._offline_archive_seconds_total = 0.0
        self._offline_smoother_build_seconds_total = 0.0
        self._offline_phase = "first_pass"
        self._offline_archive = OfflineFeatureArchive(
            total_steps=total_steps,
            sampler_name=sampler_name,
            history_storage=self.config.offline_archive_storage,
        )

    def complete_offline_capture(self) -> bool:
        archive = self._offline_archive
        if self._offline_phase != "first_pass" or archive is None:
            raise RuntimeError("offline capture is not active")
        if self._disabled:
            archive.invalidate(self._disable_reason or "base Spectrum disabled during offline first pass")
        complete = archive.complete(minimum_anchors=self.config.min_fit_points)
        self.stats.offline_archive_bytes = archive.tensor_bytes
        self.stats.offline_estimated_archive_bytes = archive.estimated_tensor_bytes
        if not complete:
            return False
        started = time.perf_counter()
        try:
            self._offline_smoother = OfflineSmoother(
                archive,
                degree=self.config.degree,
                ridge_lambda=self.config.ridge_lambda,
                blend_weight=self.config.blend_weight,
                audio_blend_weight=self.config.audio_blend_weight,
            )
            self._record_offline_smoother_stats()
        except (RuntimeError, ValueError) as exc:
            archive.invalidate(f"offline smoother construction failed: {exc}")
            return False
        finally:
            elapsed = time.perf_counter() - started
            self._offline_smoother_build_seconds_total += elapsed
            self.stats.offline_smoother_build_seconds += elapsed
        return True

    def begin_offline_replay(self) -> None:
        if self._offline_archive is None or self._offline_smoother is None:
            raise RuntimeError("offline replay requires a complete first-pass archive")
        self._offline_phase = "replay"

    def release_offline_archive(self) -> None:
        if self._offline_archive is not None:
            self._offline_archive.release()
        self._offline_archive = None
        self._offline_smoother = None
        self._offline_phase = None

    def coordinate_for_timestep(self, timestep: torch.Tensor | float) -> float:
        if self._run is None:
            raise RuntimeError("Spectrum H3 runtime is outside a sampling run")
        value_tensor = _as_cpu_float64_vector(timestep)
        if value_tensor.numel() == 0 or not bool(torch.isfinite(value_tensor).all().item()):
            raise RuntimeError("current solver timestep is empty or nonfinite")
        if not bool(torch.allclose(value_tensor, value_tensor[0].expand_as(value_tensor))):
            raise RuntimeError("current predict_noise call contains multiple solver timesteps")
        value = float(value_tensor[0].item())
        sigma_span = self._run.sigma_max - self._run.sigma_min
        if not math.isfinite(sigma_span) or sigma_span <= 0.0:
            return 0.0
        coordinate = 2.0 * (value - self._run.sigma_min) / sigma_span - 1.0
        return float(max(-1.0, min(1.0, coordinate)))

    def begin_step(self, timestep: torch.Tensor | float) -> dict[str, Any]:
        if self._run is None:
            raise RuntimeError("Spectrum H3 runtime is outside a sampling run")
        if self._step is not None:
            raise RuntimeError("previous Spectrum H3 solver step was not finalized")
        step_id = self._run.next_step_id
        if step_id >= self._run.total_steps:
            raise RuntimeError("predict_noise call count exceeded the supplied sigma schedule")
        coordinate = self.coordinate_for_timestep(timestep)

        if self._offline_phase == "replay":
            archive = self._offline_archive
            if archive is None or self._offline_smoother is None or step_id >= len(archive.steps):
                raise OfflineReplayAbort("offline replay archive is incomplete")
            record = archive.steps[step_id]
            if not math.isclose(record.coordinate, coordinate, rel_tol=1e-6, abs_tol=1e-6):
                raise OfflineReplayAbort(
                    f"offline replay coordinate changed at step {step_id}: "
                    f"{record.coordinate:.9f} != {coordinate:.9f}"
                )
            self._step = _StepState(
                step_id=step_id,
                coordinate=coordinate,
                adaptive_recompute=False,
                mode="replay",
                reason="offline smoothing replay",
            )
            self._run.next_step_id += 1
            return {
                "run_id": self._run.run_id,
                "step_id": step_id,
                "coordinate": coordinate,
                "actual": False,
                "reason": "offline smoothing replay",
            }

        effective_tail = max(self.config.tail_actual_steps, self._run.min_tail_actual_steps)
        tail_start = max(0, self._run.total_steps - effective_tail)
        advances_window = False
        bootstrap_forecast = False
        rollback_replay = False
        consumes_feedback_refresh = False
        if self.config.force_actual:
            actual, reason = True, "forced-actual validation mode"
        elif self._disabled:
            actual, reason = True, self._disable_reason or "forecasting disabled"
        elif self._forced_actual_reason is not None:
            actual, reason = True, self._forced_actual_reason
            rollback_replay = self._forced_actual_is_replay
            self._forced_actual_reason = None
            self._forced_actual_is_replay = False
        elif self._required_feedback_actuals > 0:
            actual, reason = True, "anchor residual feedback refresh"
            rollback_replay = False
            consumes_feedback_refresh = True
        elif step_id < self._run.min_actual_prefix_steps:
            actual, reason = True, "H3 Continuum actual prefix"
        elif step_id < self.config.warmup_steps:
            actual, reason = True, "warmup"
        elif step_id >= tail_start:
            actual, reason = True, "final actual tail"
        elif (
            self.config.bootstrap_first_forecast
            and self.config.degree == 1
            and step_id == 1
            and self.forecaster.history_length == 1
        ):
            actual, reason = False, "one-point bootstrap forecast"
            bootstrap_forecast = True
        elif not self.forecaster.ready(self.config.min_fit_points):
            actual, reason = True, "insufficient actual history"
        else:
            interval = max(1, math.floor(self._current_window))
            actual = ((self._consecutive_forecasts + 1) % interval) == 0
            reason = "adaptive recompute" if actual else "adaptive forecast"
            advances_window = actual

        forecast_limit = self._run.max_consecutive_forecasts
        if not actual and forecast_limit is not None and self._consecutive_forecasts >= forecast_limit:
            actual = True
            reason = "post-forecast sampler refresh"
            advances_window = False
            bootstrap_forecast = False
        if not actual and self._required_actual_refreshes > 0:
            actual = True
            reason = "post-forecast sampler refresh"
            advances_window = False
            bootstrap_forecast = False

        model_aware_decision = None
        model_aware_forced_actual = False
        if not actual and self._model_aware_enabled():
            model_aware_started = time.perf_counter()
            try:
                model_aware_decision = self.model_aware.decision(
                    forecast_horizon=float(self._consecutive_forecasts + 1),
                    history_length=self.forecaster.history_length,
                    configured_degree=self.config.degree,
                    configured_ridge_lambda=self.config.ridge_lambda,
                    configured_audio_blend=self.config.audio_blend_weight,
                    configured_video_blend=self.config.blend_weight,
                )
                if (
                    model_aware_decision.audio_subspace_telemetry.eligible
                    or model_aware_decision.video_subspace_telemetry.eligible
                ):
                    anchor_ids = self.forecaster.latest_anchor_ids(3)
                    model_aware_decision = replace(
                        model_aware_decision,
                        correction_anchor_ids=anchor_ids,
                    )
                self.stats.model_aware_risk_max = max(
                    self.stats.model_aware_risk_max,
                    model_aware_decision.combined_risk,
                )
                self.stats.model_aware_confidence_min = min(
                    self.stats.model_aware_confidence_min,
                    model_aware_decision.confidence,
                )
                self.stats.model_aware_correction_max = max(
                    self.stats.model_aware_correction_max,
                    abs(model_aware_decision.audio_correction_gain),
                    abs(model_aware_decision.video_correction_gain),
                    model_aware_decision.audio_subspace_telemetry.applied_bounded_norm_ratio,
                    model_aware_decision.video_subspace_telemetry.applied_bounded_norm_ratio,
                )
                if model_aware_decision.force_actual:
                    actual = True
                    reason = "model-aware forecast risk"
                    advances_window = False
                    bootstrap_forecast = False
                    model_aware_forced_actual = True
            except (RuntimeError, TypeError, ValueError) as exc:
                self._model_aware_disabled_reason = f"model-aware decision failed: {exc}"
                self.stats.model_aware_failures += 1
                LOG.warning(
                    "Spectrum H3 model-aware forecasting disabled for this run: %s",
                    self._model_aware_disabled_reason,
                )
            finally:
                elapsed = time.perf_counter() - model_aware_started
                self.stats.model_aware_decision_seconds += elapsed
                self.stats.model_aware_overhead_seconds += elapsed

        self._step = _StepState(
            step_id=step_id,
            coordinate=coordinate,
            adaptive_recompute=advances_window,
            mode="actual" if actual else "forecast",
            reason=reason,
            bootstrap_forecast=bootstrap_forecast,
            rollback_replay=rollback_replay,
            consumes_feedback_refresh=consumes_feedback_refresh,
            model_aware_decision=model_aware_decision,
            model_aware_forced_actual=model_aware_forced_actual,
        )
        self._run.next_step_id += 1
        return {
            "run_id": self._run.run_id,
            "step_id": step_id,
            "coordinate": coordinate,
            "actual": actual,
            "reason": reason,
        }

    def _require_step(self, run_id: int, step_id: int) -> _StepState:
        if self._run is None or self._run.run_id != int(run_id):
            raise RuntimeError("Spectrum H3 run context is stale")
        if self._step is None or self._step.step_id != int(step_id):
            raise RuntimeError("Spectrum H3 solver-step context is stale")
        return self._step

    def _disable_forecasting(self, reason: str) -> bool:
        newly_disabled = not self._disabled
        if newly_disabled:
            self._disabled = True
            self._disable_reason = str(reason)
            self.stats.disabled = True
            self.stats.disable_reason = self._disable_reason
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._rollback_requested = False
        if self._offline_phase == "first_pass" and self._offline_archive is not None:
            self._offline_archive.invalidate(reason)
        return newly_disabled

    def _fallback_or_retry(self, step: _StepState, reason: str) -> None:
        if step.mode == "replay":
            raise OfflineReplayAbort(reason)
        self._disable_forecasting(reason)
        if any(call.used_forecast for call in step.calls):
            raise ForecastRetryActual(reason)
        step.mode = "actual"
        step.reason = reason
        step.bootstrap_forecast = False
        step.fallback = True

    def fallback_current_step(self, run_id: int, step_id: int, reason: str) -> None:
        step = self._require_step(run_id, step_id)
        self._fallback_or_retry(step, reason)

    def begin_model_call(
        self,
        run_id: int,
        step_id: int,
        *,
        topology: tuple[Any, ...],
        labels: tuple[Any, ...] | None,
        expected_shape: tuple[int, ...],
    ) -> tuple[int, bool]:
        step = self._require_step(run_id, step_id)
        normalized_topology = tuple(topology)
        normalized_shape = tuple(int(v) for v in expected_shape)
        normalized_labels = None if labels is None else tuple(labels)
        if len(normalized_shape) < 2:
            self._fallback_or_retry(step, "target feature shape has no branch dimension")
        if step.mode == "replay" and self._offline_archive is not None:
            if normalized_topology != self._offline_archive.topology:
                self._fallback_or_retry(step, "offline replay topology changed")
            if self._offline_archive.feature_shape is not None and tuple(normalized_shape[1:]) != tuple(
                self._offline_archive.feature_shape[1:]
            ):
                self._fallback_or_retry(step, "offline replay target feature shape changed")
        if self._history_topology is not None and normalized_topology != self._history_topology:
            self._fallback_or_retry(step, "packed H3 topology changed within the sampling run")
        call = _CallState(normalized_topology, normalized_labels, normalized_shape)
        step.calls.append(call)
        return len(step.calls) - 1, step.mode == "actual"

    def observe_actual(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        feature: torch.Tensor,
    ) -> None:
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode != "actual":
            raise RuntimeError("actual H3 feature observed during a forecast-only step")
        if tuple(feature.shape) != call.expected_shape:
            raise RuntimeError(
                f"actual H3 feature shape {tuple(feature.shape)} does not match {call.expected_shape}"
            )
        started = time.perf_counter()
        try:
            detached = feature.detach()
            capture_storage = self.config.history_storage
            if (
                self._offline_phase == "first_pass"
                and self.config.offline_archive_storage == "vram"
            ):
                # Preserve a compact device copy when either consumer needs it.
                # Finalization can then transfer the bounded causal history to
                # CPU while the full replay archive takes ownership on device.
                capture_storage = "vram"
            if capture_storage == "vram":
                # The observed target is a view into the complete final-block
                # hidden state. A forced clone keeps only the compact target
                # storage alive and prevents later reuse of the backing tensor.
                archived = detached.clone(memory_format=torch.contiguous_format)
            else:
                archived = detached.to(device="cpu", dtype=feature.dtype, copy=True).contiguous()
        finally:
            self.stats.history_archive_seconds += time.perf_counter() - started
        call.observed_actual = True
        step.actual_records.append(_ActualRecord(archived, call.labels))

    def prepare_residual_probe(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ResidualProbe | None:
        step = self._require_step(run_id, step_id)
        if (
            step.mode != "actual"
            or step.rollback_replay
            or self._rollback_replay_active
            or not self._residual_experiment_enabled()
        ):
            return None
        if step.residual_skip_reason is not None:
            return None
        if self._last_completed_mode != "forecast":
            return None
        if self.config.anchor_residual_feedback:
            run = self._run
            if run is None:
                raise RuntimeError("residual probe lost its active run")
            if self.stats.feedback_refreshes >= _FEEDBACK_MAX_REFRESHES:
                step.residual_skip_reason = "actual-refresh budget exhausted"
                self.stats.feedback_suppressed_budget += 1
                if self.config.debug:
                    LOG.warning(
                        "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s budget=%s",
                        run_id,
                        step_id,
                        step.residual_skip_reason,
                        _FEEDBACK_MAX_REFRESHES,
                    )
                return None
            effective_tail = max(self.config.tail_actual_steps, run.min_tail_actual_steps)
            tail_start = max(0, run.total_steps - effective_tail)
            if step.step_id + 1 >= tail_start:
                step.residual_skip_reason = "no later forecast can consume feedback"
                self.stats.residual_skipped_terminal_probes += 1
                if self.config.debug:
                    LOG.warning(
                        "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s",
                        run_id,
                        step_id,
                        step.residual_skip_reason,
                    )
                return None
        if (
            self.config.selective_rollback_correction
            and self.stats.rollback_count >= _ROLLBACK_MAX_CORRECTIONS
        ):
            step.residual_skip_reason = "rollback correction budget exhausted"
            self.stats.rollback_suppressed_budget += 1
            if self.config.debug:
                LOG.warning(
                    "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s budget=%s",
                    run_id,
                    step_id,
                    step.residual_skip_reason,
                    _ROLLBACK_MAX_CORRECTIONS,
                )
            return None
        if not self.forecaster.ready(self.config.min_fit_points):
            return None

        call = step.calls[int(call_id)]
        if self._history_labels is None or call.labels is None:
            self.disable_experiment("residual measurement branch labels are missing")
            return None
        if len(call.labels) != call.expected_shape[0] or len(set(call.labels)) != len(call.labels):
            self.disable_experiment("residual measurement branch labels are duplicate or incomplete")
            return None
        positions: list[int] = []
        for label in call.labels:
            try:
                positions.append(self._history_labels.index(label))
            except ValueError:
                self.disable_experiment("residual measurement branch identity changed")
                return None
        if len(set(positions)) != len(positions):
            self.disable_experiment("residual measurement assigned a canonical row more than once")
            return None
        history_shape = self.forecaster.feature_shape
        if history_shape is None or tuple(call.expected_shape[1:]) != tuple(history_shape[1:]):
            self.disable_experiment("residual measurement target feature shape changed")
            return None

        started = time.perf_counter()
        try:
            segments = self._prediction_segments(call)
            shadow = self.forecaster.predict_segments(
                step.coordinate,
                segments,
                rows=positions,
                device=device,
                dtype=dtype,
            )
            hold = self.forecaster.predict_latest_hold(
                rows=positions,
                device=device,
                dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as exc:
            self.disable_experiment(f"residual probe prediction failed: {exc}")
            return None
        finally:
            self.stats.residual_measure_seconds += time.perf_counter() - started
        if tuple(shadow.shape) != call.expected_shape or tuple(hold.shape) != call.expected_shape:
            self.disable_experiment("residual probe prediction shape is invalid")
            return None
        if not tensor_all_finite(shadow) or not tensor_all_finite(hold):
            self.disable_experiment("residual probe prediction is nonfinite")
            return None
        step.residual_expected = True
        return ResidualProbe(shadow=shadow, hold=hold)

    def record_residual_measurement(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        probe: ResidualProbe,
        *,
        actual_feature: torch.Tensor,
        actual_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        shadow_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        hold_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        step = self._require_step(run_id, step_id)
        if not self._residual_experiment_enabled() or step.mode != "actual":
            return
        call = step.calls[int(call_id)]
        if call.labels is None:
            self.disable_experiment("residual measurement lost branch labels")
            return
        if not all(
            isinstance(output, (list, tuple)) and len(output) == 2
            for output in (actual_output, shadow_output, hold_output)
        ):
            self.disable_experiment("residual measurement output structure changed")
            return
        started = time.perf_counter()
        try:
            video_score = measure_stream_residual(
                actual_output[0], shadow_output[0], hold_output[0]
            )
            audio_score = measure_stream_residual(
                actual_output[1], shadow_output[1], hold_output[1]
            )
        except (RuntimeError, ValueError) as exc:
            self.disable_experiment(f"residual measurement failed: {exc}")
            return
        finally:
            self.stats.residual_measure_seconds += time.perf_counter() - started
        step.residual_records.append(
            _ResidualRecord(tuple(call.labels), video_score, audio_score)
        )
        self.stats.residual_max_video_score = max(
            self.stats.residual_max_video_score, video_score.score
        )
        self.stats.residual_max_audio_score = max(
            self.stats.residual_max_audio_score, audio_score.score
        )
        self.stats.residual_max_score = max(
            self.stats.residual_max_score, video_score.score, audio_score.score
        )

    def predict(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode == "actual":
            return None
        canonical_labels = (
            self._offline_archive.labels
            if step.mode == "replay" and self._offline_archive is not None
            else self._history_labels
        )
        if canonical_labels is None or call.labels is None:
            self._fallback_or_retry(step, "branch labels are missing; forecast row correspondence is unproven")
            return None
        if len(call.labels) != call.expected_shape[0] or len(set(call.labels)) != len(call.labels):
            self._fallback_or_retry(step, "branch labels are duplicate or do not match the model-call batch")
            return None
        positions = []
        for label in call.labels:
            try:
                position = canonical_labels.index(label)
            except ValueError:
                self._fallback_or_retry(step, "conditional branch identity changed")
                return None
            if position in step.used_history_rows:
                self._fallback_or_retry(step, "conditional branch row was assigned more than once")
                return None
            positions.append(position)

        history_shape = (
            self._offline_archive.feature_shape
            if step.mode == "replay" and self._offline_archive is not None
            else self.forecaster.feature_shape
        )
        if history_shape is None or tuple(call.expected_shape[1:]) != tuple(history_shape[1:]):
            self._fallback_or_retry(step, "target audio/video row count or hidden width changed")
            return None
        if step.bootstrap_forecast and self.forecaster.history_length != 1:
            self._fallback_or_retry(
                step,
                "one-point bootstrap forecast requires exactly one actual history entry",
            )
            return None
        segments = None
        model_aware_weighted_segments = None
        if step.mode != "replay" and not step.bootstrap_forecast:
            try:
                if (
                    step.model_aware_decision is not None
                    and self._model_aware_enabled()
                    and self._offline_phase is None
                ):
                    model_aware_weighted_segments = self._model_aware_weight_segments(
                        call,
                        step.model_aware_decision,
                        coordinate=step.coordinate,
                    )
                else:
                    segments = self._prediction_segments(call)
            except (RuntimeError, ValueError) as exc:
                self._fallback_or_retry(step, str(exc))
                return None
        started = time.perf_counter()
        try:
            if step.mode == "replay":
                if self._offline_smoother is None:
                    raise OfflineReplayAbort("offline replay smoother is missing")
                predicted = self._offline_smoother.predict(
                    step.step_id,
                    rows=tuple(positions),
                    device=device,
                    dtype=dtype,
                )
            elif step.bootstrap_forecast:
                predicted = self.forecaster.predict_one_point_hold(
                    rows=positions,
                    device=device,
                    dtype=dtype,
                )
            elif model_aware_weighted_segments is not None:
                predicted = self.forecaster.predict_with_segment_weights(
                    model_aware_weighted_segments,
                    rows=positions,
                    device=device,
                    dtype=dtype,
                )
            else:
                assert segments is not None
                predicted = self.forecaster.predict_segments(
                    step.coordinate,
                    segments,
                    rows=positions,
                    device=device,
                    dtype=dtype,
                )
        except (RuntimeError, ValueError) as exc:
            if step.mode == "replay":
                raise OfflineReplayAbort(f"offline replay prediction failed: {exc}") from exc
            raise
        finally:
            self.stats.forecast_prediction_seconds += time.perf_counter() - started
        if tuple(predicted.shape) != call.expected_shape:
            self._fallback_or_retry(step, "predicted target feature shape is invalid")
            return None
        step.used_history_rows.update(positions)
        call.used_forecast = True
        return predicted

    def prepare_actual_retry(self, run_id: int, step_id: int, reason: str) -> None:
        step = self._require_step(run_id, step_id)
        step.mode = "actual"
        step.reason = str(reason)
        step.bootstrap_forecast = False
        step.fallback = True
        step.calls.clear()
        step.actual_records.clear()
        step.residual_records.clear()
        step.residual_expected = False
        step.used_history_rows.clear()
        self.stats.forecast_fallbacks += 1

    @staticmethod
    def _label_key(label: Any) -> tuple[str, str]:
        return type(label).__name__, repr(label)

    def _aggregate_actual(self, step: _StepState) -> torch.Tensor | None:
        if not step.actual_records:
            self._disable_forecasting("actual solver step produced no observable target feature")
            return None
        if any(record.labels is None for record in step.actual_records):
            self._disable_forecasting("actual solver step did not provide branch labels")
            return None

        rows: list[tuple[Any, torch.Tensor]] = []
        topology = step.calls[0].topology
        for call in step.calls:
            if call.topology != topology:
                self._disable_forecasting("packed H3 topology changed between model subcalls")
                return None
        for record in step.actual_records:
            assert record.labels is not None
            if len(record.labels) != record.feature.shape[0]:
                self._disable_forecasting("branch labels do not cover actual feature rows")
                return None
            rows.extend((label, record.feature[index]) for index, label in enumerate(record.labels))
        labels = tuple(label for label, _ in rows)
        if len(set(labels)) != len(labels):
            self._disable_forecasting("duplicate conditional branch labels make row correspondence ambiguous")
            return None

        if self._history_labels is None:
            canonical_labels = tuple(sorted(labels, key=self._label_key))
        else:
            canonical_labels = self._history_labels
            if set(labels) != set(canonical_labels) or len(labels) != len(canonical_labels):
                self._disable_forecasting("conditional branch set changed across actual solver steps")
                return None
        if len(step.actual_records) == 1 and step.actual_records[0].labels == canonical_labels:
            combined = step.actual_records[0].feature
            self.stats.direct_history_updates += 1
        else:
            row_map = {label: feature for label, feature in rows}
            combined = torch.stack([row_map[label] for label in canonical_labels], dim=0).contiguous()

        if self._history_topology is None:
            self._history_topology = topology
        elif topology != self._history_topology:
            self._disable_forecasting("packed H3 topology changed across actual history")
            return None
        self._history_labels = canonical_labels
        return combined

    def _aggregate_residual(
        self,
        step: _StepState,
    ) -> _AggregatedResidual | None:
        if not step.residual_expected:
            return None
        if len(step.residual_records) != len(step.actual_records):
            self.disable_experiment("residual measurement did not cover every actual model subcall")
            return None
        if self._history_labels is None:
            self.disable_experiment("residual measurement has no canonical branch labels")
            return None
        video_score = 0.0
        audio_score = 0.0
        for record in step.residual_records:
            video_score = max(video_score, record.video_score.score)
            audio_score = max(audio_score, record.audio_score.score)
        labels = tuple(
            label
            for record in step.residual_records
            for label in record.labels
        )
        if len(set(labels)) != len(labels) or set(labels) != set(self._history_labels):
            self.disable_experiment("residual branch set is duplicate or incomplete")
            return None
        policy_score = (
            video_score
            if self.config.anchor_residual_feedback
            else max(video_score, audio_score)
        )
        if (
            not all(math.isfinite(value) for value in (video_score, audio_score, policy_score))
        ):
            self.disable_experiment("aggregated residual score is nonfinite")
            return None
        return _AggregatedResidual(
            policy_score=policy_score,
            video_score=video_score,
            audio_score=audio_score,
        )

    def _observe_model_aware_anchor(
        self,
        step: _StepState,
        combined: torch.Tensor,
        exact_head_weights: dict[str, torch.Tensor],
        stream_diagonals: dict[str, torch.Tensor],
    ) -> None:
        if not self._model_aware_enabled() or self.forecaster.history_length < 2:
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
            generic_audio, generic_video = self.model_aware.generic_correction_gains(decision)
            parameters = []
            # A zero audio spectral blend selects the linear/local audio predictor;
            # it does not bypass audio forecasting. H3 emits packed audio and video
            # features from the same transformer evaluation, so an actual anchor
            # refreshes both streams and audio error remains scheduler-relevant.
            # Offline replay later uses bracketing local interpolation, making this
            # causal audio evidence a trajectory-risk proxy rather than an exact
            # measurement of replay interpolation error.
            for name, start, end in self._stream_ranges(step.calls[0]):
                if name == "audio":
                    blend = decision.audio_blend_weight
                    model_gain = decision.audio_correction_gain
                    generic_gain = generic_audio
                    diagonal_candidate_gain = (
                        decision.audio_correction_telemetry.diagonal_candidate_gain
                    )
                    model_candidate_gain = (
                        decision.audio_correction_telemetry.model_candidate_gain
                    )
                    subspace = decision.audio_subspace_telemetry
                elif name == "video":
                    blend = decision.video_blend_weight
                    model_gain = decision.video_correction_gain
                    generic_gain = generic_video
                    diagonal_candidate_gain = (
                        decision.video_correction_telemetry.diagonal_candidate_gain
                    )
                    model_candidate_gain = (
                        decision.video_correction_telemetry.model_candidate_gain
                    )
                    subspace = decision.video_subspace_telemetry
                else:
                    if not math.isclose(
                        decision.audio_blend_weight,
                        decision.video_blend_weight,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            "packed model-aware evidence requires audio/video row metadata"
                        )
                    blend = decision.video_blend_weight
                    model_gain = 0.5 * (
                        decision.audio_correction_gain + decision.video_correction_gain
                    )
                    generic_gain = 0.5 * (generic_audio + generic_video)
                    diagonal_candidate_gain = 0.5 * (
                        decision.audio_correction_telemetry.diagonal_candidate_gain
                        + decision.video_correction_telemetry.diagonal_candidate_gain
                    )
                    model_candidate_gain = 0.5 * (
                        decision.audio_correction_telemetry.model_candidate_gain
                        + decision.video_correction_telemetry.model_candidate_gain
                    )
                    audio_subspace = decision.audio_subspace_telemetry
                    video_subspace = decision.video_subspace_telemetry
                    subspace = replace(
                        audio_subspace,
                        eligible=audio_subspace.eligible and video_subspace.eligible,
                        generic_coefficients=tuple(
                            0.5
                            * (
                                audio_subspace.generic_coefficients[index]
                                + video_subspace.generic_coefficients[index]
                            )
                            for index in range(2)
                        ),
                        exact_coefficients=tuple(
                            0.5
                            * (
                                audio_subspace.exact_coefficients[index]
                                + video_subspace.exact_coefficients[index]
                            )
                            for index in range(2)
                        ),
                        applied_coefficients=tuple(
                            0.5
                            * (
                                audio_subspace.applied_coefficients[index]
                                + video_subspace.applied_coefficients[index]
                            )
                            for index in range(2)
                        ),
                    )
                parameters.append(
                    (
                        name,
                        start,
                        end,
                        blend,
                        model_gain,
                        generic_gain,
                        diagonal_candidate_gain,
                        model_candidate_gain,
                        subspace.generic_coefficients,
                        subspace.exact_coefficients,
                        subspace.applied_coefficients,
                    )
                )
            evidence = self.forecaster.sampled_anchor_evidence(
                step.coordinate,
                combined,
                parameters,
                degree=decision.degree,
                ridge_lambda=decision.ridge_lambda,
                stream_diagonals=stream_diagonals,
                exact_head_weights=exact_head_weights,
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
                self.stats.model_aware_evidence_sensitivity_transfer_seconds += (
                    evidence.timing.sensitivity_transfer_seconds
                )
                self.stats.model_aware_evidence_scalar_transfer_seconds += (
                    evidence.timing.scalar_transfer_seconds
                )
                self.stats.model_aware_evidence_reduction_seconds += (
                    evidence.timing.reduction_seconds
                )
                self.stats.model_aware_evidence_exact_head_projection_seconds += (
                    evidence.timing.exact_head_projection_seconds
                )
                self.stats.model_aware_exact_head_projection_seconds += (
                    evidence.timing.exact_head_projection_seconds
                )
                self.stats.model_aware_exact_head_projection_calls += len(
                    exact_head_weights
                )
                self.stats.model_aware_evidence_fit_condition_seconds += (
                    evidence.timing.fit_condition_seconds
                )
                self.stats.model_aware_subspace_gram_seconds += (
                    evidence.timing.subspace_gram_seconds
                )
                self.stats.model_aware_subspace_solve_seconds += (
                    evidence.timing.subspace_solve_seconds
                )
                self.stats.model_aware_subspace_workspace_bytes = max(
                    self.stats.model_aware_subspace_workspace_bytes,
                    evidence.subspace_workspace_bytes,
                )
                self.stats.model_aware_model_corrected_ratio_mean = (
                    self.model_aware.model_corrected_ratio_mean
                )
                self.stats.model_aware_generic_corrected_ratio_mean = (
                    self.model_aware.generic_corrected_ratio_mean
                )
                if self.config.debug:
                    audio_gain = decision.audio_correction_telemetry
                    video_gain = decision.video_correction_telemetry
                    audio_subspace = decision.audio_subspace_telemetry
                    video_subspace = decision.video_subspace_telemetry
                    LOG.warning(
                        "Spectrum H3 model-aware anchor step=%s "
                        "forecast_ratio_audio=%.6f forecast_ratio_video=%.6f "
                        "forecast_ratio_aggregate=max(audio,video)=%.6f "
                        "curvature_ratio_audio=%.6f curvature_ratio_video=%.6f "
                        "curvature_ratio_aggregate=max(audio,video)=%.6f "
                        "fit_condition=%.6f "
                        "audio_generic_projection=%.6f audio_diagonal_projection=%.6f "
                        "audio_exact_head_projection=%.6f "
                        "video_generic_projection=%.6f video_diagonal_projection=%.6f "
                        "video_exact_head_projection=%.6f "
                        "audio_applied_ratio=%.6f audio_generic_ratio=%.6f "
                        "audio_diagonal_candidate_ratio=%.6f audio_exact_candidate_ratio=%.6f "
                        "audio_applied_head_ratio=%.6f audio_generic_head_ratio=%.6f "
                        "audio_diagonal_candidate_head_ratio=%.6f "
                        "audio_exact_candidate_head_ratio=%.6f "
                        "video_applied_ratio=%.6f video_generic_ratio=%.6f "
                        "video_diagonal_candidate_ratio=%.6f video_exact_candidate_ratio=%.6f "
                        "video_applied_head_ratio=%.6f video_generic_head_ratio=%.6f "
                        "video_diagonal_candidate_head_ratio=%.6f "
                        "video_exact_candidate_head_ratio=%.6f",
                        step.step_id,
                        evidence.audio.forecast_ratio,
                        evidence.video.forecast_ratio,
                        evidence.forecast_ratio,
                        evidence.audio.curvature_ratio,
                        evidence.video.curvature_ratio,
                        evidence.curvature_ratio,
                        evidence.fit_condition,
                        evidence.audio.residual_projection,
                        evidence.audio.diagonal_projection,
                        evidence.audio.model_projection,
                        evidence.video.residual_projection,
                        evidence.video.diagonal_projection,
                        evidence.video.model_projection,
                        evidence.audio.model_corrected_ratio,
                        evidence.audio.generic_corrected_ratio,
                        evidence.audio.diagonal_candidate_ratio,
                        evidence.audio.model_candidate_ratio,
                        evidence.audio.model_corrected_head_ratio,
                        evidence.audio.generic_corrected_head_ratio,
                        evidence.audio.diagonal_candidate_head_ratio,
                        evidence.audio.model_candidate_head_ratio,
                        evidence.video.model_corrected_ratio,
                        evidence.video.generic_corrected_ratio,
                        evidence.video.diagonal_candidate_ratio,
                        evidence.video.model_candidate_ratio,
                        evidence.video.model_corrected_head_ratio,
                        evidence.video.generic_corrected_head_ratio,
                        evidence.video.diagonal_candidate_head_ratio,
                        evidence.video.model_candidate_head_ratio,
                    )
                    LOG.warning(
                        "Spectrum H3 model-aware correction step=%s "
                        "audio_raw_generic_gain=%.6f audio_raw_diagonal_gain=%.6f "
                        "audio_raw_exact_gain=%.6f audio_generic_gain=%.6f "
                        "audio_diagonal_candidate_gain=%.6f audio_exact_candidate_gain=%.6f "
                        "audio_applied_gain=%.6f audio_exact_trust=%.6f "
                        "audio_exact_trust_next=%.6f audio_generic_bound_active=%s "
                        "audio_diagonal_bound_active=%s audio_exact_bound_active=%s "
                        "audio_exact_generic_delta_pre=%.6f audio_exact_generic_delta_post=%.6f "
                        "audio_exact_diagonal_delta_pre=%.6f "
                        "audio_exact_diagonal_delta_post=%.6f audio_applied_delta=%.6f "
                        "video_raw_generic_gain=%.6f video_raw_diagonal_gain=%.6f "
                        "video_raw_exact_gain=%.6f video_generic_gain=%.6f "
                        "video_diagonal_candidate_gain=%.6f video_exact_candidate_gain=%.6f "
                        "video_applied_gain=%.6f video_exact_trust=%.6f "
                        "video_exact_trust_next=%.6f video_generic_bound_active=%s "
                        "video_diagonal_bound_active=%s video_exact_bound_active=%s "
                        "video_exact_generic_delta_pre=%.6f video_exact_generic_delta_post=%.6f "
                        "video_exact_diagonal_delta_pre=%.6f "
                        "video_exact_diagonal_delta_post=%.6f video_applied_delta=%.6f",
                        step.step_id,
                        audio_gain.raw_generic_gain,
                        audio_gain.raw_diagonal_gain,
                        audio_gain.raw_model_gain,
                        audio_gain.generic_gain,
                        audio_gain.diagonal_candidate_gain,
                        audio_gain.model_candidate_gain,
                        audio_gain.model_gain,
                        audio_gain.model_trust,
                        self.model_aware.audio_model_trust,
                        audio_gain.generic_bound_active,
                        audio_gain.diagonal_bound_active,
                        audio_gain.model_bound_active,
                        audio_gain.pre_bound_delta,
                        audio_gain.post_bound_delta,
                        audio_gain.exact_diagonal_pre_bound_delta,
                        audio_gain.exact_diagonal_post_bound_delta,
                        audio_gain.applied_delta,
                        video_gain.raw_generic_gain,
                        video_gain.raw_diagonal_gain,
                        video_gain.raw_model_gain,
                        video_gain.generic_gain,
                        video_gain.diagonal_candidate_gain,
                        video_gain.model_candidate_gain,
                        video_gain.model_gain,
                        video_gain.model_trust,
                        self.model_aware.video_model_trust,
                        video_gain.generic_bound_active,
                        video_gain.diagonal_bound_active,
                        video_gain.model_bound_active,
                        video_gain.pre_bound_delta,
                        video_gain.post_bound_delta,
                        video_gain.exact_diagonal_pre_bound_delta,
                        video_gain.exact_diagonal_post_bound_delta,
                        video_gain.applied_delta,
                    )
                    LOG.warning(
                        "Spectrum H3 model-aware K=2 step=%s "
                        "audio_eligible=%s audio_fallback=%s "
                        "audio_solve_generic=(%.6f,%.6f) audio_solve_exact=(%.6f,%.6f) "
                        "audio_condition_generic=%.6f audio_condition_exact=%.6f "
                        "audio_rank_generic=%s audio_rank_exact=%s "
                        "audio_raw_generic=(%.6f,%.6f) audio_raw_exact=(%.6f,%.6f) "
                        "audio_generic=(%.6f,%.6f) audio_exact=(%.6f,%.6f) "
                        "audio_applied=(%.6f,%.6f) "
                        "audio_raw_norm_generic=%.6f audio_raw_norm_exact=%.6f "
                        "audio_bound_scale_generic=%.6f audio_bound_scale_exact=%.6f "
                        "audio_bound_norm_generic=%.6f audio_bound_norm_exact=%.6f "
                        "audio_radial_bound_generic=%s audio_radial_bound_exact=%s "
                        "audio_generic_2d_ratio=%.6f audio_exact_2d_ratio=%.6f "
                        "audio_applied_2d_ratio=%.6f audio_generic_2d_head_ratio=%.6f "
                        "audio_exact_2d_head_ratio=%.6f audio_applied_2d_head_ratio=%.6f "
                        "audio_exact_2d_trust=%.6f audio_exact_2d_trust_next=%.6f "
                        "video_eligible=%s video_fallback=%s "
                        "video_solve_generic=(%.6f,%.6f) video_solve_exact=(%.6f,%.6f) "
                        "video_condition_generic=%.6f video_condition_exact=%.6f "
                        "video_rank_generic=%s video_rank_exact=%s "
                        "video_raw_generic=(%.6f,%.6f) video_raw_exact=(%.6f,%.6f) "
                        "video_generic=(%.6f,%.6f) video_exact=(%.6f,%.6f) "
                        "video_applied=(%.6f,%.6f) "
                        "video_raw_norm_generic=%.6f video_raw_norm_exact=%.6f "
                        "video_bound_scale_generic=%.6f video_bound_scale_exact=%.6f "
                        "video_bound_norm_generic=%.6f video_bound_norm_exact=%.6f "
                        "video_radial_bound_generic=%s video_radial_bound_exact=%s "
                        "video_generic_2d_ratio=%.6f video_exact_2d_ratio=%.6f "
                        "video_applied_2d_ratio=%.6f video_generic_2d_head_ratio=%.6f "
                        "video_exact_2d_head_ratio=%.6f video_applied_2d_head_ratio=%.6f "
                        "video_exact_2d_trust=%.6f video_exact_2d_trust_next=%.6f",
                        step.step_id,
                        audio_subspace.eligible,
                        not evidence.audio.generic_2d_eligible or not evidence.audio.exact_2d_eligible,
                        *evidence.audio.generic_2d_coefficients,
                        *evidence.audio.exact_2d_coefficients,
                        evidence.audio.generic_2d_condition,
                        evidence.audio.exact_2d_condition,
                        evidence.audio.generic_2d_rank,
                        evidence.audio.exact_2d_rank,
                        *audio_subspace.raw_generic_coefficients,
                        *audio_subspace.raw_exact_coefficients,
                        *audio_subspace.generic_coefficients,
                        *audio_subspace.exact_coefficients,
                        *audio_subspace.applied_coefficients,
                        audio_subspace.generic_raw_norm_ratio,
                        audio_subspace.exact_raw_norm_ratio,
                        audio_subspace.generic_bound_scale,
                        audio_subspace.exact_bound_scale,
                        audio_subspace.generic_bounded_norm_ratio,
                        audio_subspace.exact_bounded_norm_ratio,
                        audio_subspace.generic_bound_active,
                        audio_subspace.exact_bound_active,
                        evidence.audio.generic_2d_ratio,
                        evidence.audio.exact_2d_ratio,
                        evidence.audio.applied_2d_ratio,
                        evidence.audio.generic_2d_head_ratio,
                        evidence.audio.exact_2d_head_ratio,
                        evidence.audio.applied_2d_head_ratio,
                        audio_subspace.model_trust,
                        self.model_aware.audio_subspace_model_trust,
                        video_subspace.eligible,
                        not evidence.video.generic_2d_eligible or not evidence.video.exact_2d_eligible,
                        *evidence.video.generic_2d_coefficients,
                        *evidence.video.exact_2d_coefficients,
                        evidence.video.generic_2d_condition,
                        evidence.video.exact_2d_condition,
                        evidence.video.generic_2d_rank,
                        evidence.video.exact_2d_rank,
                        *video_subspace.raw_generic_coefficients,
                        *video_subspace.raw_exact_coefficients,
                        *video_subspace.generic_coefficients,
                        *video_subspace.exact_coefficients,
                        *video_subspace.applied_coefficients,
                        video_subspace.generic_raw_norm_ratio,
                        video_subspace.exact_raw_norm_ratio,
                        video_subspace.generic_bound_scale,
                        video_subspace.exact_bound_scale,
                        video_subspace.generic_bounded_norm_ratio,
                        video_subspace.exact_bounded_norm_ratio,
                        video_subspace.generic_bound_active,
                        video_subspace.exact_bound_active,
                        evidence.video.generic_2d_ratio,
                        evidence.video.exact_2d_ratio,
                        evidence.video.applied_2d_ratio,
                        evidence.video.generic_2d_head_ratio,
                        evidence.video.exact_2d_head_ratio,
                        evidence.video.applied_2d_head_ratio,
                        video_subspace.model_trust,
                        self.model_aware.video_subspace_model_trust,
                    )
                    LOG.warning(
                        "Spectrum H3 model-aware evidence timing step=%s "
                        "evidence_weight_fit_s=%.6f evidence_sample_index_s=%.6f "
                        "evidence_device_transfer_s=%.6f evidence_scalar_transfer_s=%.6f "
                        "evidence_reduction_s=%.6f evidence_exact_head_projection_s=%.6f "
                        "evidence_fit_condition_s=%.6f subspace_gram_s=%.6f "
                        "subspace_solve_s=%.6f",
                        step.step_id,
                        evidence.timing.weight_fit_seconds,
                        evidence.timing.sample_index_seconds,
                        evidence.timing.device_transfer_seconds,
                        evidence.timing.scalar_transfer_seconds,
                        evidence.timing.reduction_seconds,
                        evidence.timing.exact_head_projection_seconds,
                        evidence.timing.fit_condition_seconds,
                        evidence.timing.subspace_gram_seconds,
                        evidence.timing.subspace_solve_seconds,
                    )
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            self._model_aware_disabled_reason = f"anchor evidence failed: {exc}"
            self.stats.model_aware_failures += 1
            LOG.warning(
                "Spectrum H3 model-aware forecasting disabled for this run: %s",
                self._model_aware_disabled_reason,
            )
        finally:
            elapsed = time.perf_counter() - started
            self.stats.model_aware_evidence_seconds += elapsed
            self.stats.model_aware_overhead_seconds += elapsed

    def _apply_residual_policy(self, step: _StepState, result: _AggregatedResidual) -> None:
        if self._experiment_disabled:
            return
        score = result.policy_score
        action = "none"
        if self.config.anchor_residual_feedback:
            if score < _FEEDBACK_SCORE_THRESHOLD:
                self.stats.feedback_suppressed_threshold += 1
                action = "below_refresh_threshold"
            elif self.stats.feedback_refreshes >= _FEEDBACK_MAX_REFRESHES:
                self.stats.feedback_suppressed_budget += 1
                action = "refresh_budget_exhausted"
            else:
                self._required_feedback_actuals = max(self._required_feedback_actuals, 1)
                action = "actual_refresh_requested"
        elif self.config.selective_rollback_correction and self._last_completed_mode == "forecast":
            if self._rollback_replay_active:
                action = "rollback_replay_ignored"
            elif score < _ROLLBACK_SCORE_THRESHOLD:
                self.stats.rollback_suppressed_threshold += 1
                action = "below_rollback_threshold"
            elif self.stats.rollback_count >= _ROLLBACK_MAX_CORRECTIONS:
                self.stats.rollback_suppressed_budget += 1
                action = "rollback_budget_exhausted"
            else:
                self._rollback_requested = True
                action = "rollback_requested"
        self.stats.residual_policy_max_score = max(
            self.stats.residual_policy_max_score,
            score,
        )
        if self.config.debug:
            LOG.warning(
                "Spectrum H3 residual anchor run_id=%s step=%s video=%.6f audio=%.6f "
                "policy=%.6f action=%s",
                self.stats.run_id,
                step.step_id,
                result.video_score,
                result.audio_score,
                score,
                action,
            )

    def finalize_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if not step.calls:
            if step.mode == "replay":
                raise OfflineReplayAbort(
                    "offline replay step did not reach the native H3 model wrapper"
                )
            if step.fallback and self._disabled:
                self._consecutive_forecasts = 0
                self.stats.actual_steps += 1
                self.stats.current_window = self._current_window
                self._step = None
                return
            reason = "solver step completed without reaching the native H3 model wrapper"
            newly_disabled = self._disable_forecasting(reason)
            self._consecutive_forecasts = 0
            self._required_actual_refreshes = 0
            self.stats.bypassed_steps += 1
            self.stats.current_window = self._current_window
            self._step = None
            if newly_disabled:
                LOG.warning(
                    "Spectrum H3 disabled for the rest of this run because a predict-noise "
                    "evaluation returned without reaching the native MiniMax H3 model wrapper; "
                    "accepting the wrapped result as a passthrough. Another model or cache patch "
                    "may have intercepted the evaluation."
                )
            return

        if step.mode == "replay":
            if any(call.observed_actual for call in step.calls) or not all(
                call.used_forecast for call in step.calls
            ):
                raise OfflineReplayAbort("offline replay model-call transaction was incomplete")
            archive_labels = self._offline_archive.labels if self._offline_archive is not None else ()
            expected_rows = set(range(len(archive_labels or ())))
            if step.used_history_rows != expected_rows:
                raise OfflineReplayAbort("offline replay branch-row allocation was incomplete")
            self.stats.offline_replay_steps += 1
            self.stats.offline_replay_model_calls += len(step.calls)
            archive = self._offline_archive
            if archive is None:
                raise OfflineReplayAbort("offline replay archive disappeared during finalization")
            if archive.steps[step.step_id].actual:
                self.stats.offline_replay_anchor_steps += 1
            else:
                self.stats.offline_replay_smoothed_steps += 1
            self._last_completed_mode = "replay"
            self._last_completed_step_id = step.step_id
            self._step = None
            return

        if step.mode == "forecast":
            if any(call.observed_actual for call in step.calls) or not all(call.used_forecast for call in step.calls):
                raise ForecastRetryActual("forecast solver step was incomplete or mixed with an actual call")
            expected_rows = set(range(len(self._history_labels or ())))
            if step.used_history_rows != expected_rows:
                raise ForecastRetryActual("forecast branch-row allocation was incomplete")
            self._consecutive_forecasts += 1
            self._required_actual_refreshes = self._run.min_actual_steps_after_forecast
            self.stats.forecast_steps += 1
            self.stats.forecast_model_calls += len(step.calls)
            if step.model_aware_decision is not None:
                self.stats.model_aware_forecasts += 1
        else:
            if any(call.used_forecast for call in step.calls):
                raise RuntimeError("actual solver step retained a forecasted subcall")
            combined = self._aggregate_actual(step)
            residual_result = self._aggregate_residual(step)
            if combined is not None and not self._disabled:
                exact_head_weights: dict[str, torch.Tensor] = {}
                stream_diagonals: dict[str, torch.Tensor] = {}
                if self._model_aware_enabled():
                    try:
                        exact_head_weights, stream_diagonals = (
                            self._model_aware_head_metrics(combined.device)
                        )
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except (RuntimeError, TypeError, ValueError) as exc:
                        self._model_aware_disabled_reason = (
                            f"head metric materialization failed: {exc}"
                        )
                        self.stats.model_aware_failures += 1
                        LOG.warning(
                            "Spectrum H3 model-aware correction disabled: %s",
                            self._model_aware_disabled_reason,
                        )
                self._observe_model_aware_anchor(
                    step,
                    combined,
                    exact_head_weights,
                    stream_diagonals,
                )
                if self._offline_phase == "first_pass" and self._offline_archive is not None:
                    assert self._history_labels is not None
                    archive_started = time.perf_counter()
                    try:
                        self._offline_archive.record_actual(
                            step.step_id,
                            step.coordinate,
                            combined,
                            labels=self._history_labels,
                            topology=step.calls[0].topology,
                            take_ownership=True,
                        )
                    finally:
                        elapsed = time.perf_counter() - archive_started
                        self.stats.offline_archive_seconds += elapsed
                        self._offline_archive_seconds_total += elapsed
                update_started = time.perf_counter()
                exact_projection_before = (
                    self.forecaster.model_aware_exact_head_projection_seconds
                )
                exact_projection_calls_before = (
                    self.forecaster.model_aware_exact_head_projection_calls
                )
                try:
                    self.forecaster.update(
                        step.coordinate,
                        combined,
                        anchor_id=step.step_id,
                        take_ownership=True,
                        evidence_segments=(
                            self._stream_ranges(step.calls[0])
                            if self._model_aware_enabled()
                            else None
                        ),
                        exact_head_weights=(
                            exact_head_weights
                            if self._model_aware_enabled()
                            else None
                        ),
                    )
                except ValueError as exc:
                    self._disable_forecasting(f"actual H3 feature is incompatible with history: {exc}")
                finally:
                    self.stats.history_update_seconds += time.perf_counter() - update_started
                    exact_projection_elapsed = max(
                        0.0,
                        self.forecaster.model_aware_exact_head_projection_seconds
                        - exact_projection_before,
                    )
                    exact_projection_calls = max(
                        0,
                        self.forecaster.model_aware_exact_head_projection_calls
                        - exact_projection_calls_before,
                    )
                    self.stats.model_aware_exact_head_projection_seconds += (
                        exact_projection_elapsed
                    )
                    self.stats.model_aware_exact_head_projection_calls += (
                        exact_projection_calls
                    )
                    self.stats.model_aware_exact_head_workspace_bytes = max(
                        self.stats.model_aware_exact_head_workspace_bytes,
                        self.forecaster.model_aware_exact_head_workspace_bytes,
                    )
                    self.stats.model_aware_overhead_seconds += exact_projection_elapsed
            self._consecutive_forecasts = 0
            self._required_actual_refreshes = max(0, self._required_actual_refreshes - 1)
            if step.consumes_feedback_refresh:
                self._required_feedback_actuals = max(0, self._required_feedback_actuals - 1)
            self.stats.actual_steps += 1
            if step.model_aware_forced_actual:
                self.stats.adaptive_extra_nfes += 1
            self.stats.actual_transformer_calls += len(step.actual_records)
            if step.consumes_feedback_refresh:
                self.stats.feedback_refreshes += 1
            if step.rollback_replay:
                self.stats.replayed_transformer_calls += len(step.actual_records)
            if (
                step.adaptive_recompute
                and not step.fallback
                and not self._disabled
                and step.step_id >= self.config.warmup_steps
            ):
                window_ceiling = max(float(self.config.window_size), float(self.config.max_history))
                self._current_window = min(
                    round(self._current_window + self.config.flex_window, 6),
                    window_ceiling,
                )
            if residual_result is not None and not self._experiment_disabled:
                self.stats.residual_anchors += 1
                self._apply_residual_policy(step, residual_result)

        if self._offline_phase == "first_pass" and self._offline_archive is not None:
            self._offline_archive.record_step(
                step.step_id,
                step.coordinate,
                step.mode == "actual",
                model_aware_decision=step.model_aware_decision,
            )

        self.stats.current_window = self._current_window
        self._last_completed_mode = step.mode
        self._last_completed_step_id = step.step_id
        self._step = None

    def abort_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if self._run is not None and self._run.next_step_id == step.step_id + 1:
            self._run.next_step_id = step.step_id
        self._rollback_requested = False
        self._step = None

    def create_rollback_snapshot(self) -> RuntimeRollbackSnapshot:
        if self._run is None or self._step is not None:
            raise RuntimeError("rollback snapshots require an idle active run")
        return RuntimeRollbackSnapshot(
            next_step_id=self._run.next_step_id,
            forecaster=self.forecaster.snapshot(),
            history_topology=self._history_topology,
            history_labels=self._history_labels,
            current_window=self._current_window,
            consecutive_forecasts=self._consecutive_forecasts,
            required_actual_refreshes=self._required_actual_refreshes,
            required_feedback_actuals=self._required_feedback_actuals,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
            experiment_disabled=self._experiment_disabled,
            experiment_disable_reason=self._experiment_disable_reason,
            last_completed_mode=self._last_completed_mode,
            last_completed_step_id=self._last_completed_step_id,
            model_aware_state=self.model_aware.snapshot(),
            stats=replace(self.stats),
        )

    def restore_rollback_snapshot(self, snapshot: RuntimeRollbackSnapshot) -> None:
        if self._run is None or self._step is not None:
            raise RuntimeError("rollback restoration requires an idle active run")
        if not isinstance(snapshot, RuntimeRollbackSnapshot):
            raise TypeError("snapshot must be a RuntimeRollbackSnapshot")
        current = self.stats
        speculative_calls = max(
            0, current.forecast_model_calls - snapshot.stats.forecast_model_calls
        )
        discarded_actual_calls = max(
            0, current.actual_transformer_calls - snapshot.stats.actual_transformer_calls
        )
        restored = replace(snapshot.stats)
        restored.forecast_model_calls += speculative_calls
        restored.actual_transformer_calls += discarded_actual_calls
        restored.speculative_forecast_calls += speculative_calls
        restored.discarded_actual_calls += discarded_actual_calls
        restored.rollback_count += 1
        for name in (
            "history_archive_seconds",
            "history_update_seconds",
            "forecast_prediction_seconds",
            "residual_measure_seconds",
            "residual_output_head_seconds",
            "offline_archive_seconds",
            "offline_smoother_build_seconds",
            "model_aware_overhead_seconds",
            "model_aware_decision_seconds",
            "model_aware_evidence_seconds",
            "model_aware_evidence_weight_fit_seconds",
            "model_aware_evidence_sample_index_seconds",
            "model_aware_evidence_device_transfer_seconds",
            "model_aware_evidence_sensitivity_transfer_seconds",
            "model_aware_evidence_scalar_transfer_seconds",
            "model_aware_evidence_reduction_seconds",
            "model_aware_evidence_exact_head_projection_seconds",
            "model_aware_evidence_fit_condition_seconds",
            "model_aware_subspace_gram_seconds",
            "model_aware_subspace_solve_seconds",
            "model_aware_head_materialization_seconds",
            "model_aware_exact_head_projection_seconds",
            "model_aware_fit_seconds",
            "model_aware_causal_correction_seconds",
            "model_aware_offline_correction_seconds",
        ):
            setattr(
                restored,
                name,
                getattr(restored, name) + max(0.0, getattr(current, name) - getattr(snapshot.stats, name)),
            )
        for name in (
            "direct_history_updates",
            "residual_anchors",
            "residual_failures",
            "residual_skipped_terminal_probes",
            "rollback_suppressed_threshold",
            "rollback_suppressed_budget",
            "offline_replay_steps",
            "offline_replay_model_calls",
            "offline_replay_anchor_steps",
            "offline_replay_smoothed_steps",
            "model_aware_anchor_updates",
            "model_aware_failures",
            "model_aware_offline_correction_applications",
            "model_aware_exact_head_projection_calls",
        ):
            setattr(
                restored,
                name,
                getattr(restored, name) + max(0, getattr(current, name) - getattr(snapshot.stats, name)),
            )
        restored.residual_max_score = max(restored.residual_max_score, current.residual_max_score)
        restored.residual_max_video_score = max(
            restored.residual_max_video_score, current.residual_max_video_score
        )
        restored.residual_max_audio_score = max(
            restored.residual_max_audio_score, current.residual_max_audio_score
        )
        restored.residual_policy_max_score = max(
            restored.residual_policy_max_score, current.residual_policy_max_score
        )
        restored.model_aware_risk_max = max(
            restored.model_aware_risk_max, current.model_aware_risk_max
        )
        restored.model_aware_confidence_min = min(
            restored.model_aware_confidence_min, current.model_aware_confidence_min
        )
        restored.model_aware_correction_max = max(
            restored.model_aware_correction_max, current.model_aware_correction_max
        )
        restored.model_aware_head_materialized_bytes = max(
            restored.model_aware_head_materialized_bytes,
            current.model_aware_head_materialized_bytes,
        )
        restored.model_aware_exact_head_workspace_bytes = max(
            restored.model_aware_exact_head_workspace_bytes,
            current.model_aware_exact_head_workspace_bytes,
        )
        restored.model_aware_subspace_workspace_bytes = max(
            restored.model_aware_subspace_workspace_bytes,
            current.model_aware_subspace_workspace_bytes,
        )

        self._run.next_step_id = snapshot.next_step_id
        self.forecaster.restore(snapshot.forecaster)
        self._history_topology = snapshot.history_topology
        self._history_labels = snapshot.history_labels
        self._current_window = snapshot.current_window
        self._consecutive_forecasts = snapshot.consecutive_forecasts
        self._required_actual_refreshes = snapshot.required_actual_refreshes
        self._required_feedback_actuals = snapshot.required_feedback_actuals
        self._disabled = snapshot.disabled
        self._disable_reason = snapshot.disable_reason
        self._experiment_disabled = snapshot.experiment_disabled
        self._experiment_disable_reason = snapshot.experiment_disable_reason
        self._last_completed_mode = snapshot.last_completed_mode
        self._last_completed_step_id = snapshot.last_completed_step_id
        self.model_aware.restore(snapshot.model_aware_state)
        restored.model_aware_correction_seconds = (
            restored.model_aware_causal_correction_seconds
            + restored.model_aware_offline_correction_seconds
        )
        restored.model_aware_model_corrected_ratio_mean = (
            self.model_aware.model_corrected_ratio_mean
        )
        restored.model_aware_generic_corrected_ratio_mean = (
            self.model_aware.generic_corrected_ratio_mean
        )
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self.stats = restored

    def consume_rollback_request(self) -> bool:
        requested = self._rollback_requested
        self._rollback_requested = False
        return requested

    def force_next_actual(self, reason: str, *, rollback_replay: bool) -> None:
        if self._step is not None:
            raise RuntimeError("cannot force an actual step while another step is active")
        self._forced_actual_reason = str(reason)
        self._forced_actual_is_replay = bool(rollback_replay)

    def begin_rollback_replay(self) -> None:
        self._rollback_replay_active = True

    def end_rollback_replay(self) -> None:
        self._rollback_replay_active = False

    def debug_summary(self) -> str:
        offline_archive_device = (
            self._offline_archive.history_device
            if self._offline_archive is not None
            else None
        )
        controller = self.model_aware
        audio_forecast_mean = controller.stream_mean("audio", "forecast_ratio")
        video_forecast_mean = controller.stream_mean("video", "forecast_ratio")
        audio_model_mean = controller.stream_mean("audio", "model_corrected_ratio")
        video_model_mean = controller.stream_mean("video", "model_corrected_ratio")
        audio_generic_mean = controller.stream_mean("audio", "generic_corrected_ratio")
        video_generic_mean = controller.stream_mean("video", "generic_corrected_ratio")
        audio_diagonal_mean = controller.stream_mean(
            "audio", "diagonal_candidate_ratio"
        )
        video_diagonal_mean = controller.stream_mean(
            "video", "diagonal_candidate_ratio"
        )
        audio_candidate_mean = controller.stream_mean("audio", "model_candidate_ratio")
        video_candidate_mean = controller.stream_mean("video", "model_candidate_ratio")
        audio_candidate_head_mean = controller.stream_mean(
            "audio", "model_candidate_head_ratio"
        )
        video_candidate_head_mean = controller.stream_mean(
            "video", "model_candidate_head_ratio"
        )
        audio_generic_head_mean = controller.stream_mean(
            "audio", "generic_corrected_head_ratio"
        )
        video_generic_head_mean = controller.stream_mean(
            "video", "generic_corrected_head_ratio"
        )
        audio_diagonal_head_mean = controller.stream_mean(
            "audio", "diagonal_candidate_head_ratio"
        )
        video_diagonal_head_mean = controller.stream_mean(
            "video", "diagonal_candidate_head_ratio"
        )
        audio_delta_pre_mean = controller.stream_mean("audio", "gain_delta_pre_abs")
        video_delta_pre_mean = controller.stream_mean("video", "gain_delta_pre_abs")
        audio_delta_post_mean = controller.stream_mean("audio", "gain_delta_post_abs")
        video_delta_post_mean = controller.stream_mean("video", "gain_delta_post_abs")
        audio_delta_applied_mean = controller.stream_mean(
            "audio", "gain_delta_applied_abs"
        )
        video_delta_applied_mean = controller.stream_mean(
            "video", "gain_delta_applied_abs"
        )
        audio_exact_diagonal_delta_mean = controller.stream_mean(
            "audio", "gain_delta_exact_diagonal_abs"
        )
        video_exact_diagonal_delta_mean = controller.stream_mean(
            "video", "gain_delta_exact_diagonal_abs"
        )
        head_metric_available = bool(
            controller.profile is not None
            and controller.profile.audio_head_weight is not None
            and controller.profile.video_head_weight is not None
            and controller.profile.audio_head_gram_diagonal is not None
            and controller.profile.video_head_gram_diagonal is not None
        )
        subspace_parts = []
        for stream in ("audio", "video"):
            for metric in (
                "generic_2d_ratio",
                "exact_2d_ratio",
                "applied_2d_ratio",
                "generic_2d_head_ratio",
                "exact_2d_head_ratio",
                "applied_2d_head_ratio",
            ):
                subspace_parts.append(
                    f"model_aware_{stream}_{metric}_mean="
                    f"{controller.stream_mean(stream, metric):.6f}"
                )
            subspace_parts.extend(
                (
                    f"model_aware_{stream}_2d_eligible={getattr(controller, f'{stream}_subspace_evidence_count')}",
                    f"model_aware_{stream}_2d_fallbacks={getattr(controller, f'{stream}_subspace_fallback_count')}",
                    f"model_aware_{stream}_2d_condition_fallbacks={getattr(controller, f'{stream}_subspace_condition_fallback_count')}",
                    f"model_aware_{stream}_2d_generic_condition_max={getattr(controller, f'{stream}_generic_2d_condition_max'):.6f}",
                    f"model_aware_{stream}_2d_exact_condition_max={getattr(controller, f'{stream}_exact_2d_condition_max'):.6f}",
                    f"model_aware_{stream}_2d_regularization_max={getattr(controller, f'{stream}_subspace_regularization_max'):.6f}",
                    f"model_aware_{stream}_exact_2d_trust={getattr(controller, f'{stream}_subspace_model_trust'):.6f}",
                )
            )
            for comparison in (
                "generic_2d_vs_scalar",
                "exact_2d_vs_exact_scalar",
                "exact_2d_vs_generic_2d",
                "exact_2d_vs_generic_scalar",
            ):
                prefix = f"{stream}_{comparison}"
                subspace_parts.extend(
                    (
                        f"model_aware_{prefix}_comparisons={getattr(controller, f'{prefix}_comparison_count')}",
                        f"model_aware_{prefix}_wins={getattr(controller, f'{prefix}_win_count')}",
                        f"model_aware_{prefix}_losses={getattr(controller, f'{prefix}_loss_count')}",
                        f"model_aware_{prefix}_advantage_mean={controller.subspace_advantage_mean(stream, comparison):.6f}",
                        f"model_aware_{prefix}_advantage_abs_max={getattr(controller, f'{prefix}_advantage_max'):.6f}",
                    )
                )
        subspace_summary = " ".join(subspace_parts)
        return (
            f"run_id={self.stats.run_id} sampler={self.stats.sampler_name} "
            f"steps={self.stats.total_steps} actual_steps={self.stats.actual_steps} "
            f"forecast_steps={self.stats.forecast_steps} "
            f"actual_transformer_calls={self.stats.actual_transformer_calls} "
            f"forecast_calls={self.stats.forecast_model_calls} "
            f"fallbacks={self.stats.forecast_fallbacks} "
            f"bypassed_steps={self.stats.bypassed_steps} disabled={self.stats.disabled} "
            f"video_blend_weight={self.config.blend_weight:.6f} "
            f"audio_blend_weight={self.config.audio_blend_weight:.6f} "
            f"causal_video_blend_weight={self.stats.causal_video_blend_weight:.6f} "
            f"causal_audio_blend_weight={self.stats.causal_audio_blend_weight:.6f} "
            f"history_archive_s={self.stats.history_archive_seconds:.3f} "
            f"history_update_s={self.stats.history_update_seconds:.3f} "
            f"forecast_predict_s={self.stats.forecast_prediction_seconds:.3f} "
            f"residual_measure_s={self.stats.residual_measure_seconds:.3f} "
            f"residual_output_head_s={self.stats.residual_output_head_seconds:.3f} "
            f"residual_anchors={self.stats.residual_anchors} "
            f"residual_failures={self.stats.residual_failures} "
            f"residual_terminal_skips={self.stats.residual_skipped_terminal_probes} "
            f"residual_score_max={self.stats.residual_max_score:.6f} "
            f"residual_video_max={self.stats.residual_max_video_score:.6f} "
            f"residual_audio_max={self.stats.residual_max_audio_score:.6f} "
            f"residual_policy_max={self.stats.residual_policy_max_score:.6f} "
            f"feedback_threshold={_FEEDBACK_SCORE_THRESHOLD:.3f} "
            f"feedback_budget={_FEEDBACK_MAX_REFRESHES} "
            f"feedback_refreshes={self.stats.feedback_refreshes} "
            f"feedback_below_threshold={self.stats.feedback_suppressed_threshold} "
            f"feedback_budget_skips={self.stats.feedback_suppressed_budget} "
            f"speculative_calls={self.stats.speculative_forecast_calls} "
            f"discarded_actual_calls={self.stats.discarded_actual_calls} "
            f"rollbacks={self.stats.rollback_count} "
            f"rollback_threshold={_ROLLBACK_SCORE_THRESHOLD:.3f} "
            f"rollback_budget={_ROLLBACK_MAX_CORRECTIONS} "
            f"rollback_below_threshold={self.stats.rollback_suppressed_threshold} "
            f"rollback_budget_skips={self.stats.rollback_suppressed_budget} "
            f"replayed_transformer_calls={self.stats.replayed_transformer_calls} "
            f"offline_archive_s={self.stats.offline_archive_seconds:.3f} "
            f"offline_smoother_build_s={self.stats.offline_smoother_build_seconds:.3f} "
            f"offline_replay_steps={self.stats.offline_replay_steps} "
            f"offline_replay_calls={self.stats.offline_replay_model_calls} "
            f"offline_replay_anchor_steps={self.stats.offline_replay_anchor_steps} "
            f"offline_replay_smoothed_steps={self.stats.offline_replay_smoothed_steps} "
            f"offline_validation_samples_per_branch={self.stats.offline_validation_samples_per_branch} "
            f"offline_validation_anchors={self.stats.offline_validation_anchors} "
            f"offline_validation_streams={self.stats.offline_validation_streams} "
            f"offline_validation_s={self.stats.offline_validation_seconds:.3f} "
            f"offline_validation_audio_max={self.stats.offline_validation_audio_max:.6f} "
            f"offline_validation_video_max={self.stats.offline_validation_video_max:.6f} "
            f"offline_validation_packed_max={self.stats.offline_validation_packed_max:.6f} "
            f"offline_attenuated_predictions={self.stats.offline_attenuated_predictions} "
            f"offline_local_only_predictions={self.stats.offline_local_only_predictions} "
            f"offline_effective_blend_min={self.stats.offline_effective_blend_min:.6f} "
            f"offline_effective_blend_mean={self.stats.offline_effective_blend_mean:.6f} "
            f"offline_effective_blend_max={self.stats.offline_effective_blend_max:.6f} "
            f"offline_effective_audio_blend_min={self.stats.offline_effective_audio_blend_min:.6f} "
            f"offline_effective_audio_blend_mean={self.stats.offline_effective_audio_blend_mean:.6f} "
            f"offline_effective_audio_blend_max={self.stats.offline_effective_audio_blend_max:.6f} "
            f"offline_effective_video_blend_min={self.stats.offline_effective_video_blend_min:.6f} "
            f"offline_effective_video_blend_mean={self.stats.offline_effective_video_blend_mean:.6f} "
            f"offline_effective_video_blend_max={self.stats.offline_effective_video_blend_max:.6f} "
            f"offline_attenuated_audio_predictions={self.stats.offline_attenuated_audio_predictions} "
            f"offline_attenuated_video_predictions={self.stats.offline_attenuated_video_predictions} "
            f"offline_local_only_audio_predictions={self.stats.offline_local_only_audio_predictions} "
            f"offline_local_only_video_predictions={self.stats.offline_local_only_video_predictions} "
            f"offline_archive_mib={self.stats.offline_archive_bytes / (1024 * 1024):.1f} "
            f"offline_full_schedule_estimated_mib={self.stats.offline_estimated_archive_bytes / (1024 * 1024):.1f} "
            f"direct_history_updates={self.stats.direct_history_updates} "
            f"model_aware_mode={self.config.model_aware_mode!r} "
            f"model_aware_profile_cache_hit={self.stats.model_profile_cache_hit} "
            f"model_aware_profile_build_s={self.stats.model_profile_build_seconds:.6f} "
            f"model_aware_profile_lookup_s={self.stats.model_profile_lookup_seconds:.6f} "
            f"model_aware_profile_bytes={self.stats.model_profile_bytes} "
            f"model_aware_profile_workspace_bytes={self.stats.model_profile_workspace_bytes} "
            f"model_aware_patch_count={self.stats.model_profile_patch_count} "
            f"model_aware_unknown_patches={self.stats.model_profile_unknown_patch_count} "
            f"model_aware_sensitivity={self.stats.model_profile_sensitivity:.6f} "
            f"model_aware_patch_perturbation={self.stats.model_profile_patch_perturbation:.6f} "
            f"model_aware_forecasts={self.stats.model_aware_forecasts} "
            f"model_aware_anchor_updates={self.stats.model_aware_anchor_updates} "
            f"model_aware_failures={self.stats.model_aware_failures} "
            f"model_aware_risk_max={self.stats.model_aware_risk_max:.6f} "
            f"model_aware_confidence_min={self.stats.model_aware_confidence_min:.6f} "
            f"model_aware_correction_max={self.stats.model_aware_correction_max:.6f} "
            f"model_aware_extra_nfes={self.stats.adaptive_extra_nfes} "
            f"model_aware_overhead_s={self.stats.model_aware_overhead_seconds:.6f} "
            f"model_aware_decision_s={self.stats.model_aware_decision_seconds:.6f} "
            f"model_aware_evidence_s={self.stats.model_aware_evidence_seconds:.6f} "
            f"model_aware_evidence_weight_fit_s={self.stats.model_aware_evidence_weight_fit_seconds:.6f} "
            f"model_aware_evidence_sample_index_s={self.stats.model_aware_evidence_sample_index_seconds:.6f} "
            f"model_aware_evidence_device_transfer_s={self.stats.model_aware_evidence_device_transfer_seconds:.6f} "
            f"model_aware_evidence_sensitivity_transfer_s={self.stats.model_aware_evidence_sensitivity_transfer_seconds:.6f} "
            f"model_aware_evidence_scalar_transfer_s={self.stats.model_aware_evidence_scalar_transfer_seconds:.6f} "
            f"model_aware_evidence_reduction_s={self.stats.model_aware_evidence_reduction_seconds:.6f} "
            f"model_aware_evidence_exact_head_projection_s={self.stats.model_aware_evidence_exact_head_projection_seconds:.6f} "
            f"model_aware_evidence_fit_condition_s={self.stats.model_aware_evidence_fit_condition_seconds:.6f} "
            f"model_aware_subspace_gram_s={self.stats.model_aware_subspace_gram_seconds:.6f} "
            f"model_aware_subspace_solve_s={self.stats.model_aware_subspace_solve_seconds:.6f} "
            f"model_aware_subspace_workspace_bytes={self.stats.model_aware_subspace_workspace_bytes} "
            f"model_aware_head_materialization_s={self.stats.model_aware_head_materialization_seconds:.6f} "
            f"model_aware_head_materialized_bytes={self.stats.model_aware_head_materialized_bytes} "
            f"model_aware_exact_head_projection_s={self.stats.model_aware_exact_head_projection_seconds:.6f} "
            f"model_aware_exact_head_projection_calls={self.stats.model_aware_exact_head_projection_calls} "
            f"model_aware_exact_head_workspace_bytes={self.stats.model_aware_exact_head_workspace_bytes} "
            f"model_aware_fit_s={self.stats.model_aware_fit_seconds:.6f} "
            f"model_aware_correction_s={self.stats.model_aware_correction_seconds:.6f} "
            f"model_aware_causal_correction_s={self.stats.model_aware_causal_correction_seconds:.6f} "
            f"model_aware_offline_replay_correction_s={self.stats.model_aware_offline_correction_seconds:.6f} "
            f"model_aware_offline_replay_correction_applications={self.stats.model_aware_offline_correction_applications} "
            f"model_aware_scheduler_forecast_aggregate=max(audio,video) "
            f"model_aware_scheduler_curvature_aggregate=max(audio,video) "
            f"model_aware_correction_metric=final_layer_exact_linear_head_space "
            f"model_aware_correction_subspace=two_causal_actual_deltas "
            f"model_aware_subspace_model_comparison=exact_2d_vs_generic_2d_head_rms "
            f"model_aware_diagonal_ablation_metric=final_layer_gram_diagonal "
            f"model_aware_model_comparison_metric=exact_linear_head_space_rms "
            f"model_aware_head_metric_available={head_metric_available} "
            f"model_aware_correction_bound=rational_softsign_0.25 "
            f"model_aware_subspace_bound=radial_rational_softsign_0.25 "
            f"{subspace_summary} "
            f"model_aware_forecast_ratio_ewma={controller.forecast_ratio_ewma:.6f} "
            f"model_aware_curvature_ratio_ewma={controller.curvature_ratio_ewma:.6f} "
            f"model_aware_audio_forecast_ratio_mean={audio_forecast_mean:.6f} "
            f"model_aware_audio_forecast_ratio_max={controller.audio_forecast_ratio_max:.6f} "
            f"model_aware_video_forecast_ratio_mean={video_forecast_mean:.6f} "
            f"model_aware_video_forecast_ratio_max={controller.video_forecast_ratio_max:.6f} "
            f"model_aware_audio_model_corrected_ratio_mean={audio_model_mean:.6f} "
            f"model_aware_video_model_corrected_ratio_mean={video_model_mean:.6f} "
            f"model_aware_audio_generic_corrected_ratio_mean={audio_generic_mean:.6f} "
            f"model_aware_video_generic_corrected_ratio_mean={video_generic_mean:.6f} "
            f"model_aware_audio_diagonal_candidate_ratio_mean={audio_diagonal_mean:.6f} "
            f"model_aware_video_diagonal_candidate_ratio_mean={video_diagonal_mean:.6f} "
            f"model_aware_audio_exact_candidate_ratio_mean={audio_candidate_mean:.6f} "
            f"model_aware_video_exact_candidate_ratio_mean={video_candidate_mean:.6f} "
            f"model_aware_audio_diagonal_candidate_head_ratio_mean={audio_diagonal_head_mean:.6f} "
            f"model_aware_video_diagonal_candidate_head_ratio_mean={video_diagonal_head_mean:.6f} "
            f"model_aware_audio_exact_candidate_head_ratio_mean={audio_candidate_head_mean:.6f} "
            f"model_aware_video_exact_candidate_head_ratio_mean={video_candidate_head_mean:.6f} "
            f"model_aware_audio_generic_head_ratio_mean={audio_generic_head_mean:.6f} "
            f"model_aware_video_generic_head_ratio_mean={video_generic_head_mean:.6f} "
            f"model_aware_audio_exact_trust={controller.audio_model_trust:.6f} "
            f"model_aware_video_exact_trust={controller.video_model_trust:.6f} "
            f"model_aware_audio_diagonal_comparisons={controller.audio_diagonal_comparison_count} "
            f"model_aware_video_diagonal_comparisons={controller.video_diagonal_comparison_count} "
            f"model_aware_audio_diagonal_candidate_wins={controller.audio_diagonal_candidate_win_count} "
            f"model_aware_video_diagonal_candidate_wins={controller.video_diagonal_candidate_win_count} "
            f"model_aware_audio_diagonal_candidate_losses={controller.audio_diagonal_candidate_loss_count} "
            f"model_aware_video_diagonal_candidate_losses={controller.video_diagonal_candidate_loss_count} "
            f"model_aware_audio_exact_comparisons={controller.audio_model_comparison_count} "
            f"model_aware_video_exact_comparisons={controller.video_model_comparison_count} "
            f"model_aware_audio_exact_candidate_wins={controller.audio_model_candidate_win_count} "
            f"model_aware_video_exact_candidate_wins={controller.video_model_candidate_win_count} "
            f"model_aware_audio_exact_candidate_losses={controller.audio_model_candidate_loss_count} "
            f"model_aware_video_exact_candidate_losses={controller.video_model_candidate_loss_count} "
            f"model_aware_audio_diagonal_bound_active={controller.audio_diagonal_bound_active_count} "
            f"model_aware_video_diagonal_bound_active={controller.video_diagonal_bound_active_count} "
            f"model_aware_audio_exact_bound_active={controller.audio_model_bound_active_count} "
            f"model_aware_video_exact_bound_active={controller.video_model_bound_active_count} "
            f"model_aware_audio_generic_bound_active={controller.audio_generic_bound_active_count} "
            f"model_aware_video_generic_bound_active={controller.video_generic_bound_active_count} "
            f"model_aware_audio_gain_delta_pre_abs_mean={audio_delta_pre_mean:.6f} "
            f"model_aware_audio_gain_delta_pre_abs_max={controller.audio_gain_delta_pre_abs_max:.6f} "
            f"model_aware_video_gain_delta_pre_abs_mean={video_delta_pre_mean:.6f} "
            f"model_aware_video_gain_delta_pre_abs_max={controller.video_gain_delta_pre_abs_max:.6f} "
            f"model_aware_audio_gain_delta_post_abs_mean={audio_delta_post_mean:.6f} "
            f"model_aware_audio_gain_delta_post_abs_max={controller.audio_gain_delta_post_abs_max:.6f} "
            f"model_aware_video_gain_delta_post_abs_mean={video_delta_post_mean:.6f} "
            f"model_aware_video_gain_delta_post_abs_max={controller.video_gain_delta_post_abs_max:.6f} "
            f"model_aware_audio_gain_delta_applied_abs_mean={audio_delta_applied_mean:.6f} "
            f"model_aware_audio_gain_delta_applied_abs_max={controller.audio_gain_delta_applied_abs_max:.6f} "
            f"model_aware_video_gain_delta_applied_abs_mean={video_delta_applied_mean:.6f} "
            f"model_aware_video_gain_delta_applied_abs_max={controller.video_gain_delta_applied_abs_max:.6f} "
            f"model_aware_audio_exact_diagonal_delta_abs_mean={audio_exact_diagonal_delta_mean:.6f} "
            f"model_aware_audio_exact_diagonal_delta_abs_max={controller.audio_gain_delta_exact_diagonal_abs_max:.6f} "
            f"model_aware_video_exact_diagonal_delta_abs_mean={video_exact_diagonal_delta_mean:.6f} "
            f"model_aware_video_exact_diagonal_delta_abs_max={controller.video_gain_delta_exact_diagonal_abs_max:.6f} "
            f"model_corrected_ratio_mean={self.stats.model_aware_model_corrected_ratio_mean:.6f} "
            f"generic_corrected_ratio_mean={self.stats.model_aware_generic_corrected_ratio_mean:.6f} "
            f"history_storage={self.config.history_storage} "
            f"offline_archive_storage={self.config.offline_archive_storage} "
            f"offline_archive_device={str(offline_archive_device)!r} "
            f"history_device={str(self.prediction_history_device)!r} "
            f"history_mib={self.prediction_history_tensor_bytes / (1024 * 1024):.1f} "
            f"model_aware_evidence_history_mib={self.forecaster.evidence_tensor_bytes / (1024 * 1024):.3f} "
            f"model_aware_generic_evidence_bytes={self.forecaster.generic_evidence_tensor_bytes} "
            f"model_aware_exact_head_evidence_bytes={self.forecaster.exact_head_evidence_tensor_bytes} "
            f"reason={self.stats.disable_reason!r} "
            f"experimental_reason={self._experiment_disable_reason!r} "
            f"model_aware_reason={self._model_aware_disabled_reason!r}"
        )
