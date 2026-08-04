from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import SpectrumH3Config
from .forecast import HistoryWeightForecaster

LOG = logging.getLogger(__name__)


class ForecastRetryActual(RuntimeError):
    """Internal signal used to discard a partial forecast attempt transactionally."""


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
    history_archive_seconds: float = 0.0
    history_update_seconds: float = 0.0
    forecast_prediction_seconds: float = 0.0
    direct_history_updates: int = 0
    current_window: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None


@dataclass(slots=True)
class _ActualRecord:
    feature: torch.Tensor
    labels: tuple[Any, ...] | None


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
    calls: list[_CallState] = field(default_factory=list)
    actual_records: list[_ActualRecord] = field(default_factory=list)
    used_history_rows: set[int] = field(default_factory=set)
    fallback: bool = False


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
    next_step_id: int = 0


class SpectrumH3Runtime:
    def __init__(self, config: SpectrumH3Config):
        self.config = config.validate()
        self.forecaster = HistoryWeightForecaster(
            degree=self.config.degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.max_history,
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
        self._disabled = False
        self._disable_reason: str | None = None

    @property
    def active_run_id(self) -> int | None:
        return None if self._run is None else self._run.run_id

    @property
    def active_step_id(self) -> int | None:
        return None if self._step is None else self._step.step_id

    @property
    def supported_sampler(self) -> bool:
        return self._run is not None and self._run.supported_sampler

    @property
    def disabled_reason(self) -> str | None:
        return self._disable_reason

    @property
    def history_labels(self) -> tuple[Any, ...] | None:
        return self._history_labels

    def start_run(
        self,
        sigmas: torch.Tensor,
        sampler_name: str,
        *,
        supported_sampler: bool,
        max_consecutive_forecasts: int | None = None,
        min_actual_steps_after_forecast: int = 0,
        min_tail_actual_steps: int = 0,
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
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        sigma_values = torch.as_tensor(sigmas).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        total_steps = max(0, sigma_values.numel() - 1)
        evaluated = sigma_values[:-1]
        finite_schedule = bool(evaluated.numel()) and bool(torch.isfinite(evaluated).all().item())
        sigma_min = float(evaluated.min().item()) if finite_schedule else 0.0
        sigma_max = float(evaluated.max().item()) if finite_schedule else 0.0
        schedule_valid = finite_schedule and math.isfinite(sigma_min) and math.isfinite(sigma_max) and sigma_max > sigma_min

        self._run_counter += 1
        effective_supported = bool(supported_sampler and schedule_valid and total_steps > 0)
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
        )
        self._step = None
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._disabled = not effective_supported
        if not supported_sampler:
            self._disable_reason = f"sampler {sampler_name!r} is not allowlisted for one-call solver-step tracking"
        elif not schedule_valid:
            self._disable_reason = "supplied sigma schedule is empty, nonfinite, or has no usable range"
        elif total_steps <= 0:
            self._disable_reason = "supplied sigma schedule has no solver steps"
        else:
            self._disable_reason = None
        self.stats = RuntimeStats(
            run_id=self._run.run_id,
            sampler_name=self._run.sampler_name,
            total_steps=total_steps,
            current_window=self._current_window,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
        )
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

    def coordinate_for_timestep(self, timestep: torch.Tensor | float) -> float:
        if self._run is None:
            raise RuntimeError("Spectrum H3 runtime is outside a sampling run")
        value_tensor = torch.as_tensor(timestep).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
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

        effective_tail = max(self.config.tail_actual_steps, self._run.min_tail_actual_steps)
        tail_start = max(0, self._run.total_steps - effective_tail)
        advances_window = False
        if self.config.force_actual:
            actual, reason = True, "forced-actual validation mode"
        elif self._disabled:
            actual, reason = True, self._disable_reason or "forecasting disabled"
        elif step_id < self.config.warmup_steps:
            actual, reason = True, "warmup"
        elif step_id >= tail_start:
            actual, reason = True, "final actual tail"
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
        if not actual and self._required_actual_refreshes > 0:
            actual = True
            reason = "post-forecast sampler refresh"
            advances_window = False

        self._step = _StepState(
            step_id=step_id,
            coordinate=coordinate,
            adaptive_recompute=advances_window,
            mode="actual" if actual else "forecast",
            reason=reason,
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

    def _disable_forecasting(self, reason: str) -> None:
        if not self._disabled:
            self._disabled = True
            self._disable_reason = str(reason)
            self.stats.disabled = True
            self.stats.disable_reason = self._disable_reason
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None

    def _fallback_or_retry(self, step: _StepState, reason: str) -> None:
        self._disable_forecasting(reason)
        if any(call.used_forecast for call in step.calls):
            raise ForecastRetryActual(reason)
        step.mode = "actual"
        step.reason = reason
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
            archived = feature.detach().to(device="cpu", dtype=feature.dtype, copy=True).contiguous()
        finally:
            self.stats.history_archive_seconds += time.perf_counter() - started
        call.observed_actual = True
        step.actual_records.append(_ActualRecord(archived, call.labels))

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
        if self._history_labels is None or call.labels is None:
            self._fallback_or_retry(step, "branch labels are missing; forecast row correspondence is unproven")
            return None
        if len(call.labels) != call.expected_shape[0] or len(set(call.labels)) != len(call.labels):
            self._fallback_or_retry(step, "branch labels are duplicate or do not match the model-call batch")
            return None
        positions = []
        for label in call.labels:
            try:
                position = self._history_labels.index(label)
            except ValueError:
                self._fallback_or_retry(step, "conditional branch identity changed")
                return None
            if position in step.used_history_rows:
                self._fallback_or_retry(step, "conditional branch row was assigned more than once")
                return None
            positions.append(position)

        history_shape = self.forecaster.feature_shape
        if history_shape is None or tuple(call.expected_shape[1:]) != tuple(history_shape[1:]):
            self._fallback_or_retry(step, "target audio/video row count or hidden width changed")
            return None
        started = time.perf_counter()
        try:
            predicted = self.forecaster.predict(
                step.coordinate,
                self.config.blend_weight,
                rows=positions,
                device=device,
                dtype=dtype,
            )
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
        step.fallback = True
        step.calls.clear()
        step.actual_records.clear()
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

    def finalize_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if not step.calls:
            if step.fallback and self._disabled:
                self._consecutive_forecasts = 0
                self.stats.actual_steps += 1
                self.stats.current_window = self._current_window
                self._step = None
                return
            self._disable_forecasting("solver step completed without an H3 model call")
            raise RuntimeError("Spectrum H3 solver step completed without an H3 model call")

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
        else:
            if any(call.used_forecast for call in step.calls):
                raise RuntimeError("actual solver step retained a forecasted subcall")
            started = time.perf_counter()
            try:
                combined = self._aggregate_actual(step)
                if combined is not None and not self._disabled:
                    try:
                        self.forecaster.update(step.coordinate, combined, take_ownership=True)
                    except ValueError as exc:
                        self._disable_forecasting(f"actual H3 feature is incompatible with history: {exc}")
            finally:
                self.stats.history_update_seconds += time.perf_counter() - started
            self._consecutive_forecasts = 0
            self._required_actual_refreshes = max(0, self._required_actual_refreshes - 1)
            self.stats.actual_steps += 1
            self.stats.actual_transformer_calls += len(step.actual_records)
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

        self.stats.current_window = self._current_window
        self._step = None

    def abort_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if self._run is not None and self._run.next_step_id == step.step_id + 1:
            self._run.next_step_id = step.step_id
        self._step = None

    def debug_summary(self) -> str:
        return (
            f"run_id={self.stats.run_id} sampler={self.stats.sampler_name} "
            f"steps={self.stats.total_steps} actual_steps={self.stats.actual_steps} "
            f"forecast_steps={self.stats.forecast_steps} "
            f"actual_transformer_calls={self.stats.actual_transformer_calls} "
            f"forecast_calls={self.stats.forecast_model_calls} "
            f"fallbacks={self.stats.forecast_fallbacks} disabled={self.stats.disabled} "
            f"history_archive_s={self.stats.history_archive_seconds:.3f} "
            f"history_update_s={self.stats.history_update_seconds:.3f} "
            f"forecast_predict_s={self.stats.forecast_prediction_seconds:.3f} "
            f"direct_history_updates={self.stats.direct_history_updates} "
            f"history_mib={self.forecaster.history_tensor_bytes / (1024 * 1024):.1f} "
            f"reason={self.stats.disable_reason!r}"
        )
