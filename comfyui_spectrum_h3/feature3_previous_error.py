from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from . import feature3_direction as _base
from . import feature3_direction_normalization as _norm
from . import minimax_h3 as _h3
from .model_aware import ModelAwareController, ModelAwareForecastDecision
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)

_ERROR_SAMPLE_ROWS = 32


@dataclass(frozen=True, slots=True)
class PreviousErrorState:
    """Strictly prior-anchor sampled residuals in hidden and native output space."""

    anchor_id: int
    hidden_residual: torch.Tensor
    output_residual: torch.Tensor | None

    @property
    def tensor_bytes(self) -> int:
        values = (self.hidden_residual, self.output_residual)
        return sum(
            value.numel() * value.element_size()
            for value in values
            if torch.is_tensor(value)
        )


@dataclass(frozen=True, slots=True)
class ErrorCandidateTelemetry:
    eligible: bool = False
    alpha_history_count: int = 0
    raw_direction_norm_ratio: float = 0.0
    normalized_direction_norm_ratio: float = 0.0
    instantaneous_alpha: float = 0.0
    alpha_used: float = 0.0
    raw_correction_norm_ratio: float = 0.0
    bounded_correction_norm_ratio: float = 0.0
    radial_scale: float = 1.0
    bound_active: bool = False
    ordinary_ratio: float = 0.0
    static_head_ratio: float = 0.0
    final_layer_ratio: float = 0.0


def static_output_adjoint(
    output_residual: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Return W^T e in row-major notation: e @ W."""
    if output_residual.ndim < 1 or head_weight.ndim != 2:
        raise ValueError("static adjoint requires [..., out] residual and [out, hidden] head")
    if int(output_residual.shape[-1]) != int(head_weight.shape[0]):
        raise ValueError("output residual width does not match FinalLayer head")
    operator = head_weight.to(device=output_residual.device, dtype=torch.float32)
    return torch.matmul(output_residual.to(torch.float32), operator)


def _alpha_used(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    confidence: float,
) -> tuple[float, int]:
    prefix = f"_feature3_error_{stream}_{kind}"
    count = int(getattr(controller, f"{prefix}_alpha_count"))
    ewma = float(getattr(controller, f"{prefix}_alpha_ewma"))
    confidence_scale = max(0.25, min(1.0, float(confidence)))
    return ewma * confidence_scale, count


def _update_alpha(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    instantaneous: float,
) -> None:
    if not math.isfinite(float(instantaneous)):
        return
    prefix = f"_feature3_error_{stream}_{kind}"
    count = int(getattr(controller, f"{prefix}_alpha_count"))
    prior = float(getattr(controller, f"{prefix}_alpha_ewma"))
    blend = 0.5 if count < 2 else 0.3
    bounded = _base._soft_limit(float(instantaneous), _base._DIRECTION_ALPHA_LIMIT)
    setattr(
        controller,
        f"{prefix}_alpha_ewma",
        bounded if count == 0 else (1.0 - blend) * prior + blend * bounded,
    )
    setattr(controller, f"{prefix}_alpha_count", count + 1)


def _instantaneous_alpha(
    residual: torch.Tensor,
    direction: torch.Tensor,
    direction_eligible: torch.Tensor,
) -> torch.Tensor:
    flat_direction = direction.reshape(-1)
    flat_residual = residual.reshape(-1)
    denominator = torch.dot(flat_direction, flat_direction)
    numerator = torch.dot(flat_residual, flat_direction)
    tiny = torch.as_tensor(
        torch.finfo(torch.float32).tiny,
        dtype=torch.float32,
        device=denominator.device,
    )
    value = numerator / denominator.clamp_min(tiny)
    valid = direction_eligible & (denominator > 0.0) & torch.isfinite(value)
    return torch.where(valid, value, torch.zeros_like(value))


def _comparison(
    controller: ModelAwareController,
    stream: str,
    comparison: str,
    baseline: float,
    candidate: float,
) -> float:
    if not all(math.isfinite(value) for value in (baseline, candidate)):
        return 0.0
    if baseline <= 0.0 or candidate <= 0.0:
        return 0.0
    advantage = (baseline - candidate) / max(baseline, 1e-12)
    prefix = f"_feature3_error_{stream}_{comparison}"
    setattr(controller, f"{prefix}_count", getattr(controller, f"{prefix}_count") + 1)
    setattr(
        controller,
        f"{prefix}_advantage_sum",
        getattr(controller, f"{prefix}_advantage_sum") + advantage,
    )
    setattr(
        controller,
        f"{prefix}_advantage_abs_max",
        max(getattr(controller, f"{prefix}_advantage_abs_max"), abs(advantage)),
    )
    if candidate < baseline:
        setattr(controller, f"{prefix}_wins", getattr(controller, f"{prefix}_wins") + 1)
    elif candidate > baseline:
        setattr(controller, f"{prefix}_losses", getattr(controller, f"{prefix}_losses") + 1)
    return advantage


def _reset_error_state(controller: ModelAwareController) -> None:
    controller._feature3_error_previous: dict[str, PreviousErrorState] = {}
    controller._feature3_error_row_indices: dict[str, torch.Tensor] = {}
    controller._feature3_error_row_shapes: dict[str, tuple[int, int]] = {}
    controller._feature3_error_last: dict[
        str,
        tuple[ErrorCandidateTelemetry, ErrorCandidateTelemetry, ErrorCandidateTelemetry],
    ] = {}
    controller._feature3_error_compute_seconds = 0.0
    controller._feature3_error_geometry_seconds = 0.0
    controller._feature3_error_static_seconds = 0.0
    controller._feature3_error_full_seconds = 0.0
    controller._feature3_error_vjp_seconds = 0.0
    controller._feature3_error_scalar_transfer_seconds = 0.0
    controller._feature3_error_workspace_bytes = 0
    controller._feature3_error_failures = 0
    controller._feature3_error_disabled_reason: str | None = None
    for stream in ("audio", "video"):
        for kind in ("residual", "static", "full"):
            prefix = f"_feature3_error_{stream}_{kind}"
            setattr(controller, f"{prefix}_alpha_ewma", 0.0)
            setattr(controller, f"{prefix}_alpha_count", 0)
            setattr(controller, f"{prefix}_eligible_count", 0)
            setattr(controller, f"{prefix}_fallback_count", 0)
            setattr(controller, f"{prefix}_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_ratio_max", 0.0)
            setattr(controller, f"{prefix}_raw_direction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_raw_direction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_normalized_direction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_normalized_direction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_bounded_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_bounded_ratio_max", 0.0)
            setattr(controller, f"{prefix}_radial_scale_sum", 0.0)
            setattr(controller, f"{prefix}_radial_scale_min", 1.0)
            setattr(controller, f"{prefix}_bound_active_count", 0)
        for comparison in (
            "residual_vs_generic",
            "static_vs_residual",
            "static_vs_generic",
            "full_vs_residual",
            "full_vs_static",
            "full_vs_generic",
        ):
            prefix = f"_feature3_error_{stream}_{comparison}"
            setattr(controller, f"{prefix}_count", 0)
            setattr(controller, f"{prefix}_wins", 0)
            setattr(controller, f"{prefix}_losses", 0)
            setattr(controller, f"{prefix}_advantage_sum", 0.0)
            setattr(controller, f"{prefix}_advantage_abs_max", 0.0)


def _numeric_keys() -> list[str]:
    keys = [
        "_feature3_error_compute_seconds",
        "_feature3_error_geometry_seconds",
        "_feature3_error_static_seconds",
        "_feature3_error_full_seconds",
        "_feature3_error_vjp_seconds",
        "_feature3_error_scalar_transfer_seconds",
        "_feature3_error_workspace_bytes",
        "_feature3_error_failures",
    ]
    for stream in ("audio", "video"):
        for kind in ("residual", "static", "full"):
            prefix = f"_feature3_error_{stream}_{kind}"
            keys.extend(
                f"{prefix}_{suffix}"
                for suffix in (
                    "alpha_ewma",
                    "alpha_count",
                    "eligible_count",
                    "fallback_count",
                    "ratio_sum",
                    "ratio_max",
                    "raw_direction_ratio_sum",
                    "raw_direction_ratio_max",
                    "normalized_direction_ratio_sum",
                    "normalized_direction_ratio_max",
                    "bounded_ratio_sum",
                    "bounded_ratio_max",
                    "radial_scale_sum",
                    "radial_scale_min",
                    "bound_active_count",
                )
            )
        for comparison in (
            "residual_vs_generic",
            "static_vs_residual",
            "static_vs_generic",
            "full_vs_residual",
            "full_vs_static",
            "full_vs_generic",
        ):
            prefix = f"_feature3_error_{stream}_{comparison}"
            keys.extend(
                f"{prefix}_{suffix}"
                for suffix in (
                    "count",
                    "wins",
                    "losses",
                    "advantage_sum",
                    "advantage_abs_max",
                )
            )
    return keys


def _evidence_bytes(controller: ModelAwareController) -> int:
    previous = sum(state.tensor_bytes for state in controller._feature3_error_previous.values())
    indices = sum(
        value.numel() * value.element_size()
        for value in controller._feature3_error_row_indices.values()
    )
    return previous + indices


def _row_indices(
    controller: ModelAwareController,
    forecaster: Any,
    stream: str,
    row_count: int,
    branch_count: int,
) -> torch.Tensor:
    shape = (int(row_count), int(branch_count))
    previous_shape = controller._feature3_error_row_shapes.get(stream)
    if previous_shape is not None and previous_shape != shape:
        raise ValueError("previous-error row correspondence changed during the run")
    indices = controller._feature3_error_row_indices.get(stream)
    if indices is None:
        indices = forecaster._complete_row_indices(
            row_count,
            branch_count,
            torch.device("cpu"),
            limit=_ERROR_SAMPLE_ROWS,
        ).contiguous()
        controller._feature3_error_row_indices[stream] = indices
        controller._feature3_error_row_shapes[stream] = shape
    return indices


def _sample_feature_rows(
    feature: torch.Tensor,
    start: int,
    end: int,
    indices_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    indices = indices_cpu.to(device=feature.device)
    selected = feature[:, int(start) : int(end)].index_select(1, indices)
    return selected.detach().to(device=device, dtype=torch.float32).contiguous()


def _history_rows(
    runtime: SpectrumH3Runtime,
    start: int,
    end: int,
    indices_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    shape = runtime.forecaster.feature_shape
    if shape is None:
        raise RuntimeError("previous-error screen lost the forecaster feature shape")
    rows = []
    for entry in runtime.forecaster._history:
        feature = entry.feature_flat.reshape(shape)
        rows.append(
            _sample_feature_rows(
                feature,
                start,
                end,
                indices_cpu,
                device=device,
            )
        )
    return rows


def _record_candidate(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    telemetry: ErrorCandidateTelemetry,
) -> None:
    prefix = f"_feature3_error_{stream}_{kind}"
    if not telemetry.eligible:
        setattr(controller, f"{prefix}_fallback_count", getattr(controller, f"{prefix}_fallback_count") + 1)
        return
    setattr(controller, f"{prefix}_eligible_count", getattr(controller, f"{prefix}_eligible_count") + 1)
    setattr(controller, f"{prefix}_ratio_sum", getattr(controller, f"{prefix}_ratio_sum") + telemetry.ordinary_ratio)
    setattr(controller, f"{prefix}_ratio_max", max(getattr(controller, f"{prefix}_ratio_max"), telemetry.ordinary_ratio))
    for suffix, value in (
        ("raw_direction_ratio", telemetry.raw_direction_norm_ratio),
        ("normalized_direction_ratio", telemetry.normalized_direction_norm_ratio),
        ("bounded_ratio", telemetry.bounded_correction_norm_ratio),
    ):
        setattr(controller, f"{prefix}_{suffix}_sum", getattr(controller, f"{prefix}_{suffix}_sum") + value)
        setattr(controller, f"{prefix}_{suffix}_max", max(getattr(controller, f"{prefix}_{suffix}_max"), value))
    setattr(controller, f"{prefix}_radial_scale_sum", getattr(controller, f"{prefix}_radial_scale_sum") + telemetry.radial_scale)
    setattr(controller, f"{prefix}_radial_scale_min", min(getattr(controller, f"{prefix}_radial_scale_min"), telemetry.radial_scale))
    if telemetry.bound_active:
        setattr(controller, f"{prefix}_bound_active_count", getattr(controller, f"{prefix}_bound_active_count") + 1)


def _consume_geometry(
    runtime: SpectrumH3Runtime,
    step_id: int,
) -> _base.FinalLayerGeometry | None:
    pending = getattr(runtime, "_feature3_error_pending_geometry", None)
    if not pending:
        return None
    entries = pending.pop(int(step_id), {})
    if not entries:
        return None
    geometries = list(entries.values())
    first = geometries[0]
    for geometry in geometries[1:]:
        if geometry.norm_eps != first.norm_eps:
            return None
        if geometry.norm_weight.shape != first.norm_weight.shape:
            return None
        if geometry.audio_scale.shape != first.audio_scale.shape:
            return None
        if geometry.video_scale.shape != first.video_scale.shape:
            return None
    return first


def _capture_geometry(
    inner: Any,
    state: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _base.FinalLayerGeometry:
    import comfy.ops

    started = time.perf_counter()
    shift, scale = inner.final_layer.adaln_proj(state.t_emb)
    del shift
    audio_row = int(state.audio_timestep_row)
    video_row = int(state.video_timestep_row)
    if scale.ndim != 2 or max(audio_row, video_row) >= int(scale.shape[0]):
        raise RuntimeError("native FinalLayer AdaLN scale shape is incompatible")
    norm = inner.final_layer.norm
    eps = float(norm.eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise RuntimeError("native FinalLayer RMSNorm epsilon is invalid")
    probe = torch.empty((), device=device, dtype=dtype)
    if getattr(norm, "weight", None) is None:
        norm_weight = torch.ones(
            int(inner.hidden_size),
            device=device,
            dtype=dtype,
        )
    else:
        weight, bias, offload_state = comfy.ops.cast_bias_weight(
            norm,
            probe,
            offloadable=True,
        )
        try:
            norm_weight = weight.detach().clone()
        finally:
            comfy.ops.uncast_bias_weight(norm, weight, bias, offload_state)
    geometry = _base.FinalLayerGeometry(
        norm_weight=norm_weight,
        norm_eps=eps,
        audio_scale=scale[audio_row].detach().clone(),
        video_scale=scale[video_row].detach().clone(),
    )
    geometry._capture_elapsed = time.perf_counter() - started if hasattr(geometry, "_capture_elapsed") else 0.0
    return geometry


def _candidate_tensors(
    direction: torch.Tensor | None,
    residual: torch.Tensor,
    delta: torch.Tensor,
    *,
    alpha_used: float,
) -> tuple[Any, Any, torch.Tensor]:
    if direction is None:
        zero = torch.zeros((), dtype=torch.float32, device=delta.device)
        normalization = _norm._TensorNormalizedDirection(
            direction=torch.zeros_like(delta),
            raw_direction_norm=zero,
            reference_delta_norm=torch.linalg.vector_norm(delta),
            raw_direction_norm_ratio=zero,
            normalized_direction_norm_ratio=zero,
            eligible=torch.zeros((), dtype=torch.bool, device=delta.device),
        )
    else:
        normalization = _norm._normalize_direction_tensor(direction, delta)
    bounded = _base._radially_bound_direction_tensor(
        alpha_used,
        normalization.direction,
        delta,
    )
    instantaneous = _instantaneous_alpha(
        residual,
        normalization.direction,
        normalization.eligible,
    )
    return normalization, bounded, instantaneous


def _observe_previous_error_anchor(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    raw_weights_by_stream: dict[str, torch.Tensor],
    exact_head_weights: dict[str, torch.Tensor],
) -> None:
    if runtime.config.model_aware_mode != "full":
        return
    if runtime._offline_phase == "replay":
        return
    controller = runtime.model_aware
    if controller._feature3_error_disabled_reason is not None:
        return
    if runtime.forecaster.history_length < 2 or runtime.forecaster.feature_shape is None:
        return

    started = time.perf_counter()
    geometry = _consume_geometry(runtime, step.step_id)
    generic_audio, generic_video = controller.generic_correction_gains(decision)
    last: dict[
        str,
        tuple[ErrorCandidateTelemetry, ErrorCandidateTelemetry, ErrorCandidateTelemetry],
    ] = {}
    try:
        for stream, start, end in runtime._stream_ranges(step.calls[0]):
            if stream not in {"audio", "video"}:
                continue
            head = exact_head_weights.get(stream)
            weights = raw_weights_by_stream.get(stream)
            if head is None or weights is None:
                continue
            branch_count = int(combined.shape[0])
            row_count = int(end) - int(start)
            indices_cpu = _row_indices(
                controller,
                runtime.forecaster,
                stream,
                row_count,
                branch_count,
            )
            history = _history_rows(
                runtime,
                start,
                end,
                indices_cpu,
                device=combined.device,
            )
            if len(history) != runtime.forecaster.history_length or len(history) < 2:
                raise RuntimeError("previous-error sampled history is not aligned")
            if len(weights) != len(history):
                raise RuntimeError("previous-error prediction weights are not aligned")

            actual_native = _sample_feature_rows(
                combined,
                start,
                end,
                indices_cpu,
                device=combined.device,
            )
            predicted = torch.zeros_like(actual_native)
            for weight_value, sample in zip(weights.tolist(), history, strict=True):
                if weight_value != 0.0:
                    predicted.add_(sample, alpha=float(weight_value))
            latest = history[-1]
            previous = history[-2]
            delta = latest - previous
            current_residual = actual_native - predicted
            hold_rms = _base._tensor_rms(actual_native - latest).clamp_min(
                _base._DIRECTION_EPS
            )
            generic_gain = generic_audio if stream == "audio" else generic_video
            exact_gain = (
                decision.audio_correction_telemetry.model_candidate_gain
                if stream == "audio"
                else decision.video_correction_telemetry.model_candidate_gain
            )
            generic_ratio = _base._tensor_rms(
                actual_native - (predicted + float(generic_gain) * delta)
            ) / hold_rms
            exact_ratio = _base._tensor_rms(
                actual_native - (predicted + float(exact_gain) * delta)
            ) / hold_rms

            prior = controller._feature3_error_previous.get(stream)
            residual_raw = None if prior is None else prior.hidden_residual
            static_started = time.perf_counter()
            static_raw = (
                None
                if prior is None or prior.output_residual is None
                else static_output_adjoint(prior.output_residual, head)
            )
            controller._feature3_error_static_seconds += time.perf_counter() - static_started
            adaln_scale = None
            if geometry is not None:
                adaln_scale = geometry.audio_scale if stream == "audio" else geometry.video_scale
            full_raw = None
            if prior is not None and prior.output_residual is not None and adaln_scale is not None:
                full_started = time.perf_counter()
                vjp_started = time.perf_counter()
                full_raw = _base.final_layer_vjp(
                    predicted.to(dtype=actual_native.dtype),
                    prior.output_residual,
                    norm_weight=geometry.norm_weight,
                    norm_eps=geometry.norm_eps,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                ).to(torch.float32)
                vjp_elapsed = time.perf_counter() - vjp_started
                controller._feature3_error_vjp_seconds += vjp_elapsed
                controller._feature3_error_full_seconds += time.perf_counter() - full_started

            residual_alpha, residual_count = _alpha_used(
                controller, stream, "residual", decision.confidence
            )
            static_alpha, static_count = _alpha_used(
                controller, stream, "static", decision.confidence
            )
            full_alpha, full_count = _alpha_used(
                controller, stream, "full", decision.confidence
            )
            residual_norm, residual_bound, residual_instant = _candidate_tensors(
                residual_raw,
                current_residual,
                delta,
                alpha_used=residual_alpha,
            )
            static_norm, static_bound, static_instant = _candidate_tensors(
                static_raw,
                current_residual,
                delta,
                alpha_used=static_alpha,
            )
            full_norm, full_bound, full_instant = _candidate_tensors(
                full_raw,
                current_residual,
                delta,
                alpha_used=full_alpha,
            )

            residual_prediction = predicted + residual_bound.correction
            static_prediction = predicted + static_bound.correction
            full_prediction = predicted + full_bound.correction
            residual_ratio = _base._tensor_rms(actual_native - residual_prediction) / hold_rms
            static_ratio = _base._tensor_rms(actual_native - static_prediction) / hold_rms
            full_ratio = _base._tensor_rms(actual_native - full_prediction) / hold_rms

            static_head_hold = _base._tensor_rms(
                _base._static_head_difference(actual_native, latest, head)
            ).clamp_min(_base._DIRECTION_EPS)
            static_head_ratio = _base._tensor_rms(
                _base._static_head_difference(actual_native, static_prediction, head)
            ) / static_head_hold
            full_final_ratio = torch.zeros((), dtype=torch.float32, device=combined.device)
            if geometry is not None and adaln_scale is not None:
                final_hold = _base._tensor_rms(
                    _base._final_layer_difference(
                        actual_native.to(dtype=actual_native.dtype),
                        latest.to(dtype=actual_native.dtype),
                        geometry=geometry,
                        adaln_scale=adaln_scale,
                        head_weight=head,
                    )
                ).clamp_min(_base._DIRECTION_EPS)
                full_final_ratio = _base._tensor_rms(
                    _base._final_layer_difference(
                        actual_native.to(dtype=actual_native.dtype),
                        full_prediction.to(dtype=actual_native.dtype),
                        geometry=geometry,
                        adaln_scale=adaln_scale,
                        head_weight=head,
                    )
                ) / final_hold

            current_output_residual = None
            if geometry is not None and adaln_scale is not None:
                current_output_residual = _base._final_layer_difference(
                    actual_native.to(dtype=actual_native.dtype),
                    predicted.to(dtype=actual_native.dtype),
                    geometry=geometry,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                ).detach().to(torch.float32).contiguous()

            transfer_started = time.perf_counter()
            values = torch.stack(
                (
                    generic_ratio,
                    exact_ratio,
                    residual_ratio,
                    static_ratio,
                    full_ratio,
                    static_head_ratio,
                    full_final_ratio,
                    residual_instant,
                    static_instant,
                    full_instant,
                    residual_norm.raw_direction_norm_ratio,
                    residual_norm.normalized_direction_norm_ratio,
                    residual_bound.raw_norm_ratio,
                    residual_bound.bounded_norm_ratio,
                    residual_bound.radial_scale,
                    residual_bound.bound_active.to(torch.float32),
                    residual_bound.eligible.to(torch.float32),
                    static_norm.raw_direction_norm_ratio,
                    static_norm.normalized_direction_norm_ratio,
                    static_bound.raw_norm_ratio,
                    static_bound.bounded_norm_ratio,
                    static_bound.radial_scale,
                    static_bound.bound_active.to(torch.float32),
                    static_bound.eligible.to(torch.float32),
                    full_norm.raw_direction_norm_ratio,
                    full_norm.normalized_direction_norm_ratio,
                    full_bound.raw_norm_ratio,
                    full_bound.bounded_norm_ratio,
                    full_bound.radial_scale,
                    full_bound.bound_active.to(torch.float32),
                    full_bound.eligible.to(torch.float32),
                )
            ).detach().to(device="cpu").tolist()
            controller._feature3_error_scalar_transfer_seconds += (
                time.perf_counter() - transfer_started
            )
            (
                generic_ratio_v,
                exact_ratio_v,
                residual_ratio_v,
                static_ratio_v,
                full_ratio_v,
                static_head_ratio_v,
                full_final_ratio_v,
                residual_instant_v,
                static_instant_v,
                full_instant_v,
                residual_raw_dir_v,
                residual_norm_dir_v,
                residual_raw_corr_v,
                residual_bounded_v,
                residual_radial_v,
                residual_active_v,
                residual_direction_eligible_v,
                static_raw_dir_v,
                static_norm_dir_v,
                static_raw_corr_v,
                static_bounded_v,
                static_radial_v,
                static_active_v,
                static_direction_eligible_v,
                full_raw_dir_v,
                full_norm_dir_v,
                full_raw_corr_v,
                full_bounded_v,
                full_radial_v,
                full_active_v,
                full_direction_eligible_v,
            ) = values

            residual_eligible = bool(
                residual_direction_eligible_v
                and residual_count > 0
                and math.isfinite(residual_ratio_v)
            )
            static_eligible = bool(
                static_direction_eligible_v
                and static_count > 0
                and math.isfinite(static_ratio_v)
            )
            full_eligible = bool(
                full_direction_eligible_v
                and full_count > 0
                and math.isfinite(full_ratio_v)
            )
            residual_telemetry = ErrorCandidateTelemetry(
                eligible=residual_eligible,
                alpha_history_count=residual_count,
                raw_direction_norm_ratio=residual_raw_dir_v,
                normalized_direction_norm_ratio=residual_norm_dir_v,
                instantaneous_alpha=residual_instant_v,
                alpha_used=residual_alpha,
                raw_correction_norm_ratio=residual_raw_corr_v,
                bounded_correction_norm_ratio=residual_bounded_v,
                radial_scale=residual_radial_v,
                bound_active=bool(residual_active_v),
                ordinary_ratio=residual_ratio_v if residual_eligible else 0.0,
            )
            static_telemetry = ErrorCandidateTelemetry(
                eligible=static_eligible,
                alpha_history_count=static_count,
                raw_direction_norm_ratio=static_raw_dir_v,
                normalized_direction_norm_ratio=static_norm_dir_v,
                instantaneous_alpha=static_instant_v,
                alpha_used=static_alpha,
                raw_correction_norm_ratio=static_raw_corr_v,
                bounded_correction_norm_ratio=static_bounded_v,
                radial_scale=static_radial_v,
                bound_active=bool(static_active_v),
                ordinary_ratio=static_ratio_v if static_eligible else 0.0,
                static_head_ratio=static_head_ratio_v if static_eligible else 0.0,
            )
            full_telemetry = ErrorCandidateTelemetry(
                eligible=full_eligible,
                alpha_history_count=full_count,
                raw_direction_norm_ratio=full_raw_dir_v,
                normalized_direction_norm_ratio=full_norm_dir_v,
                instantaneous_alpha=full_instant_v,
                alpha_used=full_alpha,
                raw_correction_norm_ratio=full_raw_corr_v,
                bounded_correction_norm_ratio=full_bounded_v,
                radial_scale=full_radial_v,
                bound_active=bool(full_active_v),
                ordinary_ratio=full_ratio_v if full_eligible else 0.0,
                final_layer_ratio=full_final_ratio_v if full_eligible else 0.0,
            )

            _record_candidate(controller, stream, "residual", residual_telemetry)
            _record_candidate(controller, stream, "static", static_telemetry)
            _record_candidate(controller, stream, "full", full_telemetry)
            if residual_eligible:
                _comparison(controller, stream, "residual_vs_generic", generic_ratio_v, residual_ratio_v)
            if static_eligible:
                if residual_eligible:
                    _comparison(controller, stream, "static_vs_residual", residual_ratio_v, static_ratio_v)
                _comparison(controller, stream, "static_vs_generic", generic_ratio_v, static_ratio_v)
            if full_eligible:
                if residual_eligible:
                    _comparison(controller, stream, "full_vs_residual", residual_ratio_v, full_ratio_v)
                if static_eligible:
                    _comparison(controller, stream, "full_vs_static", static_ratio_v, full_ratio_v)
                _comparison(controller, stream, "full_vs_generic", generic_ratio_v, full_ratio_v)

            if bool(residual_direction_eligible_v):
                _update_alpha(controller, stream, "residual", residual_instant_v)
            if bool(static_direction_eligible_v):
                _update_alpha(controller, stream, "static", static_instant_v)
            if bool(full_direction_eligible_v):
                _update_alpha(controller, stream, "full", full_instant_v)

            # Only after scoring/current-alpha fitting do we publish anchor N's
            # residuals for a later target. This prevents same-anchor hindsight.
            controller._feature3_error_previous[stream] = PreviousErrorState(
                anchor_id=int(step.step_id),
                hidden_residual=current_residual.detach().to(torch.float32).contiguous(),
                output_residual=current_output_residual,
            )
            last[stream] = (residual_telemetry, static_telemetry, full_telemetry)

            element_size = torch.tensor([], dtype=torch.float32).element_size()
            workspace = sum(
                tensor.numel()
                for tensor in (
                    predicted,
                    delta,
                    current_residual,
                    residual_norm.direction,
                    static_norm.direction,
                    full_norm.direction,
                    residual_prediction,
                    static_prediction,
                    full_prediction,
                )
            ) * element_size
            if current_output_residual is not None:
                workspace += current_output_residual.numel() * current_output_residual.element_size()
            controller._feature3_error_workspace_bytes = max(
                controller._feature3_error_workspace_bytes,
                workspace,
            )

            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 previous-error step=%s stream=%s "
                    "generic_scalar_ratio=%.6f exact_scalar_ratio=%.6f "
                    "residual_eligible=%s residual_ratio=%.6f residual_alpha=%.6e "
                    "static_eligible=%s static_ratio=%.6f static_head_ratio=%.6f static_alpha=%.6e "
                    "full_eligible=%s full_ratio=%.6f full_final_ratio=%.6f full_alpha=%.6e",
                    step.step_id,
                    stream,
                    generic_ratio_v,
                    exact_ratio_v,
                    residual_eligible,
                    residual_telemetry.ordinary_ratio,
                    residual_alpha,
                    static_eligible,
                    static_telemetry.ordinary_ratio,
                    static_telemetry.static_head_ratio,
                    static_alpha,
                    full_eligible,
                    full_telemetry.ordinary_ratio,
                    full_telemetry.final_layer_ratio,
                    full_alpha,
                )
                for kind, telemetry in (
                    ("residual", residual_telemetry),
                    ("static", static_telemetry),
                    ("full", full_telemetry),
                ):
                    LOG.warning(
                        "Spectrum H3 previous-error direction step=%s stream=%s kind=%s "
                        "eligible=%s raw_direction_norm_ratio=%.9e normalized_direction_norm_ratio=%.9e "
                        "instantaneous_alpha=%.9e alpha_used=%.9e raw_correction_norm_ratio=%.9e "
                        "bounded_correction_norm_ratio=%.9e radial_scale=%.9e bound_active=%s",
                        step.step_id,
                        stream,
                        kind,
                        telemetry.eligible,
                        telemetry.raw_direction_norm_ratio,
                        telemetry.normalized_direction_norm_ratio,
                        telemetry.instantaneous_alpha,
                        telemetry.alpha_used,
                        telemetry.raw_correction_norm_ratio,
                        telemetry.bounded_correction_norm_ratio,
                        telemetry.radial_scale,
                        telemetry.bound_active,
                    )
        controller._feature3_error_last = last
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        controller._feature3_error_failures += 1
        controller._feature3_error_disabled_reason = str(exc)
        LOG.warning(
            "Spectrum H3 previous-error telemetry disabled for this run: %s",
            exc,
        )
    finally:
        controller._feature3_error_compute_seconds += time.perf_counter() - started


def _comparison_summary(
    controller: ModelAwareController,
    stream: str,
    comparison: str,
) -> str:
    prefix = f"_feature3_error_{stream}_{comparison}"
    count = int(getattr(controller, f"{prefix}_count"))
    mean = getattr(controller, f"{prefix}_advantage_sum") / count if count else 0.0
    return (
        f"{comparison}:{getattr(controller, f'{prefix}_wins')}/"
        f"{getattr(controller, f'{prefix}_losses')}:"
        f"mean_adv={mean:.6f}:"
        f"max_abs={getattr(controller, f'{prefix}_advantage_abs_max'):.6f}"
    )


def _summary(runtime: SpectrumH3Runtime) -> str:
    controller = runtime.model_aware
    parts = [
        "feature3_previous_error_screen=residual_vs_static_adjoint_vs_local_adjoint",
        "feature3_previous_error_applied=false",
        "feature3_previous_error_units=delta_equivalent_norm",
        f"feature3_error_evidence_bytes={_evidence_bytes(controller)}",
        f"feature3_error_workspace_bytes={controller._feature3_error_workspace_bytes}",
        f"feature3_error_compute_s={controller._feature3_error_compute_seconds:.6f}",
        f"feature3_error_geometry_s={controller._feature3_error_geometry_seconds:.6f}",
        f"feature3_error_static_enqueue_s={controller._feature3_error_static_seconds:.6f}",
        f"feature3_error_full_enqueue_s={controller._feature3_error_full_seconds:.6f}",
        f"feature3_error_vjp_enqueue_s={controller._feature3_error_vjp_seconds:.6f}",
        f"feature3_error_scalar_transfer_s={controller._feature3_error_scalar_transfer_seconds:.6f}",
        f"feature3_error_failures={controller._feature3_error_failures}",
        "feature3_error_extra_transformer_nfe=0",
    ]
    for stream in ("audio", "video"):
        for kind in ("residual", "static", "full"):
            prefix = f"_feature3_error_{stream}_{kind}"
            count = int(getattr(controller, f"{prefix}_eligible_count"))
            ratio_mean = getattr(controller, f"{prefix}_ratio_sum") / count if count else 0.0
            raw_mean = getattr(controller, f"{prefix}_raw_direction_ratio_sum") / count if count else 0.0
            normalized_mean = getattr(controller, f"{prefix}_normalized_direction_ratio_sum") / count if count else 0.0
            bounded_mean = getattr(controller, f"{prefix}_bounded_ratio_sum") / count if count else 0.0
            radial_mean = getattr(controller, f"{prefix}_radial_scale_sum") / count if count else 1.0
            parts.append(
                f"feature3_error_{stream}_{kind}=eligible:{count},"
                f"fallback:{getattr(controller, f'{prefix}_fallback_count')},"
                f"ratio_mean:{ratio_mean:.6f},ratio_max:{getattr(controller, f'{prefix}_ratio_max'):.6f},"
                f"raw_dir_ratio_mean:{raw_mean:.6e},raw_dir_ratio_max:{getattr(controller, f'{prefix}_raw_direction_ratio_max'):.6e},"
                f"normalized_dir_ratio_mean:{normalized_mean:.6e},normalized_dir_ratio_max:{getattr(controller, f'{prefix}_normalized_direction_ratio_max'):.6e},"
                f"bounded_mean:{bounded_mean:.6e},bounded_max:{getattr(controller, f'{prefix}_bounded_ratio_max'):.6e},"
                f"radial_scale_min:{getattr(controller, f'{prefix}_radial_scale_min'):.6e},"
                f"radial_scale_mean:{radial_mean:.6e},bound_active:{getattr(controller, f'{prefix}_bound_active_count')}"
            )
        for comparison in (
            "residual_vs_generic",
            "static_vs_residual",
            "static_vs_generic",
            "full_vs_residual",
            "full_vs_static",
            "full_vs_generic",
        ):
            parts.append(f"feature3_error_{stream}_{_comparison_summary(controller, stream, comparison)}")
    parts.append(
        f"feature3_error_reason={controller._feature3_error_disabled_reason!r}"
    )
    return " ".join(parts)


def install_feature3_previous_error_experiment() -> None:
    """Install one telemetry-only previous-error adjoint screen."""
    if getattr(ModelAwareController, "_feature3_previous_error_installed", False):
        return

    original_reset = ModelAwareController.reset
    original_snapshot = ModelAwareController.snapshot
    original_restore = ModelAwareController.restore
    original_start = SpectrumH3Runtime.start_run
    original_end = SpectrumH3Runtime.end_run
    original_abort = SpectrumH3Runtime.abort_step
    original_disable = SpectrumH3Runtime._disable_forecasting
    original_disable_model_aware = SpectrumH3Runtime.disable_model_aware
    original_debug_summary = SpectrumH3Runtime.debug_summary
    original_execute_actual = _h3._execute_actual

    def reset(controller: ModelAwareController) -> None:
        original_reset(controller)
        _reset_error_state(controller)

    def snapshot(controller: ModelAwareController) -> dict[str, Any]:
        state: dict[str, Any] = dict(original_snapshot(controller))
        for key in _numeric_keys():
            state[key] = getattr(controller, key)
        state["_feature3_error_previous"] = dict(controller._feature3_error_previous)
        state["_feature3_error_row_indices"] = dict(controller._feature3_error_row_indices)
        state["_feature3_error_row_shapes"] = dict(controller._feature3_error_row_shapes)
        state["_feature3_error_last"] = dict(controller._feature3_error_last)
        state["_feature3_error_disabled_reason"] = controller._feature3_error_disabled_reason
        return state

    def restore(controller: ModelAwareController, state: dict[str, Any]) -> None:
        original_restore(controller, state)
        for key in _numeric_keys():
            if key in state:
                current = getattr(controller, key)
                value = state[key]
                setattr(controller, key, int(value) if isinstance(current, int) else float(value))
        controller._feature3_error_previous = dict(state.get("_feature3_error_previous", {}))
        controller._feature3_error_row_indices = dict(state.get("_feature3_error_row_indices", {}))
        controller._feature3_error_row_shapes = dict(state.get("_feature3_error_row_shapes", {}))
        controller._feature3_error_last = dict(state.get("_feature3_error_last", {}))
        controller._feature3_error_disabled_reason = state.get(
            "_feature3_error_disabled_reason"
        )

    def start(runtime: SpectrumH3Runtime, *args, **kwargs):
        runtime._feature3_error_pending_geometry = {}
        return original_start(runtime, *args, **kwargs)

    def release(runtime: SpectrumH3Runtime) -> None:
        runtime._feature3_error_pending_geometry = {}
        runtime.model_aware._feature3_error_previous = {}
        runtime.model_aware._feature3_error_row_indices = {}
        runtime.model_aware._feature3_error_row_shapes = {}

    def end(runtime: SpectrumH3Runtime, run_id: int) -> None:
        try:
            original_end(runtime, run_id)
        finally:
            release(runtime)

    def abort(runtime: SpectrumH3Runtime, run_id: int, step_id: int) -> None:
        try:
            original_abort(runtime, run_id, step_id)
        finally:
            getattr(runtime, "_feature3_error_pending_geometry", {}).pop(
                int(step_id), None
            )

    def disable(runtime: SpectrumH3Runtime, reason: str) -> bool:
        result = original_disable(runtime, reason)
        release(runtime)
        return result

    def disable_model_aware(runtime: SpectrumH3Runtime, reason: str) -> None:
        original_disable_model_aware(runtime, reason)
        release(runtime)

    def execute_actual(
        executor,
        inner: Any,
        runtime: SpectrumH3Runtime,
        run_id: int,
        step_id: int,
        call_id: int,
        layout: Any,
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload,
        kwargs,
        residual_probe=None,
    ):
        result = original_execute_actual(
            executor,
            inner,
            runtime,
            run_id,
            step_id,
            call_id,
            layout,
            x,
            timestep,
            context,
            transformer_options,
            minimax_payload,
            kwargs,
            residual_probe,
        )
        controller = runtime.model_aware
        if (
            runtime.config.model_aware_mode != "full"
            or not runtime._model_aware_enabled()
            or runtime._offline_phase == "replay"
            or controller._feature3_error_disabled_reason is not None
        ):
            return result
        started = time.perf_counter()
        try:
            state = _h3._prepare_output_state(
                inner,
                x[0],
                x[1],
                timestep,
                context,
                transformer_options,
                minimax_payload or {},
                layout,
            )
            geometry = _capture_geometry(
                inner,
                state,
                device=x[0].device,
                dtype=context.dtype,
            )
            pending = runtime._feature3_error_pending_geometry
            pending.setdefault(int(step_id), {})[int(call_id)] = geometry
        except torch.cuda.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            controller._feature3_error_failures += 1
            controller._feature3_error_disabled_reason = (
                f"current FinalLayer geometry unavailable: {exc}"
            )
            LOG.warning(
                "Spectrum H3 previous-error geometry disabled for this run: %s",
                exc,
            )
        finally:
            controller._feature3_error_geometry_seconds += time.perf_counter() - started
        return result

    def debug_summary(runtime: SpectrumH3Runtime) -> str:
        return f"{original_debug_summary(runtime)} {_summary(runtime)}"

    ModelAwareController.reset = reset
    ModelAwareController.snapshot = snapshot
    ModelAwareController.restore = restore
    SpectrumH3Runtime.start_run = start
    SpectrumH3Runtime.end_run = end
    SpectrumH3Runtime.abort_step = abort
    SpectrumH3Runtime._disable_forecasting = disable
    SpectrumH3Runtime.disable_model_aware = disable_model_aware
    SpectrumH3Runtime.debug_summary = debug_summary
    _h3._execute_actual = execute_actual
    _base.register_feature3_full_telemetry_hook(_observe_previous_error_anchor)
    ModelAwareController._feature3_previous_error_installed = True


__all__ = [
    "ErrorCandidateTelemetry",
    "PreviousErrorState",
    "install_feature3_previous_error_experiment",
    "static_output_adjoint",
]
