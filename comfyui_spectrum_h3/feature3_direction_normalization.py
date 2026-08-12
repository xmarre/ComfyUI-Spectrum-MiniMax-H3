from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from . import feature3_direction as _base
from . import minimax_h3 as _h3
from .model_aware import ModelAwareController, ModelAwareForecastDecision
from .runtime import SpectrumH3Runtime

_REFERENCE_NORM_EPS_SCALE = torch.finfo(torch.float32).eps


@dataclass(frozen=True, slots=True)
class NormalizedDirection:
    direction: torch.Tensor
    raw_direction_norm: float
    reference_delta_norm: float
    raw_direction_norm_ratio: float
    normalized_direction_norm_ratio: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class _TensorNormalizedDirection:
    direction: torch.Tensor
    raw_direction_norm: torch.Tensor
    reference_delta_norm: torch.Tensor
    raw_direction_norm_ratio: torch.Tensor
    normalized_direction_norm_ratio: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True, slots=True)
class DirectionCandidateTelemetryV2:
    eligible: bool = False
    geometry_available: bool = False
    alpha_history_count: int = 0
    direction_norm_ratio: float = 0.0
    raw_direction_norm: float = 0.0
    reference_delta_norm: float = 0.0
    raw_direction_norm_ratio: float = 0.0
    normalized_direction_norm_ratio: float = 0.0
    coefficient_used: float = 0.0
    instantaneous_coefficient: float = 0.0
    raw_correction_norm_ratio: float = 0.0
    radial_scale: float = 1.0
    bounded_norm_ratio: float = 0.0
    bound_active: bool = False
    ordinary_ratio: float = 0.0
    static_head_ratio: float = 0.0
    final_layer_ratio: float = 0.0
    generic_scalar_ratio: float = 0.0
    exact_scalar_ratio: float = 0.0
    scalar_applied_ratio: float = 0.0
    relative_advantage_vs_generic: float = 0.0
    relative_advantage_vs_exact: float = 0.0
    relative_advantage_vs_static: float = 0.0


def _reference_norm_epsilon(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        _REFERENCE_NORM_EPS_SCALE * math.sqrt(max(1, value.numel())),
        dtype=torch.float32,
        device=value.device,
    )


def _normalize_direction_tensor(
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
) -> _TensorNormalizedDirection:
    """Normalize a finite nonzero direction to the reference-delta norm."""
    if direction.shape != reference_delta.shape:
        raise ValueError("direction and reference delta must have identical shape")
    m = direction.to(torch.float32)
    d = reference_delta.to(torch.float32)
    zero = torch.zeros((), dtype=torch.float32, device=m.device)

    m_norm = torch.linalg.vector_norm(m)
    d_norm = torch.linalg.vector_norm(d)
    reference_eps = _reference_norm_epsilon(d)
    finite = (
        torch.isfinite(m).all()
        & torch.isfinite(d).all()
        & torch.isfinite(m_norm)
        & torch.isfinite(d_norm)
    )
    reference_valid = finite & (d_norm > reference_eps)
    raw_nonzero = finite & (m_norm > 0.0)
    base_eligible = reference_valid & raw_nonzero

    safe_d_norm = torch.where(reference_valid, d_norm, torch.ones_like(d_norm))
    safe_m_norm = torch.where(raw_nonzero, m_norm, torch.ones_like(m_norm))
    scale = safe_d_norm / safe_m_norm
    scale_finite = torch.isfinite(scale)
    normalized = m * torch.where(base_eligible & scale_finite, scale, zero)
    normalized_norm = torch.linalg.vector_norm(normalized)
    normalized_finite = torch.isfinite(normalized).all() & torch.isfinite(normalized_norm)
    eligible = base_eligible & scale_finite & normalized_finite & (normalized_norm > 0.0)

    normalized = torch.where(eligible, normalized, torch.zeros_like(normalized))
    raw_ratio = torch.where(base_eligible, m_norm / safe_d_norm, zero)
    normalized_ratio = torch.where(eligible, normalized_norm / safe_d_norm, zero)
    return _TensorNormalizedDirection(
        direction=normalized,
        raw_direction_norm=torch.where(finite, m_norm, zero),
        reference_delta_norm=torch.where(finite, d_norm, zero),
        raw_direction_norm_ratio=torch.where(torch.isfinite(raw_ratio), raw_ratio, zero),
        normalized_direction_norm_ratio=torch.where(
            torch.isfinite(normalized_ratio), normalized_ratio, zero
        ),
        eligible=eligible,
    )


def normalize_direction_to_reference(
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
) -> NormalizedDirection:
    normalized = _normalize_direction_tensor(direction, reference_delta)
    values = torch.stack(
        (
            normalized.raw_direction_norm,
            normalized.reference_delta_norm,
            normalized.raw_direction_norm_ratio,
            normalized.normalized_direction_norm_ratio,
            normalized.eligible.to(torch.float32),
        )
    ).detach().to(device="cpu").tolist()
    raw_norm, reference_norm, raw_ratio, normalized_ratio, eligible = values
    return NormalizedDirection(
        direction=normalized.direction,
        raw_direction_norm=float(raw_norm),
        reference_delta_norm=float(reference_norm),
        raw_direction_norm_ratio=float(raw_ratio),
        normalized_direction_norm_ratio=float(normalized_ratio),
        eligible=bool(eligible),
    )


def _radially_bound_direction_tensor(
    coefficient: float,
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    limit: float = _base._CORRECTION_NORM_LIMIT,
) -> _base._TensorBoundedDirection:
    """Stable device-local rational radial bound with the exact q->0 limit."""
    if direction.shape != reference_delta.shape:
        raise ValueError("direction and reference delta must have identical shape")
    d = reference_delta.to(torch.float32)
    m = direction.to(torch.float32)
    zero = torch.zeros((), dtype=torch.float32, device=m.device)
    one = torch.ones_like(zero)
    false = torch.zeros((), dtype=torch.bool, device=m.device)
    if (
        not math.isfinite(float(coefficient))
        or not math.isfinite(float(limit))
        or float(limit) <= 0.0
    ):
        return _base._TensorBoundedDirection(
            correction=torch.zeros_like(m),
            direction_norm_ratio=zero,
            raw_norm_ratio=zero,
            radial_scale=one,
            bounded_norm_ratio=zero,
            bound_active=false,
            eligible=false,
        )

    d_norm = torch.linalg.vector_norm(d)
    m_norm = torch.linalg.vector_norm(m)
    reference_eps = _reference_norm_epsilon(d)
    finite = (
        torch.isfinite(d).all()
        & torch.isfinite(m).all()
        & torch.isfinite(d_norm)
        & torch.isfinite(m_norm)
    )
    eligible = finite & (d_norm > reference_eps) & (m_norm > 0.0)
    safe_d_norm = torch.where(eligible, d_norm, torch.ones_like(d_norm))
    direction_ratio = torch.where(eligible, m_norm / safe_d_norm, zero)
    raw_ratio = abs(float(coefficient)) * direction_ratio
    raw_finite = torch.isfinite(raw_ratio)
    eligible = eligible & raw_finite
    raw_ratio = torch.where(eligible, raw_ratio, zero)

    # q_bounded = q / (1 + q/L), therefore q_bounded/q is exactly the
    # following expression for every positive finite q. This avoids dividing
    # by a clamped q, which spuriously suppressed 0 < q < eps.
    radial_scale = 1.0 / (1.0 + raw_ratio / float(limit))
    radial_scale = torch.where(eligible, radial_scale, one)
    bounded_ratio = torch.where(eligible, raw_ratio * radial_scale, zero)
    correction = torch.where(
        eligible,
        float(coefficient) * radial_scale * m,
        torch.zeros_like(m),
    )
    correction_finite = torch.isfinite(correction).all()
    eligible = eligible & correction_finite
    correction = torch.where(eligible, correction, torch.zeros_like(m))
    radial_scale = torch.where(eligible, radial_scale, one)
    bounded_ratio = torch.where(eligible, bounded_ratio, zero)
    direction_ratio = torch.where(eligible, direction_ratio, zero)
    raw_ratio = torch.where(eligible, raw_ratio, zero)
    bound_active = eligible & (radial_scale < 1.0 - 1e-7)
    return _base._TensorBoundedDirection(
        correction=correction,
        direction_norm_ratio=direction_ratio,
        raw_norm_ratio=raw_ratio,
        radial_scale=radial_scale,
        bounded_norm_ratio=bounded_ratio,
        bound_active=bound_active,
        eligible=eligible,
    )


def _instantaneous_alpha(
    residual: torch.Tensor,
    direction: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    flat_direction = direction.reshape(-1)
    flat_residual = residual.reshape(-1)
    denom = torch.dot(flat_direction, flat_direction)
    numerator = torch.dot(flat_residual, flat_direction)
    tiny = torch.as_tensor(torch.finfo(torch.float32).tiny, device=denom.device)
    value = numerator / denom.clamp_min(tiny)
    valid = eligible & (denom > 0.0) & torch.isfinite(value)
    return torch.where(valid, value, torch.zeros_like(value))


def _empty_normalization(reference_delta: torch.Tensor) -> _TensorNormalizedDirection:
    d = reference_delta.to(torch.float32)
    zero = torch.zeros((), dtype=torch.float32, device=d.device)
    return _TensorNormalizedDirection(
        direction=torch.zeros_like(d),
        raw_direction_norm=zero,
        reference_delta_norm=torch.where(
            torch.isfinite(d).all(), torch.linalg.vector_norm(d), zero
        ),
        raw_direction_norm_ratio=zero,
        normalized_direction_norm_ratio=zero,
        eligible=torch.zeros((), dtype=torch.bool, device=d.device),
    )


def _evaluate_model_directions(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    raw_weights_by_stream: dict[str, torch.Tensor],
    exact_head_weights: dict[str, torch.Tensor],
    geometry: _base.FinalLayerGeometry | None,
) -> dict[str, tuple[DirectionCandidateTelemetryV2, DirectionCandidateTelemetryV2]]:
    direction_started = time.perf_counter()
    controller = runtime.model_aware
    history = controller._feature3_row_history
    if len(history) != runtime.forecaster.history_length or len(history) < 2:
        return {}
    generic_audio, generic_video = controller.generic_correction_gains(decision)
    result: dict[
        str, tuple[DirectionCandidateTelemetryV2, DirectionCandidateTelemetryV2]
    ] = {}

    for name, start, end in runtime._stream_ranges(step.calls[0]):
        if name not in {"audio", "video"}:
            continue
        head = exact_head_weights.get(name)
        weights = raw_weights_by_stream.get(name)
        indices = controller._feature3_row_indices.get(name)
        if head is None or weights is None or indices is None:
            continue

        history_rows = [entry[name] for entry in history]
        actual_native = combined[:, start:end].index_select(1, indices)
        actual_rows = actual_native.to(torch.float32)
        predicted = torch.zeros_like(actual_rows)
        for weight, sample in zip(weights.tolist(), history_rows, strict=True):
            if weight != 0.0:
                predicted.add_(sample.to(torch.float32), alpha=float(weight))
        predicted_native = predicted.to(dtype=actual_native.dtype)
        latest_native = history_rows[-1]
        previous_native = history_rows[-2]
        latest = latest_native.to(torch.float32)
        previous = previous_native.to(torch.float32)
        delta = latest - previous
        residual = actual_rows - predicted
        hold_rms = _base._tensor_rms(actual_rows - latest).clamp_min(
            _base._DIRECTION_EPS
        )

        if name == "audio":
            generic_gain = generic_audio
            exact_gain = decision.audio_correction_telemetry.model_candidate_gain
            applied_gain = decision.audio_correction_gain
            adaln_scale = None if geometry is None else geometry.audio_scale
        else:
            generic_gain = generic_video
            exact_gain = decision.video_correction_telemetry.model_candidate_gain
            applied_gain = decision.video_correction_gain
            adaln_scale = None if geometry is None else geometry.video_scale

        generic_ratio = _base._tensor_rms(
            actual_rows - (predicted + generic_gain * delta)
        ) / hold_rms
        exact_ratio = _base._tensor_rms(
            actual_rows - (predicted + exact_gain * delta)
        ) / hold_rms
        applied_ratio = _base._tensor_rms(
            actual_rows - (predicted + applied_gain * delta)
        ) / hold_rms

        static_started = time.perf_counter()
        static_raw = _base.static_head_metric_direction(delta, head)
        controller._feature3_static_seconds += time.perf_counter() - static_started
        static_normalized = _normalize_direction_tensor(static_raw, delta)
        static_used, static_count = _base._direction_alpha_used(
            controller, name, "static", decision.confidence
        )
        static_bound = _radially_bound_direction_tensor(
            static_used, static_normalized.direction, delta
        )
        static_prediction = predicted + static_bound.correction
        static_ratio = _base._tensor_rms(actual_rows - static_prediction) / hold_rms
        static_head_hold = _base._tensor_rms(
            _base._static_head_difference(actual_rows, latest, head)
        ).clamp_min(_base._DIRECTION_EPS)
        static_head_ratio = _base._tensor_rms(
            _base._static_head_difference(actual_rows, static_prediction, head)
        ) / static_head_hold
        static_instant = _instantaneous_alpha(
            residual, static_normalized.direction, static_normalized.eligible
        )

        full_raw = None
        full_jvp = full_vjp = 0.0
        if geometry is not None and adaln_scale is not None:
            full_started = time.perf_counter()
            full_raw, full_jvp, full_vjp = _base.final_layer_metric_direction(
                predicted_native,
                delta.to(dtype=predicted_native.dtype),
                norm_weight=geometry.norm_weight,
                norm_eps=geometry.norm_eps,
                adaln_scale=adaln_scale,
                head_weight=head,
            )
            full_raw = full_raw.to(torch.float32)
            controller._feature3_full_seconds += time.perf_counter() - full_started
            controller._feature3_jvp_seconds += full_jvp
            controller._feature3_vjp_seconds += full_vjp

        full_used, full_count = _base._direction_alpha_used(
            controller, name, "full", decision.confidence
        )
        if full_raw is None:
            full_normalized = _empty_normalization(delta)
            full_bound = _base._TensorBoundedDirection(
                correction=torch.zeros_like(delta),
                direction_norm_ratio=torch.zeros(
                    (), dtype=torch.float32, device=delta.device
                ),
                raw_norm_ratio=torch.zeros((), dtype=torch.float32, device=delta.device),
                radial_scale=torch.ones((), dtype=torch.float32, device=delta.device),
                bounded_norm_ratio=torch.zeros(
                    (), dtype=torch.float32, device=delta.device
                ),
                bound_active=torch.zeros((), dtype=torch.bool, device=delta.device),
                eligible=torch.zeros((), dtype=torch.bool, device=delta.device),
            )
            full_prediction = predicted
            full_ratio = torch.zeros((), dtype=torch.float32, device=combined.device)
            full_static_head_ratio = torch.zeros_like(full_ratio)
            full_final_ratio = torch.zeros_like(full_ratio)
            full_instant = torch.zeros_like(full_ratio)
        else:
            full_normalized = _normalize_direction_tensor(full_raw, delta)
            full_bound = _radially_bound_direction_tensor(
                full_used, full_normalized.direction, delta
            )
            full_prediction = predicted + full_bound.correction
            full_ratio = _base._tensor_rms(actual_rows - full_prediction) / hold_rms
            full_static_head_ratio = _base._tensor_rms(
                _base._static_head_difference(actual_rows, full_prediction, head)
            ) / static_head_hold
            assert geometry is not None and adaln_scale is not None
            final_hold = _base._tensor_rms(
                _base._final_layer_difference(
                    actual_native,
                    latest_native,
                    geometry=geometry,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                )
            ).clamp_min(_base._DIRECTION_EPS)
            full_final_ratio = _base._tensor_rms(
                _base._final_layer_difference(
                    actual_native,
                    full_prediction.to(dtype=actual_native.dtype),
                    geometry=geometry,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                )
            ) / final_hold
            full_instant = _instantaneous_alpha(
                residual, full_normalized.direction, full_normalized.eligible
            )

        transfer_started = time.perf_counter()
        scalar_values = torch.stack(
            (
                generic_ratio,
                exact_ratio,
                applied_ratio,
                static_ratio,
                static_head_ratio,
                static_instant,
                full_ratio,
                full_static_head_ratio,
                full_final_ratio,
                full_instant,
                static_normalized.raw_direction_norm,
                static_normalized.reference_delta_norm,
                static_normalized.raw_direction_norm_ratio,
                static_normalized.normalized_direction_norm_ratio,
                static_bound.raw_norm_ratio,
                static_bound.radial_scale,
                static_bound.bounded_norm_ratio,
                static_bound.bound_active.to(torch.float32),
                static_bound.eligible.to(torch.float32),
                full_normalized.raw_direction_norm,
                full_normalized.reference_delta_norm,
                full_normalized.raw_direction_norm_ratio,
                full_normalized.normalized_direction_norm_ratio,
                full_bound.raw_norm_ratio,
                full_bound.radial_scale,
                full_bound.bounded_norm_ratio,
                full_bound.bound_active.to(torch.float32),
                full_bound.eligible.to(torch.float32),
            )
        ).detach().to(device="cpu").tolist()
        controller._feature3_scalar_transfer_seconds += (
            time.perf_counter() - transfer_started
        )
        (
            generic_ratio_v,
            exact_ratio_v,
            applied_ratio_v,
            static_ratio_v,
            static_head_ratio_v,
            static_instant_v,
            full_ratio_v,
            full_static_head_ratio_v,
            full_final_ratio_v,
            full_instant_v,
            static_raw_norm_v,
            static_reference_norm_v,
            static_raw_direction_ratio_v,
            static_normalized_direction_ratio_v,
            static_raw_correction_ratio_v,
            static_radial_scale_v,
            static_bounded_ratio_v,
            static_bound_active_v,
            static_bound_eligible_v,
            full_raw_norm_v,
            full_reference_norm_v,
            full_raw_direction_ratio_v,
            full_normalized_direction_ratio_v,
            full_raw_correction_ratio_v,
            full_radial_scale_v,
            full_bounded_ratio_v,
            full_bound_active_v,
            full_bound_eligible_v,
        ) = scalar_values

        static_eligible = bool(
            static_bound_eligible_v
            and static_count > 0
            and math.isfinite(static_ratio_v)
        )
        full_eligible = bool(
            full_bound_eligible_v
            and full_count > 0
            and geometry is not None
            and math.isfinite(full_ratio_v)
        )
        static_telemetry = DirectionCandidateTelemetryV2(
            eligible=static_eligible,
            geometry_available=True,
            alpha_history_count=static_count,
            direction_norm_ratio=static_raw_direction_ratio_v,
            raw_direction_norm=static_raw_norm_v,
            reference_delta_norm=static_reference_norm_v,
            raw_direction_norm_ratio=static_raw_direction_ratio_v,
            normalized_direction_norm_ratio=static_normalized_direction_ratio_v,
            coefficient_used=static_used,
            instantaneous_coefficient=static_instant_v,
            raw_correction_norm_ratio=static_raw_correction_ratio_v,
            radial_scale=static_radial_scale_v,
            bounded_norm_ratio=static_bounded_ratio_v,
            bound_active=bool(static_bound_active_v),
            ordinary_ratio=static_ratio_v if static_eligible else 0.0,
            static_head_ratio=static_head_ratio_v if static_eligible else 0.0,
            generic_scalar_ratio=generic_ratio_v,
            exact_scalar_ratio=exact_ratio_v,
            scalar_applied_ratio=applied_ratio_v,
        )
        full_telemetry = DirectionCandidateTelemetryV2(
            eligible=full_eligible,
            geometry_available=geometry is not None,
            alpha_history_count=full_count,
            direction_norm_ratio=full_raw_direction_ratio_v,
            raw_direction_norm=full_raw_norm_v,
            reference_delta_norm=full_reference_norm_v,
            raw_direction_norm_ratio=full_raw_direction_ratio_v,
            normalized_direction_norm_ratio=full_normalized_direction_ratio_v,
            coefficient_used=full_used,
            instantaneous_coefficient=full_instant_v,
            raw_correction_norm_ratio=full_raw_correction_ratio_v,
            radial_scale=full_radial_scale_v,
            bounded_norm_ratio=full_bounded_ratio_v,
            bound_active=bool(full_bound_active_v),
            ordinary_ratio=full_ratio_v if full_eligible else 0.0,
            static_head_ratio=full_static_head_ratio_v if full_eligible else 0.0,
            final_layer_ratio=full_final_ratio_v if full_eligible else 0.0,
            generic_scalar_ratio=generic_ratio_v,
            exact_scalar_ratio=exact_ratio_v,
            scalar_applied_ratio=applied_ratio_v,
        )

        if static_eligible:
            adv_g = _base._record_comparison(
                controller, name, "static_vs_generic", generic_ratio_v, static_ratio_v
            )
            adv_e = _base._record_comparison(
                controller, name, "static_vs_exact", exact_ratio_v, static_ratio_v
            )
            static_telemetry = replace(
                static_telemetry,
                relative_advantage_vs_generic=adv_g,
                relative_advantage_vs_exact=adv_e,
            )
        if full_eligible:
            adv_g = _base._record_comparison(
                controller, name, "full_vs_generic", generic_ratio_v, full_ratio_v
            )
            adv_e = _base._record_comparison(
                controller, name, "full_vs_exact", exact_ratio_v, full_ratio_v
            )
            adv_s = 0.0
            if static_eligible:
                adv_s = _base._record_comparison(
                    controller, name, "full_vs_static", static_ratio_v, full_ratio_v
                )
            full_telemetry = replace(
                full_telemetry,
                relative_advantage_vs_generic=adv_g,
                relative_advantage_vs_exact=adv_e,
                relative_advantage_vs_static=adv_s,
            )

        _base._candidate_stats(controller, name, "static", static_telemetry)
        _base._candidate_stats(controller, name, "full", full_telemetry)
        if bool(static_bound_eligible_v):
            _base._update_direction_alpha(controller, name, "static", static_instant_v)
        if full_raw is not None and bool(full_bound_eligible_v):
            _base._update_direction_alpha(controller, name, "full", full_instant_v)
        result[name] = (static_telemetry, full_telemetry)

        element_size = torch.tensor([], dtype=torch.float32).element_size()
        workspace = (
            predicted.numel()
            + delta.numel()
            + residual.numel()
            + static_raw.numel()
            + static_normalized.direction.numel()
            + static_prediction.numel()
        ) * element_size
        if full_raw is not None:
            workspace += (
                full_raw.numel()
                + full_normalized.direction.numel()
                + full_prediction.numel()
            ) * element_size
        controller._feature3_workspace_bytes = max(
            controller._feature3_workspace_bytes, workspace
        )

    controller._feature3_direction_seconds += time.perf_counter() - direction_started
    return result


def _new_stat_suffixes() -> tuple[str, ...]:
    return (
        "raw_direction_ratio_sum",
        "raw_direction_ratio_min",
        "raw_direction_ratio_max",
        "normalized_direction_ratio_sum",
        "normalized_direction_ratio_min",
        "normalized_direction_ratio_max",
        "instant_alpha_sum",
        "instant_alpha_min",
        "instant_alpha_max",
        "used_alpha_sum",
        "used_alpha_abs_max",
        "raw_correction_ratio_sum",
        "raw_correction_ratio_max",
        "radial_scale_sum",
        "radial_scale_min",
    )


def _reset_new_stats(controller: ModelAwareController) -> None:
    for stream in ("audio", "video"):
        for kind in ("static", "full"):
            prefix = f"_feature3_{stream}_{kind}"
            setattr(controller, f"{prefix}_raw_direction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_raw_direction_ratio_min", math.inf)
            setattr(controller, f"{prefix}_raw_direction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_normalized_direction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_normalized_direction_ratio_min", math.inf)
            setattr(controller, f"{prefix}_normalized_direction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_instant_alpha_sum", 0.0)
            setattr(controller, f"{prefix}_instant_alpha_min", math.inf)
            setattr(controller, f"{prefix}_instant_alpha_max", -math.inf)
            setattr(controller, f"{prefix}_used_alpha_sum", 0.0)
            setattr(controller, f"{prefix}_used_alpha_abs_max", 0.0)
            setattr(controller, f"{prefix}_raw_correction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_raw_correction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_radial_scale_sum", 0.0)
            setattr(controller, f"{prefix}_radial_scale_min", 1.0)


def _record_new_candidate_stats(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    telemetry: DirectionCandidateTelemetryV2,
) -> None:
    if not telemetry.eligible:
        return
    prefix = f"_feature3_{stream}_{kind}"
    raw = telemetry.raw_direction_norm_ratio
    normalized = telemetry.normalized_direction_norm_ratio
    instant = telemetry.instantaneous_coefficient
    used = telemetry.coefficient_used
    raw_correction = telemetry.raw_correction_norm_ratio
    radial = telemetry.radial_scale
    setattr(
        controller,
        f"{prefix}_raw_direction_ratio_sum",
        getattr(controller, f"{prefix}_raw_direction_ratio_sum") + raw,
    )
    setattr(
        controller,
        f"{prefix}_raw_direction_ratio_min",
        min(getattr(controller, f"{prefix}_raw_direction_ratio_min"), raw),
    )
    setattr(
        controller,
        f"{prefix}_raw_direction_ratio_max",
        max(getattr(controller, f"{prefix}_raw_direction_ratio_max"), raw),
    )
    setattr(
        controller,
        f"{prefix}_normalized_direction_ratio_sum",
        getattr(controller, f"{prefix}_normalized_direction_ratio_sum") + normalized,
    )
    setattr(
        controller,
        f"{prefix}_normalized_direction_ratio_min",
        min(getattr(controller, f"{prefix}_normalized_direction_ratio_min"), normalized),
    )
    setattr(
        controller,
        f"{prefix}_normalized_direction_ratio_max",
        max(getattr(controller, f"{prefix}_normalized_direction_ratio_max"), normalized),
    )
    setattr(
        controller,
        f"{prefix}_instant_alpha_sum",
        getattr(controller, f"{prefix}_instant_alpha_sum") + instant,
    )
    setattr(
        controller,
        f"{prefix}_instant_alpha_min",
        min(getattr(controller, f"{prefix}_instant_alpha_min"), instant),
    )
    setattr(
        controller,
        f"{prefix}_instant_alpha_max",
        max(getattr(controller, f"{prefix}_instant_alpha_max"), instant),
    )
    setattr(
        controller,
        f"{prefix}_used_alpha_sum",
        getattr(controller, f"{prefix}_used_alpha_sum") + used,
    )
    setattr(
        controller,
        f"{prefix}_used_alpha_abs_max",
        max(getattr(controller, f"{prefix}_used_alpha_abs_max"), abs(used)),
    )
    setattr(
        controller,
        f"{prefix}_raw_correction_ratio_sum",
        getattr(controller, f"{prefix}_raw_correction_ratio_sum") + raw_correction,
    )
    setattr(
        controller,
        f"{prefix}_raw_correction_ratio_max",
        max(
            getattr(controller, f"{prefix}_raw_correction_ratio_max"),
            raw_correction,
        ),
    )
    setattr(
        controller,
        f"{prefix}_radial_scale_sum",
        getattr(controller, f"{prefix}_radial_scale_sum") + radial,
    )
    setattr(
        controller,
        f"{prefix}_radial_scale_min",
        min(getattr(controller, f"{prefix}_radial_scale_min"), radial),
    )


def _summary_min(value: float, count: int) -> float:
    return 0.0 if count == 0 or not math.isfinite(value) else value


def _feature3_summary(runtime: SpectrumH3Runtime) -> str:
    c = runtime.model_aware
    parts = [
        "feature3_applied_correction=scalar_latest_delta",
        "feature3_direction_screen=static_head_and_full_final_layer_normalized",
        "feature3_direction_units=delta_equivalent_norm",
        "feature3_k2_runtime=retired",
        f"feature3_direction_evidence_bytes={_base._feature3_evidence_bytes(c)}",
        f"feature3_direction_workspace_bytes={c._feature3_workspace_bytes}",
        f"feature3_geometry_s={c._feature3_geometry_seconds:.6f}",
        f"feature3_direction_compute_s={c._feature3_direction_seconds:.6f}",
        f"feature3_static_enqueue_s={c._feature3_static_seconds:.6f}",
        f"feature3_full_enqueue_s={c._feature3_full_seconds:.6f}",
        f"feature3_jvp_enqueue_s={c._feature3_jvp_seconds:.6f}",
        f"feature3_vjp_enqueue_s={c._feature3_vjp_seconds:.6f}",
        f"feature3_scalar_transfer_s={c._feature3_scalar_transfer_seconds:.6f}",
        f"feature3_geometry_failures={c._feature3_geometry_failures}",
        "feature3_extra_transformer_nfe=0",
    ]
    for stream in ("audio", "video"):
        for kind in ("static", "full"):
            prefix = f"_feature3_{stream}_{kind}"
            count = int(getattr(c, f"{prefix}_eligible_count"))
            ratio_mean = getattr(c, f"{prefix}_ratio_sum") / count if count else 0.0
            raw_mean = (
                getattr(c, f"{prefix}_raw_direction_ratio_sum") / count
                if count
                else 0.0
            )
            normalized_mean = (
                getattr(c, f"{prefix}_normalized_direction_ratio_sum") / count
                if count
                else 0.0
            )
            instant_mean = (
                getattr(c, f"{prefix}_instant_alpha_sum") / count if count else 0.0
            )
            used_mean = (
                getattr(c, f"{prefix}_used_alpha_sum") / count if count else 0.0
            )
            raw_correction_mean = (
                getattr(c, f"{prefix}_raw_correction_ratio_sum") / count
                if count
                else 0.0
            )
            bounded_mean = (
                getattr(c, f"{prefix}_bounded_ratio_sum") / count if count else 0.0
            )
            radial_mean = (
                getattr(c, f"{prefix}_radial_scale_sum") / count if count else 1.0
            )
            parts.append(
                f"{stream}_{kind}=eligible:{count},"
                f"fallback:{getattr(c, f'{prefix}_fallback_count')},"
                f"ratio_mean:{ratio_mean:.6f},"
                f"ratio_max:{getattr(c, f'{prefix}_ratio_max'):.6f},"
                f"raw_dir_ratio_min:{_summary_min(getattr(c, f'{prefix}_raw_direction_ratio_min'), count):.6e},"
                f"raw_dir_ratio_mean:{raw_mean:.6e},"
                f"raw_dir_ratio_max:{getattr(c, f'{prefix}_raw_direction_ratio_max'):.6e},"
                f"normalized_dir_ratio_min:{_summary_min(getattr(c, f'{prefix}_normalized_direction_ratio_min'), count):.6e},"
                f"normalized_dir_ratio_mean:{normalized_mean:.6e},"
                f"normalized_dir_ratio_max:{getattr(c, f'{prefix}_normalized_direction_ratio_max'):.6e},"
                f"instant_alpha_min:{_summary_min(getattr(c, f'{prefix}_instant_alpha_min'), count):.6e},"
                f"instant_alpha_mean:{instant_mean:.6e},"
                f"instant_alpha_max:{_summary_min(getattr(c, f'{prefix}_instant_alpha_max'), count):.6e},"
                f"used_alpha_mean:{used_mean:.6e},"
                f"used_alpha_abs_max:{getattr(c, f'{prefix}_used_alpha_abs_max'):.6e},"
                f"raw_correction_mean:{raw_correction_mean:.6e},"
                f"raw_correction_max:{getattr(c, f'{prefix}_raw_correction_ratio_max'):.6e},"
                f"bounded_mean:{bounded_mean:.6e},"
                f"bounded_max:{getattr(c, f'{prefix}_bounded_ratio_max'):.6e},"
                f"radial_scale_min:{getattr(c, f'{prefix}_radial_scale_min'):.6e},"
                f"radial_scale_mean:{radial_mean:.6e},"
                f"bound_active:{getattr(c, f'{prefix}_bound_active_count')}"
            )
        for comparison in (
            "static_vs_generic",
            "static_vs_exact",
            "full_vs_generic",
            "full_vs_exact",
            "full_vs_static",
        ):
            parts.append(f"{stream}_{_base._summary_comparison(c, stream, comparison)}")
    return " ".join(parts)


def _precise_direction_log(runtime: SpectrumH3Runtime, step: Any) -> None:
    if runtime.config.model_aware_mode != "full" or not runtime.config.debug:
        return
    for name, pair in runtime.model_aware._feature3_last.items():
        static, full = pair
        for kind, telemetry in (("static", static), ("full", full)):
            _h3.LOG.warning(
                "Spectrum H3 normalized model-direction step=%s stream=%s kind=%s "
                "eligible=%s raw_direction_norm=%.9e reference_delta_norm=%.9e "
                "raw_direction_norm_ratio=%.9e normalized_direction_norm_ratio=%.9e "
                "instantaneous_alpha_raw=%.9e alpha_used=%.9e "
                "raw_correction_norm_ratio=%.9e bounded_correction_norm_ratio=%.9e "
                "radial_scale=%.9e bound_active=%s",
                step.step_id,
                name,
                kind,
                telemetry.eligible,
                telemetry.raw_direction_norm,
                telemetry.reference_delta_norm,
                telemetry.raw_direction_norm_ratio,
                telemetry.normalized_direction_norm_ratio,
                telemetry.instantaneous_coefficient,
                telemetry.coefficient_used,
                telemetry.raw_correction_norm_ratio,
                telemetry.bounded_norm_ratio,
                telemetry.radial_scale,
                telemetry.bound_active,
            )


def install_feature3_direction_normalization() -> None:
    """Install the scale-invariant Feature-3 direction screen once."""
    if getattr(_base, "_feature3_direction_normalization_installed", False):
        return
    if not getattr(ModelAwareController, "_feature3_direction_installed", False):
        raise RuntimeError("Feature-3 base experiment must be installed first")

    old_reset = _base._controller_feature3_reset
    old_numeric_keys = _base._feature3_numeric_keys
    old_candidate_stats = _base._candidate_stats
    old_observe = SpectrumH3Runtime._observe_model_aware_anchor
    old_debug_summary = SpectrumH3Runtime.debug_summary

    def reset(controller: ModelAwareController) -> None:
        old_reset(controller)
        _reset_new_stats(controller)

    def numeric_keys(controller: ModelAwareController) -> list[str]:
        keys = list(old_numeric_keys(controller))
        for stream in ("audio", "video"):
            for kind in ("static", "full"):
                prefix = f"_feature3_{stream}_{kind}"
                keys.extend(f"{prefix}_{suffix}" for suffix in _new_stat_suffixes())
        return keys

    def candidate_stats(
        controller: ModelAwareController,
        stream: str,
        kind: str,
        telemetry: DirectionCandidateTelemetryV2,
    ) -> None:
        old_candidate_stats(controller, stream, kind, telemetry)
        _record_new_candidate_stats(controller, stream, kind, telemetry)

    def observe(
        self: SpectrumH3Runtime,
        step: Any,
        combined: torch.Tensor,
        exact_head_weights: dict[str, torch.Tensor],
        stream_diagonals: dict[str, torch.Tensor],
    ) -> None:
        old_observe(self, step, combined, exact_head_weights, stream_diagonals)
        _precise_direction_log(self, step)

    def debug_summary(self: SpectrumH3Runtime) -> str:
        summary = old_debug_summary(self)
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
        return summary

    _base._controller_feature3_reset = reset
    _base._feature3_numeric_keys = numeric_keys
    _base._candidate_stats = candidate_stats
    _base._radially_bound_direction_tensor = _radially_bound_direction_tensor
    _base._evaluate_model_directions = _evaluate_model_directions
    _base._feature3_summary = _feature3_summary
    SpectrumH3Runtime._observe_model_aware_anchor = observe
    SpectrumH3Runtime.debug_summary = debug_summary

    # Controllers created after installation are initialized by the patched reset.
    _base._feature3_direction_normalization_installed = True


__all__ = [
    "DirectionCandidateTelemetryV2",
    "NormalizedDirection",
    "install_feature3_direction_normalization",
    "normalize_direction_to_reference",
]
