from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .forecast import HistoryWeightForecaster
from .model_aware import (
    AnchorEvidence,
    AnchorEvidenceTiming,
    ModelAwareForecastDecision,
    StreamAnchorEvidence,
)
from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)

_CORRECTION_NORM_LIMIT = 0.25
_DIRECTION_ALPHA_LIMIT = 2.0
_DIRECTION_EPS = torch.finfo(torch.float32).eps


@dataclass(frozen=True, slots=True)
class FinalLayerGeometry:
    """Detached native FinalLayer geometry for one target timestep."""

    norm_weight: torch.Tensor
    norm_eps: float
    audio_scale: torch.Tensor
    video_scale: torch.Tensor

    @property
    def tensor_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.norm_weight, self.audio_scale, self.video_scale)
        )


@dataclass(frozen=True, slots=True)
class BoundedDirection:
    correction: torch.Tensor
    direction_norm_ratio: float
    raw_norm_ratio: float
    radial_scale: float
    bounded_norm_ratio: float
    bound_active: bool
    eligible: bool


@dataclass(frozen=True, slots=True)
class _TensorBoundedDirection:
    correction: torch.Tensor
    direction_norm_ratio: torch.Tensor
    raw_norm_ratio: torch.Tensor
    radial_scale: torch.Tensor
    bounded_norm_ratio: torch.Tensor
    bound_active: torch.Tensor
    eligible: torch.Tensor


def _soft_limit(value: float, limit: float) -> float:
    value = float(value)
    limit = float(limit)
    if not math.isfinite(value) or not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("soft limit requires a finite value and positive finite limit")
    return value / (1.0 + abs(value) / limit)


def static_head_metric_direction(
    delta: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Historical helper: return W^T W d without materializing W^T W."""
    if delta.ndim < 1 or head_weight.ndim != 2:
        raise ValueError(
            "static FinalLayer direction requires [..., hidden] delta and [out, hidden] head"
        )
    if int(delta.shape[-1]) != int(head_weight.shape[1]):
        raise ValueError("head hidden width does not match delta")
    operator = head_weight.to(device=delta.device, dtype=torch.float32)
    projected = torch.matmul(delta.to(torch.float32), operator.transpose(0, 1))
    return torch.matmul(projected, operator)


def _rmsnorm_weight(
    x: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    if weight is None:
        return torch.ones(int(x.shape[-1]), dtype=x.dtype, device=x.device)
    if weight.ndim != 1 or int(weight.shape[0]) != int(x.shape[-1]):
        raise ValueError("RMSNorm weight does not match hidden width")
    return weight.to(device=x.device, dtype=x.dtype)


def rmsnorm_jvp(
    x: torch.Tensor,
    tangent: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Analytic JVP of the native H3 RMSNorm."""
    if x.shape != tangent.shape or x.ndim < 1:
        raise ValueError("RMSNorm JVP requires equal [..., hidden] tensors")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    tangent = tangent.to(device=x.device, dtype=x.dtype)
    w = _rmsnorm_weight(x, weight)
    mean_square = x.square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + float(eps))
    mean_xt = (x * tangent).mean(dim=-1, keepdim=True)
    return (inv_rms * tangent - x * inv_rms.pow(3) * mean_xt) * w


def rmsnorm_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Analytic VJP of the native H3 RMSNorm."""
    if x.shape != cotangent.shape or x.ndim < 1:
        raise ValueError("RMSNorm VJP requires equal [..., hidden] tensors")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    w = _rmsnorm_weight(x, weight)
    weighted = cotangent.to(device=x.device, dtype=x.dtype) * w
    mean_square = x.square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + float(eps))
    mean_xp = (x * weighted).mean(dim=-1, keepdim=True)
    return inv_rms * weighted - x * inv_rms.pow(3) * mean_xp


def final_layer_jvp(
    x: torch.Tensor,
    tangent: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    adaln_scale: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Analytic native FinalLayer JVP; retained for historical verification."""
    if adaln_scale.ndim != 1 or int(adaln_scale.shape[0]) != int(x.shape[-1]):
        raise ValueError("AdaLN scale does not match hidden width")
    gain = 1.0 + adaln_scale.to(device=x.device)
    norm_tangent = rmsnorm_jvp(x, tangent, norm_weight, norm_eps)
    operator = head_weight.to(device=x.device, dtype=torch.float32)
    return torch.matmul(
        (norm_tangent * gain).to(torch.float32),
        operator.transpose(0, 1),
    )


def final_layer_vjp(
    x: torch.Tensor,
    output_cotangent: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    adaln_scale: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Analytic native FinalLayer VJP."""
    if adaln_scale.ndim != 1 or int(adaln_scale.shape[0]) != int(x.shape[-1]):
        raise ValueError("AdaLN scale does not match hidden width")
    operator = head_weight.to(device=x.device, dtype=torch.float32)
    if int(output_cotangent.shape[-1]) != int(operator.shape[0]):
        raise ValueError("output cotangent does not match FinalLayer output width")
    grad_modulated = torch.matmul(output_cotangent.to(torch.float32), operator)
    gain = 1.0 + adaln_scale.to(device=x.device)
    promoted = torch.promote_types(x.dtype, gain.dtype)
    grad_normed = (grad_modulated.to(promoted) * gain.to(promoted)).to(x.dtype)
    return rmsnorm_vjp(x, grad_normed, norm_weight, norm_eps)


def final_layer_metric_direction(
    x: torch.Tensor,
    delta: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    adaln_scale: torch.Tensor,
    head_weight: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Historical helper for the rejected J^T J d experiment."""
    jvp_started = time.perf_counter()
    projected = final_layer_jvp(
        x,
        delta,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        adaln_scale=adaln_scale,
        head_weight=head_weight,
    )
    jvp_seconds = time.perf_counter() - jvp_started
    vjp_started = time.perf_counter()
    direction = final_layer_vjp(
        x,
        projected,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        adaln_scale=adaln_scale,
        head_weight=head_weight,
    )
    return direction, jvp_seconds, time.perf_counter() - vjp_started


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


def _radially_bound_direction_tensor(
    coefficient: float,
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    limit: float = _CORRECTION_NORM_LIMIT,
) -> _TensorBoundedDirection:
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
        return _TensorBoundedDirection(
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
    eps = torch.as_tensor(
        torch.finfo(torch.float32).eps * math.sqrt(max(1, d.numel())),
        dtype=torch.float32,
        device=m.device,
    )
    finite = (
        torch.isfinite(d).all()
        & torch.isfinite(m).all()
        & torch.isfinite(d_norm)
        & torch.isfinite(m_norm)
    )
    eligible = finite & (d_norm > eps) & (m_norm > 0.0)
    safe_d_norm = torch.where(eligible, d_norm, torch.ones_like(d_norm))
    direction_ratio = torch.where(eligible, m_norm / safe_d_norm, zero)
    raw_ratio = abs(float(coefficient)) * direction_ratio
    eligible = eligible & torch.isfinite(raw_ratio)
    raw_ratio = torch.where(eligible, raw_ratio, zero)
    radial_scale = 1.0 / (1.0 + raw_ratio / float(limit))
    radial_scale = torch.where(eligible, radial_scale, one)
    bounded_ratio = torch.where(eligible, raw_ratio * radial_scale, zero)
    correction = torch.where(
        eligible,
        float(coefficient) * radial_scale * m,
        torch.zeros_like(m),
    )
    eligible = eligible & torch.isfinite(correction).all()
    correction = torch.where(eligible, correction, torch.zeros_like(m))
    radial_scale = torch.where(eligible, radial_scale, one)
    bounded_ratio = torch.where(eligible, bounded_ratio, zero)
    direction_ratio = torch.where(eligible, direction_ratio, zero)
    raw_ratio = torch.where(eligible, raw_ratio, zero)
    return _TensorBoundedDirection(
        correction=correction,
        direction_norm_ratio=direction_ratio,
        raw_norm_ratio=raw_ratio,
        radial_scale=radial_scale,
        bounded_norm_ratio=bounded_ratio,
        bound_active=eligible & (radial_scale < 1.0 - 1e-7),
        eligible=eligible,
    )


def radially_bound_direction(
    coefficient: float,
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    limit: float = _CORRECTION_NORM_LIMIT,
) -> BoundedDirection:
    bounded = _radially_bound_direction_tensor(
        coefficient,
        direction,
        reference_delta,
        limit=limit,
    )
    values = torch.stack(
        (
            bounded.direction_norm_ratio,
            bounded.raw_norm_ratio,
            bounded.radial_scale,
            bounded.bounded_norm_ratio,
            bounded.bound_active.to(torch.float32),
            bounded.eligible.to(torch.float32),
        )
    ).detach().to(device="cpu").tolist()
    direction_ratio, raw_ratio, radial_scale, bounded_ratio, active, eligible = values
    return BoundedDirection(
        correction=bounded.correction,
        direction_norm_ratio=float(direction_ratio),
        raw_norm_ratio=float(raw_ratio),
        radial_scale=float(radial_scale),
        bounded_norm_ratio=float(bounded_ratio),
        bound_active=bool(active),
        eligible=bool(eligible),
    )


def _final_layer_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    geometry: FinalLayerGeometry,
    adaln_scale: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Exact sampled F(left)-F(right); FinalLayer shift and bias cancel."""
    weight = geometry.norm_weight.to(device=left.device, dtype=left.dtype)
    left_norm = torch.nn.functional.rms_norm(
        left,
        (int(left.shape[-1]),),
        weight=weight,
        eps=geometry.norm_eps,
    )
    right_norm = torch.nn.functional.rms_norm(
        right,
        (int(right.shape[-1]),),
        weight=weight,
        eps=geometry.norm_eps,
    )
    gain = 1.0 + adaln_scale.to(device=left.device)
    operator = head_weight.to(device=left.device, dtype=torch.float32)
    return torch.matmul(
        ((left_norm - right_norm) * gain).to(torch.float32),
        operator.transpose(0, 1),
    )


def _static_head_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    operator = head_weight.to(device=left.device, dtype=torch.float32)
    return torch.matmul(
        (left - right).to(torch.float32),
        operator.transpose(0, 1),
    )


@dataclass(frozen=True, slots=True)
class _ScalarEvidenceResult:
    evidence: AnchorEvidence
    raw_weights: dict[str, torch.Tensor]


def _stream_decision_values(
    decision: ModelAwareForecastDecision,
    name: str,
    generic_audio: float,
    generic_video: float,
) -> tuple[float, float, float, float, float]:
    if name == "audio":
        telemetry = decision.audio_correction_telemetry
        return (
            decision.audio_blend_weight,
            decision.audio_correction_gain,
            generic_audio,
            telemetry.diagonal_candidate_gain,
            telemetry.model_candidate_gain,
        )
    if name == "video":
        telemetry = decision.video_correction_telemetry
        return (
            decision.video_blend_weight,
            decision.video_correction_gain,
            generic_video,
            telemetry.diagonal_candidate_gain,
            telemetry.model_candidate_gain,
        )
    if not math.isclose(
        decision.audio_blend_weight,
        decision.video_blend_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("packed model-aware evidence requires audio/video row metadata")
    return (
        decision.video_blend_weight,
        0.5 * (decision.audio_correction_gain + decision.video_correction_gain),
        0.5 * (generic_audio + generic_video),
        0.5
        * (
            decision.audio_correction_telemetry.diagonal_candidate_gain
            + decision.video_correction_telemetry.diagonal_candidate_gain
        ),
        0.5
        * (
            decision.audio_correction_telemetry.model_candidate_gain
            + decision.video_correction_telemetry.model_candidate_gain
        ),
    )


def _risk_anchor_evidence(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    """Feature-2 trajectory evidence only; no Feature-3 correction work."""
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

        fit_started = time.perf_counter()
        raw_weights = forecaster.model_aware_weights(
            step.coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=0.0,
        )
        weight_fit_seconds += time.perf_counter() - fit_started
        selection_started = time.perf_counter()
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        actual = forecaster._sample_segment_device(combined, start, end)
        sample_index_seconds += time.perf_counter() - selection_started
        if any(sample.device != actual.device for sample in history_samples):
            raise RuntimeError("model-aware risk evidence device changed during actual history")

        reduction_started = time.perf_counter()
        predicted = torch.zeros_like(actual)
        for weight_value, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight_value != 0.0:
                predicted.add_(sample, alpha=float(weight_value))
        latest = history_samples[-1]
        previous = history_samples[-2]
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_DIRECTION_EPS)
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
        reduction_seconds += time.perf_counter() - reduction_started
        transfer_started = time.perf_counter()
        forecast_value, curvature_value = values.detach().to(device="cpu").tolist()
        scalar_transfer_seconds += time.perf_counter() - transfer_started
        stream_evidence[name] = StreamAnchorEvidence(
            forecast_ratio=float(forecast_value),
            curvature_ratio=float(curvature_value),
        )

    if "packed" in stream_evidence:
        audio = video = stream_evidence["packed"]
    else:
        audio = stream_evidence.get("audio", StreamAnchorEvidence())
        video = stream_evidence.get("video", StreamAnchorEvidence())
    fit_started = time.perf_counter()
    fit_condition = forecaster.fit_condition(degree=decision.degree)
    fit_condition_seconds = time.perf_counter() - fit_started
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


def _scalar_anchor_evidence(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    exact_head_weights: dict[str, torch.Tensor],
    stream_diagonals: dict[str, torch.Tensor],
) -> _ScalarEvidenceResult | None:
    """Retained scalar Feature-3 evidence; intentionally contains no K=2 solve."""
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or forecaster.feature_shape is None:
        return None
    if tuple(combined.shape) != forecaster.feature_shape:
        raise ValueError("actual feature shape changed during model-aware evidence sampling")
    if len(forecaster._evidence_history) != forecaster.history_length:
        raise RuntimeError("device-local model-aware evidence history is not aligned")

    generic_audio, generic_video = runtime.model_aware.generic_correction_gains(decision)
    stream_evidence: dict[str, StreamAnchorEvidence] = {}
    raw_weights_by_stream: dict[str, torch.Tensor] = {}
    weight_fit_seconds = sample_index_seconds = reduction_seconds = 0.0
    scalar_transfer_seconds = exact_head_projection_seconds = 0.0

    for name, start, end in runtime._stream_ranges(step.calls[0]):
        blend, model_gain, generic_gain, diagonal_gain, exact_gain = _stream_decision_values(
            decision,
            name,
            generic_audio,
            generic_video,
        )
        fit_started = time.perf_counter()
        raw_weights = forecaster.model_aware_weights(
            step.coordinate,
            blend,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
            correction_gain=0.0,
        )
        weight_fit_seconds += time.perf_counter() - fit_started
        raw_weights_by_stream[name] = raw_weights

        selection_started = time.perf_counter()
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        actual = forecaster._sample_segment_device(combined, start, end)
        diagonal = stream_diagonals.get(name)
        sampled_diagonal = None
        if diagonal is not None:
            if diagonal.device != actual.device:
                raise ValueError("FinalLayer diagonal is on the wrong evidence device")
            sampled_diagonal = forecaster._sample_channel_sensitivity(
                forecaster.feature_shape,
                start,
                end,
                diagonal,
            )

        actual_head = None
        history_head: list[torch.Tensor] = []
        head_weight = exact_head_weights.get(name)
        if head_weight is not None:
            if len(forecaster._exact_head_history) != forecaster.history_length:
                raise RuntimeError("exact head evidence history is not aligned")
            indices = forecaster._exact_head_row_indices.get(name)
            if indices is None:
                raise RuntimeError("exact head evidence row indices are missing")
            history_head = [entry[name] for entry in forecaster._exact_head_history]
            head_started = time.perf_counter()
            actual_head = forecaster._project_hidden_rows(
                combined,
                start,
                end,
                indices,
                head_weight,
            )
            exact_head_projection_seconds += time.perf_counter() - head_started
        sample_index_seconds += time.perf_counter() - selection_started

        reduction_started = time.perf_counter()
        predicted = torch.zeros_like(actual)
        for weight_value, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight_value != 0.0:
                predicted.add_(sample, alpha=float(weight_value))
        latest = history_samples[-1]
        previous = history_samples[-2]
        delta = latest - previous
        residual = actual - predicted
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_DIRECTION_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        dot_epsilon = epsilon.square() * max(1, int(delta.numel()))
        projection = torch.dot(residual, delta) / torch.dot(delta, delta).clamp_min(dot_epsilon)
        if sampled_diagonal is None:
            diagonal_projection = projection
        else:
            weighted_delta = sampled_diagonal * delta
            diagonal_projection = torch.dot(residual, weighted_delta) / torch.dot(
                delta,
                weighted_delta,
            ).clamp_min(dot_epsilon)

        predicted_head = latest_head = delta_head = None
        if actual_head is None:
            model_projection = projection
        else:
            predicted_head = torch.zeros_like(actual_head)
            for weight_value, sample in zip(raw_weights.tolist(), history_head, strict=True):
                if weight_value != 0.0:
                    predicted_head.add_(sample, alpha=float(weight_value))
            latest_head = history_head[-1]
            delta_head = latest_head - history_head[-2]
            residual_head = actual_head - predicted_head
            flat_delta = delta_head.reshape(-1)
            flat_residual = residual_head.reshape(-1)
            head_epsilon = _tensor_rms(actual_head).mul(1e-6).clamp_min(_DIRECTION_EPS)
            model_projection = torch.dot(flat_residual, flat_delta) / torch.dot(
                flat_delta,
                flat_delta,
            ).clamp_min(head_epsilon.square() * max(1, int(flat_delta.numel())))

        forecast_ratio = _tensor_rms(residual) / hold_rms
        model_ratio = _tensor_rms(actual - (predicted + float(model_gain) * delta)) / hold_rms
        generic_ratio = _tensor_rms(actual - (predicted + float(generic_gain) * delta)) / hold_rms
        diagonal_ratio = _tensor_rms(actual - (predicted + float(diagonal_gain) * delta)) / hold_rms
        exact_ratio = _tensor_rms(actual - (predicted + float(exact_gain) * delta)) / hold_rms
        if actual_head is None:
            model_head_ratio = model_ratio
            generic_head_ratio = generic_ratio
            diagonal_head_ratio = diagonal_ratio
            exact_head_ratio = exact_ratio
        else:
            assert predicted_head is not None and latest_head is not None and delta_head is not None
            head_hold_rms = _tensor_rms(actual_head - latest_head).clamp_min(_DIRECTION_EPS)
            model_head_ratio = _tensor_rms(
                actual_head - (predicted_head + float(model_gain) * delta_head)
            ) / head_hold_rms
            generic_head_ratio = _tensor_rms(
                actual_head - (predicted_head + float(generic_gain) * delta_head)
            ) / head_hold_rms
            diagonal_head_ratio = _tensor_rms(
                actual_head - (predicted_head + float(diagonal_gain) * delta_head)
            ) / head_hold_rms
            exact_head_ratio = _tensor_rms(
                actual_head - (predicted_head + float(exact_gain) * delta_head)
            ) / head_hold_rms

        if len(history_samples) >= 3:
            curvature = latest - 2.0 * previous + history_samples[-3]
            curvature_ratio = _tensor_rms(curvature) / _tensor_rms(delta).clamp_min(epsilon)
        else:
            curvature_ratio = torch.zeros((), dtype=torch.float32, device=actual.device)
        values = torch.stack(
            (
                forecast_ratio,
                curvature_ratio,
                projection,
                diagonal_projection,
                model_projection,
                model_ratio,
                generic_ratio,
                diagonal_ratio,
                exact_ratio,
                model_head_ratio,
                generic_head_ratio,
                diagonal_head_ratio,
                exact_head_ratio,
            )
        )
        reduction_seconds += time.perf_counter() - reduction_started
        transfer_started = time.perf_counter()
        unpacked = values.detach().to(device="cpu").tolist()
        scalar_transfer_seconds += time.perf_counter() - transfer_started
        (
            forecast_value,
            curvature_value,
            projection_value,
            diagonal_projection_value,
            model_projection_value,
            model_ratio_value,
            generic_ratio_value,
            diagonal_ratio_value,
            exact_ratio_value,
            model_head_value,
            generic_head_value,
            diagonal_head_value,
            exact_head_value,
        ) = unpacked
        stream_evidence[name] = StreamAnchorEvidence(
            forecast_ratio=forecast_value,
            curvature_ratio=curvature_value,
            residual_projection=projection_value,
            model_corrected_ratio=model_ratio_value,
            generic_corrected_ratio=generic_ratio_value,
            diagonal_projection=diagonal_projection_value,
            model_projection=model_projection_value,
            diagonal_candidate_ratio=diagonal_ratio_value,
            model_candidate_ratio=exact_ratio_value,
            model_corrected_head_ratio=model_head_value,
            generic_corrected_head_ratio=generic_head_value,
            diagonal_candidate_head_ratio=diagonal_head_value,
            model_candidate_head_ratio=exact_head_value,
        )

    if "packed" in stream_evidence:
        audio = video = stream_evidence["packed"]
    else:
        audio = stream_evidence.get("audio", StreamAnchorEvidence())
        video = stream_evidence.get("video", StreamAnchorEvidence())
    fit_started = time.perf_counter()
    fit_condition = forecaster.fit_condition(degree=decision.degree)
    fit_condition_seconds = time.perf_counter() - fit_started
    evidence = AnchorEvidence(
        forecast_ratio=max(audio.forecast_ratio, video.forecast_ratio),
        curvature_ratio=max(audio.curvature_ratio, video.curvature_ratio),
        fit_condition=fit_condition,
        audio_projection=audio.residual_projection,
        video_projection=video.residual_projection,
        model_corrected_ratio=max(audio.model_corrected_ratio, video.model_corrected_ratio),
        generic_corrected_ratio=max(audio.generic_corrected_ratio, video.generic_corrected_ratio),
        audio=audio,
        video=video,
        subspace_workspace_bytes=0,
        timing=AnchorEvidenceTiming(
            weight_fit_seconds=weight_fit_seconds,
            sample_index_seconds=sample_index_seconds,
            device_transfer_seconds=scalar_transfer_seconds,
            scalar_transfer_seconds=scalar_transfer_seconds,
            reduction_seconds=reduction_seconds,
            exact_head_projection_seconds=exact_head_projection_seconds,
            fit_condition_seconds=fit_condition_seconds,
            subspace_gram_seconds=0.0,
            subspace_solve_seconds=0.0,
        ),
    )
    return _ScalarEvidenceResult(evidence=evidence, raw_weights=raw_weights_by_stream)


_ORIGINAL_FORECASTER_UPDATE = HistoryWeightForecaster.update
_ORIGINAL_RUNTIME_START = SpectrumH3Runtime.start_run
_ORIGINAL_RUNTIME_END = SpectrumH3Runtime.end_run
_ORIGINAL_RUNTIME_DISABLE = SpectrumH3Runtime._disable_forecasting
_ORIGINAL_RUNTIME_DISABLE_MODEL_AWARE = SpectrumH3Runtime.disable_model_aware
_ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary

_FULL_TELEMETRY_HOOK: Callable[..., None] | None = None


def register_feature3_full_telemetry_hook(hook: Callable[..., None]) -> None:
    global _FULL_TELEMETRY_HOOK
    if _FULL_TELEMETRY_HOOK is not None and _FULL_TELEMETRY_HOOK is not hook:
        raise RuntimeError("a Feature-3 full telemetry hook is already installed")
    _FULL_TELEMETRY_HOOK = hook


def _patched_forecaster_update(
    self: HistoryWeightForecaster,
    coordinate: float,
    feature: torch.Tensor,
    **kwargs,
) -> None:
    mode = getattr(self, "_feature3_evidence_capture_mode", None)
    if mode == "schedule":
        kwargs["evidence_segments"] = None
        kwargs["exact_head_weights"] = None
    elif mode == "schedule_confidence":
        kwargs["exact_head_weights"] = None
    return _ORIGINAL_FORECASTER_UPDATE(self, coordinate, feature, **kwargs)


def _patched_runtime_weight_segments(
    self: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
):
    """Applied full path: retained trust-mixed scalar latest-delta correction only."""
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


def _record_evidence_stats(runtime: SpectrumH3Runtime, evidence: AnchorEvidence) -> None:
    runtime.stats.model_aware_anchor_updates += 1
    runtime.stats.model_aware_evidence_weight_fit_seconds += evidence.timing.weight_fit_seconds
    runtime.stats.model_aware_evidence_sample_index_seconds += evidence.timing.sample_index_seconds
    runtime.stats.model_aware_evidence_device_transfer_seconds += evidence.timing.device_transfer_seconds
    runtime.stats.model_aware_evidence_scalar_transfer_seconds += evidence.timing.scalar_transfer_seconds
    runtime.stats.model_aware_evidence_reduction_seconds += evidence.timing.reduction_seconds
    runtime.stats.model_aware_evidence_exact_head_projection_seconds += (
        evidence.timing.exact_head_projection_seconds
    )
    runtime.stats.model_aware_exact_head_projection_seconds += (
        evidence.timing.exact_head_projection_seconds
    )
    runtime.stats.model_aware_evidence_fit_condition_seconds += evidence.timing.fit_condition_seconds
    runtime.stats.model_aware_subspace_gram_seconds += 0.0
    runtime.stats.model_aware_subspace_solve_seconds += 0.0
    runtime.stats.model_aware_subspace_workspace_bytes = max(
        runtime.stats.model_aware_subspace_workspace_bytes,
        0,
    )


def _patched_runtime_observe(
    self: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    exact_head_weights: dict[str, torch.Tensor],
    stream_diagonals: dict[str, torch.Tensor],
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
        scalar_result: _ScalarEvidenceResult | None = None
        if mode == "schedule_confidence":
            evidence = _risk_anchor_evidence(self, step, combined, decision)
        else:
            scalar_result = _scalar_anchor_evidence(
                self,
                step,
                combined,
                decision,
                exact_head_weights,
                stream_diagonals,
            )
            evidence = None if scalar_result is None else scalar_result.evidence

        if evidence is not None:
            if (
                mode == "full"
                and scalar_result is not None
                and _FULL_TELEMETRY_HOOK is not None
            ):
                try:
                    _FULL_TELEMETRY_HOOK(
                        self,
                        step,
                        combined,
                        decision,
                        scalar_result.raw_weights,
                        exact_head_weights,
                    )
                except torch.cuda.OutOfMemoryError:
                    raise
                except (RuntimeError, TypeError, ValueError) as exc:
                    LOG.warning(
                        "Spectrum H3 optional Feature-3 telemetry failed closed at step=%s: %s",
                        step.step_id,
                        exc,
                    )
            self.model_aware.observe_anchor(evidence, decision)
            _record_evidence_stats(self, evidence)
            self.stats.model_aware_exact_head_projection_calls += len(exact_head_weights)
            self.stats.model_aware_model_corrected_ratio_mean = (
                self.model_aware.model_corrected_ratio_mean
            )
            self.stats.model_aware_generic_corrected_ratio_mean = (
                self.model_aware.generic_corrected_ratio_mean
            )
            if self.config.debug and mode == "full":
                LOG.warning(
                    "Spectrum H3 scalar correction anchor step=%s "
                    "audio_raw=%.6f audio_generic=%.6f audio_exact=%.6f audio_applied=%.6f "
                    "video_raw=%.6f video_generic=%.6f video_exact=%.6f video_applied=%.6f",
                    step.step_id,
                    evidence.audio.forecast_ratio,
                    evidence.audio.generic_corrected_ratio,
                    evidence.audio.model_candidate_ratio,
                    evidence.audio.model_corrected_ratio,
                    evidence.video.forecast_ratio,
                    evidence.video.generic_corrected_ratio,
                    evidence.video.model_candidate_ratio,
                    evidence.video.model_corrected_ratio,
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


def _patched_runtime_start(self: SpectrumH3Runtime, *args, **kwargs):
    run_id = _ORIGINAL_RUNTIME_START(self, *args, **kwargs)
    self.forecaster._feature3_evidence_capture_mode = self.config.model_aware_mode
    return run_id


def _release_full_head_state(runtime: SpectrumH3Runtime) -> None:
    runtime.model_aware._head_device_cache.clear()


def _patched_runtime_end(self: SpectrumH3Runtime, run_id: int) -> None:
    try:
        _ORIGINAL_RUNTIME_END(self, run_id)
    finally:
        _release_full_head_state(self)


def _patched_runtime_disable(self: SpectrumH3Runtime, reason: str) -> bool:
    result = _ORIGINAL_RUNTIME_DISABLE(self, reason)
    _release_full_head_state(self)
    return result


def _patched_runtime_disable_model_aware(self: SpectrumH3Runtime, reason: str) -> None:
    _ORIGINAL_RUNTIME_DISABLE_MODEL_AWARE(self, reason)
    _release_full_head_state(self)


def _patched_debug_summary(self: SpectrumH3Runtime) -> str:
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
        "feature3_applied_correction=scalar_latest_delta "
        "feature3_transformed_trajectory_runtime=retired "
        "feature3_k2_runtime=retired "
        "feature3_direction_evidence_bytes=0 "
        "feature3_direction_workspace_bytes=0 "
        "feature3_direction_compute_s=0.000000 "
        "feature3_static_enqueue_s=0.000000 "
        "feature3_full_enqueue_s=0.000000 "
        "feature3_jvp_enqueue_s=0.000000 "
        "feature3_vjp_enqueue_s=0.000000 "
        "feature3_extra_transformer_nfe=0"
    )


def install_feature3_direction_experiment() -> None:
    """Install scalar Feature-3/mode-boundary fixes; no rejected direction runtime."""
    if getattr(SpectrumH3Runtime, "_feature3_scalar_revision_installed", False):
        return
    HistoryWeightForecaster.update = _patched_forecaster_update
    SpectrumH3Runtime._model_aware_weight_segments = _patched_runtime_weight_segments
    SpectrumH3Runtime._observe_model_aware_anchor = _patched_runtime_observe
    SpectrumH3Runtime.start_run = _patched_runtime_start
    SpectrumH3Runtime.end_run = _patched_runtime_end
    SpectrumH3Runtime._disable_forecasting = _patched_runtime_disable
    SpectrumH3Runtime.disable_model_aware = _patched_runtime_disable_model_aware
    SpectrumH3Runtime.debug_summary = _patched_debug_summary
    SpectrumH3Runtime._feature3_scalar_revision_installed = True


__all__ = [
    "BoundedDirection",
    "FinalLayerGeometry",
    "final_layer_jvp",
    "final_layer_metric_direction",
    "final_layer_vjp",
    "install_feature3_direction_experiment",
    "radially_bound_direction",
    "register_feature3_full_telemetry_hook",
    "rmsnorm_jvp",
    "rmsnorm_vjp",
    "static_head_metric_direction",
]
