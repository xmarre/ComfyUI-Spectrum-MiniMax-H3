from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from . import minimax_h3 as _h3
from .model_aware import (
    AnchorEvidence,
    AnchorEvidenceTiming,
    ModelAwareController,
    ModelAwareForecastDecision,
    StreamAnchorEvidence,
)
from .forecast import HistoryWeightForecaster
from .runtime import SpectrumH3Runtime

_CORRECTION_NORM_LIMIT = 0.25
_DIRECTION_SAMPLE_ROWS = 32
_DIRECTION_ALPHA_LIMIT = 2.0
_DIRECTION_EPS = torch.finfo(torch.float32).eps


@dataclass(frozen=True, slots=True)
class FinalLayerGeometry:
    """Causal, detached local FinalLayer geometry for one target timestep."""

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
class DirectionCandidateTelemetry:
    eligible: bool = False
    geometry_available: bool = False
    alpha_history_count: int = 0
    direction_norm_ratio: float = 0.0
    coefficient_used: float = 0.0
    instantaneous_coefficient: float = 0.0
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
    """Return W^T W d without ever materializing W^T W."""
    if delta.ndim < 1 or head_weight.ndim != 2:
        raise ValueError("static FinalLayer direction requires [..., hidden] delta and [out, hidden] head")
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
    """Analytic JVP of PyTorch RMSNorm used by native ComfyUI H3."""
    if x.shape != tangent.shape or x.ndim < 1:
        raise ValueError("RMSNorm JVP requires equal [..., hidden] tensors")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    tangent = tangent.to(device=x.device, dtype=x.dtype)
    w = _rmsnorm_weight(x, weight)
    mean_square = x.square().mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + float(eps))
    mean_xt = (x * tangent).mean(dim=-1, keepdim=True)
    base = inv_rms * tangent - x * inv_rms.pow(3) * mean_xt
    return base * w


def rmsnorm_vjp(
    x: torch.Tensor,
    cotangent: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Analytic VJP of PyTorch RMSNorm used by native ComfyUI H3."""
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
    """Analytic JVP for h -> W[(RMSNorm(h) * (1 + scale)) + shift] + b."""
    if adaln_scale.ndim != 1 or int(adaln_scale.shape[0]) != int(x.shape[-1]):
        raise ValueError("AdaLN scale does not match hidden width")
    gain = 1.0 + adaln_scale.to(device=x.device)
    norm_tangent = rmsnorm_jvp(x, tangent, norm_weight, norm_eps)
    modulated_tangent = norm_tangent * gain
    operator = head_weight.to(device=x.device, dtype=torch.float32)
    return torch.matmul(modulated_tangent.to(torch.float32), operator.transpose(0, 1))


def final_layer_vjp(
    x: torch.Tensor,
    output_cotangent: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    adaln_scale: torch.Tensor,
    head_weight: torch.Tensor,
) -> torch.Tensor:
    """Analytic VJP for the same native FinalLayer mapping."""
    if adaln_scale.ndim != 1 or int(adaln_scale.shape[0]) != int(x.shape[-1]):
        raise ValueError("AdaLN scale does not match hidden width")
    operator = head_weight.to(device=x.device, dtype=torch.float32)
    if int(output_cotangent.shape[-1]) != int(operator.shape[0]):
        raise ValueError("output cotangent does not match FinalLayer output width")
    grad_modulated = torch.matmul(output_cotangent.to(torch.float32), operator)
    gain = 1.0 + adaln_scale.to(device=x.device)
    # Native FinalLayer casts the modulated hidden to fp32 before the output
    # projection. Autograd through that cast returns the gradient in the
    # modulated hidden's promoted dtype and then in the RMSNorm output dtype.
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
    """Return J_t^T J_t d and enqueue-only JVP/VJP timing without synchronizing."""
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
    vjp_seconds = time.perf_counter() - vjp_started
    return direction, jvp_seconds, vjp_seconds


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


@dataclass(frozen=True, slots=True)
class _TensorBoundedDirection:
    correction: torch.Tensor
    direction_norm_ratio: torch.Tensor
    raw_norm_ratio: torch.Tensor
    radial_scale: torch.Tensor
    bounded_norm_ratio: torch.Tensor
    bound_active: torch.Tensor
    eligible: torch.Tensor


def _radially_bound_direction_tensor(
    coefficient: float,
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    limit: float = _CORRECTION_NORM_LIMIT,
) -> _TensorBoundedDirection:
    """Device-local rational radial bound; performs no host synchronization."""
    if direction.shape != reference_delta.shape:
        raise ValueError("direction and reference delta must have identical shape")
    d = reference_delta.to(torch.float32)
    m = direction.to(torch.float32)
    scalar_zero = torch.zeros((), dtype=torch.float32, device=m.device)
    scalar_one = torch.ones_like(scalar_zero)
    if not math.isfinite(float(coefficient)) or not math.isfinite(float(limit)) or float(limit) <= 0.0:
        return _TensorBoundedDirection(
            correction=torch.zeros_like(m),
            direction_norm_ratio=scalar_zero,
            raw_norm_ratio=scalar_zero,
            radial_scale=scalar_one,
            bounded_norm_ratio=scalar_zero,
            bound_active=torch.zeros((), dtype=torch.bool, device=m.device),
            eligible=torch.zeros((), dtype=torch.bool, device=m.device),
        )

    d_norm = torch.linalg.vector_norm(d)
    m_norm = torch.linalg.vector_norm(m)
    eps = torch.as_tensor(
        torch.finfo(torch.float32).eps * math.sqrt(max(1, d.numel())),
        dtype=torch.float32,
        device=m.device,
    )
    finite = torch.isfinite(d).all() & torch.isfinite(m).all()
    eligible = finite & (d_norm > eps) & (m_norm > eps)
    safe_d_norm = d_norm.clamp_min(eps)
    direction_ratio = torch.where(eligible, m_norm / safe_d_norm, scalar_zero)
    raw_ratio = abs(float(coefficient)) * direction_ratio
    bounded_ratio = raw_ratio / (1.0 + raw_ratio / float(limit))
    radial_scale = torch.where(
        raw_ratio > 0.0,
        bounded_ratio / raw_ratio.clamp_min(eps),
        scalar_one,
    )
    radial_scale = torch.where(eligible, radial_scale, scalar_one)
    correction = torch.where(
        eligible,
        float(coefficient) * radial_scale * m,
        torch.zeros_like(m),
    )
    bound_active = eligible & (radial_scale < 1.0 - 1e-7)
    return _TensorBoundedDirection(
        correction=correction,
        direction_norm_ratio=direction_ratio,
        raw_norm_ratio=raw_ratio,
        radial_scale=radial_scale,
        bounded_norm_ratio=bounded_ratio,
        bound_active=bound_active,
        eligible=eligible,
    )


def radially_bound_direction(
    coefficient: float,
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
    *,
    limit: float = _CORRECTION_NORM_LIMIT,
) -> BoundedDirection:
    """Public scalar diagnostics wrapper for the device-local radial bound."""
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
    direction_ratio, raw_ratio, radial_scale, bounded_ratio, bound_active, eligible = values
    return BoundedDirection(
        correction=bounded.correction,
        direction_norm_ratio=float(direction_ratio),
        raw_norm_ratio=float(raw_ratio),
        radial_scale=float(radial_scale),
        bounded_norm_ratio=float(bounded_ratio),
        bound_active=bool(bound_active),
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
    """Exact sampled FinalLayer output difference; shift and bias cancel."""
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
    modulated_difference = (left_norm - right_norm) * gain
    operator = head_weight.to(device=left.device, dtype=torch.float32)
    return torch.matmul(
        modulated_difference.to(torch.float32),
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


def _controller_feature3_reset(controller: ModelAwareController) -> None:
    controller._feature3_row_history = []
    controller._feature3_row_indices = {}
    controller._feature3_last = {}
    controller._feature3_geometry_seconds = 0.0
    controller._feature3_direction_seconds = 0.0
    controller._feature3_static_seconds = 0.0
    controller._feature3_full_seconds = 0.0
    controller._feature3_jvp_seconds = 0.0
    controller._feature3_vjp_seconds = 0.0
    controller._feature3_scalar_transfer_seconds = 0.0
    controller._feature3_workspace_bytes = 0
    controller._feature3_geometry_failures = 0
    for stream in ("audio", "video"):
        for kind in ("static", "full"):
            prefix = f"_feature3_{stream}_{kind}"
            setattr(controller, f"{prefix}_alpha_ewma", 0.0)
            setattr(controller, f"{prefix}_alpha_count", 0)
            setattr(controller, f"{prefix}_eligible_count", 0)
            setattr(controller, f"{prefix}_fallback_count", 0)
            setattr(controller, f"{prefix}_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_ratio_max", 0.0)
            setattr(controller, f"{prefix}_direction_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_direction_ratio_max", 0.0)
            setattr(controller, f"{prefix}_bounded_ratio_sum", 0.0)
            setattr(controller, f"{prefix}_bounded_ratio_max", 0.0)
            setattr(controller, f"{prefix}_bound_active_count", 0)
        for comparison in ("static_vs_generic", "static_vs_exact", "full_vs_generic", "full_vs_exact", "full_vs_static"):
            prefix = f"_feature3_{stream}_{comparison}"
            setattr(controller, f"{prefix}_count", 0)
            setattr(controller, f"{prefix}_wins", 0)
            setattr(controller, f"{prefix}_losses", 0)
            setattr(controller, f"{prefix}_advantage_sum", 0.0)
            setattr(controller, f"{prefix}_advantage_abs_max", 0.0)


_ORIGINAL_CONTROLLER_RESET = ModelAwareController.reset
_ORIGINAL_CONTROLLER_SNAPSHOT = ModelAwareController.snapshot
_ORIGINAL_CONTROLLER_RESTORE = ModelAwareController.restore


def _patched_controller_reset(self: ModelAwareController) -> None:
    _ORIGINAL_CONTROLLER_RESET(self)
    _controller_feature3_reset(self)


def _feature3_numeric_keys(controller: ModelAwareController) -> list[str]:
    keys = [
        "_feature3_geometry_seconds",
        "_feature3_direction_seconds",
        "_feature3_static_seconds",
        "_feature3_full_seconds",
        "_feature3_jvp_seconds",
        "_feature3_vjp_seconds",
        "_feature3_scalar_transfer_seconds",
        "_feature3_workspace_bytes",
        "_feature3_geometry_failures",
    ]
    for stream in ("audio", "video"):
        for kind in ("static", "full"):
            prefix = f"_feature3_{stream}_{kind}"
            keys.extend(
                f"{prefix}_{suffix}"
                for suffix in (
                    "alpha_ewma",
                    "alpha_count",
                    "eligible_count",
                    "fallback_count",
                    "ratio_sum",
                    "ratio_max",
                    "direction_ratio_sum",
                    "direction_ratio_max",
                    "bounded_ratio_sum",
                    "bounded_ratio_max",
                    "bound_active_count",
                )
            )
        for comparison in ("static_vs_generic", "static_vs_exact", "full_vs_generic", "full_vs_exact", "full_vs_static"):
            prefix = f"_feature3_{stream}_{comparison}"
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


def _patched_controller_snapshot(self: ModelAwareController) -> dict[str, Any]:
    state: dict[str, Any] = dict(_ORIGINAL_CONTROLLER_SNAPSHOT(self))
    for key in _feature3_numeric_keys(self):
        state[key] = getattr(self, key)
    state["_feature3_row_history"] = tuple(
        {name: value for name, value in entry.items()}
        for entry in self._feature3_row_history
    )
    state["_feature3_row_indices"] = dict(self._feature3_row_indices)
    state["_feature3_last"] = dict(self._feature3_last)
    return state


def _patched_controller_restore(self: ModelAwareController, state: dict[str, Any]) -> None:
    _ORIGINAL_CONTROLLER_RESTORE(self, state)
    for key in _feature3_numeric_keys(self):
        if key in state:
            current = getattr(self, key)
            value = state[key]
            setattr(self, key, int(value) if isinstance(current, int) else float(value))
    self._feature3_row_history = [
        dict(entry) for entry in state.get("_feature3_row_history", ())
    ]
    self._feature3_row_indices = dict(state.get("_feature3_row_indices", {}))
    self._feature3_last = dict(state.get("_feature3_last", {}))


def _direction_alpha_used(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    confidence: float,
) -> tuple[float, int]:
    prefix = f"_feature3_{stream}_{kind}"
    count = int(getattr(controller, f"{prefix}_alpha_count"))
    ewma = float(getattr(controller, f"{prefix}_alpha_ewma"))
    scale = max(0.25, min(1.0, float(confidence)))
    return ewma * scale, count


def _update_direction_alpha(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    instantaneous: float,
) -> None:
    if not math.isfinite(float(instantaneous)):
        return
    prefix = f"_feature3_{stream}_{kind}"
    count_name = f"{prefix}_alpha_count"
    ewma_name = f"{prefix}_alpha_ewma"
    count = int(getattr(controller, count_name))
    alpha = 0.5 if count < 2 else 0.3
    bounded = _soft_limit(float(instantaneous), _DIRECTION_ALPHA_LIMIT)
    prior = float(getattr(controller, ewma_name))
    setattr(controller, ewma_name, bounded if count == 0 else (1.0 - alpha) * prior + alpha * bounded)
    setattr(controller, count_name, count + 1)


def _record_comparison(
    controller: ModelAwareController,
    stream: str,
    comparison: str,
    baseline: float,
    candidate: float,
) -> float:
    if not all(math.isfinite(v) for v in (baseline, candidate)) or baseline <= 0.0 or candidate <= 0.0:
        return 0.0
    advantage = (baseline - candidate) / max(baseline, 1e-12)
    prefix = f"_feature3_{stream}_{comparison}"
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


@dataclass(frozen=True, slots=True)
class _ScalarEvidenceResult:
    evidence: AnchorEvidence
    raw_weights: dict[str, torch.Tensor]


def _risk_anchor_evidence(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    """Feature-2 trajectory evidence only; no correction fit/projection/candidate work."""
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or forecaster.feature_shape is None:
        return None
    if tuple(combined.shape) != forecaster.feature_shape:
        raise ValueError("actual feature shape changed during model-aware risk evidence sampling")
    if len(forecaster._evidence_history) != forecaster.history_length:
        raise RuntimeError("device-local model-aware risk evidence history is not aligned")

    stream_evidence: dict[str, StreamAnchorEvidence] = {}
    weight_fit_seconds = 0.0
    sample_index_seconds = 0.0
    reduction_seconds = 0.0
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
        if any(sample.device != actual.device for sample in history_samples):
            raise RuntimeError("model-aware risk evidence device changed during actual history")
        sample_index_seconds += time.perf_counter() - selection_started

        reduction_started = time.perf_counter()
        predicted = torch.zeros_like(actual)
        for weight, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight != 0.0:
                predicted.add_(sample, alpha=float(weight))
        latest = history_samples[-1]
        previous = history_samples[-2]
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_DIRECTION_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        forecast_ratio = _tensor_rms(actual - predicted) / hold_rms
        if len(history_samples) >= 3:
            delta = latest - previous
            curvature = latest - 2.0 * previous + history_samples[-3]
            curvature_ratio = _tensor_rms(curvature) / _tensor_rms(delta).clamp_min(epsilon)
        else:
            curvature_ratio = torch.zeros((), dtype=torch.float32, device=actual.device)
        values = torch.stack((forecast_ratio, curvature_ratio))
        reduction_seconds += time.perf_counter() - reduction_started

        transfer_started = time.perf_counter()
        forecast_value, curvature_value = values.detach().to(device="cpu").tolist()
        scalar_transfer_seconds += time.perf_counter() - transfer_started
        # Feature 2 consumes forecast/curvature/condition only. Correction fields
        # deliberately stay zero so schedule_confidence cannot train Feature 3.
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
    weight_fit_seconds = 0.0
    sample_index_seconds = 0.0
    reduction_seconds = 0.0
    scalar_transfer_seconds = 0.0
    exact_head_projection_seconds = 0.0

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
        if any(sample.device != actual.device for sample in history_samples):
            raise RuntimeError("model-aware evidence device changed during actual history")
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
        for weight, sample in zip(raw_weights.tolist(), history_samples, strict=True):
            if weight != 0.0:
                predicted.add_(sample, alpha=float(weight))
        latest = history_samples[-1]
        previous = history_samples[-2]
        delta = latest - previous
        residual = actual - predicted
        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_DIRECTION_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        forecast_rms = _tensor_rms(residual)
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

        predicted_head = None
        latest_head = None
        delta_head = None
        if actual_head is None:
            model_projection = projection
        else:
            predicted_head = torch.zeros_like(actual_head)
            for weight, sample in zip(raw_weights.tolist(), history_head, strict=True):
                if weight != 0.0:
                    predicted_head.add_(sample, alpha=float(weight))
            latest_head = history_head[-1]
            delta_head = latest_head - history_head[-2]
            residual_head = actual_head - predicted_head
            flattened_delta = delta_head.reshape(-1)
            flattened_residual = residual_head.reshape(-1)
            head_eps = _tensor_rms(actual_head).mul(1e-6).clamp_min(_DIRECTION_EPS)
            head_dot_eps = head_eps.square() * max(1, int(flattened_delta.numel()))
            model_projection = torch.dot(flattened_residual, flattened_delta) / torch.dot(
                flattened_delta,
                flattened_delta,
            ).clamp_min(head_dot_eps)

        forecast_ratio = forecast_rms / hold_rms
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
        ) = values.detach().to(device="cpu").tolist()
        scalar_transfer_seconds += time.perf_counter() - transfer_started

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


def _feature3_retain_current_rows(
    controller: ModelAwareController,
    forecaster: Any,
    combined: torch.Tensor,
    ranges: tuple[tuple[str, int, int], ...],
) -> None:
    entry: dict[str, torch.Tensor] = {}
    for name, start, end in ranges:
        row_count = int(end) - int(start)
        indices = controller._feature3_row_indices.get(name)
        if indices is None:
            indices = forecaster._complete_row_indices(
                row_count,
                int(combined.shape[0]),
                combined.device,
                limit=_DIRECTION_SAMPLE_ROWS,
            )
            controller._feature3_row_indices[name] = indices
        selected = combined[:, start:end].index_select(1, indices)
        entry[name] = selected.detach().contiguous()
    controller._feature3_row_history.append(entry)
    if len(controller._feature3_row_history) > int(forecaster.max_history):
        controller._feature3_row_history.pop(0)


def _feature3_evidence_bytes(controller: ModelAwareController) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for entry in controller._feature3_row_history
        for tensor in entry.values()
    ) + sum(
        tensor.numel() * tensor.element_size()
        for tensor in controller._feature3_row_indices.values()
    )


def _candidate_stats(
    controller: ModelAwareController,
    stream: str,
    kind: str,
    telemetry: DirectionCandidateTelemetry,
) -> None:
    prefix = f"_feature3_{stream}_{kind}"
    if not telemetry.eligible:
        setattr(controller, f"{prefix}_fallback_count", getattr(controller, f"{prefix}_fallback_count") + 1)
        return
    setattr(controller, f"{prefix}_eligible_count", getattr(controller, f"{prefix}_eligible_count") + 1)
    setattr(controller, f"{prefix}_ratio_sum", getattr(controller, f"{prefix}_ratio_sum") + telemetry.ordinary_ratio)
    setattr(controller, f"{prefix}_ratio_max", max(getattr(controller, f"{prefix}_ratio_max"), telemetry.ordinary_ratio))
    setattr(
        controller,
        f"{prefix}_direction_ratio_sum",
        getattr(controller, f"{prefix}_direction_ratio_sum") + telemetry.direction_norm_ratio,
    )
    setattr(
        controller,
        f"{prefix}_direction_ratio_max",
        max(getattr(controller, f"{prefix}_direction_ratio_max"), telemetry.direction_norm_ratio),
    )
    setattr(
        controller,
        f"{prefix}_bounded_ratio_sum",
        getattr(controller, f"{prefix}_bounded_ratio_sum") + telemetry.bounded_norm_ratio,
    )
    setattr(
        controller,
        f"{prefix}_bounded_ratio_max",
        max(getattr(controller, f"{prefix}_bounded_ratio_max"), telemetry.bounded_norm_ratio),
    )
    if telemetry.bound_active:
        setattr(
            controller,
            f"{prefix}_bound_active_count",
            getattr(controller, f"{prefix}_bound_active_count") + 1,
        )


def _evaluate_model_directions(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    raw_weights_by_stream: dict[str, torch.Tensor],
    exact_head_weights: dict[str, torch.Tensor],
    geometry: FinalLayerGeometry | None,
) -> dict[str, tuple[DirectionCandidateTelemetry, DirectionCandidateTelemetry]]:
    direction_started = time.perf_counter()
    controller = runtime.model_aware
    history = controller._feature3_row_history
    if len(history) != runtime.forecaster.history_length or len(history) < 2:
        return {}
    generic_audio, generic_video = controller.generic_correction_gains(decision)
    result: dict[str, tuple[DirectionCandidateTelemetry, DirectionCandidateTelemetry]] = {}

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
        hold_rms = _tensor_rms(actual_rows - latest).clamp_min(_DIRECTION_EPS)

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

        generic_ratio = _tensor_rms(actual_rows - (predicted + generic_gain * delta)) / hold_rms
        exact_ratio = _tensor_rms(actual_rows - (predicted + exact_gain * delta)) / hold_rms
        applied_ratio = _tensor_rms(actual_rows - (predicted + applied_gain * delta)) / hold_rms

        static_started = time.perf_counter()
        static_direction = static_head_metric_direction(delta, head)
        controller._feature3_static_seconds += time.perf_counter() - static_started
        static_used, static_count = _direction_alpha_used(controller, name, "static", decision.confidence)
        static_bound = _radially_bound_direction_tensor(static_used, static_direction, delta)
        static_prediction = predicted + static_bound.correction
        static_ratio = _tensor_rms(actual_rows - static_prediction) / hold_rms
        static_head_hold = _tensor_rms(_static_head_difference(actual_rows, latest, head)).clamp_min(_DIRECTION_EPS)
        static_head_ratio = _tensor_rms(
            _static_head_difference(actual_rows, static_prediction, head)
        ) / static_head_hold
        static_flat = static_direction.reshape(-1)
        residual_flat = residual.reshape(-1)
        static_denom = torch.dot(static_flat, static_flat).clamp_min(
            _DIRECTION_EPS * max(1, static_flat.numel())
        )
        static_instant = torch.dot(residual_flat, static_flat) / static_denom

        full_direction = None
        full_jvp = full_vjp = 0.0
        if geometry is not None and adaln_scale is not None:
            full_started = time.perf_counter()
            full_direction, full_jvp, full_vjp = final_layer_metric_direction(
                predicted_native,
                delta.to(dtype=predicted_native.dtype),
                norm_weight=geometry.norm_weight,
                norm_eps=geometry.norm_eps,
                adaln_scale=adaln_scale,
                head_weight=head,
            )
            full_direction = full_direction.to(torch.float32)
            controller._feature3_full_seconds += time.perf_counter() - full_started
            controller._feature3_jvp_seconds += full_jvp
            controller._feature3_vjp_seconds += full_vjp

        full_used, full_count = _direction_alpha_used(controller, name, "full", decision.confidence)
        if full_direction is None:
            full_bound = _TensorBoundedDirection(
                correction=torch.zeros_like(delta),
                direction_norm_ratio=torch.zeros((), dtype=torch.float32, device=delta.device),
                raw_norm_ratio=torch.zeros((), dtype=torch.float32, device=delta.device),
                radial_scale=torch.ones((), dtype=torch.float32, device=delta.device),
                bounded_norm_ratio=torch.zeros((), dtype=torch.float32, device=delta.device),
                bound_active=torch.zeros((), dtype=torch.bool, device=delta.device),
                eligible=torch.zeros((), dtype=torch.bool, device=delta.device),
            )
            full_prediction = predicted
            full_ratio = torch.zeros((), dtype=torch.float32, device=combined.device)
            full_static_head_ratio = torch.zeros_like(full_ratio)
            full_final_ratio = torch.zeros_like(full_ratio)
            full_instant = torch.zeros_like(full_ratio)
        else:
            full_bound = _radially_bound_direction_tensor(full_used, full_direction, delta)
            full_prediction = predicted + full_bound.correction
            full_ratio = _tensor_rms(actual_rows - full_prediction) / hold_rms
            full_static_head_ratio = _tensor_rms(
                _static_head_difference(actual_rows, full_prediction, head)
            ) / static_head_hold
            final_hold = _tensor_rms(
                _final_layer_difference(
                    actual_native,
                    latest_native,
                    geometry=geometry,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                )
            ).clamp_min(_DIRECTION_EPS)
            full_final_ratio = _tensor_rms(
                _final_layer_difference(
                    actual_native,
                    full_prediction.to(dtype=actual_native.dtype),
                    geometry=geometry,
                    adaln_scale=adaln_scale,
                    head_weight=head,
                )
            ) / final_hold
            full_flat = full_direction.reshape(-1)
            full_denom = torch.dot(full_flat, full_flat).clamp_min(
                _DIRECTION_EPS * max(1, full_flat.numel())
            )
            full_instant = torch.dot(residual_flat, full_flat) / full_denom

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
                static_bound.direction_norm_ratio,
                static_bound.radial_scale,
                static_bound.bounded_norm_ratio,
                static_bound.bound_active.to(torch.float32),
                static_bound.eligible.to(torch.float32),
                full_bound.direction_norm_ratio,
                full_bound.radial_scale,
                full_bound.bounded_norm_ratio,
                full_bound.bound_active.to(torch.float32),
                full_bound.eligible.to(torch.float32),
            )
        ).detach().to(device="cpu").tolist()
        controller._feature3_scalar_transfer_seconds += time.perf_counter() - transfer_started
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
            static_direction_ratio_v,
            static_radial_scale_v,
            static_bounded_ratio_v,
            static_bound_active_v,
            static_bound_eligible_v,
            full_direction_ratio_v,
            full_radial_scale_v,
            full_bounded_ratio_v,
            full_bound_active_v,
            full_bound_eligible_v,
        ) = scalar_values

        static_eligible = bool(
            static_bound_eligible_v and static_count > 0 and math.isfinite(static_ratio_v)
        )
        full_eligible = bool(
            full_bound_eligible_v
            and full_count > 0
            and geometry is not None
            and math.isfinite(full_ratio_v)
        )
        static_telemetry = DirectionCandidateTelemetry(
            eligible=static_eligible,
            geometry_available=True,
            alpha_history_count=static_count,
            direction_norm_ratio=static_direction_ratio_v,
            coefficient_used=static_used,
            instantaneous_coefficient=static_instant_v,
            radial_scale=static_radial_scale_v,
            bounded_norm_ratio=static_bounded_ratio_v,
            bound_active=bool(static_bound_active_v),
            ordinary_ratio=static_ratio_v if static_eligible else 0.0,
            static_head_ratio=static_head_ratio_v if static_eligible else 0.0,
            generic_scalar_ratio=generic_ratio_v,
            exact_scalar_ratio=exact_ratio_v,
            scalar_applied_ratio=applied_ratio_v,
        )
        full_telemetry = DirectionCandidateTelemetry(
            eligible=full_eligible,
            geometry_available=geometry is not None,
            alpha_history_count=full_count,
            direction_norm_ratio=full_direction_ratio_v,
            coefficient_used=full_used,
            instantaneous_coefficient=full_instant_v,
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
            adv_g = _record_comparison(controller, name, "static_vs_generic", generic_ratio_v, static_ratio_v)
            adv_e = _record_comparison(controller, name, "static_vs_exact", exact_ratio_v, static_ratio_v)
            static_telemetry = replace(
                static_telemetry,
                relative_advantage_vs_generic=adv_g,
                relative_advantage_vs_exact=adv_e,
            )
        if full_eligible:
            adv_g = _record_comparison(controller, name, "full_vs_generic", generic_ratio_v, full_ratio_v)
            adv_e = _record_comparison(controller, name, "full_vs_exact", exact_ratio_v, full_ratio_v)
            adv_s = 0.0
            if static_eligible:
                adv_s = _record_comparison(controller, name, "full_vs_static", static_ratio_v, full_ratio_v)
            full_telemetry = replace(
                full_telemetry,
                relative_advantage_vs_generic=adv_g,
                relative_advantage_vs_exact=adv_e,
                relative_advantage_vs_static=adv_s,
            )

        _candidate_stats(controller, name, "static", static_telemetry)
        _candidate_stats(controller, name, "full", full_telemetry)
        _update_direction_alpha(controller, name, "static", static_instant_v)
        if full_direction is not None:
            _update_direction_alpha(controller, name, "full", full_instant_v)
        result[name] = (static_telemetry, full_telemetry)

        workspace = (
            predicted.numel()
            + delta.numel()
            + residual.numel()
            + static_direction.numel()
            + static_prediction.numel()
        ) * torch.tensor([], dtype=torch.float32).element_size()
        if full_direction is not None:
            workspace += (
                full_direction.numel() + full_prediction.numel()
            ) * torch.tensor([], dtype=torch.float32).element_size()
        controller._feature3_workspace_bytes = max(controller._feature3_workspace_bytes, workspace)

    controller._feature3_direction_seconds += time.perf_counter() - direction_started
    return result


def _consume_geometry(runtime: SpectrumH3Runtime, step_id: int) -> FinalLayerGeometry | None:
    pending = getattr(runtime, "_feature3_pending_geometry", None)
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
        if geometry.audio_scale.shape != first.audio_scale.shape or geometry.video_scale.shape != first.video_scale.shape:
            return None
    return first


def _record_geometry(
    runtime: SpectrumH3Runtime,
    step_id: int,
    call_id: int,
    geometry: FinalLayerGeometry,
    elapsed: float,
) -> None:
    pending = getattr(runtime, "_feature3_pending_geometry", None)
    if pending is None:
        runtime._feature3_pending_geometry = {}
        pending = runtime._feature3_pending_geometry
    pending.setdefault(int(step_id), {})[int(call_id)] = geometry
    runtime.model_aware._feature3_geometry_seconds += max(0.0, float(elapsed))


_ORIGINAL_RUNTIME_OBSERVE = SpectrumH3Runtime._observe_model_aware_anchor
_ORIGINAL_RUNTIME_WEIGHT_SEGMENTS = SpectrumH3Runtime._model_aware_weight_segments
_ORIGINAL_FORECASTER_UPDATE = HistoryWeightForecaster.update


def _patched_forecaster_update(
    self: HistoryWeightForecaster,
    coordinate: float,
    feature: torch.Tensor,
    **kwargs,
) -> None:
    """Keep Feature-1/2/3 evidence allocation aligned with runtime mode."""
    mode = getattr(self, "_feature3_evidence_capture_mode", None)
    if mode == "schedule":
        kwargs["evidence_segments"] = None
        kwargs["exact_head_weights"] = None
    elif mode == "schedule_confidence":
        kwargs["exact_head_weights"] = None
    return _ORIGINAL_FORECASTER_UPDATE(self, coordinate, feature, **kwargs)


_ORIGINAL_RUNTIME_START = SpectrumH3Runtime.start_run
_ORIGINAL_RUNTIME_END = SpectrumH3Runtime.end_run
_ORIGINAL_RUNTIME_ABORT = SpectrumH3Runtime.abort_step
_ORIGINAL_RUNTIME_DISABLE = SpectrumH3Runtime._disable_forecasting
_ORIGINAL_RUNTIME_DISABLE_MODEL_AWARE = SpectrumH3Runtime.disable_model_aware
_ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary


def _patched_runtime_weight_segments(
    self: SpectrumH3Runtime,
    call: Any,
    decision: ModelAwareForecastDecision,
    *,
    coordinate: float,
):
    """Feature 3 applied path: retained trust-mixed scalar correction only."""
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
                raise ValueError("packed H3 topology does not expose audio/video correction boundary")
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
        # Feature 1 is profile/prior-only. It intentionally does not train on
        # runtime trajectory evidence or allocate correction evidence.
        return
    geometry = _consume_geometry(self, step.step_id)
    ranges = self._stream_ranges(step.calls[0])
    if mode != "full":
        exact_head_weights = {}
        stream_diagonals = {}
        geometry = None

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
        scalar_result = None
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
            self.model_aware.observe_anchor(evidence, decision)
            self.stats.model_aware_anchor_updates += 1
            self.stats.model_aware_evidence_weight_fit_seconds += evidence.timing.weight_fit_seconds
            self.stats.model_aware_evidence_sample_index_seconds += evidence.timing.sample_index_seconds
            self.stats.model_aware_evidence_device_transfer_seconds += evidence.timing.device_transfer_seconds
            self.stats.model_aware_evidence_scalar_transfer_seconds += evidence.timing.scalar_transfer_seconds
            self.stats.model_aware_evidence_reduction_seconds += evidence.timing.reduction_seconds
            self.stats.model_aware_evidence_exact_head_projection_seconds += evidence.timing.exact_head_projection_seconds
            self.stats.model_aware_exact_head_projection_seconds += evidence.timing.exact_head_projection_seconds
            self.stats.model_aware_exact_head_projection_calls += len(exact_head_weights)
            self.stats.model_aware_evidence_fit_condition_seconds += evidence.timing.fit_condition_seconds
            # Explicit retirement invariant: no K=2 work is charged.
            self.stats.model_aware_subspace_gram_seconds += 0.0
            self.stats.model_aware_subspace_solve_seconds += 0.0
            self.stats.model_aware_subspace_workspace_bytes = max(
                self.stats.model_aware_subspace_workspace_bytes,
                0,
            )
            self.stats.model_aware_model_corrected_ratio_mean = self.model_aware.model_corrected_ratio_mean
            self.stats.model_aware_generic_corrected_ratio_mean = self.model_aware.generic_corrected_ratio_mean

            if mode == "full":
                assert scalar_result is not None
                direction = _evaluate_model_directions(
                    self,
                    step,
                    combined,
                    decision,
                    scalar_result.raw_weights,
                    exact_head_weights,
                    geometry,
                )
                self.model_aware._feature3_last = {
                    name: pair for name, pair in direction.items()
                }
                if self.config.debug:
                    for name, (static, full) in direction.items():
                        _h3.LOG.warning(
                            "Spectrum H3 model-direction step=%s stream=%s "
                            "generic_scalar_ratio=%.6f exact_scalar_ratio=%.6f scalar_applied_ratio=%.6f "
                            "static_eligible=%s static_dir_norm_ratio=%.6f static_alpha=%.6f "
                            "static_bounded_norm_ratio=%.6f static_radial_scale=%.6f "
                            "static_ratio=%.6f static_head_ratio=%.6f static_adv_generic=%.6f "
                            "static_adv_exact=%.6f full_eligible=%s full_geometry=%s "
                            "full_dir_norm_ratio=%.6f full_alpha=%.6f full_bounded_norm_ratio=%.6f "
                            "full_radial_scale=%.6f full_ratio=%.6f full_static_head_ratio=%.6f "
                            "full_final_layer_ratio=%.6f full_adv_generic=%.6f full_adv_exact=%.6f "
                            "full_adv_static=%.6f",
                            step.step_id,
                            name,
                            static.generic_scalar_ratio,
                            static.exact_scalar_ratio,
                            static.scalar_applied_ratio,
                            static.eligible,
                            static.direction_norm_ratio,
                            static.coefficient_used,
                            static.bounded_norm_ratio,
                            static.radial_scale,
                            static.ordinary_ratio,
                            static.static_head_ratio,
                            static.relative_advantage_vs_generic,
                            static.relative_advantage_vs_exact,
                            full.eligible,
                            full.geometry_available,
                            full.direction_norm_ratio,
                            full.coefficient_used,
                            full.bounded_norm_ratio,
                            full.radial_scale,
                            full.ordinary_ratio,
                            full.static_head_ratio,
                            full.final_layer_ratio,
                            full.relative_advantage_vs_generic,
                            full.relative_advantage_vs_exact,
                            full.relative_advantage_vs_static,
                        )
        if mode == "full":
            # Append only after all counterfactuals were scored so anchor N cannot
            # influence the candidate coefficient or predicted state used at N.
            _feature3_retain_current_rows(
                self.model_aware,
                self.forecaster,
                combined,
                ranges,
            )
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        self._model_aware_disabled_reason = f"model-aware evidence failed: {exc}"
        self.stats.model_aware_failures += 1
        _h3.LOG.warning("Spectrum H3 model-aware evidence disabled: %s", self._model_aware_disabled_reason)
    finally:
        elapsed = time.perf_counter() - started
        self.stats.model_aware_evidence_seconds += elapsed
        self.stats.model_aware_overhead_seconds += elapsed


def _patched_runtime_start(self: SpectrumH3Runtime, *args, **kwargs):
    self._feature3_pending_geometry = {}
    run_id = _ORIGINAL_RUNTIME_START(self, *args, **kwargs)
    self.forecaster._feature3_evidence_capture_mode = self.config.model_aware_mode
    return run_id


def _release_feature3_device_state(runtime: SpectrumH3Runtime) -> None:
    runtime._feature3_pending_geometry = {}
    runtime.model_aware._feature3_row_history = []
    runtime.model_aware._feature3_row_indices = {}
    # GPU copies of W/diag are per-run Feature-3 payloads; keep only detached CPU
    # tensors in the profile cache between runs.
    runtime.model_aware._head_device_cache.clear()


def _patched_runtime_end(self: SpectrumH3Runtime, run_id: int) -> None:
    try:
        _ORIGINAL_RUNTIME_END(self, run_id)
    finally:
        _release_feature3_device_state(self)


def _patched_runtime_abort(self: SpectrumH3Runtime, run_id: int, step_id: int) -> None:
    try:
        _ORIGINAL_RUNTIME_ABORT(self, run_id, step_id)
    finally:
        getattr(self, "_feature3_pending_geometry", {}).pop(int(step_id), None)


def _patched_runtime_disable(self: SpectrumH3Runtime, reason: str) -> bool:
    result = _ORIGINAL_RUNTIME_DISABLE(self, reason)
    self.model_aware._feature3_row_history = []
    self.model_aware._feature3_row_indices = {}
    self.model_aware._head_device_cache.clear()
    return result


def _patched_runtime_disable_model_aware(self: SpectrumH3Runtime, reason: str) -> None:
    _ORIGINAL_RUNTIME_DISABLE_MODEL_AWARE(self, reason)
    _release_feature3_device_state(self)


def _summary_comparison(controller: ModelAwareController, stream: str, comparison: str) -> str:
    prefix = f"_feature3_{stream}_{comparison}"
    count = int(getattr(controller, f"{prefix}_count"))
    mean = (
        float(getattr(controller, f"{prefix}_advantage_sum")) / count
        if count
        else 0.0
    )
    return (
        f"{comparison}={getattr(controller, f'{prefix}_wins')}/"
        f"{getattr(controller, f'{prefix}_losses')}"
        f":mean_adv={mean:.6f}"
        f":max_abs={getattr(controller, f'{prefix}_advantage_abs_max'):.6f}"
    )


def _feature3_summary(runtime: SpectrumH3Runtime) -> str:
    c = runtime.model_aware
    parts = [
        "feature3_k2_runtime=retired",
        f"feature3_direction_evidence_bytes={_feature3_evidence_bytes(c)}",
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
            dir_mean = getattr(c, f"{prefix}_direction_ratio_sum") / count if count else 0.0
            bounded_mean = getattr(c, f"{prefix}_bounded_ratio_sum") / count if count else 0.0
            parts.append(
                f"{stream}_{kind}=eligible:{count},fallback:{getattr(c, f'{prefix}_fallback_count')},"
                f"ratio_mean:{ratio_mean:.6f},ratio_max:{getattr(c, f'{prefix}_ratio_max'):.6f},"
                f"dir_norm_mean:{dir_mean:.6f},dir_norm_max:{getattr(c, f'{prefix}_direction_ratio_max'):.6f},"
                f"bounded_mean:{bounded_mean:.6f},bounded_max:{getattr(c, f'{prefix}_bounded_ratio_max'):.6f},"
                f"bound_active:{getattr(c, f'{prefix}_bound_active_count')}"
            )
        for comparison in (
            "static_vs_generic",
            "static_vs_exact",
            "full_vs_generic",
            "full_vs_exact",
            "full_vs_static",
        ):
            parts.append(f"{stream}_{_summary_comparison(c, stream, comparison)}")
    return " ".join(parts)


def _patched_debug_summary(self: SpectrumH3Runtime) -> str:
    return f"{_ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)} {_feature3_summary(self)}"


def _timestep_row(
    mod_segments: Any,
    start: int,
    end: int,
    modality_tag: int,
) -> int:
    matches = [
        int(row) // 3
        for a, b, row in mod_segments
        if int(a) == int(start)
        and int(b) == int(end)
        and int(row) % 3 == int(modality_tag)
    ]
    if len(matches) != 1:
        raise RuntimeError("could not resolve native FinalLayer target timestep row")
    return matches[0]


def _capture_geometry_from_block_args(
    inner: Any,
    runtime: SpectrumH3Runtime,
    step_id: int,
    call_id: int,
    layout: Any,
    args: dict[str, Any],
) -> None:
    started = time.perf_counter()
    t_emb = args.get("t_emb")
    mod_segments = args.get("mod_segments")
    hidden = args.get("img")
    if not torch.is_tensor(t_emb) or mod_segments is None or not torch.is_tensor(hidden):
        raise RuntimeError("native final-block wrapper did not expose t_emb/mod_segments/img")
    (aa, ab), (va, vb) = _h3.target_segments(layout)
    audio_row = _timestep_row(mod_segments, aa, ab, 2)
    video_row = _timestep_row(mod_segments, va, vb, 0)
    shift, scale = inner.final_layer.adaln_proj(t_emb)
    del shift
    if scale.ndim != 2 or audio_row >= scale.shape[0] or video_row >= scale.shape[0]:
        raise RuntimeError("native FinalLayer AdaLN projection shape is incompatible")
    norm = inner.final_layer.norm
    eps = float(norm.eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise RuntimeError("native FinalLayer RMSNorm epsilon is invalid")
    import comfy.ops

    probe = torch.empty((), device=hidden.device, dtype=hidden.dtype)
    if getattr(norm, "weight", None) is None:
        norm_weight = torch.ones(
            int(hidden.shape[-1]),
            device=hidden.device,
            dtype=hidden.dtype,
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
    geometry = FinalLayerGeometry(
        norm_weight=norm_weight,
        norm_eps=eps,
        audio_scale=scale[audio_row].detach().clone(),
        video_scale=scale[video_row].detach().clone(),
    )
    _record_geometry(runtime, step_id, call_id, geometry, time.perf_counter() - started)


_ORIGINAL_EXECUTE_ACTUAL = _h3._execute_actual


def _patched_execute_actual(
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
    """Native actual path with causal FinalLayer geometry captured at the last block."""
    if len(inner.blocks) == 0:
        raise RuntimeError("native MiniMax H3 has no transformer blocks to observe")
    last_index = len(inner.blocks) - 1
    local_options = dict(transformer_options)
    patches_replace = dict(local_options.get("patches_replace") or {})
    dit_replacements = dict(patches_replace.get("dit") or {})
    patches_replace["dit"] = dit_replacements
    local_options["patches_replace"] = patches_replace
    existing = dit_replacements.get(("double_block", last_index))
    observed = False
    actual_target = None

    def capture_replacement(args, replacement_context):
        nonlocal actual_target, observed
        output = (
            existing(args, replacement_context)
            if existing is not None
            else replacement_context["original_block"](args)
        )
        if not isinstance(output, dict) or "img" not in output or not torch.is_tensor(output["img"]):
            raise RuntimeError("final MiniMax H3 block replacement did not return {'img': tensor}")
        (aa, _), (_, vb) = _h3.target_segments(layout)
        hidden = output["img"]
        if hidden.ndim != 2 or hidden.shape[0] < vb:
            raise RuntimeError("final MiniMax H3 hidden feature is incompatible with the packed layout")
        target = hidden[aa:vb].unsqueeze(0)
        actual_target = target
        runtime.observe_actual(run_id, step_id, call_id, target)
        observed = True
        if runtime.config.model_aware_mode == "full" and runtime._model_aware_enabled():
            try:
                geometry_args = dict(args)
                geometry_args["img"] = hidden
                _capture_geometry_from_block_args(
                    inner,
                    runtime,
                    step_id,
                    call_id,
                    layout,
                    geometry_args,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                runtime.model_aware._feature3_geometry_failures += 1
                _h3.LOG.warning(
                    "Spectrum H3 FinalLayer direction geometry unavailable at step=%s call=%s: %s",
                    step_id,
                    call_id,
                    exc,
                )
        return output

    dit_replacements[("double_block", last_index)] = capture_replacement
    result = executor(
        x,
        timestep,
        context,
        local_options,
        minimax_payload=minimax_payload,
        **kwargs,
    )
    if not observed:
        raise RuntimeError("native MiniMax H3 final transformer block was not executed")
    if residual_probe is not None and actual_target is not None:
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
            output_head_started = time.perf_counter()
            try:
                shadow_output = _h3._execute_forecast(
                    inner,
                    residual_probe.shadow,
                    state,
                    x[0],
                    x[1],
                )
                hold_output = _h3._execute_forecast(
                    inner,
                    residual_probe.hold,
                    state,
                    x[0],
                    x[1],
                )
            finally:
                runtime.record_residual_output_head_seconds(
                    time.perf_counter() - output_head_started
                )
            runtime.record_residual_measurement(
                run_id,
                step_id,
                call_id,
                residual_probe,
                actual_feature=actual_target,
                actual_output=result,
                shadow_output=shadow_output,
                hold_output=hold_output,
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            runtime.disable_experiment(f"residual output-head evaluation failed: {exc}")
    return result


def install_feature3_direction_experiment() -> None:
    """Install the scoped Feature-3 runtime revision once per interpreter."""
    if getattr(ModelAwareController, "_feature3_direction_installed", False):
        return
    ModelAwareController.reset = _patched_controller_reset
    ModelAwareController.snapshot = _patched_controller_snapshot
    ModelAwareController.restore = _patched_controller_restore
    ModelAwareController._feature3_direction_installed = True

    HistoryWeightForecaster.update = _patched_forecaster_update

    SpectrumH3Runtime._model_aware_weight_segments = _patched_runtime_weight_segments
    SpectrumH3Runtime._observe_model_aware_anchor = _patched_runtime_observe
    SpectrumH3Runtime.start_run = _patched_runtime_start
    SpectrumH3Runtime.end_run = _patched_runtime_end
    SpectrumH3Runtime.abort_step = _patched_runtime_abort
    SpectrumH3Runtime._disable_forecasting = _patched_runtime_disable
    SpectrumH3Runtime.disable_model_aware = _patched_runtime_disable_model_aware
    SpectrumH3Runtime.debug_summary = _patched_debug_summary

    _h3._execute_actual = _patched_execute_actual


__all__ = [
    "BoundedDirection",
    "DirectionCandidateTelemetry",
    "FinalLayerGeometry",
    "final_layer_jvp",
    "final_layer_metric_direction",
    "final_layer_vjp",
    "install_feature3_direction_experiment",
    "radially_bound_direction",
    "rmsnorm_jvp",
    "rmsnorm_vjp",
    "static_head_metric_direction",
]
