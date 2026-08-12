from __future__ import annotations

import math
from dataclasses import dataclass

import torch

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
            torch.isfinite(normalized_ratio),
            normalized_ratio,
            zero,
        ),
        eligible=eligible,
    )


def normalize_direction_to_reference(
    direction: torch.Tensor,
    reference_delta: torch.Tensor,
) -> NormalizedDirection:
    """Public diagnostic wrapper for delta-equivalent direction normalization."""
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


__all__ = [
    "NormalizedDirection",
    "normalize_direction_to_reference",
]
