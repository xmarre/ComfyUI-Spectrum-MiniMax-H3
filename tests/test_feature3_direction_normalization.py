from __future__ import annotations

import math

import pytest
import torch

import comfyui_spectrum_h3.feature3_direction as feature3
from comfyui_spectrum_h3.feature3_direction_normalization import (
    normalize_direction_to_reference,
)


def test_tiny_positive_radial_ratio_is_not_spuriously_suppressed():
    delta = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    bounded = feature3.radially_bound_direction(1e-12, delta, delta)
    assert bounded.eligible
    assert bounded.raw_norm_ratio == pytest.approx(1e-12, rel=2e-5, abs=1e-15)
    assert bounded.radial_scale == pytest.approx(1.0, rel=0.0, abs=2e-7)
    assert bounded.bounded_norm_ratio == pytest.approx(
        bounded.raw_norm_ratio, rel=2e-5, abs=1e-15
    )
    assert not bounded.bound_active


@pytest.mark.parametrize("q", (1e-12, 1e-8, 1e-4, 0.01, 0.1, 0.25, 1.0, 100.0))
def test_radial_bound_matches_exact_rational_scale(q):
    delta = torch.tensor([[1.0, 2.0, -3.0]], dtype=torch.float32)
    bounded = feature3.radially_bound_direction(q, delta, delta)
    expected_scale = 1.0 / (1.0 + q / 0.25)
    expected_bounded = q * expected_scale
    assert bounded.eligible
    assert bounded.radial_scale == pytest.approx(expected_scale, rel=2e-6, abs=2e-7)
    assert bounded.bounded_norm_ratio == pytest.approx(
        expected_bounded, rel=2e-6, abs=2e-12
    )
    assert bounded.bounded_norm_ratio <= bounded.raw_norm_ratio + 1e-7


def test_direction_normalization_matches_reference_norm():
    torch.manual_seed(101)
    direction = torch.randn(3, 7)
    delta = torch.randn(3, 7)
    normalized = normalize_direction_to_reference(direction, delta)
    assert normalized.eligible
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized.direction),
        torch.linalg.vector_norm(delta),
        rtol=2e-6,
        atol=2e-6,
    )
    assert normalized.normalized_direction_norm_ratio == pytest.approx(
        1.0, rel=2e-6, abs=2e-6
    )


@pytest.mark.parametrize("scale", (1e-12, 1e-6, 0.01, 1.0, 100.0, 1e6))
def test_direction_normalization_is_invariant_to_positive_scaling(scale):
    direction = torch.tensor([[0.5, -1.25, 2.0, 0.75]], dtype=torch.float32)
    delta = torch.tensor([[2.0, 1.0, -0.5, 1.5]], dtype=torch.float32)
    reference = normalize_direction_to_reference(direction, delta)
    scaled = normalize_direction_to_reference(scale * direction, delta)
    assert reference.eligible and scaled.eligible
    torch.testing.assert_close(
        scaled.direction,
        reference.direction,
        rtol=3e-5,
        atol=3e-6,
    )
    assert scaled.normalized_direction_norm_ratio == pytest.approx(
        reference.normalized_direction_norm_ratio, rel=3e-5, abs=3e-6
    )


@pytest.mark.parametrize("scale", (1e-12, 1e-6, 0.01, 1.0, 100.0, 1e6))
def test_candidate_correction_is_invariant_to_positive_direction_scaling(scale):
    direction = torch.tensor([[0.5, -1.25, 2.0, 0.75]], dtype=torch.float32)
    delta = torch.tensor([[2.0, 1.0, -0.5, 1.5]], dtype=torch.float32)
    alpha = -0.4
    reference = normalize_direction_to_reference(direction, delta)
    scaled = normalize_direction_to_reference(scale * direction, delta)
    reference_bound = feature3.radially_bound_direction(
        alpha, reference.direction, delta
    )
    scaled_bound = feature3.radially_bound_direction(alpha, scaled.direction, delta)
    assert reference_bound.eligible and scaled_bound.eligible
    torch.testing.assert_close(
        scaled_bound.correction,
        reference_bound.correction,
        rtol=3e-5,
        atol=3e-6,
    )
    assert scaled_bound.raw_norm_ratio == pytest.approx(
        reference_bound.raw_norm_ratio, rel=3e-5, abs=3e-6
    )
    assert scaled_bound.radial_scale == pytest.approx(
        reference_bound.radial_scale, rel=3e-5, abs=3e-6
    )


def test_full_jtj_tiny_raw_scale_normalizes_before_budgeting():
    torch.manual_seed(102)
    hidden, out = 6, 3
    x = torch.randn(2, hidden, dtype=torch.float32)
    delta = torch.randn_like(x)
    norm_weight = torch.ones(hidden, dtype=torch.float32)
    adaln_scale = torch.linspace(-0.2, 0.2, hidden, dtype=torch.float32)
    # J^T J scales quadratically with the head. A 1e-4 head deliberately makes
    # the raw operator magnitude tiny while preserving a finite orientation.
    head = 1e-4 * torch.randn(out, hidden, dtype=torch.float32)
    raw, _, _ = feature3.final_layer_metric_direction(
        x,
        delta,
        norm_weight=norm_weight,
        norm_eps=1e-5,
        adaln_scale=adaln_scale,
        head_weight=head,
    )
    raw_ratio = (
        torch.linalg.vector_norm(raw.to(torch.float32))
        / torch.linalg.vector_norm(delta)
    ).item()
    assert 0.0 < raw_ratio < 1e-5

    normalized = normalize_direction_to_reference(raw, delta)
    assert normalized.eligible
    assert normalized.raw_direction_norm_ratio == pytest.approx(
        raw_ratio, rel=5e-5, abs=1e-15
    )
    assert normalized.normalized_direction_norm_ratio == pytest.approx(
        1.0, rel=3e-5, abs=3e-5
    )

    bounded = feature3.radially_bound_direction(0.2, normalized.direction, delta)
    expected_scale = 1.0 / (1.0 + 0.2 / 0.25)
    assert bounded.eligible
    assert bounded.raw_norm_ratio == pytest.approx(0.2, rel=3e-5, abs=3e-5)
    assert bounded.radial_scale == pytest.approx(
        expected_scale, rel=3e-5, abs=3e-5
    )
    assert bounded.radial_scale > 0.5


@pytest.mark.parametrize(
    "direction",
    (
        torch.zeros(1, 4),
        torch.tensor([[float("nan"), 0.0, 0.0, 1.0]]),
        torch.tensor([[float("inf"), 0.0, 0.0, 1.0]]),
    ),
)
def test_normalization_zero_or_nonfinite_direction_fails_closed(direction):
    normalized = normalize_direction_to_reference(direction, torch.ones_like(direction))
    assert not normalized.eligible
    assert torch.equal(normalized.direction, torch.zeros_like(normalized.direction))


def test_normalization_tiny_reference_delta_fails_closed():
    direction = torch.ones(1, 4)
    delta = torch.full((1, 4), torch.finfo(torch.float32).tiny)
    normalized = normalize_direction_to_reference(direction, delta)
    assert not normalized.eligible
    assert torch.equal(normalized.direction, torch.zeros_like(normalized.direction))


def test_invalid_radial_limit_fails_closed_without_increasing_correction():
    delta = torch.ones(1, 4)
    bounded = feature3.radially_bound_direction(0.5, delta, delta, limit=0.0)
    assert not bounded.eligible
    assert bounded.bounded_norm_ratio == 0.0
    assert math.isclose(bounded.radial_scale, 1.0)
    assert torch.equal(bounded.correction, torch.zeros_like(bounded.correction))
