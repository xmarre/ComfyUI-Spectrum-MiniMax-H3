from __future__ import annotations

import math

import pytest

from comfyui_spectrum_h3.generic_correction_controller import (
    GenericCorrectionController,
)
from comfyui_spectrum_h3.generic_correction_core import (
    GainState,
    RecursiveLeastSquares,
    ScalarMoments,
    apply_gain,
    coordinate_transport_scale,
    limit_gain,
    resolve_attenuation_policy,
)
from comfyui_spectrum_h3.generic_correction_topology import (
    temporal_audio_bands,
    temporal_video_regions,
)


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ((0.0, 1.0, 2.0), 1.0),
        ((2.0, 1.0, 0.0), 1.0),
        ((0.0, 0.25, 1.0), 3.0),
        ((1.0, 0.5, 0.75), -0.5),
    ],
)
def test_coordinate_transport_preserves_signed_nonuniform_spacing(
    coordinates,
    expected,
):
    scale, active = coordinate_transport_scale(*coordinates)
    assert active
    assert scale == pytest.approx(expected)


def test_coordinate_transport_falls_back_for_duplicate_or_nonfinite_spacing():
    assert coordinate_transport_scale(1.0, 1.0, 0.0) == (1.0, False)
    assert coordinate_transport_scale(0.0, float("nan"), 1.0) == (1.0, False)


def _moments() -> ScalarMoments:
    return ScalarMoments(
        sample_count=4,
        residual_sq_mean=4.0,
        residual_dot_direction_mean=2.0,
        direction_sq_mean=1.0,
        hold_error_sq_mean=1.0,
        actual_sq_mean=9.0,
        ratio_epsilon=3.0e-6,
        ratio_denominator_rms=1.0,
    ).validate()


def test_exact_quadratic_reconstruction_and_transport():
    moments = _moments()
    assert moments.oracle_gain == pytest.approx(2.0)
    assert moments.mse(0.5) == pytest.approx(2.25)
    assert moments.ratio(0.5) == pytest.approx(1.5)
    transported = moments.transported(-2.0)
    assert transported.residual_dot_direction_mean == pytest.approx(-4.0)
    assert transported.direction_sq_mean == pytest.approx(4.0)
    assert transported.oracle_gain == pytest.approx(-1.0)
    assert transported.mse(-0.25) == pytest.approx(moments.mse(0.5))


def test_rls_uses_direction_energy_and_converges_to_known_gain():
    estimator = RecursiveLeastSquares(forgetting=1.0)
    for energy in (0.01, 1.0, 4.0, 0.25):
        estimator.update(1.75 * energy, energy)
    assert estimator.gain == pytest.approx(1.75)
    assert estimator.observations == 4
    assert estimator.support == 1.0


def test_rls_forgetting_changes_recent_observation_weight():
    slow = RecursiveLeastSquares(forgetting=1.0)
    fast = RecursiveLeastSquares(forgetting=0.5)
    for estimator in (slow, fast):
        estimator.update(1.0, 1.0)
        estimator.update(3.0, 1.0)
    assert fast.gain > slow.gain


def test_reliability_rewards_stable_aligned_success_and_rejects_no_support():
    stable = GainState()
    weak = GainState()
    aligned = _moments()
    degenerate = ScalarMoments(
        sample_count=4,
        residual_sq_mean=4.0,
        residual_dot_direction_mean=0.0,
        direction_sq_mean=0.0,
        hold_error_sq_mean=1.0,
        actual_sq_mean=9.0,
        ratio_epsilon=3.0e-6,
        ratio_denominator_rms=1.0,
    ).validate()
    for _ in range(4):
        stable.reliability.update(aligned, 1.5)
        stable.rls.update(
            aligned.residual_dot_direction_mean, aligned.direction_sq_mean
        )
        weak.reliability.update(degenerate, 0.0)
        weak.rls.update(0.0, 0.0)
    assert stable.reliability.value > weak.reliability.value
    assert stable.reliability.support == 1.0
    assert weak.reliability.support == 0.0


def test_limiters_preserve_sign_and_current_rational_definition():
    assert limit_gain(0.25, "rational", 0.25) == pytest.approx(0.125)
    assert limit_gain(-0.5, "hard_clip", 0.25) == pytest.approx(-0.25)
    assert limit_gain(0.25, "tanh", 0.25) == pytest.approx(0.25 * math.tanh(1.0))


def test_controller_snapshot_restores_recursive_and_reliability_state():
    controller = GenericCorrectionController(
        "coordinate_rls_reliability",
        "tanh",
        0.4,
    )
    controller.observe_stream("audio", _moments(), 0.2)
    before = controller.application("audio", general_confidence=0.8)
    restored = GenericCorrectionController.from_snapshot(controller.snapshot())
    after = restored.application("audio", general_confidence=0.8)
    assert after == before


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("coordinate_rls", "general_confidence"),
        ("coordinate_rls_reliability", "combined_conservative"),
        ("regional", "combined_conservative"),
    ),
)
def test_mode_default_preserves_existing_advanced_attenuation(mode, expected):
    assert resolve_attenuation_policy(mode, "mode_default") == expected
    default = GenericCorrectionController(mode, "hard_clip", 0.4)
    explicit = GenericCorrectionController(
        mode,
        "hard_clip",
        0.4,
        attenuation=expected,
    )
    for controller in (default, explicit):
        controller.observe_stream("audio", _moments(), 0.2)
    assert default.application("audio", general_confidence=0.8) == explicit.application(
        "audio", general_confidence=0.8
    )


def test_explicit_no_attenuation_applies_raw_gain_before_limiter():
    controller = GenericCorrectionController(
        "coordinate_rls",
        "hard_clip",
        0.4,
        attenuation="no_attenuation",
    )
    controller.audio.rls.update(0.3, 0.5)
    application = controller.application("audio", general_confidence=0.1)
    assert application.raw_gain == pytest.approx(0.6)
    assert application.scaled_gain == pytest.approx(0.6)
    assert application.bounded_gain == pytest.approx(0.4)
    assert controller.resolved_attenuation == "no_attenuation"


def test_temporal_regions_follow_native_t_h_w_flattening():
    topology = (
        ("video_shape", (1, 24, 8, 6, 10)),
        ("video_padded", (8, 6, 10)),
        ("patch_size", (1, 2, 2)),
        ("target_video_rows", 8 * 3 * 5),
    )
    regions = temporal_video_regions(
        topology,
        video_start_row=20,
        video_end_row=140,
        requested_regions=4,
    )
    assert regions is not None
    assert [(item.start_row, item.end_row) for item in regions] == [
        (20, 50),
        (50, 80),
        (80, 110),
        (110, 140),
    ]


def test_audio_temporal_bands_follow_native_channel_major_stereo_mapping():
    topology = (
        ("audio_shape", (1, 32, 2, 160)),
        ("target_audio_rows", 320),
    )
    bands = temporal_audio_bands(
        topology,
        audio_start_row=10,
        audio_end_row=330,
    )
    assert bands is not None
    assert [band.band_id for band in bands] == [
        "audio_start",
        "audio_middle",
        "audio_end",
    ]
    assert [band.row_ranges for band in bands] == [
        ((10, 50), (170, 210)),
        ((50, 130), (210, 290)),
        ((130, 170), (290, 330)),
    ]


@pytest.mark.parametrize(
    "topology",
    [
        (("target_video_rows", 120),),
        (
            ("video_shape", (1, 24, 8, 6, 10)),
            ("video_padded", (8, 6, 10)),
            ("patch_size", (1, 2, 2)),
            ("target_video_rows", 119),
        ),
    ],
)
def test_temporal_regions_fail_closed_when_topology_is_unproven(topology):
    assert (
        temporal_video_regions(
            topology,
            video_start_row=20,
            video_end_row=140,
        )
        is None
    )


def test_causal_candidate_is_unchanged_by_future_labels():
    state = GainState()
    state.rls.update(2.0, 1.0)
    before = apply_gain(
        state,
        general_confidence=0.8,
        use_general_confidence=True,
        use_reliability=False,
        limiter="rational",
        limit=0.25,
    )
    future_a = _moments()
    future_b = future_a.transported(-3.0)
    assert future_a.oracle_gain != future_b.oracle_gain
    after = apply_gain(
        state,
        general_confidence=0.8,
        use_general_confidence=True,
        use_reliability=False,
        limiter="rational",
        limit=0.25,
    )
    assert after == before
