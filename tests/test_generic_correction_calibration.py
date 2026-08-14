from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import generic_correction_calibration as calibration
from comfyui_spectrum_h3.forecast import HistoryWeightForecaster
from comfyui_spectrum_h3.generic_correction_calibration import (
    GenericCalibrationState,
    build_block,
    exact_segment_moments,
)
from comfyui_spectrum_h3.generic_correction_core import combine_moments
from comfyui_spectrum_h3.generic_correction_topology import temporal_audio_bands


def test_exact_segment_moments_match_direct_tensor_calculation():
    forecaster = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.1,
        max_history=4,
        history_storage="system_ram",
    )
    previous = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])
    latest = previous + 2.0
    actual = previous + torch.tensor([[[3.0, 2.0], [5.0, 4.0], [7.0, 6.0]]])
    forecaster.update(-1.0, previous, anchor_id=0)
    forecaster.update(0.0, latest, anchor_id=1)
    weights = torch.tensor([-0.5, 1.5])
    moments = exact_segment_moments(
        forecaster,
        actual,
        (("video", 1, 3, weights),),
    )["video"]

    predicted = weights[0] * previous[:, 1:] + weights[1] * latest[:, 1:]
    residual = actual[:, 1:] - predicted
    direction = latest[:, 1:] - previous[:, 1:]
    hold = actual[:, 1:] - latest[:, 1:]
    assert moments.residual_sq_mean == pytest.approx(
        residual.float().square().mean().item()
    )
    assert moments.residual_dot_direction_mean == pytest.approx(
        (residual.float() * direction.float()).mean().item()
    )
    assert moments.direction_sq_mean == pytest.approx(
        direction.float().square().mean().item()
    )
    assert moments.hold_error_sq_mean == pytest.approx(
        hold.float().square().mean().item()
    )


def test_audio_start_middle_end_moments_reconstruct_aggregate_exactly():
    forecaster = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.1,
        max_history=4,
        history_storage="system_ram",
    )
    previous = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
    latest = previous + torch.linspace(0.5, 2.0, 24).reshape(1, 12, 2)
    actual = latest + torch.linspace(-1.0, 1.0, 24).reshape(1, 12, 2)
    forecaster.update(-1.0, previous, anchor_id=0)
    forecaster.update(0.0, latest, anchor_id=1)
    weights = torch.tensor([-0.25, 1.25])
    topology = (("audio_shape", (1, 32, 2, 6)), ("target_audio_rows", 12))
    bands = temporal_audio_bands(topology, audio_start_row=0, audio_end_row=12)
    assert bands is not None
    segments = [("audio", 0, 12, weights)]
    for band in bands:
        segments.extend(
            (
                f"{band.band_id}:{channel}",
                start,
                end,
                weights,
            )
            for channel, (start, end) in enumerate(band.row_ranges)
        )
    exact = exact_segment_moments(forecaster, actual, segments)
    reconstructed = combine_moments(
        [
            combine_moments(
                [exact[f"{band.band_id}:0"], exact[f"{band.band_id}:1"]]
            )
            for band in bands
        ]
    )
    aggregate = exact["audio"]
    assert reconstructed.sample_count == aggregate.sample_count
    for name in (
        "residual_sq_mean",
        "residual_dot_direction_mean",
        "direction_sq_mean",
        "hold_error_sq_mean",
        "actual_sq_mean",
    ):
        assert getattr(reconstructed, name) == pytest.approx(
            getattr(aggregate, name), rel=1e-12, abs=1e-12
        )


def test_calibration_block_contains_only_scalar_rows_and_stable_provenance():
    row = {
        "schema_version": 1,
        "target_step_id": 2,
        "stream": "video",
        "region_id": None,
        "sample_count": 4,
        "A": 1.0,
        "B": 0.2,
        "C": 0.5,
    }
    state = GenericCalibrationState(
        enabled=True,
        run_id=1,
        sampler_name="sample_euler",
        total_steps=3,
        schedule=(1.0, 0.5, 0.0, 0.0),
        config_snapshot={"generic_correction_mode": "legacy"},
        rows=[row],
        topology=(("target_video_rows", 2),),
    )
    runtime = SimpleNamespace(_spectrum_h3_observed_seed=123)
    block = build_block(runtime, state)
    rebuilt = build_block(runtime, state)
    assert block["provenance"]["seed"] == 123
    assert block["provenance"]["config_hash"]
    assert block["provenance"]["trace_fingerprint"]
    assert block["target_rows"][0]["trace_fingerprint"]
    assert rebuilt == block
    assert "trace_fingerprint" not in row
    assert not any(torch.is_tensor(value) for value in block["target_rows"][0].values())


def test_row_retention_and_serialization_fail_closed(monkeypatch):
    state = GenericCalibrationState(
        enabled=True,
        run_id=1,
        sampler_name="sample_euler",
        total_steps=3,
        schedule=(1.0, 0.5, 0.0),
        config_snapshot={},
    )
    topology = (("target_video_rows", 2),)
    monkeypatch.setattr(calibration, "MAX_RETAINED_ROWS", 1)
    calibration.record_row(state, {"value": 1.0}, topology=topology)
    calibration.record_row(state, {"value": 2.0}, topology=topology)
    assert state.rows == [{"value": 1.0}]
    assert state.failures == 1

    separate = GenericCalibrationState(
        enabled=True,
        run_id=2,
        sampler_name="sample_euler",
        total_steps=3,
        schedule=(1.0, 0.5, 0.0),
        config_snapshot={},
    )
    calibration.record_row(separate, {"bad": object()}, topology=topology)
    assert separate.rows == []
    assert separate.failures == 1
