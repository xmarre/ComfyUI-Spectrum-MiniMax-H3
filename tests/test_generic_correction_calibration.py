from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.forecast import HistoryWeightForecaster
from comfyui_spectrum_h3.generic_correction_calibration import (
    GenericCalibrationState,
    build_block,
    exact_segment_moments,
)


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
    assert block["provenance"]["seed"] == 123
    assert block["provenance"]["config_hash"]
    assert block["provenance"]["trace_fingerprint"]
    assert block["target_rows"][0]["trace_fingerprint"]
    assert not any(torch.is_tensor(value) for value in block["target_rows"][0].values())
