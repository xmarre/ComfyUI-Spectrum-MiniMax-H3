from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.forecast import HistoryWeightForecaster


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_same_dtype_one_point_hold_is_exact_and_uses_no_fp32_accumulator(dtype):
    source = torch.arange(2 * 40 * 30, dtype=torch.float32).reshape(2, 40, 30).to(dtype)
    forecaster = HistoryWeightForecaster(degree=1, max_history=3, chunk_bytes=4096)
    forecaster.update(0.0, source)
    predicted = forecaster.predict_one_point_hold(rows=(1, 0), dtype=dtype)
    assert torch.equal(predicted, source[[1, 0]])
    assert predicted.data_ptr() != forecaster._history[0].feature_flat.data_ptr()
    assert forecaster.last_prediction_chunk_count >= 2
    assert forecaster.last_prediction_max_fp32_elements == 0


def test_dtype_changing_hold_preserves_existing_conversion_path():
    source = torch.arange(2 * 40 * 30, dtype=torch.float32).reshape(2, 40, 30).half()
    forecaster = HistoryWeightForecaster(degree=1, max_history=3, chunk_bytes=4096)
    forecaster.update(0.0, source)
    predicted = forecaster.predict_one_point_hold(dtype=torch.float32)
    assert torch.equal(predicted, source.float())
    assert forecaster.last_prediction_max_fp32_elements > 0


def test_multi_anchor_forecast_still_uses_weighted_accumulation():
    forecaster = HistoryWeightForecaster(degree=1, max_history=3, chunk_bytes=4096)
    for i in range(2):
        forecaster.update(float(i), torch.full((1, 40, 30), float(i), dtype=torch.float32))
    prediction = forecaster.predict(1.5, blend_weight=0.5)
    assert torch.isfinite(prediction).all()
    assert forecaster.last_prediction_max_fp32_elements > 0
