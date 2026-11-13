from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.forecast import HistoryWeightForecaster


@pytest.fixture(params=["cpu", "cuda"])
def target_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this test environment")
    return torch.device(request.param)


def _supported(dtype, device):
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")


def _features(dtype, device, shift=0):
    # Non-integral values exercise FP16/BF16 conversions.
    return (torch.arange(3 * 40 * 30, dtype=torch.float32)
            .reshape(3, 40, 30).mul_(0.003).add_(shift)).to(device=device, dtype=dtype)


def _forecaster(history_storage):
    return HistoryWeightForecaster(
        degree=1, max_history=4, chunk_bytes=4096, history_storage=history_storage
    )


def _legacy_weighted_segment(sources, indices, start, end, weights, output_dtype, device):
    """Reference for the pre-optimization FP32 accumulator, using original add_ order."""
    blocks = []
    for index in indices:
        acc = torch.zeros((end - start, sources[0].shape[-1]), dtype=torch.float32, device=device)
        for weight, source in zip(weights, sources, strict=True):
            if float(weight) != 0.0:
                acc.add_(source[index, start:end].to(device=device, dtype=torch.float32),
                         alpha=float(weight))
        blocks.append(acc.to(dtype=output_dtype))
    return torch.stack(blocks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("history_storage", ["system_ram", "vram"])
def test_one_point_exact_identity_and_reordered_rows(dtype, history_storage, target_device):
    _supported(dtype, target_device)
    source = _features(dtype, target_device)
    forecaster = _forecaster(history_storage)
    forecaster.update(0.0, source)
    indices = [2, 0, 2, 1]
    result = forecaster.predict_one_point_hold(rows=indices, device=target_device, dtype=dtype)
    assert torch.equal(result, source[indices])
    assert result.data_ptr() != forecaster._history[0].feature_flat.data_ptr()
    assert forecaster.last_prediction_chunk_count >= len(indices)
    assert forecaster.last_prediction_max_fp32_elements == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("history_storage", ["system_ram", "vram"])
def test_latest_hold_uses_last_anchor_without_accumulator(dtype, history_storage, target_device):
    _supported(dtype, target_device)
    first, last = _features(dtype, target_device), _features(dtype, target_device, shift=1.75)
    forecaster = _forecaster(history_storage)
    forecaster.update(0.0, first)
    forecaster.update(1.0, last)
    indices = [1, 2, 0]
    actual = forecaster.predict_latest_hold(rows=indices, device=target_device, dtype=dtype)
    assert torch.equal(actual, last[indices])
    assert forecaster.last_prediction_max_fp32_elements == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("history_storage", ["system_ram", "vram"])
def test_segmented_identity_and_weighted_forecasts_match_preoptimization(
    dtype, history_storage, target_device
):
    _supported(dtype, target_device)
    sources = [_features(dtype, target_device), _features(dtype, target_device, shift=2.25)]
    forecaster = _forecaster(history_storage)
    for index, source in enumerate(sources):
        forecaster.update(float(index), source)
    segments = [
        (0, 12, torch.tensor([1.0, 0.0])),
        (12, 25, torch.tensor([0.25, 0.75])),
        (25, 40, torch.tensor([0.0, 1.0])),
    ]
    indices = [2, 0, 2]
    actual = forecaster.predict_with_segment_weights(
        segments, rows=indices, device=target_device, dtype=dtype
    )
    expected = torch.cat(
        [
            _legacy_weighted_segment(
                sources, indices, start, end, weights, dtype, target_device
            )
            for start, end, weights in segments
        ], dim=1
    )
    assert torch.equal(actual, expected)
    assert forecaster.last_prediction_max_fp32_elements > 0  # weighted segment
    assert forecaster.last_prediction_chunk_count >= 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_converting_hold_preserves_fp32_accumulator(dtype, target_device):
    _supported(dtype, target_device)
    source = _features(dtype, target_device)
    forecaster = _forecaster("system_ram")
    forecaster.update(0.0, source)
    actual = forecaster.predict_one_point_hold(
        rows=[2, 0], device=target_device, dtype=torch.float32
    )
    assert torch.equal(actual, source[[2, 0]].float())
    assert forecaster.last_prediction_max_fp32_elements > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_weighted_two_anchor_results_unchanged(dtype, target_device):
    _supported(dtype, target_device)
    sources = [_features(dtype, target_device), _features(dtype, target_device, shift=2.25)]
    forecaster = _forecaster("system_ram")
    for index, source in enumerate(sources):
        forecaster.update(float(index), source)
    weights = torch.tensor([0.125, 0.875], dtype=torch.float32)
    result = forecaster.predict_with_weights(
        weights, rows=[2, 0], device=target_device, dtype=dtype
    )
    expected = _legacy_weighted_segment(
        sources, [2, 0], 0, 40, weights, dtype, target_device
    )
    assert torch.equal(result, expected)
    assert forecaster.last_prediction_max_fp32_elements > 0
