from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.forecast import HistoryWeightForecaster


def _direct_prediction(coords, features, target, degree, ridge_lambda, blend):
    design = HistoryWeightForecaster.chebyshev_design(coords, degree)
    phi = HistoryWeightForecaster.chebyshev_design(torch.tensor([target]), degree)
    flat = features.to(torch.float32).reshape(features.shape[0], -1)
    lhs = design.T @ design + ridge_lambda * torch.eye(degree + 1)
    coefficients = torch.linalg.solve(lhs, design.T @ flat)
    spectral = (phi @ coefficients).reshape(features.shape[1:])
    spacing = float(coords[-1] - coords[-2])
    ratio = (target - float(coords[-1])) / spacing
    linear = features[-1].float() + ratio * (features[-1].float() - features[-2].float())
    return blend * spectral + (1.0 - blend) * linear


@pytest.mark.parametrize("feature_dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ridge_lambda", [0.0, 0.1])
def test_history_weights_match_direct_coefficients(feature_dtype, ridge_lambda):
    torch.manual_seed(11)
    coords = torch.tensor([-1.0, -0.7, -0.3, 0.1, 0.35], dtype=torch.float32)
    features = torch.randn(5, 3, 4, 6).to(feature_dtype)
    forecaster = HistoryWeightForecaster(
        degree=3,
        ridge_lambda=ridge_lambda,
        max_history=8,
        chunk_bytes=4096,
    )
    for coordinate, feature in zip(coords, features, strict=True):
        forecaster.update(float(coordinate), feature)
    actual = forecaster.predict(0.55, blend_weight=0.6)
    expected = _direct_prediction(coords, features, 0.55, 3, ridge_lambda, 0.6).to(feature_dtype)
    tolerance = 2e-2 if feature_dtype != torch.float32 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.shape == features.shape[1:]
    assert actual.dtype == feature_dtype


def test_chunked_row_subset_matches_full_prediction():
    torch.manual_seed(21)
    forecaster = HistoryWeightForecaster(degree=2, ridge_lambda=0.1, max_history=5, chunk_bytes=4096)
    for index, coordinate in enumerate((-1.0, -0.4, 0.2)):
        forecaster.update(coordinate, torch.randn(4, 17, 19) + index)
    full = forecaster.predict(0.7, blend_weight=0.5)
    subset = forecaster.predict(0.7, blend_weight=0.5, rows=(3, 1))
    torch.testing.assert_close(subset, full[(3, 1), ...])
    assert forecaster.last_prediction_chunk_count >= 2
    assert forecaster.last_prediction_max_fp32_elements <= 1024


def test_one_point_hold_reuses_canonical_rows_with_bounded_chunking_and_dtype_conversion():
    feature = torch.arange(3 * 40 * 40, dtype=torch.float32).reshape(3, 40, 40).to(torch.float16)
    forecaster = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.1,
        max_history=4,
        chunk_bytes=4096,
    )
    forecaster.update(-1.0, feature)
    history_before = [
        (entry.coordinate, entry.feature_flat.data_ptr(), entry.feature_flat.clone())
        for entry in forecaster._history
    ]

    full_prediction = forecaster.predict_one_point_hold()
    prediction = forecaster.predict_one_point_hold(
        rows=(2, 0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(full_prediction, feature, rtol=0.0, atol=0.0)
    torch.testing.assert_close(prediction, feature[(2, 0), ...].to(torch.float32), rtol=0.0, atol=0.0)
    assert full_prediction.dtype == feature.dtype
    assert prediction.dtype == torch.float32
    assert forecaster.last_prediction_chunk_count == 4
    assert forecaster.last_prediction_max_fp32_elements <= 1024
    assert forecaster.factorization_count == 0
    assert forecaster.history_length == 1
    for entry, (coordinate, pointer, archived) in zip(forecaster._history, history_before, strict=True):
        assert entry.coordinate == coordinate
        assert entry.feature_flat.data_ptr() == pointer
        torch.testing.assert_close(entry.feature_flat, archived, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("history_entries", [0, 2])
def test_one_point_hold_requires_exactly_one_history_entry(history_entries):
    forecaster = HistoryWeightForecaster(degree=1, ridge_lambda=0.1, max_history=4)
    for index in range(history_entries):
        forecaster.update(float(index), torch.full((2, 3, 4), float(index)))

    with pytest.raises(RuntimeError, match="exactly one actual history entry"):
        forecaster.predict_one_point_hold(device="cpu", dtype=torch.float32)

    assert forecaster.factorization_count == 0


def test_factorization_is_reused_until_history_changes():
    forecaster = HistoryWeightForecaster(degree=1, ridge_lambda=0.1, max_history=4)
    forecaster.update(-1.0, torch.zeros(1, 2, 3))
    forecaster.update(0.0, torch.ones(1, 2, 3))
    forecaster.predict(0.2, blend_weight=1.0)
    count = forecaster.factorization_count
    forecaster.predict(0.4, blend_weight=1.0)
    assert forecaster.factorization_count == count
    forecaster.update(0.5, torch.full((1, 2, 3), 2.0))
    forecaster.predict(0.7, blend_weight=1.0)
    assert forecaster.factorization_count == count + 1


def test_max_history_eviction_and_degree_change_instances():
    low = HistoryWeightForecaster(degree=1, ridge_lambda=0.1, max_history=2)
    for index in range(5):
        low.update(float(index), torch.full((1, 2, 2), float(index)))
    assert low.history_length == 2
    assert torch.isfinite(low.predict(5.0, blend_weight=0.5)).all()

    high = HistoryWeightForecaster(degree=4, ridge_lambda=0.1, max_history=5)
    for index in range(5):
        high.update(float(index), torch.full((1, 2, 2), float(index)))
    assert high.ready()


def test_repeated_coordinates_take_bounded_jitter_path_with_zero_lambda():
    forecaster = HistoryWeightForecaster(degree=2, ridge_lambda=0.0, max_history=4)
    for value in (1.0, 2.0, 3.0):
        forecaster.update(0.0, torch.full((1, 2, 2), value))
    prediction = forecaster.predict(0.0, blend_weight=1.0)
    assert torch.isfinite(prediction).all()
    assert forecaster.jitter_attempts > 0


def test_persistent_storage_has_no_full_fp32_rhs_or_coefficients():
    forecaster = HistoryWeightForecaster(degree=4, ridge_lambda=0.1, max_history=8)
    for index in range(5):
        forecaster.update(float(index), torch.full((2, 128, 64), float(index), dtype=torch.float16))
    forecaster.predict(5.0, blend_weight=0.5)
    small_matrix_ceiling = 16 * 1024
    assert forecaster.persistent_tensor_bytes <= forecaster.history_tensor_bytes + small_matrix_ceiling
    assert not hasattr(forecaster, "_rhs")
    assert not hasattr(forecaster, "_coeff")


def test_vram_storage_uses_the_feature_device_and_can_take_ownership():
    forecaster = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.1,
        max_history=2,
        history_storage="vram",
    )
    first = torch.zeros(1, 2, 3).contiguous()
    forecaster.update(-1.0, first, take_ownership=True)
    forecaster.update(0.0, torch.ones(1, 2, 3).contiguous(), take_ownership=True)

    assert forecaster.history_device == first.device
    assert forecaster._history[0].feature_flat.data_ptr() == first.data_ptr()
    assert torch.isfinite(forecaster.predict(0.5, blend_weight=0.5)).all()


def test_vram_storage_rejects_a_device_change_within_history():
    forecaster = HistoryWeightForecaster(
        degree=1,
        ridge_lambda=0.1,
        max_history=2,
        history_storage="vram",
    )
    forecaster.update(-1.0, torch.zeros(1, 2, 3), take_ownership=True)

    with pytest.raises(ValueError, match="history device changed"):
        forecaster.update(0.0, torch.empty(1, 2, 3, device="meta"), take_ownership=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_vram_and_system_ram_predictions_match_on_cuda():
    torch.manual_seed(31)
    features = [torch.randn(2, 7, 11, device="cuda", dtype=torch.float16) for _ in range(3)]
    system_ram = HistoryWeightForecaster(degree=2, max_history=3, history_storage="system_ram")
    vram = HistoryWeightForecaster(degree=2, max_history=3, history_storage="vram")
    for coordinate, feature in zip((-1.0, -0.25, 0.5), features, strict=True):
        system_ram.update(coordinate, feature)
        vram.update(coordinate, feature)

    cpu_cached = system_ram.predict(0.75, blend_weight=0.5, device="cuda", dtype=torch.float16)
    gpu_cached = vram.predict(0.75, blend_weight=0.5, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(gpu_cached, cpu_cached, rtol=0.0, atol=0.0)
    assert system_ram.history_device == torch.device("cpu")
    assert vram.history_device is not None
    assert vram.history_device.type == "cuda"
