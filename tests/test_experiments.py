from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import (
    OfflineFeatureArchive,
    OfflineSmoother,
    measure_stream_residual,
)
from comfyui_spectrum_h3.model_aware import (
    ModelAwareForecastDecision,
    SubspaceCorrectionTelemetry,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _model_decision(*, audio_blend: float, video_blend: float, correction: float = 0.0):
    return ModelAwareForecastDecision(
        trajectory_risk=0.2,
        model_risk=0.3,
        patch_risk=0.1,
        combined_risk=0.25,
        confidence=0.75,
        ridge_lambda=0.2,
        degree=1,
        audio_blend_weight=audio_blend,
        video_blend_weight=video_blend,
        audio_correction_gain=correction,
        video_correction_gain=-correction,
        forecast_horizon=1.0,
        force_actual=False,
    )


def test_offline_k2_replay_uses_archived_causal_anchor_ids_without_future_leakage():
    coefficients = (0.1, -0.05)
    telemetry = SubspaceCorrectionTelemetry(
        eligible=True,
        used_scalar_fallback=False,
        generic_coefficients=coefficients,
        exact_coefficients=coefficients,
        applied_coefficients=coefficients,
    )
    decision = ModelAwareForecastDecision(
        trajectory_risk=0.2,
        model_risk=0.3,
        patch_risk=0.1,
        combined_risk=0.25,
        confidence=0.75,
        ridge_lambda=0.2,
        degree=1,
        audio_blend_weight=0.5,
        video_blend_weight=0.5,
        audio_correction_gain=0.0,
        video_correction_gain=0.0,
        forecast_horizon=1.0,
        force_actual=False,
        audio_subspace_telemetry=telemetry,
        video_subspace_telemetry=telemetry,
        correction_anchor_ids=(0, 2, 4),
    )

    def build(with_correction: bool):
        archive = OfflineFeatureArchive(total_steps=7, sampler_name="sample_euler")
        coordinates = torch.linspace(-1.0, 1.0, 7).tolist()
        for step_id, coordinate in enumerate(coordinates):
            archive.record_step(
                step_id,
                coordinate,
                step_id % 2 == 0,
                model_aware_decision=(decision if with_correction and step_id == 5 else None),
            )
        for step_id in (0, 2, 4, 6):
            archive.record_actual(
                step_id,
                coordinates[step_id],
                torch.full((1, 2, 4), float(step_id), dtype=torch.float32),
                labels=((0, "positive"),),
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                take_ownership=True,
            )
        assert archive.complete(minimum_anchors=2)
        return OfflineSmoother(
            archive,
            degree=1,
            ridge_lambda=0.2,
            blend_weight=0.5,
            audio_blend_weight=0.5,
        )

    baseline = build(False)
    corrected = build(True)
    base_weights = baseline._forecast_weights[(5, 0, 0)]
    corrected_weights = corrected._forecast_weights[(5, 0, 0)]
    difference = corrected_weights - base_weights

    torch.testing.assert_close(
        difference,
        torch.tensor((0.05, -0.15, 0.1, 0.0), dtype=torch.float32),
    )


def _archive(right_value: float) -> OfflineFeatureArchive:
    archive = OfflineFeatureArchive(total_steps=3, sampler_name="sample_euler")
    archive.record_step(0, -1.0, True)
    archive.record_step(1, 0.0, False)
    archive.record_step(2, 1.0, True)
    labels = ((0, "positive"),)
    topology = (("shape", 1),)
    archive.record_actual(
        0,
        -1.0,
        torch.zeros(1, 1, 2, dtype=torch.float16),
        labels=labels,
        topology=topology,
        take_ownership=True,
    )
    archive.record_actual(
        2,
        1.0,
        torch.full((1, 1, 2), right_value, dtype=torch.float16),
        labels=labels,
        topology=topology,
        take_ownership=True,
    )
    assert archive.complete(minimum_anchors=2)
    return archive


def _interleaved_archive(values: list[float], *, history_storage: str = "system_ram") -> OfflineFeatureArchive:
    total_steps = len(values) * 2 - 1
    coordinates = torch.linspace(-1.0, 1.0, total_steps).tolist()
    archive = OfflineFeatureArchive(
        total_steps=total_steps,
        sampler_name="sample_euler",
        history_storage=history_storage,
    )
    for step_id, coordinate in enumerate(coordinates):
        archive.record_step(step_id, coordinate, step_id % 2 == 0)
    for anchor_index, value in enumerate(values):
        step_id = anchor_index * 2
        archive.record_actual(
            step_id,
            coordinates[step_id],
            torch.full((1, 2, 16), value, dtype=torch.float32),
            labels=((0, "positive"),),
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            take_ownership=True,
        )
    assert archive.complete(minimum_anchors=2)
    return archive


def _modality_archive(audio_values: list[float], video_values: list[float]) -> OfflineFeatureArchive:
    assert len(audio_values) == len(video_values)
    total_steps = len(audio_values) * 2 - 1
    coordinates = torch.linspace(-1.0, 1.0, total_steps).tolist()
    archive = OfflineFeatureArchive(total_steps=total_steps, sampler_name="sample_euler")
    for step_id, coordinate in enumerate(coordinates):
        archive.record_step(step_id, coordinate, step_id % 2 == 0)
    for anchor_index, (audio, video) in enumerate(zip(audio_values, video_values, strict=True)):
        step_id = anchor_index * 2
        feature = torch.empty(1, 2, 16)
        feature[:, 0].fill_(audio)
        feature[:, 1].fill_(video)
        archive.record_actual(
            step_id,
            coordinates[step_id],
            feature,
            labels=((0, "positive"),),
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            take_ownership=True,
        )
    assert archive.complete(minimum_anchors=2)
    return archive


def test_residual_score_uses_scale_aware_zero_case():
    actual = torch.full((2048,), 1000.0)
    score = measure_stream_residual(actual, actual.clone(), actual.clone(), chunk_bytes=4096)
    assert score.score == 0.0
    assert score.epsilon == pytest.approx(1e-3)
    assert score.chunks == 2


def test_residual_score_compares_forecast_with_hold_baseline():
    actual = torch.full((8,), 3.0)
    shadow = torch.zeros(8)
    hold = torch.full((8,), 2.0)
    score = measure_stream_residual(actual, shadow, hold)
    assert score.forecast_rms == pytest.approx(3.0)
    assert score.hold_rms == pytest.approx(1.0)
    assert score.score == pytest.approx(3.0)


def test_offline_smoother_uses_future_anchor_and_reuses_actual_features_exactly():
    first_archive = _archive(2.0)
    second_archive = _archive(6.0)
    first = OfflineSmoother(
        first_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.5,
    )
    second = OfflineSmoother(
        second_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.5,
    )
    kwargs = {"rows": (0,), "device": torch.device("cpu"), "dtype": torch.float16}
    first_middle = first.predict(1, **kwargs)
    second_middle = second.predict(1, **kwargs)
    assert not torch.equal(first_middle, second_middle)
    torch.testing.assert_close(first.predict(0, **kwargs), first_archive.anchors[0].feature)
    torch.testing.assert_close(first.predict(2, **kwargs), first_archive.anchors[1].feature)


def test_offline_local_component_is_bracketing_interpolation():
    archive = _archive(8.0)
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.0,
    )
    middle = smoother.predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    torch.testing.assert_close(middle, torch.full_like(middle, 4.0))


def test_offline_spectral_weights_preserve_constant_trajectories_under_ridge():
    archive = _interleaved_archive([4.0] * 5)
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=10.0,
        blend_weight=1.0,
    )

    for step_id in (1, 3, 5, 7):
        prediction = smoother.predict(
            step_id,
            rows=(0,),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(prediction, torch.full_like(prediction, 4.0))
        for stream_index in range(2):
            assert smoother._forecast_weights[(step_id, 0, stream_index)].sum().item() == pytest.approx(1.0)


def test_offline_smoother_attenuates_spectral_fit_that_loses_validation():
    archive = _interleaved_archive([0.0, 1.0, 4.0, 9.0, 16.0])
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
    )

    assert smoother.validation_samples_per_branch == 32
    assert smoother.validation_stream_count == 2
    assert smoother.validation_anchor_count == 3
    assert smoother.attenuated_prediction_count == 4
    assert smoother.local_only_prediction_counts["audio"] == 4
    assert smoother.attenuated_prediction_counts["video"] == 4
    assert smoother.effective_blend_stream_stats["audio"] == (0.0, 0.0, 0.0)
    video_min, video_mean, video_max = smoother.effective_blend_stream_stats["video"]
    assert 0.0 < video_min < video_max < 0.5
    assert video_min < video_mean < video_max


def test_offline_smoother_validates_audio_and_video_independently():
    archive = _modality_archive(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
    )
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
    )

    assert smoother.validation_stream_max_scores["audio"] > 1.0
    assert smoother.validation_stream_max_scores["video"] <= 1.0
    assert smoother.effective_blend_stream_stats["audio"] == (0.0, 0.0, 0.0)
    assert smoother.effective_blend_stream_stats["video"] == (0.5, 0.5, 0.5)
    prediction = smoother.predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert prediction.shape == (1, 2, 16)
    local = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.0,
    ).predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(prediction[:, :1], local[:, :1])
    assert not torch.equal(prediction[:, 1:], local[:, 1:])


def test_offline_replay_preserves_per_forecast_model_aware_decisions_without_index_drift():
    archive = OfflineFeatureArchive(total_steps=5, sampler_name="sample_er_sde")
    coordinates = torch.linspace(-1.0, 1.0, 5).tolist()
    first_decision = _model_decision(audio_blend=0.0, video_blend=0.2)
    second_decision = _model_decision(audio_blend=0.0, video_blend=0.8, correction=0.1)
    for step_id, coordinate in enumerate(coordinates):
        archive.record_step(
            step_id,
            coordinate,
            step_id % 2 == 0,
            model_aware_decision={1: first_decision, 3: second_decision}.get(step_id),
        )
    for step_id, value in ((0, 0.0), (2, 1.0), (4, 2.0)):
        feature = torch.empty(1, 2, 16)
        feature[:, 0].fill_(value)
        feature[:, 1].fill_(value)
        archive.record_actual(
            step_id,
            coordinates[step_id],
            feature,
            labels=((0, "positive"),),
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            take_ownership=True,
        )
    assert archive.complete(minimum_anchors=2)

    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    first_archived = archive.steps[1].model_aware_decision
    second_archived = archive.steps[3].model_aware_decision
    assert first_archived is not None and first_archived is not first_decision
    assert second_archived is not None and second_archived is not second_decision
    assert first_archived.audio_blend_weight == first_decision.audio_blend_weight
    assert first_archived.video_blend_weight == first_decision.video_blend_weight
    assert second_archived.audio_correction_gain == second_decision.audio_correction_gain
    assert second_archived.video_correction_gain == second_decision.video_correction_gain
    assert all(
        isinstance(getattr(second_archived, field), (int, float))
        for field in (
            "degree",
            "ridge_lambda",
            "audio_blend_weight",
            "video_blend_weight",
            "audio_correction_gain",
            "video_correction_gain",
        )
    )
    assert smoother.effective_blend_stream_stats["audio"] == (0.0, 0.0, 0.0)
    assert smoother.effective_blend_stream_stats["video"] == pytest.approx((0.2, 0.5, 0.8))
    assert smoother.model_aware_offline_correction_applications == 2
    assert smoother.model_aware_offline_correction_seconds > 0.0
    assert (
        smoother.model_aware_correction_seconds
        == smoother.model_aware_offline_correction_seconds
    )
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    runtime._offline_smoother = smoother
    runtime._record_offline_smoother_stats()
    assert runtime.stats.model_aware_offline_correction_applications == 2
    assert runtime.stats.model_aware_offline_correction_seconds > 0.0
    assert runtime.stats.model_aware_causal_correction_seconds == 0.0
    assert (
        runtime.stats.model_aware_correction_seconds
        == runtime.stats.model_aware_offline_correction_seconds
    )
    assert runtime.stats.model_aware_overhead_seconds >= (
        runtime.stats.model_aware_fit_seconds
        + runtime.stats.model_aware_offline_correction_seconds
    )
    summary = runtime.debug_summary()
    assert "model_aware_causal_correction_s=0.000000" in summary
    assert "model_aware_offline_replay_correction_s=" in summary
    assert "model_aware_offline_replay_correction_applications=2" in summary
    spectral = smoother._affine_spectral_weights(
        smoother._forecaster.model_aware_weights(
            coordinates[3],
            1.0,
            degree=second_decision.degree,
            ridge_lambda=second_decision.ridge_lambda,
        )
    )
    local = torch.tensor([0.0, 0.5, 0.5])
    expected_step_3_video = 0.8 * spectral + 0.2 * local
    expected_step_3_video[1] += 0.1
    expected_step_3_video[2] -= 0.1
    torch.testing.assert_close(
        smoother._forecast_weights[(3, 0, 1)],
        expected_step_3_video,
    )
    assert not torch.equal(
        smoother._forecast_weights[(1, 0, 1)],
        smoother._forecast_weights[(3, 0, 1)],
    )


def test_offline_archive_shares_owned_storage_on_selected_device():
    archive = OfflineFeatureArchive(
        total_steps=1,
        sampler_name="sample_euler",
        history_storage="vram",
    )
    archive.record_step(0, 0.0, True)
    feature = torch.ones(1, 2, 3)
    archive.record_actual(
        0,
        0.0,
        feature,
        labels=((0, "positive"),),
        topology=(("shape", 1),),
        take_ownership=True,
    )

    assert archive.anchors[0].feature.data_ptr() == feature.data_ptr()
    assert archive.history_device == feature.device


def test_offline_archive_rejects_unknown_storage():
    with pytest.raises(ValueError, match="history_storage"):
        OfflineFeatureArchive(
            total_steps=1,
            sampler_name="sample_euler",
            history_storage="automatic",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_offline_vram_archive_and_smoother_share_cuda_anchors():
    archive = OfflineFeatureArchive(
        total_steps=3,
        sampler_name="sample_euler",
        history_storage="vram",
    )
    archive.record_step(0, -1.0, True)
    archive.record_step(1, 0.0, False)
    archive.record_step(2, 1.0, True)
    features = [
        torch.full((1, 2, 4), value, device="cuda", dtype=torch.float16)
        for value in (0.0, 2.0)
    ]
    for step_id, coordinate, feature in zip((0, 2), (-1.0, 1.0), features, strict=True):
        archive.record_actual(
            step_id,
            coordinate,
            feature,
            labels=((0, "positive"),),
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            take_ownership=True,
        )
    assert archive.complete(minimum_anchors=2)

    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
    )
    assert archive.history_device is not None
    assert archive.history_device.type == "cuda"
    assert smoother.history_device == archive.history_device
    assert [anchor.feature.data_ptr() for anchor in archive.anchors] == [
        feature.data_ptr() for feature in features
    ]
    prediction = smoother.predict(
        1,
        rows=(0,),
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    assert prediction.device.type == "cuda"
    assert torch.isfinite(prediction).all()


def test_offline_archive_requires_a_future_anchor_for_every_forecast():
    archive = OfflineFeatureArchive(total_steps=2, sampler_name="sample_euler")
    archive.record_step(0, -1.0, True)
    archive.record_step(1, 0.0, False)
    archive.record_actual(
        0,
        -1.0,
        torch.zeros(1, 1, 1),
        labels=((0, "positive"),),
        topology=(("shape", 1),),
        take_ownership=True,
    )
    assert not archive.complete(minimum_anchors=1)
    assert "future actual anchor" in archive.failure_reason
