from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import OfflineFeatureArchive, OfflineSmoother
from comfyui_spectrum_h3.model_aware import ModelAwareForecastDecision
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _scalar_decision(*, gain: float, anchor_ids: tuple[int, ...]) -> ModelAwareForecastDecision:
    return ModelAwareForecastDecision(
        trajectory_risk=0.2,
        model_risk=0.3,
        patch_risk=0.0,
        combined_risk=0.25,
        confidence=0.75,
        ridge_lambda=0.1,
        degree=1,
        audio_blend_weight=0.0,
        video_blend_weight=0.0,
        audio_correction_gain=gain,
        video_correction_gain=-gain,
        forecast_horizon=1.0,
        force_actual=False,
        correction_anchor_ids=anchor_ids,
    )


def test_full_forecast_captures_latest_two_causal_anchor_ids_for_replay():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_risk_threshold=1.0,
            warmup_steps=0,
            tail_actual_steps=0,
            bootstrap_first_forecast=False,
        )
    )
    sigmas = torch.linspace(1.0, 0.0, 7, dtype=torch.float32)
    runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
    )

    profile = SimpleNamespace(
        forecast_risk_prior=0.2,
        patch_perturbation=0.0,
    )
    runtime._model_profile = profile
    runtime.model_aware.set_profile(profile)
    runtime.model_aware.anchor_count = 2
    runtime.model_aware.audio_projection_ewma = 0.5
    runtime.model_aware.video_projection_ewma = -0.4

    feature = torch.zeros(1, 2, 4, dtype=torch.float32)
    runtime.forecaster.update(0.8, feature, anchor_id=0, take_ownership=True)
    runtime.forecaster.update(0.4, feature + 1.0, anchor_id=2, take_ownership=True)
    assert runtime._run is not None
    runtime._run.next_step_id = 3

    result = runtime.begin_step(sigmas[3])
    assert not result["actual"]
    decision = runtime.active_model_aware_decision
    assert decision is not None
    assert decision.audio_correction_gain != 0.0
    assert decision.video_correction_gain != 0.0
    assert decision.correction_anchor_ids == (0, 2)

    runtime.abort_step(runtime.active_run_id, runtime.active_step_id)


def _build_scalar_replay(*, corrected: bool) -> OfflineSmoother:
    archive = OfflineFeatureArchive(total_steps=7, sampler_name="sample_er_sde")
    coordinates = torch.linspace(-1.0, 1.0, 7, dtype=torch.float32).tolist()
    decision = _scalar_decision(gain=0.1, anchor_ids=(2, 4))
    for step_id, coordinate in enumerate(coordinates):
        archive.record_step(
            step_id,
            coordinate,
            step_id in {0, 2, 4, 6},
            model_aware_decision=(decision if corrected and step_id == 5 else None),
        )

    for step_id, value in ((0, 0.0), (2, 1.0), (4, 2.0), (6, 100.0)):
        archive.record_actual(
            step_id,
            coordinates[step_id],
            torch.full((1, 2, 8), value, dtype=torch.float32),
            labels=((0, "positive"),),
            topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
            take_ownership=True,
        )
    assert archive.complete(minimum_anchors=2)
    return OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.0,
        audio_blend_weight=0.0,
    )


def test_scalar_replay_correction_uses_causal_pair_not_future_bracketing_anchor():
    baseline = _build_scalar_replay(corrected=False)
    corrected = _build_scalar_replay(corrected=True)

    base_weights = baseline._forecast_weights[(5, 0, 0)]
    corrected_weights = corrected._forecast_weights[(5, 0, 0)]
    difference = corrected_weights - base_weights

    torch.testing.assert_close(
        difference,
        torch.tensor((0.0, -0.1, 0.1, 0.0), dtype=torch.float32),
        rtol=0.0,
        atol=1e-7,
    )
    assert difference[-1].item() == pytest.approx(0.0, abs=1e-7)
