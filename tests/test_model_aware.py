from __future__ import annotations

import dataclasses
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.forecast import HistoryWeightForecaster
from comfyui_spectrum_h3.model_aware import (
    AnchorEvidence,
    ModelAwareController,
    ModelForecastabilityProfile,
    ProfileLookup,
    clear_model_profile_cache,
    get_model_forecastability_profile,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


class _AdaLN(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv_proj = torch.nn.Linear(hidden, hidden * 3, bias=False)
        self.attn.out_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(hidden, hidden * 2, bias=False)
        self.mlp.fc2 = torch.nn.Linear(hidden, hidden, bias=False)
        self.adaln_proj = _AdaLN(hidden)


class _Inner(torch.nn.Module):
    def __init__(self, hidden: int = 4):
        super().__init__()
        self.hidden_size = hidden
        self.patch_size = (1, 2, 2)
        self.latents_dim = 16
        self.audio_latents_dim = 32
        self.use_adaln_curves = False
        self.blocks = torch.nn.ModuleList([_Block(hidden), _Block(hidden)])
        self.final_layer = torch.nn.Module()
        self.final_layer.adaln_proj = _AdaLN(hidden)
        self.final_layer.video_out = torch.nn.Linear(hidden, 8, bias=False)
        self.final_layer.audio_out = torch.nn.Linear(hidden, 6, bias=False)


class _Patcher:
    def __init__(self, inner, *, base_uuid="base", patches_uuid="patches", patches=None):
        self.model = SimpleNamespace(diffusion_model=inner)
        self.clone_base_uuid = base_uuid
        self.patches_uuid = patches_uuid
        self.patches = patches or {}
        self.backup = {}
        self.injections = {}

    def get_model_object(self, key):
        value = self.model
        for part in key.split("."):
            value = getattr(value, part)
        return value


class _LoRA:
    name = "lora"

    def __init__(self, up, down, *, alpha=None):
        self.weights = (up, down, alpha, None, None, None)


def _profile(**overrides):
    values = {
        "cache_key": ("base", "patches"),
        "base_model_identity": "fake:base",
        "patch_identity": "patches",
        "active_patch_count": 0,
        "active_patch_keys": 0,
        "recognized_lora_count": 0,
        "unknown_patch_count": 0,
        "sampled_base_tensors": 8,
        "profile_confidence": 1.0,
        "aggregate_sensitivity": 0.2,
        "patch_perturbation": 0.0,
        "final_block_perturbation": 0.0,
        "audio_sensitivity": 0.8,
        "video_sensitivity": 1.2,
        "forecast_risk_prior": 0.2,
        "build_seconds": 0.001,
        "estimated_bytes": 512,
        "transient_workspace_bytes": 4096,
    }
    values.update(overrides)
    return ModelForecastabilityProfile(**values)


@pytest.fixture(autouse=True)
def _empty_profile_cache():
    clear_model_profile_cache()
    yield
    clear_model_profile_cache()


def _lora_patch(strength, *, scale=1.0):
    up = torch.eye(4) * scale
    down = torch.eye(4)
    return (strength, _LoRA(up, down), 1.0, None, None)


def test_base_profile_is_scalar_only_and_reused_across_clone_lineage():
    inner = _Inner()
    original = _Patcher(inner)
    first = get_model_forecastability_profile(original)
    clone = _Patcher(inner)
    second = get_model_forecastability_profile(clone)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.profile is second.profile
    assert first.profile.active_patch_count == 0
    assert first.profile.sampled_base_tensors == 8
    assert first.profile.estimated_bytes <= 4096
    assert all(
        not torch.is_tensor(getattr(first.profile, field.name))
        for field in dataclasses.fields(first.profile)
    )

    reference = weakref.ref(original)
    del original
    gc.collect()
    assert reference() is None


def test_lora_strength_and_composition_change_profile_without_materializing_ba():
    inner = _Inner()
    key = "diffusion_model.blocks.1.attn.out_proj.weight"
    weak = get_model_forecastability_profile(
        _Patcher(inner, patches_uuid="weak", patches={key: [_lora_patch(0.25)]})
    ).profile
    strong = get_model_forecastability_profile(
        _Patcher(inner, patches_uuid="strong", patches={key: [_lora_patch(1.0)]})
    ).profile
    multiple = get_model_forecastability_profile(
        _Patcher(
            inner,
            patches_uuid="multiple",
            patches={key: [_lora_patch(0.5), _lora_patch(0.75, scale=0.5)]},
        )
    ).profile

    assert weak.recognized_lora_count == 1
    assert strong.patch_perturbation > weak.patch_perturbation > 0.0
    assert multiple.recognized_lora_count == 2
    assert multiple.active_patch_count == 2
    assert multiple.active_patch_keys == 1
    assert multiple.transient_workspace_bytes < 1_000_000


def test_zero_strength_lora_is_inactive_and_unknown_patch_is_conservative():
    inner = _Inner()
    key = "diffusion_model.blocks.1.attn.out_proj.weight"
    zero = get_model_forecastability_profile(
        _Patcher(inner, patches_uuid="zero", patches={key: [_lora_patch(0.0)]})
    ).profile
    unknown = get_model_forecastability_profile(
        _Patcher(inner, patches_uuid="unknown", patches={key: [(1.0, object(), 1.0, None, None)]})
    ).profile

    assert zero.active_patch_count == 0
    assert zero.patch_perturbation == 0.0
    assert unknown.active_patch_count == 1
    assert unknown.unknown_patch_count == 1
    assert unknown.profile_confidence < zero.profile_confidence


def test_bypass_injection_manager_is_profiled_once_when_both_closures_capture_it():
    inner = _Inner()
    patcher = _Patcher(inner, patches_uuid="bypass")
    manager = SimpleNamespace(
        adapters={
            "diffusion_model.blocks.1.attn.out_proj": (
                _LoRA(torch.eye(4), torch.eye(4)),
                0.75,
            )
        }
    )

    def inject(_patcher):
        return manager

    def eject(_patcher):
        return manager

    patcher.injections = {"bypass_lora": [SimpleNamespace(inject=inject, eject=eject)]}
    profile = get_model_forecastability_profile(patcher).profile

    assert profile.active_patch_count == 1
    assert profile.active_patch_keys == 1
    assert profile.recognized_lora_count == 1


def test_patch_uuid_invalidates_cache_while_same_identity_hits():
    inner = _Inner()
    first = get_model_forecastability_profile(_Patcher(inner, patches_uuid="a"))
    repeated = get_model_forecastability_profile(_Patcher(inner, patches_uuid="a"))
    changed = get_model_forecastability_profile(_Patcher(inner, patches_uuid="b"))

    assert not first.cache_hit
    assert repeated.cache_hit
    assert not changed.cache_hit
    assert first.profile.cache_key != changed.profile.cache_key


def test_patch_metadata_digest_invalidates_in_place_strength_changes():
    inner = _Inner()
    key = "diffusion_model.blocks.1.attn.out_proj.weight"
    patches = {key: [_lora_patch(0.25)]}
    patcher = _Patcher(inner, patches_uuid="stable", patches=patches)
    first = get_model_forecastability_profile(patcher)
    patches[key][0] = _lora_patch(1.0)
    changed = get_model_forecastability_profile(patcher)

    assert not first.cache_hit
    assert not changed.cache_hit
    assert changed.profile.patch_perturbation > first.profile.patch_perturbation


def test_controller_combines_static_prior_with_observed_evidence_and_bounds_outputs():
    controller = ModelAwareController("full", risk_threshold=0.55)
    controller.set_profile(
        _profile(
            patch_perturbation=0.8,
            forecast_risk_prior=0.85,
            audio_sensitivity=0.5,
            video_sensitivity=1.5,
        )
    )
    initial = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=4,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.4,
        configured_video_blend=0.6,
    )
    for _ in range(4):
        controller.observe_anchor(
            AnchorEvidence(0.2, 0.1, 2.0, 0.4, -0.3, 0.3, 0.4)
        )
    calibrated = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=4,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.4,
        configured_video_blend=0.6,
    )

    assert calibrated.combined_risk < initial.combined_risk
    assert 1 <= calibrated.degree <= 4
    assert 1e-6 <= calibrated.ridge_lambda <= 10.0
    assert 0.0 <= calibrated.audio_blend_weight <= 0.4
    assert 0.0 <= calibrated.video_blend_weight <= 0.6
    assert abs(calibrated.audio_correction_gain) <= 0.25
    assert abs(calibrated.video_correction_gain) <= 0.25
    assert calibrated.audio_correction_gain != calibrated.video_correction_gain


def test_bad_observed_evidence_overrides_a_benign_static_prior():
    controller = ModelAwareController("schedule_confidence", risk_threshold=0.45)
    controller.set_profile(_profile(forecast_risk_prior=0.02))
    benign = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=4,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.4,
        configured_video_blend=0.6,
    )
    for _ in range(3):
        controller.observe_anchor(AnchorEvidence(5.0, 4.0, 1e7, 0.0, 0.0, 5.0, 5.0))
    risky = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=4,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.4,
        configured_video_blend=0.6,
    )

    assert not benign.force_actual
    assert risky.force_actual
    assert risky.combined_risk > benign.combined_risk
    assert risky.degree < benign.degree
    assert risky.ridge_lambda > benign.ridge_lambda


def test_correction_is_zero_outside_full_mode():
    controller = ModelAwareController("schedule_confidence", risk_threshold=1.0)
    controller.set_profile(_profile(audio_sensitivity=0.5, video_sensitivity=1.5))
    controller.observe_anchor(AnchorEvidence(0.5, 0.2, 2.0, 0.5, -0.5, 0.4, 0.4))
    decision = controller.decision(
        forecast_horizon=1.0,
        history_length=4,
        configured_degree=2,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.4,
        configured_video_blend=0.6,
    )

    assert decision.audio_correction_gain == 0.0
    assert decision.video_correction_gain == 0.0


def test_sampled_evidence_is_bounded_and_model_aware_weight_correction_preserves_affine_sum():
    forecaster = HistoryWeightForecaster(degree=1, ridge_lambda=0.1, max_history=4)
    forecaster.update(0.0, torch.zeros((1, 4, 3)))
    forecaster.update(0.5, torch.ones((1, 4, 3)))
    weights = forecaster.model_aware_weights(
        1.0,
        0.5,
        degree=1,
        ridge_lambda=0.2,
        correction_gain=0.2,
    )
    uncorrected = forecaster.model_aware_weights(
        1.0,
        0.5,
        degree=1,
        ridge_lambda=0.2,
        correction_gain=0.0,
    )
    evidence = forecaster.sampled_anchor_evidence(
        1.0,
        torch.full((1, 4, 3), 2.2),
        (("audio", 0, 2, 0.5, 0.2, 0.1), ("video", 2, 4, 0.5, -0.1, 0.1)),
        degree=1,
        ridge_lambda=0.2,
    )

    assert float(weights.sum()) == pytest.approx(float(uncorrected.sum()), abs=1e-6)
    assert evidence is not None
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            evidence.forecast_ratio,
            evidence.curvature_ratio,
            evidence.audio_projection,
            evidence.video_projection,
        )
    )


def test_model_aware_schedule_can_only_convert_a_legacy_forecast_to_actual():
    config = SpectrumH3Config(
        degree=1,
        max_history=4,
        warmup_steps=1,
        tail_actual_steps=0,
        window_size=2.0,
        bootstrap_first_forecast=True,
        offline_smoothing_replay=False,
        model_aware_mode="schedule",
        model_aware_risk_threshold=0.01,
    )
    runtime = SpectrumH3Runtime(config)
    runtime.set_model_profile(ProfileLookup(_profile(forecast_risk_prior=0.9), False, 0.001))
    runtime.start_run(
        torch.linspace(1.0, 0.0, 5),
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    first = runtime.begin_step(torch.tensor([1.0]))
    assert first["actual"]
    runtime.abort_step(first["run_id"], first["step_id"])
    # Seed one native anchor so the legacy degree-one bootstrap is eligible.
    first = runtime.begin_step(torch.tensor([1.0]))
    call, actual = runtime.begin_model_call(
        first["run_id"], first["step_id"], topology=(("hidden", 2),), labels=((0, "p"),), expected_shape=(1, 2, 2)
    )
    assert actual
    runtime.observe_actual(first["run_id"], first["step_id"], call, torch.ones((1, 2, 2)))
    runtime.finalize_step(first["run_id"], first["step_id"])
    second = runtime.begin_step(torch.tensor([0.75]))

    assert second["actual"]
    assert second["reason"] == "model-aware forecast risk"
    assert runtime.active_model_aware_decision is not None
    assert runtime.active_model_aware_decision.force_actual


def test_disabling_model_aware_clears_profile_and_lookup_metadata():
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="schedule"))
    runtime.set_model_profile(ProfileLookup(_profile(), True, 0.25))

    runtime.disable_model_aware("profile refresh failed")
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )

    assert runtime.model_profile is None
    assert runtime.model_aware.profile is None
    assert not runtime.stats.model_profile_cache_hit
    assert runtime.stats.model_profile_lookup_seconds == 0.0
    assert runtime.stats.model_profile_bytes == 0
    runtime.end_run(run_id)


def test_model_aware_weight_failure_uses_actual_fallback_with_resolved_coordinate(
    monkeypatch,
):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
            offline_smoothing_replay=False,
            model_aware_mode="schedule_confidence",
            model_aware_risk_threshold=1.0,
        )
    )
    runtime.set_model_profile(ProfileLookup(_profile(forecast_risk_prior=0.0), False, 0.001))
    runtime.start_run(
        torch.linspace(1.0, 0.0, 5),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    topology = (("target_audio_rows", 1), ("target_video_rows", 1))
    labels = ((0, "positive"),)
    for timestep in (1.0, 0.75):
        actual_step = runtime.begin_step(torch.tensor([timestep]))
        call_id, actual = runtime.begin_model_call(
            actual_step["run_id"],
            actual_step["step_id"],
            topology=topology,
            labels=labels,
            expected_shape=(1, 2, 2),
        )
        assert actual
        runtime.observe_actual(
            actual_step["run_id"],
            actual_step["step_id"],
            call_id,
            torch.full((1, 2, 2), timestep),
        )
        runtime.finalize_step(actual_step["run_id"], actual_step["step_id"])

    forecast_step = runtime.begin_step(torch.tensor([0.5]))
    call_id, actual = runtime.begin_model_call(
        forecast_step["run_id"],
        forecast_step["step_id"],
        topology=topology,
        labels=labels,
        expected_shape=(1, 2, 2),
    )
    assert not actual
    observed_coordinates = []

    def fail_weight_construction(_call, _decision, *, coordinate):
        observed_coordinates.append(coordinate)
        raise RuntimeError("synthetic adaptive factorization failure")

    monkeypatch.setattr(runtime, "_model_aware_weight_segments", fail_weight_construction)

    prediction = runtime.predict(
        forecast_step["run_id"],
        forecast_step["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prediction is None
    assert observed_coordinates == [pytest.approx(forecast_step["coordinate"])]
    assert runtime._step is not None
    assert runtime._step.mode == "actual"
    assert runtime._step.fallback
