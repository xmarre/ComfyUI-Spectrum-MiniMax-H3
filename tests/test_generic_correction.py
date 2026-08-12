from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import OfflineModelAwareDecision
from comfyui_spectrum_h3.model_aware import (
    ModelAwareController,
    ModelForecastabilityProfile,
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
    def __init__(self, inner):
        self.model = SimpleNamespace(diffusion_model=inner)
        self.clone_base_uuid = "base"
        self.patches_uuid = "patches"
        self.patches = {}
        self.backup = {}
        self.injections = {}

    def get_model_object(self, key):
        value = self.model
        for part in key.split("."):
            value = getattr(value, part)
        return value


def _profile() -> ModelForecastabilityProfile:
    return ModelForecastabilityProfile(
        cache_key=("base", "patches"),
        base_model_identity="fake:base",
        patch_identity="patches",
        active_patch_count=0,
        active_patch_keys=0,
        recognized_lora_count=0,
        unknown_patch_count=0,
        sampled_base_tensors=8,
        profile_confidence=1.0,
        aggregate_sensitivity=0.2,
        patch_perturbation=0.0,
        final_block_perturbation=0.0,
        audio_sensitivity=0.8,
        video_sensitivity=1.2,
        audio_head_weight=None,
        video_head_weight=None,
        audio_head_gram_diagonal=None,
        video_head_gram_diagonal=None,
        forecast_risk_prior=0.2,
        build_seconds=0.001,
        estimated_bytes=1024,
        transient_workspace_bytes=4096,
    )


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    clear_model_profile_cache()
    yield
    clear_model_profile_cache()


def test_profile_retains_only_compact_non_tensor_state():
    lookup = get_model_forecastability_profile(_Patcher(_Inner()))
    profile = lookup.profile
    assert profile.sampled_base_tensors == 8
    assert profile.audio_sensitivity > 0.0
    assert profile.video_sensitivity > 0.0
    assert profile.audio_head_weight is None
    assert profile.video_head_weight is None
    assert profile.audio_head_gram_diagonal is None
    assert profile.video_head_gram_diagonal is None
    assert profile.estimated_bytes <= 4096
    assert all(
        not torch.is_tensor(getattr(profile, field.name))
        for field in dataclasses.fields(profile)
    )


def test_full_applied_gain_is_exactly_generic_scalar_gain():
    controller = ModelAwareController("full", risk_threshold=1.0)
    controller.set_profile(_profile())
    controller.anchor_count = 3
    controller.audio_projection_ewma = 0.60
    controller.video_projection_ewma = -0.35
    # Deliberately divergent rejected-model state must not affect shipping gain.
    controller.audio_diagonal_projection_ewma = -1.5
    controller.audio_model_projection_ewma = 1.8
    controller.video_diagonal_projection_ewma = 1.4
    controller.video_model_projection_ewma = -1.7
    controller.audio_model_trust = 1.0
    controller.video_model_trust = 1.0

    decision = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=1,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.0,
        configured_video_blend=0.5,
    )
    for gain, telemetry in (
        (decision.audio_correction_gain, decision.audio_correction_telemetry),
        (decision.video_correction_gain, decision.video_correction_telemetry),
    ):
        assert gain == pytest.approx(telemetry.generic_gain)
        assert telemetry.model_gain == pytest.approx(telemetry.generic_gain)
        assert telemetry.model_candidate_gain == pytest.approx(telemetry.generic_gain)
        assert telemetry.diagonal_candidate_gain == pytest.approx(telemetry.generic_gain)
        assert telemetry.model_trust == 0.0
    assert not decision.audio_subspace_telemetry.eligible
    assert not decision.video_subspace_telemetry.eligible
    assert decision.correction_anchor_ids == ()


def test_schedule_confidence_has_no_correction():
    controller = ModelAwareController("schedule_confidence", risk_threshold=1.0)
    controller.set_profile(_profile())
    controller.anchor_count = 3
    controller.audio_projection_ewma = 0.8
    controller.video_projection_ewma = -0.8
    decision = controller.decision(
        forecast_horizon=1.0,
        history_length=5,
        configured_degree=2,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.2,
        configured_video_blend=0.5,
    )
    assert decision.audio_correction_gain == 0.0
    assert decision.video_correction_gain == 0.0


def test_full_head_materialization_is_disabled():
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    runtime.model_aware.set_profile(_profile())
    weights, diagonals = runtime._model_aware_head_metrics(torch.device("cpu"))
    assert weights == {}
    assert diagonals == {}
    assert runtime.model_aware.materialized_head_bytes == 0
    assert runtime.stats.model_aware_head_materialized_bytes == 0
    assert runtime.stats.model_aware_head_materialization_seconds == 0.0


def test_forecaster_never_retains_exact_head_evidence_after_cleanup():
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    runtime.forecaster._generic_correction_capture_mode = "full"
    feature = torch.randn(1, 4, 6)
    runtime.forecaster.update(
        0.0,
        feature,
        evidence_segments=(("audio", 0, 2), ("video", 2, 4)),
        exact_head_weights={
            "audio": torch.randn(3, 6),
            "video": torch.randn(5, 6),
        },
    )
    assert runtime.forecaster.generic_evidence_tensor_bytes > 0
    assert runtime.forecaster.exact_head_evidence_tensor_bytes == 0
    assert runtime.forecaster.model_aware_exact_head_projection_calls == 0
    assert runtime.forecaster.model_aware_exact_head_projection_seconds == 0.0
    assert runtime.forecaster.model_aware_exact_head_workspace_bytes == 0


def test_offline_replay_records_only_single_generic_scalar_coefficient():
    controller = ModelAwareController("full", risk_threshold=1.0)
    controller.set_profile(_profile())
    controller.anchor_count = 2
    controller.audio_projection_ewma = 0.5
    controller.video_projection_ewma = -0.4
    decision = controller.decision(
        forecast_horizon=1.0,
        history_length=4,
        configured_degree=1,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.0,
        configured_video_blend=0.5,
    )
    replay = OfflineModelAwareDecision.from_runtime(decision)
    assert replay.audio_correction_coefficients == pytest.approx(
        (decision.audio_correction_gain,)
    )
    assert replay.video_correction_coefficients == pytest.approx(
        (decision.video_correction_gain,)
    )
    assert replay.correction_anchor_ids == ()


def test_debug_summary_marks_all_rejected_feature3_runtime_retired():
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    summary = runtime.debug_summary()
    assert "feature3_model_informed_correction=retired_no_material_gain" in summary
    assert "feature3_applied_correction=generic_scalar_latest_delta" in summary
    assert "feature3_k2_runtime=retired" in summary
    assert "feature3_transformed_trajectory_runtime=retired" in summary
    assert "feature3_previous_error_runtime=retired" in summary
    assert "feature3_direction_evidence_bytes=0" in summary
    assert "feature3_error_evidence_bytes=0" in summary
    assert "feature3_extra_transformer_nfe=0" in summary
