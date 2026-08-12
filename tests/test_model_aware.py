from __future__ import annotations

import dataclasses
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.model_aware import (
    AnchorEvidence,
    ModelAwareController,
    ModelForecastabilityProfile,
    StreamAnchorEvidence,
    clear_model_profile_cache,
    get_model_forecastability_profile,
    radially_bound_coefficients,
)


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
    def __init__(
        self,
        inner,
        *,
        base_uuid="base",
        patches_uuid="patches",
        patches=None,
    ):
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


def _profile(**overrides) -> ModelForecastabilityProfile:
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
        "audio_head_weight": None,
        "video_head_weight": None,
        "audio_head_gram_diagonal": None,
        "video_head_gram_diagonal": None,
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


def test_base_profile_is_compact_scalar_state_and_reuses_clone_lineage():
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
    assert first.profile.audio_sensitivity > 0.0
    assert first.profile.video_sensitivity > 0.0
    assert first.profile.estimated_bytes <= 4096
    assert all(
        not torch.is_tensor(getattr(first.profile, field.name))
        for field in dataclasses.fields(first.profile)
    )
    assert first.profile.audio_head_weight is None
    assert first.profile.video_head_weight is None
    assert first.profile.audio_head_gram_diagonal is None
    assert first.profile.video_head_gram_diagonal is None

    reference = weakref.ref(original)
    del original
    gc.collect()
    assert reference() is None


def test_final_head_weights_still_contribute_to_compact_stream_sensitivity():
    inner = _Inner(hidden=4)
    with torch.no_grad():
        inner.final_layer.audio_out.weight.fill_(0.1)
        inner.final_layer.video_out.weight.fill_(1.0)
    profile = get_model_forecastability_profile(_Patcher(inner)).profile
    assert profile.video_sensitivity > profile.audio_sensitivity
    assert profile.audio_head_weight is None
    assert profile.video_head_weight is None


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
        _Patcher(
            inner,
            patches_uuid="unknown",
            patches={key: [(1.0, object(), 1.0, None, None)]},
        )
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

    patcher.injections = {
        "bypass_lora": [SimpleNamespace(inject=inject, eject=eject)]
    }
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
            AnchorEvidence(
                forecast_ratio=0.2,
                curvature_ratio=0.1,
                fit_condition=2.0,
                audio_projection=0.4,
                video_projection=-0.3,
                model_corrected_ratio=0.3,
                generic_corrected_ratio=0.4,
                audio=StreamAnchorEvidence(
                    forecast_ratio=0.2,
                    curvature_ratio=0.1,
                    residual_projection=0.4,
                    model_projection=0.4,
                    model_corrected_ratio=0.3,
                    generic_corrected_ratio=0.3,
                ),
                video=StreamAnchorEvidence(
                    forecast_ratio=0.2,
                    curvature_ratio=0.1,
                    residual_projection=-0.3,
                    model_projection=-0.3,
                    model_corrected_ratio=0.3,
                    generic_corrected_ratio=0.3,
                ),
            )
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
    assert calibrated.audio_correction_gain == pytest.approx(
        calibrated.audio_correction_telemetry.generic_gain
    )
    assert calibrated.video_correction_gain == pytest.approx(
        calibrated.video_correction_telemetry.generic_gain
    )


def test_mode_progression_preserves_public_mode_contract():
    profile = _profile(forecast_risk_prior=0.9)
    decisions = {}
    for mode in ("off", "schedule", "schedule_confidence", "full"):
        controller = ModelAwareController(mode, risk_threshold=0.5)
        controller.set_profile(profile)
        controller.anchor_count = 3
        controller.audio_projection_ewma = 0.5
        controller.video_projection_ewma = -0.4
        decisions[mode] = controller.decision(
            forecast_horizon=1.0,
            history_length=5,
            configured_degree=3,
            configured_ridge_lambda=0.1,
            configured_audio_blend=0.2,
            configured_video_blend=0.5,
        )

    assert not decisions["off"].force_actual
    assert decisions["schedule"].audio_correction_gain == 0.0
    assert decisions["schedule"].video_correction_gain == 0.0
    assert decisions["schedule_confidence"].audio_correction_gain == 0.0
    assert decisions["schedule_confidence"].video_correction_gain == 0.0
    assert decisions["full"].audio_correction_gain == pytest.approx(
        decisions["full"].audio_correction_telemetry.generic_gain
    )
    assert decisions["full"].video_correction_gain == pytest.approx(
        decisions["full"].video_correction_telemetry.generic_gain
    )


def test_generic_scalar_bound_is_monotonic_and_never_exceeds_limit():
    small = radially_bound_coefficients((0.01,), ((1.0,),), limit=0.25)
    medium = radially_bound_coefficients((0.25,), ((1.0,),), limit=0.25)
    large = radially_bound_coefficients((100.0,), ((1.0,),), limit=0.25)
    for bounded in (small, medium, large):
        assert bounded.eligible
        assert bounded.bounded_norm_ratio <= 0.25 + 1e-6
    assert (
        small.bounded_norm_ratio
        < medium.bounded_norm_ratio
        < large.bounded_norm_ratio
    )


def test_controller_snapshot_restore_preserves_live_generic_state():
    controller = ModelAwareController("full", risk_threshold=0.65)
    controller.set_profile(_profile())
    controller.observe_anchor(
        AnchorEvidence(
            forecast_ratio=0.8,
            curvature_ratio=0.2,
            fit_condition=1.5,
            audio_projection=0.35,
            video_projection=-0.25,
            audio=StreamAnchorEvidence(
                forecast_ratio=0.8,
                curvature_ratio=0.2,
                residual_projection=0.35,
            ),
            video=StreamAnchorEvidence(
                forecast_ratio=0.7,
                curvature_ratio=0.2,
                residual_projection=-0.25,
            ),
        )
    )
    state = controller.snapshot()
    expected = controller.decision(
        forecast_horizon=1.0,
        history_length=4,
        configured_degree=2,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.1,
        configured_video_blend=0.5,
    )
    controller.reset()
    controller.restore(state)
    restored = controller.decision(
        forecast_horizon=1.0,
        history_length=4,
        configured_degree=2,
        configured_ridge_lambda=0.1,
        configured_audio_blend=0.1,
        configured_video_blend=0.5,
    )
    assert restored.combined_risk == pytest.approx(expected.combined_risk)
    assert restored.ridge_lambda == pytest.approx(expected.ridge_lambda)
    assert restored.audio_correction_gain == pytest.approx(
        expected.audio_correction_gain
    )
    assert restored.video_correction_gain == pytest.approx(
        expected.video_correction_gain
    )
