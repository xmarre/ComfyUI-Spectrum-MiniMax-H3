from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import comfyui_spectrum_h3.feature3_previous_error as previous_error
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.feature3_direction import (
    FinalLayerGeometry,
    _final_layer_difference,
    final_layer_vjp,
)
from comfyui_spectrum_h3.feature3_direction_normalization import (
    normalize_direction_to_reference,
)
from comfyui_spectrum_h3.model_aware import (
    CorrectionGainTelemetry,
    ModelAwareForecastDecision,
    ModelForecastabilityProfile,
    ProfileLookup,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _heads(hidden: int = 4) -> dict[str, torch.Tensor]:
    return {
        "audio": torch.tensor(
            [[1.0, 0.5, -0.25, 0.75], [0.5, -1.0, 0.5, 0.25]],
            dtype=torch.float32,
        ),
        "video": torch.tensor(
            [
                [1.0, 0.0, 0.5, -0.5],
                [0.25, 1.0, -0.5, 0.5],
                [-0.5, 0.5, 1.0, 0.25],
            ],
            dtype=torch.float32,
        ),
    }


def _profile(hidden: int = 4) -> ModelForecastabilityProfile:
    heads = _heads(hidden)
    audio = heads["audio"]
    video = heads["video"]
    return ModelForecastabilityProfile(
        cache_key=("previous-error-test", hidden),
        base_model_identity="base",
        patch_identity="patch",
        active_patch_count=0,
        active_patch_keys=0,
        recognized_lora_count=0,
        unknown_patch_count=0,
        sampled_base_tensors=1,
        profile_confidence=1.0,
        aggregate_sensitivity=0.1,
        patch_perturbation=0.0,
        final_block_perturbation=0.0,
        audio_sensitivity=1.0,
        video_sensitivity=1.0,
        audio_head_weight=audio,
        video_head_weight=video,
        audio_head_gram_diagonal=audio.square().sum(dim=0),
        video_head_gram_diagonal=video.square().sum(dim=0),
        forecast_risk_prior=0.1,
        build_seconds=0.0,
        estimated_bytes=(audio.numel() + video.numel()) * 4,
        transient_workspace_bytes=0,
    )


def _runtime() -> SpectrumH3Runtime:
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            ridge_lambda=0.1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
            offline_smoothing_replay=False,
            model_aware_mode="full",
            model_aware_risk_threshold=1.0,
        )
    )
    runtime.set_model_profile(ProfileLookup(_profile(), False, 0.0))
    return runtime


def _call():
    return SimpleNamespace(
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        expected_shape=(1, 2, 4),
        labels=((0, "positive"),),
    )


def _step(step_id: int = 2):
    return SimpleNamespace(step_id=step_id, coordinate=1.0, calls=[_call()])


def _decision() -> ModelAwareForecastDecision:
    return ModelAwareForecastDecision(
        trajectory_risk=0.0,
        model_risk=0.0,
        patch_risk=0.0,
        combined_risk=0.0,
        confidence=1.0,
        ridge_lambda=0.1,
        degree=1,
        audio_blend_weight=0.0,
        video_blend_weight=0.0,
        audio_correction_gain=0.08,
        video_correction_gain=0.06,
        forecast_horizon=1.0,
        force_actual=False,
        audio_correction_telemetry=CorrectionGainTelemetry(
            generic_gain=0.07,
            model_candidate_gain=0.09,
            model_gain=0.08,
        ),
        video_correction_telemetry=CorrectionGainTelemetry(
            generic_gain=0.05,
            model_candidate_gain=0.07,
            model_gain=0.06,
        ),
    )


def _geometry() -> FinalLayerGeometry:
    return FinalLayerGeometry(
        norm_weight=torch.ones(4),
        norm_eps=1e-5,
        audio_scale=torch.tensor([0.1, -0.2, 0.3, 0.0]),
        video_scale=torch.tensor([-0.1, 0.2, 0.0, 0.4]),
    )


def _seed_history(runtime: SpectrumH3Runtime) -> None:
    for anchor_id, (coordinate, feature) in enumerate(
        (
            (-1.0, torch.tensor([[[0.0, 0.2, 0.4, 0.6], [0.1, 0.3, 0.5, 0.7]]])),
            (0.0, torch.tensor([[[0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6]]])),
        )
    ):
        runtime.forecaster.update(
            coordinate,
            feature,
            anchor_id=anchor_id,
        )


def _seed_previous_errors(runtime: SpectrumH3Runtime) -> dict[str, torch.Tensor]:
    prior_outputs = {
        "audio": torch.tensor([[[0.2, -0.1]]], dtype=torch.float32),
        "video": torch.tensor([[[0.1, -0.2, 0.3]]], dtype=torch.float32),
    }
    runtime.model_aware._feature3_error_previous = {
        "audio": previous_error.PreviousErrorState(
            1,
            torch.tensor([[[0.2, -0.1, 0.3, -0.2]]]),
            prior_outputs["audio"],
        ),
        "video": previous_error.PreviousErrorState(
            1,
            torch.tensor([[[-0.1, 0.2, 0.1, -0.3]]]),
            prior_outputs["video"],
        ),
    }
    for stream in ("audio", "video"):
        for kind in ("residual", "static", "full"):
            setattr(runtime.model_aware, f"_feature3_error_{stream}_{kind}_alpha_count", 1)
            setattr(runtime.model_aware, f"_feature3_error_{stream}_{kind}_alpha_ewma", 0.1)
    return prior_outputs


def test_static_output_adjoint_matches_explicit_transpose_action():
    torch.manual_seed(201)
    residual = torch.randn(2, 3, 5)
    weight = torch.randn(5, 7)
    actual = previous_error.static_output_adjoint(residual, weight)
    expected = residual @ weight
    torch.testing.assert_close(actual, expected)


def test_local_output_adjoint_matches_autograd_vjp():
    torch.manual_seed(202)
    hidden, out = 5, 3
    x = torch.randn(2, hidden, dtype=torch.float64, requires_grad=True)
    norm_weight = torch.randn(hidden, dtype=torch.float64)
    scale = torch.randn(hidden, dtype=torch.float64)
    shift = torch.randn(hidden, dtype=torch.float64)
    head = torch.randn(out, hidden, dtype=torch.float32)
    output_residual = torch.randn(2, out, dtype=torch.float32)
    eps = 1e-5
    normed = torch.nn.functional.rms_norm(
        x,
        (hidden,),
        weight=norm_weight,
        eps=eps,
    )
    output = ((normed * (1.0 + scale) + shift).to(torch.float32)) @ head.T
    expected = torch.autograd.grad((output * output_residual).sum(), x)[0]
    actual = final_layer_vjp(
        x.detach(),
        output_residual,
        norm_weight=norm_weight,
        norm_eps=eps,
        adaln_scale=scale,
        head_weight=head,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_exact_output_residual_is_native_final_layer_difference():
    torch.manual_seed(203)
    actual = torch.randn(2, 4)
    predicted = torch.randn(2, 4)
    geometry = _geometry()
    head = _heads()["audio"]
    scale = geometry.audio_scale
    measured = _final_layer_difference(
        actual,
        predicted,
        geometry=geometry,
        adaln_scale=scale,
        head_weight=head,
    )
    norm_a = torch.nn.functional.rms_norm(
        actual,
        (4,),
        weight=geometry.norm_weight,
        eps=geometry.norm_eps,
    )
    norm_p = torch.nn.functional.rms_norm(
        predicted,
        (4,),
        weight=geometry.norm_weight,
        eps=geometry.norm_eps,
    )
    expected = ((norm_a - norm_p) * (1.0 + scale)).to(torch.float32) @ head.T
    torch.testing.assert_close(measured, expected)


def test_row_correspondence_is_stable_and_capped_at_32_complete_rows():
    runtime = _runtime()
    indices = previous_error._row_indices(
        runtime.model_aware,
        runtime.forecaster,
        "video",
        100,
        2,
    )
    assert indices.device.type == "cpu"
    assert 2 * indices.numel() <= 32
    torch.testing.assert_close(
        indices,
        previous_error._row_indices(
            runtime.model_aware,
            runtime.forecaster,
            "video",
            100,
            2,
        ),
    )
    with pytest.raises(ValueError, match="row correspondence changed"):
        previous_error._row_indices(
            runtime.model_aware,
            runtime.forecaster,
            "video",
            101,
            2,
        )


def test_positive_rescaling_of_error_directions_does_not_change_normalized_candidate():
    direction = torch.tensor([[0.2, -0.4, 0.8, 0.3]])
    delta = torch.tensor([[1.0, 0.5, -0.25, 0.75]])
    base = normalize_direction_to_reference(direction, delta)
    scaled = normalize_direction_to_reference(1e-9 * direction, delta)
    assert base.eligible and scaled.eligible
    torch.testing.assert_close(base.direction, scaled.direction, rtol=3e-5, atol=3e-6)


def test_previous_anchor_state_is_used_and_current_anchor_is_published_only_after_scoring(monkeypatch):
    runtime = _runtime()
    _seed_history(runtime)
    prior_outputs = _seed_previous_errors(runtime)
    runtime._feature3_error_pending_geometry = {2: {0: _geometry()}}
    captured = []
    real_static = previous_error.static_output_adjoint

    def capture_static(output_residual, head_weight):
        captured.append(output_residual.detach().clone())
        return real_static(output_residual, head_weight)

    monkeypatch.setattr(previous_error, "static_output_adjoint", capture_static)
    heads = _heads()
    raw_weights = {
        "audio": torch.tensor([-1.0, 2.0]),
        "video": torch.tensor([-1.0, 2.0]),
    }
    combined = torch.tensor(
        [[[1.7, 1.4, 1.0, 0.8], [1.6, 1.5, 1.1, 0.9]]],
        dtype=torch.float32,
    )
    calls_before = runtime.stats.actual_transformer_calls
    previous_error._observe_previous_error_anchor(
        runtime,
        _step(),
        combined,
        _decision(),
        raw_weights,
        heads,
    )
    assert runtime.stats.actual_transformer_calls == calls_before
    assert len(captured) == 2
    assert any(torch.equal(value, prior_outputs["audio"]) for value in captured)
    assert any(torch.equal(value, prior_outputs["video"]) for value in captured)
    assert runtime.model_aware._feature3_error_previous["audio"].anchor_id == 2
    assert runtime.model_aware._feature3_error_previous["video"].anchor_id == 2
    assert not torch.equal(
        runtime.model_aware._feature3_error_previous["audio"].output_residual,
        prior_outputs["audio"],
    )
    residual, static, full = runtime.model_aware._feature3_error_last["audio"]
    assert residual.eligible
    assert static.eligible
    assert full.eligible
    assert residual.normalized_direction_norm_ratio == pytest.approx(1.0, rel=3e-5)
    assert static.normalized_direction_norm_ratio == pytest.approx(1.0, rel=3e-5)
    assert full.normalized_direction_norm_ratio == pytest.approx(1.0, rel=3e-5)


def test_previous_hidden_residual_is_an_explicit_generic_baseline():
    runtime = _runtime()
    _seed_history(runtime)
    _seed_previous_errors(runtime)
    runtime._feature3_error_pending_geometry = {2: {0: _geometry()}}
    previous_error._observe_previous_error_anchor(
        runtime,
        _step(),
        torch.tensor([[[1.6, 1.3, 1.0, 0.8], [1.5, 1.4, 1.1, 0.9]]]),
        _decision(),
        {"audio": torch.tensor([-1.0, 2.0]), "video": torch.tensor([-1.0, 2.0])},
        _heads(),
    )
    summary = runtime.debug_summary()
    assert "feature3_previous_error_screen=residual_vs_static_adjoint_vs_local_adjoint" in summary
    assert "residual_vs_generic" in summary
    assert "static_vs_residual" in summary
    assert "full_vs_residual" in summary
    assert "feature3_previous_error_applied=false" in summary


def test_previous_error_state_snapshot_restore_is_exact():
    runtime = _runtime()
    prior = _seed_previous_errors(runtime)
    runtime.model_aware._feature3_error_audio_residual_alpha_ewma = 0.25
    snapshot = runtime.model_aware.snapshot()
    runtime.model_aware._feature3_error_previous = {}
    runtime.model_aware._feature3_error_audio_residual_alpha_ewma = -1.0
    runtime.model_aware.restore(snapshot)
    assert runtime.model_aware._feature3_error_audio_residual_alpha_ewma == pytest.approx(0.25)
    restored = runtime.model_aware._feature3_error_previous["audio"]
    assert restored.anchor_id == 1
    torch.testing.assert_close(restored.output_residual, prior["audio"])


def test_previous_error_evidence_retains_only_sampled_tensors_and_has_bounded_size():
    runtime = _runtime()
    _seed_previous_errors(runtime)
    runtime.model_aware._feature3_error_row_indices = {
        "audio": torch.arange(16),
        "video": torch.arange(16),
    }
    byte_count = previous_error._evidence_bytes(runtime.model_aware)
    expected = sum(
        state.tensor_bytes
        for state in runtime.model_aware._feature3_error_previous.values()
    ) + 32 * torch.tensor([], dtype=torch.int64).element_size()
    assert byte_count == expected
    assert all(
        torch.is_tensor(value)
        for state in runtime.model_aware._feature3_error_previous.values()
        for value in (state.hidden_residual, state.output_residual)
        if value is not None
    )


def test_previous_error_hook_does_not_run_in_schedule_confidence_or_replay():
    runtime = _runtime()
    _seed_history(runtime)
    _seed_previous_errors(runtime)
    before = dict(runtime.model_aware._feature3_error_previous)
    runtime._offline_phase = "replay"
    previous_error._observe_previous_error_anchor(
        runtime,
        _step(),
        torch.ones(1, 2, 4),
        _decision(),
        {"audio": torch.tensor([-1.0, 2.0]), "video": torch.tensor([-1.0, 2.0])},
        _heads(),
    )
    assert runtime.model_aware._feature3_error_previous == before
    assert runtime.model_aware._feature3_error_compute_seconds == 0.0
