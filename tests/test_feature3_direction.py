from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.feature3_direction import (
    final_layer_jvp,
    final_layer_metric_direction,
    final_layer_vjp,
    radially_bound_direction,
    rmsnorm_jvp,
    rmsnorm_vjp,
    static_head_metric_direction,
)
from comfyui_spectrum_h3.model_aware import (
    ModelAwareForecastDecision,
    ModelForecastabilityProfile,
    ProfileLookup,
    SubspaceCorrectionTelemetry,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _profile(hidden: int = 4) -> ModelForecastabilityProfile:
    audio = torch.arange(1, 2 * hidden + 1, dtype=torch.float32).reshape(2, hidden)
    video = torch.arange(1, 3 * hidden + 1, dtype=torch.float32).reshape(3, hidden) / 3.0
    return ModelForecastabilityProfile(
        cache_key=("feature3-test", hidden),
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
        estimated_bytes=audio.numel() * 4 + video.numel() * 4,
        transient_workspace_bytes=0,
    )


def _runtime(mode: str, *, hidden: int = 4) -> SpectrumH3Runtime:
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
            model_aware_mode=mode,
            model_aware_risk_threshold=1.0,
        )
    )
    runtime.set_model_profile(ProfileLookup(_profile(hidden), False, 0.0))
    return runtime


def _call(hidden: int = 4):
    return SimpleNamespace(
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        expected_shape=(1, 2, hidden),
        labels=((0, "positive"),),
    )


def _step(coordinate: float = 1.0, step_id: int = 2, hidden: int = 4):
    return SimpleNamespace(
        coordinate=float(coordinate),
        step_id=int(step_id),
        calls=[_call(hidden)],
    )


def test_static_direction_matches_explicit_full_gram():
    torch.manual_seed(1)
    delta = torch.randn(3, 5)
    weight = torch.randn(4, 5)
    actual = static_head_metric_direction(delta, weight)
    expected = delta @ (weight.transpose(0, 1) @ weight)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_static_direction_preserves_off_diagonal_cross_channel_terms():
    delta = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    full = static_head_metric_direction(delta, weight)
    diagonal_only = delta * weight.square().sum(dim=0)
    assert full[0, 1] != 0.0
    assert not torch.equal(full, diagonal_only)


def test_static_direction_never_materializes_hidden_square_gram(monkeypatch):
    hidden = 5376
    delta = torch.ones(1, hidden)
    weight = torch.ones(96, hidden)
    real_matmul = torch.matmul
    seen = []

    def guarded(left, right, *args, **kwargs):
        seen.append((tuple(left.shape), tuple(right.shape)))
        assert tuple(left.shape[-2:]) != (hidden, hidden)
        assert tuple(right.shape[-2:]) != (hidden, hidden)
        return real_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", guarded)
    output = static_head_metric_direction(delta, weight)
    assert output.shape == delta.shape
    assert seen == [((1, hidden), (hidden, 96)), ((1, 96), (96, hidden))]


def test_rmsnorm_jvp_matches_autograd():
    torch.manual_seed(2)
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(x)
    weight = torch.randn(5, dtype=torch.float64)
    eps = 1e-5

    def fn(value):
        return torch.nn.functional.rms_norm(value, (5,), weight=weight, eps=eps)

    _, expected = torch.autograd.functional.jvp(fn, x, tangent)
    actual = rmsnorm_jvp(x.detach(), tangent, weight, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_rmsnorm_vjp_matches_autograd():
    torch.manual_seed(3)
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn_like(x)
    weight = torch.randn(5, dtype=torch.float64)
    eps = 1e-5
    output = torch.nn.functional.rms_norm(x, (5,), weight=weight, eps=eps)
    expected = torch.autograd.grad((output * cotangent).sum(), x)[0]
    actual = rmsnorm_vjp(x.detach(), cotangent, weight, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def _final_fn(x, weight, scale, shift, head, eps):
    normed = torch.nn.functional.rms_norm(
        x,
        (x.shape[-1],),
        weight=weight,
        eps=eps,
    )
    return ((normed * (1.0 + scale) + shift).to(torch.float32)) @ head.transpose(0, 1)


def test_final_layer_jvp_matches_autograd_and_native_one_plus_scale():
    torch.manual_seed(4)
    hidden, out = 5, 3
    x = torch.randn(2, hidden, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(x)
    norm_weight = torch.randn(hidden, dtype=torch.float64)
    scale = torch.randn(hidden, dtype=torch.float64)
    shift = torch.randn(hidden, dtype=torch.float64)
    head = torch.randn(out, hidden, dtype=torch.float32)
    eps = 1e-5
    fn = lambda value: _final_fn(value, norm_weight, scale, shift, head, eps)
    _, expected = torch.autograd.functional.jvp(fn, x, tangent)
    actual = final_layer_jvp(
        x.detach(),
        tangent,
        norm_weight=norm_weight,
        norm_eps=eps,
        adaln_scale=scale,
        head_weight=head,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_final_layer_vjp_matches_autograd():
    torch.manual_seed(5)
    hidden, out = 4, 3
    x = torch.randn(2, hidden, dtype=torch.float64, requires_grad=True)
    norm_weight = torch.randn(hidden, dtype=torch.float64)
    scale = torch.randn(hidden, dtype=torch.float64)
    shift = torch.randn(hidden, dtype=torch.float64)
    head = torch.randn(out, hidden, dtype=torch.float32)
    cotangent = torch.randn(2, out, dtype=torch.float32)
    eps = 1e-5
    output = _final_fn(x, norm_weight, scale, shift, head, eps)
    expected = torch.autograd.grad((output * cotangent).sum(), x)[0]
    actual = final_layer_vjp(
        x.detach(),
        cotangent,
        norm_weight=norm_weight,
        norm_eps=eps,
        adaln_scale=scale,
        head_weight=head,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_historical_jtj_helper_matches_explicit_autograd_jacobian():
    torch.manual_seed(6)
    hidden, out = 4, 3
    x = torch.randn(hidden, dtype=torch.float64)
    delta = torch.randn_like(x)
    norm_weight = torch.randn(hidden, dtype=torch.float64)
    scale = torch.randn(hidden, dtype=torch.float64)
    shift = torch.randn(hidden, dtype=torch.float64)
    head = torch.randn(out, hidden, dtype=torch.float32)
    eps = 1e-5

    def fn(value):
        return _final_fn(value, norm_weight, scale, shift, head, eps)

    jacobian = torch.autograd.functional.jacobian(fn, x).to(torch.float64)
    expected = jacobian.transpose(0, 1) @ (jacobian @ delta)
    actual, _, _ = final_layer_metric_direction(
        x,
        delta,
        norm_weight=norm_weight,
        norm_eps=eps,
        adaln_scale=scale,
        head_weight=head,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


def test_radial_bound_has_stable_tiny_q_limit_and_point_two_five_budget():
    delta = torch.tensor([[1.0, -2.0, 3.0]])
    tiny = radially_bound_direction(1e-12, delta, delta)
    assert tiny.eligible
    assert tiny.radial_scale == pytest.approx(1.0, abs=2e-7)
    assert tiny.bounded_norm_ratio == pytest.approx(1e-12, rel=2e-5, abs=1e-15)

    bounded = radially_bound_direction(1.0, delta, delta)
    expected_gain = 1.0 / (1.0 + 1.0 / 0.25)
    torch.testing.assert_close(bounded.correction, expected_gain * delta)
    assert bounded.direction_norm_ratio == pytest.approx(1.0)
    assert bounded.bounded_norm_ratio == pytest.approx(expected_gain)
    assert bounded.bound_active


@pytest.mark.parametrize(
    "direction",
    (
        torch.zeros(1, 3),
        torch.tensor([[float("nan"), 0.0, 1.0]]),
        torch.tensor([[float("inf"), 0.0, 1.0]]),
    ),
)
def test_zero_or_nonfinite_direction_fails_closed(direction):
    bounded = radially_bound_direction(0.2, direction, torch.ones_like(direction))
    assert not bounded.eligible
    assert bounded.bounded_norm_ratio == 0.0
    assert torch.equal(bounded.correction, torch.zeros_like(bounded.correction))


def test_schedule_mode_does_not_retain_feature2_or_feature3_evidence():
    forecaster = _runtime("schedule").forecaster
    forecaster._feature3_evidence_capture_mode = "schedule"
    feature = torch.ones(1, 2, 4)
    forecaster.update(
        0.0,
        feature,
        evidence_segments=(("audio", 0, 1), ("video", 1, 2)),
        exact_head_weights={
            "audio": torch.ones(2, 4),
            "video": torch.ones(3, 4),
        },
    )
    assert forecaster.generic_evidence_tensor_bytes == 0
    assert forecaster.exact_head_evidence_tensor_bytes == 0
    assert forecaster.model_aware_exact_head_projection_calls == 0


def test_schedule_confidence_retains_risk_evidence_but_no_exact_head_payload():
    forecaster = _runtime("schedule_confidence").forecaster
    forecaster._feature3_evidence_capture_mode = "schedule_confidence"
    feature = torch.ones(1, 2, 4)
    forecaster.update(
        0.0,
        feature,
        evidence_segments=(("audio", 0, 1), ("video", 1, 2)),
        exact_head_weights={
            "audio": torch.ones(2, 4),
            "video": torch.ones(3, 4),
        },
    )
    assert forecaster.generic_evidence_tensor_bytes > 0
    assert forecaster.exact_head_evidence_tensor_bytes == 0
    assert forecaster.model_aware_exact_head_projection_calls == 0


def test_schedule_confidence_does_no_k2_or_previous_error_work():
    runtime = _runtime("schedule_confidence")
    runtime.forecaster._feature3_evidence_capture_mode = "schedule_confidence"
    segments = (("audio", 0, 1), ("video", 1, 2))
    for coordinate, value in ((-1.0, 0.0), (0.0, 1.0)):
        runtime.forecaster.update(
            coordinate,
            torch.full((1, 2, 4), value),
            evidence_segments=segments,
        )
    runtime._observe_model_aware_anchor(
        _step(1.0),
        torch.tensor([[[2.2, 2.0, 1.8, 2.1], [1.9, 2.2, 2.0, 1.7]]]),
        {"audio": torch.ones(2, 4), "video": torch.ones(3, 4)},
        {"audio": torch.ones(4), "video": torch.ones(4)},
    )
    assert runtime.stats.model_aware_anchor_updates == 1
    assert runtime.stats.model_aware_subspace_solve_seconds == 0.0
    assert runtime.stats.model_aware_subspace_gram_seconds == 0.0
    assert runtime.stats.model_aware_exact_head_projection_calls == 0
    assert runtime.model_aware._feature3_error_previous == {}
    assert runtime.model_aware._feature3_error_compute_seconds == 0.0
    assert runtime.forecaster.model_aware_correction_seconds == 0.0


def test_full_applied_weight_path_ignores_retired_k2_coefficients():
    runtime = _runtime("full")
    for anchor_id, coordinate in enumerate((-1.0, 0.0, 0.5)):
        runtime.forecaster.update(
            coordinate,
            torch.full((1, 2, 4), float(anchor_id)),
            anchor_id=anchor_id,
        )
    decision = ModelAwareForecastDecision(
        trajectory_risk=0.0,
        model_risk=0.0,
        patch_risk=0.0,
        combined_risk=0.0,
        confidence=1.0,
        ridge_lambda=0.1,
        degree=1,
        audio_blend_weight=0.0,
        video_blend_weight=0.0,
        audio_correction_gain=0.1,
        video_correction_gain=-0.05,
        forecast_horizon=1.0,
        force_actual=False,
        audio_subspace_telemetry=SubspaceCorrectionTelemetry(
            eligible=True,
            used_scalar_fallback=False,
            applied_coefficients=(0.2, 0.2),
        ),
        video_subspace_telemetry=SubspaceCorrectionTelemetry(
            eligible=True,
            used_scalar_fallback=False,
            applied_coefficients=(-0.2, 0.1),
        ),
        correction_anchor_ids=(0, 1, 2),
    )
    weighted = runtime._model_aware_weight_segments(
        _call(),
        decision,
        coordinate=0.75,
    )
    expected_audio = runtime.forecaster.model_aware_weights(
        0.75,
        0.0,
        degree=1,
        ridge_lambda=0.1,
        correction_gain=0.1,
    )
    expected_video = runtime.forecaster.model_aware_weights(
        0.75,
        0.0,
        degree=1,
        ridge_lambda=0.1,
        correction_gain=-0.05,
    )
    torch.testing.assert_close(weighted[0][2], expected_audio)
    torch.testing.assert_close(weighted[1][2], expected_video)
    assert runtime.stats.model_aware_subspace_solve_seconds == 0.0


def test_debug_summary_marks_rejected_direction_runtime_retired():
    runtime = _runtime("full")
    summary = runtime.debug_summary()
    assert "feature3_applied_correction=scalar_latest_delta" in summary
    assert "feature3_transformed_trajectory_runtime=retired" in summary
    assert "feature3_direction_evidence_bytes=0" in summary
    assert "feature3_direction_workspace_bytes=0" in summary
    assert "feature3_jvp_enqueue_s=0.000000" in summary
    assert "feature3_vjp_enqueue_s=0.000000" in summary
    assert "feature3_previous_error_applied=false" in summary


def test_native_h3_final_layer_contract_matches_historical_math_assumptions():
    pytest.importorskip("comfy")
    from comfy.ldm.minimax.model import FinalLayer

    source = __import__("inspect").getsource(FinalLayer.forward)
    assert "1.0 + scale[vrow]" in source
    assert "1.0 + scale[arow]" in source
    assert ".to(torch.float32)" in source
    assert "self.video_out" in source
    assert "self.audio_out" in source
