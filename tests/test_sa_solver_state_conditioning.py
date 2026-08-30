from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import sampling as sampling_module
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    SpectrumH3Binding,
    _sa_solver_expected_model_calls,
    _sa_solver_call_topology,
    _sa_solver_is_active_pece,
    _sa_solver_is_stochastic,
    _sa_solver_option_contract,
    _sa_solver_prefix_model_calls,
    _sa_solver_sampler_contract,
    _sa_solver_tau_values,
    _run_solver_aware_sa,
    outer_sample_wrapper,
    sampler_supports_seeded_replay,
)


def _sampler(function_name: str, options: dict | None = None):
    def sampler_function():
        return None

    sampler_function.__name__ = function_name
    return SimpleNamespace(
        sampler_function=sampler_function,
        extra_options={} if options is None else dict(options),
    )


def _native_sa_functions():
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for native SA-Solver tests")
    try:
        import comfy.k_diffusion.sampling as native_sampling
    except ImportError as exc:
        pytest.skip(f"optional ComfyUI runtime dependency is unavailable: {exc}")
    return {
        name: getattr(native_sampling, name)
        for name in ("sample_sa_solver", "sample_sa_solver_pece")
    }


class _ModelSampling:
    noise_scale = 1.0

    @staticmethod
    def percent_to_sigma(percent):
        return 1.0 - float(percent)


class _Patcher:
    @staticmethod
    def get_model_object(name):
        assert name == "model_sampling"
        return _ModelSampling()


def _run_native(
    *,
    function_name="sample_sa_solver",
    sigmas=(0.9, 0.7, 0.5, 0.3, 0.0),
    use_pece=False,
    predictor_order=3,
    corrector_order=4,
    simple_order_2=False,
    tau_func="piecewise",
    s_noise=0.7,
    noise_scale=1.0,
):
    native = _native_sa_functions()[function_name]
    generator = torch.Generator(device="cpu").manual_seed(8675309)
    noise_draws = []
    noise_args = []
    callbacks = []
    model_events = []
    tau_args = []

    def noise_sampler(sigma, sigma_next):
        noise_args.append((float(sigma), float(sigma_next)))
        value = torch.randn((1, 2), generator=generator, dtype=torch.float32)
        noise_draws.append(value.clone())
        return value

    if tau_func == "piecewise":
        def configured_tau(sigma):
            value = float(sigma)
            tau_args.append(value)
            return 0.8 if 0.3 <= value <= 0.7 else 0.0
    elif tau_func == "zero":
        def configured_tau(sigma):
            value = float(sigma)
            tau_args.append(value)
            return 0.0
    elif tau_func is None:
        configured_tau = None
    else:
        raise AssertionError(f"unknown tau fixture {tau_func!r}")

    model_sampling = _ModelSampling()
    model_sampling.noise_scale = noise_scale
    patcher = SimpleNamespace(get_model_object=lambda _name: model_sampling)

    class Model:
        inner_model = SimpleNamespace(model_patcher=patcher)

        def __call__(self, x, sigma, **_extra_args):
            coordinate = float(sigma.detach().cpu().item())
            model_events.append((x.detach().clone(), coordinate))
            return torch.full_like(x, 0.25 + 0.5 * coordinate)

    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
    kwargs = dict(
        extra_args={"seed": 123},
        callback=lambda state: callbacks.append(
            {
                "x": state["x"].detach().clone(),
                "i": state["i"],
                "sigma": float(state["sigma"]),
                "sigma_hat": float(state["sigma_hat"]),
                "denoised": state["denoised"].detach().clone(),
            }
        ),
        disable=True,
        tau_func=configured_tau,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        simple_order_2=simple_order_2,
    )
    if function_name == "sample_sa_solver":
        kwargs["use_pece"] = use_pece
    result = native(
        Model(),
        torch.ones((1, 2), dtype=torch.float32),
        sigma_tensor,
        **kwargs,
    )
    return SimpleNamespace(
        result=result,
        callbacks=callbacks,
        model_events=model_events,
        noise_draws=noise_draws,
        noise_args=noise_args,
        tau_args=tau_args,
        rng_state=generator.get_state().clone(),
    )


def _run_isolated(
    modes,
    *,
    sigmas=(0.9, 0.7, 0.5, 0.3, 0.0),
    tau_func="piecewise",
    s_noise=0.7,
    predictor_order=3,
    corrector_order=4,
    simple_order_2=False,
    continuum_active=False,
    use_pece=False,
    forecast_offset=0.0,
    noise_scale=1.0,
    debug=False,
    reasons=None,
):
    generator = torch.Generator(device="cpu").manual_seed(8675309)
    noise_draws = []
    noise_args = []
    callbacks = []
    model_events = []
    tau_args = []
    runtime = SimpleNamespace(
        last_completed_mode=None,
        last_completed_step_id=None,
        last_completed_reason=None,
        config=SimpleNamespace(debug=bool(debug)),
    )
    mode_iter = iter(modes)
    reason_iter = iter(reasons) if reasons is not None else None

    def noise_sampler(sigma, sigma_next):
        noise_args.append((float(sigma), float(sigma_next)))
        value = torch.randn((1, 2), generator=generator, dtype=torch.float32)
        noise_draws.append(value.clone())
        return value

    if tau_func == "piecewise":
        def configured_tau(sigma):
            value = float(sigma)
            tau_args.append(value)
            return 0.8 if 0.3 <= value <= 0.7 else 0.0
    elif tau_func == "zero":
        def configured_tau(sigma):
            value = float(sigma)
            tau_args.append(value)
            return 0.0
    elif tau_func is None:
        configured_tau = None
    else:
        raise AssertionError(f"unknown tau fixture {tau_func!r}")

    model_sampling = _ModelSampling()
    model_sampling.noise_scale = noise_scale
    patcher = SimpleNamespace(get_model_object=lambda _name: model_sampling)

    class Model:
        inner_model = SimpleNamespace(model_patcher=patcher)

        def __call__(self, x, sigma, **_extra_args):
            coordinate = float(sigma.detach().cpu().item())
            model_events.append((x.detach().clone(), coordinate))
            runtime.last_completed_mode = next(mode_iter)
            runtime.last_completed_step_id = len(model_events) - 1
            runtime.last_completed_reason = (
                next(reason_iter) if reason_iter is not None else None
            )
            offset = forecast_offset if runtime.last_completed_mode == "forecast" else 0.0
            return torch.full_like(x, 0.25 + 0.5 * coordinate + offset)

    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32)
    result = sampling_module._sample_sa_solver_forecast_isolated(
        runtime,
        Model(),
        torch.ones((1, 2), dtype=torch.float32),
        sigma_tensor,
        extra_args={"seed": 123},
        callback=lambda state: callbacks.append(
            {
                "x": state["x"].detach().clone(),
                "i": state["i"],
                "sigma": float(state["sigma"]),
                "sigma_hat": float(state["sigma_hat"]),
                "denoised": state["denoised"].detach().clone(),
            }
        ),
        disable=True,
        tau_func=configured_tau,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        use_pece=use_pece,
        simple_order_2=simple_order_2,
        continuum_active=continuum_active,
    )
    return SimpleNamespace(
        result=result,
        callbacks=callbacks,
        model_events=model_events,
        noise_draws=noise_draws,
        noise_args=noise_args,
        tau_args=tau_args,
        runtime=runtime,
        rng_state=generator.get_state().clone(),
    )


def test_sa_dense_output_holds_latest_when_only_one_actual_anchor():
    latest = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [latest],
        [torch.tensor(0.2)],
        [3],
        target_lambda=torch.tensor(0.3),
        target_step=4,
    )

    torch.testing.assert_close(predicted, latest, rtol=0, atol=0)
    assert predicted.data_ptr() != latest.data_ptr()
    assert mode == "latest_actual_hold"
    assert anchors == (3,)
    assert float(alpha) == pytest.approx(0.0)


def test_sa_dense_output_continuum_holds_latest_even_with_valid_slope():
    previous = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    latest = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [previous, latest],
        [torch.tensor(0.2), torch.tensor(0.4)],
        [1, 3],
        target_lambda=torch.tensor(0.5),
        target_step=4,
        continuum_active=True,
    )

    torch.testing.assert_close(predicted, latest, rtol=0, atol=0)
    assert predicted.data_ptr() != latest.data_ptr()
    assert mode == "latest_actual_hold_continuum"
    assert anchors == (3,)
    assert float(alpha) == pytest.approx(0.0)


def test_continuum_uses_solver_space_hold_on_nonstochastic_forecast():
    pytest.importorskip("scipy")
    run = _run_isolated(
        ["actual", "actual", "forecast", "actual"],
        tau_func="zero",
        s_noise=0.7,
        continuum_active=True,
    )

    # The raw synthetic model value at sigma=0.5 would be 0.5. Continuum must
    # ignore it even though tau is zero and reuse the exact step-1 denoised
    # value (0.25 + 0.5*0.7 = 0.6).
    torch.testing.assert_close(
        run.callbacks[2]["denoised"],
        torch.full((1, 2), 0.6, dtype=torch.float32),
        rtol=0,
        atol=1e-6,
    )
    assert run.runtime.last_completed_mode == "actual"


def test_sa_dense_output_holds_after_consecutive_actual_refreshes():
    previous = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    latest = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [previous, latest],
        [torch.tensor(0.2), torch.tensor(0.3)],
        [7, 8],
        target_lambda=torch.tensor(0.4),
        target_step=9,
    )

    torch.testing.assert_close(predicted, latest, rtol=0, atol=0)
    assert mode == "latest_actual_hold_consecutive_anchors"
    assert anchors == (8,)
    assert float(alpha) == pytest.approx(0.0)



def test_sa_dense_output_allows_consecutive_pece_endpoint_extrapolation():
    previous = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    latest = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [previous, latest],
        [torch.tensor(0.2), torch.tensor(0.3)],
        [7, 8],
        target_lambda=torch.tensor(0.4),
        target_step=9,
        allow_consecutive_extrapolation=True,
    )

    torch.testing.assert_close(
        predicted,
        torch.tensor([[3.0, 5.0]], dtype=torch.float32),
        rtol=0,
        atol=1e-6,
    )
    assert mode == "lambda_bounded_extrapolation"
    assert anchors == (7, 8)
    assert float(alpha) == pytest.approx(1.0)

def test_sa_dense_output_uses_bounded_lambda_extrapolation_from_actuals():
    previous = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    latest = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [previous, latest],
        [torch.tensor(0.2), torch.tensor(0.4)],
        [1, 3],
        target_lambda=torch.tensor(0.5),
        target_step=4,
    )

    # alpha=(0.5-0.4)/(0.4-0.2)=0.5, so latest + 0.5*(latest-previous).
    torch.testing.assert_close(
        predicted,
        torch.tensor([[2.5, 6.0]], dtype=torch.float32),
        rtol=0,
        atol=1e-6,
    )
    assert mode == "lambda_bounded_extrapolation"
    assert anchors == (1, 3)
    assert float(alpha) == pytest.approx(0.5)


def test_sa_dense_output_guards_overlong_extrapolation_to_latest_hold():
    previous = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    latest = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    predicted, mode, anchors, alpha = sampling_module._sa_predict_causal_denoised(
        [previous, latest],
        [torch.tensor(0.2), torch.tensor(0.21)],
        [1, 3],
        target_lambda=torch.tensor(0.5),
        target_step=4,
    )

    torch.testing.assert_close(predicted, latest, rtol=0, atol=0)
    assert mode == "lambda_bounded_extrapolation"
    assert anchors == (1, 3)
    assert float(alpha) == pytest.approx(0.0)


def test_isolated_sa_adapter_is_exact_native_parity_when_every_call_is_actual():
    pytest.importorskip("scipy")
    native = _run_native()
    isolated = _run_isolated(["actual"] * 4)

    assert torch.allclose(isolated.result, native.result, atol=0.0, rtol=0.0)
    assert isolated.noise_args == pytest.approx(native.noise_args)
    assert isolated.tau_args == pytest.approx(native.tau_args)
    assert len(isolated.noise_draws) == len(native.noise_draws)
    for candidate, expected in zip(
        isolated.noise_draws,
        native.noise_draws,
        strict=True,
    ):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    torch.testing.assert_close(isolated.rng_state, native.rng_state, rtol=0, atol=0)
    assert len(isolated.model_events) == len(native.model_events)
    for candidate, expected in zip(
        isolated.model_events,
        native.model_events,
        strict=True,
    ):
        assert candidate[1] == pytest.approx(expected[1])
        torch.testing.assert_close(candidate[0], expected[0], rtol=0, atol=0)
    assert len(isolated.callbacks) == len(native.callbacks)
    for candidate, expected in zip(isolated.callbacks, native.callbacks, strict=True):
        assert candidate["i"] == expected["i"]
        assert candidate["sigma"] == pytest.approx(expected["sigma"])
        assert candidate["sigma_hat"] == pytest.approx(expected["sigma_hat"])
        assert torch.allclose(candidate["x"], expected["x"], atol=0.0, rtol=0.0)
        assert torch.allclose(
            candidate["denoised"],
            expected["denoised"],
            atol=0.0,
            rtol=0.0,
        )


def test_active_pece_adapter_is_exact_native_parity_when_every_phase_is_actual():
    pytest.importorskip("scipy")
    native = _run_native(use_pece=True)
    isolated = _run_isolated(["actual"] * 7, use_pece=True)

    torch.testing.assert_close(isolated.result, native.result, rtol=0, atol=0)
    assert isolated.noise_args == pytest.approx(native.noise_args)
    assert isolated.tau_args == pytest.approx(native.tau_args)
    assert len(isolated.noise_draws) == len(native.noise_draws)
    for candidate, expected in zip(
        isolated.noise_draws,
        native.noise_draws,
        strict=True,
    ):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    torch.testing.assert_close(isolated.rng_state, native.rng_state, rtol=0, atol=0)
    assert len(isolated.model_events) == len(native.model_events) == 7
    for candidate, expected in zip(
        isolated.model_events,
        native.model_events,
        strict=True,
    ):
        assert candidate[1] == pytest.approx(expected[1])
        torch.testing.assert_close(candidate[0], expected[0], rtol=0, atol=0)
    assert len(isolated.callbacks) == len(native.callbacks) == 4
    for candidate, expected in zip(isolated.callbacks, native.callbacks, strict=True):
        assert candidate["i"] == expected["i"]
        assert candidate["sigma"] == pytest.approx(expected["sigma"])
        assert candidate["sigma_hat"] == pytest.approx(expected["sigma_hat"])
        torch.testing.assert_close(candidate["x"], expected["x"], rtol=0, atol=0)
        torch.testing.assert_close(
            candidate["denoised"],
            expected["denoised"],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("predictor_order", "corrector_order", "tau_func", "s_noise", "simple_order_2"),
    (
        (1, 1, "piecewise", 0.7, False),
        (1, 4, "zero", 0.7, False),
        (2, 1, "piecewise", 0.7, False),
        (2, 2, "piecewise", 0.7, True),
        (3, 2, "piecewise", 0.0, False),
        (3, 3, "zero", 0.7, False),
        (3, 4, "piecewise", 0.0, False),
        (3, 4, None, 0.7, False),
    ),
)
def test_active_pece_all_actual_parity_across_order_and_stochasticity(
    predictor_order,
    corrector_order,
    tau_func,
    s_noise,
    simple_order_2,
):
    pytest.importorskip("scipy")
    native = _run_native(
        use_pece=True,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        tau_func=tau_func,
        s_noise=s_noise,
        simple_order_2=simple_order_2,
    )
    isolated = _run_isolated(
        ["actual"] * 7,
        use_pece=True,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        tau_func=tau_func,
        s_noise=s_noise,
        simple_order_2=simple_order_2,
    )

    torch.testing.assert_close(isolated.result, native.result, rtol=0, atol=0)
    assert isolated.noise_args == pytest.approx(native.noise_args)
    assert isolated.tau_args == pytest.approx(native.tau_args)



def test_active_pece_continuum_predicted_forecast_uses_corrected_endpoint_hold():
    pytest.importorskip("scipy")
    run = _run_isolated(
        ["actual", "actual", "actual", "forecast", "actual", "actual", "actual"],
        use_pece=True,
        tau_func="zero",
        s_noise=0.0,
        continuum_active=True,
        forecast_offset=50.0,
    )

    # P2's raw feature forecast would be 50.5. Continuum instead holds C1,
    # the exact persistent PECE endpoint at outer step 1. In this fixture C1's
    # exact denoiser is 0.25 + 0.5*0.7 = 0.6.
    torch.testing.assert_close(
        run.callbacks[2]["denoised"],
        torch.full((1, 2), 0.6),
        rtol=0,
        atol=1e-6,
    )


def test_active_pece_debug_telemetry_exposes_phase_value_and_persistence(caplog):
    pytest.importorskip("scipy")
    with caplog.at_level("WARNING"):
        _run_isolated(
            ["actual", "actual", "actual", "forecast", "actual", "actual", "actual"],
            use_pece=True,
            tau_func="zero",
            s_noise=0.0,
            forecast_offset=25.0,
            debug=True,
        )

    text = caplog.text
    assert "logical_step=3 sa_outer=2 sa_phase=predicted" in text
    assert "actual_model_evaluation=0 forecast=1" in text
    assert "reason=predicted_forecast_ephemeral" in text
    assert "adams_persistent=0 solver_value_source=lambda_bounded_extrapolation" in text
    assert "scope=pece_all_forecasts mode=lambda_bounded_extrapolation" in text
    assert "raw_feature_forecast=ignored" in text
    assert "history_lane=persistent_endpoint_actual_only" in text
    assert "logical_step=4 sa_outer=2 sa_phase=corrected" in text
    assert "reason=pece_exact_reanchor adams_persistent=1" in text
    assert "solver_value_source=actual_h3" in text

def test_active_pece_all_actual_parity_preserves_model_noise_scale():
    pytest.importorskip("scipy")
    native = _run_native(use_pece=True, noise_scale=1.75)
    isolated = _run_isolated(
        ["actual"] * 7,
        use_pece=True,
        noise_scale=1.75,
    )

    torch.testing.assert_close(isolated.result, native.result, rtol=0, atol=0)
    assert isolated.noise_args == pytest.approx(native.noise_args)
    assert isolated.tau_args == pytest.approx(native.tau_args)


def test_active_pece_all_actual_parity_without_terminal_zero():
    pytest.importorskip("scipy")
    sigmas = (0.9, 0.7, 0.5, 0.3, 0.1)
    native = _run_native(use_pece=True, sigmas=sigmas)
    isolated = _run_isolated(
        ["actual"] * 7,
        use_pece=True,
        sigmas=sigmas,
    )

    torch.testing.assert_close(isolated.result, native.result, rtol=0, atol=0)
    assert isolated.noise_args == pytest.approx(native.noise_args)
    assert isolated.tau_args == pytest.approx(native.tau_args)



def test_active_pece_predicted_forecast_is_current_corrector_only(monkeypatch):
    """The solver-owned predicted estimate may enter C and cannot persist past C."""
    pytest.importorskip("scipy")
    import comfy.k_diffusion.sa_solver as native_sa_solver

    original_coefficients = native_sa_solver.compute_stochastic_adams_b_coeffs
    original_tensordot = torch.tensordot
    pending_kind = None
    integrations = []
    bridge_sentinel = 37.5

    def fake_dense_output(
        actual_preds,
        actual_lambdas,
        actual_steps,
        *,
        target_lambda,
        target_step,
        continuum_active=False,
        allow_consecutive_extrapolation=False,
    ):
        assert actual_preds
        assert allow_consecutive_extrapolation
        return (
            torch.full_like(actual_preds[-1], bridge_sentinel),
            "test_endpoint_bridge",
            (actual_steps[-1],),
            target_lambda.new_zeros(()),
        )

    def capture_coefficients(*args, **kwargs):
        nonlocal pending_kind
        pending_kind = "corrector" if kwargs.get("is_corrector_step") else "predictor"
        return original_coefficients(*args, **kwargs)

    def capture_tensordot(input_tensor, other, *args, **kwargs):
        nonlocal pending_kind
        integrations.append((pending_kind, input_tensor.detach().clone()))
        pending_kind = None
        return original_tensordot(input_tensor, other, *args, **kwargs)

    monkeypatch.setattr(
        sampling_module,
        "_sa_predict_causal_denoised",
        fake_dense_output,
    )
    monkeypatch.setattr(
        native_sa_solver,
        "compute_stochastic_adams_b_coeffs",
        capture_coefficients,
    )
    monkeypatch.setattr(torch, "tensordot", capture_tensordot)

    # P2 is the only forecast. Its raw hidden forecast would produce 50.5, but
    # active PECE must ignore that solver value and use the endpoint bridge.
    _run_isolated(
        ["actual", "actual", "actual", "forecast", "actual", "actual", "actual"],
        use_pece=True,
        tau_func="zero",
        s_noise=0.0,
        forecast_offset=50.0,
    )

    bridge_integrations = [
        (kind, matrix)
        for kind, matrix in integrations
        if bool(torch.isclose(matrix, torch.tensor(bridge_sentinel)).any().item())
    ]
    assert len(bridge_integrations) == 1
    assert bridge_integrations[0][0] == "corrector"

    raw_feature_sentinel = 50.5
    assert not any(
        bool(torch.isclose(matrix, torch.tensor(raw_feature_sentinel)).any().item())
        for _, matrix in integrations
    )


def test_active_pece_supports_forecasted_predicted_phase_every_outer_step_after_p0():
    pytest.importorskip("scipy")
    run = _run_isolated(
        ["actual", "forecast", "actual", "forecast", "actual", "forecast", "actual"],
        use_pece=True,
        tau_func="zero",
        s_noise=0.0,
        forecast_offset=100.0,
    )

    assert len(run.model_events) == 7
    assert len(run.callbacks) == 4
    # Raw feature forecasts are ~100.x in this fixture. Solver-visible callback
    # denoisers must come only from exact persistent PECE endpoint anchors.
    for callback in run.callbacks[1:]:
        assert float(callback["denoised"].abs().max()) < 10.0



def test_active_pece_hard_transition_restarts_dense_endpoint_interpolation():
    pytest.importorskip("scipy")
    run = _run_isolated(
        ["actual", "forecast", "actual", "actual", "actual", "forecast", "actual"],
        use_pece=True,
        tau_func="zero",
        s_noise=0.0,
        forecast_offset=100.0,
        reasons=[
            None,
            None,
            None,
            "external patch hard sigma transition",
            None,
            None,
            None,
        ],
    )

    # P2 is exact because it crosses the hard transformer boundary. Once C2
    # becomes the exact endpoint, the dense secant history is restarted there.
    # P3 must therefore hold C2 rather than extrapolate across pre-transition C1.
    torch.testing.assert_close(
        run.callbacks[3]["denoised"],
        torch.full((1, 2), 0.5),
        rtol=0,
        atol=1e-6,
    )


def test_active_pece_rejects_a_forecasted_corrected_phase():
    pytest.importorskip("scipy")
    with pytest.raises(RuntimeError, match="exact reanchor policy"):
        _run_isolated(
            ["actual", "actual", "forecast"],
            sigmas=(0.9, 0.7, 0.0),
            use_pece=True,
            tau_func="zero",
            s_noise=0.0,
        )


def test_active_pece_rejects_forecasting_both_phases():
    pytest.importorskip("scipy")
    with pytest.raises(RuntimeError, match="exact reanchor policy"):
        _run_isolated(
            ["actual", "actual", "actual", "forecast", "forecast"],
            sigmas=(0.9, 0.7, 0.5, 0.0),
            use_pece=True,
            tau_func="zero",
            s_noise=0.0,
        )


def test_active_pece_forecast_preserves_native_tau_and_noise_call_order():
    pytest.importorskip("scipy")
    native = _run_native(use_pece=True)
    accelerated = _run_isolated(
        ["actual", "actual", "actual", "forecast", "actual", "actual", "actual"],
        use_pece=True,
        forecast_offset=25.0,
    )

    assert accelerated.tau_args == pytest.approx(native.tau_args)
    assert accelerated.noise_args == pytest.approx(native.noise_args)
    assert len(accelerated.noise_draws) == len(native.noise_draws)
    for candidate, expected in zip(
        accelerated.noise_draws,
        native.noise_draws,
        strict=True,
    ):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.rng_state, native.rng_state, rtol=0, atol=0)


def test_isolated_sa_adapter_uses_forecast_in_current_corrector_but_not_next_history(monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("torchsde")
    import comfy.k_diffusion.sa_solver as native_sa_solver
    import comfy.k_diffusion.sampling as native_sampling

    original = native_sa_solver.compute_stochastic_adams_b_coeffs
    calls = []

    def capture(
        sigma_next,
        curr_lambdas,
        lambda_s,
        lambda_t,
        tau_t,
        simple_order_2=False,
        is_corrector_step=False,
    ):
        calls.append(
            {
                "curr": curr_lambdas.detach().clone(),
                "lambda_t": lambda_t.detach().clone(),
                "corrector": bool(is_corrector_step),
            }
        )
        return original(
            sigma_next,
            curr_lambdas,
            lambda_s,
            lambda_t,
            tau_t,
            simple_order_2,
            is_corrector_step=is_corrector_step,
        )

    monkeypatch.setattr(
        native_sa_solver,
        "compute_stochastic_adams_b_coeffs",
        capture,
    )
    # Keep the following exact endpoint non-terminal so native PEC executes
    # its current corrector there; a terminal zero intentionally suppresses the
    # PEC corrector.
    sigmas = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1], dtype=torch.float32)
    lambdas = native_sampling.sigma_to_half_log_snr(
        sigmas,
        model_sampling=_ModelSampling(),
    )

    _run_isolated(
        ["actual", "actual", "forecast", "actual"],
        sigmas=tuple(float(value) for value in sigmas),
        s_noise=0.0,
    )

    # The forecast at step 2 is the current endpoint estimate and therefore
    # participates in that step's PEC correction.
    step2_correctors = [
        call
        for call in calls
        if call["corrector"] and torch.allclose(call["lambda_t"], lambdas[2])
    ]
    assert len(step2_correctors) == 1
    step2 = step2_correctors[0]["curr"]
    assert any(torch.allclose(value, lambdas[2]) for value in step2)

    # The same forecast is ephemeral: once step 2 completes, lambda[2] must not
    # survive into the persistent Adams stencil used by the next exact step.
    step3_correctors = [
        call
        for call in calls
        if call["corrector"] and torch.allclose(call["lambda_t"], lambdas[3])
    ]
    assert len(step3_correctors) == 1
    step3 = step3_correctors[0]["curr"]
    assert any(torch.allclose(value, lambdas[3]) for value in step3)
    assert not any(torch.allclose(value, lambdas[2]) for value in step3)


@pytest.mark.parametrize(
    ("function_name", "use_pece", "corrector_order", "sigmas", "expected_calls"),
    (
        ("sample_sa_solver", False, 4, (0.9,), 0),
        ("sample_sa_solver", False, 4, (0.9, 0.0), 1),
        ("sample_sa_solver", False, 4, (0.9, 0.7, 0.5, 0.0), 3),
        ("sample_sa_solver", True, 4, (0.9, 0.7, 0.5, 0.0), 5),
        ("sample_sa_solver", True, 0, (0.9, 0.7, 0.5, 0.0), 3),
        ("sample_sa_solver_pece", False, 4, (0.9, 0.7, 0.5, 0.0), 5),
        ("sample_sa_solver_pece", False, 0, (0.9, 0.7, 0.5, 0.0), 3),
        ("sample_sa_solver", True, 4, (0.9, 0.7, 0.5, 0.3), 5),
    ),
)
def test_compiled_native_sa_solver_proves_exact_model_call_topology(
    function_name,
    use_pece,
    corrector_order,
    sigmas,
    expected_calls,
):
    run = _run_native(
        function_name=function_name,
        sigmas=sigmas,
        use_pece=use_pece,
        corrector_order=corrector_order,
    )

    assert len(run.model_events) == expected_calls
    assert len(run.callbacks) == max(0, len(sigmas) - 1)


def test_pece_second_evaluation_has_same_sigma_different_state_and_no_callback():
    run = _run_native(use_pece=True)

    model_sigmas = [event[1] for event in run.model_events]
    callback_sigmas = [event["sigma"] for event in run.callbacks]
    assert model_sigmas == pytest.approx([0.9, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3])
    assert callback_sigmas == pytest.approx([0.9, 0.7, 0.5, 0.3])
    assert not torch.equal(run.model_events[1][0], run.model_events[2][0])
    assert not torch.equal(run.model_events[3][0], run.model_events[4][0])


@pytest.mark.parametrize("simple_order_2", (False, True))
def test_simple_order_2_does_not_change_sa_model_call_or_noise_topology(simple_order_2):
    run = _run_native(simple_order_2=simple_order_2)

    assert len(run.model_events) == 4
    assert len(run.callbacks) == 4
    assert len(run.noise_draws) == 3


def test_native_sa_noise_is_consumed_by_the_next_callback_visible_model_call():
    run = _run_native(
        sigmas=(0.9, 0.7, 0.5, 0.3, 0.0),
        tau_func="piecewise",
        s_noise=0.7,
    )

    assert len(run.model_events) == 4
    assert len(run.callbacks) == 4
    # tau(0.7), tau(0.5), and tau(0.3) are all active in this fixture.
    # Native SA adds each corresponding draw to x_pred after the predictor, so
    # the following outer model/callback consumes that fresh stochastic state.
    assert len(run.noise_draws) == 3
    for step_id in (1, 2, 3):
        assert torch.equal(
            run.model_events[step_id][0],
            run.callbacks[step_id]["x"],
        )
        assert not torch.equal(
            run.model_events[step_id][0],
            run.model_events[step_id - 1][0],
        )


def test_tau_zero_suppresses_noise_without_extra_tau_or_noise_calls():
    run = _run_native(tau_func="zero")

    assert run.tau_args == pytest.approx([0.7, 0.5, 0.3])
    assert run.noise_args == []
    assert run.noise_draws == []


def test_piecewise_tau_transitions_preserve_native_noise_order_and_arguments():
    run = _run_native()

    assert run.tau_args == pytest.approx([0.7, 0.5, 0.3])
    assert len(run.noise_args) == 3
    for candidate, expected in zip(
        run.noise_args,
        ((0.9, 0.7), (0.7, 0.5), (0.5, 0.3)),
        strict=True,
    ):
        assert candidate == pytest.approx(expected)
    assert len(run.noise_draws) == 3


def test_tau_none_uses_native_default_interval_without_extra_validation_calls():
    run = _run_native(tau_func=None)

    assert run.tau_args == []
    assert len(run.noise_draws) == 3


def test_review_rejects_custom_tau_without_invoking_it():
    calls = []

    def custom_tau(sigma):
        calls.append(sigma)
        return 0.0

    reason = _sa_solver_option_contract("sample_sa_solver", {"tau_func": custom_tau})

    assert reason is not None
    assert "tau_func" in reason
    assert calls == []


def test_review_accepts_native_tau_closure_and_reads_eta_without_executing_it():
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for native SA tau provenance")
    from comfy.k_diffusion.sa_solver import get_tau_interval_func

    tau_func = get_tau_interval_func(0.8, 0.2, eta=0.75)
    values = _sa_solver_tau_values(tau_func)

    assert values is not None
    assert values["eta"] == pytest.approx(0.75)
    assert _sa_solver_option_contract("sample_sa_solver", {"tau_func": tau_func}) is None


@pytest.mark.parametrize(
    ("options", "stochastic"),
    (
        ({}, True),
        ({"s_noise": 0.0}, False),
    ),
)
def test_sa_stochastic_gate_handles_default_and_disabled_noise(options, stochastic):
    assert _sa_solver_is_stochastic(_sampler("sample_sa_solver", options)) is stochastic


def test_sa_stochastic_gate_reads_native_eta_zero_closure():
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for native SA tau provenance")
    from comfy.k_diffusion.sa_solver import get_tau_interval_func

    sampler = _sampler(
        "sample_sa_solver",
        {"tau_func": get_tau_interval_func(0.8, 0.2, eta=0.0)},
    )
    assert _sa_solver_is_stochastic(sampler) is False


def test_custom_noise_sampler_is_accepted_without_speculative_draws():
    calls = []

    def noise_sampler(sigma, sigma_next):
        calls.append((sigma, sigma_next))
        return torch.zeros((1, 2))

    assert (
        _sa_solver_option_contract(
            "sample_sa_solver",
            {"noise_sampler": noise_sampler},
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize(
    ("function_name", "options", "sigmas", "expected"),
    (
        ("sample_sa_solver", {}, [0.9, 0.6, 0.3, 0.0], 3),
        ("sample_sa_solver", {}, [0.9], 0),
        ("sample_sa_solver", {"use_pece": True}, [0.9, 0.6, 0.3, 0.0], 5),
        (
            "sample_sa_solver",
            {"use_pece": True, "corrector_order": 0},
            [0.9, 0.6, 0.3, 0.0],
            3,
        ),
        ("sample_sa_solver_pece", {}, [0.9, 0.6, 0.3], 3),
        (
            "sample_sa_solver_pece",
            {"corrector_order": 0},
            [0.9, 0.6, 0.3],
            2,
        ),
    ),
)
def test_sa_model_call_accounting_tracks_pec_and_pece(
    function_name,
    options,
    sigmas,
    expected,
):
    sampler = _sampler(function_name, options)

    assert _sa_solver_expected_model_calls(sampler, torch.tensor(sigmas)) == expected


def test_active_pece_call_topology_has_explicit_same_coordinate_phase_identity():
    sampler = _sampler(
        "sample_sa_solver_pece",
        {"corrector_order": 4},
    )
    topology = _sa_solver_call_topology(
        sampler,
        torch.tensor([0.9, 0.7, 0.5, 0.0]),
    )

    assert topology is not None
    assert [
        (entry.outer_step, entry.stage_index, entry.phase)
        for entry in topology
    ] == [
        (0, 0, "predicted"),
        (1, 0, "predicted"),
        (1, 1, "corrected"),
        (2, 0, "predicted"),
        (2, 1, "corrected"),
    ]
    assert _sa_solver_is_active_pece(sampler)


def test_sa_continuum_prefix_is_outer_step_identity_for_reviewed_pec():
    sampler = _sampler("sample_sa_solver")

    assert (
        _sa_solver_prefix_model_calls(
            sampler,
            torch.tensor([0.9, 0.7, 0.5, 0.3, 0.0]),
            3,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("function_name", "options", "message"),
    (
        ("sample_sa_solver", {"predictor_order": True}, "predictor_order"),
        ("sample_sa_solver", {"corrector_order": -1}, "corrector_order"),
        ("sample_sa_solver", {"s_noise": float("nan")}, "s_noise"),
        ("sample_sa_solver", {"noise_sampler": object()}, "noise_sampler"),
        ("sample_sa_solver", {"tau_func": lambda _sigma: 0.0}, "tau_func"),
    ),
)
def test_sa_unsupported_options_fail_closed(function_name, options, message):
    reason = _sa_solver_option_contract(function_name, options)

    assert reason is not None
    assert message in reason


def test_native_sa_sampler_contract_accepts_pec_and_pece_options():
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for native SA sampler provenance")
    pytest.importorskip("torchsde")
    try:
        import comfy.samplers
    except ImportError as exc:
        pytest.skip(f"optional ComfyUI runtime dependency is unavailable: {exc}")
    from comfy.k_diffusion.sa_solver import get_tau_interval_func

    tau_func = get_tau_interval_func(0.8, 0.2, eta=0.75)
    pec = comfy.samplers.ksampler(
        "sa_solver",
        {
            "tau_func": tau_func,
            "noise_sampler": lambda _sigma, _sigma_next: torch.zeros((1, 2)),
            "predictor_order": 3,
            "corrector_order": 4,
            "use_pece": False,
            "simple_order_2": True,
        },
    )
    inert_pece = comfy.samplers.ksampler(
        "sa_solver_pece",
        {"tau_func": tau_func, "corrector_order": 0},
    )
    active_pece = comfy.samplers.ksampler(
        "sa_solver_pece",
        {"tau_func": tau_func, "corrector_order": 4},
    )

    assert _sa_solver_sampler_contract(pec) == (True, None)
    assert _sa_solver_sampler_contract(inert_pece) == (True, None)
    assert _sa_solver_sampler_contract(active_pece) == (True, None)


@pytest.mark.parametrize(
    ("sampler_name", "options"),
    (
        ("sa_solver", {"use_pece": True, "corrector_order": 4}),
        ("sa_solver_pece", {"corrector_order": 4}),
    ),
)
def test_native_sa_sampler_contract_accepts_active_pece(sampler_name, options):
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for native SA sampler provenance")
    pytest.importorskip("torchsde")
    try:
        import comfy.samplers
    except ImportError as exc:
        pytest.skip(f"optional ComfyUI runtime dependency is unavailable: {exc}")

    accepted, reason = _sa_solver_sampler_contract(
        comfy.samplers.ksampler(sampler_name, options)
    )

    assert accepted
    assert reason is None


def test_generic_sa_pece_option_is_consumed_before_adapter_forwarding(monkeypatch):
    calls = []

    class FakeSampler:
        def __init__(self):
            def sample_sa_solver():
                return None

            sample_sa_solver.__name__ = "sample_sa_solver"
            self.sampler_function = sample_sa_solver
            self.extra_options = {
                "use_pece": True,
                "corrector_order": 4,
                "predictor_order": 3,
                "simple_order_2": False,
                "s_noise": 0.0,
            }
            self.inpaint_options = {}

        def sample(
            self,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        ):
            return self.sampler_function(
                model_wrap,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable_pbar,
                **self.extra_options,
            )

    sampler = FakeSampler()

    class Executor:
        class_obj = sampler
        wrappers = (object(),)

        def __call__(self, *args, **kwargs):
            raise AssertionError("native fallback must not be used")

    def isolated(
        runtime,
        model,
        x,
        sigmas,
        *,
        use_pece,
        corrector_order,
        predictor_order,
        simple_order_2,
        s_noise,
        **kwargs,
    ):
        calls.append(
            {
                "use_pece": use_pece,
                "corrector_order": corrector_order,
                "predictor_order": predictor_order,
                "simple_order_2": simple_order_2,
                "s_noise": s_noise,
                "kwargs": kwargs,
            }
        )
        return x

    monkeypatch.setattr(
        sampling_module,
        "_sample_sa_solver_forecast_isolated",
        isolated,
    )

    runtime = SimpleNamespace(
        disable_forecasting_for_run=lambda _reason: None,
        config=SimpleNamespace(debug=False),
    )
    result = _run_solver_aware_sa(
        Executor(),
        runtime,
        object(),
        torch.tensor([1.0, 0.0]),
        {},
        None,
        torch.ones(1),
        None,
        None,
        True,
    )

    torch.testing.assert_close(result, torch.ones(1))
    assert calls == [
        {
            "use_pece": True,
            "corrector_order": 4,
            "predictor_order": 3,
            "simple_order_2": False,
            "s_noise": 0.0,
            "kwargs": {
                "extra_args": {},
                "callback": None,
                "disable": True,
                "continuum_active": False,
            },
        }
    ]


def test_sa_solver_replay_is_intentionally_causal_only():
    assert not sampler_supports_seeded_replay(_sampler("sample_sa_solver"))
    assert not sampler_supports_seeded_replay(
        _sampler("sample_sa_solver", {"s_noise": 0.0})
    )


def test_sa_outer_wrapper_enters_state_conditioned_spectrum(monkeypatch, caplog):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="off",
            offline_smoothing_replay=False,
            debug=True,
        )
    )
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {},
        },
        model_patcher=_Patcher(),
    )
    sampler = _sampler("sample_sa_solver")
    calls = []

    monkeypatch.setattr(
        sampling_module,
        "_native_sa_solver_preflight_reason",
        lambda _sampler, _model_options: None,
    )

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            calls.append(
                (
                    runtime.active_run_id,
                    runtime.state_conditioned_residual,
                    runtime.stochastic_multistage,
                    runtime.stats.total_steps,
                    args,
                    kwargs,
                )
            )
            return "state-conditioned-sa"

    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            torch.ones(1),
            torch.zeros(1),
            sampler,
            torch.tensor([1.0, 0.7, 0.4, 0.0]),
            seed=17,
        )

    assert result == "state-conditioned-sa"
    assert len(calls) == 1
    assert calls[0][0] is not None
    assert calls[0][1] is True
    assert calls[0][2] is False
    assert calls[0][3] == 3
    assert runtime.active_run_id is None
    assert "feature_geometry=state_conditioned_residual" in caplog.text


def test_sa_offline_request_runs_one_causal_spectrum_pass(monkeypatch, caplog):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="off",
            offline_smoothing_replay=True,
        )
    )
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {},
        },
        model_patcher=_Patcher(),
    )
    sampler = _sampler("sample_sa_solver")
    active_runs = []

    monkeypatch.setattr(
        sampling_module,
        "_native_sa_solver_preflight_reason",
        lambda _sampler, _model_options: None,
    )

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            active_runs.append(
                (
                    runtime.active_run_id,
                    runtime.state_conditioned_residual,
                    runtime.offline_phase,
                )
            )
            return "causal-sa"

    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            torch.ones(1),
            torch.zeros(1),
            sampler,
            torch.tensor([1.0, 0.7, 0.4, 0.0]),
            seed=17,
        )

    assert result == "causal-sa"
    assert len(active_runs) == 1
    assert active_runs[0][0] is not None
    assert active_runs[0][1] is True
    assert active_runs[0][2] is None
    assert "running one causal Spectrum pass" in caplog.text



def test_active_pece_outer_wrapper_enters_phase_aware_runtime(monkeypatch):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(model_aware_mode="off", offline_smoothing_replay=False)
    )
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {},
        },
        model_patcher=_Patcher(),
    )
    sampler = _sampler(
        "sample_sa_solver",
        {"use_pece": True, "corrector_order": 4, "s_noise": 0.0},
    )
    calls = []

    monkeypatch.setattr(
        sampling_module,
        "_native_sa_solver_preflight_reason",
        lambda _sampler, _model_options: None,
    )

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            run = runtime._run
            assert run is not None
            calls.append(
                {
                    "run_id": runtime.active_run_id,
                    "total_steps": runtime.stats.total_steps,
                    "stage_count": run.stage_count,
                    "separate_stage_histories": run.separate_stage_histories,
                    "forecastable": run.forecastable_stage_indices,
                    "history_stages": run.history_stage_indices,
                    "history_steps": run.history_step_ids,
                    "tail_stages": run.tail_actual_stage_indices,
                    "bootstrap": run.allow_state_conditioned_bootstrap,
                    "topology": run.logical_call_topology,
                    "model_aware_force_actual": run.model_aware_can_force_actual,
                }
            )
            return "phase-aware-pece"

    result = outer_sample_wrapper(
        Executor(),
        torch.ones(1),
        torch.zeros(1),
        sampler,
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        seed=17,
    )

    assert result == "phase-aware-pece"
    assert len(calls) == 1
    call = calls[0]
    assert call["run_id"] == 1
    assert call["total_steps"] == 5
    assert call["stage_count"] == 2
    assert call["separate_stage_histories"] is False
    assert call["forecastable"] == (0,)
    assert call["history_stages"] == frozenset({0, 1})
    assert call["history_steps"] == frozenset({0, 2, 4})
    assert call["tail_stages"] == frozenset({1})
    assert call["bootstrap"] is True
    assert [entry.phase for entry in call["topology"]] == [
        "predicted",
        "predicted",
        "corrected",
        "predicted",
        "corrected",
    ]
    assert call["model_aware_force_actual"] is False
    assert runtime.active_run_id is None

