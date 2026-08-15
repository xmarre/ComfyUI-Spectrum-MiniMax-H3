from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.er_sde_stochastic import (
    ERSDEStepDescriptor,
    ERSDEStochasticTracker,
    ERSDETrackingError,
)
from comfyui_spectrum_h3.runtime import OfflineReplayAbort, SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    ACTUAL_KEY,
    BINDING_KEY,
    ER_SDE_TRACKER_KEY,
    RUN_ID_KEY,
    STEP_ID_KEY,
    SpectrumH3Binding,
    _er_sde_tracking_contract,
    outer_sample_wrapper,
    predict_noise_wrapper,
    sampler_sample_wrapper,
)
import comfyui_spectrum_h3.sampling as sampling_module


def _descriptor(
    step_id: int,
    *,
    mode: str = "forecast",
    replay_source_actual: bool | None = None,
) -> ERSDEStepDescriptor:
    return ERSDEStepDescriptor(
        run_id=7,
        step_id=step_id,
        mode=mode,
        replay_source_actual=replay_source_actual,
        requires_compensation=(
            mode == "forecast"
            or (mode == "replay" and replay_source_actual is False)
        ),
    )


def _tracker(
    noise: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> ERSDEStochasticTracker:
    sample = torch.tensor([[2.0, -3.0]], dtype=dtype) if noise is None else noise
    return ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: sample,
        noise_scaler=lambda value: value**2,
        effective_s_noise=0.5,
        max_stage=3,
        debug=False,
        run_id=7,
    )


def _produce_pending(tracker: ERSDEStochasticTracker) -> torch.Tensor:
    er_lambda_t = torch.tensor(0.5)
    er_lambda_s = torch.tensor(1.0)
    tracker.noise_scaler(er_lambda_t)
    tracker.noise_scaler(er_lambda_s)
    noise = tracker.noise_sampler(torch.tensor(0.8), torch.tensor(0.4))
    r = er_lambda_t**2 / er_lambda_s**2
    expected = (torch.tensor(0.4) / er_lambda_t) * noise
    expected = expected * 0.5
    expected = expected * (
        er_lambda_t**2 - er_lambda_s**2 * r**2
    ).sqrt().nan_to_num(nan=0.0)
    return expected


def test_first_model_result_has_no_pending_noise_and_is_unchanged():
    tracker = _tracker()
    denoised = torch.tensor([[4.0, 5.0]])

    assert tracker.consume(denoised, _descriptor(0)) is denoised


def test_forecast_subtracts_exact_pending_increment_once():
    tracker = _tracker()
    expected_q = _produce_pending(tracker)
    raw = torch.tensor([[4.0, 5.0]]) + expected_q

    corrected = tracker.consume(raw, _descriptor(1))

    torch.testing.assert_close(corrected, torch.tensor([[4.0, 5.0]]), rtol=0, atol=0)
    assert not tracker.has_pending
    with pytest.raises(ERSDETrackingError, match="no preceding"):
        tracker.consume(raw, _descriptor(1))


@pytest.mark.parametrize(
    ("mode", "replay_source_actual"),
    (("actual", None), ("replay", True)),
)
def test_state_aware_actual_or_replay_anchor_consumes_without_correction(
    mode, replay_source_actual
):
    tracker = _tracker()
    _produce_pending(tracker)
    denoised = torch.tensor([[4.0, 5.0]])

    result = tracker.consume(
        denoised,
        _descriptor(
            1,
            mode=mode,
            replay_source_actual=replay_source_actual,
        ),
    )

    assert result is denoised
    assert not tracker.has_pending


def test_replay_of_smoothed_step_is_compensated():
    tracker = _tracker()
    expected_q = _produce_pending(tracker)
    base = torch.tensor([[4.0, 5.0]])

    corrected = tracker.consume(
        base + expected_q,
        _descriptor(1, mode="replay", replay_source_actual=False),
    )

    torch.testing.assert_close(corrected, base, rtol=0, atol=0)


def test_stale_step_cannot_consume_pending_increment():
    tracker = _tracker()
    _produce_pending(tracker)

    with pytest.raises(ERSDETrackingError, match="targets step 1"):
        tracker.consume(torch.ones((1, 2)), _descriptor(2))

    assert not tracker.has_pending


def test_packed_shape_dtype_and_device_must_match_without_broadcasting():
    tracker = _tracker(dtype=torch.float16)
    _produce_pending(tracker)

    with pytest.raises(ERSDETrackingError, match="exactly match"):
        tracker.consume(torch.ones((1, 2, 1), dtype=torch.float16), _descriptor(1))

    assert not tracker.has_pending


def test_noise_sample_is_returned_unchanged_and_pending_owns_new_storage():
    noise = torch.tensor([[2.0, -3.0]])
    tracker = _tracker(noise)

    _produce_pending(tracker)

    assert tracker.has_pending
    assert noise.tolist() == [[2.0, -3.0]]
    assert tracker._pending is not None
    assert tracker._pending.value.data_ptr() != noise.data_ptr()


def test_clear_releases_pending_state_between_generations():
    tracker = _tracker()
    _produce_pending(tracker)
    tracker.clear()

    assert not tracker.has_pending
    assert tracker.consume(torch.ones((1, 2)), _descriptor(0)) is not None


def test_second_noise_interval_before_consumption_fails_without_overwriting_q():
    tracker = _tracker()
    _produce_pending(tracker)
    tracker.noise_scaler(torch.tensor(0.25))
    tracker.noise_scaler(torch.tensor(0.5))

    with pytest.raises(ERSDETrackingError, match="second stochastic increment"):
        tracker.noise_sampler(torch.tensor(0.4), torch.tensor(0.2))

    assert tracker.pending_step_id == 1
    tracker.clear()


def _native_er_sde_function():
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for the synthetic native ER-SDE test")
    source_path = Path(comfyui_path) / "comfy/k_diffusion/sampling.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_er_sde"
    )
    function = copy.deepcopy(function)
    function.decorator_list = []
    compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "torch": torch,
        "trange": lambda count, disable=None: range(count),
        "default_noise_sampler": object(),
        "offset_first_sigma_for_snr": lambda sigmas, _sampling: sigmas,
        "sigma_to_half_log_snr": lambda sigmas, _sampling: -torch.log(sigmas.sqrt()),
    }
    exec(  # noqa: S102 - execute the reviewed native function in a closed test namespace
        compile(compiled, "<native sample_er_sde>", "exec"), namespace
    )
    return namespace["sample_er_sde"]


def _seeded_native_run(descriptor_factory=None):
    native_er_sde = _native_er_sde_function()
    model_sampling = SimpleNamespace(noise_scale=1.0)
    generator = torch.Generator(device="cpu").manual_seed(8675309)
    noise_draws = []
    callbacks = []

    class Patcher:
        def get_model_object(self, name):
            assert name == "model_sampling"
            return model_sampling

    def base_noise_sampler(_sigma, _sigma_next):
        noise = torch.randn((1, 2), generator=generator, dtype=torch.float32)
        noise_draws.append(noise.clone())
        return noise

    tracker = None
    if descriptor_factory is not None:
        tracker = ERSDEStochasticTracker(
            noise_sampler=base_noise_sampler,
            noise_scaler=lambda value: value**2,
            effective_s_noise=0.7,
            max_stage=3,
            debug=False,
            run_id=7,
        )
    model_calls = 0

    class Model:
        inner_model = SimpleNamespace(model_patcher=Patcher())

        def __call__(self, x, sigma, **_extra_args):
            nonlocal model_calls
            sigma_view = sigma.reshape(-1, *([1] * (x.ndim - 1)))
            denoised = x * 0.25 + sigma_view * 0.05
            if tracker is not None:
                descriptor = descriptor_factory(model_calls)
                if descriptor.requires_compensation:
                    assert tracker._pending is not None
                    denoised = denoised + tracker._pending.value
                denoised = tracker.consume(denoised, descriptor)
            model_calls += 1
            return denoised

    result = native_er_sde(
        Model(),
        torch.ones((1, 2), dtype=torch.float32),
        torch.tensor([0.9, 0.7, 0.5, 0.3, 0.0]),
        callback=lambda state: callbacks.append(state["denoised"].clone()),
        disable=True,
        s_noise=0.7,
        noise_sampler=(tracker.noise_sampler if tracker is not None else base_noise_sampler),
        noise_scaler=(tracker.noise_scaler if tracker is not None else lambda value: value**2),
        max_stage=3,
    )
    if tracker is not None:
        assert not tracker.has_pending
        assert tracker.noise_calls == len(noise_draws)
        tracker.clear()
    return result, callbacks, noise_draws


def test_all_actual_tracking_is_bitwise_identical_to_native_er_sde():
    native = _seeded_native_run()
    tracked = _seeded_native_run(
        lambda step_id: _descriptor(step_id, mode="actual")
    )

    assert torch.equal(native[0], tracked[0])
    assert all(
        torch.equal(native_value, tracked_value)
        for native_value, tracked_value in zip(native[1], tracked[1], strict=True)
    )
    assert all(
        torch.equal(native_value, tracked_value)
        for native_value, tracked_value in zip(native[2], tracked[2], strict=True)
    )


def test_seeded_first_pass_and_replay_consume_identical_stochastic_streams():
    source_actual = (True, True, False, True)
    first_pass = _seeded_native_run(
        lambda step_id: _descriptor(
            step_id,
            mode="actual" if source_actual[step_id] else "forecast",
        )
    )
    replay = _seeded_native_run(
        lambda step_id: _descriptor(
            step_id,
            mode="replay",
            replay_source_actual=source_actual[step_id],
        )
    )

    assert torch.equal(first_pass[0], replay[0])
    assert len(first_pass[2]) == len(replay[2]) == 3
    assert all(
        torch.equal(first_value, replay_value)
        for first_value, replay_value in zip(first_pass[1], replay[1], strict=True)
    )
    assert all(
        torch.equal(first_value, replay_value)
        for first_value, replay_value in zip(first_pass[2], replay[2], strict=True)
    )


def _synthetic_solver_run(max_stage: int, *, compensate: bool, state_aware: bool = False):
    native_er_sde = _native_er_sde_function()
    model_sampling = SimpleNamespace(noise_scale=1.0)
    latest_q = None
    callbacks = []

    class Patcher:
        def get_model_object(self, name):
            assert name == "model_sampling"
            return model_sampling

    deterministic_noise = torch.tensor([[0.75, -0.5]], dtype=torch.float32)

    def base_noise_sampler(sigma, sigma_next):
        nonlocal latest_q
        er_lambda_s = sigma.sqrt()
        er_lambda_t = sigma_next.sqrt()
        r = er_lambda_t**2 / er_lambda_s**2
        latest_q = (sigma_next / er_lambda_t) * deterministic_noise
        latest_q = latest_q * (
            er_lambda_t**2 - er_lambda_s**2 * r**2
        ).sqrt().nan_to_num(nan=0.0)
        return deterministic_noise

    tracker = ERSDEStochasticTracker(
        noise_sampler=base_noise_sampler,
        noise_scaler=lambda value: value**2,
        effective_s_noise=1.0,
        max_stage=max_stage,
        debug=False,
        run_id=7,
    )
    model_calls = 0

    class Model:
        inner_model = SimpleNamespace(model_patcher=Patcher())

        def __call__(self, x, _sigma, **_extra_args):
            nonlocal model_calls
            base = torch.full_like(x, 0.125)
            if model_calls > 0 and not state_aware:
                assert latest_q is not None
                raw = base + latest_q
            else:
                raw = base
            if compensate or state_aware:
                mode = "forecast" if compensate else "actual"
                raw = tracker.consume(raw, _descriptor(model_calls, mode=mode))
            model_calls += 1
            return raw

    tracked = compensate or state_aware
    result = native_er_sde(
        Model(),
        torch.zeros((1, 2), dtype=torch.float32),
        torch.tensor([0.9, 0.7, 0.5, 0.3, 0.0]),
        callback=lambda state: callbacks.append(state["denoised"].clone()),
        disable=True,
        noise_sampler=tracker.noise_sampler if tracked else base_noise_sampler,
        noise_scaler=tracker.noise_scaler if tracked else (lambda value: value**2),
        max_stage=max_stage,
    )
    if tracked:
        assert not tracker.has_pending
    tracker.clear()
    return result, callbacks


def test_native_stage_one_callback_exposes_direct_leak_and_compensation_removes_it():
    raw_result, raw_callbacks = _synthetic_solver_run(1, compensate=False)
    corrected_result, corrected_callbacks = _synthetic_solver_run(1, compensate=True)
    control_result, control_callbacks = _synthetic_solver_run(
        1, compensate=False, state_aware=True
    )

    assert not torch.equal(raw_callbacks[1], control_callbacks[1])
    torch.testing.assert_close(
        corrected_callbacks[1], control_callbacks[1], rtol=0, atol=2e-8
    )
    torch.testing.assert_close(corrected_result, control_result, rtol=0, atol=0)
    assert not torch.equal(raw_result, control_result)


@pytest.mark.parametrize("max_stage", (2, 3))
def test_corrected_denoised_enters_higher_order_history(max_stage):
    corrected_result, corrected_callbacks = _synthetic_solver_run(
        max_stage, compensate=True
    )
    control_result, control_callbacks = _synthetic_solver_run(
        max_stage, compensate=False, state_aware=True
    )

    torch.testing.assert_close(corrected_result, control_result, rtol=0, atol=0)
    for corrected, control in zip(corrected_callbacks, control_callbacks, strict=True):
        torch.testing.assert_close(corrected, control, rtol=0, atol=2e-8)


def test_native_s_noise_zero_is_exact_noop_without_noise_or_q_allocation():
    native_er_sde = _native_er_sde_function()
    model_sampling = SimpleNamespace(noise_scale=1.0)
    noise_calls = []

    class Patcher:
        def get_model_object(self, _name):
            return model_sampling

    class Model:
        inner_model = SimpleNamespace(model_patcher=Patcher())

        def __call__(self, x, _sigma, **_extra_args):
            return torch.full_like(x, 0.25)

    result = native_er_sde(
        Model(),
        torch.zeros((1, 2)),
        torch.tensor([0.9, 0.6, 0.3, 0.0]),
        disable=True,
        s_noise=0.0,
        noise_sampler=lambda *_args: noise_calls.append(True),
        noise_scaler=lambda value: value**2,
    )

    assert torch.isfinite(result).all()
    assert noise_calls == []


def test_current_native_er_sde_runtime_contract_is_accepted():
    if not os.environ.get("COMFYUI_PATH"):
        pytest.skip("COMFYUI_PATH is required for the native runtime contract")
    try:
        import comfy.samplers
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional ComfyUI runtime dependency is unavailable: {exc}")

    supported, reason = _er_sde_tracking_contract(
        comfy.samplers.ksampler("er_sde")
    )

    assert supported, reason


def _complete_runtime_step(runtime: SpectrumH3Runtime, decision, feature_value: float):
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        labels=((0, "positive"),),
        expected_shape=(1, 2, 1),
    )
    if actual:
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((1, 2, 1), feature_value),
        )
    else:
        assert runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ) is not None
    runtime.finalize_step(decision["run_id"], decision["step_id"])


def test_runtime_classifies_causal_actual_and_forecast_steps():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=0,
            max_history=4,
            window_size=3.0,
            offline_smoothing_replay=False,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]),
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    first = runtime.begin_step(torch.tensor(1.0))
    first_descriptor = runtime.describe_current_er_sde_step(run_id, first["step_id"])
    assert first_descriptor.mode == "actual"
    assert not first_descriptor.requires_compensation
    _complete_runtime_step(runtime, first, 1.0)

    second = runtime.begin_step(torch.tensor(0.8))
    _complete_runtime_step(runtime, second, 0.8)
    third = runtime.begin_step(torch.tensor(0.6))
    third_descriptor = runtime.describe_current_er_sde_step(run_id, third["step_id"])
    assert third_descriptor.mode == "forecast"
    assert third_descriptor.requires_compensation
    runtime.abort_step(run_id, third["step_id"])
    runtime.end_run(run_id)


def _runtime_ready_for_step_two_forecast():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=0,
            max_history=4,
            window_size=3.0,
            offline_smoothing_replay=False,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.0]),
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    first = runtime.begin_step(torch.tensor(1.0))
    _complete_runtime_step(runtime, first, 1.0)
    second = runtime.begin_step(torch.tensor(0.8))
    _complete_runtime_step(runtime, second, 0.8)
    return runtime, run_id


def _tracker_targeting_step_two(run_id: int):
    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: torch.tensor([[2.0, -3.0]]),
        noise_scaler=lambda value: value**2,
        effective_s_noise=0.5,
        max_stage=3,
        debug=False,
        run_id=run_id,
    )
    _produce_pending(tracker)
    tracker.consume(torch.zeros((1, 2)), ERSDEStepDescriptor(run_id, 1, "actual", None, False))
    q = _produce_pending(tracker)
    return tracker, q


def test_predict_wrapper_corrects_before_returning_forecast_to_sampler():
    runtime, run_id = _runtime_ready_for_step_two_forecast()
    tracker, q = _tracker_targeting_step_two(run_id)
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)}
    )
    base = torch.tensor([[4.0, 5.0]])

    class Executor:
        class_obj = guider

        def __call__(self, _x, _timestep, model_options, _seed):
            options = model_options["transformer_options"]
            call_id, actual = runtime.begin_model_call(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                labels=((0, "positive"),),
                expected_shape=(1, 2, 1),
            )
            assert not actual
            assert runtime.predict(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ) is not None
            return base + q

    result = predict_noise_wrapper(
        Executor(),
        torch.zeros((1, 2)),
        torch.tensor(0.6),
        {"transformer_options": {ER_SDE_TRACKER_KEY: tracker}},
        seed=9,
    )

    torch.testing.assert_close(result, base, rtol=0, atol=0)
    assert runtime.last_completed_mode == "forecast"
    assert not tracker.has_pending
    runtime.end_run(run_id)


def test_missing_pending_increment_retries_same_step_as_actual():
    runtime, run_id = _runtime_ready_for_step_two_forecast()
    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda *_args: torch.zeros((1, 2)),
        noise_scaler=lambda value: value**2,
        effective_s_noise=0.5,
        max_stage=3,
        debug=False,
        run_id=run_id,
    )
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)}
    )
    attempts = []
    actual_result = torch.tensor([[7.0, 8.0]])

    class Executor:
        class_obj = guider

        def __call__(self, _x, _timestep, model_options, _seed):
            options = model_options["transformer_options"]
            call_id, actual = runtime.begin_model_call(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                labels=((0, "positive"),),
                expected_shape=(1, 2, 1),
            )
            attempts.append(bool(options[ACTUAL_KEY]))
            if actual:
                runtime.observe_actual(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    torch.ones((1, 2, 1)),
                )
            else:
                assert runtime.predict(
                    options[RUN_ID_KEY],
                    options[STEP_ID_KEY],
                    call_id,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                ) is not None
            return actual_result

    result = predict_noise_wrapper(
        Executor(),
        torch.zeros((1, 2)),
        torch.tensor(0.6),
        {"transformer_options": {ER_SDE_TRACKER_KEY: tracker}},
        seed=9,
    )

    assert result is actual_result
    assert attempts == [False, True]
    assert runtime.last_completed_mode == "actual"
    assert runtime.stats.forecast_fallbacks == 1
    runtime.end_run(run_id)


def test_runtime_classifies_replay_anchor_and_smoothed_source():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=1,
            max_history=6,
            window_size=3.0,
            offline_smoothing_replay=True,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    runtime.begin_offline_capture(total_steps=5, sampler_name="sample_er_sde")
    capture_run = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    source_actual = []
    for index, sigma in enumerate(sigmas[:-1]):
        decision = runtime.begin_step(sigma)
        source_actual.append(bool(decision["actual"]))
        _complete_runtime_step(runtime, decision, float(index))
    runtime.end_run(capture_run)
    assert source_actual == [True, True, False, True, True]
    assert runtime.complete_offline_capture()
    runtime.begin_offline_replay()

    replay_run = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    descriptors = []
    for index, sigma in enumerate(sigmas[:-1]):
        decision = runtime.begin_step(sigma)
        descriptors.append(
            runtime.describe_current_er_sde_step(replay_run, decision["step_id"])
        )
        _complete_runtime_step(runtime, decision, float(index))
    runtime.end_run(replay_run)
    runtime.release_offline_archive()

    assert descriptors[0].mode == "replay"
    assert descriptors[0].replay_source_actual is True
    assert not descriptors[0].requires_compensation
    assert descriptors[2].replay_source_actual is False
    assert descriptors[2].requires_compensation


def test_replay_tracking_failure_aborts_to_first_pass_instead_of_retrying_actual():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            warmup_steps=1,
            tail_actual_steps=1,
            max_history=6,
            window_size=3.0,
            offline_smoothing_replay=True,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
        )
    )
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    runtime.begin_offline_capture(total_steps=5, sampler_name="sample_er_sde")
    capture_run = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    for index, sigma in enumerate(sigmas[:-1]):
        decision = runtime.begin_step(sigma)
        _complete_runtime_step(runtime, decision, float(index))
    runtime.end_run(capture_run)
    assert runtime.complete_offline_capture()
    runtime.begin_offline_replay()
    replay_run = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    for sigma in sigmas[:2]:
        decision = runtime.begin_step(sigma)
        _complete_runtime_step(runtime, decision, 0.0)

    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda *_args: torch.zeros((1, 2)),
        noise_scaler=lambda value: value**2,
        effective_s_noise=0.5,
        max_stage=3,
        debug=False,
        run_id=replay_run,
    )
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)}
    )

    class Executor:
        class_obj = guider

        def __call__(self, _x, _timestep, model_options, _seed):
            options = model_options["transformer_options"]
            call_id, actual = runtime.begin_model_call(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                labels=((0, "positive"),),
                expected_shape=(1, 2, 1),
            )
            assert not actual
            assert runtime.predict(
                options[RUN_ID_KEY],
                options[STEP_ID_KEY],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ) is not None
            return torch.zeros((1, 2))

    with pytest.raises(OfflineReplayAbort, match="failed during replay"):
        predict_noise_wrapper(
            Executor(),
            torch.zeros((1, 2)),
            torch.tensor(0.6),
            {"transformer_options": {ER_SDE_TRACKER_KEY: tracker}},
            seed=9,
        )

    assert runtime.active_step_id is None
    runtime.end_run(replay_run)
    runtime.release_offline_archive()


def test_exception_cleanup_cannot_contaminate_next_tracker():
    first = _tracker()
    _produce_pending(first)
    first.clear()
    second = _tracker()

    assert not first.has_pending
    assert second.consume(torch.ones((1, 2)), _descriptor(0)) is not None


def _active_er_sde_runtime():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            offline_smoothing_replay=False,
            model_aware_mode="off",
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    return runtime, run_id


def test_unreviewed_er_sde_contract_fails_closed_to_native_all_actual(
    monkeypatch, caplog
):
    runtime, run_id = _active_er_sde_runtime()
    model_wrap = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)}
    )
    sampler = SimpleNamespace(
        sampler_function=SimpleNamespace(__name__="sample_er_sde"),
        extra_options={"noise_sampler": object()},
    )
    calls = []

    class Executor:
        class_obj = sampler

        def __init__(self):
            self.wrappers = [sampler_sample_wrapper]

        def __call__(self, *args):
            calls.append(args)
            return "native-all-actual"

    monkeypatch.setattr(
        sampling_module,
        "_er_sde_tracking_contract",
        lambda _sampler: (False, "custom ER-SDE noise_sampler provenance is unreviewed"),
    )
    with caplog.at_level("WARNING"):
        result = sampler_sample_wrapper(
            Executor(),
            model_wrap,
            torch.tensor([1.0, 0.5, 0.0]),
            {"model_options": {}},
            None,
            torch.ones((1, 2)),
            torch.zeros((1, 2)),
            None,
            True,
        )

    assert result == "native-all-actual"
    assert len(calls) == 1
    assert runtime.disabled_reason == "custom ER-SDE noise_sampler provenance is unreviewed"
    assert "preserving native all-actual sampling" in caplog.text
    runtime.end_run(run_id)


def test_native_preflight_failure_bypasses_spectrum_before_run_start(
    monkeypatch, caplog
):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(offline_smoothing_replay=False, model_aware_mode="off")
    )
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime), "transformer_options": {}}
    )
    sampler = SimpleNamespace(
        sampler_function=SimpleNamespace(__name__="sample_er_sde"),
        extra_options={},
    )

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            assert runtime.active_run_id is None
            return "untouched-native"

    monkeypatch.setattr(
        sampling_module,
        "_native_er_sde_preflight_reason",
        lambda _sampler, _options: "reviewed source digest changed",
    )
    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            torch.ones((1, 2)),
            torch.zeros((1, 2)),
            sampler,
            torch.tensor([1.0, 0.0]),
            seed=9,
        )

    assert result == "untouched-native"
    assert runtime.active_run_id is None
    assert "untouched native sampler" in caplog.text


def test_effective_s_noise_zero_uses_untouched_native_path_without_tracker(monkeypatch):
    runtime, run_id = _active_er_sde_runtime()

    class Patcher:
        def get_model_object(self, name):
            assert name == "model_sampling"
            return SimpleNamespace(noise_scale=1.0)

    model_wrap = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)},
        model_patcher=Patcher(),
    )
    sampler = SimpleNamespace(
        sampler_function=SimpleNamespace(__name__="sample_er_sde"),
        extra_options={"s_noise": 0.0, "max_stage": 3},
    )
    observed_model_options = []

    class Executor:
        class_obj = sampler

        def __init__(self):
            self.wrappers = [sampler_sample_wrapper]

        def __call__(self, *args):
            observed_model_options.append(args[2]["model_options"])
            return args[4]

    monkeypatch.setattr(
        sampling_module,
        "_er_sde_tracking_contract",
        lambda _sampler: (True, None),
    )
    noise = torch.ones((1, 2))
    result = sampler_sample_wrapper(
        Executor(),
        model_wrap,
        torch.tensor([1.0, 0.5, 0.0]),
        {"model_options": {"transformer_options": {}}},
        None,
        noise,
        torch.zeros((1, 2)),
        None,
        True,
    )

    assert result is noise
    assert ER_SDE_TRACKER_KEY not in observed_model_options[0]["transformer_options"]
    assert runtime.disabled_reason is None
    runtime.end_run(run_id)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_increment_stays_on_device_without_dtype_promotion():
    noise = torch.tensor([[2.0, -3.0]], device="cuda", dtype=torch.float16)
    tracker = _tracker(noise, dtype=torch.float16)
    _produce_pending(tracker)
    assert tracker._pending is not None
    assert tracker._pending.value.device.type == "cuda"
    assert tracker._pending.value.dtype == torch.float16
