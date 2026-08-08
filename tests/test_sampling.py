from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import OfflineReplayAbort, SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    SpectrumH3Binding,
    max_consecutive_forecasts,
    min_actual_steps_after_forecast,
    min_tail_actual_steps,
    outer_sample_wrapper,
    predict_noise_wrapper,
    sampler_is_supported,
    sampler_name,
    sampler_sample_wrapper,
)


def _sampler(function_name: str) -> SimpleNamespace:
    def sampler_function():
        pass

    sampler_function.__name__ = function_name
    return SimpleNamespace(sampler_function=sampler_function)


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    ),
)
def test_reviewed_single_call_samplers_are_supported(function_name):
    sampler = _sampler(function_name)

    assert sampler_name(sampler) == function_name
    assert sampler_is_supported(sampler)


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler_ancestral",
        "sample_res_multistep_ancestral",
        "sample_res_multistep_ancestral_cfg_pp",
        "res_multistep",
        "sample_res_multistep_experimental",
        "sample_heun",
    ),
)
def test_unreviewed_sampler_names_do_not_match_by_prefix(function_name):
    assert not sampler_is_supported(_sampler(function_name))


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    ),
)
def test_supported_h3_samplers_limit_forecast_streaks(function_name):
    assert max_consecutive_forecasts(_sampler(function_name)) == 1


@pytest.mark.parametrize("function_name", ("sample_res_multistep", "sample_res_multistep_cfg_pp"))
def test_res_multistep_policy_refreshes_once_and_protects_tail(function_name):
    sampler = _sampler(function_name)

    assert min_actual_steps_after_forecast(sampler) == 1
    assert min_tail_actual_steps(sampler) == 3


def test_euler_policy_keeps_one_refresh_and_user_tail():
    sampler = _sampler("sample_euler")

    assert min_actual_steps_after_forecast(sampler) == 1
    assert min_tail_actual_steps(sampler) == 0


def test_unsupported_sampler_has_no_forecast_streak_policy():
    sampler = _sampler("sample_euler_ancestral")

    assert max_consecutive_forecasts(sampler) is None
    assert min_actual_steps_after_forecast(sampler) == 0
    assert min_tail_actual_steps(sampler) == 0


def test_easycache_on_same_model_bypasses_spectrum_run(caplog):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {"easycache": object()},
        }
    )
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "native-cache-result"

    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            "noise",
            "latent",
            _sampler("sample_euler"),
            "sigmas",
            denoise_mask="mask",
            callback="callback",
            disable_pbar=True,
            seed=7,
            latent_shapes=("video", "audio"),
        )

    assert result == "native-cache-result"
    assert len(calls) == 1
    assert runtime.active_run_id is None
    assert "EasyCache or LazyCache is active" in caplog.text


def test_predict_noise_passthrough_survives_a_downstream_model_bypass(caplog):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            warmup_steps=1,
            tail_actual_steps=0,
            bootstrap_first_forecast=True,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    guider = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(runtime)})
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return f"intercepted-result-{len(calls)}"

    try:
        with caplog.at_level("WARNING"):
            first = predict_noise_wrapper(
                Executor(), "latent", torch.tensor([1.0]), {"sentinel": True}, seed=7
            )
            second = predict_noise_wrapper(
                Executor(), "latent", torch.tensor([0.5]), {"sentinel": True}, seed=7
            )

        assert first == "intercepted-result-1"
        assert second == "intercepted-result-2"
        assert len(calls) == 2
        assert runtime.stats.bypassed_steps == 2
        assert runtime.active_step_id is None
        assert caplog.text.count("accepting the wrapped result as a passthrough") == 1
    finally:
        runtime.end_run(run_id)


def test_default_offline_outer_sample_reports_both_passes_and_callbacks_only_replay(monkeypatch):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            bootstrap_first_forecast=False,
            audio_blend_weight=0.5,
        )
    )
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime), "transformer_options": {}},
        conds={"positive": [{"nested": {"marker": "original"}}]},
    )
    starts = []
    callback_arguments = []
    progress_totals = []
    progress_updates = []
    topology = (("tiny", 1),)
    labels = ((0, "positive"),)

    class ProgressBar:
        def __init__(self, total):
            progress_totals.append(total)

        def update_absolute(self, value, total=None, preview=None):
            progress_updates.append((value, total, preview))

    fake_utils = ModuleType("comfy.utils")
    fake_utils.ProgressBar = ProgressBar
    fake_comfy = ModuleType("comfy")
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    callback_progress = ProgressBar(2)

    def callback(*args):
        callback_arguments.append(args)
        callback_progress.update_absolute(args[0] + 1, args[3])

    class Executor:
        class_obj = guider

        def __call__(
            self,
            run_noise,
            run_latent,
            _sampler,
            run_sigmas,
            _mask,
            run_callback,
            _disable_pbar,
            _seed,
            *,
            latent_shapes=None,
        ):
            assert guider.conds["positive"][0]["nested"]["marker"] == "original"
            starts.append(
                (
                    runtime.offline_phase,
                    run_noise.clone(),
                    run_latent.clone(),
                    run_callback,
                    _disable_pbar,
                )
            )
            for index, sigma in enumerate(run_sigmas[:-1]):
                decision = runtime.begin_step(sigma)
                call_id, actual = runtime.begin_model_call(
                    decision["run_id"],
                    decision["step_id"],
                    topology=topology,
                    labels=labels,
                    expected_shape=(1, 1, 1),
                )
                if actual:
                    runtime.observe_actual(
                        decision["run_id"],
                        decision["step_id"],
                        call_id,
                        torch.full((1, 1, 1), float(index)),
                    )
                else:
                    runtime.predict(
                        decision["run_id"],
                        decision["step_id"],
                        call_id,
                        device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                runtime.finalize_step(decision["run_id"], decision["step_id"])
                if run_callback is not None:
                    run_callback(index, run_noise, run_latent, len(run_sigmas) - 1)
            if runtime.offline_phase == "first_pass":
                guider.conds["positive"][0]["nested"]["marker"] = "processed-first-pass"
                run_noise.add_(10.0)
            return run_noise + run_latent

    noise = torch.ones(1)
    latent = torch.full((1,), 2.0)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    result = outer_sample_wrapper(
        Executor(),
        noise,
        latent,
        _sampler("sample_euler"),
        sigmas,
        callback=callback,
        seed=7,
        latent_shapes=((1,),),
    )

    assert [phase for phase, *_rest in starts] == ["first_pass", "replay"]
    assert starts[0][3] is not None
    assert starts[1][3] is not None
    assert [entry[4] for entry in starts] == [False, True]
    torch.testing.assert_close(starts[0][1], torch.ones(1))
    torch.testing.assert_close(starts[1][1], torch.ones(1))
    torch.testing.assert_close(starts[1][2], torch.full((1,), 2.0))
    assert progress_totals == [2, 4]
    assert progress_updates == [
        (1, 4, None),
        (2, 4, None),
        (3, 4, None),
        (4, 4, None),
    ]
    assert len(callback_arguments) == 2
    assert [(args[0], args[3]) for args in callback_arguments] == [(2, 4), (3, 4)]
    torch.testing.assert_close(result, torch.full((1,), 3.0))
    assert runtime.active_run_id is None
    assert runtime.offline_archive is None


def test_offline_progress_finishes_when_capture_cannot_be_replayed(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime), "transformer_options": {}},
        conds={},
    )
    progress_updates = []
    callback_arguments = []

    class ProgressBar:
        def __init__(self, total):
            assert total == 4

        def update_absolute(self, value, total=None, preview=None):
            progress_updates.append((value, total, preview))

    fake_utils = ModuleType("comfy.utils")
    fake_utils.ProgressBar = ProgressBar
    fake_comfy = ModuleType("comfy")
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)

    class Executor:
        class_obj = guider

        def __call__(
            self,
            run_noise,
            run_latent,
            _sampler,
            _sigmas,
            _mask,
            run_callback,
            _disable_pbar,
            _seed,
            *,
            latent_shapes=None,
        ):
            run_callback(0, run_noise, run_latent, 2)
            return "valid-first-pass"

    result = outer_sample_wrapper(
        Executor(),
        torch.ones(1),
        torch.zeros(1),
        _sampler("sample_euler"),
        torch.tensor([1.0, 0.5, 0.0]),
        callback=lambda *args: callback_arguments.append(args),
        seed=7,
    )

    assert result == "valid-first-pass"
    assert progress_updates == [(1, 4, None), (4, 4, None)]
    assert callback_arguments == []
    assert runtime.active_run_id is None
    assert runtime.offline_archive is None


def test_offline_progress_finishes_when_replay_aborts(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime), "transformer_options": {}},
        conds={},
    )
    progress_updates = []
    callback_arguments = []
    topology = (("tiny", 1),)
    labels = ((0, "positive"),)

    class ProgressBar:
        def __init__(self, total):
            assert total == 4

        def update_absolute(self, value, total=None, preview=None):
            progress_updates.append((value, total, preview))

    fake_utils = ModuleType("comfy.utils")
    fake_utils.ProgressBar = ProgressBar
    fake_comfy = ModuleType("comfy")
    fake_comfy.utils = fake_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)

    class Executor:
        class_obj = guider

        def __call__(
            self,
            run_noise,
            run_latent,
            _sampler,
            run_sigmas,
            _mask,
            run_callback,
            _disable_pbar,
            _seed,
            *,
            latent_shapes=None,
        ):
            if runtime.offline_phase == "replay":
                raise OfflineReplayAbort("test replay failure")

            for index, sigma in enumerate(run_sigmas[:-1]):
                decision = runtime.begin_step(sigma)
                call_id, actual = runtime.begin_model_call(
                    decision["run_id"],
                    decision["step_id"],
                    topology=topology,
                    labels=labels,
                    expected_shape=(1, 1, 1),
                )
                assert actual
                runtime.observe_actual(
                    decision["run_id"],
                    decision["step_id"],
                    call_id,
                    torch.full((1, 1, 1), float(index)),
                )
                runtime.finalize_step(decision["run_id"], decision["step_id"])
                run_callback(index, run_noise, run_latent, len(run_sigmas) - 1)
            return "valid-first-pass"

    result = outer_sample_wrapper(
        Executor(),
        torch.ones(1),
        torch.zeros(1),
        _sampler("sample_euler"),
        torch.tensor([1.0, 0.5, 0.0]),
        callback=lambda *args: callback_arguments.append(args),
        seed=7,
    )

    assert result == "valid-first-pass"
    assert progress_updates == [(1, 4, None), (2, 4, None), (4, 4, None)]
    assert callback_arguments == []
    assert runtime.active_run_id is None
    assert runtime.offline_archive is None


def test_offline_unsupported_sampler_uses_true_native_bypass(caplog):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            bootstrap_first_forecast=False,
            offline_smoothing_replay=True,
        )
    )
    guider = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime), "transformer_options": {}}
    )
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            assert runtime.active_run_id is None
            calls.append((args, kwargs))
            return "native-result"

    callback = object()
    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            torch.ones(1),
            torch.zeros(1),
            _sampler("sample_euler_ancestral"),
            torch.tensor([1.0, 0.0]),
            callback=callback,
            seed=7,
        )

    assert result == "native-result"
    assert len(calls) == 1
    assert calls[0][0][5] is callback
    assert runtime.active_run_id is None
    assert "running one native pass" in caplog.text


def test_selective_rollback_res_falls_back_before_sampler_mutation(monkeypatch, caplog):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            bootstrap_first_forecast=False,
            selective_rollback_correction=True,
            offline_smoothing_replay=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_res_multistep",
        supported_sampler=True,
    )
    model_wrap = SimpleNamespace(
        model_options={BINDING_KEY: SpectrumH3Binding(runtime)}
    )
    fake_sampling = ModuleType("comfy.k_diffusion.sampling")
    fake_sampling.sample_euler = object()
    fake_k_diffusion = ModuleType("comfy.k_diffusion")
    fake_k_diffusion.sampling = fake_sampling
    fake_comfy = ModuleType("comfy")
    fake_comfy.k_diffusion = fake_k_diffusion
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", fake_k_diffusion)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", fake_sampling)
    calls = []

    class Executor:
        class_obj = _sampler("sample_res_multistep")

        def __call__(self, *args):
            calls.append(args)
            return "ordinary-spectrum"

    with caplog.at_level("WARNING"):
        result = sampler_sample_wrapper(
            Executor(),
            model_wrap,
            torch.tensor([1.0, 0.5, 0.0]),
            {"model_options": {}},
            None,
            torch.ones(1),
            torch.zeros(1),
            None,
            True,
        )

    assert result == "ordinary-spectrum"
    assert len(calls) == 1
    assert runtime._run.next_step_id == 0
    assert "supports only the exact reviewed sample_euler contract" in runtime.experiment_disabled_reason
    assert caplog.text.count("experimental mode disabled") == 1
    runtime.end_run(run_id)
