from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    SpectrumH3Binding,
    max_consecutive_forecasts,
    min_actual_steps_after_forecast,
    min_tail_actual_steps,
    outer_sample_wrapper,
    sampler_is_supported,
    sampler_name,
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
