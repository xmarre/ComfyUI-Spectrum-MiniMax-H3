from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import sampling as sampling_module
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    RES4LYF_RES_SAMPLERS,
    SpectrumH3Binding,
    _res4lyf_call_topology,
    _res4lyf_expected_model_calls,
    _res4lyf_stage_schedule_reason,
    _res4lyf_tracking_sigmas,
    max_consecutive_forecasts,
    min_actual_steps_after_forecast,
    outer_sample_wrapper,
    sampler_is_supported,
    sampler_supports_seeded_replay,
)


def _sampler(function_name: str) -> SimpleNamespace:
    def sampler_function():
        pass

    sampler_function.__name__ = function_name
    return SimpleNamespace(sampler_function=sampler_function, extra_options={})


@pytest.mark.parametrize("function_name", sorted(RES4LYF_RES_SAMPLERS))
def test_named_res4lyf_variants_are_allowlisted_but_not_replayable(function_name):
    sampler = _sampler(function_name)

    assert sampler_is_supported(sampler)
    assert max_consecutive_forecasts(sampler) == 1
    assert not sampler_supports_seeded_replay(sampler)


def test_tracking_sigmas_match_res4lyf_terminal_zero_behavior():
    with_terminal_zero = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0])
    positive_terminal = torch.tensor([1.0, 0.8, 0.6, 0.4])

    assert torch.equal(
        _res4lyf_tracking_sigmas(with_terminal_zero),
        torch.tensor([1.0, 0.8, 0.6, 0.4]),
    )
    assert torch.equal(_res4lyf_tracking_sigmas(positive_terminal), positive_terminal)


@pytest.mark.parametrize(
    ("function_name", "stage_count", "expected"),
    (
        ("sample_res_2s", 2, 4),
        ("sample_res_2s_ode", 2, 4),
        ("sample_res_3s", 3, 6),
        ("sample_res_3s_ode", 3, 6),
        ("sample_res_5s", 5, 10),
        ("sample_res_5s_ode", 5, 10),
        ("sample_res_6s", 6, 12),
        ("sample_res_6s_ode", 6, 12),
    ),
)
def test_fixed_stage_topology_counts_native_model_calls(function_name, stage_count, expected):
    sampler = _sampler(function_name)
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.0])

    topology = _res4lyf_call_topology(sampler, sigmas)

    assert topology is not None
    assert len(topology) == expected
    assert _res4lyf_expected_model_calls(sampler, sigmas) == expected
    assert [descriptor.outer_step for descriptor in topology] == [
        outer_step
        for outer_step in range(2)
        for _ in range(stage_count)
    ]
    assert [descriptor.stage_index for descriptor in topology] == list(range(stage_count)) * 2


@pytest.mark.parametrize(
    ("function_name", "expected_stages", "expected_calls", "refreshes"),
    (
        ("sample_res_2m", ((0, 1), (0, 1), (0,), (0,)), 6, 2),
        ("sample_res_2m_ode", ((0, 1), (0, 1), (0,), (0,)), 6, 2),
        ("sample_res_3m", ((0, 1, 2), (0, 1, 2), (0, 1, 2), (0,)), 10, 3),
        ("sample_res_3m_ode", ((0, 1, 2), (0, 1, 2), (0, 1, 2), (0,)), 10, 3),
    ),
)
def test_multistep_topology_preserves_native_startup_then_single_call(
    function_name,
    expected_stages,
    expected_calls,
    refreshes,
):
    sampler = _sampler(function_name)
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.45, 0.3, 0.0])

    topology = _res4lyf_call_topology(sampler, sigmas)

    assert topology is not None
    grouped = tuple(
        tuple(
            descriptor.stage_index
            for descriptor in topology
            if descriptor.outer_step == outer_step
        )
        for outer_step in range(4)
    )
    assert grouped == expected_stages
    assert len(topology) == expected_calls
    assert min_actual_steps_after_forecast(sampler) == refreshes


@pytest.mark.parametrize("function_name", ("sample_res_2m", "sample_res_3m_ode"))
def test_multistep_dynamic_h_fallback_is_rejected(function_name):
    sampler = _sampler(function_name)
    sigmas = torch.tensor([1.0, 0.3, 0.2, 0.0])

    reason = _res4lyf_stage_schedule_reason(sampler, sigmas)

    assert reason is not None
    assert "dynamic h>=1 fallback" in reason
    assert _res4lyf_call_topology(sampler, sigmas) is None


@pytest.mark.parametrize(
    "sigmas",
    (
        torch.tensor([1.0, 0.8, 0.8, 0.0]),
        torch.tensor([1.0, 0.8, 0.0, 0.0]),
        torch.tensor([1.0, float("nan"), 0.0]),
    ),
)
def test_invalid_res4lyf_active_schedule_fails_closed(sigmas):
    reason = _res4lyf_stage_schedule_reason(_sampler("sample_res_2s"), sigmas)

    assert reason is not None


@pytest.mark.parametrize(
    "function_name",
    ("sample_res_2s", "sample_res_3s_ode", "sample_res_5s", "sample_res_6s_ode"),
)
def test_fixed_stage_res4lyf_requires_one_exact_refresh(function_name):
    assert min_actual_steps_after_forecast(_sampler(function_name)) == 1


def test_outer_wrapper_uses_res4lyf_active_intervals_and_explicit_topology(monkeypatch):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            warmup_steps=0,
            tail_actual_steps=0,
            model_aware_mode="off",
            offline_smoothing_replay=False,
        )
    )
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {},
        }
    )
    captured = {}

    class Executor:
        class_obj = guider

        def __call__(self, *_args, **_kwargs):
            run = runtime._run
            assert run is not None
            captured["total_steps"] = run.total_steps
            captured["policy_steps"] = run.policy_steps
            captured["stage_count"] = run.stage_count
            captured["prefix"] = run.min_sampler_actual_prefix_steps
            captured["forecastable"] = run.forecastable_stage_indices
            captured["state_conditioned"] = run.state_conditioned_residual
            captured["topology"] = run.logical_call_topology
            return "native-result"

    monkeypatch.setattr(sampling_module, "_res4lyf_preflight_reason", lambda *_args: None)
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.45, 0.3, 0.0])

    result = outer_sample_wrapper(
        Executor(),
        torch.zeros(1),
        torch.zeros(1),
        _sampler("sample_res_2m"),
        sigmas,
    )

    assert result == "native-result"
    assert captured["total_steps"] == 6
    assert captured["policy_steps"] == 4
    assert captured["stage_count"] == 2
    assert captured["prefix"] == 2
    assert captured["forecastable"] == (0,)
    assert captured["state_conditioned"] is True
    assert captured["topology"] is not None
    assert runtime.active_run_id is None
