from __future__ import annotations

from types import SimpleNamespace

import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import min_tail_actual_steps


def _sampler(function_name: str) -> SimpleNamespace:
    def sampler_function():
        pass

    sampler_function.__name__ = function_name
    return SimpleNamespace(sampler_function=sampler_function)


def test_er_sde_requires_two_actual_tail_steps_without_changing_other_sampler_policies():
    assert min_tail_actual_steps(_sampler("sample_er_sde")) == 2
    assert min_tail_actual_steps(_sampler("sample_res_multistep")) == 3
    assert min_tail_actual_steps(_sampler("sample_res_multistep_cfg_pp")) == 3
    assert min_tail_actual_steps(_sampler("sample_euler")) == 0
    assert min_tail_actual_steps(_sampler("_turbo_sampler")) == 0


def _penultimate_decision(total_steps: int, *, minimum_tail: int) -> dict[str, object]:
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="off",
            warmup_steps=1,
            tail_actual_steps=1,
            bootstrap_first_forecast=True,
        )
    )
    sigmas = torch.linspace(1.0, 0.0, total_steps + 1, dtype=torch.float32)
    run_id = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        min_tail_actual_steps=minimum_tail,
    )
    feature = torch.zeros(1, 2, 4, dtype=torch.float32)
    runtime.forecaster.update(0.9, feature, anchor_id=0, take_ownership=True)
    runtime.forecaster.update(0.8, feature + 1.0, anchor_id=2, take_ownership=True)
    assert runtime._run is not None
    runtime._run.next_step_id = total_steps - 2
    runtime._consecutive_forecasts = 0
    decision = runtime.begin_step(sigmas[total_steps - 2])
    runtime.abort_step(run_id, total_steps - 2)
    runtime.end_run(run_id)
    return decision


def test_er_sde_two_step_tail_removes_odd_step_penultimate_forecast():
    old_policy = _penultimate_decision(25, minimum_tail=0)
    protected = _penultimate_decision(25, minimum_tail=2)

    assert old_policy["actual"] is False
    assert old_policy["reason"] == "adaptive forecast"
    assert protected["actual"] is True
    assert protected["reason"] == "final actual tail"


def test_er_sde_tail_invariant_is_independent_of_total_step_count():
    for total_steps in (20, 25, 32):
        decision = _penultimate_decision(total_steps, minimum_tail=2)
        assert decision["actual"] is True
        assert decision["reason"] == "final actual tail"
