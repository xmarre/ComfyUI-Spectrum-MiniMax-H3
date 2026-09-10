"""Focused prototype tests for deferred first-successful-forecast entitlement.

This intentionally exercises SpectrumH3Runtime's real decision/finalization chain while
keeping the proposed scheduler policy test-local until BSA transition diagnostics justify
production history carry.
"""
from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SolverCallDescriptor, SpectrumH3Runtime

TOPOLOGY = (
    ("video", (1, 24, 2, 4, 4)),
    ("audio", (1, 32, 2, 8)),
    ("hidden", 4),
    ("target_audio_rows", 1),
    ("target_video_rows", 2),
)
LABEL = ((0, "positive"),)


class EntitlementRuntime(SpectrumH3Runtime):
    """Test-only prototype of the proposed run-local entitlement semantics."""

    def __init__(self, config):
        super().__init__(config)
        self._bootstrap_entitlement_unused = False
        self._entitlement_snapshots = {}

    def start_run(self, *args, **kwargs):
        run_id = super().start_run(*args, **kwargs)
        self._bootstrap_entitlement_unused = bool(self.config.bootstrap_first_forecast)
        self._entitlement_snapshots.clear()
        return run_id

    def begin_step(self, timestep):
        decision = super().begin_step(timestep)
        step = self._step
        run = self._run
        previous_anchor = (
            self.forecaster.latest_anchor_ids(1) == (self._last_completed_step_id,)
            if self._last_completed_step_id is not None
            else False
        )
        if (
            decision["actual"]
            and decision["reason"] == "insufficient actual history"
            and self._bootstrap_entitlement_unused
            and self.config.bootstrap_first_forecast
            and self.config.degree == 1
            and run is not None
            and not run.state_conditioned_residual
            and not run.separate_stage_histories
            and self.offline_phase is None
            and self.forecaster.history_length == 1
            and self._last_completed_mode == "actual"
            and self._last_completed_step_id == step.step_id - 1
            and previous_anchor
        ):
            step.mode = "forecast"
            step.reason = "deferred one-point bootstrap forecast"
            step.bootstrap_forecast = True
            decision["actual"] = False
            decision["reason"] = step.reason
        return decision

    def finalize_step(self, run_id, step_id):
        committed_forecast = self._step is not None and self._step.mode == "forecast"
        super().finalize_step(run_id, step_id)
        if committed_forecast:
            self._bootstrap_entitlement_unused = False

    def create_rollback_snapshot(self):
        snapshot = super().create_rollback_snapshot()
        self._entitlement_snapshots[id(snapshot)] = self._bootstrap_entitlement_unused
        return snapshot

    def restore_rollback_snapshot(self, snapshot):
        entitlement = self._entitlement_snapshots[id(snapshot)]
        super().restore_rollback_snapshot(snapshot)
        self._bootstrap_entitlement_unused = entitlement


def _runtime(**overrides):
    values = {
        "degree": 1,
        "max_history": 4,
        "warmup_steps": 0,
        "tail_actual_steps": 1,
        "window_size": 2.0,
        "bootstrap_first_forecast": True,
        "offline_smoothing_replay": False,
    }
    values.update(overrides)
    return EntitlementRuntime(SpectrumH3Config(**values))


def _start(runtime, steps, **kwargs):
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    return runtime.start_run(
        sigmas,
        "sample_res_multistep",
        supported_sampler=True,
        min_actual_steps_after_forecast=0,
        **kwargs,
    )


def _complete(runtime, timestep, *, policy=None, safe=True):
    decision = runtime.begin_step(torch.tensor([timestep]))
    if policy is not None:
        runtime.prepare_backend_history(decision["run_id"], decision["step_id"], policy, safe)
    step_mode = runtime._step.mode
    call_id, actual = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY,
        labels=LABEL, expected_shape=(1, 3, 4),
    )
    if actual:
        runtime.observe_actual(
            decision["run_id"], decision["step_id"], call_id,
            torch.full((1, 3, 4), float(decision["step_id"])),
        )
        if policy is not None:
            runtime.observe_backend_history(
                decision["run_id"], decision["step_id"], policy,
                ("receipt", policy), safe,
            )
    else:
        prediction = runtime.predict(
            decision["run_id"], decision["step_id"], call_id,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert prediction is not None
    final_mode = runtime._step.mode
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision, step_mode, final_mode


def test_native_five_step_low_consumes_entitlement_on_first_committed_forecast():
    runtime = _runtime()
    run_id = _start(runtime, 5)
    modes = []
    for index, sigma in enumerate((1.0, 0.8, 0.6, 0.4, 0.2)):
        # Dense -> sparse-cold remains a hard boundary at step 2. The future
        # proven cold->primed carry is represented by keeping the sparse identity.
        policy = "dense" if index < 2 else "sparse"
        _, _, mode = _complete(runtime, sigma, policy=policy)
        modes.append(mode)
    assert modes == ["actual", "forecast", "actual", "actual", "actual"]
    assert runtime.stats.actual_steps == 4 and runtime.stats.forecast_steps == 1
    assert runtime._bootstrap_entitlement_unused is False
    runtime.end_run(run_id)


def test_mixed_five_step_low_defers_entitlement_through_prefix_and_backend_veto():
    runtime = _runtime()
    run_id = _start(runtime, 5, min_actual_prefix_steps=2)
    modes = []
    for index, sigma in enumerate((1.0, 0.8, 0.6, 0.4, 0.2)):
        policy = "dense" if index < 2 else "sparse"
        _, _, mode = _complete(runtime, sigma, policy=policy)
        modes.append(mode)
        if index < 3:
            assert runtime._bootstrap_entitlement_unused is True
    assert modes == ["actual", "actual", "actual", "forecast", "actual"]
    assert runtime.stats.actual_steps == 4 and runtime.stats.forecast_steps == 1
    assert runtime._bootstrap_entitlement_unused is False
    runtime.end_run(run_id)


def test_three_step_high_uses_one_bootstrap_and_tail_remains_exact():
    runtime = _runtime()
    run_id = _start(runtime, 3, min_actual_prefix_steps=1)
    modes = [_complete(runtime, sigma, policy="sparse")[2]
             for sigma in (1.0, 0.5, 0.25)]
    assert modes == ["actual", "forecast", "actual"]
    assert runtime.stats.actual_steps == 2 and runtime.stats.forecast_steps == 1
    runtime.end_run(run_id)


def test_backend_vetoed_forecast_attempt_does_not_consume_entitlement():
    runtime = _runtime(tail_actual_steps=1)
    run_id = _start(runtime, 4)
    _complete(runtime, 1.0, policy="a")
    decision = runtime.begin_step(torch.tensor([0.75]))
    assert not decision["actual"] and runtime._bootstrap_entitlement_unused
    runtime.prepare_backend_history(decision["run_id"], decision["step_id"], "b", True)
    assert runtime._step.mode == "actual"
    call_id, actual = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY,
        labels=LABEL, expected_shape=(1, 3, 4),
    )
    assert actual
    runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, torch.ones(1, 3, 4))
    runtime.observe_backend_history(decision["run_id"], decision["step_id"], "b", ("receipt", "b"), True)
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime._bootstrap_entitlement_unused is True
    deferred, _, mode = _complete(runtime, 0.5, policy="b")
    assert deferred["reason"] == "deferred one-point bootstrap forecast"
    assert mode == "forecast" and runtime._bootstrap_entitlement_unused is False
    runtime.end_run(run_id)


def test_fallback_to_actual_after_prediction_does_not_consume_entitlement():
    runtime = _runtime(tail_actual_steps=0)
    run_id = _start(runtime, 3)
    _complete(runtime, 1.0)
    decision = runtime.begin_step(torch.tensor([0.5]))
    assert not decision["actual"] and runtime._bootstrap_entitlement_unused
    call_id, actual = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY,
        labels=LABEL, expected_shape=(1, 3, 4),
    )
    assert not actual
    assert runtime.predict(
        decision["run_id"], decision["step_id"], call_id,
        device=torch.device("cpu"), dtype=torch.float32,
    ) is not None
    runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], "test retry")
    retry_id, retry_actual = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY,
        labels=LABEL, expected_shape=(1, 3, 4),
    )
    assert retry_actual
    runtime.observe_actual(
        decision["run_id"], decision["step_id"], retry_id, torch.ones(1, 3, 4)
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime._bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def test_abort_does_not_consume_entitlement_and_backend_reset_never_replenishes_it():
    runtime = _runtime(tail_actual_steps=0)
    run_id = _start(runtime, 5)
    _complete(runtime, 1.0, policy="a")
    decision = runtime.begin_step(torch.tensor([0.8]))
    assert not decision["actual"] and runtime._bootstrap_entitlement_unused
    runtime.abort_step(decision["run_id"], decision["step_id"])
    assert runtime._bootstrap_entitlement_unused is True
    _complete(runtime, 0.8, policy="a")
    assert runtime._bootstrap_entitlement_unused is False
    _complete(runtime, 0.6, policy="b")
    assert runtime._bootstrap_entitlement_unused is False
    _, _, mode = _complete(runtime, 0.4, policy="b")
    assert mode == "actual"
    runtime.end_run(run_id)


def test_rollback_restores_entitlement_transactionally():
    runtime = _runtime(tail_actual_steps=0)
    run_id = _start(runtime, 4)
    _complete(runtime, 1.0, policy="a")
    snapshot = runtime.create_rollback_snapshot()
    assert runtime._bootstrap_entitlement_unused is True
    _complete(runtime, 0.75, policy="a")
    assert runtime._bootstrap_entitlement_unused is False
    runtime.restore_rollback_snapshot(snapshot)
    assert runtime._bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def test_sampler_required_exact_stage_dominates_and_does_not_consume_entitlement():
    runtime = _runtime(tail_actual_steps=0)
    run_id = _start(
        runtime,
        3,
        forecastable_stage_indices=(),
        history_stage_indices=(0,),
    )
    _first, _, first_mode = _complete(runtime, 1.0)
    second, _, second_mode = _complete(runtime, 0.5)
    assert first_mode == second_mode == "actual"
    assert second["reason"] == "sampler-required exact stage"
    assert runtime._bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def test_deferred_bootstrap_requires_the_previous_actual_to_be_the_retained_anchor():
    runtime = _runtime(tail_actual_steps=0)
    run_id = _start(
        runtime,
        4,
        forced_actual_step_ids=(1,),
        history_step_ids=(0,),
    )
    _complete(runtime, 1.0)
    exact, _, exact_mode = _complete(runtime, 0.75)
    assert exact_mode == "actual" and exact["reason"] == "sampler-required exact step"
    assert runtime.forecaster.latest_anchor_ids(1) == (0,)
    candidate, _, candidate_mode = _complete(runtime, 0.5)
    assert candidate_mode == "actual"
    assert candidate["reason"] == "insufficient actual history"
    assert runtime._bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def test_sa_pece_state_conditioned_topology_does_not_acquire_ordinary_bootstrap():
    runtime = _runtime(tail_actual_steps=0)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    topology = (
        SolverCallDescriptor(0, 0, "predicted"),
        SolverCallDescriptor(1, 0, "predicted"),
        SolverCallDescriptor(1, 1, "corrected"),
    )
    run_id = runtime.start_run(
        sigmas,
        "sample_sa_solver_pece",
        supported_sampler=True,
        expected_model_calls=len(topology),
        stage_count=2,
        logical_call_topology=topology,
        state_conditioned_residual=True,
        separate_stage_histories=False,
        forecastable_stage_indices=(0,),
        history_stage_indices=(0, 1),
        history_step_ids=(0, 2),
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=False,
        min_actual_prefix_steps=1,
        min_actual_steps_after_forecast=0,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )
    _first, _, first_mode = _complete(runtime, 1.0)
    predicted, _, predicted_mode = _complete(runtime, 0.5)
    corrected, _, corrected_mode = _complete(runtime, 0.5)
    assert first_mode == predicted_mode == corrected_mode == "actual"
    assert predicted["reason"] == "insufficient actual history"
    assert corrected["reason"] == "sampler-required exact stage"
    assert runtime._bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def test_prefix_tail_disabled_degree_and_state_conditioned_restrictions():
    prefix = _runtime()
    prefix_id = _start(prefix, 4, min_actual_prefix_steps=2)
    first = _complete(prefix, 1.0)[2]
    second = _complete(prefix, 0.75)[2]
    assert (first, second) == ("actual", "actual")
    assert prefix._bootstrap_entitlement_unused is True
    prefix.end_run(prefix_id)

    disabled = _runtime(bootstrap_first_forecast=False, tail_actual_steps=0)
    disabled_id = _start(disabled, 3)
    assert [_complete(disabled, s)[2] for s in (1.0, 0.5)] == ["actual", "actual"]
    disabled.end_run(disabled_id)

    with pytest.raises(ValueError, match="bootstrap_first_forecast requires degree == 1"):
        _runtime(degree=2, tail_actual_steps=0)

    state = _runtime(tail_actual_steps=0)
    state_id = _start(state, 3, state_conditioned_residual=True, min_actual_prefix_steps=1)
    assert [_complete(state, s)[2] for s in (1.0, 0.5)] == ["actual", "actual"]
    assert state._bootstrap_entitlement_unused is True
    state.end_run(state_id)
