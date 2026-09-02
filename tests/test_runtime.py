from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.model_aware import ModelForecastabilityProfile, ProfileLookup
from comfyui_spectrum_h3.runtime import (
    ExactCorrectorGuarantee,
    ForecastRetryActual,
    SolverCallDescriptor,
    SpectrumH3Runtime,
)

TOPOLOGY = (
    ("video", (1, 24, 2, 4, 4)),
    ("audio", (1, 32, 2, 8)),
    ("hidden", 4),
    ("target_audio_rows", 1),
    ("target_video_rows", 2),
)
LABEL = ((0, "positive"),)


def _test_model_profile() -> ModelForecastabilityProfile:
    return ModelForecastabilityProfile(
        cache_key=("test",),
        base_model_identity="test",
        patch_identity="test",
        active_patch_count=0,
        active_patch_keys=0,
        recognized_lora_count=0,
        unknown_patch_count=0,
        sampled_base_tensors=1,
        profile_confidence=1.0,
        aggregate_sensitivity=0.2,
        patch_perturbation=0.0,
        final_block_perturbation=0.0,
        audio_sensitivity=1.0,
        video_sensitivity=1.0,
        audio_head_weight=None,
        video_head_weight=None,
        audio_head_gram_diagonal=None,
        video_head_gram_diagonal=None,
        forecast_risk_prior=0.2,
        build_seconds=0.0,
        estimated_bytes=0,
        transient_workspace_bytes=0,
    )


def _runtime(**overrides):
    values = {
        "degree": 1,
        "max_history": 4,
        "warmup_steps": 2,
        "tail_actual_steps": 0,
        "window_size": 2.0,
        "bootstrap_first_forecast": False,
        "offline_smoothing_replay": False,
    }
    values.update(overrides)
    return SpectrumH3Runtime(SpectrumH3Config(**values))


def _actual_step(runtime, timestep, records):
    decision = runtime.begin_step(torch.tensor([timestep]))
    assert decision["actual"]
    for labels, feature in records:
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=labels,
            expected_shape=tuple(feature.shape),
        )
        assert actual
        runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, feature)
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision


def _forecast_step(runtime, timestep, labels=LABEL):
    decision = runtime.begin_step(torch.tensor([timestep]))
    assert not decision["actual"]
    shape = (len(labels), 3, 4)
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=labels,
        expected_shape=shape,
    )
    assert not actual
    prediction = runtime.predict(
        decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32
    )
    assert prediction is not None
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return prediction


def _complete_step(runtime, timestep, *, labels=LABEL):
    decision = runtime.begin_step(torch.tensor([timestep]))
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=labels,
        expected_shape=(len(labels), 3, 4),
    )
    if actual:
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((len(labels), 3, 4), float(decision["step_id"])),
        )
    else:
        prediction = runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert prediction is not None
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision



def test_explicit_pece_topology_separates_same_sigma_phases_and_outer_policy():
    runtime = _runtime(warmup_steps=0, tail_actual_steps=1)
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    topology = (
        SolverCallDescriptor(0, 0, "predicted"),
        SolverCallDescriptor(1, 0, "predicted"),
        SolverCallDescriptor(1, 1, "corrected"),
        SolverCallDescriptor(2, 0, "predicted"),
        SolverCallDescriptor(2, 1, "corrected"),
        SolverCallDescriptor(3, 0, "predicted"),
        SolverCallDescriptor(3, 1, "corrected"),
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
        history_step_ids=(0, 2, 4, 6),
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
        min_actual_prefix_steps=2,
        min_actual_steps_after_forecast=0,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )

    decisions = [
        _complete_step(runtime, sigma)
        for sigma in (1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25)
    ]

    assert [decision["phase"] for decision in decisions] == [
        "predicted",
        "predicted",
        "corrected",
        "predicted",
        "corrected",
        "predicted",
        "corrected",
    ]
    assert [decision["policy_step_id"] for decision in decisions] == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [decision["actual"] for decision in decisions] == [
        True,
        True,
        True,
        False,
        True,
        False,
        True,
    ]
    assert runtime.stats.phase_actual_steps == {"predicted": 2, "corrected": 3}
    assert runtime.stats.phase_forecast_steps == {"predicted": 2}
    assert runtime.forecaster.history_length == 4
    assert runtime.forecaster.latest_anchor_ids(4) == (0, 2, 4, 6)
    assert runtime._stage_forecasters == {}
    runtime.end_run(run_id)


def test_terminal_pece_exact_corrector_proof_uses_explicit_topology():
    runtime = _runtime(warmup_steps=0, tail_actual_steps=0)
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    topology = (
        SolverCallDescriptor(0, 0, "predicted"),
        SolverCallDescriptor(1, 0, "predicted"),
        SolverCallDescriptor(1, 1, "corrected"),
        SolverCallDescriptor(2, 0, "predicted"),
        SolverCallDescriptor(2, 1, "corrected"),
        SolverCallDescriptor(3, 0, "predicted"),
        SolverCallDescriptor(3, 1, "corrected"),
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
        history_step_ids=(0, 2, 4, 6),
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
        min_actual_prefix_steps=2,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )

    for sigma in (1.0, 0.75, 0.75):
        _complete_step(runtime, sigma)

    interior = runtime.begin_step(torch.tensor([0.5]))
    assert interior["phase"] == "predicted"
    assert runtime.terminal_pece_exact_corrector_after_current_step() is None
    runtime.abort_step(interior["run_id"], interior["step_id"])
    runtime.end_run(run_id)


def test_terminal_pece_exact_corrector_proof_identifies_final_predicted_call():
    runtime = _runtime(warmup_steps=0, tail_actual_steps=0)
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
        expected_model_calls=3,
        stage_count=2,
        logical_call_topology=topology,
        state_conditioned_residual=True,
        separate_stage_histories=False,
        forecastable_stage_indices=(0,),
        history_stage_indices=(0, 1),
        history_step_ids=(0, 2),
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )
    _complete_step(runtime, 1.0)
    decision = runtime.begin_step(torch.tensor([0.5]))
    assert decision["actual"] is False
    assert runtime.terminal_pece_exact_corrector_after_current_step() == (
        ExactCorrectorGuarantee(1, 2, 1)
    )
    runtime.abort_step(decision["run_id"], decision["step_id"])
    runtime.end_run(run_id)


def test_terminal_pece_exact_corrector_proof_rejects_one_lane_pec():
    runtime = _runtime(warmup_steps=0, tail_actual_steps=0)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    topology = (
        SolverCallDescriptor(0, 0, "model"),
        SolverCallDescriptor(1, 0, "model"),
    )
    run_id = runtime.start_run(
        sigmas,
        "sample_sa_solver",
        supported_sampler=True,
        expected_model_calls=2,
        stage_count=1,
        logical_call_topology=topology,
        forecastable_stage_indices=(0,),
        history_stage_indices=(0,),
        tail_actual_stage_indices=(),
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )
    _complete_step(runtime, 1.0)
    decision = runtime.begin_step(torch.tensor([0.5]))
    assert runtime.terminal_pece_exact_corrector_after_current_step() is None
    runtime.abort_step(decision["run_id"], decision["step_id"])
    runtime.end_run(run_id)


def test_active_pece_corrected_phase_owns_shared_endpoint_history():
    runtime = _runtime(force_actual=True, warmup_steps=0, tail_actual_steps=0)
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
        expected_model_calls=3,
        stage_count=2,
        logical_call_topology=topology,
        state_conditioned_residual=True,
        separate_stage_histories=False,
        forecastable_stage_indices=(0,),
        history_stage_indices=(0, 1),
        history_step_ids=(0, 2),
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
    )

    _complete_step(runtime, 1.0)
    _complete_step(runtime, 0.5)
    assert runtime.forecaster.history_length == 1
    assert runtime.forecaster.latest_anchor_ids(1) == (0,)

    decision = runtime.begin_step(torch.tensor([0.5]))
    assert decision["phase"] == "corrected"
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        torch.full((1, 3, 4), 123.0),
    )
    assert runtime._step is not None
    assert runtime._step.calls[0].observed_actual
    assert len(runtime._step.actual_records) == 1
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert runtime.stats.actual_transformer_calls == 3
    assert runtime.forecaster.history_length == 2
    assert runtime.forecaster.latest_anchor_ids(2) == (0, 2)
    assert runtime._stage_forecasters == {}
    runtime.end_run(run_id)


def test_active_pece_default_19_interval_cadence_counts_true_h3_opportunities():
    runtime = _runtime(warmup_steps=1, tail_actual_steps=1)
    sigmas = torch.cat((torch.linspace(1.0, 0.1, 19), torch.zeros(1)))
    topology = [SolverCallDescriptor(0, 0, "predicted")]
    for outer_step in range(1, 19):
        topology.extend(
            (
                SolverCallDescriptor(outer_step, 0, "predicted"),
                SolverCallDescriptor(outer_step, 1, "corrected"),
            )
        )
    topology = tuple(topology)
    history_step_ids = tuple(
        step_id
        for step_id, descriptor in enumerate(topology)
        if step_id == 0 or descriptor.phase == "corrected"
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
        history_step_ids=history_step_ids,
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
        min_actual_steps_after_forecast=0,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )

    decisions = [
        _complete_step(runtime, float(sigmas[descriptor.outer_step]))
        for descriptor in topology
    ]
    predicted = [
        decision for decision in decisions if decision["phase"] == "predicted"
    ]
    corrected = [
        decision for decision in decisions if decision["phase"] == "corrected"
    ]

    assert len(decisions) == 37
    assert len(predicted) == 19
    assert len(corrected) == 18
    assert sum(decision["actual"] for decision in predicted) == 1
    assert sum(not decision["actual"] for decision in predicted) == 18
    assert all(decision["actual"] for decision in corrected)
    assert runtime.stats.actual_steps == 19
    assert runtime.stats.forecast_steps == 18
    assert runtime.stats.actual_steps + runtime.stats.forecast_steps == 37
    assert runtime.stats.actual_transformer_calls == 19
    assert runtime.stats.phase_actual_steps == {"predicted": 1, "corrected": 18}
    assert runtime.stats.phase_forecast_steps == {"predicted": 18}
    assert runtime.forecaster.latest_anchor_ids(4) == history_step_ids[-4:]
    summary = runtime.debug_summary()
    assert "outer_steps=19" in summary
    assert "h3_logical_calls=37" in summary
    assert "phase_counts=corrected:a18/f0,predicted:a1/f18" in summary
    runtime.end_run(run_id)


@pytest.mark.parametrize(
    ("policy_prefix", "predicted_actuals", "total_actuals", "forecasts"),
    (
        (1, 1, 10, 9),
        (2, 2, 11, 8),
        (3, 3, 12, 7),
    ),
)
def test_active_pece_10_interval_policy_prefix_controls_speed_stability_tradeoff(
    policy_prefix,
    predicted_actuals,
    total_actuals,
    forecasts,
):
    runtime = _runtime(warmup_steps=1, tail_actual_steps=1)
    sigmas = torch.cat((torch.linspace(1.0, 0.1, 10), torch.zeros(1)))
    topology = [SolverCallDescriptor(0, 0, "predicted")]
    for outer_step in range(1, 10):
        topology.extend(
            (
                SolverCallDescriptor(outer_step, 0, "predicted"),
                SolverCallDescriptor(outer_step, 1, "corrected"),
            )
        )
    topology = tuple(topology)
    history_step_ids = tuple(
        step_id
        for step_id, descriptor in enumerate(topology)
        if step_id == 0 or descriptor.phase == "corrected"
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
        history_step_ids=history_step_ids,
        tail_actual_stage_indices=(1,),
        allow_state_conditioned_bootstrap=True,
        min_actual_steps_after_forecast=0,
        min_sampler_actual_prefix_steps=policy_prefix,
        max_consecutive_forecasts=1,
        model_aware_can_force_actual=False,
    )

    decisions = [
        _complete_step(runtime, float(sigmas[descriptor.outer_step]))
        for descriptor in topology
    ]
    predicted = [d for d in decisions if d["phase"] == "predicted"]
    corrected = [d for d in decisions if d["phase"] == "corrected"]

    assert len(decisions) == 19
    assert [d["actual"] for d in predicted] == (
        [True] * predicted_actuals + [False] * (10 - predicted_actuals)
    )
    assert all(d["actual"] for d in corrected)
    assert runtime.stats.actual_steps == total_actuals
    assert runtime.stats.forecast_steps == forecasts
    assert runtime.stats.actual_transformer_calls == total_actuals
    assert runtime.stats.phase_actual_steps == {
        "predicted": predicted_actuals,
        "corrected": 9,
    }
    assert runtime.stats.phase_forecast_steps == {
        "predicted": 10 - predicted_actuals,
    }
    assert runtime.forecaster.latest_anchor_ids(4) == history_step_ids[-4:]
    summary = runtime.debug_summary()
    assert f"sampler_prefix_steps={policy_prefix}" in summary
    runtime.end_run(run_id)

def test_model_aware_off_never_enters_controller_and_keeps_legacy_forecast_path(
    monkeypatch,
):
    runtime = _runtime(model_aware_mode="off")

    def unexpected_decision(**_kwargs):
        raise AssertionError("model-aware controller must not run in off mode")

    monkeypatch.setattr(runtime.model_aware, "decision", unexpected_decision)
    run_id = runtime.start_run(
        torch.linspace(1.0, 0.0, 5),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(LABEL, torch.zeros((1, 3, 4)))])
    _actual_step(runtime, 0.75, [(LABEL, torch.ones((1, 3, 4)))])

    prediction = _forecast_step(runtime, 0.5)

    assert prediction.shape == (1, 3, 4)
    assert runtime.stats.model_aware_decision_seconds == 0.0
    assert runtime.stats.model_aware_correction_seconds == 0.0
    runtime.end_run(run_id)


def test_stochastic_seeds_uses_independent_stage_forecast_histories():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=5,
        stage_count=2,
        stochastic_multistage=True,
    )

    decisions = []
    for timestep, feature_value in (
        (1.0, 0.0),
        (0.85, 100.0),
        (0.7, 2.0),
        (0.55, 102.0),
    ):
        decision = runtime.begin_step(torch.tensor([timestep]))
        decisions.append(decision)
        assert decision["actual"]
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert actual
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((1, 3, 4), feature_value),
        )
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert [decision["step_id"] for decision in decisions] == [0, 1, 2, 3]
    assert runtime._stage_forecasters[0].history_length == 2
    assert runtime._stage_forecasters[1].history_length == 2

    forecast = runtime.begin_step(torch.tensor([0.4]))
    assert forecast["step_id"] == 4
    assert runtime.active_stage_index == 0
    assert runtime.active_state_residual_mode is True
    assert forecast["actual"] is False
    call_id, actual = runtime.begin_model_call(
        forecast["run_id"],
        forecast["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert not actual
    prediction = runtime.predict(
        forecast["run_id"],
        forecast["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert prediction is not None
    assert float(prediction.abs().max()) < 20.0
    runtime.finalize_step(forecast["run_id"], forecast["step_id"])
    runtime.end_run(run_id)


def test_stochastic_multistage_disables_one_point_bootstrap_per_stage_lane():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=1,
        tail_actual_steps=0,
        bootstrap_first_forecast=True,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        expected_model_calls=5,
        stage_count=2,
        state_conditioned_residual=True,
    )

    first = _complete_step(runtime, 1.0)
    second = _complete_step(runtime, 0.9)
    third = _complete_step(runtime, 0.7)
    fourth = _complete_step(runtime, 0.6)

    assert first["actual"] and second["actual"]
    assert third["actual"] and fourth["actual"]
    assert third["reason"] == "insufficient actual history"
    assert fourth["reason"] == "insufficient actual history"
    assert "bootstrap" not in third["reason"]
    assert "bootstrap" not in fourth["reason"]
    assert runtime._stage_forecasters[0].history_length == 2
    assert runtime._stage_forecasters[1].history_length == 2
    runtime.end_run(run_id)


def test_stochastic_multistage_requires_actual_refresh_in_the_same_lane():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=9,
        stage_count=2,
        state_conditioned_residual=True,
    )

    decisions = [
        _complete_step(runtime, timestep)
        for timestep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
    ]

    assert [decision["actual"] for decision in decisions[:4]] == [True, True, True, True]
    assert decisions[4]["actual"] is False
    assert decisions[5]["actual"] is True
    assert decisions[6]["actual"] is True
    assert decisions[6]["reason"] == "post-forecast stage-lane refresh"
    assert decisions[7]["actual"] is False
    assert decisions[8]["actual"] is True
    assert runtime._stage_required_actual_refreshes[0] == 0
    assert runtime._stage_required_actual_refreshes[1] == 1
    runtime.end_run(run_id)


def test_state_conditioned_single_lane_disables_one_point_bootstrap():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=1,
        tail_actual_steps=0,
        bootstrap_first_forecast=True,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        expected_model_calls=3,
        stage_count=1,
        state_conditioned_residual=True,
    )

    first = _complete_step(runtime, 1.0)
    second = _complete_step(runtime, 0.7)
    third = _complete_step(runtime, 0.4)

    assert first["actual"]
    assert second["actual"]
    assert second["reason"] == "insufficient actual history"
    assert third["actual"] is False
    assert "bootstrap" not in second["reason"]
    runtime.end_run(run_id)


def test_stochastic_seeds_protects_outer_stage_and_forecasts_internal_stage_only():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=9,
        stage_count=2,
        state_conditioned_residual=True,
        forecastable_stage_indices=(1,),
    )

    decisions = [
        _complete_step(runtime, timestep)
        for timestep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
    ]
    stage0 = decisions[0::2]
    stage1 = decisions[1::2]
    assert all(decision["actual"] for decision in stage0)
    assert all(
        decision["reason"] == "sampler-required exact stage"
        for decision in stage0[2:]
    )
    assert [decision["actual"] for decision in stage1] == [True, True, False, True]
    assert stage1[2]["reason"] == "adaptive forecast"
    assert stage1[3]["reason"] == "post-forecast stage-lane refresh"
    runtime.end_run(run_id)


def test_stochastic_seeds_shared_history_uses_exact_outer_anchors_without_lane_refresh():
    runtime = _runtime(
        model_aware_mode="schedule_confidence",
        model_aware_risk_threshold=0.0,
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    runtime.set_model_profile(
        ProfileLookup(
            profile=_test_model_profile(),
            cache_hit=False,
            lookup_seconds=0.0,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        expected_model_calls=9,
        stage_count=2,
        state_conditioned_residual=True,
        separate_stage_histories=False,
        forecastable_stage_indices=(1,),
        model_aware_can_force_actual=False,
    )

    decisions = [
        _complete_step(runtime, timestep)
        for timestep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
    ]

    stage0 = decisions[0::2]
    stage1 = decisions[1::2]
    assert all(decision["actual"] for decision in stage0)
    assert [decision["actual"] for decision in stage1] == [True, False, False, False]
    assert all(
        decision["reason"] == "adaptive forecast"
        for decision in stage1[1:]
    )
    assert runtime._run is not None
    assert runtime._run.separate_stage_histories is False
    assert runtime._stage_forecasters == {}
    assert runtime._stage_required_actual_refreshes == {}
    assert runtime.forecaster is runtime._primary_forecaster
    assert runtime.forecaster.history_length == 4
    assert runtime.stats.actual_steps == 6
    assert runtime.stats.forecast_steps == 3
    assert runtime.stats.model_aware_veto_suppressed == 3
    assert runtime.stats.model_aware_forecasts == 3
    runtime.end_run(run_id)


def test_separate_stage_histories_validation():
    runtime = _runtime(force_actual=True)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="boolean or None"):
        runtime.start_run(
            sigmas,
            "sample_seeds_2",
            supported_sampler=True,
            expected_model_calls=3,
            stage_count=2,
            state_conditioned_residual=True,
            separate_stage_histories=1,
        )

    with pytest.raises(ValueError, match="requires an explicit multistage topology"):
        runtime.start_run(
            sigmas,
            "sample_sa_solver",
            supported_sampler=True,
            expected_model_calls=2,
            stage_count=1,
            state_conditioned_residual=True,
            separate_stage_histories=True,
        )


def test_forecastable_stage_indices_validation():
    runtime = _runtime(force_actual=True)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="duplicates"):
        runtime.start_run(
            sigmas,
            "sample_seeds_2",
            supported_sampler=True,
            expected_model_calls=3,
            stage_count=2,
            state_conditioned_residual=True,
            forecastable_stage_indices=(1, 1),
        )

    with pytest.raises(ValueError, match="within"):
        runtime.start_run(
            sigmas,
            "sample_seeds_2",
            supported_sampler=True,
            expected_model_calls=3,
            stage_count=2,
            state_conditioned_residual=True,
            forecastable_stage_indices=(2,),
        )


def test_sampler_required_exact_step_overrides_ready_single_lane_forecast():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=4,
        state_conditioned_residual=True,
        forced_actual_step_ids=(2,),
    )

    first = _complete_step(runtime, 1.0)
    second = _complete_step(runtime, 0.8)
    third = _complete_step(runtime, 0.6)
    fourth = _complete_step(runtime, 0.4)

    assert first["actual"]
    assert second["actual"]
    assert third["actual"]
    assert third["reason"] == "sampler-required exact step"
    assert fourth["actual"] is False
    assert fourth["reason"] == "adaptive forecast"
    runtime.end_run(run_id)


def test_stochastic_sa_targets_eight_alternating_forecasts_without_noise_block():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=1,
        tail_actual_steps=1,
        max_history=8,
        window_size=2.0,
        flex_window=0.75,
    )
    sigmas = torch.linspace(1.0, 0.0, 20)
    run_id = runtime.start_run(
        sigmas,
        "sample_sa_solver",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=19,
        state_conditioned_residual=True,
        forced_actual_step_ids=(),
        forced_actual_steps_advance_window=False,
        model_aware_can_force_actual=False,
    )

    decisions = [
        _complete_step(runtime, float(timestep))
        for timestep in sigmas[:-1]
    ]
    forecast_steps = [
        decision["step_id"] for decision in decisions if not decision["actual"]
    ]

    assert forecast_steps == [2, 4, 6, 8, 10, 12, 14, 16]
    assert runtime.stats.actual_steps == 11
    assert runtime.stats.forecast_steps == 8
    runtime.end_run(run_id)


def test_forced_actual_step_ids_validation():
    runtime = _runtime(force_actual=True)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="duplicates"):
        runtime.start_run(
            sigmas,
            "sample_sa_solver",
            supported_sampler=True,
            expected_model_calls=2,
            forced_actual_step_ids=(1, 1),
        )

    with pytest.raises(ValueError, match="nonnegative"):
        runtime.start_run(
            sigmas,
            "sample_sa_solver",
            supported_sampler=True,
            expected_model_calls=2,
            forced_actual_step_ids=(-1,),
        )

    with pytest.raises(ValueError, match="smaller than total"):
        runtime.start_run(
            sigmas,
            "sample_sa_solver",
            supported_sampler=True,
            expected_model_calls=2,
            forced_actual_step_ids=(2,),
        )


def test_stochastic_seeds_3_stage_lane_indexing_matches_native_call_order():
    runtime = _runtime(
        model_aware_mode="off",
        force_actual=True,
        warmup_steps=0,
        tail_actual_steps=0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_seeds_3",
        supported_sampler=True,
        expected_model_calls=4,
        stage_count=3,
        stochastic_multistage=True,
    )

    observed_stages = []
    for timestep in (1.0, 0.8, 0.6, 0.5):
        decision = runtime.begin_step(torch.tensor([timestep]))
        observed_stages.append(runtime.active_stage_index)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert actual
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.zeros((1, 3, 4)),
        )
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert observed_stages == [0, 1, 2, 0]
    assert runtime._stage_forecasters[0].history_length == 2
    assert runtime._stage_forecasters[1].history_length == 1
    assert runtime._stage_forecasters[2].history_length == 1
    runtime.end_run(run_id)


def test_state_conditioned_single_lane_is_independent_from_multistage_seeds():
    runtime = _runtime(
        model_aware_mode="off",
        warmup_steps=0,
        tail_actual_steps=0,
        bootstrap_first_forecast=False,
        window_size=2.0,
        flex_window=0.0,
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=3,
        stage_count=1,
        state_conditioned_residual=True,
    )

    assert runtime.state_conditioned_residual is True
    assert runtime.stochastic_multistage is False
    assert runtime._stage_forecasters == {}

    for timestep, value in ((1.0, 1.0), (0.7, 2.0)):
        decision = runtime.begin_step(torch.tensor([timestep]))
        assert decision["actual"]
        assert runtime.active_stage_index == 0
        assert runtime.active_state_conditioned_residual is True
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert actual
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((1, 3, 4), value),
        )
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    forecast = runtime.begin_step(torch.tensor([0.4]))
    assert forecast["actual"] is False
    assert runtime.active_stage_index == 0
    assert runtime.prediction_history_length == 2
    runtime.abort_step(forecast["run_id"], forecast["step_id"])
    runtime.end_run(run_id)


def test_sa_and_seeds_state_conditioned_runs_do_not_leak_stage_histories():
    runtime = _runtime(
        model_aware_mode="off",
        force_actual=True,
        warmup_steps=0,
        tail_actual_steps=0,
    )

    sa_run = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        expected_model_calls=2,
        stage_count=1,
        state_conditioned_residual=True,
    )
    for timestep in (1.0, 0.5):
        _complete_step(runtime, timestep)
    assert runtime.state_conditioned_residual is True
    assert runtime.stochastic_multistage is False
    assert runtime._stage_forecasters == {}
    runtime.end_run(sa_run)

    seeds_run = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        expected_model_calls=3,
        stage_count=2,
        state_conditioned_residual=True,
    )
    assert runtime.state_conditioned_residual is True
    assert runtime.stochastic_multistage is True
    assert set(runtime._stage_forecasters) == {0, 1}
    for timestep in (1.0, 0.75, 0.5):
        _complete_step(runtime, timestep)
    runtime.end_run(seeds_run)

    sa_again = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        expected_model_calls=2,
        stage_count=1,
        state_conditioned_residual=True,
    )
    assert runtime._stage_forecasters == {}
    assert runtime.prediction_history_length == 0
    assert runtime.stochastic_multistage is False
    runtime.end_run(sa_again)



@pytest.mark.parametrize(
    ("middle_sampler", "middle_stage_count", "middle_calls"),
    (
        ("sample_sa_solver", 1, 3),
        ("sample_seeds_2", 2, 6),
        ("sample_seeds_3", 3, 9),
    ),
)
def test_active_pece_runtime_isolation_across_pec_and_seeds(
    middle_sampler,
    middle_stage_count,
    middle_calls,
):
    runtime = _runtime(
        model_aware_mode="off",
        force_actual=True,
        warmup_steps=0,
        tail_actual_steps=0,
    )
    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    pece_topology = (
        SolverCallDescriptor(0, 0, "predicted"),
        SolverCallDescriptor(1, 0, "predicted"),
        SolverCallDescriptor(1, 1, "corrected"),
        SolverCallDescriptor(2, 0, "predicted"),
        SolverCallDescriptor(2, 1, "corrected"),
    )
    pece_history_step_ids = (0, 2, 4)

    def run_pece():
        run_id = runtime.start_run(
            sigmas,
            "sample_sa_solver_pece",
            supported_sampler=True,
            expected_model_calls=len(pece_topology),
            stage_count=2,
            logical_call_topology=pece_topology,
            state_conditioned_residual=True,
            separate_stage_histories=False,
            forecastable_stage_indices=(0,),
            history_stage_indices=(0, 1),
            history_step_ids=pece_history_step_ids,
            tail_actual_stage_indices=(1,),
            allow_state_conditioned_bootstrap=True,
        )
        assert runtime.prediction_history_length == 0
        assert runtime.stochastic_multistage is False
        for timestep in (1.0, 0.6, 0.6, 0.3, 0.3):
            _complete_step(runtime, timestep)
        assert runtime.forecaster.history_length == 3
        assert runtime.forecaster.latest_anchor_ids(3) == pece_history_step_ids
        assert runtime._stage_forecasters == {}
        runtime.end_run(run_id)

    run_pece()
    middle_run = runtime.start_run(
        sigmas,
        middle_sampler,
        supported_sampler=True,
        expected_model_calls=middle_calls,
        stage_count=middle_stage_count,
        state_conditioned_residual=True,
        separate_stage_histories=middle_stage_count > 1,
    )
    assert runtime.prediction_history_length == 0
    assert runtime._run is not None
    assert runtime._run.logical_call_topology is None
    assert runtime.stochastic_multistage is (middle_stage_count > 1)
    for call_id in range(middle_calls):
        _complete_step(runtime, float(sigmas[min(call_id // middle_stage_count, 2)]))
    runtime.end_run(middle_run)
    run_pece()

def test_legacy_stochastic_multistage_keyword_maps_to_generic_state_conditioning():
    runtime = _runtime(force_actual=True)
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        expected_model_calls=3,
        stage_count=2,
        stochastic_multistage=True,
    )

    assert runtime.state_conditioned_residual is True
    assert runtime.stochastic_multistage is True
    assert set(runtime._stage_forecasters) == {0, 1}
    runtime.end_run(run_id)


def test_runtime_accepts_expanded_model_call_count_for_multistage_sampler():
    runtime = _runtime(force_actual=True)
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        expected_model_calls=3,
    )

    for timestep in (1.0, 0.75, 0.5):
        _complete_step(runtime, timestep)

    assert runtime.stats.total_steps == 3
    with pytest.raises(RuntimeError, match="predict_noise call count exceeded"):
        runtime.begin_step(torch.tensor([0.25]))
    runtime.end_run(run_id)


def test_runtime_rejects_model_call_count_smaller_than_sigma_interval_count():
    runtime = _runtime(force_actual=True)

    with pytest.raises(ValueError, match="expected_model_calls"):
        runtime.start_run(
            torch.tensor([1.0, 0.75, 0.5, 0.0]),
            "sample_seeds_2",
            supported_sampler=True,
            expected_model_calls=2,
        )


def test_preliminary_runtime_defaults():
    config = SpectrumH3Config()

    assert config.degree == 1
    assert config.warmup_steps == 1
    assert config.tail_actual_steps == 1
    assert config.history_storage == "system_ram"
    assert config.offline_archive_storage == "system_ram"
    assert config.bootstrap_first_forecast is True


@pytest.mark.parametrize(
    ("steps", "expected_actual", "expected_forecast"),
    [
        (17, [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15]),
        (20, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19], [1, 3, 5, 7, 9, 11, 13, 15, 17]),
    ],
)
def test_bootstrap_degree_one_euler_schedule(steps, expected_actual, expected_forecast):
    runtime = _runtime(
        bootstrap_first_forecast=True,
        warmup_steps=1,
        tail_actual_steps=1,
        window_size=2.0,
        flex_window=0.75,
    )
    runtime.start_run(
        torch.linspace(1.0, 0.0, steps + 1),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    decisions = [
        _complete_step(runtime, float(sigma))
        for sigma in torch.linspace(1.0, 1.0 / steps, steps)
    ]
    actual_indices = [decision["step_id"] for decision in decisions if decision["actual"]]
    forecast_indices = [decision["step_id"] for decision in decisions if not decision["actual"]]

    assert actual_indices == expected_actual
    assert forecast_indices == expected_forecast
    assert runtime.stats.actual_steps == len(expected_actual)
    assert runtime.stats.forecast_steps == len(expected_forecast)
    assert decisions[1]["reason"] == "one-point bootstrap forecast"
    assert decisions[2]["reason"] == "insufficient actual history"


def test_max_consecutive_forecasts_forces_a_correction_after_bootstrap_scheduling():
    runtime = _runtime(
        bootstrap_first_forecast=True,
        warmup_steps=1,
        tail_actual_steps=0,
        window_size=4.0,
        flex_window=0.0,
    )
    runtime.start_run(
        torch.linspace(1.0, 0.0, 6),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
    )

    decisions = [_complete_step(runtime, float(sigma)) for sigma in torch.linspace(1.0, 0.2, 5)]

    assert [decision["actual"] for decision in decisions] == [True, False, True, False, True]
    assert decisions[4]["reason"] == "post-forecast sampler refresh"


def test_disabling_bootstrap_preserves_degree_one_startup_schedule():
    runtime = _runtime(warmup_steps=1, bootstrap_first_forecast=False)
    runtime.start_run(
        torch.linspace(1.0, 0.0, 5),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    decisions = [_complete_step(runtime, float(sigma)) for sigma in torch.linspace(1.0, 0.25, 4)]

    assert [decision["actual"] for decision in decisions] == [True, True, False, True]
    assert decisions[1]["reason"] == "insufficient actual history"


def test_final_tail_precedes_bootstrap_on_a_two_step_run():
    runtime = _runtime(
        bootstrap_first_forecast=True,
        warmup_steps=1,
        tail_actual_steps=1,
    )
    runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    _complete_step(runtime, 1.0)
    decision = _complete_step(runtime, 0.5)

    assert decision["actual"]
    assert decision["reason"] == "final actual tail"


@pytest.mark.parametrize(
    ("config_overrides", "supported_sampler", "expected_reason"),
    [
        ({"force_actual": True}, True, "forced-actual validation mode"),
        ({"enabled": False}, True, "forecasting disabled by configuration"),
        ({}, False, "not allowlisted"),
    ],
)
def test_native_path_precedence_overrides_bootstrap(
    config_overrides,
    supported_sampler,
    expected_reason,
):
    runtime = _runtime(
        bootstrap_first_forecast=True,
        warmup_steps=1,
        **config_overrides,
    )
    runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler" if supported_sampler else "sample_heun",
        supported_sampler=supported_sampler,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    _complete_step(runtime, 1.0)
    second = runtime.begin_step(torch.tensor([0.5]))

    assert second["actual"]
    assert expected_reason in second["reason"]
    runtime.abort_step(second["run_id"], second["step_id"])


def test_bootstrap_reordered_branch_calls_hold_the_matching_canonical_rows():
    positive = ((0, "positive"),)
    negative = ((1, "negative"),)
    both = (negative[0], positive[0])
    feature = torch.stack((torch.full((3, 4), -2.0), torch.full((3, 4), 3.0)))
    runtime = _runtime(bootstrap_first_forecast=True, warmup_steps=1)
    runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(both, feature)])

    decision = runtime.begin_step(torch.tensor([0.5]))
    predictions = {}
    for labels in (positive, negative):
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=labels,
            expected_shape=(1, 3, 4),
        )
        assert not actual
        predictions[labels] = runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    torch.testing.assert_close(predictions[positive], torch.full((1, 3, 4), 3.0))
    torch.testing.assert_close(predictions[negative], torch.full((1, 3, 4), -2.0))
    assert runtime.forecaster.history_length == 1
    assert runtime.forecaster.factorization_count == 0


def test_incomplete_bootstrap_forecast_retries_the_whole_step_as_actual():
    labels = ((0, "a"), (1, "b"))
    runtime = _runtime(bootstrap_first_forecast=True, warmup_steps=1)
    runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(labels, torch.ones(2, 3, 4))])

    decision = runtime.begin_step(torch.tensor([0.5]))
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=(labels[0],),
        expected_shape=(1, 3, 4),
    )
    assert not actual
    assert runtime.predict(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) is not None
    with pytest.raises(ForecastRetryActual, match="incomplete"):
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], "incomplete branch set")
    for label in reversed(labels):
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=(label,),
            expected_shape=(1, 3, 4),
        )
        assert actual
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((1, 3, 4), 2.0),
        )
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert runtime.stats.forecast_fallbacks == 1
    assert runtime.stats.actual_steps == 2
    assert runtime.stats.forecast_steps == 0
    assert runtime.forecaster.history_length == 2


def test_aborted_bootstrap_prediction_retries_step_one_without_scheduler_corruption():
    runtime = _runtime(bootstrap_first_forecast=True, warmup_steps=1)
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(LABEL, torch.ones(1, 3, 4))])

    first = runtime.begin_step(torch.tensor([0.5]))
    call_id, _ = runtime.begin_model_call(
        first["run_id"],
        first["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    runtime.predict(
        first["run_id"],
        first["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    runtime.abort_step(run_id, first["step_id"])

    repeated = _complete_step(runtime, 0.5)

    assert repeated["step_id"] == first["step_id"] == 1
    assert repeated["reason"] == "one-point bootstrap forecast"
    assert runtime.stats.actual_steps == 1
    assert runtime.stats.forecast_steps == 1
    assert runtime.forecaster.history_length == 1


def test_inconsistent_bootstrap_history_uses_native_fallback():
    runtime = _runtime(bootstrap_first_forecast=True, warmup_steps=1)
    runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    _actual_step(runtime, 1.0, [(LABEL, torch.ones(1, 3, 4))])
    decision = runtime.begin_step(torch.tensor([0.5]))
    runtime.forecaster.update(0.75, torch.full((1, 3, 4), 2.0))
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert not actual

    prediction = runtime.predict(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prediction is None
    assert runtime._step.mode == "actual"
    assert not runtime._step.bootstrap_forecast
    assert "requires exactly one actual history entry" in runtime.disabled_reason


def test_scheduler_counts_warmup_forecasts_recomputes_and_window_growth():
    runtime = _runtime()
    run_id = runtime.start_run(torch.linspace(1.0, 0.0, 7), "sample_euler", supported_sampler=True)
    for step, sigma in enumerate(torch.linspace(1.0, 1.0 / 6.0, 6)):
        if step in (0, 1, 3, 5):
            _actual_step(runtime, float(sigma), [(LABEL, torch.full((1, 3, 4), float(step)))])
        else:
            _forecast_step(runtime, float(sigma))
    assert runtime.stats.actual_steps == 4
    assert runtime.stats.forecast_steps == 2
    assert runtime.stats.actual_transformer_calls == 4
    assert runtime.stats.current_window == pytest.approx(3.5)
    runtime.end_run(run_id)
    assert runtime.forecaster.history_length == 0


def test_split_branches_are_canonicalized_and_reordered_transactionally():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 5), "sample_euler", supported_sampler=True)
    positive = ((0, "positive"),)
    negative = ((1, "negative"),)
    _actual_step(runtime, 1.0, [(negative, torch.full((1, 3, 4), -1.0)), (positive, torch.ones(1, 3, 4))])
    _actual_step(runtime, 0.75, [(positive, torch.full((1, 3, 4), 2.0)), (negative, torch.full((1, 3, 4), -2.0))])

    decision = runtime.begin_step(torch.tensor([0.5]))
    predictions = {}
    for labels in (negative, positive):
        call_id, actual = runtime.begin_model_call(
            decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=labels, expected_shape=(1, 3, 4)
        )
        assert not actual
        predictions[labels] = runtime.predict(
            decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32
        )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert predictions[positive].mean() > 0
    assert predictions[negative].mean() < 0


def test_single_call_actual_history_takes_ownership_without_restaking_rows():
    runtime = _runtime(force_actual=True)
    runtime.start_run(torch.linspace(1.0, 0.0, 3), "sample_euler", supported_sampler=True)
    decision = runtime.begin_step(torch.tensor([1.0]))
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=tuple(feature.shape),
    )
    assert actual
    runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, feature)
    archived_ptr = runtime._step.actual_records[0].feature.data_ptr()

    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert runtime.stats.direct_history_updates == 1
    assert runtime.forecaster._history[-1].feature_flat.data_ptr() == archived_ptr
    torch.testing.assert_close(runtime.forecaster._history[-1].feature_flat.reshape_as(feature), feature)


def test_vram_archive_is_compact_owned_storage():
    runtime = _runtime(force_actual=True, history_storage="vram")
    runtime.start_run(torch.linspace(1.0, 0.0, 3), "sample_euler", supported_sampler=True)
    decision = runtime.begin_step(torch.tensor([1.0]))
    backing = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    feature = backing[1:].unsqueeze(0).reshape(1, 3, 4)
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=tuple(feature.shape),
    )
    assert actual
    expected = feature.clone()
    runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, feature)
    archived = runtime._step.actual_records[0].feature
    backing.fill_(-1.0)
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert archived.data_ptr() != feature.data_ptr()
    assert archived.untyped_storage().nbytes() == archived.numel() * archived.element_size()
    torch.testing.assert_close(runtime.forecaster._history[-1].feature_flat.reshape_as(expected), expected)


def test_split_call_actual_history_still_canonicalizes_rows():
    runtime = _runtime(force_actual=True)
    runtime.start_run(torch.linspace(1.0, 0.0, 3), "sample_euler", supported_sampler=True)
    negative = ((1, "negative"),)
    positive = ((0, "positive"),)
    _actual_step(
        runtime,
        1.0,
        [
            (negative, torch.full((1, 3, 4), -1.0)),
            (positive, torch.ones(1, 3, 4)),
        ],
    )

    assert runtime.stats.direct_history_updates == 0
    history = runtime.forecaster._history[-1].feature_flat.reshape(2, 3, 4)
    assert history[0].mean() > 0
    assert history[1].mean() < 0


def test_incomplete_forecast_requires_whole_step_actual_retry():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 5), "sample_euler", supported_sampler=True)
    labels = (((0, "a"), (1, "b")))
    _actual_step(runtime, 1.0, [(labels, torch.ones(2, 3, 4))])
    _actual_step(runtime, 0.75, [(labels, torch.full((2, 3, 4), 2.0))])
    decision = runtime.begin_step(torch.tensor([0.5]))
    call_id, _ = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=(labels[0],), expected_shape=(1, 3, 4)
    )
    runtime.predict(decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ForecastRetryActual):
        runtime.finalize_step(decision["run_id"], decision["step_id"])
    runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], "incomplete branch set")
    for label in labels:
        call_id, actual = runtime.begin_model_call(
            decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=(label,), expected_shape=(1, 3, 4)
        )
        assert actual
        runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, torch.zeros(1, 3, 4))
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime.stats.forecast_fallbacks == 1
    assert runtime.stats.actual_steps == 3


def test_abort_rolls_back_solver_step_id():
    runtime = _runtime(force_actual=True)
    runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    first = runtime.begin_step(torch.tensor([1.0]))
    runtime.abort_step(first["run_id"], first["step_id"])
    repeated = runtime.begin_step(torch.tensor([1.0]))
    assert repeated["step_id"] == first["step_id"] == 0


def test_unsupported_sampler_never_enters_forecast_state():
    runtime = _runtime()
    run_id = runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_heun", supported_sampler=False)
    assert not runtime.supported_sampler
    assert "not allowlisted" in runtime.disabled_reason
    runtime.end_run(run_id)


def test_invalid_sigma_span_uses_a_neutral_coordinate_on_the_native_path():
    runtime = _runtime()
    runtime.start_run(torch.tensor([1.0, 1.0]), "sample_euler", supported_sampler=True)
    decision = runtime.begin_step(torch.tensor([1.0]))
    assert decision["actual"]
    assert decision["coordinate"] == 0.0
    runtime.abort_step(decision["run_id"], decision["step_id"])
    runtime.end_run(decision["run_id"])


def test_history_dtype_change_disables_forecasting_and_keeps_actual_progress():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 4), "sample_euler", supported_sampler=True)
    _actual_step(runtime, 1.0, [(LABEL, torch.ones(1, 3, 4, dtype=torch.float32))])
    _actual_step(runtime, 2.0 / 3.0, [(LABEL, torch.ones(1, 3, 4, dtype=torch.float16))])
    assert runtime.stats.actual_steps == 2
    assert runtime.stats.disabled
    assert "feature dtype changed" in runtime.disabled_reason


def test_adaptive_window_is_capped_by_the_history_bound():
    runtime = _runtime(flex_window=10.0, max_history=4)
    runtime.start_run(torch.linspace(1.0, 0.0, 6), "sample_euler", supported_sampler=True)
    _actual_step(runtime, 1.0, [(LABEL, torch.zeros(1, 3, 4))])
    _actual_step(runtime, 0.8, [(LABEL, torch.ones(1, 3, 4))])
    _forecast_step(runtime, 0.6)
    _actual_step(runtime, 0.4, [(LABEL, torch.full((1, 3, 4), 3.0))])
    assert runtime.stats.current_window == 4.0


@pytest.mark.parametrize(
    ("flex_window", "tail_actual_steps", "expected_actual", "expected_indices"),
    [
        (0.75, 1, 10, [0, 1, 2, 3, 4, 6, 8, 11, 15, 19]),
        (3.0, 1, 8, [0, 1, 2, 3, 4, 6, 11, 19]),
    ],
)
def test_twenty_step_degree_four_schedule_counts(
    flex_window, tail_actual_steps, expected_actual, expected_indices
):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=4,
            warmup_steps=5,
            flex_window=flex_window,
            tail_actual_steps=tail_actual_steps,
            bootstrap_first_forecast=False,
        )
    )
    runtime.start_run(torch.linspace(1.0, 0.0, 21), "sample_euler", supported_sampler=True)
    actual_indices = []
    for step, sigma in enumerate(torch.linspace(1.0, 0.05, 20)):
        decision = runtime.begin_step(sigma)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            actual_indices.append(step)
            runtime.observe_actual(
                decision["run_id"], decision["step_id"], call_id, torch.full((1, 3, 4), float(step))
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
    assert runtime.stats.actual_steps == expected_actual
    assert runtime.stats.forecast_steps == 20 - expected_actual
    assert actual_indices == expected_indices


def test_res_multistep_refreshes_once_without_growing_the_window():
    runtime = _runtime(warmup_steps=2, flex_window=0.75, window_size=4.0)
    runtime.start_run(
        torch.linspace(1.0, 0.0, 8),
        "sample_res_multistep",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    decisions = []
    for step, sigma in enumerate(torch.linspace(1.0, 1.0 / 7.0, 7)):
        decision = runtime.begin_step(sigma)
        decisions.append(decision)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            runtime.observe_actual(
                decision["run_id"],
                decision["step_id"],
                call_id,
                torch.full((1, 3, 4), float(step)),
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

    forecast_indices = [decision["step_id"] for decision in decisions if not decision["actual"]]
    assert forecast_indices == [2, 4, 6]
    assert decisions[3]["reason"] == "post-forecast sampler refresh"
    assert decisions[5]["reason"] == "post-forecast sampler refresh"
    assert runtime.stats.current_window == pytest.approx(4.0)


def test_twenty_step_res_schedule_refreshes_once_and_enforces_three_step_tail():
    runtime = SpectrumH3Runtime(SpectrumH3Config(tail_actual_steps=0))
    runtime.start_run(
        torch.linspace(1.0, 0.0, 21),
        "sample_res_multistep",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        min_tail_actual_steps=3,
    )
    forecast_indices = []
    previous_was_forecast = False
    for step, sigma in enumerate(torch.linspace(1.0, 0.05, 20)):
        decision = runtime.begin_step(sigma)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            runtime.observe_actual(
                decision["run_id"],
                decision["step_id"],
                call_id,
                torch.full((1, 3, 4), float(step)),
            )
        else:
            assert not previous_was_forecast
            forecast_indices.append(step)
            runtime.predict(
                decision["run_id"],
                decision["step_id"],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        previous_was_forecast = not actual
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert forecast_indices == [1, 3, 5, 7, 9, 11, 13, 15]
    assert runtime.stats.actual_steps == 12
    assert runtime.stats.forecast_steps == 8


def test_aborted_res_refresh_does_not_consume_refresh_state():
    runtime = _runtime(warmup_steps=2, window_size=4.0, tail_actual_steps=0)
    run_id = runtime.start_run(
        torch.linspace(1.0, 0.0, 7),
        "sample_res_multistep",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(LABEL, torch.zeros(1, 3, 4))])
    _actual_step(runtime, 5.0 / 6.0, [(LABEL, torch.ones(1, 3, 4))])
    _forecast_step(runtime, 4.0 / 6.0)

    decision = runtime.begin_step(torch.tensor([3.0 / 6.0]))
    assert decision["actual"]
    assert decision["reason"] == "post-forecast sampler refresh"
    runtime.abort_step(run_id, decision["step_id"])

    retried = runtime.begin_step(torch.tensor([3.0 / 6.0]))
    assert retried["step_id"] == decision["step_id"]
    assert retried["actual"]
    assert retried["reason"] == "post-forecast sampler refresh"
    runtime.abort_step(run_id, retried["step_id"])


def test_missing_h3_call_disables_forecasting_and_releases_history():
    runtime = _runtime(
        bootstrap_first_forecast=True,
        warmup_steps=1,
        tail_actual_steps=0,
    )
    runtime.start_run(
        torch.tensor([1.0, 0.5, 0.25, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(LABEL, torch.ones(1, 3, 4))])

    decision = runtime.begin_step(torch.tensor([0.5]))
    assert not decision["actual"]
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert runtime.active_step_id is None
    assert runtime.stats.bypassed_steps == 1
    assert runtime.stats.actual_steps == 1
    assert runtime.stats.forecast_steps == 0
    assert runtime.stats.disabled
    assert "without reaching the native H3 model wrapper" in runtime.disabled_reason
    assert runtime.forecaster.history_length == 0

    summary = runtime.debug_summary()
    assert "bypassed_steps=1" in summary

    next_step = runtime.begin_step(torch.tensor([0.25]))
    assert next_step["actual"]
    assert next_step["reason"] == runtime.disabled_reason
    runtime.abort_step(next_step["run_id"], next_step["step_id"])


def test_twenty_step_euler_schedule_refreshes_between_forecasts():
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    runtime.start_run(
        torch.linspace(1.0, 0.0, 21),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
    )
    forecast_indices = []
    previous_was_forecast = False
    for step, sigma in enumerate(torch.linspace(1.0, 0.05, 20)):
        decision = runtime.begin_step(sigma)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            runtime.observe_actual(
                decision["run_id"],
                decision["step_id"],
                call_id,
                torch.full((1, 3, 4), float(step)),
            )
        else:
            assert not previous_was_forecast
            forecast_indices.append(step)
            runtime.predict(
                decision["run_id"],
                decision["step_id"],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        previous_was_forecast = not actual
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert forecast_indices == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert runtime.stats.actual_steps == 11
    assert runtime.stats.forecast_steps == 9


def _measured_anchor(runtime, timestep, *, video_values, audio_values):
    decision = runtime.begin_step(torch.tensor([timestep]))
    assert decision["actual"]
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    probe = runtime.prepare_residual_probe(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert probe is not None
    actual_feature = probe.shadow + 1.0
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        actual_feature,
    )
    actual_video, shadow_video, hold_video = (
        torch.full((4,), value) for value in video_values
    )
    actual_audio, shadow_audio, hold_audio = (
        torch.full((4,), value) for value in audio_values
    )
    runtime.record_residual_measurement(
        decision["run_id"],
        decision["step_id"],
        call_id,
        probe,
        actual_feature=actual_feature,
        actual_output=[actual_video, actual_audio],
        shadow_output=[shadow_video, shadow_audio],
        hold_output=[hold_video, hold_audio],
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision


def _feedback_runtime():
    runtime = _runtime(
        anchor_residual_feedback=True,
        offline_smoothing_replay=False,
        warmup_steps=2,
        window_size=2.0,
        flex_window=0.75,
    )
    runtime.start_run(
        torch.linspace(1.0, 0.0, 8),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(runtime, 1.0, [(LABEL, torch.zeros(1, 3, 4))])
    _actual_step(runtime, 6.0 / 7.0, [(LABEL, torch.ones(1, 3, 4))])
    _forecast_step(runtime, 5.0 / 7.0)
    runtime._current_window = 4.0
    return runtime


def test_residual_probe_reraises_cuda_oom(monkeypatch):
    runtime = _feedback_runtime()
    decision = runtime.begin_step(torch.tensor([4.0 / 7.0]))
    assert decision["actual"]
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual

    def raise_oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("probe prediction OOM")

    monkeypatch.setattr(runtime.forecaster, "predict_segments", raise_oom)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="probe prediction OOM"):
            runtime.prepare_residual_probe(
                decision["run_id"],
                decision["step_id"],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        assert runtime.experiment_disabled_reason is None
        assert runtime.stats.residual_measure_seconds >= 0.0
    finally:
        runtime.abort_step(decision["run_id"], decision["step_id"])
        runtime.end_run(decision["run_id"])


def test_residual_measurement_rejects_tensor_as_two_stream_container():
    runtime = _feedback_runtime()
    decision = runtime.begin_step(torch.tensor([4.0 / 7.0]))
    assert decision["actual"]
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    probe = runtime.prepare_residual_probe(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert probe is not None
    actual_feature = probe.shadow + 1.0
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        actual_feature,
    )
    runtime.record_residual_measurement(
        decision["run_id"],
        decision["step_id"],
        call_id,
        probe,
        actual_feature=actual_feature,
        actual_output=torch.zeros(2, 4),
        shadow_output=[torch.zeros(4), torch.zeros(4)],
        hold_output=[torch.zeros(4), torch.zeros(4)],
    )
    assert runtime.experiment_disabled_reason is not None
    assert runtime.experiment_disabled_reason == (
        "residual measurement output structure changed"
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    runtime.end_run(decision["run_id"])


def test_anchor_feedback_requests_actual_refresh_without_hidden_correction():
    runtime = _feedback_runtime()
    _measured_anchor(
        runtime,
        4.0 / 7.0,
        video_values=(2.0, 0.5, 1.0),
        audio_values=(2.0, 0.5, 1.0),
    )
    assert runtime.stats.residual_max_score == pytest.approx(1.5)
    assert runtime.stats.residual_policy_max_score == pytest.approx(1.5)
    assert runtime.stats.current_window == pytest.approx(4.0)
    assert not hasattr(runtime, "_pending_residual")

    decision = runtime.begin_step(torch.tensor([3.0 / 7.0]))
    assert decision["actual"]
    assert decision["reason"] == "anchor residual feedback refresh"
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        torch.zeros(1, 3, 4),
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime.stats.feedback_refreshes == 1


def test_audio_only_residual_does_not_change_anchor_feedback_schedule():
    runtime = _feedback_runtime()
    _measured_anchor(
        runtime,
        4.0 / 7.0,
        video_values=(2.0, 1.0, 0.0),
        audio_values=(3.0, 0.0, 2.0),
    )
    assert runtime.stats.residual_max_score == pytest.approx(3.0)
    assert runtime.stats.residual_policy_max_score == pytest.approx(0.5)
    decision = runtime.begin_step(torch.tensor([3.0 / 7.0]))
    assert not decision["actual"]
    assert decision["reason"] == "adaptive forecast"
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert not actual
    expected_audio = runtime.forecaster.predict(
        decision["coordinate"],
        runtime.config.audio_blend_weight,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_video = runtime.forecaster.predict(
        decision["coordinate"],
        runtime.config.blend_weight,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    observed = runtime.predict(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(observed[:, :1], expected_audio[:, :1])
    torch.testing.assert_close(observed[:, 1:], expected_video[:, 1:])
    runtime.finalize_step(decision["run_id"], decision["step_id"])


def test_distinct_modality_blends_fail_closed_without_target_row_metadata():
    runtime = _runtime()
    runtime.start_run(
        torch.tensor([1.0, 0.75, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    topology = (("tiny", 1),)
    for timestep in (1.0, 0.75):
        decision = runtime.begin_step(torch.tensor([timestep]))
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=topology,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert actual
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.zeros(1, 3, 4),
        )
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    decision = runtime.begin_step(torch.tensor([0.5]))
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=topology,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert not actual
    assert runtime.predict(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) is None
    assert runtime.disabled_reason == "packed H3 topology does not expose the target audio/video boundary"
    runtime.abort_step(decision["run_id"], decision["step_id"])
    runtime.end_run(decision["run_id"])


def test_anchor_feedback_skips_probes_after_refresh_budget():
    runtime = _feedback_runtime()
    runtime.stats.feedback_refreshes = 3
    decision = runtime.begin_step(torch.tensor([4.0 / 7.0]))
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    assert runtime.prepare_residual_probe(
        decision["run_id"],
        decision["step_id"],
        call_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) is None
    assert runtime.stats.feedback_suppressed_budget == 1
    runtime.abort_step(decision["run_id"], decision["step_id"])


def test_terminal_feedback_probe_is_skipped_but_rollback_probe_is_retained():
    for setting, expected in (
        ("anchor_residual_feedback", False),
        ("selective_rollback_correction", True),
    ):
        runtime = _runtime(offline_smoothing_replay=False, **{setting: True})
        runtime.start_run(
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
            "sample_euler",
            supported_sampler=True,
            max_consecutive_forecasts=1,
            min_actual_steps_after_forecast=1,
        )
        _actual_step(runtime, 1.0, [(LABEL, torch.zeros(1, 3, 4))])
        _actual_step(runtime, 0.75, [(LABEL, torch.ones(1, 3, 4))])
        _forecast_step(runtime, 0.5)
        decision = runtime.begin_step(torch.tensor([0.25]))
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert actual
        probe = runtime.prepare_residual_probe(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert (probe is not None) is expected
        if setting == "anchor_residual_feedback":
            assert runtime.stats.residual_skipped_terminal_probes == 1
        runtime.abort_step(decision["run_id"], decision["step_id"])


@pytest.mark.parametrize(
    ("video_blend_weight", "audio_blend_weight"),
    ((0.5, 0.0), (1.0, 1.0)),
)
def test_offline_capture_is_local_only_while_replay_keeps_configured_blends(
    video_blend_weight,
    audio_blend_weight,
):
    config = {
        "degree": 2,
        "max_history": 4,
        "warmup_steps": 3,
        "tail_actual_steps": 0,
        "bootstrap_first_forecast": False,
        "blend_weight": video_blend_weight,
        "audio_blend_weight": audio_blend_weight,
    }
    sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    feature_values = (0.0, 1.0, 4.0)

    def prepare_forecast(runtime):
        runtime.start_run(
            sigmas,
            "sample_euler",
            supported_sampler=True,
            max_consecutive_forecasts=1,
            min_actual_steps_after_forecast=1,
        )
        for sigma, value in zip(sigmas[:3], feature_values, strict=True):
            _actual_step(
                runtime,
                float(sigma),
                [(LABEL, torch.full((1, 3, 4), value))],
            )
        decision = runtime.begin_step(sigmas[3])
        assert not decision["actual"]
        local = runtime.forecaster.predict_segments(
            decision["coordinate"],
            ((0, 1, 0.0), (1, 3, 0.0)),
            rows=(0,),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        configured = runtime.forecaster.predict_segments(
            decision["coordinate"],
            (
                (0, 1, audio_blend_weight),
                (1, 3, video_blend_weight),
            ),
            rows=(0,),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert not torch.equal(local, configured)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert not actual
        prediction = runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert prediction is not None
        runtime.finalize_step(decision["run_id"], decision["step_id"])
        return prediction, local, configured

    ordinary = _runtime(**config)
    ordinary_prediction, _ordinary_local, ordinary_configured = prepare_forecast(ordinary)
    torch.testing.assert_close(ordinary_prediction, ordinary_configured)
    assert ordinary.stats.causal_video_blend_weight == video_blend_weight
    assert ordinary.stats.causal_audio_blend_weight == audio_blend_weight
    ordinary.end_run(ordinary.active_run_id)

    offline = _runtime(offline_smoothing_replay=True, **config)
    offline.begin_offline_capture(total_steps=5, sampler_name="sample_euler")
    offline_prediction, offline_local, _offline_spectral = prepare_forecast(offline)
    torch.testing.assert_close(offline_prediction, offline_local)
    assert offline.stats.causal_video_blend_weight == 0.0
    assert offline.stats.causal_audio_blend_weight == 0.0
    summary = offline.debug_summary()
    assert "causal_video_blend_weight=0.000000" in summary
    assert "causal_audio_blend_weight=0.000000" in summary

    _actual_step(
        offline,
        float(sigmas[4]),
        [(LABEL, torch.full((1, 3, 4), 9.0))],
    )
    assert offline.complete_offline_capture()
    assert offline._offline_smoother is not None
    assert offline._offline_smoother.configured_stream_blends == {
        "audio": audio_blend_weight,
        "video": video_blend_weight,
    }
    offline.end_run(offline.active_run_id)
    offline.release_offline_archive()


@pytest.mark.parametrize(
    ("history_storage", "offline_archive_storage"),
    [
        ("system_ram", "system_ram"),
        ("vram", "system_ram"),
        ("system_ram", "vram"),
        ("vram", "vram"),
    ],
)
def test_offline_archive_survives_causal_eviction_and_replay_uses_no_actual_calls(
    history_storage,
    offline_archive_storage,
):
    runtime = _runtime(
        offline_smoothing_replay=True,
        max_history=2,
        warmup_steps=2,
        tail_actual_steps=0,
        history_storage=history_storage,
        offline_archive_storage=offline_archive_storage,
    )
    sigmas = torch.linspace(1.0, 0.0, 7)
    runtime.begin_offline_capture(total_steps=6, sampler_name="sample_euler")
    run_id = runtime.start_run(
        sigmas,
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    first_pass = []
    for step, sigma in enumerate(torch.linspace(1.0, 1.0 / 6.0, 6)):
        decision = runtime.begin_step(sigma)
        first_pass.append(decision)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            runtime.observe_actual(
                decision["run_id"],
                decision["step_id"],
                call_id,
                torch.full((1, 3, 4), float(step)),
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

    assert runtime.forecaster.history_length == 2
    assert runtime.complete_offline_capture()
    archive = runtime.offline_archive
    assert archive is not None
    assert len(archive.anchors) == 4
    assert archive.history_storage == offline_archive_storage
    assert archive.history_device == torch.device("cpu")
    if history_storage == offline_archive_storage:
        history_ptrs = {
            entry.feature_flat.data_ptr() for entry in runtime.forecaster._history
        }
        archive_ptrs = {anchor.feature.data_ptr() for anchor in archive.anchors}
        assert history_ptrs < archive_ptrs
        assert archive.anchors[-1].feature.data_ptr() not in history_ptrs
    assert runtime.stats.offline_validation_samples_per_branch == 12
    assert runtime.stats.offline_validation_anchors == 2
    assert runtime.stats.offline_attenuated_predictions == 2
    assert runtime.stats.offline_effective_blend_min < runtime.config.blend_weight
    assert runtime.stats.offline_effective_audio_blend_min == 0.0
    assert runtime.stats.offline_effective_audio_blend_mean == 0.0
    assert runtime.stats.offline_effective_audio_blend_max == 0.0
    assert runtime.stats.offline_local_only_audio_predictions == 2
    assert runtime.stats.offline_attenuated_video_predictions == 2
    assert 0.0 < runtime.stats.offline_effective_video_blend_min
    assert runtime.stats.offline_effective_video_blend_max < runtime.config.blend_weight
    actual_features = {anchor.step_id: anchor.feature.clone() for anchor in archive.anchors}
    runtime.end_run(run_id)

    runtime.begin_offline_replay()
    replay_id = runtime.start_run(
        sigmas,
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    for step, sigma in enumerate(torch.linspace(1.0, 1.0 / 6.0, 6)):
        decision = runtime.begin_step(sigma)
        assert not decision["actual"]
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        assert not actual
        prediction = runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if step in actual_features:
            torch.testing.assert_close(prediction, actual_features[step])
        runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert runtime.stats.actual_transformer_calls == 0
    assert runtime.stats.offline_replay_steps == 6
    assert runtime.stats.offline_replay_model_calls == 6
    assert runtime.stats.offline_replay_anchor_steps == 4
    assert runtime.stats.offline_replay_smoothed_steps == 2
    assert runtime.prediction_history_length == 4
    assert runtime.prediction_history_tensor_bytes == archive.tensor_bytes
    assert runtime.last_prediction_chunk_count > 0
    summary = runtime.debug_summary()
    assert "offline_replay_anchor_steps=4" in summary
    assert "offline_replay_smoothed_steps=2" in summary
    assert "offline_validation_samples_per_branch=12" in summary
    assert "offline_attenuated_predictions=2" in summary
    assert "offline_effective_blend_mean=" in summary
    assert "video_blend_weight=0.500000" in summary
    assert "audio_blend_weight=0.000000" in summary
    assert "causal_video_blend_weight=0.000000" in summary
    assert "causal_audio_blend_weight=0.000000" in summary
    assert "offline_effective_audio_blend_max=0.000000" in summary
    assert "offline_effective_video_blend_mean=" in summary
    assert "offline_local_only_audio_predictions=2" in summary
    assert "offline_full_schedule_estimated_mib=" in summary
    assert f"offline_archive_storage={offline_archive_storage}" in summary
    assert "offline_archive_device='cpu'" in summary
    assert "history_device='cpu'" in summary
    runtime.end_run(replay_id)
    runtime.release_offline_archive()
    assert runtime.offline_archive is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("history_storage", "offline_archive_storage", "history_device", "archive_device"),
    [
        ("vram", "system_ram", "cuda", "cpu"),
        ("system_ram", "vram", "cpu", "cuda"),
    ],
)
def test_offline_archive_storage_is_independent_on_cuda(
    history_storage,
    offline_archive_storage,
    history_device,
    archive_device,
):
    runtime = _runtime(
        offline_smoothing_replay=True,
        force_actual=True,
        history_storage=history_storage,
        offline_archive_storage=offline_archive_storage,
    )
    runtime.begin_offline_capture(total_steps=2, sampler_name="sample_euler")
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0], device="cuda"),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )
    _actual_step(
        runtime,
        1.0,
        [(LABEL, torch.ones((1, 3, 4), device="cuda"))],
    )
    _actual_step(
        runtime,
        0.5,
        [(LABEL, torch.full((1, 3, 4), 2.0, device="cuda"))],
    )

    archive = runtime.offline_archive
    assert archive is not None
    assert archive.anchors[0].feature.device.type == archive_device
    assert runtime.forecaster.history_device is not None
    assert runtime.forecaster.history_device.type == history_device

    runtime.end_run(run_id)
    runtime.release_offline_archive()
