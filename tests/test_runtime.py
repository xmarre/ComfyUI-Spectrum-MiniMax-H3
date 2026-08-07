from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import ForecastRetryActual, SpectrumH3Runtime

TOPOLOGY = (("video", (1, 24, 2, 4, 4)), ("audio", (1, 32, 2, 8)), ("hidden", 4))
LABEL = ((0, "positive"),)


def _runtime(**overrides):
    values = {
        "degree": 1,
        "max_history": 4,
        "warmup_steps": 2,
        "tail_actual_steps": 0,
        "window_size": 2.0,
        "bootstrap_first_forecast": False,
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


def test_preliminary_runtime_defaults():
    config = SpectrumH3Config()

    assert config.degree == 1
    assert config.warmup_steps == 1
    assert config.tail_actual_steps == 1
    assert config.history_storage == "system_ram"
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
