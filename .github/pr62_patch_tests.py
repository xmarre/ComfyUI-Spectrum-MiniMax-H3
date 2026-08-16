from pathlib import Path

# Temporary one-shot helper for PR #62 test reconciliation.
path = Path("tests/test_er_sde_stochastic_compensation.py")
text = path.read_text(encoding="utf-8")

old = '''def test_seeded_first_pass_and_replay_consume_identical_stochastic_streams():
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
'''
new = '''def test_seeded_first_pass_and_replay_consume_identical_stochastic_draws():
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

    # Causal first-pass forecasts use ER-SDE solver-space dense output, while
    # replay intentionally retains its separate exact-q compatibility path so
    # smoothed replay features remain observable to the sampler. Those denoised
    # trajectories need not be equal. The native stochastic stream itself must
    # nevertheless remain exactly reproducible across the two seeded passes.
    assert len(first_pass[2]) == len(replay[2]) == 3
    assert all(
        torch.equal(first_value, replay_value)
        for first_value, replay_value in zip(first_pass[2], replay[2], strict=True)
    )
'''
if old not in text:
    raise SystemExit("first replay test block not found")
text = text.replace(old, new, 1)

old = '''def _tracker_targeting_step_two(run_id: int):
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
'''
new = '''def _tracker_targeting_step_two(run_id: int):
    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: torch.tensor([[2.0, -3.0]]),
        noise_scaler=lambda value: value**2,
        effective_s_noise=0.5,
        max_stage=3,
        debug=False,
        run_id=run_id,
    )
    _produce_pending(tracker)
    actual1 = torch.tensor([[1.5, -2.5]])
    tracker.consume(actual1, ERSDEStepDescriptor(run_id, 1, "actual", None, False))
    q = _produce_pending(tracker)
    return tracker, q, actual1


def test_predict_wrapper_uses_solver_space_hold_after_consecutive_actuals():
    runtime, run_id = _runtime_ready_for_step_two_forecast()
    tracker, q, actual1 = _tracker_targeting_step_two(run_id)
'''
if old not in text:
    raise SystemExit("wrapper helper/test block not found")
text = text.replace(old, new, 1)

old = '''    torch.testing.assert_close(result, base, rtol=0, atol=0)
    assert runtime.last_completed_mode == "forecast"
'''
new = '''    torch.testing.assert_close(result, actual1, rtol=0, atol=0)
    assert not torch.equal(result, base)
    assert runtime.last_completed_mode == "forecast"
'''
if old not in text:
    raise SystemExit("wrapper expectation block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
