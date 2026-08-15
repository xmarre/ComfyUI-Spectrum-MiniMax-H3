from __future__ import annotations

import torch

from comfyui_spectrum_h3.er_sde_stochastic import (
    ERSDEStepDescriptor,
    ERSDEStochasticTracker,
)


def _descriptor(step_id: int, *, mode: str = "forecast", replay_source_actual=None):
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


def _tracker(*, debug: bool = False):
    return ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: torch.tensor([[0.75, -0.5]]),
        noise_scaler=lambda value: value**2,
        effective_s_noise=1.0,
        max_stage=3,
        debug=debug,
        run_id=7,
    )


def _produce_pending(
    tracker: ERSDEStochasticTracker,
    *,
    lambda_source: float,
    lambda_target: float,
    sigma_next: float,
):
    # Native sample_er_sde calls noise_scaler(er_lambda_t) first, then er_lambda_s.
    er_lambda_t = torch.tensor(lambda_target)
    er_lambda_s = torch.tensor(lambda_source)
    tracker.noise_scaler(er_lambda_t)
    tracker.noise_scaler(er_lambda_s)
    noise = tracker.noise_sampler(torch.tensor(0.9), torch.tensor(sigma_next))
    r = er_lambda_t**2 / er_lambda_s**2
    return (
        (torch.tensor(sigma_next) / er_lambda_t)
        * noise
        * (er_lambda_t**2 - er_lambda_s**2 * r**2).sqrt().nan_to_num(nan=0.0)
    )


def test_bootstrap_forecast_uses_latest_actual_solver_space_denoised():
    tracker = _tracker()
    actual0 = torch.tensor([[1.25, -0.75]])
    returned0 = tracker.consume(actual0, _descriptor(0, mode="actual"))
    assert returned0 is actual0

    q1 = _produce_pending(
        tracker,
        lambda_source=10.0,
        lambda_target=4.0,
        sigma_next=0.8,
    )
    raw_forecast = torch.tensor([[100.0, -100.0]]) + q1
    returned1 = tracker.consume(raw_forecast, _descriptor(1))

    torch.testing.assert_close(returned1, actual0, rtol=0, atol=0)
    assert not tracker.has_pending
    assert tracker.denoised_anchor_steps == (0,)


def test_two_anchor_forecast_extrapolates_denoised_in_native_er_lambda_space():
    tracker = _tracker()
    actual0 = torch.tensor([[1.0, 2.0]])
    tracker.consume(actual0, _descriptor(0, mode="actual"))

    _produce_pending(
        tracker,
        lambda_source=10.0,
        lambda_target=7.0,
        sigma_next=0.8,
    )
    # Bootstrap forecast: exact latest-actual hold.
    tracker.consume(torch.tensor([[90.0, -90.0]]), _descriptor(1))

    _produce_pending(
        tracker,
        lambda_source=7.0,
        lambda_target=4.0,
        sigma_next=0.6,
    )
    actual2 = torch.tensor([[3.0, 4.0]])
    tracker.consume(actual2, _descriptor(2, mode="actual"))

    q3 = _produce_pending(
        tracker,
        lambda_source=4.0,
        lambda_target=2.5,
        sigma_next=0.4,
    )
    raw_forecast = torch.tensor([[80.0, -80.0]]) + q3
    returned3 = tracker.consume(raw_forecast, _descriptor(3))

    alpha = (2.5 - 4.0) / (4.0 - 10.0)
    expected = actual2 + alpha * (actual2 - actual0)
    torch.testing.assert_close(returned3, expected, rtol=0, atol=1e-6)
    assert not tracker.has_pending
    assert tracker.denoised_anchor_steps == (0, 2)


def test_extrapolation_guard_uses_latest_actual_hold_instead_of_noisy_raw_forecast():
    tracker = _tracker()
    actual0 = torch.tensor([[1.0, 2.0]])
    tracker.consume(actual0, _descriptor(0, mode="actual"))
    _produce_pending(
        tracker,
        lambda_source=10.0,
        lambda_target=9.5,
        sigma_next=0.8,
    )
    tracker.consume(torch.tensor([[90.0, -90.0]]), _descriptor(1))

    _produce_pending(
        tracker,
        lambda_source=9.5,
        lambda_target=9.0,
        sigma_next=0.6,
    )
    actual2 = torch.tensor([[5.0, 6.0]])
    tracker.consume(actual2, _descriptor(2, mode="actual"))

    q3 = _produce_pending(
        tracker,
        lambda_source=9.0,
        lambda_target=1.0,
        sigma_next=0.4,
    )
    raw_forecast = torch.tensor([[1000.0, -1000.0]]) + q3
    returned3 = tracker.consume(raw_forecast, _descriptor(3))

    torch.testing.assert_close(returned3, actual2, rtol=0, atol=0)
    assert not tracker.has_pending


def test_consecutive_warmup_anchors_preserve_existing_exact_q_path_for_first_forecast():
    tracker = _tracker()
    tracker.consume(torch.tensor([[1.0, 2.0]]), _descriptor(0, mode="actual"))
    _produce_pending(
        tracker,
        lambda_source=10.0,
        lambda_target=7.0,
        sigma_next=0.8,
    )
    tracker.consume(torch.tensor([[2.0, 3.0]]), _descriptor(1, mode="actual"))

    q2 = _produce_pending(
        tracker,
        lambda_source=7.0,
        lambda_target=4.0,
        sigma_next=0.6,
    )
    base = torch.tensor([[4.0, 5.0]])
    returned2 = tracker.consume(base + q2, _descriptor(2))

    torch.testing.assert_close(returned2, base, rtol=0, atol=1e-6)
    assert tracker.denoised_anchor_steps == (0, 1)


def test_replay_keeps_exact_q_compensation_and_does_not_use_causal_dense_output():
    tracker = _tracker()
    tracker.consume(torch.tensor([[1.0, 2.0]]), _descriptor(0, mode="actual"))
    q1 = _produce_pending(
        tracker,
        lambda_source=10.0,
        lambda_target=4.0,
        sigma_next=0.8,
    )
    base = torch.tensor([[7.0, 8.0]])
    returned = tracker.consume(
        base + q1,
        _descriptor(1, mode="replay", replay_source_actual=False),
    )

    torch.testing.assert_close(returned, base, rtol=0, atol=1e-6)


def test_dense_anchor_storage_is_bounded_and_clear_releases_it():
    tracker = _tracker()
    tracker.consume(torch.tensor([[0.0, 0.0]]), _descriptor(0, mode="actual"))
    for step_id, (source, target) in enumerate(
        ((10.0, 7.0), (7.0, 4.0), (4.0, 2.0)),
        start=1,
    ):
        _produce_pending(
            tracker,
            lambda_source=source,
            lambda_target=target,
            sigma_next=max(0.1, 0.9 - 0.2 * step_id),
        )
        tracker.consume(
            torch.full((1, 2), float(step_id)),
            _descriptor(step_id, mode="actual"),
        )

    assert tracker.denoised_anchor_steps == (2, 3)
    tracker.clear()
    assert tracker.denoised_anchor_steps == ()
    assert not tracker.has_pending
