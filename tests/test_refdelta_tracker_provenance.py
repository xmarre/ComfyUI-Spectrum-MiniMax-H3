from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.er_sde_stochastic import (
    ERSDEStepDescriptor,
    ERSDEStochasticTracker,
)
from comfyui_spectrum_h3.refdelta_interop import (
    RefDeltaInteropBridge,
    RefDeltaInteropError,
)


def _descriptor(
    step_id: int,
    *,
    mode: str = "actual",
    replay_source_actual: bool | None = None,
) -> ERSDEStepDescriptor:
    return ERSDEStepDescriptor(
        run_id=41,
        step_id=step_id,
        mode=mode,
        replay_source_actual=replay_source_actual,
        requires_compensation=(
            mode == "forecast"
            or (mode == "replay" and replay_source_actual is False)
        ),
    )


def _tracker() -> ERSDEStochasticTracker:
    return ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: torch.ones((1, 2)),
        noise_scaler=lambda value: value**2,
        effective_s_noise=1.0,
        max_stage=3,
        debug=False,
        run_id=41,
        external_increment=True,
    )


def test_tracker_consumption_is_the_refdelta_classification_source():
    tracker = _tracker()
    bridge = RefDeltaInteropBridge(run_id=41, tracker=tracker)
    denoised = torch.tensor([[1.0, -2.0]])

    # Reproduce the real ComfyUI failure: the stochastic tracker consumes Spectrum's
    # descriptor, but no separate post-model bridge.note_model_result() call occurs.
    result = tracker.consume(denoised, _descriptor(0, mode="actual"))

    assert result is denoised
    assert bridge.model_result_is_actual(0)


def test_tracker_driven_provenance_still_rejects_stale_step_queries():
    tracker = _tracker()
    bridge = RefDeltaInteropBridge(run_id=41, tracker=tracker)
    tracker.consume(torch.zeros((1, 2)), _descriptor(0, mode="actual"))

    with pytest.raises(
        RefDeltaInteropError,
        match=r"requested=1, observed=0",
    ):
        bridge.model_result_is_actual(1)


def test_tracker_driven_provenance_validates_external_increment_source():
    tracker = _tracker()
    bridge = RefDeltaInteropBridge(run_id=41, tracker=tracker)
    tracker.consume(torch.zeros((1, 2)), _descriptor(0, mode="actual"))

    tracker.noise_scaler(torch.tensor(0.5))
    tracker.noise_scaler(torch.tensor(1.0))
    tracker.noise_sampler(torch.tensor(0.8), torch.tensor(0.4))
    increment = torch.tensor([[0.25, -0.5]])

    bridge.publish_stochastic_increment(0, increment)

    assert tracker.pending_step_id == 1


def test_replay_source_provenance_is_bound_to_successful_tracker_consume():
    tracker = _tracker()
    bridge = RefDeltaInteropBridge(run_id=41, tracker=tracker)
    descriptor = _descriptor(0, mode="replay", replay_source_actual=False)

    tracker.consume(torch.zeros((1, 2)), descriptor)

    assert not bridge.model_result_is_actual(0)
    assert bridge.is_replay_step


def test_deterministic_bridge_keeps_direct_descriptor_path():
    bridge = RefDeltaInteropBridge(run_id=41, tracker=None)
    bridge.note_model_result(_descriptor(0, mode="forecast"))

    assert not bridge.model_result_is_actual(0)
