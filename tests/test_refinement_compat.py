from __future__ import annotations

import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.refinement_compat import _refinement_request
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import _continuum_actual_prefix


def _options(*, refinement=None, continuum=None):
    transformer_options = {}
    if continuum is not None:
        transformer_options["h3_continuum"] = continuum
    if refinement is not None:
        transformer_options["h3_refinement"] = refinement
    return {"transformer_options": transformer_options}


def _continuum(prefix=2):
    return {"api": 1, "active": True, "min_actual_prefix_steps": prefix}


def _refinement(prefix=0, sigma_reference=1.0):
    return {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": prefix,
        "sigma_reference": sigma_reference,
    }


def _runtime():
    return SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=1,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=True,
            offline_smoothing_replay=False,
        )
    )


def _complete_first_actual(runtime):
    topology = (
        ("video", (1, 24, 2, 4, 4)),
        ("audio", (1, 32, 2, 8)),
        ("hidden", 4),
        ("target_audio_rows", 1),
        ("target_video_rows", 2),
    )
    labels = ((0, "positive"),)
    decision = runtime.begin_step(torch.tensor([0.72]))
    assert decision["actual"]
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=topology,
        labels=labels,
        expected_shape=(1, 3, 4),
    )
    assert actual
    runtime.observe_actual(
        decision["run_id"],
        decision["step_id"],
        call_id,
        torch.ones((1, 3, 4)),
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision


def test_valid_refinement_contract_is_strictly_parsed():
    request = _refinement_request(_options(refinement=_refinement()))
    assert request == _refinement()


def test_refinement_contract_overrides_inherited_continuum_prefix():
    options = _options(refinement=_refinement(prefix=0), continuum=_continuum(prefix=2))
    assert _continuum_actual_prefix(options) == 0


def test_invalid_refinement_contract_does_not_suppress_continuum_safety_prefix():
    options = _options(
        refinement=_refinement(prefix=0, sigma_reference=float("nan")),
        continuum=_continuum(prefix=2),
    )
    assert _refinement_request(options) is None
    assert _continuum_actual_prefix(options) == 2


def test_normal_continuum_generation_keeps_prefix_two():
    assert _continuum_actual_prefix(_options(continuum=_continuum(prefix=2))) == 2


def test_three_step_refinement_middle_step_is_forecast_eligible_after_warmup():
    prefix = _continuum_actual_prefix(
        _options(refinement=_refinement(prefix=0), continuum=_continuum(prefix=2))
    )
    runtime = _runtime()
    run_id = runtime.start_run(
        torch.tensor([0.72, 0.55, 0.26, 0.0]),
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        min_actual_prefix_steps=prefix,
    )
    try:
        first = _complete_first_actual(runtime)
        middle = runtime.begin_step(torch.tensor([0.55]))
        assert first["reason"] == "warmup"
        assert middle["actual"] is False
        assert "forecast" in middle["reason"]
    finally:
        runtime.end_run(run_id)
