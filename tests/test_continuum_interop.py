from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    SpectrumH3Binding,
    _continuum_actual_prefix,
    _continuum_prefix_for_phase,
    outer_sample_wrapper,
)


def _model_options(request=None):
    transformer_options = {}
    if request is not None:
        transformer_options["h3_continuum"] = request
    return {"transformer_options": transformer_options}


def _sampler(name="sample_euler"):
    def sampler_function():
        pass

    sampler_function.__name__ = name
    return SimpleNamespace(sampler_function=sampler_function)


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


def _complete_actual_step(runtime, timestep):
    topology = (
        ("video", (1, 24, 2, 4, 4)),
        ("audio", (1, 32, 2, 8)),
        ("hidden", 4),
        ("target_audio_rows", 1),
        ("target_video_rows", 2),
    )
    labels = ((0, "positive"),)
    decision = runtime.begin_step(torch.tensor([timestep]))
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
        torch.full((1, 3, 4), float(decision["step_id"])),
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision


def test_valid_continuum_api_v1_returns_prefix_two():
    assert _continuum_actual_prefix(
        _model_options(
            {"api": 1, "active": True, "min_actual_prefix_steps": 2}
        )
    ) == 2


@pytest.mark.parametrize(
    "metadata",
    (
        {"api": True, "active": True, "min_actual_prefix_steps": 2},
        {"api": 1, "active": 1, "min_actual_prefix_steps": 2},
        {"api": 1, "active": True, "min_actual_prefix_steps": True},
        {"api": 2, "active": True, "min_actual_prefix_steps": 2},
        {"api": 1, "active": True, "min_actual_prefix_steps": -1},
    ),
)
def test_invalid_continuum_metadata_fails_open(metadata):
    assert _continuum_actual_prefix(_model_options(metadata)) == 0


def test_missing_continuum_metadata_preserves_zero_prefix():
    assert _continuum_actual_prefix(None) == 0
    assert _continuum_actual_prefix({}) == 0
    assert _continuum_actual_prefix(_model_options()) == 0


@pytest.mark.parametrize(
    ("phase", "expected"),
    (
        ("single_pass", 2),
        ("single_pass_fallback", 2),
        ("offline_first_pass", 2),
        ("offline_replay", 0),
    ),
)
def test_actual_prefix_phase_contract(phase, expected):
    assert _continuum_prefix_for_phase(2, phase) == expected


def test_runtime_prefix_forces_first_two_steps_before_bootstrap():
    runtime = _runtime()
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.75, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_prefix_steps=2,
    )
    try:
        first = _complete_actual_step(runtime, 1.0)
        second = _complete_actual_step(runtime, 0.75)
        assert first["reason"] == "H3 Continuum actual prefix"
        assert second["reason"] == "H3 Continuum actual prefix"
    finally:
        runtime.end_run(run_id)


def test_runtime_clamps_prefix_to_solver_step_count():
    runtime = _runtime()
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
        min_actual_prefix_steps=99,
    )
    try:
        assert runtime._run is not None
        assert runtime._run.min_actual_prefix_steps == 2
    finally:
        runtime.end_run(run_id)


@pytest.mark.parametrize("value", (True, -1, 1.5, None))
def test_runtime_rejects_invalid_internal_prefix(value):
    runtime = _runtime()
    with pytest.raises(ValueError, match="min_actual_prefix_steps"):
        runtime.start_run(
            torch.tensor([1.0, 0.0]),
            "sample_euler",
            supported_sampler=True,
            min_actual_prefix_steps=value,
        )


def test_outer_sample_accepts_continuum_prefix_once(caplog):
    runtime = _runtime()
    guider = SimpleNamespace(
        model_options={
            BINDING_KEY: SpectrumH3Binding(runtime),
            "transformer_options": {
                "h3_continuum": {
                    "api": 1,
                    "active": True,
                    "min_actual_prefix_steps": 2,
                }
            },
        }
    )
    observed = []

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            assert runtime._run is not None
            observed.append(runtime._run.min_actual_prefix_steps)
            return "result"

    with caplog.at_level("WARNING"):
        result = outer_sample_wrapper(
            Executor(),
            torch.ones(1),
            torch.zeros(1),
            _sampler(),
            torch.tensor([1.0, 0.5, 0.0]),
            seed=7,
        )

    assert result == "result"
    assert observed == [2]
    assert caplog.text.count(
        "Spectrum H3: accepted H3 Continuum API v1, actual prefix=2"
    ) == 1


def _invoke_outer(runtime, request, observed):
    model_options = {BINDING_KEY: SpectrumH3Binding(runtime)}
    model_options.update(_model_options(request))
    guider = SimpleNamespace(model_options=model_options)

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            assert runtime._run is not None
            observed.append(runtime._run.min_actual_prefix_steps)
            return "result"

    result = outer_sample_wrapper(
        Executor(),
        torch.ones(1),
        torch.zeros(1),
        _sampler(),
        torch.tensor([1.0, 0.5, 0.0]),
        seed=7,
    )
    assert result == "result"
    assert runtime._run is None


@pytest.mark.parametrize(("chunk_count", "expected_logs"), ((3, 2), (12, 11)))
def test_continuation_chunks_log_once_each(caplog, chunk_count, expected_logs):
    request = {"api": 1, "active": True, "min_actual_prefix_steps": 2}
    observed = []

    with caplog.at_level("WARNING"):
        _invoke_outer(_runtime(), None, observed)
        for _ in range(1, chunk_count):
            _invoke_outer(_runtime(), request, observed)

    assert observed == [0] + [2] * expected_logs
    assert caplog.text.count(
        "Spectrum H3: accepted H3 Continuum API v1, actual prefix=2"
    ) == expected_logs


def test_prefix_does_not_leak_into_next_normal_run(caplog):
    request = {"api": 1, "active": True, "min_actual_prefix_steps": 2}
    observed = []
    runtime = _runtime()

    with caplog.at_level("WARNING"):
        _invoke_outer(runtime, request, observed)
        _invoke_outer(runtime, None, observed)

    assert observed == [2, 0]
    assert runtime._run is None
    assert caplog.text.count(
        "Spectrum H3: accepted H3 Continuum API v1, actual prefix=2"
    ) == 1
