from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from comfyui_spectrum_h3 import res_investigation as investigation
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import ForecastRetryActual, SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    ACTUAL_KEY, BINDING_KEY, RUN_ID_KEY, STEP_ID_KEY, SpectrumH3Binding,
    predict_noise_wrapper,
)
from test_res_multistep_trajectory_recovery import _native_res_multistep


@pytest.mark.parametrize("sigmas", [
    torch.linspace(1, 0, 26, dtype=torch.float64),
    torch.tensor([1.0, .8, .51, .17, .02, 0], dtype=torch.float32),
    torch.tensor([1.0, .7, .5, .2], dtype=torch.float64),
])
def test_coefficients_match_installed_native_updates(sigmas):
    native = _native_res_multistep()
    coefficients = investigation.res_coefficients(sigmas)
    inputs, denoised = [], []
    model_sampling = SimpleNamespace(noise_scale=1)
    def model(x, sigma, **kwargs):
        inputs.append(x.clone())
        y = .3 * x + .1 * x.square()
        denoised.append(y)
        return y
    model.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
        get_model_object=lambda name: model_sampling))
    output = native(model, torch.ones(1, dtype=sigmas.dtype), sigmas, eta=0, cfg_pp=False,
                    s_noise=0, disable=True, noise_sampler=lambda a, b: torch.zeros(1))
    for i, row in enumerate(coefficients):
        expected = inputs[i] * row["state_coefficient"] + denoised[i] * row["current_denoised_coefficient"]
        if i:
            expected += denoised[i - 1] * row["previous_denoised_coefficient"]
        target = inputs[i + 1] if i + 1 < len(inputs) else output
        torch.testing.assert_close(target, expected)


@pytest.mark.parametrize("sigmas", [
    [1, .5, .5, 0], [1, 0, 0], [1, float("nan"), 0], [0, -1], [1, 2, 0],
])
def test_invalid_schedule_is_explicit(sigmas):
    with pytest.raises(ValueError, match="RES investigation"):
        investigation.res_coefficients(torch.tensor(sigmas, dtype=torch.float32))


def _runtime():
    runtime = SpectrumH3Runtime(SpectrumH3Config(
        offline_smoothing_replay=False, flex_window=0, window_size=2, debug=False))
    run_id = runtime.start_run(torch.tensor([1., .7, .4, 0]), "sample_res_multistep",
                               supported_sampler=True, min_actual_steps_after_forecast=1)
    return runtime, run_id


def _probe(runtime, mode="replace_selected", phase="single_pass"):
    # Construction provenance is tested with the native sampler separately.
    probe = object.__new__(investigation.RESInvestigation)
    probe.settings = {"mode": mode, "selected_step": 1, "label": "test"}
    probe.phase = phase
    probe.latent_shapes = [(1, 1, 1, 1, 2), (1, 1, 1, 2)]
    probe.coefficients = investigation.res_coefficients(torch.tensor([1., .7, .4, 0]))
    probe.checkpoints = {}
    probe.rows = []
    probe.started = 0
    probe.replacements = 0
    probe.selected_status = "not_reached"
    probe.pending_error = None
    probe.report = {"steps": probe.rows}
    runtime._res_investigation = probe
    return probe


class _Executor:
    def __init__(self, runtime, fail_forecast=False, fail_actual=False):
        self.class_obj = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(runtime)})
        self.runtime = runtime
        self.fail_forecast = fail_forecast
        self.fail_actual = fail_actual
        self.modes = []

    def __call__(self, x, timestep, options, seed):
        t = options["transformer_options"]
        call_id, actual = self.runtime.begin_model_call(
            t[RUN_ID_KEY], t[STEP_ID_KEY], topology=("fixed",), labels=("positive",),
            expected_shape=(1, 4, 1))
        self.modes.append(actual)
        if actual:
            if self.fail_actual and t[STEP_ID_KEY] == 1:
                raise RuntimeError("actual failed")
            feature = (0.3 * x.reshape(1, 4, 1) + .1 * x.reshape(1, 4, 1).square())
            self.runtime.observe_actual(t[RUN_ID_KEY], t[STEP_ID_KEY], call_id, feature)
            return feature.reshape_as(x)
        if self.fail_forecast:
            raise ForecastRetryActual("synthetic retry")
        predicted = self.runtime.predict(t[RUN_ID_KEY], t[STEP_ID_KEY], call_id,
                                         device=x.device, dtype=x.dtype)
        assert not t[ACTUAL_KEY]
        return predicted.reshape_as(x)


@pytest.mark.parametrize("phase", ["single_pass", "offline_first_pass"])
def test_selected_replacement_commits_only_exact_history_and_counts_once(phase):
    runtime, run_id = _runtime()
    probe = _probe(runtime, phase=phase)
    executor = _Executor(runtime)
    x = torch.ones(1, 1, 4)
    predict_noise_wrapper(executor, x, torch.tensor([1.]), {}, 7)
    result = predict_noise_wrapper(executor, x + 1, torch.tensor([.7]), {}, 7)
    expected = .3 * (x + 1) + .1 * (x + 1).square()
    torch.testing.assert_close(result, expected)
    assert executor.modes == [True, False, True]
    assert runtime.stats.actual_steps == 2 and runtime.stats.forecast_steps == 0
    assert runtime.last_completed_mode == "actual"
    assert runtime.prediction_history_length == 2
    assert probe.replacements == 1
    assert probe.selected_status == "replaced_with_actual"
    assert set(probe.rows[1]["same_state_forecast_error"]) == {"video", "audio"}
    assert probe.rows[1]["same_state_forecast_error"]["video"]["rmse"] > 0
    # The final actual refresh must see an actual anchor, never the forecast.
    predict_noise_wrapper(executor, x, torch.tensor([.4]), {}, 7)
    assert len(probe.rows) == 3
    runtime.end_run(run_id)


def test_trace_has_no_extra_executor_calls_or_output_change():
    def run(probed):
        runtime, run_id = _runtime()
        probe = _probe(runtime, mode="trace") if probed else None
        executor = _Executor(runtime)
        outputs = [predict_noise_wrapper(executor, torch.ones(1, 1, 4), torch.tensor([s]), {}, 7)
                   for s in [1., .7, .4]]
        summary = runtime.debug_summary()
        runtime.end_run(run_id)
        return outputs, executor.modes, summary, probe
    plain, modes, summary, _ = run(False)
    traced, traced_modes, traced_summary, probe = run(True)
    for a, b in zip(plain, traced, strict=True):
        assert torch.equal(a, b)
    assert modes == traced_modes == [True, False, True]
    assert summary == traced_summary
    assert probe.replacements == 0


def test_failed_forecast_uses_existing_retry_without_unbound_hook():
    runtime, run_id = _runtime()
    executor = _Executor(runtime, fail_forecast=True)
    for s in [1., .7]:
        predict_noise_wrapper(executor, torch.ones(1, 1, 4), torch.tensor([s]), {}, 7)
    assert executor.modes == [True, False, True]
    runtime.end_run(run_id)


def test_failed_intervention_aborts_active_step():
    runtime, run_id = _runtime()
    _probe(runtime)
    executor = _Executor(runtime, fail_actual=True)
    predict_noise_wrapper(executor, torch.ones(1, 1, 4), torch.tensor([1.]), {}, 7)
    with pytest.raises(RuntimeError, match="actual failed"):
        predict_noise_wrapper(executor, torch.ones(1, 1, 4), torch.tensor([.7]), {}, 7)
    assert runtime.active_step_id is None
    runtime.end_run(run_id)


def test_checkpoint_roundtrip_and_failure_cleanup(tmp_path, monkeypatch):
    paths = ModuleType("folder_paths")
    paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", paths)
    runtime, run_id = _runtime()
    probe = _probe(runtime)
    executor = _Executor(runtime)
    for s in [1., .7]:
        predict_noise_wrapper(executor, torch.ones(1, 1, 4), torch.tensor([s]), {}, 7)
    investigation.finish(runtime, torch.ones(1, 1, 4), True)
    reports = list(tmp_path.rglob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["completed"] and report["replacement_count"] == 1
    tensors = load_file(str(reports[0].with_name(report["tensor_file"])))
    assert {"selected_forecast", "selected_exact", "final_audio", "final_video"} <= tensors.keys()
    assert not probe.checkpoints and runtime._res_investigation is None
    # A failed report write also releases retained CPU tensors.
    probe = _probe(runtime)
    probe.checkpoints["x"] = torch.ones(1)
    paths.get_output_directory = lambda: (_ for _ in ()).throw(OSError("no disk"))
    investigation.finish(runtime, None, False)
    assert not probe.checkpoints and runtime._res_investigation is None
    runtime.end_run(run_id)


def test_native_all_actual_trace_is_bitwise_equal_to_untouched_native_res():
    native = _native_res_multistep()
    sigmas = torch.tensor([1., .7, .4, 0])
    initial = torch.ones(1, 1, 4)
    runtime = SpectrumH3Runtime(SpectrumH3Config(force_actual=True, offline_smoothing_replay=False))
    run_id = runtime.start_run(sigmas, "sample_res_multistep", supported_sampler=True)
    _probe(runtime, mode="trace")
    executor = _Executor(runtime)
    model_sampling = SimpleNamespace(noise_scale=1)
    inner = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name: model_sampling))
    def wrapped(x, sigma, **kwargs):
        return predict_noise_wrapper(executor, x, sigma, {}, 7)
    wrapped.inner_model = inner
    def untouched(x, sigma, **kwargs):
        return .3 * x + .1 * x.square()
    untouched.inner_model = inner
    kwargs = {"eta": 0, "cfg_pp": False, "s_noise": 0, "disable": True,
              "noise_sampler": lambda a, b: torch.zeros_like(initial)}
    baseline = native(untouched, initial.clone(), sigmas, **kwargs)
    traced = native(wrapped, initial.clone(), sigmas, **kwargs)
    assert torch.equal(baseline, traced)
    runtime.end_run(run_id)
