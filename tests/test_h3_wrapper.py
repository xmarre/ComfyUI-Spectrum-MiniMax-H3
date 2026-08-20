from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.minimax_h3 import (
    _sanitize_prediction,
    diffusion_model_wrapper,
    is_native_minimax_h3,
    locate_minimax_h3_inner,
    require_native_minimax_h3,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    RUN_ID_KEY,
    RUNTIME_KEY,
    STEP_ID_KEY,
    SpectrumH3Binding,
    model_clone_callback,
)


def _native_shaped_fake(*, use_adaln_curves=False):
    cls = type("MiniMaxH3Model", (), {})
    cls.__module__ = "comfy.ldm.minimax.model"
    instance = cls()
    for name, value in {
        "blocks": [object()],
        "final_layer": object(),
        "hidden_size": 8,
        "patch_size": (1, 2, 2),
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
        "use_adaln_curves": use_adaln_curves,
    }.items():
        setattr(instance, name, value)
    if use_adaln_curves:
        instance.adaln_t_table = object()
    else:
        instance.time_embedder = object()
    return instance


def test_model_detection_accepts_only_native_h3_identity_and_shape():
    inner = _native_shaped_fake()
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=inner))
    assert locate_minimax_h3_inner(patcher) == (inner, "model.diffusion_model")
    assert is_native_minimax_h3(inner)
    assert require_native_minimax_h3(patcher)[0] is inner
    with pytest.raises(TypeError, match="requires ComfyUI's native"):
        require_native_minimax_h3(SimpleNamespace(model=SimpleNamespace(diffusion_model=torch.nn.Linear(2, 2))))


def test_model_detection_enforces_the_conditional_timestep_contract():
    assert is_native_minimax_h3(_native_shaped_fake(use_adaln_curves=False))
    assert is_native_minimax_h3(_native_shaped_fake(use_adaln_curves=True))

    missing_embedder = _native_shaped_fake(use_adaln_curves=False)
    del missing_embedder.time_embedder
    assert not is_native_minimax_h3(missing_embedder)

    missing_table = _native_shaped_fake(use_adaln_curves=True)
    del missing_table.adaln_t_table
    assert not is_native_minimax_h3(missing_table)


def test_non_native_diffusion_call_records_a_native_passthrough_fallback():
    runtime = SpectrumH3Runtime(SpectrumH3Config(degree=1, max_history=4))
    run_id = runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    decision = runtime.begin_step(torch.tensor([1.0]))

    class Executor:
        class_obj = object()

        def __call__(self, *args, **kwargs):
            return "native-output"

    options = {
        RUNTIME_KEY: runtime,
        RUN_ID_KEY: decision["run_id"],
        STEP_ID_KEY: decision["step_id"],
    }
    result = diffusion_model_wrapper(Executor(), object(), torch.tensor([1000.0]), object(), options)
    runtime.finalize_step(decision["run_id"], decision["step_id"])

    assert result == "native-output"
    assert runtime.stats.actual_steps == 1
    assert "not ComfyUI's native MiniMax H3" in runtime.disabled_reason
    runtime.end_run(run_id)


def test_forecast_sanitization_clamps_and_replaces_nonfinite_values():
    source = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e20, 2.0])
    sanitized, event = _sanitize_prediction(source, torch.float16)
    assert event is not None
    assert torch.isfinite(sanitized).all()
    assert sanitized.dtype == torch.float16
    all_bad, event = _sanitize_prediction(torch.tensor([float("nan"), float("inf")]), torch.float16)
    assert all_bad is None
    assert "no finite" in event["reason"]


def test_model_clone_callback_provisions_an_isolated_runtime():
    source_runtime = SpectrumH3Runtime(SpectrumH3Config(degree=1, max_history=4))
    source = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(source_runtime)})
    clone = SimpleNamespace(model_options={BINDING_KEY: source.model_options[BINDING_KEY]})
    model_clone_callback(source, clone)
    clone_runtime = clone.model_options[BINDING_KEY].runtime
    assert clone_runtime is not source_runtime
    assert clone_runtime.config == source_runtime.config
