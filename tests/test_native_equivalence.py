from __future__ import annotations

import copy

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.forecast import HistoryWeightForecaster
from comfyui_spectrum_h3.minimax_h3 import diffusion_model_wrapper
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    ACTUAL_KEY,
    COORDINATE_KEY,
    RUN_ID_KEY,
    RUNTIME_KEY,
    STEP_ID_KEY,
)


def _native_imports():
    try:
        import comfy.cli_args

        comfy.cli_args.args.cpu = True
        import comfy.patcher_extension
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    except Exception as exc:  # noqa: BLE001 - any unavailable/broken external ComfyUI import skips this fixture
        pytest.skip(f"current ComfyUI source is unavailable: {exc}")
    return comfy.patcher_extension, MiniMaxH3Model, PackedLayout


class _CountingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        self.calls += 1
        return x + t_emb[0].mean().to(x.dtype) * 0.01


def _tiny_model():
    _, MiniMaxH3Model, _ = _native_imports()
    torch.manual_seed(3)
    model = MiniMaxH3Model(
        hidden_size=8,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=8,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=4,
        rope_inv_freq_len=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=torch.nn,
    )
    block = _CountingBlock()
    model.blocks[0] = block
    return model, block


def _inputs(PackedLayout):
    video = torch.randn(1, 2, 1, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 8)
    payload = {"layout": PackedLayout(2, 1, 4, 4, 3), "seed": 5}
    return [video, audio], context, payload


def _wrapped_call(model, runtime, sigma, model_timestep, x, context, payload, label="positive"):
    patcher_extension, _, _ = _native_imports()
    decision = runtime.begin_step(torch.tensor([sigma]))
    options = {
        RUNTIME_KEY: runtime,
        RUN_ID_KEY: decision["run_id"],
        STEP_ID_KEY: decision["step_id"],
        COORDINATE_KEY: decision["coordinate"],
        ACTUAL_KEY: decision["actual"],
        "cond_or_uncond": [0],
        "uuids": [label],
    }
    executor = patcher_extension.WrapperExecutor.new_class_executor(
        model._forward, model, [diffusion_model_wrapper]
    )
    output = executor.execute(x, torch.tensor([model_timestep]), context, options, minimax_payload=payload)
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return output, decision


def test_forced_actual_wrapper_is_native_equivalent_and_does_not_mutate_options():
    _, _, PackedLayout = _native_imports()
    model, _ = _tiny_model()
    x, context, payload = _inputs(PackedLayout)
    native_options = {"sentinel": {"value": 1}}
    before = copy.deepcopy(native_options)
    native = model._forward(x, torch.tensor([500.0]), context, native_options, minimax_payload=payload)

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(degree=1, max_history=4, force_actual=True, warmup_steps=0, tail_actual_steps=0)
    )
    run_id = runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    wrapped, _ = _wrapped_call(model, runtime, 1.0, 500.0, x, context, payload)
    assert native_options == before
    assert isinstance(wrapped, list) and len(wrapped) == 2
    for native_part, wrapped_part in zip(native, wrapped, strict=True):
        assert native_part.shape == wrapped_part.shape
        assert native_part.dtype == wrapped_part.dtype
        torch.testing.assert_close(native_part, wrapped_part, rtol=0.0, atol=0.0)
    runtime.end_run(run_id)


def test_actual_capture_avoids_full_audio_video_concatenation(monkeypatch):
    _, _, PackedLayout = _native_imports()
    model, _ = _tiny_model()
    x, context, payload = _inputs(PackedLayout)
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(degree=1, max_history=4, force_actual=True, warmup_steps=0, tail_actual_steps=0)
    )
    run_id = runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    original_cat = torch.cat

    def guarded_cat(tensors, *args, **kwargs):
        values = tuple(tensors)
        if len(values) == 2 and all(
            value.ndim == 2 and value.shape[-1] == model.hidden_size for value in values
        ):
            pytest.fail("actual target capture materialized a full audio/video concatenation")
        return original_cat(values, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", guarded_cat)
    wrapped, _ = _wrapped_call(model, runtime, 1.0, 500.0, x, context, payload)

    assert isinstance(wrapped, list) and len(wrapped) == 2
    assert runtime.stats.direct_history_updates == 1
    runtime.end_run(run_id)


def test_exact_forecast_reproduces_the_native_audio_video_velocity(monkeypatch):
    """A forecast that recovers the true final-block feature must equal native _forward.

    This pins the audio-velocity convention: cores that expose time_shift_slope want
    the forecast pre-scaled by d(sigma_a)/d(sigma_v), newer cores convert outside the
    wrapper and want it unscaled. Getting it wrong only shows up as audio drift.
    """
    _, _, PackedLayout = _native_imports()
    model, _ = _tiny_model()
    x, context, payload = _inputs(PackedLayout)
    native = model._forward(x, torch.tensor([500.0]), context, {}, minimax_payload=payload)

    captured = {}
    original_observe = SpectrumH3Runtime.observe_actual

    def recording_observe(self, run_id, step_id, call_id, target):
        captured["target"] = target.detach().clone()
        return original_observe(self, run_id, step_id, call_id, target)

    monkeypatch.setattr(SpectrumH3Runtime, "observe_actual", recording_observe)
    capture_runtime = SpectrumH3Runtime(
        SpectrumH3Config(degree=1, max_history=4, force_actual=True, warmup_steps=0, tail_actual_steps=0)
    )
    capture_run = capture_runtime.start_run(torch.tensor([1.0, 0.0]), "sample_euler", supported_sampler=True)
    _wrapped_call(model, capture_runtime, 1.0, 500.0, x, context, payload)
    capture_runtime.end_run(capture_run)
    monkeypatch.setattr(SpectrumH3Runtime, "observe_actual", original_observe)
    assert "target" in captured

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(degree=1, max_history=4, warmup_steps=2, tail_actual_steps=0, window_size=2.0)
    )
    run_id = runtime.start_run(torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), "sample_euler", supported_sampler=True)
    _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload)
    _wrapped_call(model, runtime, 0.75, 750.0, x, context, payload)

    # Patch the forecaster, not the runtime, so all step bookkeeping still runs.
    monkeypatch.setattr(
        HistoryWeightForecaster,
        "predict",
        lambda self, coordinate, blend_weight, *, rows, device, dtype: captured["target"].to(
            device=device, dtype=dtype
        ),
    )
    forecast, forecast_decision = _wrapped_call(model, runtime, 0.5, 500.0, x, context, payload)
    assert not forecast_decision["actual"]
    for native_part, forecast_part in zip(native, forecast, strict=True):
        assert native_part.shape == forecast_part.shape
        assert native_part.dtype == forecast_part.dtype
        torch.testing.assert_close(native_part, forecast_part, rtol=1e-5, atol=1e-5)
    runtime.end_run(run_id)


def test_forecast_step_skips_every_transformer_block_and_preserves_output_shapes():
    _, _, PackedLayout = _native_imports()
    model, block = _tiny_model()
    x, context, payload = _inputs(PackedLayout)
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(degree=1, max_history=4, warmup_steps=2, tail_actual_steps=0, window_size=2.0)
    )
    run_id = runtime.start_run(torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), "sample_euler", supported_sampler=True)
    _first, first_decision = _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload)
    second, second_decision = _wrapped_call(model, runtime, 0.75, 750.0, x, context, payload)
    calls_before = block.calls
    forecast, forecast_decision = _wrapped_call(model, runtime, 0.5, 500.0, x, context, payload)
    assert first_decision["actual"] and second_decision["actual"]
    assert not forecast_decision["actual"]
    assert block.calls == calls_before
    assert [part.shape for part in forecast] == [part.shape for part in second]
    assert [part.dtype for part in forecast] == [part.dtype for part in second]
    assert all(torch.isfinite(part).all() for part in forecast)
    runtime.end_run(run_id)
