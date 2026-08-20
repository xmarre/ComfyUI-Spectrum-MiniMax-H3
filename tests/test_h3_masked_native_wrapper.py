from __future__ import annotations

import pytest
import torch

import comfyui_spectrum_h3.minimax_h3 as minimax_h3_runtime
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


def _native_imports_allow_legacy():
    try:
        import comfy.cli_args

        comfy.cli_args.args.cpu = True
        import comfy.patcher_extension
        from comfy.ldm.minimax import model as minimax_model
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    except Exception as exc:  # noqa: BLE001 - external fixture availability
        pytest.skip(f"current ComfyUI source is unavailable: {exc}")
    return comfy.patcher_extension, minimax_model, MiniMaxH3Model, PackedLayout


def _native_imports():
    patcher_extension, minimax_model, MiniMaxH3Model, PackedLayout = _native_imports_allow_legacy()
    if not hasattr(minimax_model, "mask_row_values"):
        pytest.skip("reviewed ComfyUI revision predates native MiniMax H3 per-token masks")
    return patcher_extension, MiniMaxH3Model, PackedLayout


class _CountingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        self.calls += 1
        return x + t_emb[0].mean().to(x.dtype) * 0.01


def _tiny_model(*, allow_legacy=False):
    if allow_legacy:
        _, _, MiniMaxH3Model, _ = _native_imports_allow_legacy()
    else:
        _, MiniMaxH3Model, _ = _native_imports()
    torch.manual_seed(31)
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
    video = torch.randn(1, 2, 2, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 8)
    payload = {"layout": PackedLayout(2, 2, 4, 4, 3), "seed": 13}
    video_mask = torch.ones((1, 1, 2, 4, 4), dtype=torch.float32)
    video_mask[:, :, 0] = 0.0
    audio_mask = torch.ones((1, 1, 2, 3), dtype=torch.float32)
    audio_mask[:, :, :, 0] = 0.0
    return [video, audio], context, payload, video_mask, audio_mask


def _wrapped_call(
    model,
    runtime,
    sigma,
    model_timestep,
    x,
    context,
    payload,
    video_mask,
    audio_mask,
):
    patcher_extension, _, _, _ = _native_imports_allow_legacy()
    decision = runtime.begin_step(torch.tensor([sigma]))
    options = {
        RUNTIME_KEY: runtime,
        RUN_ID_KEY: decision["run_id"],
        STEP_ID_KEY: decision["step_id"],
        COORDINATE_KEY: decision["coordinate"],
        ACTUAL_KEY: decision["actual"],
        "cond_or_uncond": [0],
        "uuids": ["positive"],
    }
    executor = patcher_extension.WrapperExecutor.new_class_executor(
        model._forward,
        model,
        [diffusion_model_wrapper],
    )
    output = executor.execute(
        x,
        torch.tensor([model_timestep]),
        context,
        options,
        minimax_payload=payload,
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return output, decision


def test_exact_masked_forecast_reconstructs_native_velocity_and_skips_transformer(monkeypatch):
    _, _, PackedLayout = _native_imports()
    model, block = _tiny_model()
    x, context, payload, video_mask, audio_mask = _inputs(PackedLayout)

    native = model._forward(
        x,
        torch.tensor([500.0]),
        context,
        {},
        minimax_payload=payload,
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    captured = {}
    original_observe = SpectrumH3Runtime.observe_actual

    def recording_observe(self, run_id, step_id, call_id, target):
        captured["target"] = target.detach().clone()
        return original_observe(self, run_id, step_id, call_id, target)

    monkeypatch.setattr(SpectrumH3Runtime, "observe_actual", recording_observe)
    capture_runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            force_actual=True,
            warmup_steps=0,
            tail_actual_steps=0,
        )
    )
    capture_run = capture_runtime.start_run(
        torch.tensor([0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )
    _wrapped_call(
        model,
        capture_runtime,
        0.5,
        500.0,
        x,
        context,
        payload,
        video_mask,
        audio_mask,
    )
    capture_runtime.end_run(capture_run)
    monkeypatch.setattr(SpectrumH3Runtime, "observe_actual", original_observe)
    assert "target" in captured

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )
    _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload, video_mask, audio_mask)
    _wrapped_call(model, runtime, 0.75, 750.0, x, context, payload, video_mask, audio_mask)
    calls_before_forecast = block.calls

    monkeypatch.setattr(
        HistoryWeightForecaster,
        "predict_segments",
        lambda self, coordinate, segment_blends, *, rows, device, dtype: captured["target"].to(
            device=device,
            dtype=dtype,
        ),
    )
    forecast, decision = _wrapped_call(
        model,
        runtime,
        0.5,
        500.0,
        x,
        context,
        payload,
        video_mask,
        audio_mask,
    )

    assert not decision["actual"]
    assert block.calls == calls_before_forecast
    assert runtime.disabled_reason is None
    for native_part, forecast_part in zip(native, forecast, strict=True):
        assert native_part.shape == forecast_part.shape
        assert native_part.dtype == forecast_part.dtype
        torch.testing.assert_close(native_part, forecast_part, rtol=1e-5, atol=1e-5)
    runtime.end_run(run_id)


def test_masked_forecast_selects_native_fallback_when_core_lacks_mask_row_values(monkeypatch):
    _, minimax_model, _, PackedLayout = _native_imports_allow_legacy()
    model, _ = _tiny_model(allow_legacy=True)
    x, context, payload, video_mask, audio_mask = _inputs(PackedLayout)

    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=1,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=True,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_euler",
        supported_sampler=True,
    )
    _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload, video_mask, audio_mask)

    sentinel = [torch.zeros_like(x[0]), torch.zeros_like(x[1])]
    fallback_calls = []

    def fake_execute_actual(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return sentinel

    # Deleting the Core helper deliberately creates a legacy/unsupported masked
    # output-head environment. The compatibility invariant under test is that
    # Spectrum selects its native-actual fallback before trying forecast
    # reconstruction. Do not execute that deliberately broken Core path here:
    # some historical revisions reference mask_row_values from native _forward
    # and therefore cannot themselves service a masked call after the helper is
    # removed.
    monkeypatch.delattr(minimax_model, "mask_row_values", raising=False)
    monkeypatch.setattr(minimax_h3_runtime, "_execute_actual", fake_execute_actual)

    output, decision = _wrapped_call(
        model,
        runtime,
        0.5,
        500.0,
        x,
        context,
        payload,
        video_mask,
        audio_mask,
    )

    assert not decision["actual"]
    assert output is sentinel
    assert len(fallback_calls) == 1
    assert "lacks mask_row_values" in (runtime.disabled_reason or "")
    runtime.end_run(run_id)
