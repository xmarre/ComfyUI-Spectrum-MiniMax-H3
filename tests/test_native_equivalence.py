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

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        self.calls += 1
        return x + t_emb[0].mean().to(x.dtype) * 0.01


class _CountingFinalLayer(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


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


def _tiny_state_residual_model():
    _, MiniMaxH3Model, _ = _native_imports()
    torch.manual_seed(37)
    model = MiniMaxH3Model(
        hidden_size=8,
        num_layers=2,
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
    blocks = [_CountingBlock(), _CountingBlock()]
    model.blocks[0] = blocks[0]
    model.blocks[1] = blocks[1]
    return model, blocks


def _inputs(PackedLayout):
    video = torch.randn(1, 2, 1, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 8)
    payload = {"layout": PackedLayout(2, 1, 4, 4, 3), "seed": 5}
    return [video, audio], context, payload


def _reference_inputs(PackedLayout):
    video = torch.randn(1, 2, 1, 4, 4)
    audio = torch.randn(1, 2, 2, 3)
    context = torch.randn(1, 2, 8)
    ref_video = torch.randn(1, 2, 1, 4, 4)
    ref_audio = torch.randn(1, 2, 2, 2)
    refs = [
        {
            "kind": "video_audio",
            "latent_t": 1,
            "latent_h": 4,
            "latent_w": 4,
            "ref_audio_t": 2,
            "latent": ref_video,
            "audio_latent": ref_audio,
        }
    ]
    payload = {
        "layout": PackedLayout(2, 1, 4, 4, 3, refs=refs),
        "refs": refs,
        "cond_video_latents": [ref_video],
        "cond_audio_latents": [ref_audio],
        "seed": 5,
    }
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


def test_stochastic_seeds_state_residual_forecast_uses_exact_current_input_and_skips_blocks():
    _, _, PackedLayout = _native_imports()
    model, blocks = _tiny_state_residual_model()
    base_x, context, payload = _inputs(PackedLayout)
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            model_aware_mode="off",
            warmup_steps=0,
            tail_actual_steps=0,
            bootstrap_first_forecast=False,
            window_size=2.0,
            flex_window=0.0,
            offline_smoothing_replay=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_seeds_2",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=5,
        stage_count=2,
        stochastic_multistage=True,
    )

    for index, sigma in enumerate((1.0, 0.85, 0.7, 0.55)):
        x = [
            base_x[0] + 0.05 * index,
            base_x[1] - 0.03 * index,
        ]
        output, decision = _wrapped_call(
            model,
            runtime,
            sigma,
            sigma * 1000.0,
            x,
            context,
            payload,
        )
        assert decision["actual"]
        assert all(torch.isfinite(part).all() for part in output)

    calls_before_forecast = [block.calls for block in blocks]
    forecast_x = [
        base_x[0] + 0.4,
        base_x[1] - 0.2,
    ]
    output, decision = _wrapped_call(
        model,
        runtime,
        0.4,
        400.0,
        forecast_x,
        context,
        payload,
    )

    assert decision["actual"] is False
    assert runtime.active_stage_index == 0
    assert [block.calls for block in blocks] == calls_before_forecast
    assert all(torch.isfinite(part).all() for part in output)
    assert runtime.stats.actual_transformer_calls == 4
    assert runtime.stats.forecast_model_calls == 1
    runtime.end_run(run_id)


def test_sa_solver_state_conditioned_forecast_uses_exact_current_input_and_skips_blocks():
    _, _, PackedLayout = _native_imports()
    model, blocks = _tiny_state_residual_model()
    base_x, context, payload = _inputs(PackedLayout)
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            model_aware_mode="off",
            warmup_steps=0,
            tail_actual_steps=0,
            bootstrap_first_forecast=False,
            window_size=2.0,
            flex_window=0.0,
            offline_smoothing_replay=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.7, 0.4, 0.0]),
        "sample_sa_solver",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=0,
        expected_model_calls=3,
        stage_count=1,
        state_conditioned_residual=True,
    )

    for index, sigma in enumerate((1.0, 0.7)):
        x = [
            base_x[0] + 0.12 * index,
            base_x[1] - 0.09 * index,
        ]
        output, decision = _wrapped_call(
            model,
            runtime,
            sigma,
            sigma * 1000.0,
            x,
            context,
            payload,
        )
        assert decision["actual"]
        assert all(torch.isfinite(part).all() for part in output)

    calls_before_forecast = [block.calls for block in blocks]
    forecast_x = [
        base_x[0] + 0.55,
        base_x[1] - 0.35,
    ]
    output, decision = _wrapped_call(
        model,
        runtime,
        0.4,
        400.0,
        forecast_x,
        context,
        payload,
    )

    assert decision["actual"] is False
    assert runtime.active_stage_index == 0
    assert runtime.state_conditioned_residual is True
    assert runtime.stochastic_multistage is False
    assert [block.calls for block in blocks] == calls_before_forecast
    assert all(torch.isfinite(part).all() for part in output)
    assert runtime.stats.actual_transformer_calls == 2
    assert runtime.stats.forecast_model_calls == 1
    runtime.end_run(run_id)


def test_forced_actual_wrapper_is_native_equivalent_and_preserves_existing_option_values():
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
    assert native_options["sentinel"] == before["sentinel"]
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
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
        )
    )
    run_id = runtime.start_run(torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), "sample_euler", supported_sampler=True)
    _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload)
    _wrapped_call(model, runtime, 0.75, 750.0, x, context, payload)

    # Patch the forecaster, not the runtime, so all step bookkeeping still runs.
    monkeypatch.setattr(
        HistoryWeightForecaster,
        "predict_segments",
        lambda self, coordinate, segment_blends, *, rows, device, dtype: captured["target"].to(
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
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            bootstrap_first_forecast=False,
        )
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


def test_ref2va_forecast_skips_transformer_with_visual_and_audio_references():
    _, _, PackedLayout = _native_imports()
    model, block = _tiny_model()
    x, context, payload = _reference_inputs(PackedLayout)
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
    _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload)
    second, second_decision = _wrapped_call(model, runtime, 0.75, 750.0, x, context, payload)
    calls_before = block.calls
    forecast, forecast_decision = _wrapped_call(model, runtime, 0.5, 500.0, x, context, payload)

    assert second_decision["actual"]
    assert not forecast_decision["actual"]
    assert block.calls == calls_before
    assert [part.shape for part in forecast] == [part.shape for part in second]
    assert [part.dtype for part in forecast] == [part.dtype for part in second]
    assert all(torch.isfinite(part).all() for part in forecast)
    runtime.end_run(run_id)


def test_bootstrap_forecast_skips_transformers_and_runs_the_current_output_head():
    _, _, PackedLayout = _native_imports()
    model, block = _tiny_model()
    final_layer = _CountingFinalLayer(model.final_layer)
    model.final_layer = final_layer
    x, context, payload = _inputs(PackedLayout)
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
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    actual, actual_decision = _wrapped_call(model, runtime, 1.0, 1000.0, x, context, payload)
    block_calls = block.calls
    head_calls = final_layer.calls
    forecast, forecast_decision = _wrapped_call(model, runtime, 0.5, 500.0, x, context, payload)

    assert actual_decision["actual"]
    assert not forecast_decision["actual"]
    assert forecast_decision["reason"] == "one-point bootstrap forecast"
    assert block.calls == block_calls
    assert final_layer.calls == head_calls + 1
    assert [part.shape for part in forecast] == [part.shape for part in actual]
    assert all(torch.isfinite(part).all() for part in forecast)
    runtime.end_run(run_id)


def test_anchor_residual_probe_runs_current_heads_without_marking_actual_as_forecast():
    _, _, PackedLayout = _native_imports()
    model, block = _tiny_model()
    final_layer = _CountingFinalLayer(model.final_layer)
    model.final_layer = final_layer
    x, context, payload = _inputs(PackedLayout)
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=4,
            warmup_steps=2,
            tail_actual_steps=0,
            window_size=2.0,
            flex_window=0.0,
            bootstrap_first_forecast=False,
            anchor_residual_feedback=True,
            offline_smoothing_replay=False,
        )
    )
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        "sample_euler",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
    )

    decisions = []
    for sigma, model_timestep in ((1.0, 1000.0), (0.75, 750.0), (0.5, 500.0), (0.25, 250.0)):
        _output, decision = _wrapped_call(
            model,
            runtime,
            sigma,
            model_timestep,
            x,
            context,
            payload,
        )
        decisions.append(decision)

    assert [decision["actual"] for decision in decisions] == [True, True, False, True]
    assert block.calls == 3
    assert final_layer.calls == 6
    assert runtime.stats.actual_steps == 3
    assert runtime.stats.forecast_steps == 1
    assert runtime.stats.actual_transformer_calls == 3
    assert runtime.stats.residual_anchors == 1
    assert runtime.stats.residual_measure_seconds >= 0.0
    assert runtime.stats.residual_output_head_seconds >= 0.0
    runtime.end_run(run_id)