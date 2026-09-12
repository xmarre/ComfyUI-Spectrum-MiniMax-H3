from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import comfyui_spectrum_h3.backend_history as backend_history
import comfyui_spectrum_h3.minimax_h3 as minimax_h3


class _RowLocalFinalLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_out = SimpleNamespace(out_features=4)
        self.audio_out = SimpleNamespace(out_features=4)
        self.calls = 0

    @staticmethod
    def _project(x, segment, bias):
        first, last, selector = segment
        value = x[first:last].clone()
        if torch.is_tensor(selector) and selector.ndim > 0:
            modulation = selector.to(device=x.device, dtype=x.dtype).reshape(-1, 1)
        else:
            modulation = torch.as_tensor(
                float(selector),
                device=x.device,
                dtype=x.dtype,
            )
        return value + modulation + float(bias)

    def forward(self, x, t_emb, video_segment, audio_segment):
        del t_emb
        self.calls += 1
        return (
            self._project(x, video_segment, 10.0),
            self._project(x, audio_segment, -5.0),
        )


class _PDDRowLocalFinalLayer(_RowLocalFinalLayer):
    def forward(
        self,
        x,
        t_emb,
        video_segment,
        audio_segment,
        sigma=None,
        sample_sigmas=None,
        shifts=None,
    ):
        del t_emb
        self.calls += 1
        assert sigma is not None
        assert sample_sigmas is not None
        assert shifts is not None
        pdd_bias = (
            float(sigma)
            + float(sample_sigmas.sum())
            + float(shifts[0])
            + float(shifts[1])
        )
        return (
            self._project(x, video_segment, 10.0 + pdd_bias),
            self._project(x, audio_segment, -5.0 + pdd_bias),
        )


class _ForeignFinalLayer(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.video_out = inner.video_out
        self.audio_out = inner.audio_out
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


class _H3OptimizationsCompatibleFinalLayer(_RowLocalFinalLayer):
    def forward(self, x, t_emb, video_segment, audio_segment):
        return super().forward(x, t_emb, video_segment, audio_segment)


_H3OptimizationsCompatibleFinalLayer.forward.__module__ = (
    "h3_optimizations.memory.final_layer"
)
_H3OptimizationsCompatibleFinalLayer.forward._h3_optimizations_final_layer = True
_H3OptimizationsCompatibleFinalLayer.forward._h3_optimizations_final_layer_signature = 2
_H3OptimizationsCompatibleFinalLayer.forward._h3_optimizations_cube_order_state = None


def _state(audio_selector, video_selector):
    return minimax_h3._OutputState(
        layout=None,
        t_emb=torch.zeros(1, 4),
        video_timestep_row=video_selector,
        audio_timestep_row=audio_selector,
        sigma_v=torch.tensor(0.5),
        sample_sigmas=torch.tensor([1.0, 0.5, 0.0]),
        shift_v=1.1,
        shift_a=0.9,
        original_video_shape=(1, 1, 1),
        padded_video_shape=(1, 1, 1),
    )


@pytest.mark.parametrize("pdd", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_streamed_final_layer_matches_monolithic_for_selectors_and_pdd(
    monkeypatch,
    pdd,
    per_token,
):
    torch.manual_seed(7)
    audio_rows = 5
    video_rows = 6
    hidden = 4
    compact = torch.randn(audio_rows + video_rows, hidden)
    layer = _PDDRowLocalFinalLayer() if pdd else _RowLocalFinalLayer()
    inner = SimpleNamespace(hidden_size=hidden, final_layer=layer)
    module = SimpleNamespace(FinalLayer=type(layer))
    audio_selector = (
        torch.arange(audio_rows, dtype=torch.long)
        if per_token
        else 2
    )
    video_selector = (
        torch.arange(video_rows, dtype=torch.long) + 3
        if per_token
        else 4
    )
    state = _state(audio_selector, video_selector)

    video_segment = (
        audio_rows,
        audio_rows + video_rows,
        state.video_timestep_row,
    )
    audio_segment = (0, audio_rows, state.audio_timestep_row)
    monolithic_video, monolithic_audio = minimax_h3._final_layer_project(
        inner,
        module,
        compact,
        state,
        video_segment,
        audio_segment,
    )

    # Force multiple slabs even on this tiny CPU fixture.
    monkeypatch.setattr(
        minimax_h3,
        "_FORECAST_HEAD_CHUNK_BYTES",
        hidden * compact.element_size() * 2,
    )
    streamed_audio = minimax_h3._project_cpu_forecast_stream(
        inner,
        module,
        compact,
        state,
        global_start=0,
        stream_rows=audio_rows,
        selector=state.audio_timestep_row,
        video=False,
        device=torch.device("cpu"),
    )
    streamed_video = minimax_h3._project_cpu_forecast_stream(
        inner,
        module,
        compact,
        state,
        global_start=audio_rows,
        stream_rows=video_rows,
        selector=state.video_timestep_row,
        video=True,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(streamed_audio, monolithic_audio)
    torch.testing.assert_close(streamed_video, monolithic_video)


def test_cube_order_stream_reorders_selector_and_restores_video_output(monkeypatch):
    hidden = 4
    raster = torch.arange(16, dtype=torch.float32).reshape(4, hidden)
    forward = (2, 0, 3, 1)
    cube = raster.index_select(0, torch.tensor(forward))
    selector = torch.tensor([11, 22, 33, 44], dtype=torch.long)
    topology = SimpleNamespace(forward=forward)
    layer = _RowLocalFinalLayer()
    inner = SimpleNamespace(hidden_size=hidden, final_layer=layer)
    module = SimpleNamespace(FinalLayer=_RowLocalFinalLayer)
    state = _state(0, selector)

    expected, _ = minimax_h3._final_layer_project(
        inner,
        module,
        raster,
        state,
        (0, 4, selector),
        (4, 4, 0),
    )
    monkeypatch.setattr(
        minimax_h3,
        "_FORECAST_HEAD_CHUNK_BYTES",
        hidden * raster.element_size() * 2,
    )
    streamed = minimax_h3._project_cpu_forecast_stream(
        inner,
        module,
        cube,
        state,
        global_start=0,
        stream_rows=4,
        selector=selector,
        video=True,
        device=torch.device("cpu"),
        topology=topology,
        reorder_selector=True,
    )

    torch.testing.assert_close(streamed, expected)


def test_streamed_state_reconstruction_matches_monolithic_projection(monkeypatch):
    torch.manual_seed(9)
    rows = 5
    hidden = 4
    compact = torch.randn(rows, hidden)
    state_rows = torch.randn(rows, 3)
    projection = torch.nn.Linear(3, hidden, bias=False)
    scale = 0.75
    layer = _RowLocalFinalLayer()
    inner = SimpleNamespace(hidden_size=hidden, final_layer=layer)
    module = SimpleNamespace(FinalLayer=_RowLocalFinalLayer)
    state = _state(1, 2)

    reconstructed = compact + projection(state_rows).to(compact.dtype) * scale
    _, expected = minimax_h3._final_layer_project(
        inner,
        module,
        reconstructed,
        state,
        (rows, rows, 0),
        (0, rows, state.audio_timestep_row),
    )

    monkeypatch.setattr(
        minimax_h3,
        "_FORECAST_HEAD_CHUNK_BYTES",
        hidden * compact.element_size() * 2,
    )
    streamed = minimax_h3._project_cpu_forecast_stream(
        inner,
        module,
        compact,
        state,
        global_start=0,
        stream_rows=rows,
        selector=state.audio_timestep_row,
        video=False,
        device=torch.device("cpu"),
        state_rows_cpu=state_rows,
        state_projection=projection,
        state_scale=scale,
    )

    torch.testing.assert_close(streamed, expected)


def test_same_dtype_sanitizer_reuses_storage_and_repairs_nonfinite():
    feature = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
    sanitized, event = minimax_h3._sanitize_prediction(feature, torch.bfloat16)
    assert sanitized is feature
    assert event is None

    damaged = torch.tensor(
        [1.0, float("nan"), float("inf"), -float("inf")],
        dtype=torch.bfloat16,
    )
    sanitized, event = minimax_h3._sanitize_prediction(damaged, torch.bfloat16)
    assert sanitized is damaged
    assert event == {"nonfinite": 3, "below": 0, "above": 0}
    assert torch.isfinite(damaged).all()


def test_streaming_identity_gate_rejects_foreign_final_layer(monkeypatch):
    native = _RowLocalFinalLayer()
    foreign = _ForeignFinalLayer(native)
    module = SimpleNamespace(FinalLayer=_RowLocalFinalLayer)
    monkeypatch.setattr(minimax_h3, "_native_module", lambda inner: module)

    native_inner = SimpleNamespace(hidden_size=4, final_layer=native)
    foreign_inner = SimpleNamespace(hidden_size=4, final_layer=foreign)
    assert minimax_h3._final_layer_stream_kind(native_inner) == "native"
    assert minimax_h3._final_layer_stream_kind(foreign_inner) is None
    assert minimax_h3._resolve_final_layer_stream_compat(foreign_inner, 4) is None
    assert minimax_h3._forecast_prediction_device(
        foreign_inner,
        torch.device("cuda:0"),
    ) == torch.device("cuda:0")


def test_h3_optimizations_contract_is_source_gated(monkeypatch):
    layer = _H3OptimizationsCompatibleFinalLayer()
    module = SimpleNamespace(FinalLayer=_RowLocalFinalLayer)
    monkeypatch.setattr(minimax_h3, "_native_module", lambda inner: module)
    inner = SimpleNamespace(hidden_size=4, final_layer=layer)

    assert minimax_h3._final_layer_stream_kind(inner) == "h3_optimizations"
    compat = minimax_h3._resolve_final_layer_stream_compat(inner, 4)
    assert compat == (None, None, False, 2)

    # Markers alone are insufficient when the callable does not originate from
    # the reviewed H3-Optimizations FinalLayer module.
    layer.forward.__func__.__module__ = __name__
    try:
        assert minimax_h3._final_layer_stream_kind(inner) is None
    finally:
        layer.forward.__func__.__module__ = "h3_optimizations.memory.final_layer"



def test_unrecognized_final_layer_keeps_one_shot_projection(monkeypatch):
    native = _RowLocalFinalLayer()
    foreign = _ForeignFinalLayer(native)
    inner = SimpleNamespace(
        hidden_size=4,
        final_layer=foreign,
        patch_size=(1, 1, 1),
        latents_dim=1,
    )
    module = SimpleNamespace(
        FinalLayer=_RowLocalFinalLayer,
        unpatchify_video=lambda value, *args: torch.zeros(1, 1, 1, 1, 1),
        unpack_audio=lambda value: value,
    )
    monkeypatch.setattr(minimax_h3, "_native_module", lambda inner: module)
    layout = SimpleNamespace(
        segments=((0, 1, "audio"), (1, 2, "video")),
        seq_len=2,
    )
    state = minimax_h3._OutputState(
        layout=layout,
        t_emb=torch.zeros(1, 4),
        video_timestep_row=1,
        audio_timestep_row=0,
        sigma_v=torch.tensor(0.5),
        sample_sigmas=torch.tensor([1.0, 0.5, 0.0]),
        shift_v=1.0,
        shift_a=1.0,
        original_video_shape=(1, 1, 1),
        padded_video_shape=(1, 1, 1),
    )
    predicted = torch.randn(1, 2, 4)
    video_x = torch.zeros(1, 1, 1, 1, 1)
    audio_x = torch.zeros(1, 1, 1, 1)

    minimax_h3._execute_forecast(
        inner,
        predicted,
        state,
        video_x,
        audio_x,
    )

    assert foreign.calls == 1

@pytest.mark.parametrize("error_type", [AttributeError, ValueError])
def test_state_conditioned_reconstruction_failure_falls_back_to_actual(
    monkeypatch,
    error_type,
):
    class FakeRuntime:
        def __init__(self):
            self.config = SimpleNamespace(debug=False)
            self.active_state_conditioned_residual = True
            self.offline_phase = None
            self.active_stage_index = 0
            self.last_prediction_chunk_count = 1
            self.prediction_history_length = 2
            self.fallback_reason = None

        def begin_model_call(self, *args, **kwargs):
            return 3, False

        def predict(self, *args, **kwargs):
            return torch.zeros(1, 2, 4)

        def fallback_current_step(self, run_id, step_id, reason):
            self.fallback_reason = reason

    runtime = FakeRuntime()
    layout = SimpleNamespace(
        segments=((0, 1, "audio"), (1, 2, "video")),
        seq_len=2,
    )
    inner = SimpleNamespace(hidden_size=4)
    executor = SimpleNamespace(class_obj=inner)
    options = {
        minimax_h3.RUNTIME_KEY: runtime,
        minimax_h3.RUN_ID_KEY: 1,
        minimax_h3.STEP_ID_KEY: 2,
    }
    x = [
        torch.zeros(1, 1, 1, 1, 1),
        torch.zeros(1, 1, 1, 1),
    ]
    context = torch.zeros(1, 1, 4)
    actual_result = [torch.tensor([123.0]), torch.tensor([456.0])]

    monkeypatch.setattr(minimax_h3, "SpectrumH3Runtime", FakeRuntime)
    monkeypatch.setattr(minimax_h3, "is_native_minimax_h3", lambda inner: True)
    monkeypatch.setattr(minimax_h3, "_resolve_layout", lambda *args, **kwargs: layout)
    monkeypatch.setattr(
        minimax_h3,
        "topology_signature",
        lambda *args, **kwargs: ("fixture",),
    )
    monkeypatch.setattr(
        minimax_h3,
        "_forecast_prediction_device",
        lambda inner, device: device,
    )
    monkeypatch.setattr(
        minimax_h3,
        "_prepare_output_state",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        minimax_h3,
        "_execute_forecast",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            error_type("fixture reconstruction failure")
        ),
    )
    monkeypatch.setattr(
        minimax_h3,
        "_execute_actual",
        lambda *args, **kwargs: actual_result,
    )
    monkeypatch.setattr(
        backend_history,
        "prepare",
        lambda runtime, run_id, step_id, options, layout, inner: (options, None),
    )

    result = minimax_h3.diffusion_model_wrapper(
        executor,
        x,
        torch.tensor([500.0]),
        context,
        options,
        minimax_payload={},
    )

    assert result is actual_result
    assert runtime.fallback_reason is not None
    assert "state-conditioned residual forecast reconstruction failed" in (
        runtime.fallback_reason
    )
