"""Native MiniMax H3 FinalLayer projection contracts across ComfyUI revisions."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import minimax_h3


class _LegacyFinalLayer:
    def forward(self, x, t_emb, video_seg, audio_seg):
        pass


class _ShiftFinalLayer:
    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        pass


class _GridFinalLayer:
    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
        pass


class _PartialFinalLayer:
    def forward(self, x, t_emb, video_seg, audio_seg, sigma, grids):
        pass


class _AmbiguousFinalLayer:
    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts, grids):
        pass


@pytest.mark.parametrize(
    ("layer_type", "expected"),
    [
        (_LegacyFinalLayer, None),
        (_ShiftFinalLayer, "shifts"),
        (_GridFinalLayer, "grids"),
    ],
)
def test_native_projection_contract_variants(layer_type, expected):
    assert minimax_h3._final_layer_pdd_contract(SimpleNamespace(FinalLayer=layer_type)) == expected


@pytest.mark.parametrize("layer_type", [_PartialFinalLayer, _AmbiguousFinalLayer])
def test_unreviewed_native_pdd_contract_fails_closed(layer_type):
    with pytest.raises(RuntimeError, match="incomplete or ambiguous PDD projection contract"):
        minimax_h3._final_layer_pdd_contract(SimpleNamespace(FinalLayer=layer_type))


class _GridRecordingLayer:
    pdd_steps = 3

    def __init__(self):
        self.seen = []

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, grids):
        self.seen.append((sigma, sample_sigmas, grids))
        va, vb, _ = video_seg
        aa, ab, _ = audio_seg
        return x[va:vb], x[aa:ab]

    __call__ = forward


def _state(*, video_selector=1, audio_selector=0):
    return minimax_h3._OutputState(
        layout=None,
        t_emb=torch.zeros(3, 4),
        video_timestep_row=video_selector,
        audio_timestep_row=audio_selector,
        sigma_v=torch.tensor(0.5),
        sample_sigmas=torch.tensor([1.0, 0.5, 0.0]),
        shift_v=12.0,
        shift_a=3.0,
        original_video_shape=(1, 1, 1),
        padded_video_shape=(1, 1, 1),
    )


def test_grid_contract_uses_native_builder_and_caches_once_per_forecast():
    calls = []

    def pdd_grid(n, shift):
        calls.append((n, shift))
        return [shift * (n - k) / n for k in range(n + 1)]

    layer = _GridRecordingLayer()
    module = SimpleNamespace(FinalLayer=_GridRecordingLayer, pdd_grid=pdd_grid)
    inner = SimpleNamespace(final_layer=layer)
    state = _state()
    hidden = torch.randn(5, 4)
    for _ in range(2):
        video, audio = minimax_h3._final_layer_project(
            inner, module, hidden, state, (2, 5, 1), (0, 2, 0)
        )
        torch.testing.assert_close(video, hidden[2:5])
        torch.testing.assert_close(audio, hidden[0:2])
    assert calls == [(3, 12.0), (3, 3.0)]
    assert len(layer.seen) == 2
    assert layer.seen[0][2] is layer.seen[1][2] is state.pdd_grids


def test_grid_contract_requires_native_grid_builder():
    module = SimpleNamespace(FinalLayer=_GridRecordingLayer)
    with pytest.raises(RuntimeError, match="missing pdd_grid"):
        minimax_h3._final_layer_project(
            SimpleNamespace(final_layer=_GridRecordingLayer()),
            module,
            torch.zeros(3, 4),
            _state(),
            (1, 3, 0),
            (0, 1, 0),
        )


@pytest.mark.parametrize("pdd_steps", [1, 3])
@pytest.mark.parametrize("per_token", [False, True])
def test_native_gridded_pdd_matches_monolithic_and_streamed(monkeypatch, pdd_steps, per_token):
    try:
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
        from comfy.ldm.minimax import model as native
    except Exception as exc:
        pytest.skip(f"native ComfyUI source is unavailable: {exc}")

    if "grids" not in inspect.signature(native.FinalLayer.forward).parameters:
        pytest.skip("reviewed ComfyUI revision uses a pre-grid FinalLayer contract")

    torch.manual_seed(29)
    hidden = 4
    audio_rows, video_rows = 4, 6
    layer = native.FinalLayer(
        hidden, 4, 4, 4, 1e-5,
        pdd_steps=pdd_steps,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=torch.nn,
    )
    inner = SimpleNamespace(final_layer=layer, hidden_size=hidden)
    features = torch.randn(audio_rows + video_rows, hidden)
    state = _state(
        video_selector=(torch.tensor([0, 1, 2, 0, 1, 2]) if per_token else 2),
        audio_selector=(torch.tensor([2, 0, 1, 2]) if per_token else 1),
    )
    state.t_emb = torch.randn(3, 4)
    sigma = state.sigma_v
    sample_sigmas = state.sample_sigmas
    video_seg = (audio_rows, audio_rows + video_rows, state.video_timestep_row)
    audio_seg = (0, audio_rows, state.audio_timestep_row)
    native_grids = (
        native.pdd_grid(pdd_steps, state.shift_v),
        native.pdd_grid(pdd_steps, state.shift_a),
    )
    expected_video, expected_audio = layer(
        features, state.t_emb, video_seg, audio_seg, sigma, sample_sigmas, native_grids
    )

    video, audio = minimax_h3._final_layer_project(
        inner, native, features, state, video_seg, audio_seg
    )
    torch.testing.assert_close(video, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(audio, expected_audio, rtol=0, atol=0)
    assert state.pdd_grids == native_grids

    monkeypatch.setattr(
        minimax_h3, "_FORECAST_HEAD_CHUNK_BYTES", hidden * features.element_size() * 2
    )
    streamed_audio = minimax_h3._project_cpu_forecast_stream(
        inner, native, features, state,
        global_start=0, stream_rows=audio_rows, selector=state.audio_timestep_row,
        video=False, device=torch.device("cpu"),
    )
    streamed_video = minimax_h3._project_cpu_forecast_stream(
        inner, native, features, state,
        global_start=audio_rows, stream_rows=video_rows, selector=state.video_timestep_row,
        video=True, device=torch.device("cpu"),
    )
    torch.testing.assert_close(streamed_audio, expected_audio, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(streamed_video, expected_video, rtol=1e-5, atol=1e-6)
