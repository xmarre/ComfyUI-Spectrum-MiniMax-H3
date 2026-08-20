from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.minimax_h3 import _execute_forecast, _prepare_output_state


class _FakeFinalLayer:
    def __init__(self):
        self.calls = []

    def __call__(self, compact, t_emb, video_segment, audio_segment):
        self.calls.append((compact.clone(), t_emb.clone(), video_segment, audio_segment))
        video_rows = video_segment[1] - video_segment[0]
        audio_rows = audio_segment[1] - audio_segment[0]
        return (
            torch.zeros((video_rows, 96), dtype=torch.float32),
            torch.zeros((audio_rows, 32), dtype=torch.float32),
        )


def _fake_module(monkeypatch):
    def time_shift_sigma(sigma, from_shift, to_shift):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)

    def mask_row_values(mask, latent_t, lat_h, lat_w):
        m = torch.nn.functional.pad(
            mask,
            (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]),
            mode="replicate",
        )
        m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
        values = m.reshape(-1)
        return None if bool((values >= 1.0 - 1e-3).all()) else values

    module = SimpleNamespace(
        time_shift_sigma=time_shift_sigma,
        mask_row_values=mask_row_values,
        VISUAL_COND_TIMESTEP=0.999,
        AUDIO_COND_TIMESTEP=1.0,
        unpack_audio=lambda rows: rows.reshape(2, rows.shape[0] // 2, rows.shape[-1])
        .permute(2, 0, 1)
        .unsqueeze(0),
        unpatchify_video=lambda rows, t, h, w, c, patch_size: torch.zeros(
            (1, c, t, h * 2, w * 2)
        ),
    )
    monkeypatch.setattr(
        "comfyui_spectrum_h3.minimax_h3._native_module",
        lambda _inner: module,
    )


def _inner():
    final_layer = _FakeFinalLayer()
    return SimpleNamespace(
        patch_size=(1, 2, 2),
        hidden_size=4,
        latents_dim=24,
        audio_latents_dim=32,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        use_adaln_curves=False,
        time_embedder=lambda values: values[:, None].repeat(1, 3),
        final_layer=final_layer,
    )


def _layout():
    return SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, 4, "ref_img"),
            (4, 8, "audio"),
            (8, 16, "video"),
        ],
        seq_len=16,
    )


def _masked_state(monkeypatch):
    _fake_module(monkeypatch)
    inner = _inner()
    video_x = torch.zeros((1, 24, 2, 4, 4))
    audio_x = torch.zeros((1, 32, 2, 2))
    context = torch.zeros((1, 2, 4))
    video_mask = torch.ones((1, 1, 2, 4, 4))
    video_mask[:, :, 0] = 0.0
    audio_mask = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    state = _prepare_output_state(
        inner,
        video_x,
        audio_x,
        torch.tensor([500.0]),
        context,
        {},
        {},
        _layout(),
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )
    return inner, video_x, audio_x, state


def test_masked_output_state_matches_core_per_row_timestep_contract(monkeypatch):
    _, _, _, state = _masked_state(monkeypatch)

    assert torch.is_tensor(state.video_timestep_row)
    assert torch.is_tensor(state.audio_timestep_row)
    assert state.video_timestep_row.shape == (8,)
    assert state.audio_timestep_row.shape == (4,)
    assert state.video_timestep_row.dtype == torch.long
    assert state.audio_timestep_row.dtype == torch.long
    assert state.video_timestep_row[:4].unique().numel() == 1
    assert state.video_timestep_row[4:].unique().numel() == 1
    assert int(state.video_timestep_row[0]) != int(state.video_timestep_row[-1])
    assert int(state.audio_timestep_row[0]) != int(state.audio_timestep_row[1])


def test_masked_forecast_passes_vector_timestep_rows_to_native_final_layer(monkeypatch):
    inner, video_x, audio_x, state = _masked_state(monkeypatch)
    predicted = torch.zeros((1, 12, 4))

    output = _execute_forecast(inner, predicted, state, video_x, audio_x)

    assert len(output) == 2
    assert len(inner.final_layer.calls) == 1
    _, _, video_segment, audio_segment = inner.final_layer.calls[0]
    assert video_segment[:2] == (4, 12)
    assert audio_segment[:2] == (0, 4)
    assert torch.equal(video_segment[2], state.video_timestep_row)
    assert torch.equal(audio_segment[2], state.audio_timestep_row)


def test_fully_generating_masks_keep_scalar_fast_path(monkeypatch):
    _fake_module(monkeypatch)
    inner = _inner()
    state = _prepare_output_state(
        inner,
        torch.zeros((1, 24, 2, 4, 4)),
        torch.zeros((1, 32, 2, 2)),
        torch.tensor([500.0]),
        torch.zeros((1, 2, 4)),
        {},
        {},
        _layout(),
        denoise_mask=torch.ones((1, 1, 2, 4, 4)),
        audio_denoise_mask=torch.ones((1, 1, 2, 2)),
    )

    assert isinstance(state.video_timestep_row, int)
    assert isinstance(state.audio_timestep_row, int)


def test_audio_mask_row_mismatch_fails_closed(monkeypatch):
    _fake_module(monkeypatch)
    inner = _inner()
    with pytest.raises(RuntimeError, match="audio denoise mask row count"):
        _prepare_output_state(
            inner,
            torch.zeros((1, 24, 2, 4, 4)),
            torch.zeros((1, 32, 2, 2)),
            torch.tensor([500.0]),
            torch.zeros((1, 2, 4)),
            {},
            {},
            _layout(),
            audio_denoise_mask=torch.ones((1, 1, 1, 2)),
        )
