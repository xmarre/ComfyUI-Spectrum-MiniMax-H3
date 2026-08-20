from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.minimax_h3 import _prepare_output_state


class _MaskCapableInner:
    patch_size = (1, 2, 2)
    hidden_size = 4
    latents_dim = 24
    audio_latents_dim = 32
    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0
    use_adaln_curves = False

    @staticmethod
    def time_embedder(values):
        return values[:, None].repeat(1, 3)


def _native_module():
    module = importlib.import_module("comfy.ldm.minimax.model")
    if not hasattr(module, "mask_row_values"):
        pytest.skip("reviewed ComfyUI revision predates native MiniMax H3 per-token masks")
    return module


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


def _row_indices(rows_t, timestep_row):
    levels = rows_t.unique()
    base = torch.tensor(
        [timestep_row[value] for value in levels.tolist()],
        dtype=torch.long,
        device=rows_t.device,
    )
    return base[torch.searchsorted(levels, rows_t)]


def test_spectrum_masked_output_rows_match_native_core_contract():
    module = _native_module()
    _MaskCapableInner.__module__ = module.__name__
    inner = _MaskCapableInner()

    video_x = torch.zeros((1, 24, 2, 4, 4))
    audio_x = torch.zeros((1, 32, 2, 2))
    context = torch.zeros((1, 2, 4))
    timestep = torch.tensor([500.0])
    video_mask = torch.ones((1, 1, 2, 4, 4))
    video_mask[:, :, 0] = 0.0
    audio_mask = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])

    state = _prepare_output_state(
        inner,
        video_x,
        audio_x,
        timestep,
        context,
        {},
        {},
        _layout(),
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(
        1.0
        - module.time_shift_sigma(
            sigma_v,
            inner.sigma_shift_video,
            inner.sigma_shift_audio,
        )
    )

    video_mask_rows = module.mask_row_values(video_mask[0, 0].float(), 2, 4, 4)
    assert video_mask_rows is not None
    video_rows_t = (1.0 - video_mask_rows * sigma_v).clamp(
        max=max(t_v, module.VISUAL_COND_TIMESTEP)
    )

    audio_mask_rows = audio_mask[0, 0].float().reshape(-1)
    sigma_a = 1.0 - t_a
    audio_rows_t = (1.0 - audio_mask_rows * sigma_a).clamp(
        max=max(t_a, module.AUDIO_COND_TIMESTEP)
    )

    unique_t = sorted(
        {t_v, t_a, max(t_v, module.VISUAL_COND_TIMESTEP)}
        | set(video_rows_t.unique().tolist())
        | set(audio_rows_t.unique().tolist())
    )
    timestep_row = {value: index for index, value in enumerate(unique_t)}

    assert torch.equal(
        state.video_timestep_row,
        _row_indices(video_rows_t, timestep_row),
    )
    assert torch.equal(
        state.audio_timestep_row,
        _row_indices(audio_rows_t, timestep_row),
    )
    assert state.t_emb.shape[0] == len(unique_t)


def test_native_final_layer_supports_per_row_modulation_indices():
    module = _native_module()
    mod_row = getattr(module, "_mod_row", None)
    if not callable(mod_row):
        pytest.fail("mask-capable native H3 Core is missing _mod_row vector-index support")

    vectors = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    rows = torch.tensor([1, 4, 2], dtype=torch.long)
    result = mod_row(vectors, rows, torch.float32)

    assert torch.equal(result, vectors[rows])
