from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import model_aware
from comfyui_spectrum_h3 import keyless_compat


class Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1

    def __init__(self, provenance):
        self.provenance_identity = provenance

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.routing_source,
            self.retrieval_source,
            self.provenance_identity,
        )


def _keyless_inner(provenance="a"):
    attention = lambda: SimpleNamespace(
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=object(),
        route_norm=object(),
    )
    inner = SimpleNamespace(
        blocks=[SimpleNamespace(attn=attention()) for _ in range(50)],
        token_refiner=SimpleNamespace(blocks=[object(), object()]),
        final_layer=object(),
        hidden_size=5376,
        patch_size=(1, 2, 2),
        latents_dim=16,
        audio_latents_dim=8,
        sigma_shift_video=1.0,
        sigma_shift_audio=1.0,
        use_adaln_curves=True,
        video_patch_proj=object(),
        audio_patch_proj=object(),
        adaln_t_table=object(),
    )
    setattr(inner, keyless_compat.KEYLESS_CONTRACT_KEY, Contract(provenance))
    return inner


def test_model_aware_selects_qv_instead_of_nonexistent_qkv_for_keyless():
    keys = model_aware._selected_base_keys(_keyless_inner())
    assert "diffusion_model.blocks.49.attn.qv_proj.weight" in keys
    assert "diffusion_model.blocks.49.attn.qkv_proj.weight" not in keys
    assert "diffusion_model.blocks.49.attn.out_proj.weight" in keys


def test_model_aware_architecture_signature_binds_keyless_provenance():
    first = model_aware._architecture_signature(_keyless_inner("artifact-a"))
    second = model_aware._architecture_signature(_keyless_inner("artifact-b"))
    assert first != second
    assert first[-1][0] == "keyless_semantic_identity"
    assert first[-1][1][-1] == "artifact-a"


def test_native_model_aware_projection_selection_is_unchanged():
    inner = SimpleNamespace(
        blocks=[object(), object()],
        hidden_size=5376,
        patch_size=(1, 2, 2),
        latents_dim=16,
        audio_latents_dim=8,
        use_adaln_curves=True,
    )
    keys = model_aware._selected_base_keys(inner)
    assert "diffusion_model.blocks.1.attn.qkv_proj.weight" in keys
    assert not any("qv_proj" in key for key in keys)


def test_model_aware_does_not_fall_back_to_qkv_for_malformed_keyless_contract():
    inner = _keyless_inner()
    getattr(inner, keyless_compat.KEYLESS_CONTRACT_KEY).projection_attr = "qkv_proj"
    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="projection_attr"):
        model_aware._selected_base_keys(inner)
