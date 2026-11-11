from __future__ import annotations

from types import SimpleNamespace

import pytest

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

    def __init__(self, provenance="artifact-a"):
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


def _attention(*, fake_qkv=False):
    attention = SimpleNamespace(
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=SimpleNamespace(),
        route_norm=SimpleNamespace(),
    )
    if fake_qkv:
        attention.qkv_proj = SimpleNamespace(weight=SimpleNamespace(shape=(21504, 5376)))
    return attention


def _inner(*, contract=None, fake_qkv=False):
    inner = SimpleNamespace(
        blocks=[SimpleNamespace(attn=_attention(fake_qkv=fake_qkv)) for _ in range(50)],
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
    if contract is not None:
        setattr(inner, keyless_compat.KEYLESS_CONTRACT_KEY, contract)
    return inner


def test_valid_keyless_contract_selects_real_qv_projection_and_stable_identity():
    inner = _inner(contract=Contract("artifact-a"))
    contract = keyless_compat.validate_keyless_contract(inner)
    assert contract is getattr(inner, keyless_compat.KEYLESS_CONTRACT_KEY)
    assert keyless_compat.attention_projection_key(inner, 49) == (
        "diffusion_model.blocks.49.attn.qv_proj.weight"
    )
    identity = keyless_compat.keyless_semantic_identity(inner)
    assert identity[0] == keyless_compat.KEYLESS_CONTRACT_KEY
    assert identity[-1] == "artifact-a"


def test_keyless_identity_separates_checkpoint_provenance():
    first = keyless_compat.keyless_semantic_identity(_inner(contract=Contract("artifact-a")))
    second = keyless_compat.keyless_semantic_identity(_inner(contract=Contract("artifact-b")))
    assert first != second


def test_native_models_keep_qkv_projection_key_when_no_keyless_contract():
    inner = SimpleNamespace()
    assert keyless_compat.validate_keyless_contract(inner) is None
    assert keyless_compat.attention_projection_key(inner, 3) == (
        "diffusion_model.blocks.3.attn.qkv_proj.weight"
    )


def test_present_but_malformed_keyless_contract_fails_closed():
    contract = Contract()
    contract.routing_source = "key"
    inner = _inner(contract=contract)
    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="routing_source"):
        keyless_compat.validate_keyless_contract(inner)


def test_keyless_contract_rejects_fake_qkv_compatibility_alias():
    inner = _inner(contract=Contract(), fake_qkv=True)
    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="fake/dead K"):
        keyless_compat.validate_keyless_contract(inner)


def test_require_spectrum_minimax_h3_accepts_valid_keyless_model():
    inner = _inner(contract=Contract())
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=inner))
    resolved, path = keyless_compat.require_spectrum_minimax_h3(patcher)
    assert resolved is inner
    assert path == "model.diffusion_model"


def test_require_spectrum_minimax_h3_does_not_hide_invalid_advertised_contract(monkeypatch):
    contract = Contract()
    contract.qv_order = "v;q"
    inner = _inner(contract=contract)
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=inner))
    monkeypatch.setattr(keyless_compat, "is_native_minimax_h3", lambda value: True)
    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="qv_order"):
        keyless_compat.require_spectrum_minimax_h3(patcher)
