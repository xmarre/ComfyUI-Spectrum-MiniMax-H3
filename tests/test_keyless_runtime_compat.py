from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import keyless_compat, keyless_runtime_compat, minimax_h3


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
    provenance_identity = "artifact-a"

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


def _populate_keyless(instance):
    def attention():
        return SimpleNamespace(
            qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
            q_norm=object(),
            route_norm=object(),
        )

    instance.blocks = [SimpleNamespace(attn=attention()) for _ in range(50)]
    instance.token_refiner = SimpleNamespace(blocks=[object(), object()])
    instance.final_layer = object()
    instance.hidden_size = 5376
    instance.patch_size = (1, 2, 2)
    instance.latents_dim = 16
    instance.audio_latents_dim = 8
    instance.sigma_shift_video = 1.0
    instance.sigma_shift_audio = 1.0
    instance.use_adaln_curves = True
    instance.video_patch_proj = object()
    instance.audio_patch_proj = object()
    instance.adaln_t_table = object()
    setattr(instance, keyless_compat.KEYLESS_CONTRACT_KEY, Contract())
    return instance


def _native_module(native_type):
    return SimpleNamespace(
        MiniMaxH3Model=native_type,
        PackedLayout=object(),
        unpatchify_video=lambda *args, **kwargs: None,
        unpack_audio=lambda *args, **kwargs: None,
        time_shift_sigma=lambda *args, **kwargs: None,
        patchify_video=lambda *args, **kwargs: None,
        pack_audio=lambda *args, **kwargs: None,
    )


def test_runtime_predicate_accepts_valid_keyless_contract():
    inner = _populate_keyless(SimpleNamespace())
    assert keyless_runtime_compat._is_spectrum_minimax_h3(inner) is True
    # Package initialization installs this exact predicate into the live wrapper.
    assert minimax_h3.is_native_minimax_h3(inner) is True


def test_runtime_predicate_fails_closed_on_malformed_advertised_contract():
    inner = _populate_keyless(SimpleNamespace())
    getattr(inner, keyless_compat.KEYLESS_CONTRACT_KEY).routing_source = "key"
    with pytest.raises(keyless_compat.KeylessCompatibilityError, match="routing_source"):
        keyless_runtime_compat._is_spectrum_minimax_h3(inner)


def test_runtime_predicate_delegates_native_models(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_IS_NATIVE_MINIMAX_H3",
        lambda value: value is marker,
    )
    assert keyless_runtime_compat._is_spectrum_minimax_h3(marker) is True
    assert keyless_runtime_compat._is_spectrum_minimax_h3(object()) is False


def test_keyless_runtime_helpers_resolve_from_native_h3_module(monkeypatch):
    class Native:
        pass

    class Keyless(Native):
        pass

    inner = _populate_keyless(Keyless())
    module = _native_module(Native)
    monkeypatch.setattr(
        keyless_runtime_compat.importlib,
        "import_module",
        lambda name: module,
    )
    assert keyless_runtime_compat._runtime_native_module(inner) is module


def test_keyless_runtime_helpers_require_real_native_subclass(monkeypatch):
    class Native:
        pass

    inner = _populate_keyless(SimpleNamespace())
    module = _native_module(Native)
    monkeypatch.setattr(
        keyless_runtime_compat.importlib,
        "import_module",
        lambda name: module,
    )
    with pytest.raises(TypeError, match="not a subclass"):
        keyless_runtime_compat._runtime_native_module(inner)


def test_keyless_runtime_helpers_require_inherited_native_helpers(monkeypatch):
    class Native:
        pass

    class Keyless(Native):
        pass

    inner = _populate_keyless(Keyless())
    module = _native_module(Native)
    del module.pack_audio
    monkeypatch.setattr(
        keyless_runtime_compat.importlib,
        "import_module",
        lambda name: module,
    )
    with pytest.raises(RuntimeError, match="pack_audio"):
        keyless_runtime_compat._runtime_native_module(inner)


def test_native_runtime_helper_resolution_is_unchanged(monkeypatch):
    inner = SimpleNamespace()
    sentinel = object()
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_NATIVE_MODULE",
        lambda value: sentinel if value is inner else None,
    )
    assert keyless_runtime_compat._runtime_native_module(inner) is sentinel
