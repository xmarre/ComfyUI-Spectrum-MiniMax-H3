from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


def _populate_keyless(instance, provenance="artifact-a"):
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
    setattr(instance, keyless_compat.KEYLESS_CONTRACT_KEY, Contract(provenance))
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


def test_keyless_topology_binds_architecture_and_checkpoint_provenance(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )
    first = keyless_runtime_compat._topology_signature(
        _populate_keyless(SimpleNamespace(), "artifact-a"),
        None,
        None,
        None,
        None,
        {},
        {},
    )
    second = keyless_runtime_compat._topology_signature(
        _populate_keyless(SimpleNamespace(), "artifact-b"),
        None,
        None,
        None,
        None,
        {},
        {},
    )
    assert first != second
    assert first[-2][0] == "keyless_semantic_identity"
    assert first[-2][1][-1] == "artifact-a"
    assert first[-1][0] == "keyless_runtime_identity"


def test_keyless_runtime_identity_is_stable_for_unchanged_options(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )

    class Provider:
        api = 1

        def __call__(self, **_kwargs):
            return None

    provider = Provider()
    mask = torch.ones(4, dtype=torch.bool)
    options = {
        "minimax_h3_keyless_provider_v1": provider,
        "minimax_h3_keyless_mask_v1": mask,
        "minimax_h3_keyless_value_domain_v1": (0, 1, 2, 3),
    }
    inner = _populate_keyless(SimpleNamespace())
    first = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    second = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    assert first == second


def test_callable_declared_provider_identity_is_stable(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )

    class Provider:
        api = 1

        def __call__(self, **_kwargs):
            return None

        def identity(self):
            return ("provider", "v1")

    provider = Provider()
    options = {"minimax_h3_keyless_provider_v1": provider}
    inner = _populate_keyless(SimpleNamespace())
    first = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    second = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    assert first == second


def test_keyless_provider_change_invalidates_topology(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )

    class Provider:
        api = 1

        def __call__(self, **_kwargs):
            return None

    inner = _populate_keyless(SimpleNamespace())
    first = keyless_runtime_compat._topology_signature(
        inner,
        None,
        None,
        None,
        None,
        {"minimax_h3_keyless_provider_v1": Provider()},
        {},
    )
    second = keyless_runtime_compat._topology_signature(
        inner,
        None,
        None,
        None,
        None,
        {"minimax_h3_keyless_provider_v1": Provider()},
        {},
    )
    assert first != second


def test_keyless_preprocessor_and_value_domain_changes_invalidate_topology(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )

    def route_transform(value):
        return value

    class Preprocessor:
        def __init__(self, identity):
            self.identity = identity
            self.fn = route_transform

    inner = _populate_keyless(SimpleNamespace())
    base_options = {
        "minimax_h3_keyless_routing_preprocessors_v1": (Preprocessor("untwist-a"),),
        "minimax_h3_keyless_value_domain_v1": SimpleNamespace(
            start=0, stop=8, indices=None, identity="domain-a"
        ),
    }
    first = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, base_options, {}
    )
    changed_preprocessor = dict(base_options)
    changed_preprocessor["minimax_h3_keyless_routing_preprocessors_v1"] = (
        Preprocessor("untwist-b"),
    )
    second = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, changed_preprocessor, {}
    )
    assert first != second

    changed_domain = dict(base_options)
    changed_domain["minimax_h3_keyless_value_domain_v1"] = SimpleNamespace(
        start=0, stop=7, indices=None, identity="domain-b"
    )
    third = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, changed_domain, {}
    )
    assert first != third


def test_keyless_in_place_mask_change_invalidates_topology(monkeypatch):
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: (("shape", "same"),),
    )
    inner = _populate_keyless(SimpleNamespace())
    mask = torch.ones(4, dtype=torch.bool)
    options = {"minimax_h3_keyless_mask_v1": mask}
    first = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    mask[0] = False
    second = keyless_runtime_compat._topology_signature(
        inner, None, None, None, None, options, {}
    )
    assert first != second


def test_native_topology_signature_is_unchanged(monkeypatch):
    expected = (("shape", "native"),)
    monkeypatch.setattr(
        keyless_runtime_compat,
        "_ORIGINAL_TOPOLOGY_SIGNATURE",
        lambda *args, **kwargs: expected,
    )
    assert keyless_runtime_compat._topology_signature(
        SimpleNamespace(), None, None, None, None, {}, {}
    ) == expected
