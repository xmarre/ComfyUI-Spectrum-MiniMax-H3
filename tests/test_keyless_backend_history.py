from __future__ import annotations

from types import SimpleNamespace

from comfyui_spectrum_h3 import (
    backend_history,
    core_bsa_forecast_recovery,
    keyless_backend_history,
    keyless_compat,
    keyless_core_bsa_fallback,
)


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


def _keyless_model():
    def attention():
        return SimpleNamespace(
            qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
            q_norm=object(),
            route_norm=object(),
        )

    model = SimpleNamespace(
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
    setattr(model, keyless_compat.KEYLESS_CONTRACT_KEY, Contract())
    return model


def test_plain_materialized_keyless_route_is_self_qualified():
    identity, safe, audit = keyless_backend_history._preflight(
        {}, object(), _keyless_model()
    )
    assert identity[0] == keyless_backend_history.HISTORY_KEY
    assert identity[4] == ("mode", "materialized_route")
    assert safe is True
    assert audit is None


def test_plain_materialized_observation_uses_empty_exact_receipt():
    model = _keyless_model()
    options = {}
    identity, safe, _audit = keyless_backend_history._preflight(options, object(), model)
    observed = []
    runtime = SimpleNamespace(
        observe_backend_history=lambda *args: observed.append(args)
    )
    keyless_backend_history._observe(
        runtime,
        3,
        8,
        options,
        (identity, safe),
    )
    assert observed == [(3, 8, identity, (), True)]


def test_opaque_keyless_provider_is_actual_only_without_receipt_policy():
    class Provider:
        api = 1

        def __call__(self, **_kwargs):
            return None

    identity, safe, audit = keyless_backend_history._preflight(
        {"minimax_h3_keyless_provider_v1": Provider()},
        object(),
        _keyless_model(),
    )
    assert identity[0] == keyless_backend_history.HISTORY_KEY
    assert identity[4] == ("mode", "opaque_provider")
    assert safe is False
    assert audit is None


def test_declared_keyless_provider_policy_composes_with_existing_receipts():
    class Provider:
        api = 1

        def __call__(self, **_kwargs):
            return None

    class Policy:
        def __call__(self, **_kwargs):
            return ("keyless-sol", 1)

        def accept_receipts(self, receipts):
            return tuple(receipts) == (("keyless-sol", "exact"),)

    policy = Policy()
    options = {
        "minimax_h3_keyless_provider_v1": Provider(),
        backend_history.POLICIES: {"sol": policy},
        backend_history.RECEIPTS: [("keyless-sol", "exact")],
    }
    identity, safe, audit = keyless_backend_history._preflight(
        options, object(), _keyless_model()
    )
    assert identity[4] == ("mode", "declared_provider")
    assert identity[5][1] == (("sol", ("keyless-sol", 1)),)
    assert safe is True
    assert audit is None

    observed = []
    runtime = SimpleNamespace(
        config=SimpleNamespace(debug=False),
        observe_backend_history=lambda *args: observed.append(args),
    )
    keyless_backend_history._observe(runtime, 2, 5, options, (identity, safe))
    assert observed == [
        (2, 5, identity, (("keyless-sol", "exact"),), True)
    ]


def test_explicit_optimized_attention_override_is_actual_only_without_policy():
    identity, safe, _audit = keyless_backend_history._preflight(
        {"optimized_attention_override": lambda *args, **kwargs: None},
        object(),
        _keyless_model(),
    )
    assert identity[4] == ("mode", "opaque_provider")
    assert safe is False


def test_direct_core_bsa_never_self_qualifies_as_keyless(monkeypatch):
    monkeypatch.setattr(
        keyless_backend_history.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda _options: True,
    )
    identity, safe, _audit = keyless_backend_history._preflight(
        {}, object(), _keyless_model()
    )
    assert identity[4] == ("mode", "core_bsa_requires_reference_bypass")
    assert safe is False


def test_native_qkv_preflight_delegates_unchanged(monkeypatch):
    expected = (("native",), True, "audit")
    monkeypatch.setattr(
        keyless_backend_history,
        "_ORIGINAL_PREFLIGHT",
        lambda options, layout, model: expected,
    )
    assert keyless_backend_history._preflight({}, object(), SimpleNamespace()) == expected


def test_install_order_keeps_keyless_history_under_bsa_wrappers():
    assert keyless_core_bsa_fallback._ORIGINAL_PREFLIGHT is keyless_backend_history._preflight
    assert core_bsa_forecast_recovery._ORIGINAL_OBSERVE is keyless_backend_history._observe
