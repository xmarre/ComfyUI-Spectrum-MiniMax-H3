from __future__ import annotations

from types import SimpleNamespace

from comfyui_spectrum_h3 import (
    keyless_compat,
    keyless_core_bsa_fallback,
    keyless_runtime_compat,
    keyless_untwist_compat,
)
from flux_untwist.keyless_h3 import make_keyless_untwist_routing_preprocessor


_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"
_PREPROCESSORS_KEY = "minimax_h3_keyless_routing_preprocessors_v1"
_PROVIDER = "comfyui-flux2-untwisting-rope"


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
    provenance_identity = "untwist-route-test"

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


def _model():
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


def _cfg(*, progress: float, ranges=((2, 4),), high=0.95):
    return {
        "enabled": True,
        "reference_ranges": [list(item) for item in ranges],
        "reference_scope": "image_and_video",
        "rope_axis_count": 3,
        "rope_freqs_per_axis": 16,
        "high_scale_start": high,
        "high_scale_end": 1.0,
        "low_scale_start": 1.0,
        "low_scale_end": 1.05,
        "beta": 2.0,
        "start_percent": 0.0,
        "end_percent": 0.9,
        "progress": progress,
        "scale_temporal_axis": False,
    }


def _options(preprocessor, *, progress: float, instance_id="untwist-keyless-1"):
    return {
        _PREPROCESSORS_KEY: (preprocessor,),
        _RUNTIME_KEY: (
            {
                "schema_version": 2,
                "provider": _PROVIDER,
                "instance_id": instance_id,
                "schedule_progress": progress,
                "active": True,
            },
        ),
    }


def _preprocessor(*, progress: float, ranges=((2, 4),), high=0.95):
    return make_keyless_untwist_routing_preprocessor(
        _cfg(progress=progress, ranges=ranges, high=high),
        instance_id="untwist-keyless-1",
        expected_rows=12,
    )


def test_reviewed_keyless_untwist_source_is_exactly_pinned():
    preprocessor = _preprocessor(progress=0.25)
    identity = keyless_untwist_compat.reviewed_keyless_untwist_identity(
        preprocessor,
        _options(preprocessor, progress=0.25),
    )
    assert identity is not None
    assert identity[0] == "reviewed_keyless_untwist_route_v1"
    assert identity[1] in keyless_untwist_compat.AUDITED_UNTWIST_KEYLESS_GIT_BLOBS


def test_reviewed_untwist_progress_changes_do_not_reset_keyless_backend_history_identity():
    early = _preprocessor(progress=0.25)
    late = _preprocessor(progress=0.50)
    assert early is not late
    assert early.identity != late.identity

    early_identity = keyless_runtime_compat._keyless_runtime_identity(
        _options(early, progress=0.25)
    )
    late_identity = keyless_runtime_compat._keyless_runtime_identity(
        _options(late, progress=0.50)
    )
    assert early_identity == late_identity


def test_reviewed_untwist_static_route_semantics_remain_history_bearing():
    base = _preprocessor(progress=0.25)
    changed_ranges = _preprocessor(progress=0.25, ranges=((3, 5),))
    changed_scale = _preprocessor(progress=0.25, high=0.90)

    base_identity = keyless_runtime_compat._keyless_runtime_identity(
        _options(base, progress=0.25)
    )
    assert base_identity != keyless_runtime_compat._keyless_runtime_identity(
        _options(changed_ranges, progress=0.25)
    )
    assert base_identity != keyless_runtime_compat._keyless_runtime_identity(
        _options(changed_scale, progress=0.25)
    )


def test_runtime_progress_mismatch_refuses_reviewed_stable_identity():
    preprocessor = _preprocessor(progress=0.25)
    mismatched = _options(preprocessor, progress=0.50)
    assert keyless_untwist_compat.reviewed_keyless_untwist_identity(
        preprocessor,
        mismatched,
    ) is None
    generic = keyless_runtime_compat._preprocessor_identity(preprocessor, mismatched)
    assert generic[0] == preprocessor.identity


def test_declared_digest_must_match_reviewed_snapshot(monkeypatch):
    preprocessor = _preprocessor(progress=0.25)
    monkeypatch.setattr(
        preprocessor,
        "identity",
        "minimax_h3_untwist_keyless_route_v1:" + "0" * 64,
    )
    options = _options(preprocessor, progress=0.25)
    assert keyless_untwist_compat.reviewed_keyless_untwist_identity(
        preprocessor,
        options,
    ) is None
    generic = keyless_runtime_compat._preprocessor_identity(preprocessor, options)
    assert generic[0] == preprocessor.identity


def test_missing_runtime_descriptor_refuses_reviewed_stable_identity():
    preprocessor = _preprocessor(progress=0.25)
    assert keyless_untwist_compat.reviewed_keyless_untwist_identity(
        preprocessor,
        {_PREPROCESSORS_KEY: (preprocessor,)},
    ) is None


def test_core_bsa_reference_fallback_preserves_keyless_route_preprocessor(monkeypatch):
    model = _model()
    preprocessor = _preprocessor(progress=0.25)
    dit = {("double_block", index): object() for index in range(50)}
    options = {
        "patches_replace": {"dit": dit},
        "optimized_attention_override": object(),
        _PREPROCESSORS_KEY: (preprocessor,),
        _RUNTIME_KEY: _options(preprocessor, progress=0.25)[_RUNTIME_KEY],
    }
    proof = keyless_core_bsa_fallback.CoreBSAReferenceProof(
        module=object(),
        source_blob="0" * 40,
        patch=object(),
        settings_identity=(False, 1.3, 0.0, 0, 1.0, 0.0, 12288, (), "exact_kv"),
        patch_generation=7,
    )
    monkeypatch.setattr(
        keyless_core_bsa_fallback.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda _options: True,
    )
    monkeypatch.setattr(
        keyless_core_bsa_fallback,
        "_resolve_direct_core_bsa",
        lambda _options, _model: proof,
    )

    prepared, bypass_identity = keyless_core_bsa_fallback.prepare_reference_options(
        options,
        model,
    )

    assert bypass_identity[:-1] == proof.identity(model)
    assert bypass_identity[-1][0] == "keyless_runtime_identity"
    assert prepared[_PREPROCESSORS_KEY] == (preprocessor,)
    assert prepared[_RUNTIME_KEY] == options[_RUNTIME_KEY]
    assert "optimized_attention_override" not in prepared
    assert prepared["patches_replace"]["dit"] == {}
    assert options[_PREPROCESSORS_KEY] == (preprocessor,)


def test_bsa_bypass_rejects_replaced_legacy_untwist_override():
    model = _model()
    proof = keyless_core_bsa_fallback.CoreBSAReferenceProof(
        module=object(),
        source_blob="0" * 40,
        patch=object(),
        settings_identity=(False, 1.3, 0.0, 0, 1.0, 0.0, 12288, (), "exact_kv"),
        patch_generation=7,
        preprocess_identity=(("legacy-untwist",), ("runtime",)),
    )
    options = {"optimized_attention_override": object()}
    identity = keyless_core_bsa_fallback._route_bound_bypass_identity(
        proof,
        model,
        options,
    )
    options[keyless_core_bsa_fallback.BYPASS_KEY] = identity

    result_identity, safe, audit = keyless_core_bsa_fallback._preflight(
        options,
        None,
        model,
    )

    assert result_identity[0] == keyless_core_bsa_fallback.BYPASS_KEY
    assert result_identity[1] == "invalid_or_opaque_bypass"
    assert safe is False
    assert audit is None


def test_unreviewed_duck_preprocessor_keeps_generic_lifetime_identity():
    class Preprocessor:
        identity = "minimax_h3_untwist_keyless_route_v1:" + "a" * 64

        def __call__(self, route):
            return route

    first = Preprocessor()
    second = Preprocessor()
    first_identity = keyless_runtime_compat._preprocessor_identity(first, {})
    second_identity = keyless_runtime_compat._preprocessor_identity(second, {})
    assert first_identity != second_identity
