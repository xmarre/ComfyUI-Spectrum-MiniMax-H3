from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import (
    core_bsa_forecast_recovery,
    core_bsa_preprocess_compat,
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


def _proof():
    return keyless_core_bsa_fallback.CoreBSAReferenceProof(
        module=object(),
        source_blob="0" * 40,
        patch=object(),
        settings_identity=(False, 1.3, 0.0, 0, 1.0, 0.0, 12288, (), "exact_kv"),
        patch_generation=7,
    )


def _direct_bsa_options():
    dit = {
        ("double_block", index): object()
        for index in range(50)
    }
    dit[("foreign", 3)] = "keep-me"
    return {
        "patches_replace": {"dit": dit, "other": {"x": "keep"}},
        "optimized_attention_override": object(),
        "callbacks": {"prepare": {"block_sparse_attention": object()}},
        "unrelated": "keep",
    }


def _reviewed_bsa_options(model):
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core BSA is unavailable in this reviewed ComfyUI fixture: {exc}")
    blob = keyless_core_bsa_fallback.core_bsa_compat._module_blob_sha(nodes)
    if blob not in keyless_core_bsa_fallback.core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        pytest.skip("this ComfyUI fixture is not the reviewed core BSA source")
    patch = nodes.SparseAttnPatch(
        tau=1.3,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=0,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    dit = {
        ("double_block", index): nodes.make_h3_block_patch(block, index, patch)
        for index, block in enumerate(model.blocks)
    }
    return nodes, patch, {
        "patches_replace": {"dit": dit},
        "optimized_attention_override": override,
        "callbacks": {"prepare": {"block_sparse_attention": object()}},
    }


def _reviewed_untwist_factory():
    try:
        from flux_untwist import patches
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Untwist fixture is unavailable: {exc}")
    blob = keyless_core_bsa_fallback.core_bsa_compat._module_blob_sha(patches)
    if blob not in core_bsa_preprocess_compat.AUDITED_UNTWIST_GIT_BLOBS:
        pytest.skip("this Untwist fixture is not the reviewed v0.2.4 source")
    return patches.make_minimax_h3_attention_override


def _with_reviewed_untwist(options, inherited, *, instance_id="untwist-keyless-1"):
    outer = _reviewed_untwist_factory()(inherited)
    out = dict(options)
    out["optimized_attention_override"] = outer
    out["minimax_h3_untwist_rope"] = {
        "enabled": True,
        "progress": 0.5,
        "start_percent": 0.0,
        "end_percent": 1.0,
        "high_scale_start": 0.5,
        "high_scale_end": 0.5,
        "low_scale_start": 1.0,
        "low_scale_end": 1.0,
        "beta": 2.0,
        "rope_axis_count": 3,
        "rope_freqs_per_axis": 1,
        "reference_ranges": [(0, 2)],
    }
    out["spectrum_h3_visual_reference_patch_runtime"] = (
        {
            "schema_version": 2,
            "provider": "comfyui-flux2-untwisting-rope",
            "instance_id": instance_id,
            "schedule_progress": 0.5,
            "active": True,
        },
    )
    return out, outer


def test_direct_core_bsa_is_removed_only_from_local_keyless_options(monkeypatch):
    model = _keyless_model()
    options = _direct_bsa_options()
    monkeypatch.setattr(
        keyless_core_bsa_fallback.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda value: True,
    )
    monkeypatch.setattr(
        keyless_core_bsa_fallback,
        "_resolve_direct_core_bsa",
        lambda value, inner: _proof(),
    )

    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(options, model)
    assert prepared is not options
    assert options["optimized_attention_override"] is not None
    assert len(options["patches_replace"]["dit"]) == 51
    assert "optimized_attention_override" not in prepared
    assert prepared["patches_replace"]["dit"] == {("foreign", 3): "keep-me"}
    assert prepared["patches_replace"]["other"] == {"x": "keep"}
    assert prepared["callbacks"] is options["callbacks"]
    assert prepared["unrelated"] == "keep"
    assert prepared[keyless_core_bsa_fallback.BYPASS_KEY] == identity
    assert identity[0] == keyless_core_bsa_fallback.BYPASS_KEY
    assert identity[-1] == "dense_materialized_route"


def test_reviewed_core_bsa_ownership_proof_never_requires_fake_keyless_qkv():
    model = _keyless_model()
    _nodes, _patch, options = _reviewed_bsa_options(model)
    assert all(not hasattr(block.attn, "qkv_proj") for block in model.blocks)

    proof = keyless_core_bsa_fallback._resolve_direct_core_bsa(options, model)
    assert proof is not None
    assert proof.source_blob in keyless_core_bsa_fallback.core_bsa_compat.AUDITED_BSA_GIT_BLOBS

    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(options, model)
    assert identity == proof.identity(model)
    assert "optimized_attention_override" not in prepared
    assert prepared["patches_replace"]["dit"] == {}
    assert all(not hasattr(block.attn, "qkv_proj") for block in model.blocks)


def test_reviewed_outer_untwist_is_rebuilt_without_bsa_and_keeps_raw_v():
    model = _keyless_model()
    _nodes, _patch, options = _reviewed_bsa_options(model)
    bsa_override = options["optimized_attention_override"]
    options, outer = _with_reviewed_untwist(options, bsa_override)

    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(options, model)
    restored = prepared["optimized_attention_override"]
    assert restored is not outer
    assert restored is not bsa_override
    transform, previous = restored.attention_preprocess_v1
    assert previous is None
    assert core_bsa_preprocess_compat._audited_untwist_preprocess(transform) is not None
    assert identity[-2][0] == "outer_preprocess"
    assert identity[-2][1] is not None

    q = torch.ones(1, 2, 4, 128)
    route = torch.ones_like(q)
    raw_v = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)

    def original(q_in, route_in, v_in, _heads, **_kwargs):
        return q_in, route_in, v_in

    q_out, route_out, v_out = restored(
        original,
        q,
        route,
        raw_v,
        2,
        transformer_options=prepared,
    )
    assert torch.equal(q_out, q)
    assert not torch.equal(route_out[:, :, :2, :], route[:, :, :2, :])
    assert torch.equal(route_out[:, :, 2:, :], route[:, :, 2:, :])
    assert torch.equal(v_out, raw_v)


def test_reviewed_untwist_below_bsa_is_restored_directly():
    model = _keyless_model()
    nodes, patch, options = _reviewed_bsa_options(model)
    options, untwist = _with_reviewed_untwist(options, None, instance_id="untwist-below")
    bsa_override = nodes.make_attention_override(patch, untwist)
    patch.installed.add(bsa_override)
    options["optimized_attention_override"] = bsa_override

    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(options, model)
    assert prepared["optimized_attention_override"] is untwist
    assert prepared["patches_replace"]["dit"] == {}
    assert identity[-2][0] == "outer_preprocess"
    assert identity[-2][1] is not None


def test_unknown_inherited_override_is_not_silently_discarded():
    model = _keyless_model()
    nodes, patch, options = _reviewed_bsa_options(model)

    def opaque(*args, **kwargs):
        return None

    bsa_override = nodes.make_attention_override(patch, opaque)
    patch.installed.add(bsa_override)
    options["optimized_attention_override"] = bsa_override
    with pytest.raises(RuntimeError, match="silently discard an opaque"):
        keyless_core_bsa_fallback.prepare_reference_options(options, model)


def test_opaque_keyless_core_bsa_fails_before_qkv_execution(monkeypatch):
    model = _keyless_model()
    options = _direct_bsa_options()
    monkeypatch.setattr(
        keyless_core_bsa_fallback.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda value: True,
    )
    monkeypatch.setattr(
        keyless_core_bsa_fallback,
        "_resolve_direct_core_bsa",
        lambda value, inner: None,
    )
    with pytest.raises(RuntimeError, match="Refusing to execute or silently discard"):
        keyless_core_bsa_fallback.prepare_reference_options(options, model)


def test_native_qkv_options_are_not_touched(monkeypatch):
    options = _direct_bsa_options()
    native = SimpleNamespace()
    monkeypatch.setattr(
        keyless_core_bsa_fallback.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda value: True,
    )
    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(options, native)
    assert prepared is options
    assert identity is None


def test_bypass_marker_is_forecast_safe_without_core_bsa_receipts(monkeypatch):
    identity = (keyless_core_bsa_fallback.BYPASS_KEY, 1, "semantic")
    original_called = False

    def original(*args, **kwargs):
        nonlocal original_called
        original_called = True
        return None

    monkeypatch.setattr(keyless_core_bsa_fallback, "_ORIGINAL_PREFLIGHT", original)
    result = keyless_core_bsa_fallback._preflight(
        {keyless_core_bsa_fallback.BYPASS_KEY: identity}, None, None
    )
    assert result == (identity, True, None)
    assert original_called is False

    observed = []
    runtime = SimpleNamespace(
        observe_backend_history=lambda *args: observed.append(args)
    )
    monkeypatch.setattr(keyless_core_bsa_fallback, "_ORIGINAL_OBSERVE", original)
    keyless_core_bsa_fallback._observe(
        runtime,
        4,
        9,
        {keyless_core_bsa_fallback.BYPASS_KEY: identity},
        (identity, True),
    )
    assert observed == [(4, 9, identity, (), True)]
    assert original_called is False


def test_prepare_strips_bsa_before_delegating_to_history_stack(monkeypatch):
    model = _keyless_model()
    options = _direct_bsa_options()
    monkeypatch.setattr(
        keyless_core_bsa_fallback.core_bsa_compat,
        "has_core_bsa_evidence",
        lambda value: True,
    )
    monkeypatch.setattr(
        keyless_core_bsa_fallback,
        "_resolve_direct_core_bsa",
        lambda value, inner: _proof(),
    )
    delegated = []

    def original(runtime, run_id, step_id, prepared, layout, inner):
        delegated.append(prepared)
        return prepared, (prepared[keyless_core_bsa_fallback.BYPASS_KEY], True)

    monkeypatch.setattr(keyless_core_bsa_fallback, "_ORIGINAL_PREPARE", original)
    prepared, policy = keyless_core_bsa_fallback._prepare(
        object(), 1, 2, options, object(), model
    )
    assert delegated == [prepared]
    assert "optimized_attention_override" not in prepared
    assert not any(
        key[0] == "double_block"
        for key in prepared["patches_replace"]["dit"]
    )
    assert policy[0] == prepared[keyless_core_bsa_fallback.BYPASS_KEY]


def test_installation_wraps_final_core_bsa_recovery_prepare():
    # This ordering is an invariant: recovery installs a custom history.prepare
    # implementation, so Keyless stripping must wrap it rather than be hidden by it.
    assert keyless_core_bsa_fallback._ORIGINAL_PREPARE is core_bsa_forecast_recovery._prepare
