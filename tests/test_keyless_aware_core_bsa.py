from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import keyless_backend_history, keyless_core_bsa_fallback


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
    setattr(model, "minimax_h3_keyless_contract_v1", Contract())
    return model


def _reviewed_nodes():
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except ModuleNotFoundError as exc:
        if exc.name != "comfy_extras.nodes_sparse_attention":
            raise
        message = "fixture predates comfy_extras.nodes_sparse_attention"
        if os.environ.get("SPECTRUM_REQUIRE_KEYLESS_BSA_FIXTURE") == "1":
            pytest.fail(message)
        pytest.skip(message)

    blob = keyless_core_bsa_fallback.core_bsa_compat._module_blob_sha(nodes)
    if blob not in keyless_core_bsa_fallback.KEYLESS_AWARE_BSA_GIT_BLOBS:
        message = f"fixture is not the reviewed Keyless-aware Core-BSA source: {blob}"
        if os.environ.get("SPECTRUM_REQUIRE_KEYLESS_BSA_FIXTURE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return nodes


def _options(nodes, *, previous=None, block_replacement=None):
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
    override = nodes.make_attention_override(patch, previous)
    patch.installed.add(override)
    options = {
        "optimized_attention_override": override,
        "callbacks": {"prepare": {"block_sparse_attention": object()}},
    }
    if block_replacement is not None:
        options["patches_replace"] = {
            "dit": {("double_block", 0): block_replacement}
        }
    return patch, override, options


def test_reviewed_keyless_aware_bsa_is_preserved_and_not_promoted_forecast_safe():
    nodes = _reviewed_nodes()
    model = _keyless_model()
    patch, _override, options = _options(nodes)
    options["patches_replace"] = {
        "dit": {("double_block", 7): object()}
    }

    proof = keyless_core_bsa_fallback._keyless_aware_bsa_proof(options, model)
    assert proof is not None
    assert proof.patch is patch
    assert proof.source_blob in keyless_core_bsa_fallback.KEYLESS_AWARE_BSA_GIT_BLOBS

    prepared, identity = keyless_core_bsa_fallback.prepare_reference_options(
        options,
        model,
    )
    assert prepared is options
    assert identity is None
    assert prepared["optimized_attention_override"] is options["optimized_attention_override"]
    assert prepared["patches_replace"]["dit"][("double_block", 7)] is options[
        "patches_replace"
    ]["dit"][("double_block", 7)]

    history_identity, safe, _audit = keyless_backend_history._preflight(
        options,
        None,
        model,
    )
    assert safe is False
    assert history_identity[0] == keyless_backend_history.HISTORY_KEY


def test_keyless_aware_bsa_with_unknown_inherited_attention_owner_fails_closed():
    nodes = _reviewed_nodes()
    model = _keyless_model()

    def opaque(*args, **kwargs):
        return None

    _patch, _override, options = _options(nodes, previous=opaque)
    assert keyless_core_bsa_fallback._keyless_aware_bsa_proof(options, model) is None

    with pytest.raises(RuntimeError, match="Refusing to execute or silently discard"):
        keyless_core_bsa_fallback.prepare_reference_options(options, model)


def test_keyless_aware_source_rejects_native_bsa_block_replacement():
    nodes = _reviewed_nodes()
    model = _keyless_model()
    patch, _override, options = _options(nodes)
    replacement = nodes.make_h3_block_patch(model.blocks[0], 0, patch)
    options["patches_replace"] = {"dit": {("double_block", 0): replacement}}

    assert keyless_core_bsa_fallback._keyless_aware_bsa_proof(options, model) is None
    with pytest.raises(RuntimeError, match="Refusing to execute or silently discard"):
        keyless_core_bsa_fallback.prepare_reference_options(options, model)


def test_keyless_aware_bsa_source_drift_is_not_accepted(monkeypatch):
    nodes = _reviewed_nodes()
    model = _keyless_model()
    _patch, _override, options = _options(nodes)
    monkeypatch.setattr(
        keyless_core_bsa_fallback,
        "KEYLESS_AWARE_BSA_GIT_BLOBS",
        frozenset(),
    )

    assert keyless_core_bsa_fallback._keyless_aware_bsa_proof(options, model) is None
    with pytest.raises(RuntimeError, match="Refusing to execute or silently discard"):
        keyless_core_bsa_fallback.prepare_reference_options(options, model)


def test_real_comfy_loader_alias_of_reviewed_source_is_recognized(monkeypatch):
    nodes = _reviewed_nodes()
    source = Path(nodes.__file__).resolve()
    alias_name = "nodes_sparse_attention_keyless_fixture"
    spec = importlib.util.spec_from_file_location(alias_name, source)
    assert spec is not None and spec.loader is not None
    alias = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias_name, alias)
    spec.loader.exec_module(alias)

    model = _keyless_model()
    _patch, _override, options = _options(alias)
    proof = keyless_core_bsa_fallback._keyless_aware_bsa_proof(options, model)

    assert proof is not None
    assert proof.module is alias
    assert proof.source_blob in keyless_core_bsa_fallback.KEYLESS_AWARE_BSA_GIT_BLOBS
