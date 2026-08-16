from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import external_patch_compat as compat
from comfyui_spectrum_h3 import external_patch_hardening as hardening


def _contract():
    return {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "kind": "text_activation_modulation",
        "architecture": "minimax_h3",
        "instance_id": "diffaid-h3-1",
        "block_indices_0based": [0, 4],
        "model_block_count": 5,
        "strength": 0.2,
        "sigma_start": 0.55,
        "sigma_end": 1.0,
        "sigma_ramp": 0.0,
        "token_weight_mode": "none",
        "token_tail": 0.35,
        "cond_only": True,
        "scope": "native_mod_segments_tag_1_only",
    }


def test_runtime_validation_does_not_mutate_producer_owned_metadata():
    parsed = compat.parse_external_patch_contracts(
        {compat.EXTERNAL_PATCH_CONTRACTS_KEY: [_contract()]},
        block_count=5,
    )
    entry = {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "instance_id": "diffaid-h3-1",
        "normalized_sigma": 0.5,
    }
    transformer_options = {compat.EXTERNAL_PATCH_RUNTIME_KEY: (entry,)}
    before = copy.deepcopy(transformer_options)

    assert compat._runtime_entries(transformer_options, parsed) == pytest.approx((0.5,))
    assert transformer_options == before


def test_malformed_profile_metadata_returns_base_profile_until_runtime_fail_safe(
    monkeypatch,
):
    sentinel = SimpleNamespace(profile="base-profile", cache_hit=False)

    def fail(_model_patcher):
        raise compat.ExternalPatchContractError("malformed declaration")

    monkeypatch.setattr(
        compat,
        "get_model_forecastability_profile_with_external_patches",
        fail,
    )
    monkeypatch.setattr(compat, "_ORIGINAL_PROFILE_LOOKUP", lambda _model: sentinel)

    assert hardening.get_model_forecastability_profile_fail_safe(object()) is sentinel


def test_hardening_install_requires_compat_prerequisite(monkeypatch):
    monkeypatch.setattr(hardening, "_INSTALLED", False)
    monkeypatch.setattr(compat, "_INSTALLED", False)
    monkeypatch.setattr(compat, "_ORIGINAL_PROFILE_LOOKUP", None)

    with pytest.raises(
        RuntimeError,
        match=r"requires install_external_patch_compat\(\) first",
    ):
        hardening.install_external_patch_hardening()


def test_clearing_model_profile_cache_also_clears_external_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hardening,
        "_ORIGINAL_CLEAR_MODEL_PROFILE_CACHE",
        lambda: calls.append("base"),
    )
    monkeypatch.setattr(
        compat,
        "clear_external_profile_cache",
        lambda: calls.append("external"),
    )

    hardening._clear_all_model_profile_caches()
    assert calls == ["base", "external"]
