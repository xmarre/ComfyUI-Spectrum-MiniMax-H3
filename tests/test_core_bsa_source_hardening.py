from pathlib import Path

import pytest

from comfyui_spectrum_h3 import core_bsa_compat, core_bsa_preprocess_compat


def _audited_untwist_module():
    try:
        from flux_untwist import patches
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Untwist fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(patches)
        not in core_bsa_preprocess_compat.AUDITED_UNTWIST_GIT_BLOBS
    ):
        pytest.skip("Untwist fixture is not the reviewed source")
    return patches


def test_untwist_code_proof_does_not_trust_replaced_factory_binding(monkeypatch):
    patches = _audited_untwist_module()
    original_factory = patches.make_minimax_h3_attention_override
    real_outer = original_factory(None)
    real_transform, _previous = real_outer.attention_preprocess_v1
    assert core_bsa_preprocess_compat._audited_untwist_preprocess(real_transform) is not None

    source_path = str(Path(patches.__file__).resolve())
    malicious_source = """
def make_minimax_h3_attention_override(previous):
    def preprocess(q, k, v, heads, **kwargs):
        return q * 2, k, v
    return preprocess
"""
    namespace = {"__name__": patches.__name__}
    exec(compile(malicious_source, source_path, "exec"), namespace)  # noqa: S102 - intentional synthetic source
    fake_factory = namespace["make_minimax_h3_attention_override"]
    fake_transform = fake_factory(None)

    # The old proof derived its expected nested code from this mutable module
    # binding and could therefore accept a coordinated factory+transform swap.
    monkeypatch.setattr(patches, "make_minimax_h3_attention_override", fake_factory)
    assert fake_transform.__module__ == patches.__name__
    assert fake_transform.__qualname__ == real_transform.__qualname__
    assert Path(fake_transform.__code__.co_filename).resolve() == Path(patches.__file__).resolve()

    assert core_bsa_preprocess_compat._audited_untwist_preprocess(fake_transform) is None
