from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3.core_bsa_compat import _module_blob_sha
from comfyui_spectrum_h3.sampling import RES4LYF_AUDITED_GIT_BLOBS


RES4LYF_REVIEWED_SOURCES = {
    "wrappers": "beta/__init__.py",
    "sampler": "beta/rk_sampler_beta.py",
    "coefficients": "beta/rk_coefficients_beta.py",
    "method": "beta/rk_method_beta.py",
    "noise_sampler": "beta/rk_noise_sampler_beta.py",
    "guide": "beta/rk_guide_func_beta.py",
    "helper": "helper.py",
}


def _fixture_root() -> Path:
    path = os.environ.get("RES4LYF_PATH")
    if not path:
        pytest.skip("reviewed RES4LYF source fixture is unavailable")
    root = Path(path)
    if not root.is_dir():
        pytest.fail(f"RES4LYF_PATH is not a directory: {root}")
    return root


@pytest.mark.parametrize(("source_key", "relative_path"), RES4LYF_REVIEWED_SOURCES.items())
def test_reviewed_res4lyf_source_hash_matches_runtime_normalization(
    source_key: str,
    relative_path: str,
):
    root = _fixture_root()
    source = root / relative_path
    if not source.is_file():
        pytest.fail(f"reviewed RES4LYF fixture is missing {relative_path}")

    normalized_blob = _module_blob_sha(SimpleNamespace(__file__=str(source)))

    assert normalized_blob in RES4LYF_AUDITED_GIT_BLOBS[source_key]


def test_reviewed_fixture_exercises_committed_crlf_sources():
    root = _fixture_root()

    # These reviewed upstream files are committed with CRLF. This is the exact
    # case that originally made raw GitHub blob IDs disagree with
    # _module_blob_sha(), which intentionally normalizes CRLF to LF.
    for relative_path in (
        "beta/__init__.py",
        "beta/rk_coefficients_beta.py",
        "beta/rk_method_beta.py",
        "beta/rk_noise_sampler_beta.py",
    ):
        assert b"\r\n" in (root / relative_path).read_bytes()
