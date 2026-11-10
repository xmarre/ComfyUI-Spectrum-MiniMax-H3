from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

RES4LYF_SOURCES = (
    "beta/__init__.py",
    "beta/rk_sampler_beta.py",
    "beta/rk_coefficients_beta.py",
    "beta/phi_functions.py",
    "beta/rk_method_beta.py",
    "beta/rk_noise_sampler_beta.py",
    "beta/rk_guide_func_beta.py",
    "helper.py",
)


def _fixture_root() -> Path:
    path = os.environ.get("RES4LYF_PATH")
    if not path:
        pytest.skip("RES4LYF source fixture is unavailable")
    root = Path(path)
    if not root.is_dir():
        pytest.fail(f"RES4LYF_PATH is not a directory: {root}")
    return root


@pytest.mark.parametrize("relative_path", RES4LYF_SOURCES)
def test_res4lyf_fixture_source_is_available_and_parses(
    relative_path: str,
):
    root = _fixture_root()
    source = root / relative_path
    if not source.is_file():
        pytest.fail(f"RES4LYF fixture is missing {relative_path}")

    assert ast.parse(source.read_bytes(), filename=str(source)) is not None


def test_fixture_exercises_committed_crlf_sources():
    root = _fixture_root()

    # Both supported fixtures contain CRLF sources. Live source-code comparisons
    # must accept those line endings as well as LF working-tree copies.
    for relative_path in (
        "beta/__init__.py",
        "beta/rk_coefficients_beta.py",
        "beta/rk_method_beta.py",
        "beta/rk_noise_sampler_beta.py",
        "beta/phi_functions.py",
    ):
        assert b"\r\n" in (root / relative_path).read_bytes()
