import os

import pytest

from comfyui_spectrum_h3 import (
    core_bsa_compat,
    core_bsa_flow_compat,
    core_bsa_preprocess_compat,
)


def _required() -> bool:
    return os.environ.get("SPECTRUM_REQUIRE_REVIEWED_BSA_FIXTURE") == "1"


def test_required_reviewed_bsa_untwist_and_flow_fixtures_are_real():
    if not _required():
        pytest.skip("reviewed BSA/Untwist/Flow fixture gate is enabled only for the pinned CI lane")

    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"required reviewed core BSA fixture could not import: {exc}")

    bsa_blob = core_bsa_compat._module_blob_sha(nodes)
    assert bsa_blob in core_bsa_compat.AUDITED_BSA_GIT_BLOBS, (
        f"required reviewed core BSA fixture digest mismatch: {bsa_blob}"
    )

    try:
        from flux_untwist import patches
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"required reviewed Untwist fixture could not import: {exc}")

    untwist_blob = core_bsa_compat._module_blob_sha(patches)
    assert untwist_blob in core_bsa_preprocess_compat.AUDITED_UNTWIST_GIT_BLOBS, (
        f"required reviewed Untwist fixture digest mismatch: {untwist_blob}"
    )

    try:
        from h3_flow_regenerate import attention, mixed_grid
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"required reviewed Flow fixture could not import: {exc}")

    attention_blob = core_bsa_compat._module_blob_sha(attention)
    assert attention_blob in core_bsa_flow_compat.AUDITED_FLOW_ATTENTION_GIT_BLOBS, (
        f"required reviewed Flow attention fixture digest mismatch: {attention_blob}"
    )
    mixed_blob = core_bsa_compat._module_blob_sha(mixed_grid)
    assert mixed_blob in core_bsa_flow_compat.AUDITED_FLOW_MIXED_GRID_GIT_BLOBS, (
        f"required reviewed Flow mixed-grid fixture digest mismatch: {mixed_blob}"
    )
