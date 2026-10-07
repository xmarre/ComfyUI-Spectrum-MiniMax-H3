"""Require installed H3 interoperability fixtures without source-revision allowlists."""
import os

import pytest

from comfyui_spectrum_h3 import core_bsa_compat


def test_required_core_bsa_untwist_flow_runtime_fixtures():
    if os.environ.get("SPECTRUM_REQUIRE_REVIEWED_BSA_FIXTURE") != "1":
        pytest.skip("required fixture lane only")
    try:
        import comfy_extras.nodes_sparse_attention as nodes
        from flux_untwist import patches
        from h3_flow_regenerate import attention, mixed_grid
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"required H3 interoperability fixture unavailable: {exc}")

    module, source_id = core_bsa_compat._load_audited_module()
    assert module is nodes
    assert isinstance(source_id, str) and source_id
    assert callable(patches.make_minimax_h3_attention_override)
    assert callable(attention.make_layout_block_wrapper)
    assert callable(attention.mark_layout_wrapper)
    assert callable(mixed_grid.mixed_diffusion_wrapper)
    assert callable(mixed_grid.MixedGridPlan)
