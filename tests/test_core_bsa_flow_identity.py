from types import SimpleNamespace

from comfyui_spectrum_h3 import core_bsa_preprocess_compat


def _raw_flow_identity(generation: int, *, mixed_signature=("mixed", 160)):
    carrier = (
        ("signature", (64, 1, 8, 8, 16)),
        ("segments", ((0, 96, "conditioning"), (96, 128, "video"))),
    )
    mixed = (
        ("signature", mixed_signature),
        ("segments", ((0, 96, "conditioning"), (96, 160, "video"))),
    )
    return (
        "reviewed_flow_h3_wrapper_chain_v1",
        "mixed_grid_v0.3.3",
        (
            (
                (
                    "flow_mixed_grid_wrapper",
                    "mixed-blob",
                    0,
                    ("module", "qualname", generation),
                    generation + 10,
                    generation + 20,
                    carrier,
                    mixed,
                ),
                (
                    "flow_layout_wrapper",
                    "attention-blob",
                    "layout",
                    0,
                    ("module", "qualname", generation + 30),
                    generation + 40,
                ),
            ),
        ),
        ("carrier_layout", carrier),
        ("mixed_layout", (generation + 10, generation + 20, mixed)),
    )


def _audit(raw):
    return SimpleNamespace(
        flow_identity=raw,
        identity=(
            ("core", "stable"),
            ("outer_block_wrappers", raw),
        ),
    )


def test_recreated_dynamic_flow_wrappers_keep_same_backend_identity():
    first = _audit(_raw_flow_identity(100))
    second = _audit(_raw_flow_identity(900))

    assert core_bsa_preprocess_compat._stabilize_flow_audit_identity(first)
    assert core_bsa_preprocess_compat._stabilize_flow_audit_identity(second)
    assert first.identity == second.identity
    assert first.flow_identity == second.flow_identity


def test_structural_mixed_layout_change_still_changes_backend_identity():
    first = _audit(_raw_flow_identity(100, mixed_signature=("mixed", 160)))
    second = _audit(_raw_flow_identity(900, mixed_signature=("mixed", 192)))

    assert core_bsa_preprocess_compat._stabilize_flow_audit_identity(first)
    assert core_bsa_preprocess_compat._stabilize_flow_audit_identity(second)
    assert first.identity != second.identity


def test_malformed_flow_identity_fails_closed():
    audit = SimpleNamespace(
        flow_identity=("reviewed_flow_h3_wrapper_chain_v1", "broken"),
        identity=(("outer_block_wrappers", "broken"),),
    )
    assert not core_bsa_preprocess_compat._stabilize_flow_audit_identity(audit)
