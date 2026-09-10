"""Disable automatic BSA shadow probes after CUDA evidence collection.

The transition probe remains available through SPECTRUM_H3_BSA_DIAGNOSTICS, but
production-candidate runs must not silently pay its sparse-attention shadow-call
overhead.  This installer preserves the existing explicit environment override
and only suppresses the audited-safe-CUDA automatic default.
"""
from functools import wraps


def install_bsa_transition_probe_manual_only() -> None:
    from . import bsa_transition_probe as probe

    current = probe._diagnostic_directory
    if getattr(current, "_spectrum_manual_only", False):
        return

    @wraps(current)
    def manual_only(*, automatic):
        return current(automatic=False)

    manual_only._spectrum_manual_only = True
    probe._diagnostic_directory = manual_only
