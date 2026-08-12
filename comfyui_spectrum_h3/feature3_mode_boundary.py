from __future__ import annotations

from typing import Any

import torch

from .runtime import SpectrumH3Runtime


def install_feature3_mode_boundary_guard() -> None:
    """Make non-full modes unable to consume Feature-3 head evidence."""
    if getattr(SpectrumH3Runtime, "_feature3_mode_boundary_guard_installed", False):
        return

    original_observe = SpectrumH3Runtime._observe_model_aware_anchor

    def observe(
        runtime: SpectrumH3Runtime,
        step: Any,
        combined: torch.Tensor,
        exact_head_weights: dict[str, torch.Tensor],
        stream_diagonals: dict[str, torch.Tensor],
    ) -> None:
        if runtime.config.model_aware_mode != "full":
            exact_head_weights = {}
            stream_diagonals = {}
        return original_observe(
            runtime,
            step,
            combined,
            exact_head_weights,
            stream_diagonals,
        )

    SpectrumH3Runtime._observe_model_aware_anchor = observe
    SpectrumH3Runtime._feature3_mode_boundary_guard_installed = True


__all__ = ["install_feature3_mode_boundary_guard"]
