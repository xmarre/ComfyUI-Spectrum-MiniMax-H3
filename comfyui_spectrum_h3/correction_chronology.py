from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .runtime import SpectrumH3Runtime

LOG = logging.getLogger(__name__)


def _uses_generic_scalar_correction(decision: Any) -> bool:
    return bool(
        float(decision.audio_correction_gain) != 0.0
        or float(decision.video_correction_gain) != 0.0
    )


def install_correction_chronology_fix() -> None:
    """Bind each retained scalar replay gain to its causal first-pass anchor pair."""
    if getattr(SpectrumH3Runtime, "_correction_chronology_fix_installed", False):
        return

    original_begin_step = SpectrumH3Runtime.begin_step

    def begin_step(self: SpectrumH3Runtime, timestep: Any) -> dict[str, Any]:
        result = original_begin_step(self, timestep)
        step = self._step
        if (
            step is None
            or step.mode != "forecast"
            or self.config.model_aware_mode != "full"
            or step.model_aware_decision is None
            or not _uses_generic_scalar_correction(step.model_aware_decision)
        ):
            return result

        anchor_ids = self.forecaster.latest_anchor_ids(2)
        if len(anchor_ids) != 2:
            # A scalar gain is only meaningful when replay can apply it to the
            # same causal latest-delta stencil that the first-pass forecast
            # would have used. Disable model-aware behavior rather than let
            # replay reinterpret the gain against a future bracketing anchor.
            reason = "generic scalar correction is missing two causal actual anchor IDs"
            self._model_aware_disabled_reason = reason
            self.stats.model_aware_failures += 1
            step.model_aware_decision = None
            LOG.warning("Spectrum H3 model-aware forecasting disabled for this run: %s", reason)
            return result

        step.model_aware_decision = replace(
            step.model_aware_decision,
            correction_anchor_ids=anchor_ids,
        )
        return result

    SpectrumH3Runtime.begin_step = begin_step
    SpectrumH3Runtime._correction_chronology_fix_installed = True


__all__ = ["install_correction_chronology_fix"]
