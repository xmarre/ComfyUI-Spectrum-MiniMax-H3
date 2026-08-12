from __future__ import annotations

from typing import Any

from . import sampling as _sampling

_ORIGINAL_MIN_TAIL_ACTUAL_STEPS = _sampling.min_tail_actual_steps


def _min_tail_actual_steps(sampler: Any) -> int:
    """Protect ER-SDE from a penultimate forecast caused by schedule parity."""
    baseline = int(_ORIGINAL_MIN_TAIL_ACTUAL_STEPS(sampler))
    if _sampling.sampler_name(sampler) in _sampling.ER_SDE_SAMPLERS:
        return max(2, baseline)
    return baseline


def install_er_sde_tail_policy() -> None:
    if getattr(_sampling, "_er_sde_tail_policy_installed", False):
        return
    _sampling.min_tail_actual_steps = _min_tail_actual_steps
    _sampling._er_sde_tail_policy_installed = True


__all__ = ["install_er_sde_tail_policy"]
