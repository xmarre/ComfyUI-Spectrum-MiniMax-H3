from __future__ import annotations

from typing import Any

from .runtime import SpectrumH3Runtime

_ORIGINAL_RUNTIME_START = SpectrumH3Runtime.start_run


def _start_run(self: SpectrumH3Runtime, *args: Any, **kwargs: Any) -> int:
    sampler_name = str(args[1]) if len(args) >= 2 else str(kwargs.get("sampler_name", ""))
    if sampler_name == "sample_er_sde":
        kwargs["min_tail_actual_steps"] = max(
            2,
            int(kwargs.get("min_tail_actual_steps", 0)),
        )
    return _ORIGINAL_RUNTIME_START(self, *args, **kwargs)


def install_er_sde_tail_policy() -> None:
    """Keep ER-SDE's final two logical steps on exact archived features."""
    if getattr(SpectrumH3Runtime, "_er_sde_tail_policy_installed", False):
        return
    SpectrumH3Runtime.start_run = _start_run
    SpectrumH3Runtime._er_sde_tail_policy_installed = True


__all__ = ["install_er_sde_tail_policy"]
