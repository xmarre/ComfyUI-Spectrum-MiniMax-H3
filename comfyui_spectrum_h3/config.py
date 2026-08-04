from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpectrumH3Config:
    enabled: bool = True
    blend_weight: float = 0.50
    degree: int = 4
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    tail_actual_steps: int = 1
    max_history: int = 8
    history_storage: str = "system_ram"
    debug: bool = False
    force_actual: bool = False

    @property
    def min_fit_points(self) -> int:
        return max(2, self.degree + 1)

    def validate(self) -> SpectrumH3Config:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.debug, bool):
            raise TypeError("debug must be a boolean")
        if not isinstance(self.force_actual, bool):
            raise TypeError("force_actual must be a boolean")
        if not math.isfinite(self.blend_weight) or not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be finite and in [0, 1]")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 1:
            raise ValueError("degree must be an integer >= 1")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be finite and >= 0")
        if not math.isfinite(self.window_size) or self.window_size < 1.0:
            raise ValueError("window_size must be finite and >= 1")
        if not math.isfinite(self.flex_window) or self.flex_window < 0.0:
            raise ValueError("flex_window must be finite and >= 0")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be an integer >= 0")
        if (
            isinstance(self.tail_actual_steps, bool)
            or not isinstance(self.tail_actual_steps, int)
            or self.tail_actual_steps < 0
        ):
            raise ValueError("tail_actual_steps must be an integer >= 0")
        if (
            isinstance(self.max_history, bool)
            or not isinstance(self.max_history, int)
            or self.max_history < self.min_fit_points
        ):
            raise ValueError(
                f"max_history must be an integer >= {self.min_fit_points} for degree {self.degree}"
            )
        if not isinstance(self.history_storage, str) or self.history_storage not in {
            "system_ram",
            "vram",
        }:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        return self


CONSERVATIVE_PRESET = SpectrumH3Config()
AGGRESSIVE_PRESET = SpectrumH3Config(
    blend_weight=0.75,
    flex_window=3.0,
    tail_actual_steps=1,
)
