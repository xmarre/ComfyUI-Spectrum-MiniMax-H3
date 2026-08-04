from __future__ import annotations

from .config import SpectrumH3Config
from .minimax_h3 import install_h3_wrapper, require_native_minimax_h3
from .runtime import SpectrumH3Runtime
from .sampling import install_sampler_wrappers


class SpectrumApplyMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "degree": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
                "tail_actual_steps": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "max_history": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "history_storage": (["system_ram", "vram"], {"default": "system_ram"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(
        self,
        model,
        enabled,
        blend_weight,
        degree,
        ridge_lambda,
        window_size,
        flex_window,
        warmup_steps,
        tail_actual_steps,
        max_history,
        debug,
        history_storage="system_ram",
    ):
        if not enabled:
            return (model,)
        require_native_minimax_h3(model)
        config = SpectrumH3Config(
            enabled=bool(enabled),
            blend_weight=float(blend_weight),
            degree=int(degree),
            ridge_lambda=float(ridge_lambda),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=int(warmup_steps),
            tail_actual_steps=int(tail_actual_steps),
            max_history=int(max_history),
            history_storage=str(history_storage),
            debug=bool(debug),
        ).validate()
        patched = model.clone()
        require_native_minimax_h3(patched)
        runtime = SpectrumH3Runtime(config)
        install_sampler_wrappers(patched, runtime)
        install_h3_wrapper(patched)
        return (patched,)


NODE_CLASS_MAPPINGS = {"SpectrumApplyMiniMaxH3": SpectrumApplyMiniMaxH3}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumApplyMiniMaxH3": "Spectrum Apply MiniMax H3"}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SpectrumApplyMiniMaxH3",
]
