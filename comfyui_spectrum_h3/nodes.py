from __future__ import annotations

import logging

from .config import SpectrumH3Config
from .minimax_h3 import install_h3_wrapper, require_native_minimax_h3
from .runtime import SpectrumH3Runtime
from .sampling import install_sampler_wrappers

LOG = logging.getLogger(__name__)


def _effective_bootstrap_first_forecast(
    *,
    requested: bool,
    degree: int,
    warmup_steps: int,
) -> bool:
    if not requested or (degree == 1 and warmup_steps <= 1):
        return requested
    LOG.warning(
        "Spectrum H3: bootstrap_first_forecast was enabled but requires "
        "degree=1 and warmup_steps<=1; got degree=%d and warmup_steps=%d. "
        "Disabling bootstrap_first_forecast for this node execution.",
        degree,
        warmup_steps,
    )
    return False


class SpectrumApplyMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_weight": (
                    "FLOAT",
                    {
                        "default": 0.50,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Direct video spectral share. Audio uses the separate audio_blend_weight setting. "
                            "In ordinary single-pass H3, video forecasts can still affect later audio through joint transformer calls."
                        ),
                    },
                ),
                "degree": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Polynomial degree. Values other than 1 disable bootstrap_first_forecast.",
                    },
                ),
                "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
                "window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "flex_window": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 8.0, "step": 0.05}),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Initial native solver steps. Values above 1 disable bootstrap_first_forecast.",
                    },
                ),
                "tail_actual_steps": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
                "max_history": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "history_storage": (
                    ["system_ram", "vram"],
                    {
                        "default": "system_ram",
                        "tooltip": (
                            "Storage for the bounded causal history, capped by max_history. "
                            "Offline replay uses the separate offline_archive_storage setting."
                        ),
                    },
                ),
                "bootstrap_first_forecast": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Forecast solver step 1 from the actual step-0 feature. Requires degree=1 "
                            "and warmup_steps<=1; incompatible settings disable it with a console warning."
                        ),
                    },
                ),
                "anchor_residual_feedback": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental video-scored actual-refresh guard; never injects a hidden residual. "
                            "Disable offline_smoothing_replay before enabling this mode."
                        ),
                    },
                ),
                "selective_rollback_correction": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental thresholded, budgeted rollback for the reviewed deterministic Euler sampler only. "
                            "Disable offline_smoothing_replay before enabling this mode."
                        ),
                    },
                ),
                "offline_smoothing_replay": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Default audio-quality path: capture a local-only trajectory, then apply the "
                            "configured video/audio blends using past and future anchors without causal "
                            "video-to-audio feedback. Uses a second sampler pass and retains every actual anchor. "
                            "Disable only for single-pass comparisons or either research mode."
                        ),
                    },
                ),
                "audio_blend_weight": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Direct audio spectral share. The default 0 prevents spectral mixing of audio rows. "
                            "The default offline replay path also isolates capture from video-to-audio trajectory coupling."
                        ),
                    },
                ),
                "offline_archive_storage": (
                    ["system_ram", "vram"],
                    {
                        "default": "system_ram",
                        "tooltip": (
                            "Storage for every actual anchor retained until offline replay completes. "
                            "This archive is not capped by max_history. Keep system_ram for constrained "
                            "GPUs; vram is an explicit speed/memory tradeoff."
                        ),
                    },
                ),
                "model_aware_mode": (
                    ["off", "schedule", "schedule_confidence", "full"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Experimental model/patch-aware scheduling and confidence. 'schedule' may replace "
                            "risky forecasts with actual evaluations. 'schedule_confidence' also adapts ridge "
                            "regularization, usable degree, and spectral share without applying a correction. "
                            "'full' additionally applies the bounded generic latest-delta residual correction; "
                            "the investigated model-specific Feature-3 correction families did not meet the "
                            "materiality gate and are not applied. No extra denoiser forward is performed. "
                            "Existing workflows remain unchanged when off."
                        ),
                    },
                ),
                "model_aware_risk_threshold": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Advanced threshold for converting a prospective forecast into an actual model "
                            "evaluation. Lower values are more conservative and may spend more NFEs."
                        ),
                    },
                ),
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
        bootstrap_first_forecast=True,
        anchor_residual_feedback=False,
        selective_rollback_correction=False,
        offline_smoothing_replay=True,
        audio_blend_weight=0.0,
        offline_archive_storage="system_ram",
        model_aware_mode="off",
        model_aware_risk_threshold=0.65,
    ):
        if not enabled:
            return (model,)
        require_native_minimax_h3(model)
        resolved_degree = int(degree)
        resolved_warmup_steps = int(warmup_steps)
        if not isinstance(bootstrap_first_forecast, bool):
            raise TypeError("bootstrap_first_forecast must be a boolean")
        for name, value in (
            ("anchor_residual_feedback", anchor_residual_feedback),
            ("selective_rollback_correction", selective_rollback_correction),
            ("offline_smoothing_replay", offline_smoothing_replay),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        effective_bootstrap = _effective_bootstrap_first_forecast(
            requested=bootstrap_first_forecast,
            degree=resolved_degree,
            warmup_steps=resolved_warmup_steps,
        )
        config = SpectrumH3Config(
            enabled=bool(enabled),
            blend_weight=float(blend_weight),
            degree=resolved_degree,
            ridge_lambda=float(ridge_lambda),
            window_size=float(window_size),
            flex_window=float(flex_window),
            warmup_steps=resolved_warmup_steps,
            tail_actual_steps=int(tail_actual_steps),
            max_history=int(max_history),
            history_storage=str(history_storage),
            bootstrap_first_forecast=effective_bootstrap,
            anchor_residual_feedback=anchor_residual_feedback,
            selective_rollback_correction=selective_rollback_correction,
            offline_smoothing_replay=offline_smoothing_replay,
            audio_blend_weight=float(audio_blend_weight),
            offline_archive_storage=str(offline_archive_storage),
            model_aware_mode=str(model_aware_mode),
            model_aware_risk_threshold=float(model_aware_risk_threshold),
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
