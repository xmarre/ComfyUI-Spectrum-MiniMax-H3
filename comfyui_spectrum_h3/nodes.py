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
                "tail_actual_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Requested final native tail. RES enforces its three-step solver tail. "
                            "ER-SDE offline replay promotes only a penultimate step that the normal schedule would forecast, preserving a future exact terminal anchor without a blanket two-step tail."
                        ),
                    },
                ),
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
                            "Compatibility-safe default and historically validated H3 audio/stutter path: capture a local-only trajectory, "
                            "then apply configured blends using past and future anchors without causal video-to-audio feedback. "
                            "It uses a second sampler pass and retains every actual anchor. Current controlled native ER-SDE testing "
                            "favored full single-pass for one recurring temporal facial artifact; replay remains supported."
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
                            "'full' additionally applies the bounded generic latest-delta residual correction. "
                            "Current controlled native ER-SDE testing prefers full single-pass among the compared "
                            "model-aware quality modes. No equivalent conclusion is established for other samplers. "
                            "The correction itself adds no denoiser forward."
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
                "model_aware_trust_shrinkage": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental research/reproduction switch for model_aware_mode='full'. The supported "
                            "default is false and current native ER-SDE perceptual A/B testing does not recommend "
                            "promotion. Offline replay keeps the rejected causal-kappa transfer disabled and uses "
                            "shadow-only diagnostics. No transformer evaluation is added."
                        ),
                    },
                ),
                "model_aware_replay_generic_correction": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Offline replay-only legacy/ablation switch for model_aware_mode='full'. False is "
                            "the supported default: do not transplant the causal PR #39 latest-delta scalar onto "
                            "the different future-bracket replay direction. True explicitly restores that old "
                            "replay transfer for regression/scientific reproduction. The causal PR #39 correction, "
                            "validation attenuation, local/spectral blending, scheduling, and transformer NFE are unchanged."
                        ),
                    },
                ),
                "generic_correction_mode": (
                    [
                        "legacy",
                        "coordinate_rls",
                        "coordinate_rls_reliability",
                        "regional",
                    ],
                    {
                        "default": "legacy",
                        "tooltip": (
                            "Advanced full-mode correction controller. Legacy exactly preserves the validated "
                            "latest-delta EWMA path. The other modes add coordinate transport, recursive "
                            "least-squares gain estimation, correction-specific reliability, and topology-proven "
                            "coarse video temporal regions. Experimental modes add no transformer evaluation."
                        ),
                    },
                ),
                "generic_correction_limiter": (
                    ["rational", "hard_clip", "tanh"],
                    {
                        "default": "rational",
                        "tooltip": (
                            "Advanced generic-correction gain limiter. Rational with limit 0.25 is the exact "
                            "validated baseline. Hard clip and tanh are experimental A/B paths."
                        ),
                    },
                ),
                "generic_correction_limit": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Advanced symmetric gain-limit scale. Keep 0.25 for the validated baseline; "
                            "alternative values are research-only until real A/B validation."
                        ),
                    },
                ),
                # Appended after every pre-existing widget so saved workflow
                # widget arrays keep their historical positional meaning.
                "generic_correction_attenuation": (
                    [
                        "mode_default",
                        "no_attenuation",
                        "general_confidence",
                        "correction_reliability",
                        "combined_conservative",
                    ],
                    {
                        "default": "mode_default",
                        "tooltip": (
                            "Advanced experimental attenuation policy. mode_default exactly preserves each "
                            "existing mode. Explicit policies reproduce evaluator candidates without changing "
                            "legacy production defaults or adding a transformer evaluation."
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
        model_aware_trust_shrinkage=False,
        model_aware_replay_generic_correction=False,
        generic_correction_mode="legacy",
        generic_correction_limiter="rational",
        generic_correction_limit=0.25,
        generic_correction_attenuation="mode_default",
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
            ("model_aware_trust_shrinkage", model_aware_trust_shrinkage),
            (
                "model_aware_replay_generic_correction",
                model_aware_replay_generic_correction,
            ),
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
            model_aware_trust_shrinkage=model_aware_trust_shrinkage,
            model_aware_replay_generic_correction=(
                model_aware_replay_generic_correction
            ),
            generic_correction_mode=str(generic_correction_mode),
            generic_correction_attenuation=str(generic_correction_attenuation),
            generic_correction_limiter=str(generic_correction_limiter),
            generic_correction_limit=float(generic_correction_limit),
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
