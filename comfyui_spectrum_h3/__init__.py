from .config import AGGRESSIVE_PRESET, CONSERVATIVE_PRESET, SpectrumH3Config
from .er_sde_policy import install_er_sde_tail_policy
from .forecast import HistoryWeightForecaster
from .generic_correction import install_generic_residual_correction
from .minimax_h3 import locate_minimax_h3_inner, require_native_minimax_h3
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .objective_media_nodes import SpectrumH3ObjectiveSequentialCapture
from .replay_calibration import install_replay_calibration
from .replay_calibration_provenance import install_replay_calibration_provenance
from .replay_calibration_validation import install_replay_calibration_validation
from .replay_component_shadow import install_replay_component_decomposition
from .replay_generic_correction_gate import install_replay_generic_correction_gate
from .replay_shadow_composition import install_replay_shadow_composition
from .replay_spectral_alpha_shadow import install_replay_spectral_alpha_shadow
from .replay_spectral_mixture_shadow import install_replay_spectral_mixture_shadow
from .replay_trust_shadow import install_replay_native_trust_shadow
from .runtime import SpectrumH3Runtime
from .trust_probe import install_forecast_trust_probe


class _SpectrumH3ObjectiveSequentialCaptureLinkedSeed(
    SpectrumH3ObjectiveSequentialCapture
):
    """Expose benchmark seed as a normal linkable INT without seed randomization."""

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        required["generation_seed"] = (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "step": 1,
                "tooltip": (
                    "Actual generation seed. Connect the same fixed INT seed output used by the generation workflow. "
                    "This input deliberately has no control_after_generate randomizer."
                ),
            },
        )
        return {**schema, "required": required}


# The recommended sequential benchmark must accept the exact seed already driving
# the generation graph. Current ComfyUI exposes a socket alongside ordinary INT
# widgets; omitting control_after_generate keeps the local fallback fixed while
# still allowing a direct INT link from the user's seed node.
NODE_CLASS_MAPPINGS[
    "SpectrumH3ObjectiveSequentialCapture"
] = _SpectrumH3ObjectiveSequentialCaptureLinkedSeed

install_generic_residual_correction()
install_forecast_trust_probe()
install_replay_native_trust_shadow()
install_replay_component_decomposition()
install_replay_shadow_composition()
install_replay_spectral_mixture_shadow()
install_replay_spectral_alpha_shadow()
install_replay_generic_correction_gate()
install_replay_calibration()
install_replay_calibration_validation()
install_replay_calibration_provenance()
install_er_sde_tail_policy()

__all__ = [
    "AGGRESSIVE_PRESET",
    "CONSERVATIVE_PRESET",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "HistoryWeightForecaster",
    "SpectrumH3Config",
    "SpectrumH3Runtime",
    "locate_minimax_h3_inner",
    "require_native_minimax_h3",
]
