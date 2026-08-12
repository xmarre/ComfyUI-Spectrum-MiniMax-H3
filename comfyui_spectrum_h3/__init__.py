from .config import AGGRESSIVE_PRESET, CONSERVATIVE_PRESET, SpectrumH3Config
from .correction_chronology import install_correction_chronology_fix
from .forecast import HistoryWeightForecaster
from .generic_correction import install_generic_residual_correction
from .minimax_h3 import locate_minimax_h3_inner, require_native_minimax_h3
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .runtime import SpectrumH3Runtime

install_generic_residual_correction()
install_correction_chronology_fix()

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
