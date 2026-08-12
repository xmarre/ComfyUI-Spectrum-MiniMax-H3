from .config import AGGRESSIVE_PRESET, CONSERVATIVE_PRESET, SpectrumH3Config
from .feature3_direction import install_feature3_direction_experiment
from .feature3_previous_error import install_feature3_previous_error_experiment
from .forecast import HistoryWeightForecaster
from .minimax_h3 import locate_minimax_h3_inner, require_native_minimax_h3
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .runtime import SpectrumH3Runtime

install_feature3_direction_experiment()
install_feature3_previous_error_experiment()

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
