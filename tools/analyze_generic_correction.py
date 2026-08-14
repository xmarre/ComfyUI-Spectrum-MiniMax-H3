from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "comfyui_spectrum_h3"
    / "generic_correction_evaluator.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "spectrum_generic_correction_evaluator",
    _CORE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load the shared generic-correction evaluator")
_CORE = sys.modules.get(_SPEC.name)
if _CORE is None:
    _CORE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _CORE
    _SPEC.loader.exec_module(_CORE)

# Keep the forensic CLI's existing import surface, including test-only private
# helpers, while executing the single runtime-owned evaluator implementation.
for _name, _value in vars(_CORE).items():
    if _name not in {
        "__builtins__",
        "__cached__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(_CORE.main())
