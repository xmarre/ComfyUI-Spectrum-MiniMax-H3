from __future__ import annotations

import importlib
import json
import sys
import traceback
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

_PACKAGE_NAME = "comfyui_spectrum_h3"


def _load_research_module():
    """Load the stdlib-only research path without executing package __init__."""
    package_dir = Path(__file__).resolve().parent
    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(_PACKAGE_NAME)
        package.__file__ = str(package_dir / "__init__.py")
        package.__package__ = _PACKAGE_NAME
        package.__path__ = [str(package_dir)]
        package.__spec__ = ModuleSpec(_PACKAGE_NAME, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(package_dir)]
        sys.modules[_PACKAGE_NAME] = package
    return importlib.import_module(f"{_PACKAGE_NAME}.generic_correction_research")


def _run_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("worker payload must be a JSON object")
    block = payload.get("block")
    root = payload.get("root")
    if not isinstance(block, dict):
        raise ValueError("worker payload is missing a calibration block")
    if not isinstance(root, str) or not root:
        raise ValueError("worker payload is missing the research root")

    research = _load_research_module()
    result = research.persist_and_analyze(block, root=Path(root))
    return {
        "ok": True,
        "duplicate": bool(result.duplicate),
        "console_summary": result.console_summary,
        "elapsed_seconds": float(result.elapsed_seconds),
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = _run_payload(payload)
    except Exception:  # noqa: BLE001 - child reports failure to the parent process
        traceback.print_exc(file=sys.stderr)
        return 1

    json.dump(result, sys.stdout, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
