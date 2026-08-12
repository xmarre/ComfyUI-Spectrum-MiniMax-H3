from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

comfyui_path = os.environ.get("COMFYUI_PATH")
if comfyui_path and comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)

# Native ComfyUI source-contract tests run on CPU-only GitHub runners. Prime
# ComfyUI's public CLI args once with --cpu before any source module imports
# model_management. Restore pytest's argv immediately afterward.
if comfyui_path and not torch.cuda.is_available():
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [original_argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401
    finally:
        sys.argv[:] = original_argv
