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
        import comfy.cli_args
    finally:
        sys.argv[:] = original_argv

    # The compatibility matrix intentionally pins a minimal comfy-kitchen that
    # is sufficient for the reviewed H3 fixture. Newer ComfyUI imports probe a
    # capability helper added after that pinned package. Source-contract tests
    # do not execute INT8 attention, so provide the missing negative capability
    # answer without installing a moving dependency into the historical matrix.
    import comfy_kitchen

    if not hasattr(comfy_kitchen, "int8_attention_is_available"):
        comfy_kitchen.int8_attention_is_available = lambda: False
