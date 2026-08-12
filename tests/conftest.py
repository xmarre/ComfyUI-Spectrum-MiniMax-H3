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

# Native ComfyUI source-contract tests run on CPU-only GitHub runners. Set the
# same public ComfyUI CPU flag before any source module imports model_management
# so those imports do not attempt torch.cuda.current_device().
if comfyui_path and not torch.cuda.is_available() and "--cpu" not in sys.argv:
    sys.argv.append("--cpu")
