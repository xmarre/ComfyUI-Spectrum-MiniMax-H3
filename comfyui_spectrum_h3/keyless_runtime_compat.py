from __future__ import annotations

import importlib
from typing import Any

from . import minimax_h3
from .keyless_compat import KEYLESS_CONTRACT_KEY, validate_keyless_contract

_NATIVE_H3_MODULE = "comfy.ldm.minimax.model"
_NATIVE_HELPERS = (
    "PackedLayout",
    "unpatchify_video",
    "unpack_audio",
    "time_shift_sigma",
    "patchify_video",
    "pack_audio",
)

_ORIGINAL_IS_NATIVE_MINIMAX_H3 = minimax_h3.is_native_minimax_h3
_ORIGINAL_NATIVE_MODULE = minimax_h3._native_module
_INSTALLED = False


def _is_spectrum_minimax_h3(inner: Any) -> bool:
    if inner is not None and hasattr(inner, KEYLESS_CONTRACT_KEY):
        validate_keyless_contract(inner)
        return True
    return _ORIGINAL_IS_NATIVE_MINIMAX_H3(inner)


def _runtime_native_module(inner: Any):
    """Resolve inherited native H3 helpers for Keyless subclasses.

    The Keyless model class intentionally lives in the Keyless package. Spectrum's
    existing helper lookup used the concrete class module, which is correct for the
    native class but would look for PackedLayout/unpack helpers in the Keyless package.
    Keyless v1 inherits ComfyUI's MiniMaxH3Model packing/final-layer semantics, so
    resolve those helpers from the audited native module after proving the public
    contract and actual subclass relationship.
    """
    contract = validate_keyless_contract(inner)
    if contract is None:
        return _ORIGINAL_NATIVE_MODULE(inner)
    module = importlib.import_module(_NATIVE_H3_MODULE)
    native_type = getattr(module, "MiniMaxH3Model", None)
    if native_type is None or not isinstance(inner, native_type):
        raise TypeError(
            "Keyless H3 contract is present but the diffusion model is not a subclass "
            "of ComfyUI's current MiniMaxH3Model"
        )
    missing = [name for name in _NATIVE_HELPERS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "native MiniMax H3 module is missing Keyless-required inherited helpers: "
            + ", ".join(missing)
        )
    return module


def install_keyless_runtime_compat() -> None:
    """Extend Spectrum's H3 wrapper to exact Keyless v1 subclasses without fake QKV."""
    global _INSTALLED
    if _INSTALLED:
        return
    minimax_h3.is_native_minimax_h3 = _is_spectrum_minimax_h3
    minimax_h3._native_module = _runtime_native_module
    _INSTALLED = True
