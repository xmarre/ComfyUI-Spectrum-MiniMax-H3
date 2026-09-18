from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

from . import model_aware
from .keyless_compat import (
    KEYLESS_ARCHITECTURE,
    attention_projection_key,
    keyless_semantic_identity,
    validate_keyless_contract,
)

_ORIGINAL_ARCHITECTURE_SIGNATURE = model_aware._architecture_signature
_ORIGINAL_SELECTED_BASE_KEYS = model_aware._selected_base_keys
_ORIGINAL_BUILD_PROFILE = model_aware._build_profile
_INSTALLED = False


def _architecture_signature(inner: Any) -> tuple[Any, ...]:
    base = _ORIGINAL_ARCHITECTURE_SIGNATURE(inner)
    identity = keyless_semantic_identity(inner)
    if identity is None:
        return base
    return (*base, ("keyless_semantic_identity", identity))


def _selected_base_keys(inner: Any) -> tuple[str, ...]:
    base = _ORIGINAL_SELECTED_BASE_KEYS(inner)
    if validate_keyless_contract(inner) is None:
        return base
    last = max(0, len(getattr(inner, "blocks", ())) - 1)
    native_qkv = f"diffusion_model.blocks.{last}.attn.qkv_proj.weight"
    keyless_qv = attention_projection_key(inner, last)
    if native_qkv not in base:
        raise RuntimeError(
            "Spectrum model-aware base-key contract changed; refusing to guess the Keyless projection slot"
        )
    return tuple(keyless_qv if key == native_qkv else key for key in base)


def _build_profile(model_patcher: Any, inner: Any, cache_key: tuple[Any, ...]):
    profile = _ORIGINAL_BUILD_PROFILE(model_patcher, inner, cache_key)
    identity = keyless_semantic_identity(inner)
    if identity is None:
        return profile
    digest = hashlib.sha256(
        repr(identity).encode("utf-8", "backslashreplace")
    ).hexdigest()[:16]
    return replace(
        profile,
        base_model_identity=(
            f"{KEYLESS_ARCHITECTURE}:{digest}:"
            f"{getattr(model_patcher, 'clone_base_uuid', 'unknown-base')}"
        ),
    )


def install_keyless_model_aware_compat() -> None:
    """Teach model-aware profiling about QV/provenance without changing native-QKV behavior."""
    global _INSTALLED
    if _INSTALLED:
        return
    model_aware._architecture_signature = _architecture_signature
    model_aware._selected_base_keys = _selected_base_keys
    model_aware._build_profile = _build_profile
    _INSTALLED = True
