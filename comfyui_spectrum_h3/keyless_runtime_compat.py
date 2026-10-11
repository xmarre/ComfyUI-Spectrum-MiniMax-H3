from __future__ import annotations

import importlib
from typing import Any

import torch

from . import core_bsa_compat, minimax_h3
from .keyless_compat import (
    KEYLESS_CONTRACT_KEY,
    keyless_semantic_identity,
    validate_keyless_contract,
)

_NATIVE_H3_MODULE = "comfy.ldm.minimax.model"
_NATIVE_HELPERS = (
    "PackedLayout",
    "unpatchify_video",
    "unpack_audio",
    "time_shift_sigma",
    "patchify_video",
    "pack_audio",
)
_KEYLESS_PROVIDER = "minimax_h3_keyless_provider_v1"
_KEYLESS_PREPROCESSORS = "minimax_h3_keyless_routing_preprocessors_v1"
_KEYLESS_VALUE_DOMAIN = "minimax_h3_keyless_value_domain_v1"
_KEYLESS_ROUTING_POSITION_DOMAIN = "minimax_h3_keyless_routing_position_domain_v1"
_KEYLESS_QUERY_DOMAIN = "minimax_h3_keyless_query_domain_v1"
_KEYLESS_MASK = "minimax_h3_keyless_mask_v1"
_KEYLESS_LOG_MEASURE = "minimax_h3_keyless_log_measure_v1"
_KEYLESS_EXACT_BLOCKS = "minimax_h3_keyless_exact_blocks_v1"

_ORIGINAL_IS_NATIVE_MINIMAX_H3 = minimax_h3.is_native_minimax_h3
_ORIGINAL_NATIVE_MODULE = minimax_h3._native_module
_ORIGINAL_TOPOLOGY_SIGNATURE = minimax_h3.topology_signature
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


def _lifetime_identity(value: Any) -> int:
    """Return a process-lifetime identity that cannot alias after address reuse."""
    return core_bsa_compat._lifetime_generation(value)


def _freeze_option(value: Any) -> Any:
    """Build a bounded hashable identity for Keyless numerical routing options.

    Tensor contents are deliberately not copied or synchronized. A process-lifetime
    object generation plus PyTorch's mutation version invalidates Spectrum history
    both when a tensor is replaced and when the same tensor is edited in-place,
    without relying on recyclable CPython addresses.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        try:
            version = int(value._version)
        except (AttributeError, RuntimeError):
            version = None
        return (
            "tensor",
            tuple(int(v) for v in value.shape),
            str(value.dtype),
            str(value.device),
            _lifetime_identity(value),
            version,
        )
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_option(item)) for key, item in value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_option(item) for item in value), key=repr))

    # Keyless RowDomain is deliberately duck-typed so Spectrum does not import
    # the Keyless package. Preserve both physical coordinates and semantic label.
    if all(hasattr(value, name) for name in ("start", "stop", "indices", "identity")):
        return (
            "row_domain",
            getattr(value, "start"),
            getattr(value, "stop"),
            _freeze_option(getattr(value, "indices")),
            _freeze_option(getattr(value, "identity")),
        )
    return (
        "object",
        type(value).__module__,
        type(value).__qualname__,
        _lifetime_identity(value),
    )


def _callable_identity(value: Any) -> tuple[Any, ...]:
    base = getattr(value, "__func__", value)
    owner = getattr(value, "__self__", None)
    return (
        str(getattr(base, "__module__", type(base).__module__)),
        str(getattr(base, "__qualname__", type(base).__qualname__)),
        _lifetime_identity(base),
        None if owner is None else _lifetime_identity(owner),
    )


def _declared_identity(value: Any) -> Any:
    """Freeze a declared identity without depending on ephemeral bound-method objects."""
    if callable(value):
        return ("callable", _callable_identity(value))
    return _freeze_option(value)


def _preprocessor_identity(value: Any) -> tuple[Any, ...]:
    declared = getattr(value, "identity", None)
    fn = getattr(value, "fn", None)
    if callable(fn):
        implementation = _callable_identity(fn)
    elif callable(value):
        implementation = _callable_identity(value)
    else:
        implementation = (
            "non_callable",
            type(value).__module__,
            type(value).__qualname__,
            _lifetime_identity(value),
        )
    return (_declared_identity(declared), implementation)


def _keyless_runtime_identity(transformer_options: dict[str, Any]) -> tuple[Any, ...]:
    """Identity the numerical Keyless route independently of packed H3 geometry.

    Spectrum history is a prediction of final H3 hidden states. A QV provider,
    routing-only preprocessor, physical row domain, mask, or row measure can change
    those states without changing video/audio tensor shapes. Bind those semantics to
    the topology so an old QKV/Keyless or old-Keyless anchor cannot silently prime a
    numerically different call.
    """
    provider = transformer_options.get(_KEYLESS_PROVIDER)
    provider_identity = (
        ("none",)
        if provider is None
        else (
            "provider",
            getattr(provider, "api", None),
            _callable_identity(provider),
            _declared_identity(getattr(provider, "identity", None)),
        )
    )
    preprocessors = transformer_options.get(_KEYLESS_PREPROCESSORS, ())
    try:
        preprocessor_identity = tuple(_preprocessor_identity(item) for item in preprocessors)
    except TypeError:
        preprocessor_identity = (
            (
                "invalid_container",
                type(preprocessors).__module__,
                type(preprocessors).__qualname__,
                _lifetime_identity(preprocessors),
            ),
        )

    return (
        ("provider", provider_identity),
        ("preprocessors", preprocessor_identity),
        ("value_domain", _freeze_option(transformer_options.get(_KEYLESS_VALUE_DOMAIN))),
        (
            "routing_position_domain",
            _freeze_option(transformer_options.get(_KEYLESS_ROUTING_POSITION_DOMAIN)),
        ),
        ("query_domain", _freeze_option(transformer_options.get(_KEYLESS_QUERY_DOMAIN))),
        ("mask", _freeze_option(transformer_options.get(_KEYLESS_MASK))),
        ("log_measure", _freeze_option(transformer_options.get(_KEYLESS_LOG_MEASURE))),
        ("exact_blocks", _freeze_option(transformer_options.get(_KEYLESS_EXACT_BLOCKS))),
    )


def _topology_signature(
    inner: Any,
    video_x,
    audio_x,
    context,
    layout: Any,
    transformer_options: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    base = _ORIGINAL_TOPOLOGY_SIGNATURE(
        inner,
        video_x,
        audio_x,
        context,
        layout,
        transformer_options,
        payload,
    )
    identity = keyless_semantic_identity(inner)
    if identity is None:
        return base
    return (
        *base,
        ("keyless_semantic_identity", identity),
        ("keyless_runtime_identity", _keyless_runtime_identity(transformer_options)),
    )


def install_keyless_runtime_compat() -> None:
    """Extend Spectrum's H3 wrapper to exact Keyless v1 subclasses without fake QKV."""
    global _INSTALLED
    if _INSTALLED:
        return
    minimax_h3.is_native_minimax_h3 = _is_spectrum_minimax_h3
    minimax_h3._native_module = _runtime_native_module
    minimax_h3.topology_signature = _topology_signature
    _INSTALLED = True
