"""Bridge composable attention preprocessors around reviewed core BSA.

Core MiniMax-H3 BSA owns the block replacements, but model-function wrappers may
legitimately place an ``attention_preprocess_v1`` provider above BSA at call time.
The outer provider can be recreated for every model call, so object identity is
not a stable numerical identity.  This module unwraps only the explicit generic
preprocess contract, asks the source-gated core-BSA auditor to prove the actual
BSA owner, then restores the real call-time provider for actual-call validation.
Unknown wrappers still fail closed.
"""
from __future__ import annotations

import hashlib
import math
import types
from typing import Any

from . import core_bsa_compat

_MISSING = object()


def _freeze_contract_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        frozen = tuple(_freeze_contract_value(item) for item in value)
        return None if any(item is None and source is not None for item, source in zip(frozen, value)) else frozen
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            frozen = _freeze_contract_value(item)
            if frozen is None and item is not None:
                return None
            items.append((str(key), frozen))
        return tuple(sorted(items))
    return None


def _stable_callable_semantics(function: Any) -> tuple[Any, ...] | None:
    """Describe recreated contract transforms by code semantics, not object id."""
    base = getattr(function, "__func__", function)
    code = getattr(base, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    closure = core_bsa_compat._closure_values(base)
    if closure is None:
        return None
    frozen_closure = []
    for name in code.co_freevars:
        value = closure.get(name, _MISSING)
        if value is _MISSING:
            return None
        frozen = _freeze_contract_value(value)
        if frozen is None and value is not None:
            return None
        frozen_closure.append((name, frozen))

    digest = hashlib.sha256()
    digest.update(code.co_code)
    digest.update(repr(code.co_names).encode("utf-8"))
    digest.update(repr(code.co_freevars).encode("utf-8"))
    digest.update(repr(code.co_argcount).encode("ascii"))
    digest.update(repr(code.co_kwonlyargcount).encode("ascii"))
    return (
        str(getattr(base, "__module__", type(base).__module__)),
        str(getattr(base, "__qualname__", type(base).__qualname__)),
        digest.hexdigest(),
        tuple(frozen_closure),
    )


def _unwrap_preprocess_chain(options: dict[str, Any]) -> tuple[Any, tuple[Any, ...], str | None]:
    current = options.get("optimized_attention_override")
    transforms = []
    seen = set()

    while not core_bsa_compat._looks_like_core_bsa_callable(
        current, "make_attention_override", "override"
    ):
        if current is None:
            return None, (), "bsa_override_not_found"
        object_id = id(current)
        if object_id in seen:
            return None, (), "outer_preprocess_cycle"
        seen.add(object_id)

        contract = getattr(current, "attention_preprocess_v1", _MISSING)
        if contract is _MISSING:
            return None, (), "outer_attention_uncontracted"
        if not isinstance(contract, tuple) or len(contract) != 2:
            return None, (), "outer_preprocess_contract_invalid"
        transform, previous = contract
        transform_identity = _stable_callable_semantics(transform)
        if transform_identity is None:
            return None, (), "outer_preprocess_identity_unproven"
        transforms.append(transform_identity)
        current = previous

    return current, tuple(transforms), None


def probe(options: dict[str, Any], layout: Any, model: Any):
    """Probe core BSA through explicit outer ``attention_preprocess_v1`` layers."""
    real_override = options.get("optimized_attention_override")
    if core_bsa_compat._looks_like_core_bsa_callable(
        real_override, "make_attention_override", "override"
    ):
        return core_bsa_compat.probe(options, layout, model)

    bsa_override, preprocess_identity, reason = _unwrap_preprocess_chain(options)
    if bsa_override is None:
        return None, reason

    normalized = dict(options)
    normalized["optimized_attention_override"] = bsa_override
    audit, reason = core_bsa_compat.probe(normalized, layout, model)
    if audit is None:
        return None, reason

    # The underlying BSA ownership was proven against the reviewed source.  Add
    # the semantic preprocess chain to history without using the recreated outer
    # wrapper's address, then validate the exact real wrapper during this actual
    # call so a mid-forward provider swap still fails closed.
    audit.identity = (*audit.identity, ("outer_preprocess", preprocess_identity))
    audit.current_override = real_override
    return audit, None
