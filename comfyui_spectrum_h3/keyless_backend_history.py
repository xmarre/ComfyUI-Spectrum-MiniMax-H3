"""Spectrum numerical-history contract for MiniMax-H3-Keyless routes.

Keyless routing can change numerically without changing packed H3 tensor geometry:
the QV provider, routing-only preprocessors, row domains, masks and row measures
all live in transformer options.  Bind those semantics to Spectrum's backend
history.  The built-in materialized Q/route/V path is self-qualified; opaque
Keyless providers or optimized-attention overrides remain actual-only unless an
existing ``attention_backend_history_v1`` provider supplies receipts.
"""
from __future__ import annotations

from typing import Any

from . import backend_history, core_bsa_compat, keyless_runtime_compat
from .keyless_compat import keyless_semantic_identity, validate_keyless_contract

HISTORY_KEY = "spectrum_keyless_history_v1"
HISTORY_VERSION = 1

_ORIGINAL_PREFLIGHT = None
_ORIGINAL_OBSERVE = None
_INSTALLED = False


def _has_declared_backend_policy(options: dict[str, Any]) -> bool:
    policies = options.get(backend_history.POLICIES, {})
    return isinstance(policies, dict) and bool(policies)


def _is_plain_materialized_route(options: dict[str, Any]) -> bool:
    """Whether Keyless will use its ordinary materialized reference/backend path.

    Core BSA is deliberately excluded even though the Keyless fallback eventually
    strips it: the outer core-BSA adapter owns that proof and publishes a stronger
    source/patch identity.  Any explicit Keyless provider, optimized-attention
    override or declared backend policy needs its own actual receipts.
    """
    return (
        options.get(keyless_runtime_compat._KEYLESS_PROVIDER) is None
        and options.get("optimized_attention_override") is None
        and not _has_declared_backend_policy(options)
        and not core_bsa_compat.has_core_bsa_evidence(options)
    )


def _keyless_identity(
    model: Any,
    options: dict[str, Any],
    *,
    mode: str,
    delegated: Any = None,
) -> tuple[Any, ...]:
    semantic = keyless_semantic_identity(model)
    if semantic is None:
        raise RuntimeError("Keyless backend-history identity requested for native QKV H3")
    return (
        HISTORY_KEY,
        HISTORY_VERSION,
        semantic,
        keyless_runtime_compat._keyless_runtime_identity(options),
        ("mode", str(mode)),
        ("delegated", delegated),
    )


def _preflight(options: dict[str, Any], layout: Any, model: Any):
    if validate_keyless_contract(model) is None:
        if _ORIGINAL_PREFLIGHT is None:
            raise RuntimeError("Keyless backend-history compatibility was not installed")
        return _ORIGINAL_PREFLIGHT(options, layout, model)

    if _is_plain_materialized_route(options):
        return _keyless_identity(model, options, mode="materialized_route"), True, None

    if _ORIGINAL_PREFLIGHT is None:
        raise RuntimeError("Keyless backend-history compatibility was not installed")
    delegated_identity, delegated_safe, audit = _ORIGINAL_PREFLIGHT(options, layout, model)

    # A Core-BSA/QV composition must pass through the dedicated source-gated
    # reference bypass.  A direct preflight cannot make the QKV producer safe.
    if core_bsa_compat.has_core_bsa_evidence(options):
        return (
            _keyless_identity(
                model,
                options,
                mode="core_bsa_requires_reference_bypass",
                delegated=delegated_identity,
            ),
            False,
            audit,
        )

    declared = _has_declared_backend_policy(options)
    mode = "declared_provider" if declared else "opaque_provider"
    safe = bool(declared and delegated_identity is not None and delegated_safe)
    return (
        _keyless_identity(model, options, mode=mode, delegated=delegated_identity),
        safe,
        audit,
    )


def _is_materialized_policy(identity: Any) -> bool:
    return (
        isinstance(identity, tuple)
        and len(identity) == 6
        and identity[0] == HISTORY_KEY
        and identity[1] == HISTORY_VERSION
        and identity[4] == ("mode", "materialized_route")
    )


def _observe(runtime, run_id, step_id, options, policy):
    if policy is not None:
        identity, safe = policy
        if _is_materialized_policy(identity):
            current_runtime_identity = keyless_runtime_compat._keyless_runtime_identity(options)
            unchanged = (
                _is_plain_materialized_route(options)
                and identity[3] == current_runtime_identity
            )
            runtime.observe_backend_history(
                run_id,
                step_id,
                identity,
                (),
                bool(safe and unchanged),
            )
            return

    if _ORIGINAL_OBSERVE is None:
        raise RuntimeError("Keyless backend-history compatibility was not installed")
    return _ORIGINAL_OBSERVE(runtime, run_id, step_id, options, policy)


def install_keyless_backend_history() -> None:
    """Install before Core-BSA recovery so later wrappers preserve this contract."""
    global _INSTALLED, _ORIGINAL_PREFLIGHT, _ORIGINAL_OBSERVE
    if _INSTALLED:
        return
    _ORIGINAL_PREFLIGHT = backend_history._preflight
    _ORIGINAL_OBSERVE = backend_history.observe
    backend_history._preflight = _preflight
    backend_history.observe = _observe
    _INSTALLED = True
