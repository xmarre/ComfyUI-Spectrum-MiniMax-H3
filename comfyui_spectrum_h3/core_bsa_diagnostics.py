"""Debug-only ownership diagnostics for the source-gated core BSA adapter.

This module does not participate in acceptance.  It mirrors the strict ownership
checks closely enough to report which proof boundary rejected a real runtime
stack.  All failures remain actual-only in the real adapter.
"""
from __future__ import annotations

from typing import Any

from . import core_bsa_compat, core_bsa_flow_compat


def _callable_label(value: Any) -> str:
    if value is None:
        return "None"
    base = getattr(value, "__func__", value)
    module = getattr(base, "__module__", type(base).__module__)
    qualname = getattr(base, "__qualname__", type(base).__qualname__)
    return f"{module}.{qualname}"


def _effective_override(options: dict[str, Any]) -> tuple[Any, str]:
    current = options.get("optimized_attention_override")
    if core_bsa_compat._looks_like_core_bsa_callable(
        current, "make_attention_override", "override"
    ):
        return current, "direct_bsa"
    contract = getattr(current, "attention_preprocess_v1", None)
    if isinstance(contract, tuple) and len(contract) == 2:
        transform, previous = contract
        return previous, f"preprocess:{_callable_label(transform)}"
    return current, f"unrecognized_outer:{_callable_label(current)}"


def ownership_diagnostic(
    options: dict[str, Any], layout: Any, model: Any
) -> str:
    """Return one compact first-failure description without changing acceptance."""
    try:
        module, source = core_bsa_compat._load_audited_module()
        if module is None:
            return f"module:{source or 'unavailable'}"
        block_patch_code = core_bsa_compat._nested_code(module.make_h3_block_patch, "block_patch")
        attention_code = core_bsa_compat._nested_code(module.make_h3_block_patch, "attention")
        override_code = core_bsa_compat._nested_code(module.make_attention_override, "override")
        if block_patch_code is None or attention_code is None or override_code is None:
            return "source:nested_code_missing"

        blocks = getattr(model, "blocks", None)
        if blocks is None:
            return "model:blocks_missing"
        patches = options.get("patches_replace")
        if not isinstance(patches, dict):
            return f"patches_replace:{type(patches).__name__}"
        dit = patches.get("dit")
        if not isinstance(dit, dict):
            return f"dit:{type(dit).__name__}"

        patch = None
        flow_wrapped = 0
        direct = 0
        for index, block in enumerate(blocks):
            replacement = dit.get(("double_block", index))
            if replacement is None:
                return f"block[{index}]:replacement_missing"
            if core_bsa_compat._looks_like_core_bsa_callable(
                replacement, "make_h3_block_patch", "block_patch"
            ):
                underlying = replacement
                direct += 1
            else:
                underlying, identities, _mixed, reason = core_bsa_flow_compat._unwrap_replacement(
                    replacement, index, layout, model
                )
                if underlying is None:
                    return (
                        f"block[{index}]:flow_unwrap={reason or 'unknown'} "
                        f"outer={_callable_label(replacement)}"
                    )
                flow_wrapped += 1
                if not identities:
                    return f"block[{index}]:flow_unwrap_without_identity"

            if getattr(underlying, "__code__", None) is not block_patch_code:
                return (
                    f"block[{index}]:block_patch_code_mismatch "
                    f"underlying={_callable_label(underlying)} source={source}"
                )
            closure = core_bsa_compat._closure_values(underlying)
            if closure is None:
                return f"block[{index}]:block_closure_unreadable"
            owner = closure.get("patch")
            if type(owner) is not module.SparseAttnPatch:
                return (
                    f"block[{index}]:patch_type_mismatch "
                    f"actual={type(owner).__module__}.{type(owner).__qualname__}"
                )
            if patch is None:
                patch = owner
            elif owner is not patch:
                return f"block[{index}]:patch_owner_changed"
            if closure.get("block") is not block:
                return (
                    f"block[{index}]:block_identity_mismatch "
                    f"captured={id(closure.get('block'))} runtime={id(block)}"
                )
            if closure.get("block_index") != index:
                return f"block[{index}]:block_index={closure.get('block_index')!r}"
            attention = closure.get("attention")
            if getattr(attention, "__code__", None) is not attention_code:
                return (
                    f"block[{index}]:attention_code_mismatch "
                    f"attention={_callable_label(attention)}"
                )
            attention_closure = core_bsa_compat._closure_values(attention)
            if attention_closure is None:
                return f"block[{index}]:attention_closure_unreadable"
            if attention_closure.get("patch") is not patch:
                return f"block[{index}]:attention_patch_owner_mismatch"
            if attention_closure.get("block") is not block:
                return f"block[{index}]:attention_block_identity_mismatch"
            if attention_closure.get("block_index") != index:
                return f"block[{index}]:attention_block_index_mismatch"

        if patch is None:
            return "patch:none"

        current = options.get("optimized_attention_override")
        effective, outer_mode = _effective_override(options)
        if getattr(effective, "__code__", None) is not override_code:
            contract = getattr(current, "attention_preprocess_v1", None)
            contract_shape = "none"
            if isinstance(contract, tuple):
                contract_shape = f"tuple[{len(contract)}]"
            elif contract is not None:
                contract_shape = type(contract).__name__
            return (
                "override:code_mismatch "
                f"mode={outer_mode} current={_callable_label(current)} "
                f"effective={_callable_label(effective)} contract={contract_shape} "
                f"blocks=direct:{direct}/flow:{flow_wrapped}"
            )
        closure = core_bsa_compat._closure_values(effective)
        if closure is None:
            return f"override:closure_unreadable mode={outer_mode}"
        if closure.get("patch") is not patch:
            return f"override:patch_owner_mismatch mode={outer_mode}"
        installed = getattr(patch, "installed", None)
        if not isinstance(installed, set):
            return f"override:installed_type={type(installed).__name__}"
        if effective not in installed:
            return f"override:not_in_installed mode={outer_mode} installed={len(installed)}"
        previous = closure.get("previous")
        if getattr(previous, "__code__", None) is override_code:
            return f"override:stacked_bsa_previous mode={outer_mode}"
        previous_closure = (
            core_bsa_compat._closure_values(previous) if previous is not None else None
        )
        if (
            previous_closure is not None
            and type(previous_closure.get("patch")) is module.SparseAttnPatch
        ):
            return f"override:previous_owns_bsa_patch mode={outer_mode}"

        return (
            "ownership_checks_passed_unexpectedly "
            f"mode={outer_mode} blocks=direct:{direct}/flow:{flow_wrapped} "
            f"current={_callable_label(current)}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never alter sampling
        return f"diagnostic_exception:{type(exc).__name__}:{exc}"
