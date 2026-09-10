"""Debug-only ownership diagnostics for the source-gated core BSA adapter.

This module does not participate in acceptance. It mirrors the strict ownership
checks closely enough to report which proof boundary rejected a real runtime
stack. All failures remain actual-only in the real adapter.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from . import core_bsa_compat, core_bsa_flow_compat


def _callable_label(value: Any) -> str:
    if value is None:
        return "None"
    base = getattr(value, "__func__", value)
    module = getattr(base, "__module__", type(base).__module__)
    qualname = getattr(base, "__qualname__", type(base).__qualname__)
    return f"{module}.{qualname}"


def _callable_source_diagnostic(value: Any) -> str:
    """Describe a rejected callable without changing any acceptance decision."""
    if value is None:
        return "callable=None"
    base = getattr(value, "__func__", value)
    module_name = getattr(base, "__module__", None)
    qualname = getattr(base, "__qualname__", None)
    defaults = getattr(base, "__defaults__", None)
    kwdefaults = getattr(base, "__kwdefaults__", None)
    code = getattr(base, "__code__", None)
    code_name = getattr(code, "co_filename", None)
    module = sys.modules.get(module_name) if isinstance(module_name, str) else None
    source = getattr(module, "__file__", None) if module is not None else None
    try:
        source_path = str(Path(source).resolve()) if isinstance(source, str) else None
    except (OSError, RuntimeError, TypeError, ValueError):
        source_path = f"<unresolvable:{source!r}>"
    try:
        code_path = str(Path(code_name).resolve()) if isinstance(code_name, str) else None
    except (OSError, RuntimeError, TypeError, ValueError):
        code_path = f"<unresolvable:{code_name!r}>"
    blob = core_bsa_compat._module_blob_sha(module) if module is not None else None
    closure = core_bsa_compat._closure_values(base)
    closure_keys = (
        tuple(sorted(str(key) for key in closure)) if isinstance(closure, dict) else None
    )
    return (
        f"module={module_name!r} qualname={qualname!r} registered={module is not None} "
        f"source={source_path!r} code={code_path!r} blob={blob!r} "
        f"defaults={defaults!r} kwdefaults={kwdefaults!r} closure_keys={closure_keys!r}"
    )


def _flow_layout_review_diagnostic(wrapper: Any, index: int) -> str:
    """Mirror every reviewed Flow layout-wrapper gate and report the first miss."""
    base = getattr(wrapper, "__func__", wrapper)
    if getattr(base, "__qualname__", None) != core_bsa_flow_compat._LAYOUT_QUALNAME:
        return f"layout:qualname={getattr(base, '__qualname__', None)!r}"

    module_name = getattr(base, "__module__", None)
    if not isinstance(module_name, str):
        return f"layout:module_name_type={type(module_name).__name__}"
    if (
        module_name != core_bsa_flow_compat._FLOW_ATTENTION_TOPLEVEL
        and not module_name.endswith(core_bsa_flow_compat._FLOW_ATTENTION_SUFFIX)
    ):
        return f"layout:module_name_unreviewed={module_name!r}"

    module = sys.modules.get(module_name)
    if module is None:
        return f"layout:module_not_registered={module_name!r}"
    source = getattr(module, "__file__", None)
    code = getattr(base, "__code__", None)
    if not isinstance(source, str) or code is None:
        return (
            "layout:source_or_code_missing "
            f"source_type={type(source).__name__} code_type={type(code).__name__}"
        )
    try:
        source_path = Path(source).resolve()
        code_path = Path(code.co_filename).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"layout:path_resolution={type(exc).__name__}:{exc}"
    if source_path != code_path:
        return f"layout:path_mismatch source={source_path} code={code_path}"
    if source_path.name != "attention.py":
        return f"layout:filename={source_path.name!r}"
    if source_path.parent.name != "h3_flow_regenerate":
        return f"layout:parent={source_path.parent.name!r}"

    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in core_bsa_flow_compat.AUDITED_FLOW_ATTENTION_GIT_BLOBS:
        return f"layout:source_blob_unreviewed={blob!r} path={source_path}"
    if not core_bsa_flow_compat._source_matches(
        base,
        module,
        ("make_layout_block_wrapper", "wrapper"),
    ):
        return f"layout:source_code_mismatch blob={blob} path={source_path}"
    if getattr(base, "__defaults__", None) is not None:
        return f"layout:defaults={getattr(base, '__defaults__', None)!r}"
    if getattr(base, "__kwdefaults__", None):
        return f"layout:kwdefaults={getattr(base, '__kwdefaults__', None)!r}"

    closure = core_bsa_compat._closure_values(base)
    if closure is None:
        return "layout:closure_unreadable"
    keys = frozenset(closure)
    if keys != core_bsa_flow_compat._LAYOUT_CLOSURE_SCHEMA:
        return f"layout:closure_schema={tuple(sorted(keys))!r}"
    if type(closure["layer"]) is not int or closure["layer"] != index:
        return f"layout:layer={closure['layer']!r} expected={index}"
    if type(closure["record_layout"]) is not bool:
        return f"layout:record_layout_type={type(closure['record_layout']).__name__}"

    previous = closure["previous"]
    marker = getattr(wrapper, "_h3_flow_layout_wrapper", None)
    if marker is not True:
        return f"layout:marker={marker!r}"
    marker_previous = getattr(wrapper, "_h3_flow_previous", None)
    if marker_previous is not previous:
        return (
            "layout:previous_marker_mismatch "
            f"closure={_callable_label(previous)} marker={_callable_label(marker_previous)}"
        )
    scope = getattr(wrapper, "_h3_flow_layout_scope", None)
    if scope not in {"layout", "attention"}:
        return f"layout:scope={scope!r}"
    metrics = closure["metrics"]
    marker_metrics = getattr(wrapper, "_h3_flow_metrics", None)
    if marker_metrics is not metrics:
        return (
            "layout:metrics_marker_mismatch "
            f"closure_id={id(metrics)} marker_id={id(marker_metrics)}"
        )
    if not callable(getattr(metrics, "increment", None)):
        return f"layout:metrics_increment={type(getattr(metrics, 'increment', None)).__name__}"
    if not callable(getattr(metrics, "event", None)):
        return f"layout:metrics_event={type(getattr(metrics, 'event', None)).__name__}"
    return (
        "layout:all_review_checks_passed "
        f"blob={blob} scope={scope!r} module={module_name!r} "
        f"previous={_callable_label(previous)} "
        f"previous_detail=({_callable_source_diagnostic(previous)})"
    )


def _flow_wrapper_review_diagnostic(wrapper: Any, index: int) -> str:
    base = getattr(wrapper, "__func__", wrapper)
    qualname = getattr(base, "__qualname__", None)
    if qualname == core_bsa_flow_compat._LAYOUT_QUALNAME:
        return _flow_layout_review_diagnostic(wrapper, index)
    if qualname == core_bsa_flow_compat._MIXED_QUALNAME:
        return "mixed:review_failed_before_detailed_diagnostic"
    return f"unknown:qualname={qualname!r} detail=({_callable_source_diagnostic(wrapper)})"


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
                    review = _flow_wrapper_review_diagnostic(replacement, index)
                    return (
                        f"block[{index}]:flow_unwrap={reason or 'unknown'} "
                        f"review={review} outer={_callable_label(replacement)}"
                    )
                flow_wrapped += 1
                if not identities:
                    return f"block[{index}]:flow_unwrap_without_identity"

            if getattr(underlying, "__code__", None) is not block_patch_code:
                return (
                    f"block[{index}]:block_patch_code_mismatch "
                    f"underlying={_callable_label(underlying)} source={source} "
                    f"underlying_detail=({_callable_source_diagnostic(underlying)})"
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
                f"blocks=direct:{direct}/flow:{flow_wrapped} "
                f"effective_detail=({_callable_source_diagnostic(effective)})"
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
