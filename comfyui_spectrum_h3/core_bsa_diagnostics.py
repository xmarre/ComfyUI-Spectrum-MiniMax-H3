"""Debug-only runtime-structure diagnostics for the core BSA adapter.

This module does not participate in acceptance. It reports which live patch-chain
invariant prevented Spectrum from deriving a stable core-BSA numerical identity.
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


def runtime_structure_diagnostic(
    options: dict[str, Any], layout: Any, model: Any
) -> str:
    """Return the first live-structure mismatch without changing acceptance."""
    try:
        blocks = getattr(model, "blocks", None)
        if blocks is None or not hasattr(blocks, "__len__") or len(blocks) == 0:
            return f"model:blocks={type(blocks).__name__}"
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

            closure = core_bsa_compat._closure_values(underlying)
            if closure is None:
                return f"block[{index}]:block_closure_unreadable"
            owner = closure.get("patch")
            if owner is None:
                return f"block[{index}]:patch_missing"
            if patch is None:
                patch = owner
            elif owner is not patch:
                return f"block[{index}]:patch_changed"
            if closure.get("block") is not block:
                return f"block[{index}]:block_identity_mismatch"
            if closure.get("block_index") != index:
                return f"block[{index}]:block_index={closure.get('block_index')!r}"
            attention = closure.get("attention")
            if not callable(attention):
                return f"block[{index}]:attention_not_callable"
            attention_closure = core_bsa_compat._closure_values(attention)
            if attention_closure is None:
                return f"block[{index}]:attention_closure_unreadable"
            if attention_closure.get("patch") is not patch:
                return f"block[{index}]:attention_patch_mismatch"
            if attention_closure.get("block") is not block:
                return f"block[{index}]:attention_block_mismatch"
            if attention_closure.get("block_index") != index:
                return f"block[{index}]:attention_block_index_mismatch"

        if patch is None:
            return "patch:none"
        if core_bsa_compat._settings_identity(patch) is None:
            return "patch:settings_unrecognized"

        current = options.get("optimized_attention_override")
        effective, outer_mode = _effective_override(options)
        if not callable(effective):
            return (
                f"override:not_callable mode={outer_mode} "
                f"current={_callable_label(current)}"
            )
        closure = core_bsa_compat._closure_values(effective)
        if closure is None:
            return f"override:closure_unreadable mode={outer_mode}"
        if closure.get("patch") is not patch:
            return f"override:patch_mismatch mode={outer_mode}"
        installed = getattr(patch, "installed", None)
        if not isinstance(installed, set):
            return f"override:installed_type={type(installed).__name__}"
        if effective not in installed:
            return f"override:not_in_installed mode={outer_mode} installed={len(installed)}"
        previous = closure.get("previous")
        previous_closure = (
            core_bsa_compat._closure_values(previous) if callable(previous) else None
        )
        previous_patch = (
            previous_closure.get("patch")
            if isinstance(previous_closure, dict)
            else None
        )
        if (
            previous_patch is not None
            and core_bsa_compat._settings_identity(previous_patch) is not None
        ):
            return f"override:stacked_bsa_previous mode={outer_mode}"

        return (
            "runtime_structure_checks_passed_unexpectedly "
            f"mode={outer_mode} blocks=direct:{direct}/flow:{flow_wrapped} "
            f"current={_callable_label(current)}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never alter sampling
        return f"diagnostic_exception:{type(exc).__name__}:{exc}"
