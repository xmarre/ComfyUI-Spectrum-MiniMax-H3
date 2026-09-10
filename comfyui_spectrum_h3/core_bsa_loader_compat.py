"""Resolve reviewed core BSA callables under ComfyUI's real builtin-node loader.

ComfyUI loads builtin ``comfy_extras/nodes_*.py`` files through ``load_custom_node``
using the file path stem as ``sys.modules`` key. Therefore the runtime
``__module__`` for core BSA callables is normally an absolute path, not
``comfy_extras.nodes_sparse_attention``. Importing the canonical dotted module
creates a second module instance with different function/class identities and is
not a valid ownership oracle for the live model.

This compatibility layer keeps the existing strict BSA audit but resolves the
exact runtime module instance from the live replacement/override callables. The
module is accepted only when it is the current ComfyUI checkout's
``comfy_extras/nodes_sparse_attention.py``, has the reviewed Git blob and source
shape, and the live nested callable semantics match that reviewed on-disk source.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import torch

from . import core_bsa_compat, source_code_audit

_BSA_FILE = "nodes_sparse_attention.py"
_BSA_PARENT = "comfy_extras"
_REQUIRED = (
    "SparseAttnPatch",
    "make_attention_override",
    "make_h3_block_patch",
    "HEAD_DIM",
    "BLOCK_SIZE",
    "PRODUCER_CHUNK",
    "ck",
)
_INSTALLED = False


def _comfy_extras_dir() -> Path | None:
    try:
        import comfy

        source = getattr(comfy, "__file__", None)
        if not isinstance(source, str):
            return None
        root = Path(source).resolve().parent.parent
        extras = (root / _BSA_PARENT).resolve()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return extras


def _runtime_module_from_callable(
    function: Any,
    owner: str,
    local_name: str,
) -> tuple[Any, str] | None:
    base = getattr(function, "__func__", function)
    if getattr(base, "__qualname__", None) != f"{owner}.<locals>.{local_name}":
        return None
    code = getattr(base, "__code__", None)
    module_name = getattr(base, "__module__", None)
    if not isinstance(code, types.CodeType) or not isinstance(module_name, str):
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        return None
    try:
        source_path = Path(source).resolve()
        code_path = Path(code.co_filename).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    extras = _comfy_extras_dir()
    if (
        extras is None
        or source_path != code_path
        or source_path.name != _BSA_FILE
        or source_path.parent != extras
    ):
        return None
    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        return None
    if any(not hasattr(module, name) for name in _REQUIRED):
        return None
    try:
        constants_ok = (
            int(module.HEAD_DIM) == 128
            and int(module.BLOCK_SIZE) == 64
            and int(module.PRODUCER_CHUNK) == 4096
        )
    except (TypeError, ValueError):
        return None
    if not constants_ok:
        return None
    if not source_code_audit.matches_nested_source_code(
        base,
        source_path,
        (owner, local_name),
    ):
        return None
    if getattr(base, "__defaults__", None) is not None or getattr(base, "__kwdefaults__", None):
        return None
    return module, blob


def _looks_like_runtime_core_bsa_callable(function: Any, owner: str, local_name: str) -> bool:
    return _runtime_module_from_callable(function, owner, local_name) is not None


def _runtime_module_for_options(
    options: dict[str, Any], model: Any
) -> tuple[Any, str] | None:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, (tuple, list, torch.nn.ModuleList)) or not blocks:
        return None
    patches = options.get("patches_replace")
    if not isinstance(patches, dict):
        return None
    dit = patches.get("dit")
    if not isinstance(dit, dict):
        return None

    runtime_module = None
    runtime_blob = None
    for index in range(len(blocks)):
        replacement = dit.get(("double_block", index))
        resolved = _runtime_module_from_callable(
            replacement,
            "make_h3_block_patch",
            "block_patch",
        )
        if resolved is None:
            return None
        module, blob = resolved
        if runtime_module is None:
            runtime_module, runtime_blob = module, blob
        elif module is not runtime_module or blob != runtime_blob:
            return None

    override = options.get("optimized_attention_override")
    resolved_override = _runtime_module_from_callable(
        override,
        "make_attention_override",
        "override",
    )
    if resolved_override is None:
        return None
    module, blob = resolved_override
    if module is not runtime_module or blob != runtime_blob:
        return None
    return runtime_module, runtime_blob


def _runtime_probe(
    options: dict[str, Any], layout: Any, model: Any
) -> tuple[core_bsa_compat.CoreBSAAudit | None, str | None]:
    """Run the existing BSA audit against the actual ComfyUI-loaded module instance."""
    try:
        resolved = _runtime_module_for_options(options, model)
        if resolved is None:
            return None, "ownership_unproven"
        module, source_blob = resolved

        ownership = core_bsa_compat._replacement_ownership(module, model, options)
        if ownership is None:
            return None, "ownership_unproven"
        patch, ownership_identity = ownership
        patch_generation = core_bsa_compat._lifetime_generation(patch)

        normalized_layout = core_bsa_compat._normalize_layout(layout)
        if normalized_layout is None:
            return None, "layout_unproven"
        seq_len, layout_identity = normalized_layout
        uuids = core_bsa_compat._normalize_uuids(options)
        if uuids is None:
            return None, "uuids_unproven"
        settings_identity = core_bsa_compat._settings_identity(patch)
        execution_identity = core_bsa_compat._runtime_execution_identity(model)
        if settings_identity is None or execution_identity is None:
            return None, "execution_shape_unproven"
        if settings_identity[0]:
            return None, "vsa_unsupported"

        try:
            pool_specs = tuple(
                (
                    int(block.attn.heads),
                    int(block.attn.head_dim),
                    (
                        str(block.attn.qkv_proj.weight.device)
                        if torch.device(block.attn.qkv_proj.weight.device).type == "cuda"
                        else None
                    ),
                )
                for block in model.blocks
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None, "pool_shape_unproven"

        sparse_runtime_ok = core_bsa_compat._sparse_runtime_eligible(model, module)
        routed = core_bsa_compat._route_specs(
            patch,
            len(model.blocks),
            seq_len,
            uuids,
            layout_identity,
            options,
            sparse_runtime_ok,
            pool_specs,
        )
        if routed is None:
            return None, "route_unproven"
        route_specs, safe = routed
        pool_ownership = core_bsa_compat._pool_ownership_identity(
            patch,
            len(model.blocks),
            seq_len,
            uuids,
            pool_specs,
        )
        expected_receipts = tuple(
            (
                core_bsa_compat.ADAPTER_KEY,
                core_bsa_compat.ADAPTER_VERSION,
                patch_generation,
                index,
                route,
                seq_len,
                sink,
                sink_q,
            )
            for index, (route, sink, sink_q) in enumerate(route_specs)
        )
        mode = "sol-attn" if settings_identity[2] == 0.0 else "sla"
        identity = (
            core_bsa_compat.ADAPTER_KEY,
            core_bsa_compat.ADAPTER_VERSION,
            source_blob,
            patch_generation,
            ("mode", mode),
            ("settings", settings_identity),
            ("layout", layout_identity),
            ("uuids", core_bsa_compat._freeze(uuids)),
            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            ("ownership", ownership_identity),
            ("execution", execution_identity),
        )
        return core_bsa_compat.CoreBSAAudit(
            identity=identity,
            safe=bool(safe),
            patch=patch,
            patch_generation=patch_generation,
            block_count=len(model.blocks),
            seq_len=seq_len,
            uuids=uuids,
            layout_identity=layout_identity,
            settings_identity=settings_identity,
            route_specs=route_specs,
            pool_specs=pool_specs,
            expected_receipts=expected_receipts,
            source_blob=source_blob,
            current_override=options.get("optimized_attention_override"),
        ), None
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - source/ownership introspection stays fail closed
        return None, "adapter_introspection_failed"


def install_core_bsa_loader_compat() -> None:
    """Install loader-aware BSA callable/module resolution once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    core_bsa_compat._looks_like_core_bsa_callable = _looks_like_runtime_core_bsa_callable
    core_bsa_compat.probe = _runtime_probe
    _INSTALLED = True
