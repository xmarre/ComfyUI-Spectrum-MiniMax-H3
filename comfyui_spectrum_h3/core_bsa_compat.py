"""Fail-closed Spectrum compatibility audit for ComfyUI core BlockSparseAttention.

This adapter deliberately recognizes only reviewed upstream implementations.  It
does not modify ComfyUI or ask core BSA to publish Spectrum-specific metadata.
Instead, Spectrum proves ownership/configuration before a model call and wraps
only the copied MiniMax-H3 block replacements for actual-call verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import itertools
import math
from pathlib import Path
import threading
import types
from typing import Any
import weakref

import torch

ADAPTER_KEY = "spectrum_core_bsa_v1"
ADAPTER_VERSION = 1
PRIVATE_AUDIT_KEY = "_spectrum_core_bsa_audit_v1"

# Comfy-Org/ComfyUI comfy_extras/nodes_sparse_attention.py as reviewed at
# be92396834f9b6e3e361cfe30e7eed693137a448 (unchanged from its parent).
# Any source change is actual-only until the new implementation is audited.
AUDITED_BSA_GIT_BLOBS = frozenset(
    {"006d1eb352f946a7c85595edf75d3cda9ac79194"}
)

_MISSING = object()
_IDENTITY_LOCK = threading.RLock()
_IDENTITY_COUNTER = itertools.count(1)
_IDENTITY_REGISTRY: dict[int, tuple[weakref.ReferenceType[Any] | None, int, Any | None]] = {}


def _release_lifetime_generation(
    object_id: int,
    generation: int,
    reference: weakref.ReferenceType[Any],
) -> None:
    with _IDENTITY_LOCK:
        current = _IDENTITY_REGISTRY.get(object_id)
        if current is not None and current[0] is reference and current[1] == generation:
            _IDENTITY_REGISTRY.pop(object_id, None)


def _lifetime_generation(value: Any) -> int:
    """Return a process-lifetime generation immune to CPython address reuse."""
    if value is None:
        return 0
    object_id = id(value)
    with _IDENTITY_LOCK:
        current = _IDENTITY_REGISTRY.get(object_id)
        if current is not None:
            reference, generation, strong = current
            owner = strong if reference is None else reference()
            if owner is value:
                return generation
            if owner is None:
                _IDENTITY_REGISTRY.pop(object_id, None)

        generation = next(_IDENTITY_COUNTER)

        def cleanup(reference, *, object_id=object_id, generation=generation):
            _release_lifetime_generation(object_id, generation, reference)

        try:
            reference = weakref.ref(value, cleanup)
        except TypeError:
            # Rare non-weakrefable callable owners are kept alive rather than
            # allowing their address to be recycled into backend history.
            _IDENTITY_REGISTRY[object_id] = (None, generation, value)
        else:
            _IDENTITY_REGISTRY[object_id] = (reference, generation, None)
        return generation


@dataclass
class CoreBSAAudit:
    identity: tuple[Any, ...]
    safe: bool
    patch: Any
    patch_generation: int
    block_count: int
    seq_len: int
    uuids: tuple[Any, ...]
    layout_identity: tuple[Any, ...]
    settings_identity: tuple[Any, ...]
    route_specs: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]
    pool_specs: tuple[tuple[int, int, str | None], ...]
    expected_receipts: tuple[tuple[Any, ...], ...]
    source_blob: str
    current_override: Any
    failure: str | None = None


def has_core_bsa_callback(options: dict[str, Any]) -> bool:
    callbacks = options.get("callbacks", {})
    if not isinstance(callbacks, dict):
        return False
    try:
        return any("block_sparse_attention" in group for group in callbacks.values())
    except Exception:  # noqa: BLE001 - malformed metadata is not trusted
        return False


def _looks_like_core_bsa_callable(function: Any, owner: str, local_name: str) -> bool:
    base = getattr(function, "__func__", function)
    return (
        getattr(base, "__module__", None) == "comfy_extras.nodes_sparse_attention"
        and getattr(base, "__qualname__", None) == f"{owner}.<locals>.{local_name}"
    )


def has_core_bsa_evidence(options: dict[str, Any]) -> bool:
    """Detect current core BSA even if another node dropped its callback metadata."""
    if has_core_bsa_callback(options):
        return True
    if _looks_like_core_bsa_callable(
        options.get("optimized_attention_override"), "make_attention_override", "override"
    ):
        return True
    patches_replace = options.get("patches_replace", {})
    if not isinstance(patches_replace, dict):
        return False
    dit = patches_replace.get("dit", {})
    if not isinstance(dit, dict):
        return False
    return any(
        _looks_like_core_bsa_callable(replacement, "make_h3_block_patch", "block_patch")
        for replacement in dit.values()
    )


@lru_cache(maxsize=8)
def _hash_git_blob(path: str, mtime_ns: int, size: int) -> str | None:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if len(data) != size:
        return None
    payload = f"blob {len(data)}\0".encode() + data
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def _module_blob_sha(module: Any) -> str | None:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        return None
    path = Path(source)
    if path.suffix != ".py":
        return None
    try:
        stat = path.stat()
        resolved = str(path.resolve())
    except OSError:
        return None
    return _hash_git_blob(resolved, int(stat.st_mtime_ns), int(stat.st_size))


def _load_audited_module() -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module("comfy_extras.nodes_sparse_attention")
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - unavailable/old ComfyUI stays actual-only
        return None, "module_unavailable"
    blob = _module_blob_sha(module)
    if blob not in AUDITED_BSA_GIT_BLOBS:
        return None, "source_unreviewed"
    required = (
        "SparseAttnPatch",
        "make_attention_override",
        "make_h3_block_patch",
        "HEAD_DIM",
        "BLOCK_SIZE",
        "PRODUCER_CHUNK",
        "ck",
    )
    if any(not hasattr(module, name) for name in required):
        return None, "source_shape_changed"
    if (
        int(module.HEAD_DIM) != 128
        or int(module.BLOCK_SIZE) != 64
        or int(module.PRODUCER_CHUNK) != 4096
    ):
        return None, "source_constants_changed"
    return module, blob


def _nested_code(function: Any, name: str) -> types.CodeType | None:
    code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    pending = [code]
    while pending:
        parent = pending.pop()
        for value in parent.co_consts:
            if isinstance(value, types.CodeType):
                if value.co_name == name:
                    return value
                pending.append(value)
    return None


def _closure_values(function: Any) -> dict[str, Any] | None:
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None)
    if not isinstance(code, types.CodeType):
        return None
    if not code.co_freevars:
        return {}
    if closure is None or len(closure) != len(code.co_freevars):
        return None
    values: dict[str, Any] = {}
    try:
        for name, cell in zip(code.co_freevars, closure):
            values[name] = cell.cell_contents
    except ValueError:
        return None
    return values


def _callable_identity(function: Any) -> tuple[Any, ...]:
    if function is None:
        return ("comfy.default",)
    base = getattr(function, "__func__", function)
    return (
        str(getattr(base, "__module__", type(base).__module__)),
        str(getattr(base, "__qualname__", type(base).__qualname__)),
        _lifetime_generation(base),
    )


def _attention_owner_identity(provider: Any) -> tuple[Any, ...]:
    transforms = []
    seen = set()
    current = provider
    while current is not None and hasattr(current, "attention_preprocess_v1"):
        if id(current) in seen:
            return ("cyclic_preprocess", _lifetime_generation(current))
        seen.add(id(current))
        contract = current.attention_preprocess_v1
        if not isinstance(contract, tuple) or len(contract) != 2:
            return ("invalid_preprocess", _callable_identity(current))
        transform, current = contract
        transforms.append(_callable_identity(transform))
    return (tuple(transforms), _callable_identity(current))


def _freeze(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return repr(value)


def _normalize_layout(layout: Any) -> tuple[int, tuple[Any, ...]] | None:
    seq_len = getattr(layout, "seq_len", None)
    signature = getattr(layout, "signature", None)
    segments = getattr(layout, "segments", None)
    if type(seq_len) is not int or seq_len <= 0 or not isinstance(segments, (tuple, list)):
        return None
    normalized = []
    cursor = 0
    try:
        for start, stop, kind in segments:
            a, b = int(start), int(stop)
            if a != cursor or b <= a or b > seq_len:
                return None
            normalized.append((a, b, str(kind)))
            cursor = b
    except (TypeError, ValueError):
        return None
    if cursor != seq_len:
        return None
    return seq_len, (
        ("signature", _freeze(signature)),
        ("segments", tuple(normalized)),
    )


def _normalize_uuids(options: dict[str, Any]) -> tuple[Any, ...] | None:
    try:
        uuids = tuple(options.get("uuids", ()))
        hash(uuids)
    except (TypeError, ValueError):
        return None
    return uuids


def _sigma_value(options: dict[str, Any]) -> float | None:
    sigmas = options.get("sigmas")
    if sigmas is None:
        return None
    try:
        if len(sigmas) == 0:
            return None
        value = float(sigmas[0])
        return value if math.isfinite(value) else None
    except torch.cuda.OutOfMemoryError:
        raise
    except (TypeError, ValueError, RuntimeError):
        return None


def _settings_identity(patch: Any) -> tuple[Any, ...] | None:
    try:
        dense_blocks = tuple(sorted(int(index) for index in patch.dense_blocks))
        tau = float(patch.tau)
        topk_ratio = float(patch.topk_ratio)
        sigma_start = float(patch.sigma_start)
        sigma_end = float(patch.sigma_end)
        if not all(math.isfinite(value) for value in (tau, topk_ratio, sigma_start, sigma_end)):
            return None
        settings = (
            bool(patch.vsa),
            tau,
            topk_ratio,
            int(patch.extra_tokens),
            sigma_start,
            sigma_end,
            int(patch.min_tokens),
            dense_blocks,
            str(patch.sink_conditioning),
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return settings


def _runtime_execution_identity(model: Any) -> tuple[Any, ...] | None:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, (tuple, list, torch.nn.ModuleList)) or not blocks:
        return None
    forwards = []
    qkv = []
    try:
        for block in blocks:
            attn = block.attn
            projection = attn.qkv_proj
            weight = projection.weight
            forwards.append(_callable_identity(attn.forward))
            qkv.append(
                (
                    int(attn.head_dim),
                    str(weight.dtype),
                    type(projection).__module__,
                    type(projection).__qualname__,
                    type(weight).__module__,
                    type(weight).__qualname__,
                )
            )
    except (AttributeError, TypeError, ValueError):
        return None
    return (
        ("model_dtype", str(getattr(model, "dtype", None))),
        ("attention_forwards", tuple(forwards)),
        ("qkv_execution", tuple(qkv)),
    )


def _candidate_cuda_device(blocks: Any) -> torch.device | None:
    try:
        cuda_devices = {
            torch.device(block.attn.qkv_proj.weight.device)
            for block in blocks
            if torch.device(block.attn.qkv_proj.weight.device).type == "cuda"
        }
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if len(cuda_devices) == 1:
        return next(iter(cuda_devices))
    if len(cuda_devices) > 1:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return torch.device("cuda", torch.cuda.current_device())
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError):
        return None


def _sparse_runtime_eligible(model: Any, module: Any) -> bool:
    blocks = getattr(model, "blocks", ())
    if not blocks:
        return False
    compute_dtype = getattr(model, "dtype", None)
    try:
        weight_dtypes = {block.attn.qkv_proj.weight.dtype for block in blocks}
        head_dims = {int(block.attn.head_dim) for block in blocks}
    except (AttributeError, TypeError, ValueError):
        return False
    if compute_dtype is None and len(weight_dtypes) == 1:
        compute_dtype = next(iter(weight_dtypes))
    if compute_dtype is not torch.bfloat16:
        return False
    if head_dims != {int(module.HEAD_DIM)}:
        return False
    device = _candidate_cuda_device(blocks)
    if device is None:
        return False
    try:
        return bool(module.ck.sol_attn_is_available(device))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - kernel introspection fails closed
        return False


def _conditioning_sinks(
    patch: Any, layout_identity: tuple[Any, ...], seq_len: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    segments = dict()
    try:
        segment_rows = dict(layout_identity)["segments"]
        for start, stop, kind in segment_rows:
            segments.setdefault(kind, (int(start), int(stop)))
        mode = str(patch.sink_conditioning)
    except (TypeError, ValueError, AttributeError):
        return None
    if mode == "off":
        return (0, 0), (0, 0)
    if mode not in {"exact_kv", "exact_kv_and_rows"}:
        return None
    video = segments.get("video")
    if video is None or video[0] <= 0:
        return (0, 0), (0, 0)
    block_size = 64
    sink = (0, (video[0] + block_size - 1) // block_size)
    if mode != "exact_kv_and_rows":
        return sink, (0, 0)
    audio = segments.get("audio")
    if audio is None:
        return sink, sink
    sink_q = (audio[0] // block_size, sink[1])
    if not (0 <= sink_q[0] <= sink_q[1] <= (seq_len + block_size - 1) // block_size):
        return None
    return sink, sink_q


def _replacement_ownership(
    module: Any, model: Any, options: dict[str, Any]
) -> tuple[Any, tuple[Any, ...]] | None:
    block_patch_code = _nested_code(module.make_h3_block_patch, "block_patch")
    attention_code = _nested_code(module.make_h3_block_patch, "attention")
    override_code = _nested_code(module.make_attention_override, "override")
    if block_patch_code is None or attention_code is None or override_code is None:
        return None

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return None
    replacements = options.get("patches_replace", {})
    if not isinstance(replacements, dict):
        return None
    dit = replacements.get("dit", {})
    if not isinstance(dit, dict):
        return None

    patch = None
    replacement_ids = []
    for index, block in enumerate(blocks):
        replacement = dit.get(("double_block", index))
        if getattr(replacement, "__code__", None) is not block_patch_code:
            return None
        closure = _closure_values(replacement)
        if closure is None:
            return None
        owner = closure.get("patch")
        if type(owner) is not module.SparseAttnPatch:
            return None
        if patch is None:
            patch = owner
        elif owner is not patch:
            return None
        if closure.get("block") is not block or closure.get("block_index") != index:
            return None
        attention = closure.get("attention")
        if getattr(attention, "__code__", None) is not attention_code:
            return None
        attention_closure = _closure_values(attention)
        if (
            attention_closure is None
            or attention_closure.get("patch") is not patch
            or attention_closure.get("block") is not block
            or attention_closure.get("block_index") != index
        ):
            return None
        replacement_ids.append(_callable_identity(replacement))

    if patch is None:
        return None
    current = options.get("optimized_attention_override")
    if getattr(current, "__code__", None) is not override_code:
        return None
    closure = _closure_values(current)
    if closure is None or closure.get("patch") is not patch:
        return None
    installed = getattr(patch, "installed", None)
    if not isinstance(installed, set) or current not in installed:
        return None
    previous = closure.get("previous")
    if getattr(previous, "__code__", None) is override_code:
        return None
    previous_closure = _closure_values(previous) if previous is not None else None
    if previous_closure is not None and type(previous_closure.get("patch")) is module.SparseAttnPatch:
        return None

    return patch, (
        ("inherited_attention", _attention_owner_identity(previous)),
        ("h3_replacements", tuple(replacement_ids)),
        ("bsa_override", _callable_identity(current)),
    )


def _pool_entry(
    patch: Any,
    key: tuple[Any, ...],
    expected: tuple[int, int, str | None] | None = None,
) -> tuple[Any, ...]:
    pooled = getattr(patch, "pooled", None)
    if not isinstance(pooled, dict):
        return ("invalid",)
    entry = pooled.get(key, _MISSING)
    if entry is _MISSING:
        return ("missing",)
    if (
        not isinstance(entry, (tuple, list))
        or len(entry) != 2
        or not all(torch.is_tensor(value) for value in entry)
    ):
        return ("invalid",)
    if expected is not None:
        heads, head_dim, device = expected
        required_shape = (heads, head_dim)
        if any(
            tuple(value.shape) != required_shape
            or value.dtype is not torch.float32
            or (device is not None and str(value.device) != device)
            for value in entry
        ):
            return ("invalid",)
    return ("present", entry[0], entry[1])


def _pool_ownership_identity(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    pool_specs: tuple[tuple[int, int, str | None], ...],
) -> tuple[Any, ...]:
    ownership = []
    for index in range(block_count):
        state = _pool_entry(patch, (index, seq_len, uuids), pool_specs[index])
        if state[0] == "present":
            ownership.append(
                (
                    index,
                    _lifetime_generation(state[1]),
                    _lifetime_generation(state[2]),
                    str(state[1].device),
                    str(state[2].device),
                )
            )
        else:
            ownership.append((index, state[0]))
    return tuple(ownership)


def _classify_pool_transition(
    before: tuple[Any, ...],
    after: tuple[Any, ...],
    *,
    sparse_selected: bool,
) -> str:
    if before[0] == "missing":
        if sparse_selected and after[0] == "present":
            return "h3_chunked_sparse_cold"
        if not sparse_selected and after[0] == "missing":
            return "h3_dense"
        return "unknown"
    if before[0] != "present" or after[0] != "present":
        return "unknown"
    if before[1] is not after[1] or before[2] is not after[2]:
        return "unknown"
    return "h3_chunked_sparse_primed" if sparse_selected else "h3_dense"


def _route_specs(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    layout_identity: tuple[Any, ...],
    options: dict[str, Any],
    sparse_runtime_ok: bool,
    pool_specs: tuple[tuple[int, int, str | None], ...],
) -> tuple[tuple[tuple[str, tuple[int, int], tuple[int, int]], ...], bool] | None:
    sigma = _sigma_value(options)
    if sigma is None:
        return None
    settings = _settings_identity(patch)
    if settings is None:
        return None
    vsa, _tau, _topk, _extra, sigma_start, sigma_end, min_tokens, dense_blocks, _sink_mode = settings
    if vsa:
        return None

    inside_window = sigma_end <= sigma <= sigma_start
    sinks = _conditioning_sinks(patch, layout_identity, seq_len)
    if sinks is None:
        return None
    sink, sink_q = sinks
    dense_set = set(dense_blocks)
    specs = []
    safe = True
    for index in range(block_count):
        if not inside_window or seq_len < min_tokens or index in dense_set:
            specs.append(("h3_dense", (0, 0), (0, 0)))
            continue
        if not sparse_runtime_ok:
            safe = False
            specs.append(("h3_sparse_unproven", sink, sink_q))
            continue
        state = _pool_entry(patch, (index, seq_len, uuids), pool_specs[index])
        if state[0] == "missing":
            route = "h3_chunked_sparse_cold"
        elif state[0] == "present":
            route = "h3_chunked_sparse_primed"
        else:
            safe = False
            route = "h3_sparse_unproven"
        specs.append((route, sink, sink_q))
    return tuple(specs), safe


def probe(
    options: dict[str, Any], layout: Any, model: Any
) -> tuple[CoreBSAAudit | None, str | None]:
    """Recognize current core BSA and derive the next-call numerical identity."""
    try:
        module, source_blob = _load_audited_module()
        if module is None or source_blob is None:
            return None, source_blob or "module_unavailable"

        ownership = _replacement_ownership(module, model, options)
        if ownership is None:
            return None, "ownership_unproven"
        patch, ownership_identity = ownership
        patch_generation = _lifetime_generation(patch)

        normalized_layout = _normalize_layout(layout)
        if normalized_layout is None:
            return None, "layout_unproven"
        seq_len, layout_identity = normalized_layout
        uuids = _normalize_uuids(options)
        if uuids is None:
            return None, "uuids_unproven"
        settings_identity = _settings_identity(patch)
        execution_identity = _runtime_execution_identity(model)
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

        sparse_runtime_ok = _sparse_runtime_eligible(model, module)
        routed = _route_specs(
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
        pool_ownership = _pool_ownership_identity(
            patch, len(model.blocks), seq_len, uuids, pool_specs
        )
        expected_receipts = tuple(
            (
                ADAPTER_KEY,
                ADAPTER_VERSION,
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
            ADAPTER_KEY,
            ADAPTER_VERSION,
            source_blob,
            patch_generation,
            ("mode", mode),
            ("settings", settings_identity),
            ("layout", layout_identity),
            ("uuids", _freeze(uuids)),
            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            ("ownership", ownership_identity),
            ("execution", execution_identity),
        )
        return CoreBSAAudit(
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
    except Exception:  # noqa: BLE001 - all adapter introspection is fail closed
        return None, "adapter_introspection_failed"


def _current_route_matches(
    audit: CoreBSAAudit,
    index: int,
    args: dict[str, Any],
) -> bool:
    try:
        hidden = args["img"]
        call_options = args["transformer_options"]
    except (KeyError, TypeError):
        return False
    if not torch.is_tensor(hidden) or hidden.ndim < 1 or int(hidden.shape[0]) != audit.seq_len:
        return False

    metadata_ok = True
    if _normalize_uuids(call_options) != audit.uuids:
        metadata_ok = False
    if _settings_identity(audit.patch) != audit.settings_identity:
        metadata_ok = False
    if call_options.get("optimized_attention_override") is not audit.current_override:
        metadata_ok = False
    actual_layout = _normalize_layout(call_options.get("minimax_h3_layout"))
    if (
        actual_layout is None
        or actual_layout[1] != audit.layout_identity
        or actual_layout[0] != audit.seq_len
    ):
        metadata_ok = False
    sigma = _sigma_value(call_options)
    if sigma is None:
        metadata_ok = False
        inside_window = False
    else:
        sigma_start = audit.settings_identity[4]
        sigma_end = audit.settings_identity[5]
        inside_window = sigma_end <= sigma <= sigma_start

    min_tokens = audit.settings_identity[6]
    dense_blocks = audit.settings_identity[7]
    dense = not inside_window or audit.seq_len < min_tokens or index in dense_blocks
    expected_route = audit.route_specs[index][0]
    if expected_route != "h3_dense" and (
        hidden.dtype is not torch.bfloat16 or hidden.device.type != "cuda"
    ):
        metadata_ok = False
    if not metadata_ok:
        return False
    if expected_route == "h3_dense":
        return dense
    return not dense


def _make_actual_wrapper(
    audit: CoreBSAAudit,
    index: int,
    replacement: Any,
    receipts: list[Any],
):
    key = (index, audit.seq_len, audit.uuids)
    replacement_closure = _closure_values(replacement)
    expected_attention = (
        replacement_closure.get("attention")
        if replacement_closure is not None
        else None
    )

    def audited_replacement(args, replacement_context):
        try:
            metadata_ok = _current_route_matches(audit, index, args)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - observation must not abort the actual block
            metadata_ok = False
            audit.failure = "actual_metadata_failed"
        try:
            before = _pool_entry(audit.patch, key, audit.pool_specs[index])
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001
            before = ("invalid",)
            metadata_ok = False

        sparse_selected = False
        original_block_calls = 0
        original_block = replacement_context.get("original_block")
        if expected_attention is None or not callable(original_block):
            metadata_ok = False
            output = replacement(args, replacement_context)
        else:
            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)

            observed_context = dict(replacement_context)
            observed_context["original_block"] = audited_original_block
            output = replacement(args, observed_context)
            if original_block_calls != 1:
                metadata_ok = False

        try:
            after = _pool_entry(audit.patch, key, audit.pool_specs[index])
            observed = _classify_pool_transition(
                before,
                after,
                sparse_selected=sparse_selected,
            )
            expected_route, expected_sink, expected_sink_q = audit.route_specs[index]
            if not metadata_ok or observed != expected_route:
                audit.failure = "actual_route_mismatch"
            receipt = (
                ADAPTER_KEY,
                ADAPTER_VERSION,
                audit.patch_generation,
                index,
                observed,
                audit.seq_len,
                expected_sink,
                expected_sink_q,
            )
            receipts.append(receipt)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 - observation failure must not abort sampling
            audit.failure = "actual_observation_failed"
        return output

    return audited_replacement


def instrument_actual_options(
    options: dict[str, Any], audit: CoreBSAAudit, receipts_key: str
) -> dict[str, Any]:
    """Wrap copied main-H3 replacements so an actual call proves its route."""
    prepared = dict(options)
    receipts = prepared.get(receipts_key)
    if not isinstance(receipts, list):
        audit.failure = "receipt_buffer_missing"
        return prepared
    patches_replace = prepared.get("patches_replace")
    if not isinstance(patches_replace, dict):
        audit.failure = "replacement_table_missing"
        return prepared
    dit = patches_replace.get("dit")
    if not isinstance(dit, dict):
        audit.failure = "dit_replacement_table_missing"
        return prepared

    local_patches = dict(patches_replace)
    local_dit = dict(dit)
    for index in range(audit.block_count):
        key = ("double_block", index)
        replacement = local_dit.get(key)
        if replacement is None:
            audit.failure = "main_h3_replacement_missing"
            return prepared
        local_dit[key] = _make_actual_wrapper(audit, index, replacement, receipts)
    local_patches["dit"] = local_dit
    prepared["patches_replace"] = local_patches
    prepared[PRIVATE_AUDIT_KEY] = audit
    return prepared


def accepts_actual(audit: CoreBSAAudit, receipts: tuple[Any, ...]) -> bool:
    if audit.failure is not None or not audit.safe:
        return False
    if len(receipts) != audit.block_count:
        return False
    seen = set()
    for receipt in receipts:
        if not isinstance(receipt, tuple) or len(receipt) != 8:
            return False
        if receipt[0] != ADAPTER_KEY or receipt[1] != ADAPTER_VERSION:
            return False
        if receipt[2] != audit.patch_generation:
            return False
        block = receipt[3]
        if type(block) is not int or block < 0 or block >= audit.block_count or block in seen:
            return False
        seen.add(block)
        if receipt != audit.expected_receipts[block]:
            return False
    return seen == set(range(audit.block_count))
