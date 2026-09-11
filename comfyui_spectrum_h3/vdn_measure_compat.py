"""Source-gated VDN ownership and completion receipts for weighted H3 measure.

The Flow API-2 external-sequence contract describes Mixed-Grid geometry; it does
not imply that VDN is installed.  This module distinguishes an absent VDN owner
from the reviewed VDN attention owner and, when VDN is present, binds Spectrum's
numerical identity to the concrete learned-gate/output-projection epilogue.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Mapping

import torch

VDN_FORWARD_MARKER = "_vdn_forward"
VDN_EXTERNAL_SEQUENCE_API_ATTR = "_vdn_external_sequence_api"
VDN_EPILOGUE_KEY = "vdn_h3_external_softmax_epilogue_v1"
VDN_EPILOGUE_RECEIPTS_KEY = "vdn_h3_external_softmax_epilogue_receipts_v1"
VDN_EXTERNAL_SEQUENCE_API = 2

AUDITED_VDN_HYBRID_GIT_BLOBS = frozenset(
    {"bdaf4c59e6ff26acfaa6bffb902423f103b5c18b"}
)
AUDITED_VDN_EPILOGUE_GIT_BLOBS = frozenset(
    {"e363ecdf814144fdf3c700b3eeb551776207a773"}
)

_EXTERNAL_COUNT_FIELDS = (
    "native_sequence_rows",
    "sequence_rows",
    "video_start",
    "temporal",
    "prefix_t",
    "source_rows_per_frame",
    "prefix_rows_per_frame",
)


@dataclass(frozen=True)
class VDNMeasureAudit:
    active: bool
    attentions: tuple[Any, ...]
    forwards: tuple[Any, ...]
    capabilities: tuple[Any, ...]
    owner_generations: tuple[str, ...]
    config_digests: tuple[str, ...]
    weight_owner_digests: tuple[str, ...]
    gate_expected: tuple[bool, ...]
    external_digest: str | None
    external_counts: tuple[tuple[str, int], ...]
    expected_receipts: tuple[tuple[tuple[str, Any], ...], ...]
    hybrid_blob: str | None
    epilogue_blob: str | None
    epilogue_module: Any | None


def _core():
    # Imported lazily because core_bsa_compat delegates its VDN proof here.
    from . import core_bsa_compat

    return core_bsa_compat


def _loaded_exact_module(name: str, expected_blobs: frozenset[str]):
    module = sys.modules.get(name)
    if module is None or getattr(module, "__name__", None) != name:
        return None, None
    blob = _core()._module_blob_sha(module)
    if blob not in expected_blobs:
        return None, blob
    return module, blob


def _external_digest(epilogue_module, external: Any) -> str | None:
    if not isinstance(external, Mapping) or external.get("api") != VDN_EXTERNAL_SEQUENCE_API:
        return None
    if any(type(external.get(name)) is not int for name in _EXTERNAL_COUNT_FIELDS):
        return None
    normalized = {name: int(external[name]) for name in _EXTERNAL_COUNT_FIELDS}
    try:
        digest = epilogue_module._digest(normalized)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - audited companion introspection fails closed
        return None
    return digest if isinstance(digest, str) and digest else None


def _state_layout_matches_external(
    state: Any, external_counts: tuple[tuple[str, int], ...]
) -> bool:
    counts = dict(external_counts)
    layout = getattr(state, "layout", None)
    if layout is None:
        return False
    try:
        return (
            int(layout.seq_len) == counts["native_sequence_rows"]
            and int(layout.video_start) == counts["video_start"]
            and int(layout.num_frames) == counts["temporal"]
            and int(layout.tokens_per_frame) == counts["source_rows_per_frame"]
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _current_capability_fields(
    epilogue_module,
    capability,
    attention,
    index: int,
    external_digest: str,
    external_counts: tuple[tuple[str, int], ...],
    *,
    require_active_layout: bool = True,
):
    try:
        if (
            getattr(capability, "api", None) != 1
            or getattr(capability, "block_index", None) != index
            or getattr(capability, "out_proj", None) is not attention.out_proj
            or int(capability.heads) != int(attention.heads)
            or int(capability.head_dim) != int(attention.head_dim)
        ):
            return None
        state = capability.state
        branches = state.branches
        if not isinstance(branches, (tuple, list)) or index >= len(branches):
            return None
        branch = branches[index]
        if capability.branch_owner is not branch:
            return None
        weight_owner = epilogue_module._weight_owner(state, branch)
        if capability.weight_owner is not weight_owner:
            return None
        owner_generation = capability.owner_generation
        config_digest = capability.config_digest
        weight_owner_digest = capability.weight_owner_digest
        if not all(isinstance(value, str) and value for value in (
            owner_generation, config_digest, weight_owner_digest
        )):
            return None
        if epilogue_module._digest(
            epilogue_module._config_identity(state, index, branch)
        ) != config_digest:
            return None
        expected_weight_digest = epilogue_module._digest(
            {
                "epilogue_owner_generation": owner_generation,
                "owner_type": None
                if weight_owner is None
                else f"{type(weight_owner).__module__}.{type(weight_owner).__qualname__}",
                "block_index": index,
                "branch_present": branch is not None,
            }
        )
        if expected_weight_digest != weight_owner_digest:
            return None
        cfg = state.cfg
        if not isinstance(cfg, Mapping):
            return None
        if require_active_layout and not _state_layout_matches_external(
            state, external_counts
        ):
            return None
        gate_expected = bool(branch is not None and cfg.get("enable_softmax_gate", True))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - ownership proof fails closed
        return None

    fields = (
        ("vdn_owner_generation", owner_generation),
        ("vdn_config_digest", config_digest),
        ("vdn_weight_owner_digest", weight_owner_digest),
        ("vdn_external_digest", external_digest),
        ("vdn_gate_expected", gate_expected),
        ("vdn_gate_calls", 1 if gate_expected else 0),
        ("vdn_projection_calls", 1),
        ("vdn_completed", True),
    )
    return fields


def _spectrum_wrapper(function: Any) -> bool:
    module = sys.modules.get("comfyui_spectrum_h3.minimax_h3")
    expected = None if module is None else getattr(module, "diffusion_model_wrapper", None)
    return function is expected and callable(expected)


def _deferred_layout_context_proven(
    hybrid: Any, state: Any, options: Any
) -> bool:
    if not isinstance(options, Mapping):
        return False
    wrappers = options.get("wrappers")
    if not isinstance(wrappers, Mapping):
        return False
    groups = wrappers.get("diffusion_model")
    if not isinstance(groups, Mapping):
        return False

    expected_code = _core()._nested_code(hybrid.make_layout_wrapper, "wrap")
    if expected_code is None:
        return False

    flattened = []
    try:
        for key, functions in groups.items():
            if not isinstance(functions, (tuple, list)):
                return False
            for function in functions:
                flattened.append((key, function))
    except Exception:  # noqa: BLE001 - malformed wrapper metadata fails closed
        return False

    spectrum_indices = [
        index
        for index, (key, function) in enumerate(flattened)
        if key == "spectrum_minimax_h3" and _spectrum_wrapper(function)
    ]
    vdn_indices = []
    for index, (key, function) in enumerate(flattened):
        if key != "vdn_h3":
            continue
        base = getattr(function, "__func__", function)
        if (
            getattr(base, "__module__", None) != "vdn_h3.hybrid"
            or getattr(base, "__qualname__", None)
            != "make_layout_wrapper.<locals>.wrap"
            or getattr(base, "__code__", None) is not expected_code
        ):
            continue
        closure = _core()._closure_values(base)
        if closure is None or set(closure) != {"state"} or closure.get("state") is not state:
            continue
        vdn_indices.append(index)

    # If VDN has already executed outside Spectrum, state.layout is live and this
    # helper is not needed.  A missing layout is safe to defer only when the exact
    # wrapper for this VDN state is still downstream of this Spectrum wrapper.
    return (
        len(spectrum_indices) == 1
        and len(vdn_indices) == 1
        and spectrum_indices[0] < vdn_indices[0]
    )


def probe(
    model: Any, external: Any, block_count: int, options: Any = None
):
    """Return reviewed per-block VDN epilogue ownership, or a fail-closed reason."""
    try:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, (tuple, list, torch.nn.ModuleList)) or len(blocks) != block_count:
            return None, "vdn_model_shape_unproven"
        attentions = tuple(block.attn for block in blocks)
        forwards = tuple(attention.forward for attention in attentions)
        marked = tuple(
            getattr(forward, VDN_FORWARD_MARKER, False) is True for forward in forwards
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001
        return None, "vdn_owner_introspection_failed"

    if not any(marked):
        return VDNMeasureAudit(
            active=False,
            attentions=attentions,
            forwards=forwards,
            capabilities=(),
            owner_generations=(),
            config_digests=(),
            weight_owner_digests=(),
            gate_expected=(),
            external_digest=None,
            external_counts=(),
            expected_receipts=(),
            hybrid_blob=None,
            epilogue_blob=None,
            epilogue_module=None,
        ), None
    if not all(marked):
        return None, "vdn_owner_incomplete"

    if not isinstance(external, Mapping) or external.get("api") != VDN_EXTERNAL_SEQUENCE_API:
        return None, "vdn_external_contract_missing"
    if external.get("mode") != "dense_gate_no_linear" or external.get("topology") != "mixed_grid_low_suffix":
        return None, "vdn_external_contract_unreviewed"

    hybrid, hybrid_blob = _loaded_exact_module(
        "vdn_h3.hybrid", AUDITED_VDN_HYBRID_GIT_BLOBS
    )
    epilogue, epilogue_blob = _loaded_exact_module(
        "vdn_h3.mixed_measure_epilogue", AUDITED_VDN_EPILOGUE_GIT_BLOBS
    )
    if hybrid is None or epilogue is None:
        return None, "vdn_source_unreviewed"
    required_hybrid = ("make_vdn_forward", "make_layout_wrapper")
    required_epilogue = (
        "ExternalSoftmaxEpilogueCapability",
        "_config_identity",
        "_weight_owner",
        "_digest",
    )
    if any(not hasattr(hybrid, name) for name in required_hybrid) or any(
        not hasattr(epilogue, name) for name in required_epilogue
    ):
        return None, "vdn_source_shape_changed"
    expected_code = _core()._nested_code(hybrid.make_vdn_forward, "vdn_forward")
    if expected_code is None:
        return None, "vdn_forward_code_unproven"

    external_digest = _external_digest(epilogue, external)
    if external_digest is None:
        return None, "vdn_external_digest_unproven"
    external_counts = tuple(
        (name, int(external[name])) for name in _EXTERNAL_COUNT_FIELDS
    )

    capabilities = []
    owner_generations = []
    config_digests = []
    weight_owner_digests = []
    gate_expected = []
    expected_receipts = []
    states = []
    for index, (attention, forward) in enumerate(zip(attentions, forwards)):
        if (
            getattr(forward, "__module__", None) != "vdn_h3.hybrid"
            or getattr(forward, "__qualname__", None)
            != "make_vdn_forward.<locals>.vdn_forward"
            or getattr(forward, "__code__", None) is not expected_code
            or getattr(forward, VDN_EXTERNAL_SEQUENCE_API_ATTR, None)
            != VDN_EXTERNAL_SEQUENCE_API
        ):
            return None, "vdn_forward_owner_unproven"
        capability = getattr(forward, VDN_EPILOGUE_KEY, None)
        if type(capability) is not epilogue.ExternalSoftmaxEpilogueCapability:
            return None, "vdn_epilogue_capability_unproven"
        active_layout = _state_layout_matches_external(
            capability.state, external_counts
        )
        if not active_layout and not _deferred_layout_context_proven(
            hybrid, capability.state, options
        ):
            return None, "vdn_execution_context_unproven"
        fields = _current_capability_fields(
            epilogue,
            capability,
            attention,
            index,
            external_digest,
            external_counts,
            require_active_layout=active_layout,
        )
        if fields is None:
            return None, "vdn_epilogue_owner_unproven"
        capabilities.append(capability)
        states.append(capability.state)
        owner_generations.append(dict(fields)["vdn_owner_generation"])
        config_digests.append(dict(fields)["vdn_config_digest"])
        weight_owner_digests.append(dict(fields)["vdn_weight_owner_digest"])
        gate_expected.append(bool(dict(fields)["vdn_gate_expected"]))
        expected_receipts.append(fields)
    if states and any(state is not states[0] for state in states[1:]):
        return None, "vdn_state_owner_inconsistent"

    return VDNMeasureAudit(
        active=True,
        attentions=attentions,
        forwards=forwards,
        capabilities=tuple(capabilities),
        owner_generations=tuple(owner_generations),
        config_digests=tuple(config_digests),
        weight_owner_digests=tuple(weight_owner_digests),
        gate_expected=tuple(gate_expected),
        external_digest=external_digest,
        external_counts=external_counts,
        expected_receipts=tuple(expected_receipts),
        hybrid_blob=hybrid_blob,
        epilogue_blob=epilogue_blob,
        epilogue_module=epilogue,
    ), None


def identity(audit: VDNMeasureAudit):
    if not audit.active:
        return ("absent",)
    return (
        "reviewed_vdn_external_softmax_epilogue_v1",
        ("sources", audit.hybrid_blob, audit.epilogue_blob),
        ("external_digest", audit.external_digest),
        (
            "blocks",
            tuple(
                (
                    index,
                    audit.owner_generations[index],
                    audit.config_digests[index],
                    audit.weight_owner_digests[index],
                    audit.gate_expected[index],
                )
                for index in range(len(audit.forwards))
            ),
        ),
    )


def _same_forward_owner(current: Any, expected: Any) -> bool:
    """Compare a callable and its bound owner without relying on bound-method identity."""
    current_func = getattr(current, "__func__", None)
    expected_func = getattr(expected, "__func__", None)
    if current_func is not None or expected_func is not None:
        return (
            current_func is expected_func
            and getattr(current, "__self__", None) is getattr(expected, "__self__", None)
        )
    return current is expected


def runtime_matches(audit: VDNMeasureAudit, index: int) -> bool:
    if type(index) is not int or index < 0 or index >= len(audit.attentions):
        return False
    try:
        forward = audit.attentions[index].forward
    except Exception:  # noqa: BLE001
        return False
    if not _same_forward_owner(forward, audit.forwards[index]):
        return False
    if not audit.active:
        return getattr(forward, VDN_FORWARD_MARKER, False) is not True
    if (
        getattr(forward, VDN_FORWARD_MARKER, False) is not True
        or getattr(forward, VDN_EXTERNAL_SEQUENCE_API_ATTR, None)
        != VDN_EXTERNAL_SEQUENCE_API
        or getattr(forward, VDN_EPILOGUE_KEY, None) is not audit.capabilities[index]
    ):
        return False
    fields = _current_capability_fields(
        audit.epilogue_module,
        audit.capabilities[index],
        audit.attentions[index],
        index,
        audit.external_digest,
        audit.external_counts,
    )
    return fields == audit.expected_receipts[index]


def expected_receipt(audit: VDNMeasureAudit, index: int, route: str):
    if not audit.active or route == "h3_dense":
        return None
    if type(index) is not int or index < 0 or index >= len(audit.expected_receipts):
        return None
    return audit.expected_receipts[index]


def validate_receipt_delta(
    audit: VDNMeasureAudit,
    index: int,
    route: str,
    sink: Any,
    before: int | None,
):
    """Validate exactly the receipt produced by this block invocation."""
    expected = expected_receipt(audit, index, route)
    if not audit.active:
        return True, None
    if not isinstance(sink, list) or type(before) is not int or before < 0:
        return False, None
    if expected is None:
        return (len(sink) == before), None
    if len(sink) != before + 1:
        return False, None
    item = sink[-1]
    if (
        not isinstance(item, tuple)
        or len(item) != 2
        or item[0] != index
        or item[1] != expected
    ):
        return False, None
    return True, item[1]


def validate_final_receipts(
    audit: VDNMeasureAudit,
    route_specs: tuple[tuple[str, Any, Any], ...],
    sink: Any,
) -> bool:
    if not audit.active:
        return sink is None
    if not isinstance(sink, list):
        return False
    expected = tuple(
        (index, audit.expected_receipts[index])
        for index, (route, _sink, _sink_q) in enumerate(route_specs)
        if route != "h3_dense"
    )
    return tuple(sink) == expected


__all__ = [
    "AUDITED_VDN_EPILOGUE_GIT_BLOBS",
    "AUDITED_VDN_HYBRID_GIT_BLOBS",
    "VDN_EPILOGUE_KEY",
    "VDN_EPILOGUE_RECEIPTS_KEY",
    "VDNMeasureAudit",
    "expected_receipt",
    "identity",
    "probe",
    "runtime_matches",
    "validate_final_receipts",
    "validate_receipt_delta",
]
