from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement anchor, found {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: str, old: str, new: str, expected: int) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} replacement anchors, found {count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


CORE = "comfyui_spectrum_h3/core_bsa_compat.py"
LOADER = "comfyui_spectrum_h3/core_bsa_loader_compat.py"
FLOW = "comfyui_spectrum_h3/core_bsa_flow_compat.py"
RECOVERY = "comfyui_spectrum_h3/core_bsa_forecast_recovery.py"
WORKFLOW = ".github/workflows/tests.yml"

replace_once(
    CORE,
    '''# Comfy-Org/ComfyUI comfy_extras/nodes_sparse_attention.py as reviewed at
# be92396834f9b6e3e361cfe30e7eed693137a448 (unchanged from its parent).
# Any source change is actual-only until the new implementation is audited.
AUDITED_BSA_GIT_BLOBS = frozenset(
    {"006d1eb352f946a7c85595edf75d3cda9ac79194"}
)

_MISSING = object()
_IDENTITY_LOCK = threading.RLock()
_IDENTITY_COUNTER = itertools.count(1)
''',
    '''# Comfy-Org/ComfyUI comfy_extras/nodes_sparse_attention.py sources reviewed
# for Spectrum numerical-history compatibility. The second blob is the generic
# attention-measure implementation from ComfyUI PR #16239; unreviewed sources
# remain actual-only.
AUDITED_BSA_GIT_BLOBS = frozenset(
    {
        "006d1eb352f946a7c85595edf75d3cda9ac79194",
        "a2b0d601529d776265b7b37cfe49b804158ae4a2",
    }
)
MEASURE_CAPABLE_BSA_GIT_BLOBS = frozenset(
    {"a2b0d601529d776265b7b37cfe49b804158ae4a2"}
)
AUDITED_BSA_MEASURE_GIT_BLOBS = frozenset(
    {"7789d179f778134f3b3b649a65653980ef9e5fe3"}
)
AUDITED_ATTENTION_MEASURE_GIT_BLOBS = frozenset(
    {"439f2798f8514d38ea56f12c330068b78e8fa539"}
)
ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_CAPABILITIES_KEY = "attention_measure_capabilities_v1"
CORE_BSA_MEASURE_PROVIDER = "comfy.core.block_sparse_attention"
SPARSE_MEASURE_PROFILE = "weighted_exact_blocks_v1"
SPARSE_MEASURE_ROUTE = "core_bsa_h3_chunked"
SPARSE_MEASURE_PREPROCESS = "core_h3_chunked_rms_rope_split_half_v1"
DENSE_MEASURE_PROFILE = "dense_exact_v1"
DENSE_MEASURE_ROUTE = "core_dense_sdpa"
DENSE_MEASURE_PREPROCESS = "caller_attention_domain_v1"

_MISSING = object()
_IDENTITY_LOCK = threading.RLock()
_IDENTITY_COUNTER = itertools.count(1)
_MEASURE_CALL_COUNTER = itertools.count(1)
''',
)

replace_once(
    CORE,
    '''@dataclass
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
''',
    '''@dataclass(frozen=True)
class CoreBSAMeasureAudit:
    bsa_module: Any
    adapter: Any
    core: Any
    capability: Any
    semantic_digest: str
    normalized_request: Any
    external_sequence: Any
    owner_generation: str
    provider_identity: str
    exact_k_block_range: tuple[int, int]
    exact_range_digest: str
    pool_keys: tuple[tuple[Any, ...], ...]
    call_tokens: tuple[tuple[int, int], ...]
    core_blob: str
    adapter_blob: str
    chunked_key_bias: bool


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
    measure: CoreBSAMeasureAudit | None = None
    failure: str | None = None
''',
)

helper_anchor = '''def _pool_entry(
    patch: Any,
    key: tuple[Any, ...],
    expected: tuple[int, int, str | None] | None = None,
) -> tuple[Any, ...]:
'''
helper_code = r'''def _measure_audit(
    module: Any,
    source_blob: str,
    patch: Any,
    options: dict[str, Any],
    layout: Any,
    seq_len: int,
    uuids: tuple[Any, ...],
    layout_identity: tuple[Any, ...],
    block_count: int,
) -> tuple[CoreBSAMeasureAudit | None, str | None]:
    request = options.get(ATTENTION_MEASURE_KEY, _MISSING)
    if request is _MISSING:
        return None, None
    if source_blob not in MEASURE_CAPABLE_BSA_GIT_BLOBS:
        return None, "measure_bsa_source_unreviewed"

    adapter = getattr(module, "measure", None)
    if adapter is None:
        return None, "measure_adapter_missing"
    try:
        core = importlib.import_module("comfy.attention_measure")
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - source-gated optional contract
        return None, "measure_core_missing"

    adapter_blob = _module_blob_sha(adapter)
    core_blob = _module_blob_sha(core)
    if adapter_blob not in AUDITED_BSA_MEASURE_GIT_BLOBS:
        return None, "measure_adapter_source_unreviewed"
    if core_blob not in AUDITED_ATTENTION_MEASURE_GIT_BLOBS:
        return None, "measure_core_source_unreviewed"

    required_adapter = (
        "ATTENTION_MEASURE_KEY",
        "PROVIDER_IDENTITY",
        "Capability",
        "VDN_EXTERNAL_SEQUENCE_KEY",
        "supports_key_bias",
    )
    required_core = (
        "ATTENTION_MEASURE_CAPABILITIES_KEY",
        "normalize",
        "semantic_digest",
        "validate_h3",
        "merge_exact_k_blocks",
    )
    if any(not hasattr(adapter, name) for name in required_adapter) or any(
        not hasattr(core, name) for name in required_core
    ):
        return None, "measure_source_shape_changed"
    if (
        adapter.ATTENTION_MEASURE_KEY != ATTENTION_MEASURE_KEY
        or core.ATTENTION_MEASURE_CAPABILITIES_KEY
        != ATTENTION_MEASURE_CAPABILITIES_KEY
        or adapter.PROVIDER_IDENTITY != CORE_BSA_MEASURE_PROVIDER
    ):
        return None, "measure_contract_identity_changed"

    registry = options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
    capability = registry.get(CORE_BSA_MEASURE_PROVIDER) if isinstance(registry, dict) else None
    owner_generation = getattr(patch, "measure_owner_generation", None)
    if (
        capability is None
        or capability is not getattr(patch, "measure_capability", None)
        or type(capability) is not adapter.Capability
        or getattr(capability, "patch", None) is not patch
        or not isinstance(owner_generation, str)
        or not owner_generation
        or not isinstance(getattr(patch, "measure_plans", None), dict)
    ):
        return None, "measure_capability_unproven"

    external = options.get(adapter.VDN_EXTERNAL_SEQUENCE_KEY)
    try:
        normalized = core.validate_h3(
            request,
            layout=layout,
            q_rows=seq_len,
            kv_rows=seq_len,
            external_sequence=external,
        )
        digest = core.semantic_digest(normalized)
        sinks = _conditioning_sinks(patch, layout_identity, seq_len)
        if sinks is None:
            return None, "measure_sink_unproven"
        exact_range = core.merge_exact_k_blocks(normalized, 64, sinks[0])
        exact_range_digest = hashlib.sha256(
            f"{exact_range[0]}:{exact_range[1]}".encode("ascii")
        ).hexdigest()
        chunked_key_bias = bool(adapter.supports_key_bias(module.ck.sol_attn_chunked))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - malformed/stale measure must be actual-only
        return None, "measure_contract_unproven"

    plan_identity = (
        ATTENTION_MEASURE_KEY,
        digest,
        CORE_BSA_MEASURE_PROVIDER,
        owner_generation,
        SPARSE_MEASURE_ROUTE,
        SPARSE_MEASURE_PREPROCESS,
        SPARSE_MEASURE_PROFILE,
    )
    pool_keys = tuple(
        (index, seq_len, uuids, plan_identity) for index in range(block_count)
    )
    with _IDENTITY_LOCK:
        call_generation = next(_MEASURE_CALL_COUNTER)
    call_tokens = tuple((call_generation, index) for index in range(block_count))
    return CoreBSAMeasureAudit(
        bsa_module=module,
        adapter=adapter,
        core=core,
        capability=capability,
        semantic_digest=digest,
        normalized_request=_freeze(normalized),
        external_sequence=_freeze(external),
        owner_generation=owner_generation,
        provider_identity=CORE_BSA_MEASURE_PROVIDER,
        exact_k_block_range=tuple(exact_range),
        exact_range_digest=exact_range_digest,
        pool_keys=pool_keys,
        call_tokens=call_tokens,
        core_blob=core_blob,
        adapter_blob=adapter_blob,
        chunked_key_bias=chunked_key_bias,
    ), None


def _measure_identity(measure: CoreBSAMeasureAudit) -> tuple[Any, ...]:
    return (
        ATTENTION_MEASURE_KEY,
        measure.semantic_digest,
        ("provider", measure.provider_identity, measure.owner_generation),
        ("sources", measure.core_blob, measure.adapter_blob),
        ("rows", len(measure.normalized_request),),
        ("exact_range", measure.exact_k_block_range, measure.exact_range_digest),
        ("mask", "none"),
        ("sparse_profile", SPARSE_MEASURE_PROFILE, SPARSE_MEASURE_ROUTE, SPARSE_MEASURE_PREPROCESS),
        ("dense_profile", DENSE_MEASURE_PROFILE, DENSE_MEASURE_ROUTE, DENSE_MEASURE_PREPROCESS),
        ("external_sequence", measure.external_sequence),
        ("chunked_key_bias", measure.chunked_key_bias),
        ("calibration_key_policy", "measure_bound_v1"),
    )


def _pool_key(audit: CoreBSAAudit, index: int) -> tuple[Any, ...]:
    if audit.measure is not None:
        return audit.measure.pool_keys[index]
    return index, audit.seq_len, audit.uuids


def _measure_runtime_matches(
    audit: CoreBSAAudit,
    index: int,
    call_options: Any,
    *,
    sparse_selected: bool,
) -> bool:
    if not isinstance(call_options, dict):
        return False
    measure = audit.measure
    request = call_options.get(ATTENTION_MEASURE_KEY, _MISSING)
    if measure is None:
        return request is _MISSING
    if request is _MISSING:
        return False
    try:
        registry = call_options.get(ATTENTION_MEASURE_CAPABILITIES_KEY)
        capability = (
            registry.get(measure.provider_identity) if isinstance(registry, dict) else None
        )
        if (
            capability is not measure.capability
            or capability is not getattr(audit.patch, "measure_capability", None)
            or getattr(capability, "patch", None) is not audit.patch
            or getattr(audit.patch, "measure_owner_generation", None)
            != measure.owner_generation
        ):
            return False
        layout = call_options.get("minimax_h3_layout")
        external = call_options.get(measure.adapter.VDN_EXTERNAL_SEQUENCE_KEY)
        normalized = measure.core.validate_h3(
            request,
            layout=layout,
            q_rows=audit.seq_len,
            kv_rows=audit.seq_len,
            external_sequence=external,
        )
        if (
            measure.core.semantic_digest(normalized) != measure.semantic_digest
            or _freeze(normalized) != measure.normalized_request
            or _freeze(external) != measure.external_sequence
        ):
            return False
        if sparse_selected and not measure.adapter.supports_key_bias(
            measure.bsa_module.ck.sol_attn_chunked
        ):
            return False
        return type(index) is int and 0 <= index < audit.block_count
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - execution metadata mismatch is fail closed
        return False


def _measure_receipt_fields(
    measure: CoreBSAMeasureAudit,
    index: int,
    route: str,
    *,
    completed: bool,
) -> tuple[Any, ...]:
    if route == "h3_dense":
        profile = DENSE_MEASURE_PROFILE
        numerical_route = DENSE_MEASURE_ROUTE
        preprocess = DENSE_MEASURE_PREPROCESS
    else:
        profile = SPARSE_MEASURE_PROFILE
        numerical_route = SPARSE_MEASURE_ROUTE
        preprocess = SPARSE_MEASURE_PREPROCESS
    return (
        measure.call_tokens[index],
        index,
        measure.owner_generation,
        measure.semantic_digest,
        profile,
        numerical_route,
        measure.pool_keys[index][1],
        measure.pool_keys[index][1],
        measure.exact_range_digest,
        preprocess,
        bool(completed),
    )


def _receipt(
    audit: CoreBSAAudit,
    index: int,
    route: str,
    *,
    completed: bool,
) -> tuple[Any, ...]:
    _expected_route, sink, sink_q = audit.route_specs[index]
    receipt = (
        ADAPTER_KEY,
        ADAPTER_VERSION,
        audit.patch_generation,
        index,
        route,
        audit.seq_len,
        sink,
        sink_q,
    )
    if audit.measure is None:
        return receipt
    return (
        *receipt,
        (
            ATTENTION_MEASURE_KEY,
            *_measure_receipt_fields(
                audit.measure,
                index,
                route,
                completed=completed,
            ),
        ),
    )


def _expected_receipts(audit: CoreBSAAudit) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        _receipt(audit, index, route, completed=True)
        for index, (route, _sink, _sink_q) in enumerate(audit.route_specs)
    )


'''
replace_once(CORE, helper_anchor, helper_code + helper_anchor)

replace_once(
    CORE,
    '''def _pool_ownership_identity(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    pool_specs: tuple[tuple[int, int, str | None], ...],
) -> tuple[Any, ...]:
    ownership = []
    for index in range(block_count):
        state = _pool_entry(patch, (index, seq_len, uuids), pool_specs[index])
''',
    '''def _pool_ownership_identity(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    pool_specs: tuple[tuple[int, int, str | None], ...],
    measure: CoreBSAMeasureAudit | None = None,
) -> tuple[Any, ...]:
    ownership = []
    for index in range(block_count):
        key = measure.pool_keys[index] if measure is not None else (index, seq_len, uuids)
        state = _pool_entry(patch, key, pool_specs[index])
''',
)

replace_once(
    CORE,
    '''def _route_specs(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    layout_identity: tuple[Any, ...],
    options: dict[str, Any],
    sparse_runtime_ok: bool,
    pool_specs: tuple[tuple[int, int, str | None], ...],
) -> tuple[tuple[tuple[str, tuple[int, int], tuple[int, int]], ...], bool] | None:
''',
    '''def _route_specs(
    patch: Any,
    block_count: int,
    seq_len: int,
    uuids: tuple[Any, ...],
    layout_identity: tuple[Any, ...],
    options: dict[str, Any],
    sparse_runtime_ok: bool,
    pool_specs: tuple[tuple[int, int, str | None], ...],
    measure: CoreBSAMeasureAudit | None = None,
) -> tuple[tuple[tuple[str, tuple[int, int], tuple[int, int]], ...], bool] | None:
''',
)
replace_once(
    CORE,
    '''    sink, sink_q = sinks
    dense_set = set(dense_blocks)
''',
    '''    sink, sink_q = sinks
    if measure is not None:
        sink = measure.exact_k_block_range
    dense_set = set(dense_blocks)
''',
)
replace_once(
    CORE,
    '''        if not sparse_runtime_ok:
            safe = False
            specs.append(("h3_sparse_unproven", sink, sink_q))
            continue
        state = _pool_entry(patch, (index, seq_len, uuids), pool_specs[index])
''',
    '''        if not sparse_runtime_ok or (measure is not None and not measure.chunked_key_bias):
            safe = False
            specs.append(("h3_sparse_unproven", sink, sink_q))
            continue
        key = measure.pool_keys[index] if measure is not None else (index, seq_len, uuids)
        state = _pool_entry(patch, key, pool_specs[index])
''',
)

replace_once(
    CORE,
    '''        sparse_runtime_ok = _sparse_runtime_eligible(model, module)
        routed = _route_specs(
''',
    '''        measure_audit, measure_reason = _measure_audit(
            module,
            source_blob,
            patch,
            options,
            layout,
            seq_len,
            uuids,
            layout_identity,
            len(model.blocks),
        )
        if measure_reason is not None:
            return None, measure_reason

        sparse_runtime_ok = _sparse_runtime_eligible(model, module)
        routed = _route_specs(
''',
)
replace_once(
    CORE,
    '''            sparse_runtime_ok,
            pool_specs,
        )
''',
    '''            sparse_runtime_ok,
            pool_specs,
            measure_audit,
        )
''',
)
replace_once(
    CORE,
    '''        pool_ownership = _pool_ownership_identity(
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
''',
    '''        pool_ownership = _pool_ownership_identity(
            patch,
            len(model.blocks),
            seq_len,
            uuids,
            pool_specs,
            measure_audit,
        )
        mode = "sol-attn" if settings_identity[2] == 0.0 else "sla"
        measure_identity = (
            ()
            if measure_audit is None
            else (("attention_measure", _measure_identity(measure_audit)),)
        )
        identity = (
''',
)
replace_once(
    CORE,
    '''            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            ("ownership", ownership_identity),
''',
    '''            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            *measure_identity,
            ("ownership", ownership_identity),
''',
)
replace_once(
    CORE,
    '''        return CoreBSAAudit(
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
''',
    '''        audit = CoreBSAAudit(
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
            expected_receipts=(),
            source_blob=source_blob,
            current_override=options.get("optimized_attention_override"),
            measure=measure_audit,
        )
        audit.expected_receipts = _expected_receipts(audit)
        return audit, None
''',
)

replace_once(
    CORE,
    '''    key = (index, audit.seq_len, audit.uuids)
    replacement_closure = _closure_values(replacement)
''',
    '''    key = _pool_key(audit, index)
    replacement_closure = _closure_values(replacement)
''',
)
replace_once(
    CORE,
    '''            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)
''',
    '''            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls, metadata_ok
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if not _measure_runtime_matches(
                    audit,
                    index,
                    call_args.get("transformer_options"),
                    sparse_selected=sparse_selected,
                ):
                    metadata_ok = False
                    audit.failure = "actual_measure_metadata_failed"
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)
''',
)
replace_once(
    CORE,
    '''            receipt = (
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
''',
    '''            receipt = _receipt(
                audit,
                index,
                observed,
                completed=bool(metadata_ok and observed == expected_route),
            )
            receipts.append(receipt)
''',
)
replace_once(
    CORE,
    '''    for receipt in receipts:
        if not isinstance(receipt, tuple) or len(receipt) != 8:
            return False
''',
    '''    for receipt in receipts:
        if not isinstance(receipt, tuple) or len(receipt) < 8:
            return False
''',
)

# Loader-aware path duplicates the canonical probe because ComfyUI can path-load
# builtin nodes. Keep it numerically identical to the canonical measure audit.
replace_once(
    LOADER,
    '''        sparse_runtime_ok = core_bsa_compat._sparse_runtime_eligible(model, module)
        routed = core_bsa_compat._route_specs(
''',
    '''        measure_audit, measure_reason = core_bsa_compat._measure_audit(
            module,
            source_blob,
            patch,
            options,
            layout,
            seq_len,
            uuids,
            layout_identity,
            len(model.blocks),
        )
        if measure_reason is not None:
            return None, measure_reason

        sparse_runtime_ok = core_bsa_compat._sparse_runtime_eligible(model, module)
        routed = core_bsa_compat._route_specs(
''',
)
replace_once(
    LOADER,
    '''            sparse_runtime_ok,
            pool_specs,
        )
''',
    '''            sparse_runtime_ok,
            pool_specs,
            measure_audit,
        )
''',
)
replace_once(
    LOADER,
    '''        pool_ownership = core_bsa_compat._pool_ownership_identity(
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
''',
    '''        pool_ownership = core_bsa_compat._pool_ownership_identity(
            patch,
            len(model.blocks),
            seq_len,
            uuids,
            pool_specs,
            measure_audit,
        )
        mode = "sol-attn" if settings_identity[2] == 0.0 else "sla"
        measure_identity = (
            ()
            if measure_audit is None
            else (("attention_measure", core_bsa_compat._measure_identity(measure_audit)),)
        )
        identity = (
''',
)
replace_once(
    LOADER,
    '''            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            ("ownership", ownership_identity),
''',
    '''            ("routes", route_specs),
            ("pool_ownership", pool_ownership),
            *measure_identity,
            ("ownership", ownership_identity),
''',
)
replace_once(
    LOADER,
    '''        return core_bsa_compat.CoreBSAAudit(
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
''',
    '''        audit = core_bsa_compat.CoreBSAAudit(
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
            expected_receipts=(),
            source_blob=source_blob,
            current_override=options.get("optimized_attention_override"),
            measure=measure_audit,
        )
        audit.expected_receipts = core_bsa_compat._expected_receipts(audit)
        return audit, None
''',
)

# Flow's outer wrapper must observe the same measure-bound calibration key and
# validate the final inner mixed options, not only the carrier options it receives.
replace_once(
    FLOW,
    '''    return core_bsa_compat._pool_entry(
        audit.patch,
        (index, audit.seq_len, audit.uuids),
        audit.pool_specs[index],
    )
''',
    '''    return core_bsa_compat._pool_entry(
        audit.patch,
        core_bsa_compat._pool_key(audit, index),
        audit.pool_specs[index],
    )
''',
)
replace_once(
    FLOW,
    '''            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)
''',
    '''            def audited_original_block(call_args):
                nonlocal sparse_selected, original_block_calls, metadata_ok
                original_block_calls += 1
                sparse_selected = call_args.get("attention") is expected_attention
                if not core_bsa_compat._measure_runtime_matches(
                    audit,
                    index,
                    call_args.get("transformer_options"),
                    sparse_selected=sparse_selected,
                ):
                    metadata_ok = False
                    audit.failure = "actual_measure_metadata_failed"
                if sparse_selected:
                    from .bsa_transition_probe import attention
                    call_args = {**call_args, "attention": attention(expected_attention, audit, index)}
                return original_block(call_args)
''',
)
replace_once(
    FLOW,
    '''            receipts.append(
                (
                    core_bsa_compat.ADAPTER_KEY,
                    core_bsa_compat.ADAPTER_VERSION,
                    audit.patch_generation,
                    index,
                    observed,
                    audit.seq_len,
                    expected_sink,
                    expected_sink_q,
                )
            )
''',
    '''            receipts.append(
                core_bsa_compat._receipt(
                    audit,
                    index,
                    observed,
                    completed=bool(metadata_ok and observed == expected_route),
                )
            )
''',
)

# Preserve the accepted cold->primed proof while switching only its pool lookup
# to the exact measure-bound key used by core BSA.
replace_all(
    RECOVERY,
    '''            (index, audit.seq_len, audit.uuids),
''',
    '''            core_bsa_compat._pool_key(audit, index),
''',
    2,
)

# Add one source-pinned fixture lane for the new core measure implementation.
replace_once(
    WORKFLOW,
    '''          - comfy_ref: be92396834f9b6e3e361cfe30e7eed693137a448
            python_version: "3.12"
            comfy_kitchen_version: "0.2.33"
            comfy_aimdo_version: "0.5.3"
            fixture_required: true
''',
    '''          - comfy_ref: be92396834f9b6e3e361cfe30e7eed693137a448
            python_version: "3.12"
            comfy_kitchen_version: "0.2.33"
            comfy_aimdo_version: "0.5.3"
            fixture_required: true
          - comfy_ref: efd54b9a17ce422a2cdbed76d5e7a87671f253c2
            comfy_repository: xmarre/ComfyUI
            python_version: "3.12"
            comfy_kitchen_version: "0.2.33"
            comfy_aimdo_version: "0.5.3"
            measure_fixture_required: true
''',
)
replace_once(
    WORKFLOW,
    '''          repository: comfyanonymous/ComfyUI
          ref: ${{ matrix.comfy_ref }}
''',
    '''          repository: ${{ matrix.comfy_repository || 'comfyanonymous/ComfyUI' }}
          ref: ${{ matrix.comfy_ref }}
''',
)
replace_once(
    WORKFLOW,
    '''            comfyui_spectrum_h3/core_bsa_loader_compat.py \\
            comfyui_spectrum_h3/core_bsa_preprocess_compat.py \\
''',
    '''            comfyui_spectrum_h3/core_bsa_loader_compat.py \\
            comfyui_spectrum_h3/core_bsa_preprocess_compat.py \\
''',
)
replace_once(
    WORKFLOW,
    '''            tests/test_core_bsa_forecast_recovery.py \\
            tests/test_core_bsa_preprocess_compat.py \\
''',
    '''            tests/test_core_bsa_forecast_recovery.py \\
            tests/test_core_bsa_measure_history.py \\
            tests/test_core_bsa_preprocess_compat.py \\
''',
)
replace_once(
    WORKFLOW,
    '''          SPECTRUM_REQUIRE_REVIEWED_BSA_FIXTURE: ${{ matrix.fixture_required && '1' || '0' }}
''',
    '''          SPECTRUM_REQUIRE_REVIEWED_BSA_FIXTURE: ${{ matrix.fixture_required && '1' || '0' }}
          SPECTRUM_REQUIRE_REVIEWED_BSA_MEASURE_FIXTURE: ${{ matrix.measure_fixture_required && '1' || '0' }}
''',
)

Path("tests/test_core_bsa_measure_history.py").write_text(
    r'''from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat


def _measure_nodes():
    required = os.getenv("SPECTRUM_REQUIRE_REVIEWED_BSA_MEASURE_FIXTURE") == "1"
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        if required:
            pytest.fail(f"required measure-capable core BSA fixture failed to import: {exc}")
        pytest.skip(f"measure-capable core BSA fixture unavailable: {exc}")
    blob = core_bsa_compat._module_blob_sha(nodes)
    if blob not in core_bsa_compat.MEASURE_CAPABLE_BSA_GIT_BLOBS:
        if required:
            pytest.fail(f"required measure-capable core BSA blob not loaded: {blob}")
        pytest.skip("this ComfyUI fixture is not the reviewed measure-capable BSA source")
    return nodes


class _Attention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 128
        self.qkv_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16, device="cpu")
        )

    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x


class _Block:
    def __init__(self):
        self.attn = _Attention()


class _Model:
    def __init__(self, count=2):
        self.blocks = [_Block() for _ in range(count)]
        self.dtype = torch.bfloat16


def _layout():
    return SimpleNamespace(
        seq_len=192,
        signature=(64, 2, 16, 16, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, 192, "video"),
        ],
    )


def _request(*, source_grid=(4, 8), prefix_grid=(8, 8)):
    source_rows = source_grid[0] * source_grid[1]
    prefix_rows = prefix_grid[0] * prefix_grid[1]
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
        "q_rows": 192,
        "kv_rows": 192,
        "video_start": 96,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": list(source_grid),
        "prefix_grid": list(prefix_grid),
        "segments": [
            {"start": 0, "stop": 96, "mass_num": 1, "mass_den": 1},
            {
                "start": 96,
                "stop": 96 + prefix_rows,
                "mass_num": source_rows,
                "mass_den": prefix_rows,
            },
            {
                "start": 96 + prefix_rows,
                "stop": 192,
                "mass_num": 1,
                "mass_den": 1,
            },
        ],
    }


def _installation(monkeypatch, *, count=2):
    nodes = _measure_nodes()
    model = _Model(count)
    patch = nodes.SparseAttnPatch(
        tau=1.0,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=0,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {
            "dit": {
                ("double_block", index): nodes.make_h3_block_patch(block, index, patch)
                for index, block in enumerate(model.blocks)
            }
        },
        "sigmas": torch.tensor([0.5]),
        "uuids": ("positive",),
        "minimax_h3_layout": _layout(),
        core_bsa_compat.ATTENTION_MEASURE_KEY: _request(),
    }
    nodes.measure.register(patch, options)
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    monkeypatch.setattr(nodes.measure, "supports_key_bias", lambda _provider: True)
    return nodes, model, patch, options


def test_measure_fixture_is_source_pinned_when_required():
    nodes = _measure_nodes()
    assert core_bsa_compat._module_blob_sha(nodes) in core_bsa_compat.MEASURE_CAPABLE_BSA_GIT_BLOBS
    assert core_bsa_compat._module_blob_sha(nodes.measure) in core_bsa_compat.AUDITED_BSA_MEASURE_GIT_BLOBS
    import comfy.attention_measure as core_measure
    assert core_bsa_compat._module_blob_sha(core_measure) in core_bsa_compat.AUDITED_ATTENTION_MEASURE_GIT_BLOBS


def test_weighted_measure_changes_exact_sink_and_uses_measure_bound_pool(monkeypatch):
    _nodes, model, patch, options = _installation(monkeypatch)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None
    assert audit is not None and audit.safe and audit.measure is not None
    assert audit.measure.semantic_digest
    assert audit.measure.exact_k_block_range == (0, 3)
    assert {spec[0] for spec in audit.route_specs} == {"h3_chunked_sparse_cold"}
    assert {spec[1] for spec in audit.route_specs} == {(0, 3)}
    assert len(audit.expected_receipts[0]) == 9
    measure_receipt = audit.expected_receipts[0][8]
    assert measure_receipt[0] == core_bsa_compat.ATTENTION_MEASURE_KEY
    assert measure_receipt[2] == 0
    assert measure_receipt[5] == core_bsa_compat.SPARSE_MEASURE_PROFILE
    assert measure_receipt[6] == core_bsa_compat.SPARSE_MEASURE_ROUTE
    assert measure_receipt[-1] is True

    shape = (model.blocks[0].attn.heads, model.blocks[0].attn.head_dim)
    for index in range(len(model.blocks)):
        key = audit.measure.pool_keys[index]
        assert key != (index, audit.seq_len, audit.uuids)
        patch.pooled[key] = (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    primed, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and primed is not None and primed.safe
    assert {spec[0] for spec in primed.route_specs} == {"h3_chunked_sparse_primed"}


def test_measure_digest_and_pool_key_change_for_anisotropic_grid_identity(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch)
    first, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and first is not None and first.measure is not None

    changed = dict(options)
    changed[core_bsa_compat.ATTENTION_MEASURE_KEY] = _request(
        source_grid=(8, 4), prefix_grid=(8, 8)
    )
    second, reason = core_bsa_compat.probe(changed, _layout(), model)
    assert reason is None and second is not None and second.measure is not None
    assert first.measure.semantic_digest != second.measure.semantic_digest
    assert first.measure.pool_keys != second.measure.pool_keys
    assert first.identity != second.identity


def test_foreign_measure_capability_fails_closed(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch)
    registry = dict(options[core_bsa_compat.ATTENTION_MEASURE_CAPABILITIES_KEY])
    registry[core_bsa_compat.CORE_BSA_MEASURE_PROVIDER] = object()
    options[core_bsa_compat.ATTENTION_MEASURE_CAPABILITIES_KEY] = registry
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "measure_capability_unproven"


def test_missing_chunked_key_bias_is_actual_only_for_sparse_route(monkeypatch):
    nodes, model, _patch, options = _installation(monkeypatch)
    monkeypatch.setattr(nodes.measure, "supports_key_bias", lambda _provider: False)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None
    assert not audit.safe
    assert {spec[0] for spec in audit.route_specs} == {"h3_sparse_unproven"}


def test_measure_disappearance_between_preflight_and_actual_is_rejected(monkeypatch):
    _nodes, model, _patch, options = _installation(monkeypatch, count=1)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is not None
    call_options = dict(options)
    call_options.pop(core_bsa_compat.ATTENTION_MEASURE_KEY)
    assert not core_bsa_compat._measure_runtime_matches(
        audit,
        0,
        call_options,
        sparse_selected=True,
    )


def test_unweighted_fixture_keeps_legacy_eight_field_receipt(monkeypatch):
    nodes = _measure_nodes()
    model = _Model(1)
    patch = nodes.SparseAttnPatch(
        tau=1.0,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=0,
        verbose=False,
    )
    override = nodes.make_attention_override(patch, None)
    patch.installed.add(override)
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {
            "dit": {("double_block", 0): nodes.make_h3_block_patch(model.blocks[0], 0, patch)}
        },
        "sigmas": torch.tensor([0.5]),
        "uuids": ("positive",),
    }
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.measure is None
    assert len(audit.expected_receipts[0]) == 8
''',
    encoding="utf-8",
)
