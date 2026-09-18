"""Reference fallback for ComfyUI core H3 BlockSparseAttention on Keyless H3.

Core ComfyUI's MiniMax-H3 sparse producer owns a QKV-specific block replacement:
it projects ``attn.qkv_proj`` and reads ``attn.k_norm`` before the ordinary model
attention method can run. A Keyless block has neither object. When Spectrum can
prove the exact reviewed core-BSA ownership, this adapter removes only those
per-call QKV producer replacements from Spectrum's local transformer-options copy.
The real Keyless attention then executes through its materialized Q/route/V path.

Current canonical Untwist composes through Keyless's routing-preprocessor chain and
therefore survives this BSA removal without owning attention. The older reviewed
logical-K Untwist override is still preserved exactly once when it is the sole layer
immediately above or below core BSA. Unknown attention wrappers remain fail-closed:
silently discarding them could change the numerical workflow even if doing so
happened to avoid the QKV attribute crash. The shared ModelPatcher is never mutated.

The bypass history identity binds the complete remaining Keyless numerical route
after BSA removal. Opaque Keyless providers or receipt policies remain actual-only;
source-proving the removed BSA producer does not qualify unrelated providers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Any

from . import (
    backend_history,
    core_bsa_compat,
    core_bsa_loader_compat,
    core_bsa_preprocess_compat,
    keyless_runtime_compat,
    source_code_audit,
)
from .keyless_compat import keyless_semantic_identity, validate_keyless_contract

BYPASS_KEY = "spectrum_keyless_core_bsa_reference_bypass_v1"
BYPASS_VERSION = 2

# xmarre/ComfyUI PR #7, based on Comfy-Org/ComfyUI master
# 9a77c1db9eff68d7320dcd98f0757b4450161d4b. This source explicitly recognizes
# h3_keyless_core50_v1 and keeps Keyless on the generic materialized Q/route(V)/V
# override without installing the native H3 QKV block producer.
KEYLESS_AWARE_BSA_GIT_BLOBS = frozenset(
    {"3f508f899f5d5629d8a5b31d9ef2c26ccd7ae919"}
)

_ORIGINAL_PREPARE = None
_ORIGINAL_PREFLIGHT = None
_ORIGINAL_OBSERVE = None
_INSTALLED = False


@dataclass(frozen=True)
class KeylessAwareCoreBSAProof:
    module: Any
    source_blob: str
    patch: Any
    settings_identity: tuple[Any, ...]
    patch_generation: int


def _keyless_aware_module_from_override(override: Any) -> tuple[Any, str] | None:
    """Resolve only the reviewed Keyless-aware generic Core-BSA override source."""
    base = getattr(override, "__func__", override)
    if getattr(base, "__qualname__", None) != "make_attention_override.<locals>.override":
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
    extras = core_bsa_loader_compat._comfy_extras_dir()
    if (
        extras is None
        or source_path != code_path
        or source_path.name != "nodes_sparse_attention.py"
        or source_path.parent != extras
    ):
        return None

    blob = core_bsa_compat._module_blob_sha(module)
    if blob not in KEYLESS_AWARE_BSA_GIT_BLOBS:
        return None
    required = (
        "SparseAttnPatch",
        "make_attention_override",
        "make_h3_block_patch",
        "_keyless_h3_contract",
        "KEYLESS_H3_CONTRACT_KEY",
        "KEYLESS_H3_ARCHITECTURE",
        "HEAD_DIM",
        "BLOCK_SIZE",
        "PRODUCER_CHUNK",
    )
    if any(not hasattr(module, name) for name in required):
        return None
    if (
        module.KEYLESS_H3_CONTRACT_KEY != "minimax_h3_keyless_contract_v1"
        or module.KEYLESS_H3_ARCHITECTURE != "h3_keyless_core50_v1"
    ):
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
        ("make_attention_override", "override"),
    ):
        return None
    if not core_bsa_loader_compat._reviewed_defaults(
        base,
        "make_attention_override",
        "override",
    ):
        return None
    return module, blob


def _keyless_aware_bsa_proof(
    options: dict[str, Any],
    model: Any,
) -> KeylessAwareCoreBSAProof | None:
    """Prove Core BSA's Keyless-aware generic override without promoting forecasts.

    The reviewed source executes only the already-materialized Q/route(V)/V
    override for Keyless and installs no QKV-only H3 block producer. Spectrum
    therefore leaves it intact. Backend history remains actual-only until a
    separate route/receipt contract proves dense/sparse transitions.
    """
    current = options.get("optimized_attention_override")
    resolved = _keyless_aware_module_from_override(current)
    if resolved is None:
        return None
    module, source_blob = resolved

    try:
        if module._keyless_h3_contract(model) is None:
            return None
    except (AttributeError, TypeError, ValueError):
        return None

    closure = core_bsa_compat._closure_values(current)
    if closure is None or set(closure) != {"patch", "previous"}:
        return None
    patch = closure["patch"]
    if type(patch) is not module.SparseAttnPatch:
        return None
    if closure["previous"] is not None:
        # A lower optimized-attention owner needs its own Keyless audit. Canonical
        # Untwist v0.2.4+ routes through the Keyless routing-preprocessor chain
        # instead and therefore does not require an inherited attention override.
        return None
    if current not in getattr(patch, "installed", set()):
        return None

    settings = core_bsa_compat._settings_identity(patch)
    if settings is None or settings[0]:
        return None

    # The defining Keyless-aware property is absence of the native H3 sparse
    # producer. Preserve unrelated block owners such as ControlNet, but reject
    # any Core-BSA block replacement even if someone forged the marker manually.
    block_patch_code = core_bsa_compat._nested_code(module.make_h3_block_patch, "block_patch")
    dit = (
        options.get("patches_replace", {})
        .get("dit", {})
        if isinstance(options.get("patches_replace", {}), dict)
        else {}
    )
    if not isinstance(dit, dict):
        return None
    if block_patch_code is not None and any(
        getattr(getattr(replacement, "__func__", replacement), "__code__", None)
        is block_patch_code
        for replacement in dit.values()
    ):
        return None

    return KeylessAwareCoreBSAProof(
        module=module,
        source_blob=source_blob,
        patch=patch,
        settings_identity=settings,
        patch_generation=core_bsa_compat._lifetime_generation(patch),
    )


@dataclass(frozen=True)
class CoreBSAReferenceProof:
    module: Any
    source_blob: str
    patch: Any
    settings_identity: tuple[Any, ...]
    patch_generation: int
    restored_override: Any | None = None
    preprocess_identity: tuple[Any, ...] | None = None

    def identity(self, model: Any) -> tuple[Any, ...]:
        semantic = keyless_semantic_identity(model)
        if semantic is None:
            raise RuntimeError("Keyless core-BSA fallback was asked to identify a native-QKV model")
        return (
            BYPASS_KEY,
            BYPASS_VERSION,
            semantic,
            self.source_blob,
            self.patch_generation,
            self.settings_identity,
            ("outer_preprocess", self.preprocess_identity),
            "dense_materialized_route",
        )


def _proof_from_module(
    module: Any,
    source_blob: str,
    options: dict[str, Any],
    model: Any,
    *,
    restored_override: Any | None = None,
    preprocess_identity: tuple[Any, ...] | None = None,
):
    ownership = core_bsa_compat._replacement_ownership(module, model, options)
    if ownership is None:
        return None
    patch, _ownership_identity = ownership
    settings = core_bsa_compat._settings_identity(patch)
    if settings is None:
        return None
    return CoreBSAReferenceProof(
        module=module,
        source_blob=str(source_blob),
        patch=patch,
        settings_identity=settings,
        patch_generation=core_bsa_compat._lifetime_generation(patch),
        restored_override=restored_override,
        preprocess_identity=preprocess_identity,
    )


def _resolve_bsa_proof(
    options: dict[str, Any],
    model: Any,
    *,
    restored_override: Any | None = None,
    preprocess_identity: tuple[Any, ...] | None = None,
) -> CoreBSAReferenceProof | None:
    """Prove one exact reviewed BSA layer, canonical import or live loader alias."""
    module, blob = core_bsa_compat._load_audited_module()
    if module is not None and blob is not None:
        proof = _proof_from_module(
            module,
            blob,
            options,
            model,
            restored_override=restored_override,
            preprocess_identity=preprocess_identity,
        )
        if proof is not None:
            return proof

    resolved = core_bsa_loader_compat._runtime_module_for_options(options, model)
    if resolved is None:
        return None
    module, blob = resolved
    if blob not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        return None
    return _proof_from_module(
        module,
        blob,
        options,
        model,
        restored_override=restored_override,
        preprocess_identity=preprocess_identity,
    )


def _reviewed_untwist_restore(
    override: Any,
    options: dict[str, Any],
) -> tuple[Any, tuple[Any, ...]] | None:
    """Accept a standalone reviewed legacy Untwist layer whose inheritance is dense."""
    contract = getattr(override, "attention_preprocess_v1", None)
    if not isinstance(contract, tuple) or len(contract) != 2:
        return None
    transform, inherited = contract
    if inherited is not None:
        return None
    preprocess_identity = core_bsa_preprocess_compat._audited_untwist_preprocess(transform)
    runtime_identity = core_bsa_preprocess_compat._untwist_runtime_identity(options)
    if preprocess_identity is None or runtime_identity is None:
        return None
    return override, (preprocess_identity, runtime_identity)


def _bsa_previous(override: Any) -> Any:
    closure = core_bsa_compat._closure_values(override)
    if closure is None or "previous" not in closure:
        return ...
    return closure["previous"]


def _rebuild_outer_untwist_without_bsa(
    options: dict[str, Any],
    bsa_override: Any,
    preprocess_identity: tuple[Any, ...],
) -> Any | None:
    """Recreate reviewed legacy Untwist with dense inheritance after removing BSA."""
    if _bsa_previous(bsa_override) is not None:
        return None
    current = options.get("optimized_attention_override")
    contract = getattr(current, "attention_preprocess_v1", None)
    if not isinstance(contract, tuple) or len(contract) != 2:
        return None
    transform, previous = contract
    if previous is not bsa_override:
        return None
    module = core_bsa_preprocess_compat._loaded_untwist_module(
        getattr(transform, "__func__", transform)
    )
    if module is None:
        return None
    factory = getattr(module, "make_minimax_h3_attention_override", None)
    if not callable(factory):
        return None
    rebuilt = factory(None)
    rebuilt_contract = getattr(rebuilt, "attention_preprocess_v1", None)
    if not isinstance(rebuilt_contract, tuple) or len(rebuilt_contract) != 2:
        return None
    rebuilt_transform, rebuilt_previous = rebuilt_contract
    if rebuilt_previous is not None:
        return None
    if (
        core_bsa_preprocess_compat._audited_untwist_preprocess(rebuilt_transform)
        != preprocess_identity[0]
    ):
        return None
    return rebuilt


def _resolve_direct_core_bsa(options: dict[str, Any], model: Any) -> CoreBSAReferenceProof | None:
    """Resolve reviewed BSA directly or through the one reviewed legacy Untwist layer."""
    current = options.get("optimized_attention_override")

    if core_bsa_compat._looks_like_core_bsa_callable(
        current, "make_attention_override", "override"
    ):
        previous = _bsa_previous(current)
        if previous is ...:
            return None
        if previous is None:
            return _resolve_bsa_proof(options, model)
        restored = _reviewed_untwist_restore(previous, options)
        if restored is None:
            return None
        restored_override, preprocess_identity = restored
        return _resolve_bsa_proof(
            options,
            model,
            restored_override=restored_override,
            preprocess_identity=preprocess_identity,
        )

    bsa_override, preprocess_identity, reason = (
        core_bsa_preprocess_compat._unwrap_reviewed_untwist(options)
    )
    if bsa_override is None or preprocess_identity is None or reason is not None:
        return None
    restored_override = _rebuild_outer_untwist_without_bsa(
        options,
        bsa_override,
        preprocess_identity,
    )
    if restored_override is None:
        return None
    normalized = dict(options)
    normalized["optimized_attention_override"] = bsa_override
    return _resolve_bsa_proof(
        normalized,
        model,
        restored_override=restored_override,
        preprocess_identity=preprocess_identity,
    )


def _route_bound_bypass_identity(
    proof: CoreBSAReferenceProof,
    model: Any,
    options: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        *proof.identity(model),
        (
            "keyless_runtime_identity",
            keyless_runtime_compat._keyless_runtime_identity(options),
        ),
    )


def _bypass_forecast_safe(options: dict[str, Any], identity: Any) -> bool:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 9
        or identity[0] != BYPASS_KEY
        or identity[1] != BYPASS_VERSION
        or not isinstance(identity[-1], tuple)
        or len(identity[-1]) != 2
        or identity[-1][0] != "keyless_runtime_identity"
    ):
        return False
    current_runtime = keyless_runtime_compat._keyless_runtime_identity(options)
    if identity[-1][1] != current_runtime:
        return False

    # BSA ownership proves only the removed QKV producer. It must not promote an
    # unrelated opaque Keyless provider or receipt-policy composition to forecast-safe.
    if options.get(keyless_runtime_compat._KEYLESS_PROVIDER) is not None:
        return False
    policies = options.get(backend_history.POLICIES)
    if isinstance(policies, dict) and policies:
        return False
    if policies not in (None, {}, ()):
        return False

    override = options.get("optimized_attention_override")
    if override is None:
        return True

    # The only override that prepare_reference_options may intentionally restore is
    # the source-gated legacy Untwist layer represented in the BSA proof itself.
    outer_preprocess = identity[6]
    if (
        not isinstance(outer_preprocess, tuple)
        or len(outer_preprocess) != 2
        or outer_preprocess[0] != "outer_preprocess"
        or outer_preprocess[1] is None
    ):
        return False
    restored = _reviewed_untwist_restore(override, options)
    return restored is not None and restored[1] == outer_preprocess[1]


def prepare_reference_options(
    options: dict[str, Any], model: Any
) -> tuple[dict[str, Any], tuple[Any, ...] | None]:
    """Return a per-call dense Keyless reference route for exact core H3 BSA.

    Native-QKV models and Keyless models without core-BSA evidence are returned
    unchanged. Any BSA evidence on Keyless must be completely source/ownership
    proven before it can be removed; otherwise execution aborts before the
    incompatible QKV producer reaches the model.
    """
    if validate_keyless_contract(model) is None:
        return options, None
    if not core_bsa_compat.has_core_bsa_evidence(options):
        return options, None

    keyless_aware = _keyless_aware_bsa_proof(options, model)
    if keyless_aware is not None:
        # This source no longer carries a QKV-only block producer on Keyless, so
        # there is nothing to strip. Keep the sparse materialized override intact.
        # The underlying backend-history stack sees this new source as unreported
        # and conservatively forces actual calls until a dedicated route contract
        # is validated.
        return options, None

    proof = _resolve_direct_core_bsa(options, model)
    if proof is None:
        raise RuntimeError(
            "Keyless H3 detected core BlockSparseAttention state that Spectrum cannot "
            "prove as the reviewed H3 producer with a Keyless-safe inherited attention "
            "route. Refusing to execute or silently discard an opaque QKV/attention "
            "composition against h3_keyless_core50_v1. Remove/reorder the sparse "
            "attention composition or use a reviewed Keyless-aware provider."
        )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("validated Keyless H3 model unexpectedly has no core blocks")
    copied = dict(options)
    patches = dict(copied.get("patches_replace") or {})
    dit = dict(patches.get("dit") or {})
    for index in range(len(blocks)):
        key = ("double_block", index)
        if key not in dit:
            raise RuntimeError(
                f"reviewed core-BSA ownership lost Keyless block replacement {index} during fallback"
            )
        dit.pop(key)
    patches["dit"] = dit
    copied["patches_replace"] = patches

    if proof.restored_override is None:
        copied.pop("optimized_attention_override", None)
    else:
        copied["optimized_attention_override"] = proof.restored_override
    copied.pop(core_bsa_compat.PRIVATE_AUDIT_KEY, None)
    identity = _route_bound_bypass_identity(proof, model, copied)
    copied[BYPASS_KEY] = identity
    return copied, identity


def _preflight(options: dict[str, Any], layout: Any, model: Any):
    identity = options.get(BYPASS_KEY)
    if identity is not None:
        safe = _bypass_forecast_safe(options, identity)
        return identity if safe else (BYPASS_KEY, "invalid_or_opaque_bypass", identity), safe, None
    if _ORIGINAL_PREFLIGHT is None:
        raise RuntimeError("Keyless core-BSA fallback was not installed")
    return _ORIGINAL_PREFLIGHT(options, layout, model)


def _prepare(runtime, run_id, step_id, options, layout, model):
    if _ORIGINAL_PREPARE is None:
        raise RuntimeError("Keyless core-BSA fallback was not installed")
    prepared, _identity = prepare_reference_options(options, model)
    return _ORIGINAL_PREPARE(runtime, run_id, step_id, prepared, layout, model)


def _observe(runtime, run_id, step_id, options, policy):
    if policy is not None:
        identity, safe = policy
        if (
            safe
            and isinstance(identity, tuple)
            and identity
            and identity[0] == BYPASS_KEY
            and options.get(BYPASS_KEY) == identity
            and _bypass_forecast_safe(options, identity)
        ):
            runtime.observe_backend_history(run_id, step_id, identity, (), True)
            return
    if _ORIGINAL_OBSERVE is None:
        raise RuntimeError("Keyless core-BSA fallback was not installed")
    return _ORIGINAL_OBSERVE(runtime, run_id, step_id, options, policy)


def install_keyless_core_bsa_fallback() -> None:
    """Wrap the final backend-history stack after core-BSA forecast recovery."""
    global _INSTALLED, _ORIGINAL_PREPARE, _ORIGINAL_PREFLIGHT, _ORIGINAL_OBSERVE
    if _INSTALLED:
        return
    _ORIGINAL_PREPARE = backend_history.prepare
    _ORIGINAL_PREFLIGHT = backend_history._preflight
    _ORIGINAL_OBSERVE = backend_history.observe
    backend_history._preflight = _preflight
    backend_history.prepare = _prepare
    backend_history.observe = _observe
    _INSTALLED = True
