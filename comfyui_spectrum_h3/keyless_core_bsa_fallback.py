"""Reference fallback for ComfyUI core H3 BlockSparseAttention on Keyless H3.

Core ComfyUI's MiniMax-H3 sparse producer owns a QKV-specific block replacement:
it projects ``attn.qkv_proj`` and reads ``attn.k_norm`` before the ordinary model
attention method can run. A Keyless block has neither object. When Spectrum can
prove the exact reviewed core-BSA ownership, this adapter removes only those
per-call QKV producer replacements and the BSA attention override from Spectrum's
local transformer-options copy. The real Keyless attention then executes through
its dense/materialized-route reference path.

Unknown, wrapped or source-moved BSA ownership fails closed instead of executing a
QKV producer against a Keyless block. The shared ModelPatcher is never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import backend_history, core_bsa_compat, core_bsa_loader_compat
from .keyless_compat import keyless_semantic_identity, validate_keyless_contract

BYPASS_KEY = "spectrum_keyless_core_bsa_reference_bypass_v1"
BYPASS_VERSION = 1

_ORIGINAL_PREPARE = None
_ORIGINAL_PREFLIGHT = None
_ORIGINAL_OBSERVE = None
_INSTALLED = False


@dataclass(frozen=True)
class CoreBSAReferenceProof:
    module: Any
    source_blob: str
    patch: Any
    settings_identity: tuple[Any, ...]
    patch_generation: int

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
            "dense_materialized_route",
        )


def _proof_from_module(module: Any, source_blob: str, options: dict[str, Any], model: Any):
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
    )


def _resolve_direct_core_bsa(options: dict[str, Any], model: Any) -> CoreBSAReferenceProof | None:
    """Resolve only exact reviewed direct H3-BSA ownership, canonical or loader alias."""
    module, blob = core_bsa_compat._load_audited_module()
    if module is not None and blob is not None:
        proof = _proof_from_module(module, blob, options, model)
        if proof is not None:
            return proof

    resolved = core_bsa_loader_compat._runtime_module_for_options(options, model)
    if resolved is None:
        return None
    module, blob = resolved
    if blob not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        return None
    return _proof_from_module(module, blob, options, model)


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

    proof = _resolve_direct_core_bsa(options, model)
    if proof is None:
        raise RuntimeError(
            "Keyless H3 detected core BlockSparseAttention state that Spectrum cannot "
            "prove as the reviewed direct H3 producer. Refusing to execute an opaque "
            "QKV block replacement against h3_keyless_core50_v1. Remove/reorder the "
            "sparse-attention composition or use a reviewed Keyless-aware provider."
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

    # Reference means reference: do not pass materialized Keyless Q/route/V through
    # the core-BSA override or an opaque override that happened to sit below it.
    # Comfy's normal dense optimized_attention selection remains available.
    copied.pop("optimized_attention_override", None)
    copied.pop(core_bsa_compat.PRIVATE_AUDIT_KEY, None)
    identity = proof.identity(model)
    copied[BYPASS_KEY] = identity
    return copied, identity


def _preflight(options: dict[str, Any], layout: Any, model: Any):
    identity = options.get(BYPASS_KEY)
    if identity is not None:
        if not isinstance(identity, tuple) or not identity or identity[0] != BYPASS_KEY:
            return (BYPASS_KEY, "invalid_marker"), False, None
        return identity, True, None
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
