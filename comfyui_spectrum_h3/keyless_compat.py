from __future__ import annotations

from typing import Any

from .minimax_h3 import is_native_minimax_h3, locate_minimax_h3_inner

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_PROJECTION_ATTR = "qv_proj"
KEYLESS_PROJECTION_ROWS = 14_336
KEYLESS_HIDDEN_SIZE = 5_376
KEYLESS_HEADS = 56
KEYLESS_HEAD_DIM = 128
KEYLESS_INNER_DIM = 7_168
KEYLESS_CORE_BLOCKS = 50
KEYLESS_TOKEN_REFINER_BLOCKS = 2
KEYLESS_NORM_EPSILON = 1e-5
KEYLESS_ROPE_POLICY = "h3_split_half_96_v1"
KEYLESS_QV_ORDER = "q_effective;v"

_COMMON_H3_ATTRIBUTES = (
    "blocks",
    "token_refiner",
    "final_layer",
    "hidden_size",
    "patch_size",
    "latents_dim",
    "audio_latents_dim",
    "sigma_shift_video",
    "sigma_shift_audio",
    "use_adaln_curves",
    "video_patch_proj",
    "audio_patch_proj",
)


class KeylessCompatibilityError(TypeError):
    """A model advertises the Keyless contract but does not satisfy Spectrum's v1 boundary."""


def _contract_value(contract: Any, name: str) -> Any:
    if not hasattr(contract, name):
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY} is missing required field {name!r}"
        )
    return getattr(contract, name)


def validate_keyless_contract(inner: Any) -> Any | None:
    """Return a validated Keyless v1 contract, or None when the model does not advertise it.

    The adapter is deliberately duck-typed so Spectrum does not import or own the
    MiniMax-H3-Keyless package. A present-but-malformed contract fails closed.
    """
    if inner is None or not hasattr(inner, KEYLESS_CONTRACT_KEY):
        return None
    contract = getattr(inner, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": KEYLESS_CORE_BLOCKS,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": KEYLESS_TOKEN_REFINER_BLOCKS,
        "heads": KEYLESS_HEADS,
        "head_dim": KEYLESS_HEAD_DIM,
        "inner_dim": KEYLESS_INNER_DIM,
        "hidden_size": KEYLESS_HIDDEN_SIZE,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": KEYLESS_NORM_EPSILON,
        "rope_policy": KEYLESS_ROPE_POLICY,
        "qv_order": KEYLESS_QV_ORDER,
        "projection_attr": KEYLESS_PROJECTION_ATTR,
        "checkpoint_format_version": 1,
    }
    mismatches = []
    for name, expected_value in expected.items():
        actual = _contract_value(contract, name)
        if actual != expected_value:
            mismatches.append(f"{name}={actual!r} (expected {expected_value!r})")
    if mismatches:
        raise KeylessCompatibilityError(
            f"unsupported {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches)
        )

    missing_common = [name for name in _COMMON_H3_ATTRIBUTES if not hasattr(inner, name)]
    if missing_common:
        raise KeylessCompatibilityError(
            "Keyless H3 model is missing inherited MiniMax H3 fields: "
            + ", ".join(missing_common)
        )
    if int(getattr(inner, "hidden_size")) != KEYLESS_HIDDEN_SIZE:
        raise KeylessCompatibilityError("Keyless H3 hidden_size disagrees with its semantic contract")
    if not isinstance(getattr(inner, "use_adaln_curves"), bool):
        raise KeylessCompatibilityError("Keyless H3 use_adaln_curves must be boolean")
    timestep_attribute = "adaln_t_table" if inner.use_adaln_curves else "time_embedder"
    if not hasattr(inner, timestep_attribute):
        raise KeylessCompatibilityError(
            f"Keyless H3 model is missing inherited timestep field {timestep_attribute!r}"
        )

    blocks = getattr(inner, "blocks")
    try:
        block_count = len(blocks)
    except TypeError as exc:
        raise KeylessCompatibilityError("Keyless H3 blocks are not sized") from exc
    if block_count != KEYLESS_CORE_BLOCKS:
        raise KeylessCompatibilityError(
            f"Keyless H3 has {block_count} core blocks; expected {KEYLESS_CORE_BLOCKS}"
        )
    refiner_blocks = getattr(getattr(inner, "token_refiner"), "blocks", None)
    if refiner_blocks is None or len(refiner_blocks) != KEYLESS_TOKEN_REFINER_BLOCKS:
        raise KeylessCompatibilityError(
            "Keyless H3 must preserve exactly two native-QKV token-refiner blocks"
        )

    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise KeylessCompatibilityError(f"Keyless block {index} has no attention module")
        if hasattr(attention, "qkv_proj"):
            raise KeylessCompatibilityError(
                f"Keyless block {index} exposes qkv_proj; Spectrum will not accept a fake/dead K path"
            )
        projection = getattr(attention, KEYLESS_PROJECTION_ATTR, None)
        weight = getattr(projection, "weight", None)
        if weight is None or tuple(int(v) for v in getattr(weight, "shape", ())) != (
            KEYLESS_PROJECTION_ROWS,
            KEYLESS_HIDDEN_SIZE,
        ):
            raise KeylessCompatibilityError(
                f"Keyless block {index} does not expose canonical qv_proj.weight geometry"
            )
        if not hasattr(attention, "route_norm") or not hasattr(attention, "q_norm"):
            raise KeylessCompatibilityError(
                f"Keyless block {index} is missing q_norm/route_norm routing semantics"
            )

    identity_fn = getattr(contract, "identity", None)
    if not callable(identity_fn):
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY} must expose callable identity()"
        )
    try:
        identity = tuple(identity_fn())
        hash(identity)
    except (TypeError, ValueError) as exc:
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() must return a hashable tuple-like value"
        ) from exc
    if not identity:
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY}.identity() may not be empty")
    return contract


def is_keyless_minimax_h3(inner: Any) -> bool:
    return validate_keyless_contract(inner) is not None


def keyless_semantic_identity(inner: Any) -> tuple[Any, ...] | None:
    contract = validate_keyless_contract(inner)
    if contract is None:
        return None
    return (KEYLESS_CONTRACT_KEY, *tuple(contract.identity()))


def attention_projection_key(inner: Any, block_index: int) -> str:
    """Return the real attention projection key without synthesizing a QKV alias."""
    contract = validate_keyless_contract(inner)
    suffix = "qv_proj.weight" if contract is not None else "qkv_proj.weight"
    return f"diffusion_model.blocks.{int(block_index)}.attn.{suffix}"


def require_spectrum_minimax_h3(model: Any) -> tuple[Any, str]:
    """Accept native QKV H3 or the exact public Keyless v1 semantic contract."""
    inner, path = locate_minimax_h3_inner(model)
    if inner is not None and hasattr(inner, KEYLESS_CONTRACT_KEY):
        validate_keyless_contract(inner)
    elif is_native_minimax_h3(inner):
        assert path is not None
        return inner, path
    else:
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        raise TypeError(
            "Spectrum Apply MiniMax H3 requires ComfyUI's native MiniMaxH3Model or "
            f"{KEYLESS_CONTRACT_KEY}; discovered {actual}"
        )
    assert path is not None
    return inner, path
