"""Semantic contract for the optional H3 BSA cold-to-primed history carry.

Ordinary BSA admission uses the live runtime chain on every source revision.
Only the carry exception compares the active calibration implementation with the
validated recipe below. Comments, line numbers and unrelated node changes do not
matter. A changed recipe takes one actual primed call before rebuilding history.
The reference is compiled for the running Python version without executing it.
"""
from __future__ import annotations

from functools import lru_cache
import sys
import types

from . import source_code_audit

# Calibration/dispatch recipe from ComfyUI be92396834f9b6e3e361cfe30e7eed693137a448.
# This is executable-code semantics, never a whole-file source admission list.
_REFERENCE_SOURCE = '''
from __future__ import annotations
import comfy_kitchen as ck
import torch
import comfy.model_management
import comfy.model_prefetch


class SparseAttnPatch:

    def dense_reason(self, transformer_options, tokens, block_index):
        """Why this call stays dense regardless of its tensors, or None."""
        sigmas = transformer_options.get('sigmas')
        if sigmas is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return f'sigma {sigma:.3g} outside the start/end window'
        if tokens < self.min_tokens:
            return f'{tokens} tokens < min_tokens {self.min_tokens}'
        if self.dense_blocks:
            if block_index is None:
                self.log_once('no_block_index', 'this model does not report block indices; dense_blocks ignored')
            elif block_index in self.dense_blocks:
                return f'block {block_index} in dense_blocks'
        return None

    def sinks(self, transformer_options, tokens):
        """MiniMax-H3 conditioning rows as (exact-KV blocks, dense-query blocks):
        the packed prefix stays exact for every query, and optionally the
        target-audio query rows run dense."""
        layout = transformer_options.get('minimax_h3_layout')
        if self.sink_conditioning == 'off' or layout is None or layout.seq_len != tokens:
            return ((0, 0), (0, 0))
        video = next(((a, b) for a, b, kind in layout.segments if kind == 'video'), None)
        if video is None or video[0] <= 0:
            return ((0, 0), (0, 0))
        blocks = (0, (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if self.sink_conditioning != 'exact_kv_and_rows':
            return (blocks, (0, 0))
        audio = next(((a, b) for a, b, kind in layout.segments if kind == 'audio'), None)
        if audio is None:
            return (blocks, blocks)
        return (blocks, (audio[0] // BLOCK_SIZE, blocks[1]))

def h3_eligible(attn, x, rope_freqs, transformer_options, patch: SparseAttnPatch, block_index):
    """Whether this H3 block call takes the sparse producer (decided before any work)."""
    n_tokens = x.shape[0]
    if rope_freqs is None or x.dtype != torch.bfloat16 or x.device.type != "cuda" or attn.head_dim != HEAD_DIM:
        return False
    reason = patch.dense_reason(transformer_options, n_tokens, block_index)
    if reason is None and not ck.sol_attn_is_available(x.device):
        reason = "no compiled sol_attn kernel for this GPU"
    if reason is not None:
        patch.log_once(("dense", n_tokens, reason), f"dense ({n_tokens} tokens): {reason}")
        return False
    if patch.vsa:
        layout = transformer_options.get("minimax_h3_layout")
        if layout is None or layout.seq_len != n_tokens:
            patch.log_once("no_layout", "no H3 layout for this call; running dense")
            return False
    return True

def h3_sparse_attention(attn, x, rope_freqs, transformer_options, patch: SparseAttnPatch, block_index):
    """H3 attention through the chunked producer: qkv projected in 4K-token
    slices straight into the kernel's int8 carriers, full Q/K/V never built."""
    n_tokens = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
    extra, plan, gate = {}, None, None
    n, freqs = n_tokens, rope_freqs
    with comfy.model_prefetch.pause_malloc_graph():
        if patch.vsa:
            plan = patch.vsa_plan(transformer_options["minimax_h3_layout"], x.device)
            n = plan["n"]
            freqs = patch.vsa_rope_freqs(rope_freqs, plan)

        key = (block_index, n, tuple(transformer_options.get("uuids", ())))   # statistics per conditioning branch
        pooled = patch.pooled.get(key)
        first = pooled is None
        if first:
            pooled = (
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
                torch.empty((heads, head_dim), dtype=torch.float32, device=x.device),
            )

    if patch.vsa:
        sink = sink_q = (0, plan["n_prefix"])
        extra = {"tail": False, "block_len": plan["block_len"]}
        gate = attn.to_gate_compress
        if gate is not None:
            extra["coarse_gate"] = x.new_empty(n, heads * head_dim).view(1, n, heads, head_dim)
    else:
        sink, sink_q = patch.sinks(transformer_options, n_tokens)

    def chunks():
        for i in range(0, n, PRODUCER_CHUNK):
            if plan is None:
                yield attn.qkv_proj(x[i:i + PRODUCER_CHUNK])
                continue
            idx = plan["src"][i:i + PRODUCER_CHUNK]
            xc = x[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1).to(x.dtype)   # pad rows zero
            if gate is not None:
                extra["coarse_gate"].view(n, heads * head_dim)[i:i + xc.shape[0]] = gate(xc)
            yield attn.qkv_proj(xc)

    out, kmean, vscale = ck.sol_attn_chunked(
        chunks, n, heads, freqs, (qw, kw),
        kmean=None if first else pooled[0],
        vscale=None if first else pooled[1],
        tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=patch.extra_tokens,
        sink_blocks=list(sink), sink_q=list(sink_q),
        rope_eps=attn.q_norm.eps, **extra)
    pooled[0].copy_(kmean)
    pooled[1].copy_(vscale)
    patch.pooled[key] = pooled
    mode = f"VSA tiles ({n} padded rows, {sink[1]} prefix tiles)" if plan is not None else f"sinks {sink}/{sink_q}"
    patch.log_once(("producer", n), f"sparse producer path: {n_tokens} tokens, {mode}")
    out = out.view(n, heads * head_dim)
    if plan is not None:
        out = out[plan["inv"]]
    return attn.out_proj(out)

def make_h3_block_patch(block, block_index, patch: SparseAttnPatch):
    """Runs the block with its attention swapped for the sparse producer."""
    def attention(h, rope_freqs=None, transformer_options={}):
        return h3_sparse_attention(block.attn, h, rope_freqs, transformer_options, patch, block_index)

    def block_patch(args, extra):
        if h3_eligible(block.attn, args["img"], args["rope_freqs"], args["transformer_options"], patch, block_index):
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch
'''

@lru_cache(maxsize=1)
def _reference_codes():
    root = compile(_REFERENCE_SOURCE, "<h3-bsa-calibration-contract>", "exec",
                   dont_inherit=True, optimize=sys.flags.optimize)
    paths = (
        ("h3_sparse_attention",),
        ("h3_eligible",),
        ("make_h3_block_patch", "attention"),
        ("make_h3_block_patch", "block_patch"),
        ("SparseAttnPatch", "sinks"),
        ("SparseAttnPatch", "dense_reason"),
    )
    result = []
    for path in paths:
        current = root
        for name in path:
            current = source_code_audit._direct_nested_code(current, name)
            if current is None:
                raise RuntimeError("BSA calibration reference is incomplete")
        result.append(source_code_audit._code_semantics(current))
    return tuple(result)


@lru_cache(maxsize=64)
def _live_semantics(code):
    return source_code_audit._code_semantics(code)


def supports_cold_successor(replacement, attention, patch):
    """Qualify only the carry exception against live code and helper bindings."""
    globals_ = getattr(attention, "__globals__", {})
    eligible = getattr(replacement, "__globals__", {}).get("h3_eligible")
    functions = (
        globals_.get("h3_sparse_attention"), eligible, attention, replacement,
        getattr(patch, "sinks", None), getattr(patch, "dense_reason", None),
    )
    defaults = (None, None, (None, {}), None, None, None)
    for function, reference, expected_defaults in zip(functions, _reference_codes(), defaults):
        base = getattr(function, "__func__", function)
        code = getattr(base, "__code__", None)
        if not isinstance(code, types.CodeType):
            return False
        if _live_semantics(code) != reference:
            return False
        if getattr(base, "__defaults__", None) != expected_defaults or getattr(base, "__kwdefaults__", None):
            return False
    return True
