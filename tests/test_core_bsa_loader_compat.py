from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat, core_bsa_flow_compat


class _FakeAttention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 128
        self.qkv_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16, device="cpu")
        )

    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttention()


class _FakeModel:
    def __init__(self, count=2):
        self.blocks = [_FakeBlock() for _ in range(count)]
        self.dtype = torch.bfloat16


def _layout(seq_len=128):
    return SimpleNamespace(
        seq_len=seq_len,
        signature=(64, 8, 4, 4, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, seq_len, "video"),
        ],
    )


def _runtime_loaded_bsa(monkeypatch):
    try:
        import comfy_extras.nodes_sparse_attention as canonical
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core BSA fixture unavailable: {exc}")
    source = Path(canonical.__file__).resolve()
    if core_bsa_compat._module_blob_sha(canonical) not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        pytest.skip("ComfyUI fixture is not the reviewed core BSA source")

    # Mirror ComfyUI nodes.load_custom_node for builtin extra-node files: the
    # sys.modules key is the source path stem, not comfy_extras.<module>.
    alias_name = str(source.with_suffix(""))
    spec = importlib.util.spec_from_file_location(alias_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias_name, module)
    spec.loader.exec_module(module)
    assert module.__name__ == alias_name
    return module


def _installation(module, *, count=2):
    model = _FakeModel(count)
    patch = module.SparseAttnPatch(
        tau=1.3,
        topk_ratio=0.0,
        vsa=False,
        sigma_start=0.8,
        sigma_end=0.0,
        min_tokens=1,
        dense_blocks=set(),
        sink_conditioning="exact_kv",
        extra_tokens=64,
        verbose=False,
    )
    override = module.make_attention_override(patch, None)
    patch.installed.add(override)
    replacements = {
        ("double_block", index): module.make_h3_block_patch(block, index, patch)
        for index, block in enumerate(model.blocks)
    }
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {"dit": replacements},
        # Outside BSA's sparse sigma window so CPU fixture availability is not
        # part of this loader-identity regression.
        "sigmas": torch.tensor([1.0]),
        "uuids": ("positive",),
    }
    return model, patch, override, options


def test_runtime_path_loaded_core_bsa_is_recognized(monkeypatch):
    module = _runtime_loaded_bsa(monkeypatch)
    model, patch, override, options = _installation(module)

    replacement = options["patches_replace"]["dit"][("double_block", 0)]
    assert replacement.__module__ == module.__name__
    assert module.__name__ != "comfy_extras.nodes_sparse_attention"
    assert core_bsa_compat._looks_like_core_bsa_callable(
        replacement,
        "make_h3_block_patch",
        "block_patch",
    )
    assert core_bsa_compat._looks_like_core_bsa_callable(
        override,
        "make_attention_override",
        "override",
    )

    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.safe
    assert audit.patch is patch
    assert all(spec[0] == "h3_dense" for spec in audit.route_specs)


def test_runtime_path_loaded_bsa_under_reviewed_flow_wrapper(monkeypatch):
    module = _runtime_loaded_bsa(monkeypatch)
    model, patch, _override, options = _installation(module, count=1)
    try:
        from h3_flow_regenerate import attention
        from h3_flow_regenerate.metrics import H3FlowMetrics
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"reviewed Flow fixture unavailable: {exc}")
    if (
        core_bsa_compat._module_blob_sha(attention)
        not in core_bsa_flow_compat.AUDITED_FLOW_ATTENTION_GIT_BLOBS
    ):
        pytest.skip("Flow attention fixture is not the reviewed source")

    key = ("double_block", 0)
    previous = options["patches_replace"]["dit"][key]
    metrics = H3FlowMetrics()
    wrapper = attention.make_layout_block_wrapper(0, metrics, previous)
    wrapper = attention.mark_layout_wrapper(
        wrapper,
        metrics=metrics,
        previous=previous,
        scope="layout",
    )
    options["patches_replace"]["dit"][key] = wrapper

    audit, reason = core_bsa_flow_compat.probe(options, _layout(), model)
    assert reason is None and audit is not None and audit.safe
    assert audit.patch is patch
    assert audit.flow_outer_replacements[0] is wrapper
    assert audit.flow_bsa_replacements[0] is previous
