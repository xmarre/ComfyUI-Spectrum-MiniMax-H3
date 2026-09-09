from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import core_bsa_compat
from comfyui_spectrum_h3.backend_history import POLICIES, RECEIPTS, preflight


class _FakeAttention:
    def __init__(self, heads=2):
        self.heads = heads
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


def _audited_nodes():
    try:
        import comfy_extras.nodes_sparse_attention as nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"core BSA is unavailable in this reviewed ComfyUI fixture: {exc}")
    if core_bsa_compat._module_blob_sha(nodes) not in core_bsa_compat.AUDITED_BSA_GIT_BLOBS:
        pytest.skip("this ComfyUI fixture is not the reviewed core BSA source")
    return nodes


def _layout(seq_len=128):
    return SimpleNamespace(
        seq_len=seq_len,
        signature=(64, 1, 8, 8, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, seq_len, "video"),
        ],
    )


def _installation(*, count=2, previous=None, sigma=1.0, uuids=("positive",)):
    nodes = _audited_nodes()
    model = _FakeModel(count)
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
    override = nodes.make_attention_override(patch, previous)
    patch.installed.add(override)
    dit = {
        ("double_block", index): nodes.make_h3_block_patch(block, index, patch)
        for index, block in enumerate(model.blocks)
    }
    options = {
        "callbacks": {"on_prepare_state": {"block_sparse_attention": []}},
        "optimized_attention_override": override,
        "patches_replace": {"dit": dit},
        "sigmas": torch.tensor([sigma]),
        "uuids": uuids,
    }
    return nodes, model, patch, options


def _actual_args(options, layout, seq_len):
    call_options = dict(options)
    call_options["minimax_h3_layout"] = layout
    return {
        "img": torch.zeros(seq_len, 4, dtype=torch.bfloat16),
        "rope_freqs": torch.zeros(seq_len, 1),
        "transformer_options": call_options,
    }


def test_reviewed_core_bsa_dense_route_is_structurally_recognized():
    _nodes, model, _patch, options = _installation(sigma=1.0)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None
    assert audit is not None
    assert audit.safe
    assert {spec[0] for spec in audit.route_specs} == {"h3_dense"}


def test_structural_bsa_evidence_survives_missing_callback():
    _nodes, model, _patch, options = _installation(sigma=1.0)
    options["callbacks"] = {}
    assert core_bsa_compat.has_core_bsa_evidence(options)
    identity, safe = preflight(options, _layout(), model)
    assert identity[0] == core_bsa_compat.ADAPTER_KEY
    assert safe


def test_structural_bsa_evidence_fails_closed_after_override_replacement():
    _nodes, model, _patch, options = _installation(sigma=1.0)
    options["callbacks"] = {}
    options["optimized_attention_override"] = lambda *args, **kwargs: None
    assert core_bsa_compat.has_core_bsa_evidence(options)
    identity, safe = preflight(options, _layout(), model)
    assert identity == ("core_bsa_unreported", "ownership_unproven")
    assert not safe


def test_explicit_provider_contract_keeps_precedence(monkeypatch):
    class Provider:
        def __call__(self, **_kwargs):
            return ("explicit",)

    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("core BSA adapter must not run with an explicit provider")

    monkeypatch.setattr(core_bsa_compat, "probe", forbidden_probe)
    identity, safe = preflight({POLICIES: {"explicit": Provider()}}, None, None)
    assert identity == (("explicit", ("explicit",)),)
    assert safe


def test_unreviewed_core_bsa_source_fails_closed(monkeypatch):
    _nodes, model, _patch, options = _installation()
    monkeypatch.setattr(core_bsa_compat, "_module_blob_sha", lambda _module: "unreviewed")
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "source_unreviewed"
    identity, safe = preflight(options, _layout(), model)
    assert identity == ("core_bsa_unreported", "source_unreviewed")
    assert not safe


def test_missing_or_foreign_h3_replacement_fails_closed():
    _nodes, model, _patch, options = _installation()
    options["patches_replace"]["dit"][("double_block", 1)] = lambda args, extra: extra[
        "original_block"
    ](args)
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "ownership_unproven"


def test_stacked_bsa_ownership_fails_closed():
    nodes, model, first_patch, first_options = _installation()
    first_override = first_options["optimized_attention_override"]
    second_patch = nodes.SparseAttnPatch(
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
    second_override = nodes.make_attention_override(second_patch, first_override)
    second_patch.installed.add(second_override)
    options = dict(first_options)
    options["optimized_attention_override"] = second_override
    options["patches_replace"] = {
        "dit": {
            ("double_block", index): nodes.make_h3_block_patch(
                block, index, second_patch
            )
            for index, block in enumerate(model.blocks)
        }
    }
    assert first_patch is not second_patch
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "ownership_unproven"


def test_inherited_attention_owner_change_changes_policy_identity():
    nodes, model, patch, options = _installation(previous=lambda *args, **kwargs: None)
    first, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and first is not None

    def replacement_previous(*args, **kwargs):
        return None

    override = nodes.make_attention_override(patch, replacement_previous)
    patch.installed.add(override)
    options["optimized_attention_override"] = override
    second, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and second is not None
    assert first.identity != second.identity


def test_lifetime_generation_and_callable_identity_are_stable_and_unique():
    class Owner:
        pass

    first_owner = Owner()
    second_owner = Owner()
    first_generation = core_bsa_compat._lifetime_generation(first_owner)
    assert first_generation == core_bsa_compat._lifetime_generation(first_owner)
    assert first_generation != core_bsa_compat._lifetime_generation(second_owner)

    def first_callable():
        return None

    def second_callable():
        return None

    first_identity = core_bsa_compat._callable_identity(first_callable)
    assert first_identity == core_bsa_compat._callable_identity(first_callable)
    assert first_identity != core_bsa_compat._callable_identity(second_callable)


def test_execution_identity_ignores_transient_weight_residency():
    model = _FakeModel(1)
    first = core_bsa_compat._runtime_execution_identity(model)
    model.blocks[0].attn.qkv_proj.weight = torch.empty(
        1, dtype=torch.bfloat16, device="meta"
    )
    second = core_bsa_compat._runtime_execution_identity(model)
    assert first == second


def test_dense_sparse_and_cold_primed_transitions_change_identity(monkeypatch):
    _nodes, model, patch, options = _installation(sigma=1.0)
    dense, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and dense is not None

    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    options["sigmas"] = torch.tensor([0.5])
    cold, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and cold is not None and cold.safe
    assert {spec[0] for spec in cold.route_specs} == {"h3_chunked_sparse_cold"}

    for index, block in enumerate(model.blocks):
        shape = (block.attn.heads, block.attn.head_dim)
        patch.pooled[(index, 128, ("positive",))] = (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )
    primed, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and primed is not None and primed.safe
    assert {spec[0] for spec in primed.route_specs} == {
        "h3_chunked_sparse_primed"
    }
    assert primed.expected_receipts[0][2] == primed.patch_generation
    assert dense.identity != cold.identity
    assert cold.identity != primed.identity


def test_pool_tensor_replacement_changes_policy_identity(monkeypatch):
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    _nodes, model, patch, options = _installation(sigma=0.5, count=1)
    shape = (model.blocks[0].attn.heads, model.blocks[0].attn.head_dim)
    key = (0, 128, ("positive",))
    patch.pooled[key] = (
        torch.zeros(shape, dtype=torch.float32),
        torch.zeros(shape, dtype=torch.float32),
    )
    first, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and first is not None and first.safe

    patch.pooled[key] = (
        torch.zeros(shape, dtype=torch.float32),
        torch.zeros(shape, dtype=torch.float32),
    )
    second, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and second is not None and second.safe
    assert first.identity != second.identity


def test_layout_and_uuid_changes_change_policy_identity(monkeypatch):
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    _nodes, model, _patch, options = _installation(sigma=0.5)
    first, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and first is not None

    options["uuids"] = ("other",)
    uuid_changed, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None and uuid_changed is not None
    assert first.identity != uuid_changed.identity

    layout_changed = SimpleNamespace(
        seq_len=192,
        signature=(64, 1, 8, 16, 16),
        segments=[
            (0, 64, "text"),
            (64, 96, "audio"),
            (96, 192, "video"),
        ],
    )
    layout_identity, reason = core_bsa_compat.probe(options, layout_changed, model)
    assert reason is None and layout_identity is not None
    assert uuid_changed.identity != layout_identity.identity


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sigma_is_unproven(value):
    assert core_bsa_compat._sigma_value({"sigmas": [value]}) is None


def test_pool_transition_classifier_rejects_ambiguous_state():
    patch = SimpleNamespace(pooled={})
    key = (0, 128, ("positive",))
    assert core_bsa_compat._classify_pool_transition(
        core_bsa_compat._pool_entry(patch, key),
        core_bsa_compat._pool_entry(patch, key),
        sparse_selected=False,
    ) == "h3_dense"

    first = torch.zeros((2, 128), dtype=torch.float32)
    second = torch.zeros((2, 128), dtype=torch.float32)
    before = core_bsa_compat._pool_entry(patch, key)
    patch.pooled[key] = (first, second)
    after = core_bsa_compat._pool_entry(patch, key)
    assert core_bsa_compat._classify_pool_transition(
        before, after, sparse_selected=True
    ) == "h3_chunked_sparse_cold"

    before = core_bsa_compat._pool_entry(patch, key)
    first.copy_(torch.ones_like(first))
    second.copy_(torch.ones_like(second))
    after = core_bsa_compat._pool_entry(patch, key)
    assert core_bsa_compat._classify_pool_transition(
        before, after, sparse_selected=True
    ) == "h3_chunked_sparse_primed"
    assert core_bsa_compat._classify_pool_transition(
        before, after, sparse_selected=False
    ) == "h3_dense"

    before = after
    patch.pooled[key] = (
        torch.zeros_like(first),
        torch.zeros_like(second),
    )
    replacement = core_bsa_compat._pool_entry(patch, key)
    assert core_bsa_compat._classify_pool_transition(
        before, replacement, sparse_selected=True
    ) == "unknown"


def test_pool_entry_accepts_inference_mode_tensors():
    key = (0, 128, ("positive",))
    with torch.inference_mode():
        first = torch.zeros((2, 128), dtype=torch.float32)
        second = torch.zeros((2, 128), dtype=torch.float32)
    patch = SimpleNamespace(pooled={key: (first, second)})

    with pytest.raises(RuntimeError, match="version counter"):
        _ = first._version

    state = core_bsa_compat._pool_entry(patch, key, (2, 128, None))
    assert state[0] == "present"
    assert state[1] is first
    assert state[2] is second


def test_actual_wrapper_observes_sparse_selection_without_version_counter(monkeypatch):
    key = (0, 128, ("positive",))
    with torch.inference_mode():
        first = torch.zeros((2, 128), dtype=torch.float32)
        second = torch.zeros((2, 128), dtype=torch.float32)
    patch = SimpleNamespace(pooled={key: (first, second)})

    def attention(*_args, **_kwargs):
        return None

    def replacement(args, extra):
        return extra["original_block"]({**args, "attention": attention})

    audit = SimpleNamespace(
        seq_len=128,
        uuids=("positive",),
        patch=patch,
        pool_specs=((2, 128, None),),
        route_specs=(("h3_chunked_sparse_primed", (0, 0), (0, 0)),),
        patch_generation=11,
        failure=None,
    )
    monkeypatch.setattr(core_bsa_compat, "_current_route_matches", lambda *_args: True)
    receipts = []
    wrapped = core_bsa_compat._make_actual_wrapper(audit, 0, replacement, receipts)
    output = wrapped(
        {"img": torch.zeros(128, 4)},
        {"original_block": lambda call_args: {"img": call_args["img"]}},
    )

    assert "img" in output
    assert audit.failure is None
    assert receipts == [
        (
            core_bsa_compat.ADAPTER_KEY,
            core_bsa_compat.ADAPTER_VERSION,
            11,
            0,
            "h3_chunked_sparse_primed",
            128,
            (0, 0),
            (0, 0),
        )
    ]


def test_main_block_audit_does_not_count_unrelated_attention():
    _nodes, model, _patch, options = _installation(sigma=1.0, count=1)
    layout = _layout()
    audit, reason = core_bsa_compat.probe(options, layout, model)
    assert reason is None and audit is not None

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    receipts = prepared[RECEIPTS]

    override = prepared["optimized_attention_override"]
    q = torch.zeros(1, 4, 256, dtype=torch.bfloat16)
    override(
        lambda q, _k, _v, _heads, **_kwargs: q,
        q,
        q,
        q,
        2,
        mask=torch.ones(1),
        transformer_options={},
    )
    assert receipts == []

    args = _actual_args(prepared, layout, 128)
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    output = wrapped(args, {"original_block": lambda call_args: {"img": call_args["img"]}})
    assert "img" in output
    assert len(receipts) == 1
    assert core_bsa_compat.accepts_actual(audit, tuple(receipts))


def test_actual_route_mismatch_fails_receipt_acceptance(monkeypatch):
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    _nodes, model, _patch, options = _installation(sigma=0.5, count=1)
    layout = _layout()
    audit, reason = core_bsa_compat.probe(options, layout, model)
    assert reason is None and audit is not None and audit.safe
    assert audit.route_specs[0][0] == "h3_chunked_sparse_cold"

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    args = _actual_args(prepared, layout, 128)
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]

    # The fake H3 activation is on CPU, so the real reviewed BSA replacement
    # declines its sparse producer. The Spectrum audit must observe dense rather
    # than trusting the preflight prediction.
    wrapped(args, {"original_block": lambda call_args: {"img": call_args["img"]}})
    receipts = tuple(prepared[RECEIPTS])
    assert receipts[0][4] == "h3_dense"
    assert audit.failure == "actual_route_mismatch"
    assert not core_bsa_compat.accepts_actual(audit, receipts)


def test_adapter_propagates_cuda_oom_from_sigma_introspection():
    class OOMSigmas:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            raise torch.cuda.OutOfMemoryError("synthetic metadata OOM")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        core_bsa_compat._sigma_value({"sigmas": OOMSigmas()})


def test_invalid_pooled_shape_is_not_forecast_safe(monkeypatch):
    monkeypatch.setattr(
        core_bsa_compat, "_sparse_runtime_eligible", lambda _model, _module: True
    )
    _nodes, model, patch, options = _installation(sigma=0.5)
    patch.pooled[(0, 128, ("positive",))] = (
        torch.zeros((1, 128), dtype=torch.float32),
        torch.zeros((1, 128), dtype=torch.float32),
    )
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert reason is None
    assert audit is not None
    assert not audit.safe
    assert audit.route_specs[0][0] == "h3_sparse_unproven"


def test_vsa_remains_fail_closed():
    _nodes, model, patch, options = _installation()
    patch.vsa = True
    audit, reason = core_bsa_compat.probe(options, _layout(), model)
    assert audit is None
    assert reason == "vsa_unsupported"


def test_actual_attention_owner_change_invalidates_receipt():
    _nodes, model, _patch, options = _installation(sigma=1.0, count=1)
    layout = _layout()
    audit, reason = core_bsa_compat.probe(options, layout, model)
    assert reason is None and audit is not None

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    args = _actual_args(prepared, layout, 128)
    args["transformer_options"]["optimized_attention_override"] = lambda *args, **kwargs: None
    wrapped = prepared["patches_replace"]["dit"][("double_block", 0)]
    wrapped(args, {"original_block": lambda call_args: {"img": call_args["img"]}})

    assert audit.failure == "actual_route_mismatch"
    assert not core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))


def test_midforward_attention_owner_change_is_revalidated_per_block():
    _nodes, model, _patch, options = _installation(sigma=1.0, count=2)
    layout = _layout()
    audit, reason = core_bsa_compat.probe(options, layout, model)
    assert reason is None and audit is not None

    prepared = {**options, RECEIPTS: []}
    prepared = core_bsa_compat.instrument_actual_options(prepared, audit, RECEIPTS)
    args = _actual_args(prepared, layout, 128)
    context = {"original_block": lambda call_args: {"img": call_args["img"]}}

    first = prepared["patches_replace"]["dit"][("double_block", 0)]
    first(args, context)
    assert audit.failure is None

    args["transformer_options"]["optimized_attention_override"] = lambda *args, **kwargs: None
    second = prepared["patches_replace"]["dit"][("double_block", 1)]
    second(args, context)

    assert audit.failure == "actual_route_mismatch"
    assert not core_bsa_compat.accepts_actual(audit, tuple(prepared[RECEIPTS]))
