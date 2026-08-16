from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import external_patch_compat as compat
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3 import model_aware
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def contract(**overrides):
    value = {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "kind": "text_activation_modulation",
        "architecture": "minimax_h3",
        "instance_id": "diffaid-h3-1",
        "block_indices_0based": [0, 1, 2, 3, 4],
        "model_block_count": 5,
        "strength": 0.2,
        "sigma_start": 0.0,
        "sigma_end": 1.0,
        "sigma_ramp": 0.0,
        "token_weight_mode": "none",
        "token_tail": 0.35,
        "cond_only": True,
        "scope": "native_mod_segments_tag_1_only",
    }
    value.update(overrides)
    return value


def parsed(*values, block_count=5):
    return compat.parse_external_patch_contracts(
        {compat.EXTERNAL_PATCH_CONTRACTS_KEY: list(values)},
        block_count=block_count,
    )


def runtime_entry(instance_id="diffaid-h3-1", sigma=0.5, provider="comfyui-diffaid-patches"):
    return {
        "schema_version": 1,
        "provider": provider,
        "instance_id": instance_id,
        "normalized_sigma": sigma,
    }


def runtime_options(*entries):
    return {compat.EXTERNAL_PATCH_RUNTIME_KEY: tuple(entries)}


def test_parser_accepts_valid_diffaid_contract_and_exposes_properties():
    result = parsed(contract(sigma_start=0.55, sigma_end=1.0))
    descriptor = result.descriptors[0]

    assert descriptor.provider == "comfyui-diffaid-patches"
    assert descriptor.block_indices_0based == (0, 1, 2, 3, 4)
    assert descriptor.affects_final_block
    assert descriptor.has_hard_temporal_transition
    assert not descriptor.inert
    assert descriptor.active_at(0.55)
    assert descriptor.active_at(1.0)
    assert not descriptor.active_at(math.nextafter(0.55, 0.0))


def test_no_contract_is_empty_and_deterministic():
    first = compat.parse_external_patch_contracts({}, block_count=5)
    second = compat.parse_external_patch_contracts({}, block_count=5)
    assert first.descriptors == ()
    assert first.canonical == ()
    assert first.fingerprint == second.fingerprint == "none"


@pytest.mark.parametrize(
    "bad,match",
    [
        (contract(schema_version=2), "unsupported schema_version"),
        (contract(block_indices_0based=[5]), "outside 0..4"),
        (contract(strength=float("nan")), "must be finite"),
        (contract(strength=float("inf")), "must be finite"),
        (
            contract(sigma_start=0.8, sigma_end=0.2),
            r"0 <= sigma_start <= sigma_end <= 1",
        ),
        (contract(sigma_start=-0.1), r"0 <= sigma_start <= sigma_end <= 1"),
        (contract(sigma_end=1.1), r"0 <= sigma_start <= sigma_end <= 1"),
        (contract(sigma_ramp=-0.1), "sigma_ramp"),
        (contract(model_block_count=4), "declares 4 blocks"),
        (contract(block_indices_0based=[0, 0]), "duplicate block indices"),
    ],
)
def test_parser_rejects_malformed_forecast_compatibility_metadata(bad, match):
    with pytest.raises(compat.ExternalPatchContractError, match=match):
        parsed(bad)


def test_stacked_descriptors_preserve_order_instances_and_fingerprint():
    first = contract(instance_id="diffaid-h3-1", strength=0.2)
    second = contract(instance_id="diffaid-h3-2", strength=0.2)
    result = parsed(first, second)
    reversed_result = parsed(second, first)

    assert [value.instance_id for value in result.descriptors] == [
        "diffaid-h3-1",
        "diffaid-h3-2",
    ]
    assert len(result.descriptors) == 2
    assert result.fingerprint != reversed_result.fingerprint


def test_duplicate_instance_id_is_rejected_instead_of_deduplicated():
    with pytest.raises(compat.ExternalPatchContractError, match="instance_id values must be unique"):
        parsed(contract(), contract())


def test_fingerprint_changes_for_every_behaviorally_relevant_setting():
    base = parsed(contract()).fingerprint
    variants = [
        contract(strength=0.21),
        contract(block_indices_0based=[1, 2, 3, 4]),
        contract(sigma_start=0.1),
        contract(sigma_end=0.9),
        contract(sigma_ramp=0.1),
        contract(token_weight_mode="linear"),
        contract(token_tail=0.4),
        contract(cond_only=False),
        contract(scope="different_scope"),
    ]
    for variant in variants[:-1]:
        assert parsed(variant).fingerprint != base
    with pytest.raises(compat.ExternalPatchContractError):
        parsed(variants[-1])


class _AdaLN(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv_proj = torch.nn.Linear(hidden, hidden * 3, bias=False)
        self.attn.out_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(hidden, hidden * 2, bias=False)
        self.mlp.fc2 = torch.nn.Linear(hidden, hidden, bias=False)
        self.adaln_proj = _AdaLN(hidden)


class _Inner(torch.nn.Module):
    def __init__(self, blocks=5, hidden=4):
        super().__init__()
        self.hidden_size = hidden
        self.patch_size = (1, 2, 2)
        self.latents_dim = 16
        self.audio_latents_dim = 32
        self.use_adaln_curves = False
        self.blocks = torch.nn.ModuleList([_Block(hidden) for _ in range(blocks)])
        self.final_layer = torch.nn.Module()
        self.final_layer.adaln_proj = _AdaLN(hidden)
        self.final_layer.video_out = torch.nn.Linear(hidden, 8, bias=False)
        self.final_layer.audio_out = torch.nn.Linear(hidden, 6, bias=False)


class _Patcher:
    def __init__(self, inner, *, model_options=None):
        self.model = SimpleNamespace(diffusion_model=inner)
        self.model_options = model_options or {}
        self.clone_base_uuid = "base"
        self.patches_uuid = "patches"
        self.patches = {}
        self.backup = {}
        self.injections = {}

    def get_model_object(self, key):
        value = self.model
        for part in key.split("."):
            value = getattr(value, part)
        return value


@pytest.fixture(autouse=True)
def clear_profile_caches():
    model_aware.clear_model_profile_cache()
    compat.clear_external_profile_cache()
    yield
    model_aware.clear_model_profile_cache()
    compat.clear_external_profile_cache()


def _profile_for(descriptors=()):
    inner = _Inner()
    options = (
        {}
        if descriptors is None
        else {compat.EXTERNAL_PATCH_CONTRACTS_KEY: list(descriptors)}
    )
    return model_aware.get_model_forecastability_profile(
        _Patcher(inner, model_options=options)
    )


def test_no_contract_profile_behavior_remains_base_profile():
    lookup = _profile_for(None)
    assert not hasattr(lookup.profile, "recognized_runtime_patch_count")
    assert lookup.profile.active_patch_count == 0
    assert lookup.profile.unknown_patch_count == 0


def test_valid_runtime_patch_is_recognized_without_becoming_unknown_lora_patch():
    lookup = _profile_for([contract()])
    profile = lookup.profile
    assert profile.recognized_runtime_patch_count == 1
    assert profile.recognized_lora_count == 0
    assert profile.unknown_patch_count == 0
    assert profile.active_patch_count == 1
    assert profile.active_patch_keys == 0
    assert profile.runtime_patch_kinds == ("text_activation_modulation",)
    assert profile.runtime_patch_perturbation > 0.0
    assert profile.patch_perturbation >= profile.runtime_patch_perturbation


def test_profile_cache_key_changes_for_strength_blocks_and_sigma_window():
    base = _profile_for(None).profile.cache_key
    full = _profile_for([contract()]).profile.cache_key
    strength = _profile_for([contract(strength=0.25)]).profile.cache_key
    blocks = _profile_for([contract(block_indices_0based=[0, 1, 2, 3])]).profile.cache_key
    window = _profile_for([contract(sigma_start=0.55)]).profile.cache_key
    assert len({base, full, strength, blocks, window}) == 5


def test_identical_external_profile_hits_its_effective_cache():
    inner = _Inner()
    options = {compat.EXTERNAL_PATCH_CONTRACTS_KEY: [contract()]}
    patcher = _Patcher(inner, model_options=options)
    first = model_aware.get_model_forecastability_profile(patcher)
    second = model_aware.get_model_forecastability_profile(patcher)
    assert not first.cache_hit
    assert second.cache_hit
    assert first.profile is second.profile


def test_final_block_and_nonfinal_runtime_perturbations_are_distinguished():
    final_profile = _profile_for([contract(block_indices_0based=[4])]).profile
    nonfinal_profile = _profile_for([contract(block_indices_0based=[0, 1, 2, 3])]).profile
    assert final_profile.runtime_final_block_perturbation > 0.0
    assert nonfinal_profile.runtime_final_block_perturbation == 0.0
    assert final_profile.final_block_perturbation > 0.0
    assert nonfinal_profile.final_block_perturbation == 0.0


def test_negative_strength_uses_magnitude_and_zero_strength_is_inert():
    positive = _profile_for([contract(strength=0.2)]).profile
    negative = _profile_for([contract(strength=-0.2)]).profile
    zero = _profile_for([contract(strength=0.0)]).profile
    assert negative.runtime_patch_perturbation == pytest.approx(
        positive.runtime_patch_perturbation
    )
    assert zero.recognized_runtime_patch_count == 0
    assert zero.runtime_patch_perturbation == 0.0
    assert zero.runtime_final_block_perturbation == 0.0


def test_runtime_risk_values_remain_finite_and_bounded_for_stacked_strong_patches():
    profile = _profile_for(
        [
            contract(instance_id="diffaid-h3-1", strength=3.0),
            contract(instance_id="diffaid-h3-2", strength=-4.0),
        ]
    ).profile
    assert profile.recognized_runtime_patch_count == 2
    assert math.isfinite(profile.aggregate_sensitivity)
    assert math.isfinite(profile.forecast_risk_prior)
    assert 0.0 <= profile.aggregate_sensitivity <= 1.0
    assert 0.0 <= profile.forecast_risk_prior <= 1.0


class _FakeRuntime:
    def __init__(self, parsed_contracts, *, replay=False, model_aware_mode="off"):
        self.config = SimpleNamespace(debug=False, model_aware_mode=model_aware_mode)
        self._step = None
        self.disabled_reason = None
        state = compat._compat_state(self)
        state.parsed = parsed_contracts
        state.contract_failure = None
        state.run = compat._ExternalRunState(
            run_id=1,
            parsed=parsed_contracts,
            replay=replay,
        )

    def set_step(self, step_id, mode):
        self._step = SimpleNamespace(
            step_id=step_id,
            mode=mode,
            reason="scheduled",
            adaptive_recompute=False,
            bootstrap_forecast=False,
            model_aware_decision=None,
            model_aware_forced_actual=False,
        )
        setattr(
            self,
            compat._CURRENT_DECISION_ATTR,
            {"run_id": 1, "step_id": step_id, "actual": mode == "actual", "reason": "scheduled"},
        )

    def _disable_forecasting(self, reason):
        self.disabled_reason = str(reason)


def _commit(fake):
    run = compat._compat_state(fake).run
    assert run is not None
    run.committed_active = run.pending_active
    run.committed_sigma = run.pending_sigma
    run.committed_step_id = run.pending_step_id
    run.pending_step_id = None
    run.pending_active = None
    run.pending_sigma = None
    run.pending_transition_indices = ()


def _observe(fake, step_id, sigma, mode="forecast", instance_ids=("diffaid-h3-1",)):
    fake.set_step(step_id, mode)
    entries = [runtime_entry(instance_id=value, sigma=sigma) for value in instance_ids]
    compat.observe_external_patch_runtime(fake, runtime_options(*entries))
    return fake._step.mode


def test_full_window_has_no_transition_forced_actuals():
    fake = _FakeRuntime(parsed(contract()))
    for step_id, sigma in enumerate((1.0, 0.75, 0.5, 0.25, 0.0)):
        assert _observe(fake, step_id, sigma, "forecast") == "forecast"
        _commit(fake)
    run = compat._compat_state(fake).run
    assert run.transitions == 0
    assert run.forced_actuals == 0


def test_high_sigma_hard_window_promotes_current_leaving_step():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    assert _observe(fake, 0, 0.70, "actual") == "actual"
    _commit(fake)
    assert _observe(fake, 1, 0.54, "forecast") == "actual"
    run = compat._compat_state(fake).run
    assert run.transitions == 1
    assert run.forced_actuals == 1
    assert fake._step.reason == "external patch hard sigma transition"
    assert getattr(fake, compat._CURRENT_DECISION_ATTR)["actual"] is True


def test_low_sigma_hard_window_promotes_current_entering_step():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.0, sigma_end=0.45)))
    _observe(fake, 0, 0.60, "actual")
    _commit(fake)
    assert _observe(fake, 1, 0.45, "forecast") == "actual"


def test_interior_hard_window_protects_entering_and_leaving_boundaries():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.30, sigma_end=0.70)))
    _observe(fake, 0, 0.80, "actual")
    _commit(fake)
    assert _observe(fake, 1, 0.70, "forecast") == "actual"
    _commit(fake)
    assert _observe(fake, 2, 0.50, "forecast") == "forecast"
    _commit(fake)
    assert _observe(fake, 3, math.nextafter(0.30, 0.0), "forecast") == "actual"
    run = compat._compat_state(fake).run
    assert run.transitions == 2
    assert run.forced_actuals == 2


def test_hard_window_equality_is_inclusive_at_start_and_end():
    descriptor = parsed(contract(sigma_start=0.30, sigma_end=0.70)).descriptors[0]
    assert descriptor.active_at(0.30)
    assert descriptor.active_at(0.70)
    assert not descriptor.active_at(math.nextafter(0.30, 0.0))
    assert not descriptor.active_at(math.nextafter(0.70, 1.0))


def test_smooth_ramp_never_uses_hard_boundary_guard():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.30, sigma_end=0.70, sigma_ramp=0.10)))
    _observe(fake, 0, 0.80, "actual")
    _commit(fake)
    assert _observe(fake, 1, 0.70, "forecast") == "forecast"
    _commit(fake)
    assert _observe(fake, 2, 0.29, "forecast") == "forecast"
    run = compat._compat_state(fake).run
    assert run.transitions == 0
    assert run.forced_actuals == 0


def test_already_actual_transition_does_not_duplicate_model_evaluation():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    _observe(fake, 0, 0.70, "actual")
    _commit(fake)
    assert _observe(fake, 1, 0.50, "actual") == "actual"
    run = compat._compat_state(fake).run
    assert run.transitions == 1
    assert run.forced_actuals == 0


def test_two_contracts_crossing_same_step_force_only_one_actual():
    contracts = parsed(
        contract(instance_id="diffaid-h3-1", sigma_start=0.55, sigma_end=1.0),
        contract(instance_id="diffaid-h3-2", sigma_start=0.60, sigma_end=1.0),
    )
    fake = _FakeRuntime(contracts)
    _observe(fake, 0, 0.70, "actual", ("diffaid-h3-1", "diffaid-h3-2"))
    _commit(fake)
    assert _observe(
        fake, 1, 0.50, "forecast", ("diffaid-h3-1", "diffaid-h3-2")
    ) == "actual"
    run = compat._compat_state(fake).run
    assert run.transitions == 2
    assert run.forced_actuals == 1


def test_zero_strength_and_model_aware_off_do_not_disable_boundary_correctness():
    zero = _FakeRuntime(parsed(contract(strength=0.0, sigma_start=0.55)))
    _observe(zero, 0, 0.70, "actual")
    _commit(zero)
    assert _observe(zero, 1, 0.50, "forecast") == "forecast"

    active = _FakeRuntime(
        parsed(contract(strength=0.2, sigma_start=0.55)),
        model_aware_mode="off",
    )
    _observe(active, 0, 0.70, "actual")
    _commit(active)
    assert _observe(active, 1, 0.50, "forecast") == "actual"


def test_repeated_model_call_and_retry_do_not_double_count_transition():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    _observe(fake, 0, 0.70, "actual")
    _commit(fake)
    fake.set_step(1, "forecast")
    options = runtime_options(runtime_entry(sigma=0.50))
    assert compat.observe_external_patch_runtime(fake, options)
    assert compat.observe_external_patch_runtime(fake, options)
    run = compat._compat_state(fake).run
    assert run.transitions == 1
    assert run.forced_actuals == 1


def test_aborted_step_does_not_commit_external_state(monkeypatch):
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    _observe(fake, 0, 0.70, "actual")
    _commit(fake)
    _observe(fake, 1, 0.50, "forecast")
    run = compat._compat_state(fake).run
    before = run.committed_active
    monkeypatch.setattr(compat, "_ORIGINAL_ABORT_STEP", lambda self, run_id, step_id: None)
    compat._runtime_abort_step(fake, 1, 1)
    assert run.committed_active == before
    assert run.pending_step_id is None


def test_finalize_commits_exactly_once(monkeypatch):
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    _observe(fake, 0, 0.70, "actual")
    run = compat._compat_state(fake).run
    monkeypatch.setattr(compat, "_ORIGINAL_FINALIZE_STEP", lambda self, run_id, step_id: None)
    compat._runtime_finalize_step(fake, 1, 0)
    assert run.committed_step_id == 0
    assert run.pending_step_id is None


def test_rollback_restore_rewinds_committed_external_state(monkeypatch):
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    run = compat._compat_state(fake).run
    run.committed_active = (True,)
    run.committed_sigma = (0.70,)
    run.committed_step_id = 0
    snapshot = object()
    monkeypatch.setattr(compat, "_ORIGINAL_CREATE_ROLLBACK_SNAPSHOT", lambda self: snapshot)
    monkeypatch.setattr(compat, "_ORIGINAL_RESTORE_ROLLBACK_SNAPSHOT", lambda self, value: None)
    assert compat._runtime_create_rollback_snapshot(fake) is snapshot
    run.committed_active = (False,)
    run.committed_sigma = (0.50,)
    run.committed_step_id = 1
    compat._runtime_restore_rollback_snapshot(fake, snapshot)
    assert run.committed_active == (True,)
    assert run.committed_sigma == (0.70,)
    assert run.committed_step_id == 0


def test_malformed_runtime_state_fails_safe_to_current_actual_and_future_all_actual():
    fake = _FakeRuntime(parsed(contract(sigma_start=0.55, sigma_end=1.0)))
    fake.set_step(0, "forecast")
    assert compat.observe_external_patch_runtime(fake, {})
    run = compat._compat_state(fake).run
    assert fake._step.mode == "actual"
    assert fake.disabled_reason is not None
    assert run.contract_failures == 1
    assert run.failed_safe

    fake.set_step(1, "forecast")
    assert compat.observe_external_patch_runtime(fake, {}) is False
    assert run.contract_failures == 1


def test_offline_replay_does_not_reintroduce_transition_promotion():
    fake = _FakeRuntime(
        parsed(contract(sigma_start=0.55, sigma_end=1.0)),
        replay=True,
    )
    run = compat._compat_state(fake).run
    run.committed_active = (True,)
    fake.set_step(4, "forecast")
    compat.observe_external_patch_runtime(fake, runtime_options(runtime_entry(sigma=0.50)))
    assert fake._step.mode == "forecast"
    assert run.transitions == 0
    assert run.forced_actuals == 0


def test_predict_noise_wrapper_transaction_parity_tripwire():
    original = compat._ORIGINAL_PREDICT_NOISE_WRAPPER
    assert original is not None
    base_source = inspect.getsource(original)
    compat_source = inspect.getsource(compat._predict_noise_wrapper)
    transaction_tokens = (
        "runtime.begin_step(",
        "copy_model_options_with_step(",
        "runtime.describe_current_er_sde_step(",
        "tracker.consume(",
        "tracker.clear(",
        "ForecastRetryActual",
        "OfflineReplayAbort",
        "ERSDETrackingError",
        "runtime.log_offline_transition(",
        "runtime.prepare_actual_retry(",
        'retry_decision["actual"] = True',
        "runtime.finalize_step(",
        "runtime.abort_step(",
    )
    for token in transaction_tokens:
        assert compat_source.count(token) == base_source.count(token), token
    assert compat_source.index("_log_effective_step(runtime, decision)") > compat_source.index(
        "result = execute_attempt(decision)"
    )


def test_transition_forced_actual_is_visible_to_er_sde_descriptor_immediately():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            offline_smoothing_replay=False,
            model_aware_mode="off",
            bootstrap_first_forecast=False,
            warmup_steps=0,
        )
    )
    parsed_contracts = parsed(contract(sigma_start=0.55, sigma_end=1.0))
    state = compat._compat_state(runtime)
    state.parsed = parsed_contracts
    state.contract_failure = None
    run_id = runtime.start_run(
        torch.tensor([1.0, 0.5, 0.0]),
        "sample_er_sde",
        supported_sampler=True,
    )
    run = compat._compat_state(runtime).run
    run.committed_active = (True,)
    run.committed_sigma = (0.70,)
    run.committed_step_id = 0
    decision = runtime.begin_step(torch.tensor([0.5]))
    runtime._step.mode = "forecast"
    decision["actual"] = False
    compat.observe_external_patch_runtime(runtime, runtime_options(runtime_entry(sigma=0.50)))
    descriptor = runtime.describe_current_er_sde_step(run_id, decision["step_id"])
    assert descriptor.mode == "actual"
    assert descriptor.requires_compensation is False
    assert decision["actual"] is True
    runtime.abort_step(run_id, decision["step_id"])
    runtime.end_run(run_id)
