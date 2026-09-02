from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import external_patch_compat as compat
from comfyui_spectrum_h3 import external_patch_visual_reference as visual


def diffaid_contract(**overrides):
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


def untwist_profile(**overrides):
    value = {
        "schema_version": 1,
        "provider": visual.VISUAL_PATCH_PROVIDER,
        "kind": visual.VISUAL_PATCH_KIND,
        "architecture": visual.VISUAL_PATCH_ARCHITECTURE,
        "instance_id": "untwist-h3-1",
        "block_indices_0based": [0, 1, 2, 3, 4],
        "model_block_count": 5,
        "strength": 0.25,
        "progress_start": 0.0,
        "progress_end": 0.90,
        "hard_start": False,
        "hard_end": True,
        "scope": "image_and_video",
        "high_scale_start": 0.75,
        "high_scale_end": 0.90,
        "low_scale_start": 1.0,
        "low_scale_end": 1.05,
        "beta": 2.0,
        "scale_temporal_axis": False,
    }
    value.update(overrides)
    return value


def parse(*, include_diffaid=True, profile=None):
    options = {visual.VISUAL_PATCH_PROFILES_KEY: [profile or untwist_profile()]}
    if include_diffaid:
        options[compat.EXTERNAL_PATCH_CONTRACTS_KEY] = [diffaid_contract()]
    return compat.parse_external_patch_contracts(options, block_count=5)


def visual_runtime(progress, *, active=True, schema_version=1):
    return {
        visual.VISUAL_PATCH_RUNTIME_KEY: (
            {
                "schema_version": schema_version,
                "provider": visual.VISUAL_PATCH_PROVIDER,
                "instance_id": "untwist-h3-1",
                "schedule_progress": progress,
                "active": active,
            },
        )
    }


def test_stacked_diffaid_and_untwist_profiles_preserve_distinct_kinds():
    parsed = parse()
    assert len(parsed.descriptors) == 2
    assert [value.kind for value in parsed.descriptors] == [
        "text_activation_modulation",
        "visual_reference_attention_modulation",
    ]
    visual_descriptor = parsed.descriptors[1]
    assert visual_descriptor.scope == "image_and_video"
    assert visual_descriptor.sigma_start == pytest.approx(0.10)
    assert visual_descriptor.sigma_end == 1.0
    assert visual_descriptor.has_hard_temporal_transition is True


def test_soft_end_widens_visual_guard_to_sigma_zero():
    parsed = parse(
        include_diffaid=False,
        profile=untwist_profile(hard_end=False),
    )
    descriptor = parsed.descriptors[0]
    assert descriptor.sigma_start == 0.0
    assert descriptor.sigma_end == 1.0
    assert descriptor.has_hard_temporal_transition is False


def test_v2_weak_late_spatial_profile_declares_terminal_pece_capability():
    profile = untwist_profile(
        schema_version=2,
        strength=0.05,
        progress_end=0.95,
        high_scale_start=0.95,
        high_scale_end=1.0,
        low_scale_start=1.0,
        low_scale_end=1.05,
        terminal_pece_exact_corrector_safe=True,
    )
    parsed = parse(include_diffaid=False, profile=profile)
    descriptor = parsed.descriptors[0]
    assert descriptor.schema_version == 2
    assert descriptor.terminal_pece_exact_corrector_safe is True
    assert compat._runtime_entries(
        visual_runtime(1.0, active=False, schema_version=2),
        parsed,
    ) == pytest.approx((0.0,))


@pytest.mark.parametrize(
    "override",
    [
        {"progress_start": 0.1, "hard_start": True},
        {"progress_end": 0.89},
        {"scope": "all_visual_including_continuum"},
        {"scale_temporal_axis": True},
        {"strength": 0.10, "high_scale_start": 0.90},
    ],
)
def test_v2_terminal_pece_capability_rejects_unreviewed_profiles(override):
    values = {
        "schema_version": 2,
        "strength": 0.05,
        "high_scale_start": 0.95,
        "high_scale_end": 1.0,
        "low_scale_start": 1.0,
        "low_scale_end": 1.05,
        "terminal_pece_exact_corrector_safe": True,
    }
    values.update(override)
    profile = untwist_profile(**values)
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="reviewed weak spatial-only late hard-end envelope",
    ):
        parse(include_diffaid=False, profile=profile)


def test_v1_profile_cannot_smuggle_terminal_pece_capability():
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="requires schema_version=2",
    ):
        parse(
            include_diffaid=False,
            profile=untwist_profile(
                terminal_pece_exact_corrector_safe=True,
            ),
        )


def test_visual_runtime_schema_must_match_static_profile():
    parsed = parse(
        include_diffaid=False,
        profile=untwist_profile(
            schema_version=2,
            strength=0.05,
            high_scale_start=0.95,
            high_scale_end=1.0,
            low_scale_start=1.0,
            low_scale_end=1.05,
            terminal_pece_exact_corrector_safe=True,
        ),
    )
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="runtime/profile schema changed",
    ):
        compat._runtime_entries(visual_runtime(1.0, schema_version=1), parsed)


def test_hard_start_maps_inclusive_progress_start_to_sigma_end():
    parsed = parse(
        include_diffaid=False,
        profile=untwist_profile(progress_start=0.20, hard_start=True),
    )
    descriptor = parsed.descriptors[0]
    assert descriptor.sigma_start == pytest.approx(0.10)
    assert descriptor.sigma_end == pytest.approx(0.80)
    assert descriptor.has_hard_temporal_transition is True


def test_visual_profile_rejects_strength_inconsistent_with_scale_endpoints():
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="strength does not match",
    ):
        parse(
            include_diffaid=False,
            profile=untwist_profile(strength=0.24),
        )


def test_visual_profile_rejects_duplicate_instance_id_across_patch_kinds():
    options = {
        compat.EXTERNAL_PATCH_CONTRACTS_KEY: [
            diffaid_contract(instance_id="untwist-h3-1")
        ],
        visual.VISUAL_PATCH_PROFILES_KEY: [untwist_profile()],
    }
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="unique across all profile kinds",
    ):
        compat.parse_external_patch_contracts(options, block_count=5)


def test_visual_profile_rejects_unsupported_scope():
    with pytest.raises(
        compat.ExternalPatchContractError,
        match="unsupported scope",
    ):
        parse(
            include_diffaid=False,
            profile=untwist_profile(scope="unsupported"),
        )


def test_visual_strength_metadata_changes_external_profile_fingerprint():
    first = parse(include_diffaid=False)
    second = parse(
        include_diffaid=False,
        profile=untwist_profile(
            strength=0.20,
            high_scale_start=0.80,
        ),
    )
    assert first.fingerprint != second.fingerprint


def test_external_model_profile_counts_diffaid_and_untwist_separately():
    parsed = parse()
    base = SimpleNamespace(
        active_patch_count=0,
        active_patch_keys=0,
        profile_confidence=1.0,
        aggregate_sensitivity=0.0,
        patch_perturbation=0.0,
        final_block_perturbation=0.0,
        forecast_risk_prior=0.0,
        build_seconds=0.0,
        estimated_bytes=0,
        transient_workspace_bytes=0,
        patch_identity="base",
    )
    lookup = SimpleNamespace(profile=base, cache_hit=True)
    profile = compat._external_profile_from_base(
        lookup,
        parsed,
        cache_key=("test", parsed.fingerprint),
        adjustment_started=time.perf_counter(),
    )
    assert profile.recognized_runtime_patch_count == 2
    assert profile.active_patch_count == 2
    assert profile.runtime_patch_kinds == (
        "text_activation_modulation",
        "visual_reference_attention_modulation",
    )


class FakeRuntime:
    def __init__(self, parsed_contracts):
        self.config = SimpleNamespace(debug=False, model_aware_mode="off")
        self._step = None
        self.disabled_reason = None
        state = compat._compat_state(self)
        state.parsed = parsed_contracts
        state.contract_failure = None
        state.run = compat._ExternalRunState(
            run_id=1,
            parsed=parsed_contracts,
            replay=False,
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
            {
                "run_id": 1,
                "step_id": step_id,
                "actual": mode == "actual",
                "reason": "scheduled",
            },
        )

    def _disable_forecasting(self, reason):
        self.disabled_reason = str(reason)


def commit_pending(fake):
    run = compat._compat_state(fake).run
    run.committed_active = run.pending_active
    run.committed_sigma = run.pending_sigma
    run.committed_step_id = run.pending_step_id
    run.pending_step_id = None
    run.pending_active = None
    run.pending_sigma = None
    run.pending_transition_indices = ()


def test_visual_runtime_active_flag_does_not_override_sigma_derived_schedule_state():
    parsed = parse(include_diffaid=False)
    progress = 17 / 18
    active_entry = compat._runtime_entries(
        visual_runtime(progress, active=True),
        parsed,
    )
    inactive_entry = compat._runtime_entries(
        visual_runtime(progress, active=False),
        parsed,
    )
    assert active_entry == pytest.approx((1.0 - progress,))
    assert inactive_entry == pytest.approx(active_entry)


def test_untwist_end_percent_transition_forces_actual_after_point_nine():
    parsed = parse(include_diffaid=False)
    fake = FakeRuntime(parsed)

    fake.set_step(16, "actual")
    assert compat.observe_external_patch_runtime(
        fake,
        visual_runtime(16 / 18, active=True),
    ) is True
    commit_pending(fake)

    fake.set_step(17, "forecast")
    assert compat.observe_external_patch_runtime(
        fake,
        visual_runtime(17 / 18, active=False),
    ) is True
    assert fake._step.mode == "actual"
    assert fake._step.reason == "external patch hard sigma transition"
    run = compat._compat_state(fake).run
    assert run.transitions == 1
    assert run.forced_actuals == 1


def test_visual_runtime_is_required_when_static_profile_exists():
    parsed = parse(include_diffaid=False)
    with pytest.raises(compat.ExternalPatchContractError, match="visual_reference_patch_runtime"):
        compat._runtime_entries({}, parsed)
