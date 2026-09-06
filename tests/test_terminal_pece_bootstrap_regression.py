from __future__ import annotations

from types import SimpleNamespace

from comfyui_spectrum_h3 import external_patch_compat as compat
from comfyui_spectrum_h3 import external_patch_visual_reference as visual
from comfyui_spectrum_h3.runtime import ExactCorrectorGuarantee


INSTANCE = "untwist-h3-bootstrap-regression"


class _FakeRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(debug=False, model_aware_mode="off")
        self._step = SimpleNamespace(
            step_id=1,
            policy_step_id=1,
            phase="predicted",
            mode="forecast",
            reason="solver-owned one-point bootstrap forecast",
            adaptive_recompute=False,
            bootstrap_forecast=True,
            model_aware_decision=None,
            model_aware_forced_actual=False,
        )
        self.disabled_reason = None
        setattr(
            self,
            compat._CURRENT_DECISION_ATTR,
            {
                "run_id": 1,
                "step_id": 1,
                "actual": False,
                "reason": "solver-owned one-point bootstrap forecast",
            },
        )

    def terminal_pece_exact_corrector_after_current_step(self):
        return ExactCorrectorGuarantee(
            predicted_step_id=1,
            corrected_step_id=2,
            outer_step=1,
        )

    def _disable_forecasting(self, reason: str) -> None:
        self.disabled_reason = str(reason)


def _parsed_terminal_safe_contract() -> compat.ParsedExternalPatchContracts:
    profile = {
        "schema_version": 2,
        "provider": visual.VISUAL_PATCH_PROVIDER,
        "kind": visual.VISUAL_PATCH_KIND,
        "architecture": visual.VISUAL_PATCH_ARCHITECTURE,
        "instance_id": INSTANCE,
        "block_indices_0based": list(range(50)),
        "model_block_count": 50,
        "strength": 0.05,
        "progress_start": 0.0,
        "progress_end": 0.95,
        "hard_start": False,
        "hard_end": True,
        "scope": "image_and_video",
        "high_scale_start": 0.95,
        "high_scale_end": 1.0,
        "low_scale_start": 1.0,
        "low_scale_end": 1.05,
        "beta": 2.0,
        "scale_temporal_axis": False,
        "terminal_pece_exact_corrector_safe": True,
    }
    return compat.parse_external_patch_contracts(
        {visual.VISUAL_PATCH_PROFILES_KEY: (profile,)},
        block_count=50,
    )


def _runtime_options(progress: float) -> dict:
    return {
        visual.VISUAL_PATCH_RUNTIME_KEY: (
            {
                "schema_version": 2,
                "provider": visual.VISUAL_PATCH_PROVIDER,
                "instance_id": INSTANCE,
                "schedule_progress": progress,
                "active": progress <= 0.95,
            },
        )
    }


def test_terminal_pece_deferral_preserves_one_point_bootstrap_execution_mode():
    runtime = _FakeRuntime()
    parsed = _parsed_terminal_safe_contract()
    descriptor = parsed.descriptors[0]
    assert descriptor.terminal_pece_exact_corrector_safe is True
    assert descriptor.active_at(0.10) is True
    assert descriptor.active_at(0.0) is False

    state = compat._compat_state(runtime)
    state.parsed = parsed
    state.run = compat._ExternalRunState(
        run_id=1,
        parsed=parsed,
        replay=False,
        committed_active=(True,),
        committed_sigma=(0.10,),
        committed_step_id=0,
    )

    # Reproduces a fresh max_speed PECE lifetime after its single exact prefix:
    # P1 is a valid solver-owned one-point bootstrap, while Untwist leaves its
    # reviewed late hard window and qualifies for the exact same-outer corrector
    # deferral. The external-patch layer may replace the reason, but must not
    # replace the bootstrap execution mechanism with a multi-point spectral fit.
    assert compat.observe_external_patch_runtime(runtime, _runtime_options(1.0)) is False

    assert runtime._step.mode == "forecast"
    assert runtime._step.reason == compat.TERMINAL_PECE_DEFERRED_REASON
    assert runtime._step.bootstrap_forecast is True
    assert runtime.disabled_reason is None

    decision = getattr(runtime, compat._CURRENT_DECISION_ATTR)
    assert decision["actual"] is False
    assert decision["reason"] == compat.TERMINAL_PECE_DEFERRED_REASON

    assert state.run.transitions == 1
    assert state.run.forced_actuals == 0
    assert state.run.terminal_pece_deferred == 1
    assert state.run.pending_terminal_pece_transition is not None
