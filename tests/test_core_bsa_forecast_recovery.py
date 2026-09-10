from types import SimpleNamespace as N

import torch

from comfyui_spectrum_h3 import core_bsa_compat
from comfyui_spectrum_h3.backend_history import BackendHistory
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.core_bsa_forecast_recovery import (
    _capture_transition_proof,
    _prove_forecast_carry,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


TOPOLOGY = (
    ("video", (1, 24, 2, 4, 4)),
    ("audio", (1, 32, 2, 8)),
    ("hidden", 4),
    ("target_audio_rows", 1),
    ("target_video_rows", 2),
)
LABEL = ((0, "positive"),)


def _runtime(**overrides):
    values = {
        "degree": 1,
        "max_history": 4,
        "warmup_steps": 0,
        "tail_actual_steps": 0,
        "window_size": 2.0,
        "bootstrap_first_forecast": True,
        "offline_smoothing_replay": False,
    }
    values.update(overrides)
    return SpectrumH3Runtime(SpectrumH3Config(**values))


def _start(runtime, steps, **kwargs):
    return runtime.start_run(
        torch.linspace(1.0, 0.0, steps + 1),
        "sample_res_multistep",
        supported_sampler=True,
        min_actual_steps_after_forecast=0,
        **kwargs,
    )


def _complete(runtime, timestep, *, policy=None):
    decision = runtime.begin_step(torch.tensor([timestep]))
    if policy is not None:
        runtime.prepare_backend_history(
            decision["run_id"], decision["step_id"], policy, True
        )
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    if actual:
        runtime.observe_actual(
            decision["run_id"],
            decision["step_id"],
            call_id,
            torch.full((1, 3, 4), float(decision["step_id"])),
        )
        if policy is not None:
            runtime.observe_backend_history(
                decision["run_id"],
                decision["step_id"],
                policy,
                ("receipt", policy),
                True,
            )
    else:
        assert runtime.predict(
            decision["run_id"],
            decision["step_id"],
            call_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ) is not None
    final_mode = runtime._step.mode
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision, final_mode


def test_real_runtime_consumes_entitlement_only_after_committed_forecast():
    runtime = _runtime()
    run_id = _start(runtime, 3)
    _complete(runtime, 1.0)
    decision, mode = _complete(runtime, 0.5)
    assert decision["reason"] == "one-point bootstrap forecast"
    assert mode == "forecast"
    assert runtime._core_bsa_bootstrap_entitlement_unused is False
    runtime.end_run(run_id)


def test_real_runtime_defers_bootstrap_after_exact_prefix_and_backend_reset():
    runtime = _runtime()
    run_id = _start(runtime, 4, min_actual_prefix_steps=2)
    _complete(runtime, 1.0, policy="dense")
    second, mode = _complete(runtime, 0.75, policy="sparse-cold")
    assert mode == "actual"
    assert second["reason"] == "H3 Continuum actual prefix"
    assert runtime.forecaster.latest_anchor_ids(1) == (1,)
    assert runtime._core_bsa_bootstrap_entitlement_unused is True

    decision, mode = _complete(runtime, 0.5, policy="sparse-cold")
    assert decision["reason"] == "deferred one-point bootstrap forecast"
    assert mode == "forecast"
    assert runtime._core_bsa_bootstrap_entitlement_unused is False
    runtime.end_run(run_id)


def test_backend_vetoed_bootstrap_does_not_consume_entitlement():
    runtime = _runtime()
    run_id = _start(runtime, 4)
    _complete(runtime, 1.0, policy="a")

    decision = runtime.begin_step(torch.tensor([0.75]))
    assert decision["actual"] is False
    runtime.prepare_backend_history(
        decision["run_id"], decision["step_id"], "b", True
    )
    assert runtime._step.mode == "actual"
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=LABEL,
        expected_shape=(1, 3, 4),
    )
    assert actual
    runtime.observe_actual(
        decision["run_id"], decision["step_id"], call_id, torch.ones(1, 3, 4)
    )
    runtime.observe_backend_history(
        decision["run_id"], decision["step_id"], "b", ("receipt", "b"), True
    )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime._core_bsa_bootstrap_entitlement_unused is True

    deferred, mode = _complete(runtime, 0.5, policy="b")
    assert deferred["reason"] == "deferred one-point bootstrap forecast"
    assert mode == "forecast"
    runtime.end_run(run_id)


def test_rollback_restores_entitlement_state():
    runtime = _runtime()
    run_id = _start(runtime, 4)
    _complete(runtime, 1.0)
    snapshot = runtime.create_rollback_snapshot()
    _complete(runtime, 0.75)
    assert runtime._core_bsa_bootstrap_entitlement_unused is False
    runtime.restore_rollback_snapshot(snapshot)
    assert runtime._core_bsa_bootstrap_entitlement_unused is True
    runtime.end_run(run_id)


def _audit(route, patch, owners, *, identity_tail="same"):
    routes = (
        ("h3_dense", (0, 0), (0, 0)),
        ("h3_dense", (0, 0), (0, 0)),
        (route, (0, 1), (0, 0)),
    )
    identity = (
        core_bsa_compat.ADAPTER_KEY,
        core_bsa_compat.ADAPTER_VERSION,
        "006d1eb352f946a7c85595edf75d3cda9ac79194",
        17,
        ("settings", (False, 1.0, 0.0)),
        ("layout", ("layout", 1)),
        ("routes", routes),
        ("pool_ownership", ((0, "missing"), (1, "missing"), (2, *owners))),
        ("ownership", identity_tail),
    )
    return N(
        identity=identity,
        safe=True,
        patch=patch,
        patch_generation=17,
        block_count=3,
        seq_len=64,
        uuids=("u",),
        layout_identity=("layout", 1),
        settings_identity=(False, 1.0, 0.0),
        route_specs=routes,
        pool_specs=((1, 2, None), (1, 2, None), (1, 2, None)),
        expected_receipts=(("r", 0), ("r", 1), ("r", 2)),
        source_blob="006d1eb352f946a7c85595edf75d3cda9ac79194",
    )


def test_cold_to_primed_carry_requires_exact_adjacent_calibration_owners(monkeypatch):
    kmean = torch.zeros(1, 2, dtype=torch.float32)
    vscale = torch.ones(1, 2, dtype=torch.float32)
    patch = N(pooled={(2, 64, ("u",)): (kmean, vscale)})
    cold = _audit("h3_chunked_sparse_cold", patch, ("missing",))
    primed = _audit("h3_chunked_sparse_primed", patch, (11, 12))
    # Route/pool fields are intentionally the only semantic difference.
    assert cold.identity != primed.identity

    runtime = N(
        _backend_history=BackendHistory(cold.identity, cold.expected_receipts, True),
        _step=N(mode="actual"),
        config=N(debug=False),
    )
    monkeypatch.setattr(core_bsa_compat, "accepts_actual", lambda *_args: True)
    _capture_transition_proof(runtime, 4, 2, cold, cold.expected_receipts)
    assert runtime._core_bsa_cold_successor_proof is not None

    runtime._step.mode = "forecast"
    assert _prove_forecast_carry(runtime, 4, 3, primed, primed.identity, True)
    assert runtime._backend_history.policy == primed.identity
    assert runtime._backend_history.receipt == cold.expected_receipts
    assert runtime._core_bsa_cold_successor_proof is None


def test_cold_to_primed_carry_rejects_replaced_calibration_owner(monkeypatch):
    kmean = torch.zeros(1, 2, dtype=torch.float32)
    vscale = torch.ones(1, 2, dtype=torch.float32)
    patch = N(pooled={(2, 64, ("u",)): (kmean, vscale)})
    cold = _audit("h3_chunked_sparse_cold", patch, ("missing",))
    primed = _audit("h3_chunked_sparse_primed", patch, (11, 12))
    runtime = N(
        _backend_history=BackendHistory(cold.identity, cold.expected_receipts, True),
        _step=N(mode="actual"),
        config=N(debug=False),
    )
    monkeypatch.setattr(core_bsa_compat, "accepts_actual", lambda *_args: True)
    _capture_transition_proof(runtime, 4, 2, cold, cold.expected_receipts)

    patch.pooled[(2, 64, ("u",))] = (kmean.clone(), vscale)
    runtime._step.mode = "forecast"
    assert not _prove_forecast_carry(runtime, 4, 3, primed, primed.identity, True)
    assert runtime._backend_history.policy == cold.identity


def test_carry_is_not_used_for_actual_steps(monkeypatch):
    kmean = torch.zeros(1, 2, dtype=torch.float32)
    vscale = torch.ones(1, 2, dtype=torch.float32)
    patch = N(pooled={(2, 64, ("u",)): (kmean, vscale)})
    cold = _audit("h3_chunked_sparse_cold", patch, ("missing",))
    primed = _audit("h3_chunked_sparse_primed", patch, (11, 12))
    runtime = N(
        _backend_history=BackendHistory(cold.identity, cold.expected_receipts, True),
        _step=N(mode="actual"),
        config=N(debug=False),
    )
    monkeypatch.setattr(core_bsa_compat, "accepts_actual", lambda *_args: True)
    _capture_transition_proof(runtime, 4, 2, cold, cold.expected_receipts)
    assert not _prove_forecast_carry(runtime, 4, 3, primed, primed.identity, True)
