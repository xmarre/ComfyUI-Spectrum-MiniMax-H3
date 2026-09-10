"""Source-gated recovery of Spectrum forecasts across core-BSA cold->primed transitions.

CUDA diagnostics proved that ComfyUI core BlockSparseAttention's cold sparse call
creates the exact calibration tensor owners consumed by the adjacent primed call.
This module preserves Spectrum history only for that one-way, structurally proven
transition when the current Spectrum step was already selected as a forecast.
Dense->sparse and every unproven/foreign transition keep the normal hard reset.

The same integration promotes the previously test-only deferred one-point
bootstrap entitlement.  The entitlement belongs to one Spectrum run and is
consumed only by a successfully finalized forecast.  It is not a target-NFE
special case; it only lets an ordinary degree-1 bootstrap move past exact-prefix
or backend-veto steps until the immediately preceding retained actual anchor can
support it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import logging
from typing import Any

from .backend_history import BackendHistory

LOG = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL_PREPARE = None
_ORIGINAL_OBSERVE = None


@dataclass(slots=True)
class _ColdSuccessorProof:
    run_id: int
    step_id: int
    semantic_identity: tuple[Any, ...]
    patch: Any
    patch_generation: int
    source_blob: str
    seq_len: int
    uuids: tuple[Any, ...]
    sparse_blocks: tuple[int, ...]
    owners: tuple[tuple[Any, Any], ...]


def _semantic_identity(identity: Any) -> Any:
    """Drop only route/pool state; all other numerical identity remains binding."""
    if isinstance(identity, tuple):
        return tuple(
            _semantic_identity(item)
            for item in identity
            if not (
                isinstance(item, tuple)
                and item
                and item[0] in ("routes", "pool_ownership")
            )
        )
    return identity


def _clear_transition_proof(runtime) -> None:
    runtime._core_bsa_cold_successor_proof = None


def _capture_transition_proof(runtime, run_id: int, step_id: int, audit, receipts) -> None:
    """Capture exact post-cold calibration owners after an accepted actual call."""
    from . import core_bsa_compat

    _clear_transition_proof(runtime)
    if (
        audit is None
        or not audit.safe
        or not core_bsa_compat.accepts_actual(audit, receipts)
        or runtime._backend_history.policy != audit.identity
        or not runtime._backend_history.forecast_safe
    ):
        return

    sparse = tuple(
        index
        for index, spec in enumerate(audit.route_specs)
        if spec[0] != "h3_dense"
    )
    if not sparse or any(
        audit.route_specs[index][0] != "h3_chunked_sparse_cold"
        for index in sparse
    ):
        return

    owners = []
    for index in sparse:
        entry = core_bsa_compat._pool_entry(
            audit.patch,
            (index, audit.seq_len, audit.uuids),
            audit.pool_specs[index],
        )
        if entry[0] != "present":
            return
        owners.append((entry[1], entry[2]))

    runtime._core_bsa_cold_successor_proof = _ColdSuccessorProof(
        run_id=int(run_id),
        step_id=int(step_id),
        semantic_identity=_semantic_identity(audit.identity),
        patch=audit.patch,
        patch_generation=int(audit.patch_generation),
        source_blob=str(audit.source_blob),
        seq_len=int(audit.seq_len),
        uuids=tuple(audit.uuids),
        sparse_blocks=sparse,
        owners=tuple(owners),
    )


def _prove_forecast_carry(runtime, run_id: int, step_id: int, audit, identity, safe: bool) -> bool:
    """Prove the exact adjacent cold->primed transition needed by a forecast step."""
    from . import core_bsa_compat

    proof = getattr(runtime, "_core_bsa_cold_successor_proof", None)
    if proof is None:
        return False
    if proof.run_id != int(run_id) or proof.step_id + 1 != int(step_id):
        _clear_transition_proof(runtime)
        return False
    step = runtime._step
    if step is None or step.mode != "forecast" or not safe:
        _clear_transition_proof(runtime)
        return False
    if (
        audit is None
        or not audit.safe
        or audit.patch is not proof.patch
        or int(audit.patch_generation) != proof.patch_generation
        or str(audit.source_blob) != proof.source_blob
        or int(audit.seq_len) != proof.seq_len
        or tuple(audit.uuids) != proof.uuids
        or _semantic_identity(identity) != proof.semantic_identity
    ):
        _clear_transition_proof(runtime)
        return False

    sparse = tuple(
        index
        for index, spec in enumerate(audit.route_specs)
        if spec[0] != "h3_dense"
    )
    if sparse != proof.sparse_blocks or any(
        audit.route_specs[index][0] != "h3_chunked_sparse_primed"
        for index in sparse
    ):
        _clear_transition_proof(runtime)
        return False

    for index, owners in zip(sparse, proof.owners):
        entry = core_bsa_compat._pool_entry(
            audit.patch,
            (index, audit.seq_len, audit.uuids),
            audit.pool_specs[index],
        )
        if (
            entry[0] != "present"
            or entry[1] is not owners[0]
            or entry[2] is not owners[1]
        ):
            _clear_transition_proof(runtime)
            return False

    old = runtime._backend_history
    if old.policy is None or old.receipt is None or not old.forecast_safe:
        _clear_transition_proof(runtime)
        return False

    # Rebind only the policy identity.  The old accepted cold receipt remains as
    # provenance until the next actual call supplies a real primed receipt.  BSA
    # calibration itself is intentionally not modified or pretended refreshed.
    runtime._backend_history = BackendHistory(
        policy=identity,
        receipt=old.receipt,
        forecast_safe=True,
    )
    runtime._core_bsa_carry_step = int(step_id)
    _clear_transition_proof(runtime)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 core-BSA history carry run_id=%s step=%s "
            "transition=cold->primed proof=adjacent_exact_calibration_tensor_owners",
            run_id,
            step_id,
        )
    return True


def _prepare(runtime, run_id, step_id, options, layout, model):
    """Backend-history prepare with one narrow cold->primed carry exception."""
    from . import backend_history as history
    from . import core_bsa_flow_compat

    identity, safe, audit = history._preflight(options, layout, model)
    history._debug_core_bsa_preflight(
        runtime,
        run_id,
        step_id,
        identity,
        safe,
        audit,
        options=options,
        layout=layout,
        model=model,
    )
    if identity is None:
        if runtime._backend_history.policy is None:
            return options, None
        identity, safe = ("provider_removed",), False

    if runtime._backend_history.policy is None and not history._runtime_has_backend_evidence(runtime):
        runtime._backend_history = BackendHistory(policy=identity)

    _prove_forecast_carry(runtime, run_id, step_id, audit, identity, safe)
    runtime.prepare_backend_history(run_id, step_id, identity, safe)

    prepared = {**options, history.RECEIPTS: []}
    if audit is not None:
        prepared = core_bsa_flow_compat.instrument_actual_options(
            prepared,
            audit,
            history.RECEIPTS,
        )
    return prepared, (identity, safe)


def _observe(runtime, run_id, step_id, options, policy):
    """Run normal receipt observation, then retain exact post-cold owners if valid."""
    assert _ORIGINAL_OBSERVE is not None
    _ORIGINAL_OBSERVE(runtime, run_id, step_id, options, policy)
    if policy is None:
        _clear_transition_proof(runtime)
        return
    from . import backend_history as history
    from . import core_bsa_compat

    audit = options.get(core_bsa_compat.PRIVATE_AUDIT_KEY)
    receipts = tuple(options.get(history.RECEIPTS, ()))
    _capture_transition_proof(runtime, run_id, step_id, audit, receipts)


def _entitlement_available(runtime) -> bool:
    return bool(getattr(runtime, "_core_bsa_bootstrap_entitlement_unused", False))


def _can_defer_bootstrap(runtime, decision) -> bool:
    step = runtime._step
    run = runtime._run
    if (
        step is None
        or run is None
        or not decision.get("actual", True)
        or decision.get("reason") != "insufficient actual history"
        or not _entitlement_available(runtime)
        or not runtime.config.bootstrap_first_forecast
        or runtime.config.degree != 1
        or run.state_conditioned_residual
        or run.separate_stage_histories
        or runtime.offline_phase is not None
        or runtime.forecaster.history_length != 1
        or runtime._last_completed_mode != "actual"
        or runtime._last_completed_step_id != step.step_id - 1
    ):
        return False
    return runtime.forecaster.latest_anchor_ids(1) == (runtime._last_completed_step_id,)


def install_core_bsa_forecast_recovery() -> None:
    """Install the CUDA-evidence-backed production candidate exactly once."""
    global _INSTALLED, _ORIGINAL_PREPARE, _ORIGINAL_OBSERVE
    if _INSTALLED:
        return

    from . import backend_history as history
    from .runtime import SpectrumH3Runtime

    _ORIGINAL_PREPARE = history.prepare
    _ORIGINAL_OBSERVE = history.observe
    history.prepare = _prepare
    history.observe = _observe

    original_start = SpectrumH3Runtime.start_run
    original_begin = SpectrumH3Runtime.begin_step
    original_finalize = SpectrumH3Runtime.finalize_step
    original_abort = SpectrumH3Runtime.abort_step
    original_snapshot = SpectrumH3Runtime.create_rollback_snapshot
    original_restore = SpectrumH3Runtime.restore_rollback_snapshot
    original_end = SpectrumH3Runtime.end_run

    @wraps(original_start)
    def start_run(self, *args, **kwargs):
        run_id = original_start(self, *args, **kwargs)
        self._core_bsa_bootstrap_entitlement_unused = bool(
            self.config.bootstrap_first_forecast
        )
        self._core_bsa_cold_successor_proof = None
        self._core_bsa_carry_step = None
        self._core_bsa_recovery_snapshots = {}
        return run_id

    @wraps(original_begin)
    def begin_step(self, timestep):
        decision = original_begin(self, timestep)
        if _can_defer_bootstrap(self, decision):
            step = self._step
            step.mode = "forecast"
            step.reason = "deferred one-point bootstrap forecast"
            step.bootstrap_forecast = True
            decision["actual"] = False
            decision["reason"] = step.reason
            if self.config.debug:
                LOG.warning(
                    "Spectrum H3 deferred bootstrap entitlement run_id=%s step=%s "
                    "anchor_step=%s action=forecast",
                    decision["run_id"],
                    decision["step_id"],
                    self._last_completed_step_id,
                )
        return decision

    @wraps(original_finalize)
    def finalize_step(self, run_id, step_id):
        original_finalize(self, run_id, step_id)
        if (
            self._last_completed_step_id == int(step_id)
            and self._last_completed_mode == "forecast"
        ):
            self._core_bsa_bootstrap_entitlement_unused = False

    @wraps(original_abort)
    def abort_step(self, run_id, step_id):
        # No forecast has committed, so entitlement is intentionally untouched.
        return original_abort(self, run_id, step_id)

    @wraps(original_snapshot)
    def create_rollback_snapshot(self):
        snapshot = original_snapshot(self)
        state = (
            bool(getattr(self, "_core_bsa_bootstrap_entitlement_unused", False)),
            getattr(self, "_core_bsa_cold_successor_proof", None),
            getattr(self, "_core_bsa_carry_step", None),
        )
        self._core_bsa_recovery_snapshots[id(snapshot)] = state
        return snapshot

    @wraps(original_restore)
    def restore_rollback_snapshot(self, snapshot):
        state = self._core_bsa_recovery_snapshots.get(id(snapshot))
        original_restore(self, snapshot)
        if state is not None:
            (
                self._core_bsa_bootstrap_entitlement_unused,
                self._core_bsa_cold_successor_proof,
                self._core_bsa_carry_step,
            ) = state

    @wraps(original_end)
    def end_run(self, *args, **kwargs):
        try:
            return original_end(self, *args, **kwargs)
        finally:
            self._core_bsa_cold_successor_proof = None
            self._core_bsa_carry_step = None
            self._core_bsa_recovery_snapshots = {}

    SpectrumH3Runtime.start_run = start_run
    SpectrumH3Runtime.begin_step = begin_step
    SpectrumH3Runtime.finalize_step = finalize_step
    SpectrumH3Runtime.abort_step = abort_step
    SpectrumH3Runtime.create_rollback_snapshot = create_rollback_snapshot
    SpectrumH3Runtime.restore_rollback_snapshot = restore_rollback_snapshot
    SpectrumH3Runtime.end_run = end_run

    _INSTALLED = True
