"""Numerical attention policy/receipt identity, independent of any sparse package.

A provider in attention_backend_history_v1 maps the current layout/options/model
into a hashable preflight identity, or None when routing cannot be predicted.
Actual attention providers append hashable receipts before the last block returns.
Opaque routing is actual-only per call; it never aborts sampling.
"""
from dataclasses import dataclass

import torch

POLICIES = "attention_backend_history_v1"
RECEIPTS = "attention_backend_receipts_v1"


@dataclass
class BackendHistory:
    policy: object = None
    receipt: object = None
    forecast_safe: bool = False


def _provider_identity(provider, *, layout, options, model):
    try:
        return provider(layout=layout, options=options, model=model)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - metadata providers fail closed to actual-only
        return None


def _provider_accepts(provider, receipts):
    accept = getattr(provider, "accept_receipts", None)
    if not callable(accept):
        return False
    try:
        return bool(accept(receipts))
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - metadata providers fail closed to actual-only
        return False


def _runtime_has_backend_evidence(runtime) -> bool:
    history = runtime._backend_history
    if history.policy is not None or history.receipt is not None:
        return True
    if runtime._history_topology is not None or runtime._history_labels is not None:
        return True
    forecasters = [runtime._primary_forecaster, *runtime._stage_forecasters.values()]
    if any(forecaster.history_length for forecaster in forecasters):
        return True
    step = runtime._step
    if step is not None and (step.calls or step.actual_records or step.used_history_rows):
        return True
    archive = runtime._offline_archive
    if archive is not None and (archive.steps or archive.anchors):
        return True
    return runtime._offline_smoother is not None


def preflight(options, layout, model):
    policies = options.get(POLICIES, {})
    if not policies:
        # Core BSA predates the provider contract. Preserve its operator while
        # requiring actual execution; never forecast across its unreported schedule.
        callbacks = options.get("callbacks", {})
        if any("block_sparse_attention" in group for group in callbacks.values()):
            return ("core_bsa_unreported",), False
        return None, True
    identities = []
    safe = True
    for name, provider in sorted(policies.items()):
        identity = _provider_identity(provider, layout=layout, options=options, model=model)
        safe = safe and identity is not None
        identities.append((name, identity))
    return tuple(identities), safe


def prepare(runtime, run_id, step_id, options, layout, model):
    identity, safe = preflight(options, layout, model)
    if identity is None:
        if runtime._backend_history.policy is None:
            return options, None
        identity, safe = ("provider_removed",), False
    # The first declared provider establishes the run's numerical identity. There
    # is nothing to invalidate until Spectrum has retained backend-dependent
    # evidence. A provider appearing later in the run still goes through the full
    # transition/reset path below.
    if runtime._backend_history.policy is None and not _runtime_has_backend_evidence(runtime):
        runtime._backend_history = BackendHistory(policy=identity)
    runtime.prepare_backend_history(run_id, step_id, identity, safe)
    return {**options, RECEIPTS: []}, (identity, safe)


def observe(runtime, run_id, step_id, options, policy):
    if policy is None:
        return
    identity, safe = policy
    receipts = tuple(options.get(RECEIPTS, ()))
    # A fallback is not proof that the next call follows the same route. Until a
    # provider can predict it, execute actuals and re-enable forecasts after SOL
    # eligibility returns. Stable dense warmup is predictable through its policy.
    safe = safe and bool(receipts) and all(
        _provider_accepts(provider, receipts)
        for provider in options.get(POLICIES, {}).values()
    )
    runtime.observe_backend_history(run_id, step_id, identity, receipts, safe)
