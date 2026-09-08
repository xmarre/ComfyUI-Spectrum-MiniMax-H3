"""Numerical attention policy/receipt identity, independent of any sparse package.

A provider in attention_backend_history_v1 maps the current layout/options/model
into a hashable preflight identity, or None when routing cannot be predicted.
Actual attention providers append hashable receipts before the last block returns.
Opaque routing is actual-only per call; it never aborts sampling.
"""
from dataclasses import dataclass

POLICIES = "attention_backend_history_v1"
RECEIPTS = "attention_backend_receipts_v1"


@dataclass
class BackendHistory:
    policy: object = None
    receipt: object = None
    forecast_safe: bool = False


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
        identity = provider(layout=layout, options=options, model=model)
        safe = safe and identity is not None
        identities.append((name, identity))
    return tuple(identities), safe


def prepare(runtime, run_id, step_id, options, layout, model):
    identity, safe = preflight(options, layout, model)
    if identity is None:
        if runtime._backend_history.policy is None:
            return options, None
        identity, safe = ("provider_removed",), False
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
        callable(getattr(provider, "accept_receipts", None)) and provider.accept_receipts(receipts)
        for provider in options.get(POLICIES, {}).values())
    runtime.observe_backend_history(run_id, step_id, identity, receipts, safe)
