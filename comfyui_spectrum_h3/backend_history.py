"""Numerical attention policy/receipt identity, independent of any sparse package.

A provider in attention_backend_history_v1 maps the current layout/options/model
into a hashable preflight identity, or None when routing cannot be predicted.
Actual attention providers append hashable receipts before the last block returns.
Opaque routing is actual-only per call; it never aborts sampling.
"""
from dataclasses import dataclass
import logging

import torch

POLICIES = "attention_backend_history_v1"
RECEIPTS = "attention_backend_receipts_v1"

LOG = logging.getLogger(__name__)


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


def _preflight(options, layout, model):
    policies = options.get(POLICIES, {})
    if policies:
        identities = []
        safe = True
        for name, provider in sorted(policies.items()):
            identity = _provider_identity(provider, layout=layout, options=options, model=model)
            safe = safe and identity is not None
            identities.append((name, identity))
        return tuple(identities), safe, None

    # Core ComfyUI BSA does not publish the generic provider contract. Spectrum
    # owns narrowly source-gated compatibility audits for the exact reviewed BSA,
    # Untwist preprocessor and Flow block-wrapper composition. Any unknown source
    # or ownership remains actual-only.
    from . import core_bsa_flow_compat, core_bsa_preprocess_compat

    if core_bsa_flow_compat.has_core_bsa_evidence(options):
        audit, reason = core_bsa_preprocess_compat.probe(options, layout, model)
        if audit is not None:
            return audit.identity, audit.safe, audit
        return ("core_bsa_unreported", reason or "unrecognized"), False, None

    return None, True, None


def preflight(options, layout, model):
    identity, safe, _audit = _preflight(options, layout, model)
    return identity, safe


def _debug_core_bsa_preflight(
    runtime,
    run_id,
    step_id,
    identity,
    safe,
    audit,
    *,
    options,
    layout,
    model,
):
    if not runtime.config.debug:
        return
    if audit is not None:
        routes = ",".join(str(spec[0]) for spec in audit.route_specs)
        flow_mode = getattr(audit, "flow_mixed", None)
        LOG.warning(
            "Spectrum H3 core-BSA preflight run_id=%s step=%s result=audited safe=%s "
            "seq_len=%s routes=%s flow_mixed=%s failure=%s",
            run_id,
            step_id,
            bool(safe),
            audit.seq_len,
            routes,
            flow_mode,
            audit.failure,
        )
        return
    if (
        isinstance(identity, tuple)
        and len(identity) >= 2
        and identity[0] == "core_bsa_unreported"
    ):
        detail = None
        if identity[1] == "ownership_unproven":
            try:
                from . import core_bsa_diagnostics

                detail = core_bsa_diagnostics.ownership_diagnostic(
                    options,
                    layout,
                    model,
                )
            except Exception as exc:  # noqa: BLE001 - debug detail cannot affect sampling
                detail = f"diagnostic_failed:{type(exc).__name__}:{exc}"
        LOG.warning(
            "Spectrum H3 core-BSA preflight run_id=%s step=%s result=unreported "
            "safe=False reason=%s detail=%s",
            run_id,
            step_id,
            identity[1],
            detail,
        )


def prepare(runtime, run_id, step_id, options, layout, model):
    identity, safe, audit = _preflight(options, layout, model)
    _debug_core_bsa_preflight(
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
    # The first declared provider establishes the run's numerical identity. There
    # is nothing to invalidate until Spectrum has retained backend-dependent
    # evidence. A provider appearing later in the run still goes through the full
    # transition/reset path below.
    if runtime._backend_history.policy is None and not _runtime_has_backend_evidence(runtime):
        runtime._backend_history = BackendHistory(policy=identity)
    runtime.prepare_backend_history(run_id, step_id, identity, safe)

    prepared = {**options, RECEIPTS: []}
    if audit is not None:
        from . import core_bsa_flow_compat

        prepared = core_bsa_flow_compat.instrument_actual_options(
            prepared, audit, RECEIPTS
        )
    return prepared, (identity, safe)


def observe(runtime, run_id, step_id, options, policy):
    if policy is None:
        return
    identity, safe = policy
    receipts = tuple(options.get(RECEIPTS, ()))

    from . import core_bsa_compat

    audit = options.get(core_bsa_compat.PRIVATE_AUDIT_KEY)
    if audit is not None:
        accepted = core_bsa_compat.accepts_actual(audit, receipts)
        safe = safe and accepted
        if runtime.config.debug:
            routes = ",".join(
                str(item[4]) if isinstance(item, tuple) and len(item) > 4 else "malformed"
                for item in receipts
            )
            LOG.warning(
                "Spectrum H3 core-BSA observe run_id=%s step=%s accepted=%s "
                "preflight_safe=%s receipts=%s expected=%s failure=%s routes=%s",
                run_id,
                step_id,
                bool(accepted),
                bool(policy[1]),
                len(receipts),
                len(audit.expected_receipts),
                audit.failure,
                routes,
            )
    else:
        # A fallback is not proof that the next call follows the same route. Until
        # a provider can predict it, execute actuals and re-enable forecasts after
        # the provider reports a stable route.
        safe = safe and bool(receipts) and all(
            _provider_accepts(provider, receipts)
            for provider in options.get(POLICIES, {}).values()
        )
        if (
            runtime.config.debug
            and isinstance(identity, tuple)
            and len(identity) >= 2
            and identity[0] == "core_bsa_unreported"
        ):
            LOG.warning(
                "Spectrum H3 core-BSA observe run_id=%s step=%s accepted=False "
                "reason=%s receipts=%s",
                run_id,
                step_id,
                identity[1],
                len(receipts),
            )
    runtime.observe_backend_history(run_id, step_id, identity, receipts, safe)
