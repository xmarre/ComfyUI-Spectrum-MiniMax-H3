from __future__ import annotations

from types import SimpleNamespace

from comfyui_spectrum_h3 import backend_history, core_bsa_compat, core_bsa_forecast_recovery


def _audit():
    core = SimpleNamespace(ATTENTION_MEASURE_KEY=core_bsa_compat.ATTENTION_MEASURE_KEY)
    return SimpleNamespace(measure=SimpleNamespace(core=core))


def _receipt(*, call_generation=1, route="h3_chunked_sparse_primed", digest="measure-a"):
    block = 0
    measure = (
        core_bsa_compat.ATTENTION_MEASURE_KEY,
        (call_generation, block),
        block,
        "owner-generation",
        digest,
        core_bsa_compat.SPARSE_MEASURE_PROFILE,
        core_bsa_compat.SPARSE_MEASURE_ROUTE,
        192,
        192,
        "exact-range-digest",
        core_bsa_compat.SPARSE_MEASURE_PREPROCESS,
        True,
    )
    return (
        core_bsa_compat.ADAPTER_KEY,
        core_bsa_compat.ADAPTER_VERSION,
        7,
        block,
        route,
        192,
        (0, 3),
        (0, 0),
        measure,
    )


def test_weighted_bsa_call_token_is_not_numerical_receipt_identity():
    audit = _audit()
    first = backend_history._core_bsa_numerical_receipts(
        audit, (_receipt(call_generation=11),)
    )
    second = backend_history._core_bsa_numerical_receipts(
        audit, (_receipt(call_generation=12),)
    )

    assert first == second
    normalized_measure = first[0][8]
    assert normalized_measure[0] == core_bsa_compat.ATTENTION_MEASURE_KEY
    assert (11, 0) not in normalized_measure
    assert normalized_measure[1] == 0


def test_weighted_bsa_numerical_receipt_keeps_route_and_measure_identity():
    audit = _audit()
    baseline = backend_history._core_bsa_numerical_receipts(
        audit, (_receipt(call_generation=11),)
    )
    changed_route = backend_history._core_bsa_numerical_receipts(
        audit, (_receipt(call_generation=12, route="h3_chunked_sparse_cold"),)
    )
    changed_measure = backend_history._core_bsa_numerical_receipts(
        audit, (_receipt(call_generation=12, digest="measure-b"),)
    )

    assert changed_route != baseline
    assert changed_measure != baseline


def test_unweighted_bsa_receipts_remain_byte_for_byte_compatible():
    receipt = (
        core_bsa_compat.ADAPTER_KEY,
        core_bsa_compat.ADAPTER_VERSION,
        7,
        0,
        "h3_chunked_sparse_primed",
        192,
        (0, 1),
        (0, 0),
    )
    audit = SimpleNamespace(measure=None)
    receipts = (receipt,)
    assert backend_history._core_bsa_numerical_receipts(audit, receipts) is receipts


def test_observe_verifies_raw_receipt_but_stores_token_free_identity(monkeypatch):
    audit = _audit()
    raw = (_receipt(call_generation=31),)
    accepted = []

    def accepts_actual(observed_audit, receipts):
        accepted.append((observed_audit, receipts))
        return True

    monkeypatch.setattr(core_bsa_compat, "accepts_actual", accepts_actual)

    class Runtime:
        config = SimpleNamespace(debug=False)

        def __init__(self):
            self.observed = None

        def observe_backend_history(self, run_id, step_id, identity, receipts, safe):
            self.observed = (run_id, step_id, identity, receipts, safe)

    runtime = Runtime()
    options = {
        core_bsa_compat.PRIVATE_AUDIT_KEY: audit,
        backend_history.RECEIPTS: list(raw),
    }
    original_observe = core_bsa_forecast_recovery._ORIGINAL_OBSERVE
    assert original_observe is not None
    original_observe(runtime, 4, 9, options, ("policy", True))

    assert accepted == [(audit, raw)]
    assert runtime.observed[:3] == (4, 9, "policy")
    assert runtime.observed[4] is True
    assert runtime.observed[3] == backend_history._core_bsa_numerical_receipts(audit, raw)
    assert runtime.observed[3] != raw
