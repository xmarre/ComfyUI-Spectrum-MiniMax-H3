from types import SimpleNamespace as N
import json
import random

import pytest
import torch

from comfyui_spectrum_h3 import bsa_transition_probe as probe


def test_pool_transaction_restores_contents_keys_and_all_identities_in_inference_mode():
    with torch.inference_mode():
        entry = (torch.ones(2, 3), torch.full((2, 3), 2.0))
        mapping = {("block",): entry}
        patch = N(pooled=mapping)
        with probe.isolated_pool(patch, torch.device("cpu")):
            entry[0].zero_()
            patch.pooled = {"foreign": (torch.zeros(1, 1), torch.zeros(1, 1))}
        assert patch.pooled is mapping
        assert patch.pooled[("block",)] is entry
        assert torch.equal(entry[0], torch.ones(2, 3))
        assert torch.equal(entry[1], torch.full((2, 3), 2.0))


@pytest.mark.parametrize("error", [RuntimeError, torch.cuda.OutOfMemoryError])
def test_transaction_restores_rng_and_pool_when_shadow_raises(error):
    patch = N(pooled={0: [torch.ones(1, 1), torch.ones(1, 1)]})
    tensor = patch.pooled[0][0]
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    with pytest.raises(error):
        with probe.isolated_pool(patch, torch.device("cpu")):
            tensor.zero_()
            torch.rand(5)
            random.random()
            patch.pooled[1] = (tensor, tensor)
            raise error("shadow failed")
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    assert list(patch.pooled) == [0]
    assert patch.pooled[0][0] is tensor and tensor.item() == 1


def test_malformed_pool_fails_before_diagnostic_mutation():
    patch = N(pooled={0: (None, torch.zeros(1, 1))})
    with pytest.raises(ValueError, match="malformed"):
        with probe.isolated_pool(patch, torch.device("cpu")):
            pytest.fail("should never enter")


def test_chunked_error_metrics_and_nonfinite():
    ref = torch.arange(262150, dtype=torch.float64).reshape(2, -1)
    report = probe.delta(ref, ref + 2)
    assert report["rmse"] == 2
    assert report["max_abs"] == 2
    assert report["mae"] == 2
    assert probe.delta(torch.ones(1), torch.tensor([float("nan")]))["finite"] is False


def _fixture(tmp_path):
    config = N(bootstrap_first_forecast=True, degree=1)
    p = probe.Probe(tmp_path, 1, config)
    rt = N(config=config, stats=N(forecast_model_calls=0),
           _run=N(min_actual_prefix_steps=2, state_conditioned_residual=False),
           _step=N(policy_step_id=0))
    patch = N(pooled={})
    def audit(route, seq=4):
        return N(identity=("source", ("layout", seq), ("routes", route),
                           ("pool_ownership", "cold" if route.endswith("cold") else "live")),
                 route_specs=((route, (0, 0), (0, 0)),), patch=patch,
                 seq_len=seq, uuids=("positive",), block_count=1, flow_mixed=True)
    return p, rt, patch, audit


def test_candidate_and_next_actual_measurements_preserve_control(tmp_path):
    p, rt, patch, audit = _fixture(tmp_path)
    key = (0, 4, ("positive",))
    # Complete cold control anchor.
    cold = probe.Actual(p, rt, 2, 0, audit("h3_chunked_sparse_cold"))
    patch.pooled[key] = (torch.ones(1, 2), torch.ones(1, 2))
    cold.capture_hidden(torch.ones(1, 4, 2), 1)
    cold.complete(True)
    entry = patch.pooled[key]
    calls = []
    def attention(h):
        existing = patch.pooled.get(key)
        scale = existing[1][0, 0].item() if existing is not None else 4.0
        calls.append(scale)
        if existing is None:
            patch.pooled[key] = (torch.zeros(1, 2), torch.zeros(1, 2))
        for tensor in patch.pooled[key]:
            tensor.add_(1)
        return h * scale
    candidate = probe.Actual(p, rt, 3, 0, audit("h3_chunked_sparse_primed"))
    assert candidate.candidate
    control = candidate.wrap_attention(attention, 0)(torch.ones(4, 2))
    assert calls == [1.0, 4.0]
    assert torch.equal(control, torch.ones(4, 2))
    assert patch.pooled[key] is entry and entry[1][0, 0] == 2
    candidate.capture_hidden(torch.full((1, 4, 2), 2.0), 1)
    rt._step.policy_step_id = 3
    candidate.complete(True)
    next_call = probe.Actual(p, rt, 4, 0, audit("h3_chunked_sparse_primed"))
    assert next_call.next_actual and not next_call.candidate
    control = next_call.wrap_attention(attention, 0)(torch.ones(4, 2))
    assert calls == [1, 4, 2, 1]
    assert torch.equal(control, torch.full((4, 2), 2.0))
    assert patch.pooled[key] is entry and entry[1][0, 0] == 3
    next_call.complete(True)
    p.close()
    events = [json.loads(line) for line in p.path.read_text().splitlines()]
    assert len([e for e in events if e["event"] == "forecast_vs_actual"]) == 1
    kinds = [m["measurement"] for e in events if e["event"] == "actual" for m in e["measurements"]]
    assert kinds == ["cold_vs_primed", "skipped_refresh_next_actual"]
    assert events[-1]["diagnostic_attention_calls_completed"] == 2
    assert events[-1]["diagnostic_transformer_nfe"] == 0


@pytest.mark.parametrize("change", ["layout", "nonadjacent", "owner", "dense"])
def test_candidate_requires_adjacent_matching_cold_anchor(tmp_path, change):
    p, rt, patch, audit = _fixture(tmp_path)
    cold = probe.Actual(p, rt, 2, 0, audit("h3_chunked_sparse_cold"))
    cold.capture_hidden(torch.ones(1, 4, 2), 1)
    cold.complete(True)
    a = audit("h3_dense" if change == "dense" else "h3_chunked_sparse_primed",
              seq=5 if change == "layout" else 4)
    if change == "owner":
        a.identity += (("owner", "new"),)
    next_call = probe.Actual(p, rt, 4 if change == "nonadjacent" else 3, 0, a)
    assert not next_call.candidate
    p.close()


def test_disabled_scope_and_attention_are_noops(monkeypatch):
    monkeypatch.delenv(probe.ENV, raising=False)
    function = lambda h: h
    with probe.actual_scope(N(), 1, 0, 0, {}) as scope:
        assert scope is None
        assert probe.attention(function, N(), 0) is function
