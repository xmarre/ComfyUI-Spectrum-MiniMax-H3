from types import ModuleType, SimpleNamespace as N
import json

import torch

from comfyui_spectrum_h3 import bsa_transition_probe as probe
from comfyui_spectrum_h3 import bsa_transition_probe_compat as compat


def test_source_built_comfy_kitchen_is_not_rejected_by_version_or_blob(monkeypatch, tmp_path):
    source = tmp_path / "cuda.py"
    source.write_text("# locally built current comfy-kitchen source\n")
    cuda = ModuleType("comfy_kitchen.backends.cuda")
    cuda.__file__ = str(source)

    def sol_attn_chunked(qkv_chunks, t, h, rope_freqs, qk_norm_weights, *, kmean=None, vscale=None):
        return qkv_chunks, kmean, vscale

    cuda.sol_attn_chunked = sol_attn_chunked
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "999.0.dev-current")
    monkeypatch.setattr(compat.importlib, "import_module", lambda _name: cuda)

    version, blob, signature = compat._source_provenance()
    assert version == "999.0.dev-current"
    assert blob is not None
    assert "kmean" in signature and "vscale" in signature


def test_missing_sol_attn_chunked_is_a_type_contract_error(monkeypatch):
    cuda = ModuleType("comfy_kitchen.backends.cuda")
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "0.2.33")
    monkeypatch.setattr(compat.importlib, "import_module", lambda _name: cuda)

    try:
        compat._source_provenance()
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing sol_attn_chunked must be rejected")
    assert "sol_attn_chunked" in message


def test_only_required_sol_attn_chunked_api_contract_is_enforced(monkeypatch):
    cuda = ModuleType("comfy_kitchen.backends.cuda")

    def sol_attn_chunked(qkv_chunks, t, h):
        return qkv_chunks

    cuda.sol_attn_chunked = sol_attn_chunked
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "0.2.33")
    monkeypatch.setattr(compat.importlib, "import_module", lambda _name: cuda)

    try:
        compat._source_provenance()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing calibration parameters must be rejected")
    assert "kmean" in message and "vscale" in message


def test_installed_probe_uses_compatibility_provenance_without_exact_build_pin(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(probe, "_diagnostic_directory", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(
        compat,
        "_source_provenance",
        lambda: ("0.2.33+source", "different-current-source-blob", "(..., kmean=None, vscale=None)"),
    )
    runtime = N(_run=object(), config=N())

    current = probe.get_probe(runtime, 17, automatic=True)
    try:
        assert current is not None
        assert runtime._bsa_transition_probe[1] is current
        record = current.path.read_text().splitlines()[1]
        assert "different-current-source-blob" in record
        assert "runtime_api_contract_not_exact_build_pin" in record
    finally:
        current.close()
        runtime._bsa_transition_probe = None


def test_one_jsonl_is_shared_by_all_runtime_segments_of_one_generation(monkeypatch, tmp_path):
    compat._close_sinks()
    monkeypatch.setattr(compat, "_execution_generation_id", lambda: "prompt-generation-123")
    config = N()

    first = probe.Probe(tmp_path, 1, config)
    first.write("marker", segment="low")
    first.close()
    second = probe.Probe(tmp_path, 3, config)
    second.write("marker", segment="high")
    second.close()

    try:
        assert first.path == second.path
        assert len(list(tmp_path.glob("bsa-*.jsonl"))) == 1
        events = [json.loads(line) for line in first.path.read_text().splitlines()]
        starts = [event for event in events if event["event"] == "start"]
        assert len(starts) == 2
        assert {event["run_id"] for event in starts} == {1, 3}
        assert len({event["run_instance"] for event in starts}) == 2
        assert all(event["schema_version"] == 4 for event in starts)
        markers = [event["segment"] for event in events if event["event"] == "marker"]
        assert markers == ["low", "high"]
    finally:
        compat._close_sinks()


def test_new_prompt_generation_gets_a_new_jsonl(monkeypatch, tmp_path):
    compat._close_sinks()
    generation = ["prompt-a"]
    monkeypatch.setattr(compat, "_execution_generation_id", lambda: generation[0])

    first = probe.Probe(tmp_path, 1, N())
    first.close()
    generation[0] = "prompt-b"
    second = probe.Probe(tmp_path, 1, N())
    second.close()

    try:
        assert first.path != second.path
        assert len(list(tmp_path.glob("bsa-*.jsonl"))) == 2
    finally:
        compat._close_sinks()


def test_next_actual_measurement_can_cross_settings_identity_with_same_pool_owners(
    monkeypatch, tmp_path
):
    compat._close_sinks()
    monkeypatch.setattr(compat, "_execution_generation_id", lambda: None)
    diagnostic = probe.Probe(tmp_path, 3, N())
    entry = (torch.ones(1, 2), torch.ones(1, 2))
    patch = N(pooled={(0, 4, ("positive",)): entry})
    diagnostic.pending = {
        "identity": ("settings", "previous"),
        "step": 1,
        "stale": {0: (entry, tuple(tensor.clone() for tensor in entry))},
    }
    runtime = N()
    audit = N(
        identity=("settings", "current"),
        route_specs=(("h3_chunked_sparse_primed", (0, 0), (0, 0)),),
        patch=patch,
        seq_len=4,
        uuids=("positive",),
    )

    try:
        current = probe.Actual(diagnostic, runtime, 2, 0, audit)
        assert current.next_actual is True
        assert current.next_actual_identity_changed is True
    finally:
        diagnostic.close()
        compat._close_sinks()
