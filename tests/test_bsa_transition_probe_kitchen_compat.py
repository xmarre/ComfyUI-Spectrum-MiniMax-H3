from types import ModuleType, SimpleNamespace as N

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
