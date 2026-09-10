"""Compatibility shim for the diagnostic probe's comfy-kitchen provenance check.

The diagnostic needs the public ``sol_attn_chunked`` calibration contract, not a
particular wheel build or source blob.  Local/source builds from current upstream
may legitimately keep the same package version while changing unrelated CUDA
wrapper code, so exact version/blob pinning is inappropriate here.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path


def _source_provenance():
    """Return installed comfy-kitchen provenance after checking the API contract."""
    version = importlib.metadata.version("comfy-kitchen")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    function = getattr(cuda, "sol_attn_chunked", None)
    if not callable(function):
        raise RuntimeError(
            "BSA diagnostics require a comfy-kitchen CUDA backend exposing sol_attn_chunked"
        )

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "BSA diagnostics could not inspect comfy-kitchen sol_attn_chunked"
        ) from exc
    missing = sorted({"kmean", "vscale"} - set(parameters))
    if missing:
        raise RuntimeError(
            "BSA diagnostics require sol_attn_chunked calibration parameters: "
            + ", ".join(missing)
        )

    source_path = getattr(cuda, "__file__", None)
    source = b""
    if source_path:
        try:
            source = Path(source_path).read_bytes()
        except OSError:
            pass
    blob = (
        hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
        if source
        else None
    )
    return version, blob, str(inspect.signature(function))


def install_bsa_transition_probe_compat():
    """Replace only the diagnostic's obsolete exact comfy-kitchen build pin."""
    from . import bsa_transition_probe as probe

    if getattr(probe.get_probe, "_spectrum_compatible_kitchen", False):
        return

    def get_probe(runtime, run_id, *, automatic=False):
        directory = probe._diagnostic_directory(automatic=automatic)
        if directory is None:
            return None
        run = runtime._run
        item = getattr(runtime, "_bsa_transition_probe", None)
        if item is not None and item[0] is not run:
            item[1].close()
            item = None
        if item is None:
            version, blob, signature = _source_provenance()
            current = probe.Probe(directory, run_id, runtime.config)
            current.write(
                "source",
                comfy_kitchen=version,
                cuda_wrapper_git_blob=blob,
                sol_attn_chunked_signature=signature,
                compatibility_basis="runtime_api_contract_not_exact_build_pin",
            )
            runtime._bsa_transition_probe = (run, current)
            return current
        return item[1]

    get_probe._spectrum_compatible_kitchen = True
    probe.get_probe = get_probe
