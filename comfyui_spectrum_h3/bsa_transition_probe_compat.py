"""Compatibility and lifecycle helpers for the BSA transition diagnostic.

The diagnostic accepts source-built/current comfy-kitchen as long as the public
``sol_attn_chunked`` calibration contract is present.  It also groups all Spectrum
runtime segments executed by one ComfyUI prompt into one JSONL report, because one
video generation can legitimately contain several low/high sampler invocations.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import threading
import uuid


_SINK_LOCK = threading.Lock()
_SINKS = {}


def _source_provenance():
    """Return installed comfy-kitchen provenance after checking the API contract."""
    version = importlib.metadata.version("comfy-kitchen")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    function = getattr(cuda, "sol_attn_chunked", None)
    if not callable(function):
        raise TypeError(
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


def _execution_generation_id():
    """Use ComfyUI's prompt id as the generation boundary when available."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except (ImportError, AttributeError, RuntimeError):
        return None
    prompt_id = getattr(context, "prompt_id", None)
    return None if prompt_id is None else str(prompt_id)


def _generation_digest(generation_id):
    return hashlib.sha256(generation_id.encode()).hexdigest()[:12]


class _GenerationSink:
    def __init__(self, directory, generation_id, *, persistent, session):
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.generation_id = generation_id
        self.generation = _generation_digest(generation_id)
        self.persistent = persistent
        self.path = directory / f"bsa-{session}-generation-{self.generation}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.active = 0
        self.closed = False

    def write(self, payload):
        with self.lock:
            if self.closed:
                raise RuntimeError("BSA diagnostic generation report is already closed")
            self.file.write(json.dumps(payload, allow_nan=False) + "\n")
            self.file.flush()

    def close(self):
        with self.lock:
            if not self.closed:
                self.file.close()
                self.closed = True


def _acquire_sink(directory, generation_id, *, session):
    persistent = generation_id is not None
    if generation_id is None:
        generation_id = f"standalone-{uuid.uuid4().hex}"
    key = (str(Path(directory).expanduser().resolve()), generation_id, session)
    with _SINK_LOCK:
        for old_key, old in list(_SINKS.items()):
            if old_key != key and old.active == 0:
                old.close()
                del _SINKS[old_key]
        sink = _SINKS.get(key)
        created = sink is None or sink.closed
        if created:
            sink = _GenerationSink(
                directory,
                generation_id,
                persistent=persistent,
                session=session,
            )
            _SINKS[key] = sink
        sink.active += 1
    return key, sink, created


def _release_sink(key, sink):
    with _SINK_LOCK:
        sink.active = max(0, sink.active - 1)
        if sink.active == 0 and not sink.persistent:
            sink.close()
            _SINKS.pop(key, None)


def _close_sinks():
    with _SINK_LOCK:
        for sink in _SINKS.values():
            sink.close()
        _SINKS.clear()


atexit.register(_close_sinks)


def install_bsa_transition_probe_compat():
    """Install source-build compatibility and generation-scoped report aggregation."""
    from . import bsa_transition_probe as probe

    if getattr(probe.get_probe, "_spectrum_compatible_kitchen", False):
        return

    base_probe = probe.Probe
    base_actual = probe.Actual

    class GenerationProbe(base_probe):
        """One run segment writing into a prompt-scoped generation report."""

        def __init__(self, directory, run_id, config):
            generation_id = _execution_generation_id()
            self._sink_key, self._sink, created = _acquire_sink(
                directory,
                generation_id,
                session=probe._SESSION,
            )
            self.path = self._sink.path
            self.file = self._sink.file
            self.generation = self._sink.generation
            self.run_instance = uuid.uuid4().hex[:8]
            self.run_id = run_id
            self.config = config
            self.previous = None
            self.anchor = None
            self.pending = None
            self.examined = False
            self.extra_attention_calls = 0
            self.started_attention_calls = 0
            self.errors = 0
            self.calls = 0
            self._closed = False
            self.write(
                "start",
                schema_version=4,
                torch_version=probe.torch.__version__,
                scope="matched-input sparse attention; full target-hidden one-point hold",
                diagnostic_transformer_nfe=0,
                max_attention_calls_per_run=6,
                production_policy="unchanged",
                source_revision=probe._revision(),
                report_scope="one_jsonl_per_comfyui_prompt_generation",
            )
            if created:
                probe.LOG.warning("Spectrum H3 BSA diagnostic report: %s", self.path)

        def write(self, event, **data):
            self._sink.write({
                "event": event,
                "generation": self.generation,
                "run_instance": self.run_instance,
                "run_id": self.run_id,
                **data,
            })

        def close(self):
            if self._closed:
                return
            self.write(
                "end",
                tracked_actual_calls=self.calls,
                diagnostic_attention_calls_started=self.started_attention_calls,
                diagnostic_attention_calls_completed=self.extra_attention_calls,
                diagnostic_transformer_nfe=0,
                errors=self.errors,
                pending_next_actual=self.pending is not None,
            )
            self.anchor = self.pending = self.previous = None
            self._closed = True
            _release_sink(self._sink_key, self._sink)

    class GenerationActual(base_actual):
        """Permit candidate-C measurement across a settings-only identity change.

        The shadow still executes the current actual step's own attention closure.
        The only substituted state is the stale calibration from the immediately
        preceding candidate, and exact tensor ownership is proved before enabling
        the measurement.  Sequence/pool discontinuities therefore remain rejected.
        """

        def __init__(self, diagnostic, runtime, step_id, call_id, audit):
            super().__init__(diagnostic, runtime, step_id, call_id, audit)
            self.next_actual_identity_changed = False
            pending = diagnostic.pending
            if self.next_actual or pending is None or step_id != pending["step"] + 1:
                return
            if not self.selected or any(
                self.routes[index] != "h3_chunked_sparse_primed" for index in self.selected
            ):
                return
            for index in self.selected:
                stored = pending["stale"].get(index)
                key = (index, audit.seq_len, audit.uuids)
                entry = probe._pool_entry(audit.patch, key)
                if (
                    stored is None
                    or entry is None
                    or any(owner is not current for owner, current in zip(stored[0], entry))
                ):
                    return
            self.next_actual = True
            self.next_actual_identity_changed = pending["identity"] != self.identity

        def complete(self, accepted):
            if self.next_actual:
                self.probe.write(
                    "next_actual_scope",
                    step=self.step_id,
                    semantic_identity_changed=self.next_actual_identity_changed,
                    safety_basis=(
                        "adjacent call + same calibration tensor owners + current-step attention closure"
                    ),
                )
            return super().complete(accepted)

    probe.Probe = GenerationProbe
    probe.Actual = GenerationActual

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
