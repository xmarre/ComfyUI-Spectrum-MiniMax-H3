"""Opt-in, bounded matched-input BSA calibration and hidden-hold diagnostics.

Attention comparisons repeat only the reviewed sparse attention closure. They do
not replay a transformer or simulate the downstream trajectory of a skipped NFE.
The unmodified actual output and pool update remain the control trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path
import random
import threading
import uuid

import torch

LOG = logging.getLogger(__name__)
_ACTIVE = ContextVar("spectrum_bsa_transition_probe", default=None)
_LOCK = threading.Lock()
_SESSION = uuid.uuid4().hex[:12]
ENV = "SPECTRUM_H3_BSA_DIAGNOSTICS"


def delta(reference, candidate):
    """Chunked FP64 reductions; no full FP32 duplicate of the hidden tensor."""
    if reference.shape != candidate.shape:
        raise ValueError("diagnostic tensor shapes differ")
    a, b = reference.detach().reshape(-1), candidate.detach().reshape(-1)
    sums = torch.zeros(5, dtype=torch.float64, device=a.device)
    maximum = torch.zeros((), dtype=torch.float64, device=a.device)
    for start in range(0, a.numel(), 262144):
        x = a[start:start + 262144].to(dtype=torch.float64)
        y = b[start:start + 262144].to(device=a.device, dtype=torch.float64)
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            return {"finite": False, "elements": a.numel()}
        diff = y - x
        sums += torch.stack((diff.square().sum(), x.square().sum(),
                             y.square().sum(), (x * y).sum(), diff.abs().sum()))
        maximum = torch.maximum(maximum, diff.abs().max())
    err, ref, cand, dot, abs_sum = sums.tolist()
    n = a.numel()
    return {"finite": True, "elements": n, "rmse": (err / max(n, 1)) ** 0.5,
            "relative_l2": (err / ref) ** 0.5 if ref else None,
            "reference_rms": (ref / max(n, 1)) ** 0.5,
            "mae": abs_sum / max(n, 1), "max_abs": maximum.item(),
            "cosine": dot / (ref * cand) ** 0.5 if ref and cand else None}


class PoolSnapshot:
    """Preserve dict, entry and tensor identities as well as tensor contents."""
    def __init__(self, patch):
        if not isinstance(patch.pooled, dict):
            raise ValueError("BSA pool is not a dictionary")
        self.patch, self.mapping = patch, patch.pooled
        self.entries = dict(patch.pooled)
        self.values = {}
        for key, entry in self.entries.items():
            if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                    or not all(torch.is_tensor(t) and t.dtype == torch.float32
                               and t.ndim == 2 for t in entry)):
                raise ValueError("BSA pool entry is malformed")
            self.values[key] = tuple(t.detach().clone() for t in entry)

    def restore(self):
        with torch.no_grad():
            for key, entry in self.entries.items():
                for tensor, saved in zip(entry, self.values[key]):
                    tensor.copy_(saved)
            self.mapping.clear()
            self.mapping.update(self.entries)
            self.patch.pooled = self.mapping

    def verify(self):
        if self.patch.pooled is not self.mapping or self.mapping.keys() != self.entries.keys():
            raise RuntimeError("BSA diagnostic failed to restore pool dictionary")
        for key, entry in self.entries.items():
            if self.mapping[key] is not entry:
                raise RuntimeError("BSA diagnostic failed to restore pool entry identity")
            for tensor, saved in zip(entry, self.values[key]):
                # Byte comparison also verifies NaN payloads, without _version.
                if not torch.equal(tensor.contiguous().view(torch.uint8),
                                   saved.contiguous().view(torch.uint8)):
                    raise RuntimeError("BSA diagnostic failed to restore pool contents")


@contextmanager
def isolated_pool(patch, device):
    """Restore after success, Python exceptions and OOM; never swallow failures."""
    saved = PoolSnapshot(patch)
    python_rng = random.getstate()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_rng)
        saved.restore()
        saved.verify()


def _semantic_identity(value):
    # Diagnostics only. This never supplies a production backend identity.
    if isinstance(value, tuple):
        return tuple(_semantic_identity(x) for x in value
                     if not (isinstance(x, tuple) and x
                             and x[0] in ("routes", "pool_ownership")))
    return value


def _identity_digest(identity):
    return hashlib.sha256(repr(identity).encode()).hexdigest()


class Probe:
    def __init__(self, directory, run_id, config):
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"bsa-{_SESSION}-{uuid.uuid4().hex[:8]}-run{run_id}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
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
        self.write("start", schema_version=1, torch_version=torch.__version__,
                   scope="matched-input sparse attention; full target-hidden one-point hold",
                   diagnostic_transformer_nfe=0, max_attention_calls_per_run=6,
                   production_policy="unchanged", source_revision=_revision())
        LOG.warning("Spectrum H3 BSA diagnostic report: %s", self.path)

    def write(self, event, **data):
        self.file.write(json.dumps({"event": event, "run_id": self.run_id, **data},
                                   allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        if self.file.closed:
            return
        self.write("end", tracked_actual_calls=self.calls,
                   diagnostic_attention_calls_started=self.started_attention_calls,
                   diagnostic_attention_calls_completed=self.extra_attention_calls,
                   diagnostic_transformer_nfe=0, errors=self.errors,
                   pending_next_actual=self.pending is not None)
        self.anchor = self.pending = self.previous = None
        self.file.close()


def _revision():
    # Hash installed diagnostic source, without invoking git in the sampling path.
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def get_probe(runtime, run_id):
    directory = os.environ.get(ENV)
    if not directory:
        return None
    run = runtime._run
    item = getattr(runtime, "_bsa_transition_probe", None)
    if item is not None and item[0] is not run:
        item[1].close()
        item = None
    if item is None:
        version = importlib.metadata.version("comfy-kitchen")
        spec = importlib.util.find_spec("comfy_kitchen.backends.cuda")
        source = Path(spec.origin).read_bytes() if spec and spec.origin else b""
        blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
        if version != "0.2.33" or blob != "15fcab4a9d166fae67db6ca4f3410274ad6840d9":
            raise RuntimeError("BSA diagnostics require reviewed comfy-kitchen 0.2.33 CUDA wrapper")
        probe = Probe(directory, run_id, runtime.config)
        probe.write("source", comfy_kitchen=version, cuda_wrapper_git_blob=blob)
        runtime._bsa_transition_probe = (run, probe)
        return probe
    return item[1]


def finish(runtime):
    item = getattr(runtime, "_bsa_transition_probe", None)
    if item is not None:
        item[1].close()
        runtime._bsa_transition_probe = None


class Actual:
    def __init__(self, probe, runtime, step_id, call_id, audit):
        self.probe, self.runtime, self.audit = probe, runtime, audit
        self.step_id, self.call_id = step_id, call_id
        self.identity = _semantic_identity(audit.identity)
        sparse = [i for i, spec in enumerate(audit.route_specs) if spec[0] != "h3_dense"]
        self.selected = set((sparse[0], sparse[len(sparse)//2], sparse[-1])) if sparse else set()
        self.routes = tuple(spec[0] for spec in audit.route_specs)
        prev = probe.previous
        self.candidate = bool(not probe.examined and prev and prev[0] == step_id - 1
                              and prev[1] == self.identity and sparse
                              and all(self.routes[i] == "h3_chunked_sparse_primed"
                                      and prev[2][i] == "h3_chunked_sparse_cold" for i in sparse))
        self.next_actual = bool(probe.pending and probe.pending["identity"] == self.identity
                                and step_id == probe.pending["step"] + 1)
        self.stale = {}
        self.hidden = None
        self.rows = None
        self.measurements = []

    def wrap_attention(self, function, index):
        if index not in self.selected or not (self.candidate or self.next_actual):
            return function

        def measured(h, *args, **kwargs):
            key = (index, self.audit.seq_len, self.audit.uuids)
            patch = self.audit.patch
            entry = patch.pooled.get(key)
            if entry is None:
                raise RuntimeError("BSA diagnostic expected primed pool")
            old = tuple(t.detach().clone() for t in entry)
            # Only this call contributes to the control trajectory and receipts.
            actual = function(h, *args, **kwargs)
            fresh = tuple(t.detach().clone() for t in patch.pooled[key])
            if self.candidate:
                self.stale[index] = (entry, old)
                replacement = None
                label = "cold_vs_primed"
            else:
                stored = self.probe.pending["stale"].get(index)
                if stored is None or any(a is not b for a, b in zip(stored[0], entry)):
                    raise RuntimeError("BSA diagnostic pool ownership changed before next actual")
                replacement = stored[1]
                label = "skipped_refresh_next_actual"
            self.probe.started_attention_calls += 1
            if self.probe.started_attention_calls > 6:
                raise RuntimeError("BSA diagnostic attention-call budget exceeded")
            self.probe.write("attention_begin", step=self.step_id, block=index, measurement=label)
            with isolated_pool(patch, h.device):
                if replacement is None:
                    del patch.pooled[key]
                else:
                    for tensor, saved in zip(patch.pooled[key], replacement):
                        tensor.copy_(saved)
                alternative = function(h, *args, **kwargs)
                measurement = delta(actual, alternative)
            self.probe.extra_attention_calls += 1
            self.measurements.append({"measurement": label, "block": index,
                                      "scope": "attention_output_projection_at_fixed_actual_block_input",
                                      "error": measurement, "pool_restored": True,
                                      "kmean_change": delta(old[0], fresh[0]),
                                      "vscale_change": delta(old[1], fresh[1]),
                                      "vscale_range_growth_max": (fresh[1] / old[1].clamp_min(1e-8)).max().item(),
                                      "alternative_scale_headroom_exceeded_channels": (
                                          int((fresh[1] / replacement[1].clamp_min(1e-8) > 1.1).sum())
                                          if replacement is not None else None)})
            return actual
        return measured

    def capture_hidden(self, target, audio_rows):
        self.rows = audio_rows
        # Retain only a cold anchor, or the candidate target used for comparison.
        if self.candidate or "h3_chunked_sparse_cold" in self.routes:
            self.hidden = target.detach().to(device="cpu", copy=True)

    def complete(self, accepted):
        p = self.probe
        p.calls += 1
        p.write("actual", step=self.step_id, call=self.call_id, accepted=bool(accepted),
                receipts_expected=self.audit.block_count,
                seq_len=self.audit.seq_len, flow_mixed=bool(getattr(self.audit, "flow_mixed", False)),
                identity=_identity_digest(self.identity), candidate=self.candidate,
                next_actual=self.next_actual, measurements=self.measurements)
        if not accepted:
            raise RuntimeError("BSA diagnostic actual audit rejected; report is incomplete")
        if self.candidate:
            if p.anchor is None or self.hidden is None or p.anchor.shape != self.hidden.shape:
                raise RuntimeError("BSA diagnostic missing compatible hidden anchor")
            run = self.runtime._run
            hold = p.anchor
            # Exactly the values produced by predict_one_point_hold. Do not call
            # runtime.predict: it consumes production history rows and counters.
            p.write("forecast_vs_actual", step=self.step_id,
                    forecast="one_point_hold_candidate_not_executed",
                    scope="complete_target_hidden_audio_and_video",
                    audio=delta(self.hidden[:, :self.rows], hold[:, :self.rows]),
                    video=delta(self.hidden[:, self.rows:], hold[:, self.rows:]),
                    previous_to_current_actual=delta(hold, self.hidden),
                    prior_committed_forecasts=self.runtime.stats.forecast_model_calls,
                    bootstrap_config=bool(self.runtime.config.bootstrap_first_forecast),
                    degree=self.runtime.config.degree,
                    continuum_prefix=run.min_actual_prefix_steps,
                    state_conditioned_residual=bool(run.state_conditioned_residual),
                    policy_step=self.runtime._step.policy_step_id)
            p.pending = {"identity": self.identity, "step": self.step_id, "stale": self.stale}
            p.examined = True
            p.anchor = None
        elif self.next_actual:
            p.pending = None
        elif p.pending is not None:
            p.write("skipped", reason="next actual was not the adjacent compatible call")
            p.pending = None
        if "h3_chunked_sparse_cold" in self.routes and not p.examined:
            p.anchor = self.hidden
        p.previous = (self.step_id, self.identity, self.routes)


@contextmanager
def actual_scope(runtime, run_id, step_id, call_id, options):
    probe = get_probe(runtime, run_id)
    if probe is None:
        yield None
        return
    from .core_bsa_compat import PRIVATE_AUDIT_KEY
    audit = options.get(PRIVATE_AUDIT_KEY)
    if audit is None or not audit.safe:
        probe.write("skipped", step=step_id, reason="audited safe core BSA required")
        probe.anchor = probe.previous = probe.pending = None
        yield None
        return
    if (runtime.active_state_conditioned_residual or runtime.offline_phase is not None
            or runtime._run.separate_stage_histories or call_id != 0):
        raise RuntimeError("BSA diagnostics require single-pass absolute-hidden single-call stages")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("concurrent BSA diagnostic calls are unsupported")
    token = None
    try:
        current = Actual(probe, runtime, step_id, call_id, audit)
        token = _ACTIVE.set(current)
        yield current
    except BaseException as exc:
        probe.errors += 1
        probe.write("error", step=step_id, type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if token is not None:
            _ACTIVE.reset(token)
        _LOCK.release()


def attention(function, audit, index):
    current = _ACTIVE.get()
    if current is None or current.audit is not audit:
        return function
    return current.wrap_attention(function, index)


def hidden(target, audio_rows):
    current = _ACTIVE.get()
    if current is not None:
        current.capture_hidden(target, audio_rows)
