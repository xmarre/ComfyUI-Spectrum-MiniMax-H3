"""Bounded matched-input BSA calibration and hidden-hold diagnostics.

On this diagnostic PR, audited-safe core-BSA CUDA runs enable the probe
automatically and write reports below ComfyUI's normal output directory. The
SPECTRUM_H3_BSA_DIAGNOSTICS environment variable remains only as an optional
override/disable switch; no shell setup is required for the production handoff.

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
import sys
import threading
import uuid

import torch

LOG = logging.getLogger(__name__)
_ACTIVE = ContextVar("spectrum_bsa_transition_probe", default=None)
_LOCK = threading.Lock()
_SESSION = uuid.uuid4().hex[:12]
ENV = "SPECTRUM_H3_BSA_DIAGNOSTICS"
AUTO_OUTPUT_SUBDIR = "spectrum_h3_bsa_diagnostics"
_DISABLED = frozenset({"", "0", "false", "off", "no", "disable", "disabled"})


def _diagnostic_directory(*, automatic):
    """Resolve an explicit override, or auto-enable for an audited-safe CUDA run."""
    configured = os.environ.get(ENV)
    if configured is not None:
        value = configured.strip()
        if value.lower() in _DISABLED:
            return None
        return Path(value).expanduser()
    if not automatic or not torch.cuda.is_available():
        return None
    try:
        import folder_paths
        output_root = Path(folder_paths.get_output_directory())
    except (ImportError, AttributeError, TypeError, RuntimeError):
        output_root = Path.cwd() / "output"
    return output_root.expanduser() / AUTO_OUTPUT_SUBDIR


def delta(reference, candidate):
    """Chunked FP32 vector math with FP64 accumulation of chunk reductions."""
    if reference.shape != candidate.shape:
        raise ValueError("diagnostic tensor shapes differ")
    a, b = reference.detach().reshape(-1), candidate.detach().reshape(-1)
    n = a.numel()
    if n == 0:
        return {"finite": True, "elements": 0, "rmse": 0.0, "relative_l2": None,
                "reference_rms": 0.0, "mae": 0.0, "max_abs": 0.0, "cosine": None}
    sums = torch.zeros(5, dtype=torch.float64, device=a.device)
    maximum = torch.zeros((), dtype=torch.float32, device=a.device)
    finite = torch.ones((), dtype=torch.bool, device=a.device)
    for start in range(0, n, 262144):
        x = a[start:start + 262144].to(dtype=torch.float32)
        y = b[start:start + 262144].to(device=a.device, dtype=torch.float32)
        finite &= torch.isfinite(x).all() & torch.isfinite(y).all()
        diff = y - x
        chunk = torch.stack((diff.square().sum(), x.square().sum(),
                             y.square().sum(), (x * y).sum(), diff.abs().sum()))
        sums += chunk.to(dtype=torch.float64)
        maximum = torch.maximum(maximum, diff.abs().max())
    if not bool(finite.item()):
        return {"finite": False, "elements": n}
    err, ref, cand, dot, abs_sum = sums.tolist()
    return {"finite": True, "elements": n, "rmse": (err / n) ** 0.5,
            "relative_l2": (err / ref) ** 0.5 if ref else None,
            "reference_rms": (ref / n) ** 0.5,
            "mae": abs_sum / n, "max_abs": maximum.item(),
            "cosine": dot / (ref * cand) ** 0.5 if ref and cand else None}


def _tensor_state(tensor):
    value = tensor.detach().to(dtype=torch.float32)
    stats = torch.stack((value.min(), value.max(), value.mean(), value.square().mean().sqrt()))
    minimum, maximum, mean, rms = stats.tolist()
    return {"shape": list(value.shape), "dtype": str(tensor.dtype),
            "min": minimum, "max": maximum, "mean": mean, "rms": rms}


def _calibration_state(values):
    return {"kmean": _tensor_state(values[0]), "vscale": _tensor_state(values[1])}


def _same_bytes(tensor, saved):
    return torch.equal(
        tensor.contiguous().view(torch.uint8),
        saved.contiguous().view(torch.uint8),
    )


def _pool_entry(patch, key):
    pooled = getattr(patch, "pooled", None)
    if not isinstance(pooled, dict):
        return None
    entry = pooled.get(key)
    if (not isinstance(entry, (tuple, list)) or len(entry) != 2
            or not all(torch.is_tensor(value) and value.dtype == torch.float32
                       and value.ndim == 2 for value in entry)):
        return None
    return entry


def _capture_cold_successor(audit, routes):
    """Retain exact post-cold pool owners and bytes for the adjacent transition proof."""
    state = {}
    for index, route in enumerate(routes):
        if route != "h3_chunked_sparse_cold":
            continue
        key = (index, audit.seq_len, audit.uuids)
        entry = _pool_entry(audit.patch, key)
        if entry is None:
            raise RuntimeError(
                f"BSA diagnostic lost accepted cold pool state for block {index}"
            )
        state[index] = (
            entry[0],
            entry[1],
            entry[0].detach().clone(),
            entry[1].detach().clone(),
        )
    return state


def _verify_cold_successor(audit, sparse, predecessor):
    """Prove current primed state is exactly the previous accepted cold output."""
    for index in sparse:
        expected = predecessor.get(index)
        if expected is None:
            return False, index, "predecessor_snapshot_missing"
        key = (index, audit.seq_len, audit.uuids)
        entry = _pool_entry(audit.patch, key)
        if entry is None:
            return False, index, "current_pool_missing_or_malformed"
        if entry[0] is not expected[0] or entry[1] is not expected[1]:
            return False, index, "pool_tensor_owner_changed"
        if not _same_bytes(entry[0], expected[2]) or not _same_bytes(entry[1], expected[3]):
            return False, index, "pool_tensor_contents_changed"
    return True, None, None


class PoolSnapshot:
    """Preserve dict, entry and tensor identities as well as tensor contents."""
    def __init__(self, patch):
        if not isinstance(patch.pooled, dict):
            raise TypeError("BSA pool is not a dictionary")
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
                if not _same_bytes(tensor, saved):
                    raise RuntimeError("BSA diagnostic failed to restore pool contents")


class IsolationStatus:
    def __init__(self):
        self.pool_restored = False
        self.rng_restored = False


@contextmanager
def isolated_pool(patch, device):
    """Restore after success, Python exceptions and OOM without masking the primary failure."""
    saved = PoolSnapshot(patch)
    status = IsolationStatus()
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state().clone()
    cuda_device = None
    cuda_rng = None
    if device.type == "cuda":
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
        cuda_rng = torch.cuda.get_rng_state(cuda_device).clone()
    devices = [] if cuda_device is None else [cuda_device]
    try:
        with torch.random.fork_rng(devices=devices):
            yield status
    finally:
        primary = sys.exception()
        restoration_errors = []
        try:
            random.setstate(python_rng)
        except (TypeError, ValueError) as exc:  # pragma: no cover - stdlib state restore is deterministic
            restoration_errors.append(f"Python RNG restore failed: {exc}")
        try:
            saved.restore()
            saved.verify()
            status.pool_restored = True
        except (RuntimeError, TypeError, ValueError) as exc:
            restoration_errors.append(f"pool restore failed: {exc}")
        try:
            cpu_ok = torch.equal(torch.get_rng_state(), cpu_rng)
            cuda_ok = True if cuda_rng is None else torch.equal(
                torch.cuda.get_rng_state(cuda_device), cuda_rng
            )
            python_ok = random.getstate() == python_rng
            status.rng_restored = bool(cpu_ok and cuda_ok and python_ok)
            if not status.rng_restored:
                restoration_errors.append("RNG state did not restore exactly")
        except (RuntimeError, TypeError, ValueError) as exc:
            restoration_errors.append(f"RNG verification failed: {exc}")
        if restoration_errors:
            message = "BSA diagnostic restoration failure: " + "; ".join(restoration_errors)
            if primary is not None:
                try:
                    primary.add_note(message)
                except AttributeError:  # pragma: no cover - Python 3.11+ has add_note
                    pass
            else:
                raise RuntimeError(message)


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
        self.write("start", schema_version=3, torch_version=torch.__version__,
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


def get_probe(runtime, run_id, *, automatic=False):
    directory = _diagnostic_directory(automatic=automatic)
    if directory is None:
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
        self.selected = {sparse[0], sparse[len(sparse)//2], sparse[-1]} if sparse else set()
        self.routes = tuple(spec[0] for spec in audit.route_specs)
        prev = probe.previous
        transition_shape = bool(
            not probe.examined
            and prev
            and prev[0] == step_id - 1
            and prev[1] == self.identity
            and sparse
            and all(
                self.routes[i] == "h3_chunked_sparse_primed"
                and prev[2][i] == "h3_chunked_sparse_cold"
                for i in sparse
            )
        )
        self.predecessor_pool_continuity = None
        self.predecessor_pool_mismatch_block = None
        self.predecessor_pool_mismatch_reason = None
        if transition_shape:
            continuity, block, reason = _verify_cold_successor(audit, sparse, prev[3])
            self.predecessor_pool_continuity = continuity
            self.predecessor_pool_mismatch_block = block
            self.predecessor_pool_mismatch_reason = reason
            if not continuity:
                probe.write(
                    "skipped",
                    step=step_id,
                    reason="cold_to_primed_pool_successor_discontinuity",
                    block=block,
                    mismatch=reason,
                    identity=_identity_digest(self.identity),
                )
        self.candidate = bool(transition_shape and self.predecessor_pool_continuity)
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
                alternative_mode = "cold_current_input_two_pass"
            else:
                stored = self.probe.pending["stale"].get(index)
                if stored is None or any(a is not b for a, b in zip(stored[0], entry)):
                    raise RuntimeError("BSA diagnostic pool ownership changed before next actual")
                replacement = stored[1]
                label = "skipped_refresh_next_actual"
                alternative_mode = "primed_with_stale_pre_skipped_step_calibration"
            self.probe.started_attention_calls += 1
            if self.probe.started_attention_calls > 6:
                raise RuntimeError("BSA diagnostic attention-call budget exceeded")
            self.probe.write("attention_begin", step=self.step_id, block=index, measurement=label,
                             route=self.routes[index])
            actual_nfe_before = self.runtime.stats.actual_transformer_calls
            forecast_calls_before = self.runtime.stats.forecast_model_calls
            with isolated_pool(patch, h.device) as isolation:
                if replacement is None:
                    del patch.pooled[key]
                else:
                    for tensor, saved in zip(patch.pooled[key], replacement):
                        tensor.copy_(saved)
                alternative = function(h, *args, **kwargs)
            if (self.runtime.stats.actual_transformer_calls != actual_nfe_before
                    or self.runtime.stats.forecast_model_calls != forecast_calls_before):
                raise RuntimeError("BSA diagnostic attention repeat changed transformer/forecast counters")
            measurement = delta(actual, alternative)
            self.probe.extra_attention_calls += 1
            growth = fresh[1] / old[1].clamp_min(1e-8)
            stale_growth = None if replacement is None else (
                fresh[1] / replacement[1].clamp_min(1e-8)
            )
            headroom_count = None if stale_growth is None else int((stale_growth > 1.1).sum().item())
            self.measurements.append({
                "measurement": label,
                "block": index,
                "route": self.routes[index],
                "scope": "attention_output_projection_at_fixed_actual_block_input",
                "error": measurement,
                "control_output_source": "unmodified_primary_attention_call",
                "alternative_output_discarded": True,
                "alternative_calibration_mode": alternative_mode,
                "control_pool_entry_identity": id(entry),
                "control_kmean_tensor_identity": id(entry[0]),
                "control_vscale_tensor_identity": id(entry[1]),
                "control_supplied_calibration": _calibration_state(old),
                "alternative_supplied_calibration": (
                    None if replacement is None else _calibration_state(replacement)
                ),
                "next_calibration": _calibration_state(fresh),
                "pool_restored": isolation.pool_restored,
                "rng_restored": isolation.rng_restored,
                "diagnostic_transformer_nfe_delta": (
                    self.runtime.stats.actual_transformer_calls - actual_nfe_before
                ),
                "diagnostic_forecast_counter_delta": (
                    self.runtime.stats.forecast_model_calls - forecast_calls_before
                ),
                "kmean_change": delta(old[0], fresh[0]),
                "vscale_change": delta(old[1], fresh[1]),
                "vscale_range_growth_max": growth.max().item(),
                "alternative_scale_headroom_exceeded_channels": headroom_count,
                "alternative_scale_channels": (
                    None if stale_growth is None else stale_growth.numel()
                ),
                "alternative_scale_growth_max": (
                    None if stale_growth is None else stale_growth.max().item()
                ),
            })
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
        successor_state = _capture_cold_successor(self.audit, self.routes) if accepted else {}
        p.write("actual", step=self.step_id, call=self.call_id, accepted=bool(accepted),
                receipts_expected=self.audit.block_count,
                seq_len=self.audit.seq_len, flow_mixed=bool(getattr(self.audit, "flow_mixed", False)),
                identity=_identity_digest(self.identity), candidate=self.candidate,
                predecessor_pool_continuity=self.predecessor_pool_continuity,
                predecessor_pool_mismatch_block=self.predecessor_pool_mismatch_block,
                predecessor_pool_mismatch_reason=self.predecessor_pool_mismatch_reason,
                captured_cold_successor_blocks=len(successor_state),
                next_actual=self.next_actual, core_bsa_source_blob=self.audit.source_blob,
                patch_generation=self.audit.patch_generation,
                routes=list(self.routes), measurements=self.measurements)
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
                    policy_step=self.runtime._step.policy_step_id,
                    candidate_backend_transition="h3_chunked_sparse_cold->h3_chunked_sparse_primed",
                    predecessor_pool_continuity=True,
                    diagnostic_transformer_nfe=0)
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
        p.previous = (self.step_id, self.identity, self.routes, successor_state)


@contextmanager
def actual_scope(runtime, run_id, step_id, call_id, options):
    from .core_bsa_compat import PRIVATE_AUDIT_KEY
    audit = options.get(PRIVATE_AUDIT_KEY)
    probe = get_probe(runtime, run_id, automatic=bool(audit is not None and audit.safe))
    if probe is None:
        yield None
        return
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
