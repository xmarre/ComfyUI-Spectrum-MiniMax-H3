"""Opt-in native RES trajectory tracing and same-state actual intervention.

The intervention commits an actual evaluation through the normal retry path.
It is not a shadow oracle: the exact hidden feature becomes a causal anchor and,
in offline mode, a capture anchor. No sampler equations or production policy change.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
import re
import time
import uuid

import torch

from .bsa_transition_probe import delta

LOG = logging.getLogger(__name__)
SETTINGS_KEY = "spectrum_h3_res_investigation"


def res_coefficients(sigmas):
    """Evaluate the native deterministic RES expressions in schedule dtype.

    These are denoised-value coefficients, not a bound on nonlinear trajectory
    error. The zero endpoint uses native Euler arithmetic, not log(0).
    """
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("RES investigation requires a one-dimensional sigma schedule")
    s = sigmas.detach().cpu()
    if (
        not s.dtype.is_floating_point
        or not bool(torch.isfinite(s).all())
        or not bool((s[:-1] > 0).all())
        or s[-1] < 0
        or not bool((s[:-1] > s[1:]).all())
    ):
        raise ValueError(
            "RES investigation requires finite, strictly decreasing positive evaluated sigmas"
        )
    rows = []
    for i in range(len(s) - 1):
        if i == 0 or s[i + 1] == 0:
            current = float(((s[i] - s[i + 1]) / s[i]).item())
            previous = 0.0
            decay = float((s[i + 1] / s[i]).item())
            h = None
            branch = "euler"
        else:
            t, t_old = -s[i].log(), -s[i].log()
            t_next, t_prev = -s[i + 1].log(), -s[i - 1].log()
            h_tensor = t_next - t
            c2 = (t_prev - t_old) / h_tensor
            phi1 = torch.expm1(-h_tensor) / -h_tensor
            phi2 = (phi1 - 1.0) / -h_tensor
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            current = float((h_tensor * b1).item())
            previous = float((h_tensor * b2).item())
            decay = float(h_tensor.neg().exp().item())
            h = float(h_tensor.item())
            branch = "multistep"
        if not all(math.isfinite(v) for v in (current, previous, decay)):
            raise ValueError(
                "RES investigation encountered nonfinite native coefficients"
            )
        rows.append(
            {
                "step": i,
                "sigma": float(s[i]),
                "next_sigma": float(s[i + 1]),
                "branch": branch,
                "h": h,
                "state_coefficient": decay,
                "current_denoised_coefficient": current,
                "previous_denoised_coefficient": previous,
            }
        )
    return rows


def _streams(value, latent_shapes):
    if value.is_nested:
        pieces = value.unbind()
        return tuple((f"stream_{i}", part) for i, part in enumerate(pieces))
    if (
        latent_shapes
        and len(latent_shapes) == 2
        and value.ndim == 3
        and value.shape[1] == 1
    ):
        sizes = [math.prod(shape[1:]) for shape in latent_shapes]
        if sum(sizes) == value.shape[-1]:
            # ComfyUI H3 packs video followed by audio (distinct from hidden-token order).
            return (
                ("video", value[..., : sizes[0]]),
                ("audio", value[..., sizes[0] :]),
            )
    return (("packed", value),)


def _summary(value):
    flat = value.detach().reshape(-1)
    count = flat.numel()
    if not count:
        return {"shape": list(value.shape), "finite": True, "rms": 0.0, "mean": 0.0}
    totals = torch.zeros(2, device=flat.device, dtype=torch.float64)
    finite = torch.ones((), device=flat.device, dtype=torch.bool)
    for start in range(0, count, 262144):
        part = flat[start : start + 262144].float()
        finite &= torch.isfinite(part).all()
        totals += torch.stack((part.sum(), part.square().sum())).double()
    is_finite = bool(finite.item())
    mean, square = totals.cpu().tolist()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": is_finite,
        "sha256": hashlib.sha256(
            value.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest(),
        "mean": mean / count if is_finite else None,
        "rms": (square / count) ** 0.5 if is_finite else None,
    }


class RESInvestigation:
    def __init__(
        self, settings, runtime, sigmas, sampler, phase, seed, latent_shapes, masked
    ):
        self.settings = dict(settings)
        self.phase = phase
        self.latent_shapes = latent_shapes
        self.coefficients = res_coefficients(sigmas)
        self.checkpoints = {}
        self.rows = []
        self.started = time.perf_counter()
        self.replacements = 0
        self.discarded_forecast_model_calls = 0
        self.selected_status = "not_reached"
        self.pending_error = None
        self.selected_input = None
        source = inspect.getsource(sampler.sampler_function)
        import comfy.k_diffusion.sampling as native

        core_source = inspect.getsource(native.res_multistep)
        self.report = {
            "schema": "spectrum_h3_res_investigation_v1",
            "settings": self.settings,
            "phase": phase,
            "seed": seed,
            "config": asdict(runtime.config),
            "sampler": sampler.sampler_function.__name__,
            "native_source_sha256": hashlib.sha256(
                (source + core_source).encode()
            ).hexdigest(),
            "sigmas": sigmas.detach().cpu().tolist(),
            "sigma_dtype": str(sigmas.dtype),
            "latent_shapes": [list(shape) for shape in latent_shapes or ()],
            "masked": masked,
            "measurement_space": "CFGGuider.predict_noise output before KSampler inpaint output masking",
            "coefficient_scope": "native deterministic unmasked denoised coefficients; not a trajectory error bound",
            "steps": self.rows,
        }

    def before_attempt(self, runtime, decision, x):
        if (
            self.settings["mode"] == "replace_selected"
            and decision["step_id"] == self.settings["selected_step"]
            and self.phase != "offline_replay"
            and runtime._step.mode == "forecast"
        ):
            self.selected_input = x.detach().to(device="cpu", copy=True)

    def _verify_selected_input(self, x):
        if (
            self.selected_input is None
            or x.shape != self.selected_input.shape
            or x.dtype != self.selected_input.dtype
        ):
            raise RuntimeError("RES intervention lost its selected input witness")
        current = x.detach().to(device="cpu").contiguous().view(torch.uint8)
        saved = self.selected_input.contiguous().view(torch.uint8)
        if not torch.equal(current, saved):
            raise RuntimeError(
                "RES intervention input was mutated by a model-call wrapper"
            )

    def maybe_replace(self, runtime, decision, x, result, execute_attempt):
        if decision["step_id"] != self.settings["selected_step"]:
            return result
        step = runtime._step
        if self.phase == "offline_replay":
            self.selected_status = "replay_observation_only"
            return result
        if step is None or step.mode != "forecast":
            self.selected_status = "selected_step_was_actual"
            return result
        self.selected_status = "selected_forecast_traced"
        if self.settings["mode"] != "replace_selected":
            return result
        # Only intervene in a complete forecast transaction. Partial transactions
        # retain the standard finalization/retry behavior and cannot be called a probe.
        if (
            not step.calls
            or any(call.observed_actual for call in step.calls)
            or not all(call.used_forecast for call in step.calls)
        ):
            self.selected_status = "incomplete_forecast_transaction"
            return result
        forecast = result.detach().to(device="cpu", copy=True)
        self._verify_selected_input(x)
        self.discarded_forecast_model_calls += len(step.calls)
        runtime.prepare_actual_retry(
            decision["run_id"],
            decision["step_id"],
            "RES investigation selected forecast replacement",
        )
        decision["actual"] = True
        decision["reason"] = "RES investigation selected forecast replacement"
        actual = execute_attempt(decision)
        self._verify_selected_input(x)
        if not runtime._step.calls or not all(
            call.observed_actual for call in runtime._step.calls
        ):
            raise RuntimeError(
                "RES intervention did not execute the native H3 actual path"
            )
        errors = {}
        for (name, ref), (other_name, cand) in zip(
            _streams(actual, self.latent_shapes),
            _streams(forecast, self.latent_shapes),
            strict=True,
        ):
            if name != other_name:
                raise ValueError("RES intervention stream mapping changed")
            errors[name] = delta(ref, cand)
        self.pending_error = errors
        self.checkpoints["selected_forecast"] = forecast
        self.checkpoints["selected_exact"] = actual.detach().to(device="cpu", copy=True)
        self.replacements += 1
        self.selected_status = "replaced_with_actual"
        return actual

    def record(self, runtime, decision, x, result):
        i = decision["step_id"]
        row = {
            "step": i,
            "coordinate": decision["coordinate"],
            "mode": runtime.last_completed_mode,
            "reason": runtime.last_completed_reason,
            "native_transaction_verified": runtime.last_completed_step_id == i,
            "input": {
                name: _summary(part) for name, part in _streams(x, self.latent_shapes)
            },
            "denoised": {
                name: _summary(part)
                for name, part in _streams(result, self.latent_shapes)
            },
            "coefficients": self.coefficients[i],
        }
        if self.pending_error is not None:
            row["same_state_forecast_error"] = self.pending_error
            row["same_state_input_verified"] = True
            self.pending_error = None
        self.selected_input = None
        self.rows.append(row)
        if i == 0 or abs(i - self.settings["selected_step"]) <= 1:
            self.checkpoints[f"step_{i:03d}_input"] = x.detach().to(
                device="cpu", copy=True
            )
            self.checkpoints[f"step_{i:03d}_denoised"] = result.detach().to(
                device="cpu", copy=True
            )

    def finish(self, runtime, result, success):
        try:
            self.report.update(
                {
                    "completed": success,
                    "selected_status": self.selected_status,
                    "valid_trace": success
                    and len(self.rows) == len(self.coefficients)
                    and all(row["native_transaction_verified"] for row in self.rows),
                    "replacement_count": self.replacements,
                    "discarded_forecast_model_calls": self.discarded_forecast_model_calls,
                    "wall_seconds_with_diagnostics": time.perf_counter() - self.started,
                    "runtime_summary": runtime.debug_summary(),
                }
            )
            if result is not None:
                for name, part in _streams(result, self.latent_shapes):
                    self.checkpoints[f"final_{name}"] = part.detach().to(
                        device="cpu", copy=True
                    )
            self.report["checkpoint_tensor_bytes"] = sum(
                v.numel() * v.element_size() for v in self.checkpoints.values()
            )
            import folder_paths
            from safetensors.torch import save_file

            directory = (
                Path(folder_paths.get_output_directory())
                / "spectrum_h3_res_investigation"
            )
            directory.mkdir(parents=True, exist_ok=True)
            label = (
                re.sub(r"[^A-Za-z0-9_-]+", "_", self.settings["label"])[:64] or "res"
            )
            stem = f"{label}_{self.phase}_{uuid.uuid4().hex}"
            tensor_path = directory / f"{stem}.safetensors"
            report_path = directory / f"{stem}.json"
            if self.checkpoints:
                save_file(
                    {
                        key: value.contiguous()
                        for key, value in self.checkpoints.items()
                    },
                    str(tensor_path),
                )
                self.report["tensor_file"] = tensor_path.name
            report_path.write_text(
                json.dumps(self.report, indent=2, allow_nan=False), encoding="utf-8"
            )
            LOG.warning("Spectrum H3 RES investigation report: %s", report_path)
        finally:
            self.checkpoints.clear()
            self.selected_input = None


def begin(runtime, model_options, sigmas, sampler, phase, seed, latent_shapes, masked):
    runtime._res_investigation = None
    settings = model_options.get(SETTINGS_KEY)
    if settings is None:
        return
    import comfy.k_diffusion.sampling as native

    if sampler.sampler_function is not native.sample_res_multistep:
        raise ValueError(
            "RES Investigation requires native res_multistep (CFG++ and ancestral are separate contracts)"
        )
    if "multigpu_clones" in model_options:
        raise ValueError("RES Investigation requires a single GPU model-call stream")
    if not 0 <= settings["selected_step"] < len(sigmas) - 1:
        raise ValueError(
            "RES Investigation selected_step is outside the supplied schedule"
        )
    runtime._res_investigation = RESInvestigation(
        settings, runtime, sigmas, sampler, phase, seed, latent_shapes, masked
    )


def before_attempt(runtime, decision, x):
    probe = getattr(runtime, "_res_investigation", None)
    if probe is not None:
        probe.before_attempt(runtime, decision, x)


def maybe_replace(runtime, decision, x, result, execute_attempt):
    probe = getattr(runtime, "_res_investigation", None)
    if probe is None:
        return result
    return probe.maybe_replace(runtime, decision, x, result, execute_attempt)


def record(runtime, decision, x, result):
    probe = getattr(runtime, "_res_investigation", None)
    if probe is not None:
        probe.record(runtime, decision, x, result)


def finish(runtime, result, success):
    probe = getattr(runtime, "_res_investigation", None)
    runtime._res_investigation = None
    if probe is None:
        return
    try:
        probe.finish(runtime, result, success)
    except Exception:
        # File output failure must not replace the original sampling exception
        # or discard completed media. Report the failure explicitly to the user.
        LOG.exception("Spectrum H3 RES investigation report could not be saved")


class SpectrumRESInvestigation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["trace", "replace_selected", "all_actual"],),
                "selected_step": (
                    "INT",
                    {
                        "default": 23,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "Zero-based selected forecast. Replacement runs in the causal/capture pass and commits an actual anchor.",
                    },
                ),
                "label": ("STRING", {"default": "res_test"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Spectrum/MiniMax H3/Research"

    def apply(self, model, mode, selected_step, label):
        from .runtime import SpectrumH3Runtime
        from .sampling import (
            BINDING_KEY,
            SpectrumH3Binding,
            _binding_from_model_options,
        )

        if mode not in {"trace", "replace_selected", "all_actual"}:
            raise ValueError("unknown RES investigation mode")
        if type(selected_step) is not int or selected_step < 0:
            raise ValueError("selected_step must be a nonnegative integer")
        binding = _binding_from_model_options(model.model_options)
        if binding is None:
            raise ValueError(
                "Connect Spectrum Apply MiniMax H3 before RES Investigation"
            )
        patched = model.clone()
        config = replace(
            binding.runtime.config,
            force_actual=(mode == "all_actual" or binding.runtime.config.force_actual),
            offline_smoothing_replay=(
                False
                if mode == "all_actual"
                else binding.runtime.config.offline_smoothing_replay
            ),
        )
        patched.model_options[BINDING_KEY] = SpectrumH3Binding(
            SpectrumH3Runtime(config)
        )
        patched.model_options[SETTINGS_KEY] = {
            "mode": mode,
            "selected_step": selected_step,
            "label": str(label),
        }
        return (patched,)
