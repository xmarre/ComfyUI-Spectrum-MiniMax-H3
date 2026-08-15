from __future__ import annotations

import copy
import inspect
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

from .er_sde_ksampler_contract import (
    KSamplerSampleContract,
    validate_ksampler_sample,
)
from .er_sde_stochastic import (
    ERSDEStochasticTracker,
    ERSDETrackingError,
    function_ast_digest,
    native_default_er_sde_noise_scaler,
)
from .model_aware import get_model_forecastability_profile
from .rollback import run_selective_rollback_euler
from .runtime import ForecastRetryActual, OfflineReplayAbort, SpectrumH3Runtime

LOG = logging.getLogger(__name__)

BINDING_KEY = "spectrum_h3_binding"
RUNTIME_KEY = "spectrum_h3_runtime"
RUN_ID_KEY = "spectrum_h3_run_id"
STEP_ID_KEY = "spectrum_h3_step_id"
COORDINATE_KEY = "spectrum_h3_coordinate"
ACTUAL_KEY = "spectrum_h3_actual"
REASON_KEY = "spectrum_h3_reason"
ER_SDE_TRACKER_KEY = "spectrum_h3_er_sde_stochastic_tracker"
WRAPPER_KEY = "spectrum_minimax_h3"
KJ_PREVIEW_WRAPPER_KEY = "kj_preview_override"

SUPPORTED_SINGLE_CALL_SAMPLERS = frozenset(
    {
        "_turbo_sampler",
        "sample_euler",
        "sample_er_sde",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

RES_MULTISTEP_SAMPLERS = frozenset(
    {
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

ER_SDE_SAMPLERS = frozenset({"sample_er_sde"})
ER_SDE_NATIVE_SCALER_MODULE = "comfy_extras.nodes_custom_sampler"
ER_SDE_NATIVE_SCALER_FREEVARS = {
    "SamplerER_SDE.execute.<locals>.er_sde_noise_scaler": ("eta",),
    "SamplerER_SDE.execute.<locals>.reverse_time_sde_noise_scaler": ("eta",),
    "SamplerER_SDE.execute.<locals>.ode_noise_scaler": (),
}
ER_SDE_NATIVE_FUNCTION_DIGEST = (
    "55b76bd3a76d44fbd363de39f2ab3ea672c78de9f001f47168b47ec6ff6d2447"
)
ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST = (
    "11cfe81f36f0b43e96c12eff32a4f074f35227a53ca116e837bf268b6383f9ad"
)
ER_SDE_KSAMPLER_SAMPLE_DIGEST = (
    "cacf00387fb2b9d0e076c68e0bd3f75d104801aa8d34239115b3208153ed8dac"
)
ER_SDE_TRACKED_OPTIONS = frozenset(
    {"s_noise", "noise_sampler", "noise_scaler", "max_stage"}
)


@dataclass(slots=True)
class SpectrumH3Binding:
    runtime: SpectrumH3Runtime


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def sampler_is_supported(sampler: Any) -> bool:
    return sampler_name(sampler) in SUPPORTED_SINGLE_CALL_SAMPLERS


def _er_sde_noise_scaler_supports_replay(noise_scaler: Any) -> bool:
    """Accept only the reviewed native SamplerER_SDE scaler closures."""
    if noise_scaler is None:
        return True
    if getattr(noise_scaler, "__module__", None) != ER_SDE_NATIVE_SCALER_MODULE:
        return False

    qualname = getattr(noise_scaler, "__qualname__", None)
    expected_freevars = ER_SDE_NATIVE_SCALER_FREEVARS.get(qualname)
    if expected_freevars is None:
        return False

    code = getattr(noise_scaler, "__code__", None)
    if code is None or tuple(code.co_freevars) != expected_freevars:
        return False

    native_module = sys.modules.get(ER_SDE_NATIVE_SCALER_MODULE)
    if native_module is None or getattr(noise_scaler, "__globals__", None) is not vars(native_module):
        return False
    sampler_class = getattr(native_module, "SamplerER_SDE", None)
    execute = getattr(sampler_class, "execute", None)
    execute_function = getattr(execute, "__func__", execute)
    execute_code = getattr(execute_function, "__code__", None)
    if execute_code is None or not any(item is code for item in execute_code.co_consts):
        return False

    closure = getattr(noise_scaler, "__closure__", None) or ()
    if len(closure) != len(expected_freevars):
        return False
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            return False
        if type(value) not in (int, float):
            return False
    return True


def sampler_supports_seeded_replay(sampler: Any) -> bool:
    """Return whether a fresh invocation can reconstruct the sampler's random stream."""
    if not sampler_is_supported(sampler):
        return False
    if sampler_name(sampler) not in ER_SDE_SAMPLERS:
        return True

    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False
    if options.get("noise_sampler") is not None:
        return False
    return _er_sde_noise_scaler_supports_replay(options.get("noise_scaler"))


def _ksampler_sample_contract(sampler: Any) -> KSamplerSampleContract:
    import comfy.samplers

    sample_method = vars(type(sampler)).get("sample")
    return validate_ksampler_sample(
        sample_method,
        expected_adapter=comfy.samplers.KSamplerX0Inpaint,
        expected_reference_digest=ER_SDE_KSAMPLER_SAMPLE_DIGEST,
    )


def _copy_ksampler_with_options(
    sampler: Any,
    options: dict[str, Any],
) -> tuple[Any | None, str | None]:
    """Prove and perform the shallow copy used to isolate tracked options."""
    if not inspect.isfunction(vars(type(sampler)).get("sample")):
        return None, "KSAMPLER.sample is not a plain function descriptor"
    if vars(type(sampler)).get("__copy__") is not None:
        return None, "KSAMPLER defines an unreviewed custom __copy__ method"
    try:
        original_state = vars(sampler)
        original_options = sampler.extra_options
        copied = copy.copy(sampler)
    except Exception as exc:  # noqa: BLE001 - fail closed on custom runtime state
        return None, f"KSAMPLER state/copy inspection failed: {type(exc).__name__}: {exc}"
    if copied is sampler or type(copied) is not type(sampler):
        return None, "KSAMPLER shallow copy did not create a distinct native instance"
    try:
        for name, value in original_state.items():
            if getattr(copied, name, object()) is not value:
                return None, f"KSAMPLER shallow copy changed instance field {name!r}"
        copied.extra_options = options
        if sampler.extra_options is not original_options:
            return None, "KSAMPLER shallow copy mutated the original extra_options"
        if copied.extra_options is not options:
            return None, "KSAMPLER shallow copy did not accept invocation-local options"
        if copied.sampler_function is not sampler.sampler_function:
            return None, "KSAMPLER shallow copy changed sampler_function binding"
        if copied.inpaint_options is not sampler.inpaint_options:
            return None, "KSAMPLER shallow copy changed inpaint_options ownership"
    except Exception as exc:  # noqa: BLE001 - fail closed on custom runtime state
        return None, f"KSAMPLER copied-state validation failed: {type(exc).__name__}: {exc}"
    return copied, None


def _er_sde_tracking_contract(sampler: Any) -> tuple[bool, str | None]:
    """Accept only the native solver/wrapper semantics reviewed by this project."""
    import comfy.k_diffusion.sampling as native_sampling
    import comfy.samplers

    function = getattr(sampler, "sampler_function", None)
    if function is not native_sampling.sample_er_sde:
        return False, "sampler function is not native ComfyUI sample_er_sde"
    if function_ast_digest(function) != ER_SDE_NATIVE_FUNCTION_DIGEST:
        return False, "native sample_er_sde implementation is not a reviewed revision"
    if type(sampler) is not comfy.samplers.KSAMPLER:
        return False, "ER-SDE sampler object is not native ComfyUI KSAMPLER"
    if (
        comfy.samplers.KSAMPLER.__module__ != "comfy.samplers"
        or comfy.samplers.KSAMPLER.__name__ != "KSAMPLER"
    ):
        return False, "native ComfyUI KSAMPLER class provenance is unreviewed"
    if (
        comfy.samplers.KSamplerX0Inpaint.__module__ != "comfy.samplers"
        or comfy.samplers.KSamplerX0Inpaint.__name__ != "KSamplerX0Inpaint"
    ):
        return False, "native KSamplerX0Inpaint adapter provenance is unreviewed"
    sample_contract = _ksampler_sample_contract(sampler)
    if not sample_contract.accepted:
        return False, (
            f"KSAMPLER.sample contract rejected: {sample_contract.failure}; "
            f"{sample_contract.provenance.log_fields()}"
        )
    _, copy_failure = _copy_ksampler_with_options(sampler, {})
    if copy_failure is not None:
        return False, f"KSAMPLER copy contract rejected: {copy_failure}"
    if (
        function_ast_digest(native_sampling.default_noise_sampler)
        != ER_SDE_DEFAULT_NOISE_SAMPLER_DIGEST
    ):
        return False, "native default_noise_sampler implementation is not a reviewed revision"

    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False, "ER-SDE sampler options are not a dictionary"
    unknown = set(options) - ER_SDE_TRACKED_OPTIONS
    if unknown:
        return False, f"ER-SDE sampler has unreviewed options: {sorted(unknown)}"
    try:
        s_noise = float(options.get("s_noise", 1.0))
    except (TypeError, ValueError):
        return False, "ER-SDE s_noise is not numeric"
    if not torch.isfinite(torch.tensor(s_noise)) or s_noise < 0.0:
        return False, "ER-SDE s_noise must be finite and nonnegative"
    max_stage = options.get("max_stage", 3)
    if isinstance(max_stage, bool) or not isinstance(max_stage, int) or not 1 <= max_stage <= 3:
        return False, "ER-SDE max_stage must be an integer in [1, 3]"
    if s_noise > 0.0:
        if options.get("noise_sampler") is not None:
            return False, "custom ER-SDE noise_sampler provenance is unreviewed"
        if not _er_sde_noise_scaler_supports_replay(options.get("noise_scaler")):
            return False, "custom ER-SDE noise_scaler provenance is unreviewed"
    return True, None


def _native_er_sde_preflight_reason(
    sampler: Any,
    model_options: dict[str, Any] | None,
) -> str | None:
    """Resolve native ER-SDE failures before Spectrum starts retaining state."""
    function = getattr(sampler, "sampler_function", None)
    if getattr(function, "__module__", None) != "comfy.k_diffusion.sampling":
        return None
    supported, reason = _er_sde_tracking_contract(sampler)
    if not supported:
        return reason or "native ER-SDE compensation contract is unproven"
    options = getattr(sampler, "extra_options", {}) or {}
    if float(options.get("s_noise", 1.0)) == 0.0:
        return None

    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.get_all_wrappers(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        model_options or {},
        is_model_options=True,
    )
    if len(wrappers) != 1 or wrappers[0] is not sampler_sample_wrapper:
        return "another SAMPLER_SAMPLE wrapper makes ER-SDE ordering unproven"
    return None


def max_consecutive_forecasts(sampler: Any) -> int | None:
    return 1 if sampler_is_supported(sampler) else None


def min_actual_steps_after_forecast(sampler: Any) -> int:
    return 1 if sampler_name(sampler) in SUPPORTED_SINGLE_CALL_SAMPLERS else 0


def min_tail_actual_steps(sampler: Any) -> int:
    return 3 if sampler_name(sampler) in RES_MULTISTEP_SAMPLERS else 0


def _binding_from_model_options(
    model_options: dict[str, Any] | None,
) -> SpectrumH3Binding | None:
    binding = (model_options or {}).get(BINDING_KEY)
    return binding if isinstance(binding, SpectrumH3Binding) else None


def _copy_condition_structure(value: Any) -> Any:
    """Copy mutable conditioning containers while sharing tensor/model payloads."""
    if isinstance(value, dict):
        return {key: _copy_condition_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_condition_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_condition_structure(item) for item in value)
    return value


def _offline_progress_callbacks(callback, total_steps: int):
    """Report both passes while keeping previews and callback side effects replay-only."""
    if callback is None or total_steps <= 0:
        return None, callback, None

    import comfy.utils

    total_work = total_steps * 2
    progress = comfy.utils.ProgressBar(total_work)
    replay_finished = False

    def capture_callback(step, _x0, _x, _pass_steps):
        progress.update_absolute(step + 1, total_work)

    def replay_callback(step, x0, x, _pass_steps):
        nonlocal replay_finished
        callback(total_steps + step, x0, x, total_work)
        replay_finished = step + 1 >= total_steps

    def complete_progress():
        if not replay_finished:
            progress.update_absolute(total_work, total_work)

    return capture_callback, replay_callback, complete_progress


def copy_model_options_with_step(
    model_options: dict[str, Any] | None,
    runtime: SpectrumH3Runtime,
    decision: dict[str, Any],
) -> dict[str, Any]:
    copied = dict(model_options or {})
    transformer_options = dict(copied.get("transformer_options") or {})
    copied["transformer_options"] = transformer_options
    transformer_options[RUNTIME_KEY] = runtime
    transformer_options[RUN_ID_KEY] = int(decision["run_id"])
    transformer_options[STEP_ID_KEY] = int(decision["step_id"])
    transformer_options[COORDINATE_KEY] = float(decision["coordinate"])
    transformer_options[ACTUAL_KEY] = bool(decision["actual"])
    transformer_options[REASON_KEY] = str(decision["reason"])
    return copied


def outer_sample_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None:
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    transformer_options = (
        (getattr(guider, "model_options", None) or {}).get("transformer_options")
        or {}
    )
    if transformer_options.get("easycache") is not None:
        LOG.warning(
            "Spectrum H3 disabled for this run because EasyCache or LazyCache is active on the same model"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    runtime = binding.runtime
    name = sampler_name(sampler)
    if name in ER_SDE_SAMPLERS:
        preflight_reason = _native_er_sde_preflight_reason(
            sampler,
            getattr(guider, "model_options", None),
        )
        if preflight_reason is not None:
            LOG.warning(
                "Spectrum H3 disabled for this ER-SDE run; running the untouched "
                "native sampler because stochastic compensation is unavailable: %s",
                preflight_reason,
            )
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
    profile_eligible = sampler_is_supported(sampler) and (
        not runtime.config.offline_smoothing_replay
        or sampler_supports_seeded_replay(sampler)
    )
    if runtime.config.model_aware_mode != "off" and profile_eligible:
        try:
            lookup = get_model_forecastability_profile(guider.model_patcher)
            runtime.set_model_profile(lookup)
            if runtime.config.debug:
                profile = lookup.profile
                LOG.warning(
                    "Spectrum H3 model-aware profile base=%s patches=%s patch_keys=%s "
                    "recognized_lora=%s unknown_patches=%s cache=%s build_s=%.6f "
                    "lookup_s=%.6f memory_bytes=%s sensitivity=%.6f perturbation=%.6f "
                    "final_perturbation=%.6f confidence=%.6f "
                    "profile_payload=compact_scalar_sensitivities retained_head_bytes=0",
                    profile.base_model_identity,
                    profile.active_patch_count,
                    profile.active_patch_keys,
                    profile.recognized_lora_count,
                    profile.unknown_patch_count,
                    "hit" if lookup.cache_hit else "miss",
                    profile.build_seconds,
                    lookup.lookup_seconds,
                    profile.estimated_bytes,
                    profile.aggregate_sensitivity,
                    profile.patch_perturbation,
                    profile.final_block_perturbation,
                    profile.profile_confidence,
                )
        except torch.cuda.OutOfMemoryError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            runtime.disable_model_aware(f"profile construction failed: {exc}")
            LOG.warning(
                "Spectrum H3 model-aware forecasting unavailable for this run: %s",
                exc,
            )

    def execute_run(
        run_noise,
        run_latent,
        run_sigmas,
        run_mask,
        run_callback,
        run_disable_pbar,
        *,
        phase: str,
    ):
        run_id = runtime.start_run(
            run_sigmas,
            name,
            supported_sampler=sampler_is_supported(sampler),
            max_consecutive_forecasts=max_consecutive_forecasts(sampler),
            min_actual_steps_after_forecast=min_actual_steps_after_forecast(sampler),
            min_tail_actual_steps=min_tail_actual_steps(sampler),
        )
        if name in ER_SDE_SAMPLERS and (
            runtime.config.anchor_residual_feedback
            or runtime.config.selective_rollback_correction
        ):
            runtime.disable_experiment(
                "ER-SDE is reviewed only for ordinary Spectrum and offline smoothing replay"
            )
        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 run start phase=%s run_id=%s sampler=%s steps=%s supported=%s",
                phase,
                run_id,
                name,
                runtime.stats.total_steps,
                runtime.supported_sampler,
            )
        started = time.perf_counter()
        try:
            result = executor(
                run_noise,
                run_latent,
                sampler,
                run_sigmas,
                run_mask,
                run_callback,
                run_disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
            runtime.log_offline_transition(
                "first_pass_executor_return" if phase == "offline_first_pass" else "executor_return",
                phase=phase,
                run_id=run_id,
                elapsed_s=f"{time.perf_counter() - started:.6f}",
            )
            return result
        finally:
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 run summary phase=%s wall_s=%.3f %s",
                    phase,
                    time.perf_counter() - started,
                    runtime.debug_summary(),
                )
            runtime.end_run(run_id)
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 run teardown phase=%s run_id=%s", phase, run_id
                )

    if not runtime.config.offline_smoothing_replay:
        result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass",
        )
        return result

    if not sampler_is_supported(sampler):
        LOG.warning(
            "Spectrum H3 offline smoothing replay is unsupported for sampler %s; running one native pass",
            name,
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not sampler_supports_seeded_replay(sampler):
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires ER-SDE's native seeded "
            "noise_sampler and noise_scaler; running one native pass"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not all(torch.is_tensor(value) for value in (noise, latent_image, sigmas)):
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires tensor sampling inputs; running one ordinary pass"
        )
        result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass_fallback",
        )
        return result

    replay_noise = noise.detach().clone()
    replay_latent = latent_image.detach().clone()
    replay_sigmas = sigmas.detach().clone()
    replay_mask = (
        denoise_mask.detach().clone()
        if torch.is_tensor(denoise_mask)
        else denoise_mask
    )
    initial_conds = (
        _copy_condition_structure(guider.conds) if hasattr(guider, "conds") else None
    )
    offline_steps = max(0, sigmas.numel() - 1)
    capture_callback, replay_callback, complete_progress = _offline_progress_callbacks(
        callback,
        offline_steps,
    )
    runtime.begin_offline_capture(total_steps=offline_steps, sampler_name=name)
    try:
        first_result = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            capture_callback,
            disable_pbar,
            phase="offline_first_pass",
        )
        try:
            capture_complete = runtime.complete_offline_capture()
        except Exception as exc:  # noqa: BLE001 - preserve completed first-pass output
            archive = runtime.offline_archive
            if archive is not None:
                archive.invalidate(f"offline capture completion failed: {exc}")
            capture_complete = False
            runtime.log_offline_transition(
                "complete_offline_capture_failed",
                error_type=type(exc).__name__,
                reason=exc,
            )
            LOG.warning(
                "Spectrum H3 offline capture completion failed; preserving the valid "
                "first-pass result: %s",
                exc,
            )
        if not capture_complete:
            if complete_progress is not None:
                complete_progress()
            reason = (
                runtime.offline_archive.failure_reason
                if runtime.offline_archive is not None
                else "offline archive was not retained"
            )
            LOG.warning(
                "Spectrum H3 offline replay skipped; returning the valid first-pass result: %s",
                reason,
            )
            return first_result

        try:
            runtime.begin_offline_replay()
        except Exception as exc:  # noqa: BLE001 - preserve completed first-pass output
            runtime.log_offline_transition(
                "begin_offline_replay_failed",
                error_type=type(exc).__name__,
                reason=exc,
            )
            if complete_progress is not None:
                complete_progress()
            LOG.warning(
                "Spectrum H3 offline replay setup failed; returning the valid first-pass "
                "result: %s",
                exc,
            )
            return first_result
        if initial_conds is not None:
            guider.conds = _copy_condition_structure(initial_conds)
        try:
            replay_result = execute_run(
                replay_noise,
                replay_latent,
                replay_sigmas,
                replay_mask,
                replay_callback,
                True,
                phase="offline_replay",
            )
            if complete_progress is not None:
                complete_progress()
        except OfflineReplayAbort as exc:
            if complete_progress is not None:
                complete_progress()
            LOG.warning(
                "Spectrum H3 offline replay aborted; returning the valid first-pass result: %s",
                exc,
            )
            return first_result
        return replay_result
    finally:
        runtime.release_offline_archive()


def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if (
        binding is None
        or binding.runtime.active_run_id is None
        or not binding.runtime.supported_sampler
    ):
        return executor(x, timestep, model_options or {}, seed)

    if "multigpu_clones" in (model_options or {}):
        if binding.runtime.config.debug:
            LOG.warning(
                "Spectrum H3 native fallback: multi-GPU parallel model calls are not transactionally supported"
            )
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    decision = runtime.begin_step(timestep)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 step run_id=%s step=%s coordinate=%.6f decision=%s reason=%s history=%s window=%.3f",
            decision["run_id"],
            decision["step_id"],
            decision["coordinate"],
            "actual" if decision["actual"] else "forecast",
            decision["reason"],
            runtime.prediction_history_length,
            runtime.stats.current_window,
        )
        model_decision = runtime.active_model_aware_decision
        if model_decision is not None:
            audio_gain = model_decision.audio_correction_telemetry
            video_gain = model_decision.video_correction_telemetry
            LOG.warning(
                "Spectrum H3 model-aware step=%s trajectory_risk=%.6f model_risk=%.6f "
                "patch_risk=%.6f combined_risk=%.6f confidence=%.6f horizon=%.3f "
                "degree=%s ridge=%.8f audio_blend=%.6f video_blend=%.6f "
                "audio_generic_projection=%.6f audio_raw_generic_gain=%.6f "
                "audio_generic_gain=%.6f audio_applied_gain=%.6f "
                "audio_generic_bound_active=%s "
                "video_generic_projection=%.6f video_raw_generic_gain=%.6f "
                "video_generic_gain=%.6f video_applied_gain=%.6f "
                "video_generic_bound_active=%s model_informed_correction=retired decision=%s",
                decision["step_id"],
                model_decision.trajectory_risk,
                model_decision.model_risk,
                model_decision.patch_risk,
                model_decision.combined_risk,
                model_decision.confidence,
                model_decision.forecast_horizon,
                model_decision.degree,
                model_decision.ridge_lambda,
                model_decision.audio_blend_weight,
                model_decision.video_blend_weight,
                audio_gain.residual_projection,
                audio_gain.raw_generic_gain,
                audio_gain.generic_gain,
                model_decision.audio_correction_gain,
                audio_gain.generic_bound_active,
                video_gain.residual_projection,
                video_gain.raw_generic_gain,
                video_gain.generic_gain,
                model_decision.video_correction_gain,
                video_gain.generic_bound_active,
                "ACTUAL" if decision["actual"] else "FORECAST",
            )

    def execute_attempt(attempt_decision: dict[str, Any]):
        patched = copy_model_options_with_step(model_options, runtime, attempt_decision)
        return executor(x, timestep, patched, seed)

    def consume_er_sde_increment(
        result: torch.Tensor,
        attempt_decision: dict[str, Any],
    ) -> torch.Tensor:
        transformer_options = (model_options or {}).get("transformer_options") or {}
        tracker = transformer_options.get(ER_SDE_TRACKER_KEY)
        if not isinstance(tracker, ERSDEStochasticTracker):
            return result
        descriptor = None
        try:
            descriptor = runtime.describe_current_er_sde_step(
                int(attempt_decision["run_id"]),
                int(attempt_decision["step_id"]),
            )
            return tracker.consume(result, descriptor)
        except ERSDETrackingError as exc:
            tracker.clear()
            if descriptor is not None and descriptor.mode == "replay":
                raise OfflineReplayAbort(
                    f"ER-SDE stochastic-state compensation failed during replay: {exc}"
                ) from exc
            if not bool(attempt_decision["actual"]):
                raise ForecastRetryActual(
                    f"ER-SDE stochastic-state compensation failed: {exc}"
                ) from exc
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 ER-SDE tracker discarded on state-aware actual "
                    "step=%s reason=%s",
                    attempt_decision["step_id"],
                    exc,
                )
            return result

    try:
        try:
            result = execute_attempt(decision)
            runtime.log_offline_transition(
                "actual_executor_return" if decision["actual"] else "forecast_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
            )
            result = consume_er_sde_increment(result, decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
        except ForecastRetryActual as retry:
            runtime.prepare_actual_retry(
                decision["run_id"], decision["step_id"], str(retry)
            )
            retry_decision = dict(decision)
            retry_decision["actual"] = True
            retry_decision["reason"] = f"forecast transaction retry: {retry}"
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 forecast retry run_id=%s step=%s reason=%s",
                    decision["run_id"],
                    decision["step_id"],
                    retry,
                )
            result = execute_attempt(retry_decision)
            runtime.log_offline_transition(
                "actual_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
                retry=True,
            )
            result = consume_er_sde_increment(result, retry_decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


def _run_tracked_er_sde(
    executor,
    runtime: SpectrumH3Runtime,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image,
    denoise_mask,
    disable_pbar,
):
    supported, reason = _er_sde_tracking_contract(executor.class_obj)
    if len(executor.wrappers) != 1:
        supported = False
        reason = "another SAMPLER_SAMPLE wrapper makes ER-SDE ordering unproven"
    if not supported:
        assert reason is not None
        runtime.disable_forecasting_for_run(reason)
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving native all-actual "
            "sampling because stochastic compensation is unavailable: %s",
            reason,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    sampler = executor.class_obj
    options = dict(getattr(sampler, "extra_options", {}) or {})
    configured_s_noise = float(options.get("s_noise", 1.0))
    model_sampling = model_wrap.model_patcher.get_model_object("model_sampling")
    noise_scale = float(getattr(model_sampling, "noise_scale", 1.0))
    effective_s_noise = configured_s_noise * noise_scale
    if not torch.isfinite(torch.tensor(effective_s_noise)) or effective_s_noise < 0.0:
        reason = "effective ER-SDE s_noise is not finite and nonnegative"
        runtime.disable_forecasting_for_run(reason)
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving native all-actual "
            "sampling because %s",
            reason,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if effective_s_noise == 0.0:
        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 ER-SDE compensation q_pending=false applied=false "
                "reason=no_stochastic_increment max_stage=%s stochastic=false",
                options.get("max_stage", 3),
            )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if not torch.is_tensor(noise):
        reason = "ER-SDE sampler noise is not a packed tensor"
        runtime.disable_forecasting_for_run(reason)
        LOG.warning("Spectrum H3 disabled for this ER-SDE run: %s", reason)
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    import comfy.k_diffusion.sampling as native_sampling

    base_noise_sampler = native_sampling.default_noise_sampler(
        noise,
        seed=extra_args.get("seed"),
    )
    base_noise_scaler = options.get("noise_scaler")
    if base_noise_scaler is None:
        base_noise_scaler = native_default_er_sde_noise_scaler
    tracker = ERSDEStochasticTracker(
        noise_sampler=base_noise_sampler,
        noise_scaler=base_noise_scaler,
        effective_s_noise=effective_s_noise,
        max_stage=int(options.get("max_stage", 3)),
        debug=runtime.config.debug,
        run_id=int(runtime.active_run_id),
    )
    tracked_options = dict(options)
    tracked_options["noise_sampler"] = tracker.noise_sampler
    tracked_options["noise_scaler"] = tracker.noise_scaler
    tracked_sampler, copy_failure = _copy_ksampler_with_options(
        sampler,
        tracked_options,
    )
    if copy_failure is not None:
        tracker.clear()
        runtime.disable_forecasting_for_run(copy_failure)
        LOG.warning(
            "Spectrum H3 disabled for this ER-SDE run; preserving native all-actual "
            "sampling because KSAMPLER copy validation failed: %s",
            copy_failure,
        )
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    assert tracked_sampler is not None

    if runtime.config.debug:
        sample_contract = _ksampler_sample_contract(sampler)
        LOG.warning(
            "Spectrum H3 ER-SDE stochastic tracking active run_id=%s sampler=%s "
            "max_stage=%s configured_s_noise=%.8f effective_s_noise=%.8f "
            "ksampler_contract=accepted %s",
            runtime.active_run_id,
            sampler_name(sampler),
            options.get("max_stage", 3),
            configured_s_noise,
            effective_s_noise,
            sample_contract.provenance.log_fields(),
        )

    tracked_extra_args = dict(extra_args)
    tracked_model_options = dict(tracked_extra_args.get("model_options") or {})
    tracked_transformer_options = dict(
        tracked_model_options.get("transformer_options") or {}
    )
    tracked_transformer_options[ER_SDE_TRACKER_KEY] = tracker
    tracked_model_options["transformer_options"] = tracked_transformer_options
    tracked_extra_args["model_options"] = tracked_model_options
    try:
        return tracked_sampler.sample(
            model_wrap,
            sigmas,
            tracked_extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    finally:
        tracker.clear()


def sampler_sample_wrapper(
    executor,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image=None,
    denoise_mask=None,
    disable_pbar=False,
):
    binding = _binding_from_model_options(
        getattr(model_wrap, "model_options", None)
    )
    if binding is None or binding.runtime.active_run_id is None:
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    runtime = binding.runtime
    sampler = executor.class_obj
    if sampler_name(sampler) in ER_SDE_SAMPLERS:
        return _run_tracked_er_sde(
            executor,
            runtime,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    if (
        not runtime.config.selective_rollback_correction
        or runtime.experiment_disabled_reason is not None
    ):
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )

    import comfy.k_diffusion.sampling as native_sampling

    function = getattr(sampler, "sampler_function", None)
    options = dict(getattr(sampler, "extra_options", {}) or {})
    unsupported_reason = None
    if function is not native_sampling.sample_euler:
        unsupported_reason = (
            "selective rollback supports only the exact reviewed sample_euler "
            f"contract; got {sampler_name(sampler)}"
        )
    elif set(options) - {"s_churn", "s_tmin", "s_tmax", "s_noise"}:
        unsupported_reason = "selective rollback received unknown Euler sampler options"
    elif float(options.get("s_churn", 0.0)) != 0.0:
        unsupported_reason = "selective rollback does not support Euler churn"
    elif "multigpu_clones" in (extra_args.get("model_options") or {}):
        unsupported_reason = "selective rollback does not support multi-GPU parallel sampling"
    elif len(executor.wrappers) != 1:
        unsupported_reason = "selective rollback does not support another SAMPLER_SAMPLE wrapper"
    else:
        import comfy.patcher_extension

        predict_wrappers = comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            getattr(model_wrap, "model_options", {}) or {},
            is_model_options=True,
        )
        if any(wrapper is not predict_noise_wrapper for wrapper in predict_wrappers):
            unsupported_reason = (
                "selective rollback does not support another PREDICT_NOISE wrapper"
            )

    if unsupported_reason is not None:
        runtime.disable_experiment(unsupported_reason)
        return executor(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
    return run_selective_rollback_euler(
        sampler,
        runtime,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image,
        denoise_mask,
        disable_pbar,
    )


def model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_binding = _binding_from_model_options(
        getattr(source_model, "model_options", None)
    )
    if source_binding is None:
        return
    if not hasattr(cloned_model, "model_options") or cloned_model.model_options is None:
        cloned_model.model_options = {}
    cloned_model.model_options[BINDING_KEY] = SpectrumH3Binding(
        SpectrumH3Runtime(source_binding.runtime.config)
    )


def _place_kj_preview_inside_offline_wrapper(
    model: Any,
    outer_wrapper_type: str,
) -> None:
    """Ensure KJ's observational preview wrapper is entered once for each offline pass."""
    outer_wrappers = (getattr(model, "wrappers", None) or {}).get(outer_wrapper_type)
    if not isinstance(outer_wrappers, dict):
        return
    keys = list(outer_wrappers)
    if (
        KJ_PREVIEW_WRAPPER_KEY not in outer_wrappers
        or WRAPPER_KEY not in outer_wrappers
    ):
        return
    if keys.index(KJ_PREVIEW_WRAPPER_KEY) > keys.index(WRAPPER_KEY):
        return

    preview_wrappers = outer_wrappers.pop(KJ_PREVIEW_WRAPPER_KEY)
    reordered = {}
    for key, wrappers in outer_wrappers.items():
        reordered[key] = wrappers
        if key == WRAPPER_KEY:
            reordered[KJ_PREVIEW_WRAPPER_KEY] = preview_wrappers
    outer_wrappers.clear()
    outer_wrappers.update(reordered)


def install_sampler_wrappers(model: Any, runtime: SpectrumH3Runtime) -> None:
    import comfy.patcher_extension

    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    model.model_options[BINDING_KEY] = SpectrumH3Binding(runtime)
    model.model_options.setdefault("transformer_options", {})

    wrapper_types = comfy.patcher_extension.WrappersMP
    existing_outer = model.get_wrappers(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY)
    if not existing_outer:
        model.add_wrapper_with_key(
            wrapper_types.OUTER_SAMPLE,
            WRAPPER_KEY,
            outer_sample_wrapper,
        )
    if runtime.config.offline_smoothing_replay:
        _place_kj_preview_inside_offline_wrapper(model, wrapper_types.OUTER_SAMPLE)
    existing_predict = model.get_wrappers(wrapper_types.PREDICT_NOISE, WRAPPER_KEY)
    if not existing_predict:
        model.add_wrapper_with_key(
            wrapper_types.PREDICT_NOISE,
            WRAPPER_KEY,
            predict_noise_wrapper,
        )
    existing_sampler = model.get_wrappers(wrapper_types.SAMPLER_SAMPLE, WRAPPER_KEY)
    if not existing_sampler:
        model.add_wrapper_with_key(
            wrapper_types.SAMPLER_SAMPLE,
            WRAPPER_KEY,
            sampler_sample_wrapper,
        )
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLONE
    if not model.get_callbacks(callback_type, WRAPPER_KEY):
        model.add_callback_with_key(callback_type, WRAPPER_KEY, model_clone_callback)
