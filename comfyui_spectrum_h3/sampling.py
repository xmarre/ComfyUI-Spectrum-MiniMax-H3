from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch

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
WRAPPER_KEY = "spectrum_minimax_h3"
KJ_PREVIEW_WRAPPER_KEY = "kj_preview_override"
CONTINUUM_REQUEST_KEY = "h3_continuum"

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


@dataclass(slots=True)
class SpectrumH3Binding:
    runtime: SpectrumH3Runtime


def _continuum_actual_prefix(model_options: dict[str, Any] | None) -> int:
    transformer_options = (model_options or {}).get("transformer_options")
    if not isinstance(transformer_options, dict):
        return 0
    request = transformer_options.get(CONTINUUM_REQUEST_KEY)
    if not isinstance(request, dict):
        return 0

    api = request.get("api")
    active = request.get("active")
    prefix = request.get("min_actual_prefix_steps")
    if type(api) is not int or type(active) is not bool or type(prefix) is not int:
        return 0
    if api != 1 or active is not True or prefix < 0:
        return 0
    return prefix


def _continuum_prefix_for_phase(prefix: int, phase: str) -> int:
    return 0 if phase == "offline_replay" else prefix


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
    continuum_prefix = _continuum_actual_prefix(getattr(guider, "model_options", None))
    continuum_log_emitted = False
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
        nonlocal continuum_log_emitted
        phase_prefix = _continuum_prefix_for_phase(continuum_prefix, phase)
        run_id = runtime.start_run(
            run_sigmas,
            name,
            supported_sampler=sampler_is_supported(sampler),
            max_consecutive_forecasts=max_consecutive_forecasts(sampler),
            min_actual_steps_after_forecast=min_actual_steps_after_forecast(sampler),
            min_tail_actual_steps=min_tail_actual_steps(sampler),
            min_actual_prefix_steps=phase_prefix,
        )
        if phase_prefix > 0 and runtime.supported_sampler and not continuum_log_emitted:
            LOG.warning(
                "Spectrum H3: accepted H3 Continuum API v1, actual prefix=%s",
                min(phase_prefix, max(0, run_sigmas.numel() - 1)),
            )
            continuum_log_emitted = True
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

    try:
        try:
            result = execute_attempt(decision)
            runtime.log_offline_transition(
                "actual_executor_return" if decision["actual"] else "forecast_executor_return",
                run_id=decision["run_id"],
                step=decision["step_id"],
            )
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
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


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

    sampler = executor.class_obj
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
