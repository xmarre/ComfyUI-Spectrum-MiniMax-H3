from __future__ import annotations

import importlib
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from .runtime import OfflineReplayAbort, SpectrumH3Runtime
from .sampling import (
    RUN_ID_KEY,
    RUNTIME_KEY,
    STEP_ID_KEY,
    WRAPPER_KEY,
)

LOG = logging.getLogger(__name__)

_STATE_BASIS_CHUNK_BYTES = 16 * 1024 * 1024
_SANITIZE_CHUNK_BYTES = 16 * 1024 * 1024
_FORECAST_HEAD_CHUNK_BYTES = 16 * 1024 * 1024


def locate_minimax_h3_inner(model: Any) -> tuple[Any | None, str | None]:
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None)
    if inner is not None:
        return inner, "model.diffusion_model"
    inner = getattr(model, "diffusion_model", None)
    if inner is not None:
        return inner, "diffusion_model"
    return None, None


def is_native_minimax_h3(inner: Any) -> bool:
    if inner is None:
        return False
    class_match = (
        type(inner).__name__ == "MiniMaxH3Model"
        and type(inner).__module__ == "comfy.ldm.minimax.model"
    )
    required = (
        "blocks",
        "final_layer",
        "hidden_size",
        "patch_size",
        "latents_dim",
        "audio_latents_dim",
        "sigma_shift_video",
        "sigma_shift_audio",
        "use_adaln_curves",
        "video_patch_proj",
        "audio_patch_proj",
    )
    if not class_match or not all(hasattr(inner, name) for name in required):
        return False
    if not isinstance(inner.use_adaln_curves, bool):
        return False
    timestep_attribute = "adaln_t_table" if inner.use_adaln_curves else "time_embedder"
    return hasattr(inner, timestep_attribute)


def require_native_minimax_h3(model: Any) -> tuple[Any, str]:
    inner, path = locate_minimax_h3_inner(model)
    if not is_native_minimax_h3(inner):
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        if actual == "comfy.ldm.minimax.model.MiniMaxH3Model":
            actual += " with an incompatible native attribute contract"
        raise TypeError(
            "Spectrum Apply MiniMax H3 requires ComfyUI's native "
            f"comfy.ldm.minimax.model.MiniMaxH3Model; discovered {actual}"
        )
    assert path is not None
    return inner, path


def branch_labels(transformer_options: dict[str, Any]) -> tuple[Any, ...] | None:
    conds = transformer_options.get("cond_or_uncond")
    uuids = transformer_options.get("uuids")
    if conds is None or uuids is None:
        return None
    try:
        cond_values = tuple(int(value) for value in conds)
        uuid_values = tuple(str(value) for value in uuids)
    except (TypeError, ValueError):
        return None
    if not cond_values or len(cond_values) != len(uuid_values):
        return None
    return tuple((cond_values[index], uuid_values[index]) for index in range(len(cond_values)))


def target_segments(layout: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    audio = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "audio"]
    video = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "video"]
    if len(audio) != 1 or len(video) != 1:
        raise RuntimeError("native MiniMax H3 layout must contain one target audio and one target video segment")
    aa, ab = audio[0]
    va, vb = video[0]
    if not (0 <= aa < ab <= va < vb <= int(layout.seq_len)):
        raise RuntimeError("native MiniMax H3 target order is not [audio | video] at the packed tail")
    if ab != va or vb != int(layout.seq_len):
        raise RuntimeError("native MiniMax H3 target audio/video rows are not contiguous tail segments")
    return (aa, ab), (va, vb)


def _native_module(inner: Any):
    module = importlib.import_module(type(inner).__module__)
    # time_shift_slope is deliberately absent: cores that still expose it want the
    # audio velocity pre-scaled here, newer ones convert outside this wrapper.
    required = (
        "PackedLayout",
        "unpatchify_video",
        "unpack_audio",
        "time_shift_sigma",
        "patchify_video",
        "pack_audio",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"native MiniMax H3 module is missing required helpers: {', '.join(missing)}")
    return module


def _padded_shape(shape: tuple[int, ...], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    t, h, w = shape[-3:]
    pt, ph, pw = patch_size
    return (
        ((t + pt - 1) // pt) * pt,
        ((h + ph - 1) // ph) * ph,
        ((w + pw - 1) // pw) * pw,
    )


def _apply_exact_state_input_embedding_(
    target: torch.Tensor,
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Add/subtract H3's exact target input embedding without running transformer blocks."""
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("state-conditioned residual target must be a batch-one [1, rows, hidden] tensor")
    if target.shape[-1] != int(inner.hidden_size):
        raise ValueError("state-conditioned residual target hidden width does not match native H3")

    module = _native_module(inner)
    common_dit = importlib.import_module("comfy.ldm.common_dit")
    padded_video = common_dit.pad_to_patch_size(video_x, tuple(inner.patch_size))
    video_rows = module.patchify_video(
        padded_video.to(torch.float32),
        tuple(inner.patch_size),
    )
    audio_rows = module.pack_audio(audio_x.to(torch.float32))
    if audio_rows.ndim != 2 or video_rows.ndim != 2:
        raise ValueError("native H3 patch helpers returned an unexpected row layout")

    compact = target[0]
    total_rows = int(audio_rows.shape[0] + video_rows.shape[0])
    if compact.shape[0] != total_rows:
        raise ValueError(
            "state-conditioned residual target rows do not match native H3 audio/video input rows"
        )

    element_size = torch.empty((), dtype=compact.dtype).element_size()
    rows_per_chunk = max(
        1,
        _STATE_BASIS_CHUNK_BYTES
        // max(1, int(inner.hidden_size) * element_size),
    )
    offset = 0
    for rows, projection in (
        (audio_rows, inner.audio_patch_proj),
        (video_rows, inner.video_patch_proj),
    ):
        count = int(rows.shape[0])
        for start in range(0, count, rows_per_chunk):
            stop = min(count, start + rows_per_chunk)
            embedded = projection(rows[start:stop]).to(compact.dtype)
            compact[offset + start : offset + stop].add_(embedded, alpha=float(scale))
        offset += count
    return target


def _resolve_layout(inner: Any, context: torch.Tensor, video_x: torch.Tensor, audio_x: torch.Tensor, payload: dict[str, Any]):
    module = _native_module(inner)
    latent_t, latent_h, latent_w = _padded_shape(tuple(video_x.shape), tuple(inner.patch_size))
    signature = (int(context.shape[1]), latent_t, latent_h, latent_w, int(audio_x.shape[-1]))
    layout = payload.get("layout")
    if layout is None or tuple(getattr(layout, "signature", ())) != signature:
        layout = module.PackedLayout(
            signature[0],
            signature[1],
            signature[2],
            signature[3],
            signature[4],
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )
    return layout


def topology_signature(
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    context: torch.Tensor,
    layout: Any,
    transformer_options: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[Any, ...]:
    (aa, ab), (va, vb) = target_segments(layout)
    segments = tuple((kind, int(b) - int(a)) for a, b, kind in layout.segments)
    refs = tuple(
        (
            ref.get("kind"),
            ref.get("latent_t"),
            ref.get("latent_h"),
            ref.get("latent_w"),
            ref.get("ref_audio_t"),
        )
        for ref in (payload.get("refs") or ())
    )
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", inner.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", inner.sigma_shift_audio))
    return (
        ("video_shape", tuple(int(v) for v in video_x.shape)),
        ("video_padded", _padded_shape(tuple(video_x.shape), tuple(inner.patch_size))),
        ("audio_shape", tuple(int(v) for v in audio_x.shape)),
        ("text_length", int(context.shape[1])),
        ("hidden_width", int(inner.hidden_size)),
        ("target_audio_rows", ab - aa),
        ("target_video_rows", vb - va),
        ("segments", segments),
        ("patch_size", tuple(int(v) for v in inner.patch_size)),
        ("refs", refs),
        ("keyframes", tuple(kf.get("resolved_frame_index") for kf in (payload.get("keyframes") or ()))),
        ("sigma_shifts", (shift_v, shift_a)),
        ("adaln_curves", bool(inner.use_adaln_curves)),
    )


@dataclass(slots=True)
class _OutputState:
    layout: Any
    t_emb: torch.Tensor
    video_timestep_row: int | torch.Tensor
    audio_timestep_row: int | torch.Tensor
    sigma_v: torch.Tensor
    sample_sigmas: torch.Tensor | None
    shift_v: float
    shift_a: float
    original_video_shape: tuple[int, int, int]
    padded_video_shape: tuple[int, int, int]


def _prepare_output_state(
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any],
    payload: dict[str, Any],
    layout: Any,
    denoise_mask: torch.Tensor | None = None,
    audio_denoise_mask: torch.Tensor | None = None,
) -> _OutputState:
    import comfy.model_management

    module = _native_module(inner)
    device = video_x.device
    dtype = context.dtype
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", inner.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", inner.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - module.time_shift_sigma(sigma_v, shift_v, shift_a))
    visual_aug = float(payload.get("visual_cond_noise_aug", getattr(module, "VISUAL_COND_TIMESTEP", 0.999)))
    audio_aug = float(payload.get("audio_cond_noise_aug", getattr(module, "AUDIO_COND_TIMESTEP", 1.0)))
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, visual_aug),
        "ref_img": max(t_v, visual_aug),
        "cond_audio": max(t_a, audio_aug),
        "ref_audio": max(t_a, audio_aug),
    }

    padded_t, padded_h, padded_w = _padded_shape(tuple(video_x.shape), tuple(inner.patch_size))
    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        mask_row_values = getattr(module, "mask_row_values", None)
        if not callable(mask_row_values):
            raise RuntimeError(
                "native MiniMax H3 module does not expose mask_row_values for per-token VIDEO masks"
            )
        mask_rows = mask_row_values(
            denoise_mask[0, 0].to(torch.float32),
            padded_t,
            padded_h,
            padded_w,
        )
        if mask_rows is not None:
            rows_t = (1.0 - mask_rows * sigma_v.to(mask_rows.device)).clamp(
                max=max(t_v, getattr(module, "VISUAL_COND_TIMESTEP", 0.999))
            )
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t

    if audio_denoise_mask is not None:
        mask_rows = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        expected_audio_rows = int(audio_x.shape[-1]) * int(audio_x.shape[-2])
        if int(mask_rows.numel()) != expected_audio_rows:
            raise RuntimeError(
                "native MiniMax H3 audio denoise mask row count does not match target AUDIO rows: "
                f"expected {expected_audio_rows}, got {int(mask_rows.numel())}"
            )
        if not bool((mask_rows >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - mask_rows * sigma_a).clamp(
                max=max(t_a, getattr(module, "AUDIO_COND_TIMESTEP", 1.0))
            )
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    unique_t = sorted(
        {t_v, t_a}
        | {seg_t[kind] for _, _, kind in layout.segments}
        | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
        | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set())
    )
    timestep_row = {value: index for index, value in enumerate(unique_t)}

    def rows_to_timestep_index(rows_t: torch.Tensor) -> torch.Tensor:
        levels = rows_t.unique()
        base = torch.tensor(
            [timestep_row[value] for value in levels.tolist()],
            dtype=torch.long,
            device=device,
        )
        positions = torch.searchsorted(levels, rows_t).to(device=device)
        return base[positions]

    video_timestep_row: int | torch.Tensor
    audio_timestep_row: int | torch.Tensor
    if video_rows_t is None:
        video_timestep_row = timestep_row[seg_t["video"]]
    else:
        video_timestep_row = rows_to_timestep_index(video_rows_t)
    if audio_rows_t is None:
        audio_timestep_row = timestep_row[seg_t["audio"]]
    else:
        audio_timestep_row = rows_to_timestep_index(audio_rows_t)

    values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if inner.use_adaln_curves:
        table = comfy.model_management.cast_to(inner.adaln_t_table, device=device)
        position = values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    else:
        t_emb = inner.time_embedder(values).to(dtype)
    return _OutputState(
        layout=layout,
        t_emb=t_emb,
        video_timestep_row=video_timestep_row,
        audio_timestep_row=audio_timestep_row,
        sigma_v=sigma_v,
        sample_sigmas=transformer_options.get("sample_sigmas"),
        shift_v=shift_v,
        shift_a=shift_a,
        original_video_shape=tuple(int(v) for v in video_x.shape[-3:]),
        padded_video_shape=(padded_t, padded_h, padded_w),
    )


def _sanitize_prediction(feature: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if not dtype.is_floating_point:
        return None, {"reason": "target dtype is not floating point"}
    if not feature.dtype.is_floating_point:
        return None, {"reason": "forecast dtype is not floating point"}

    # Spectrum predicts directly in the native H3/context dtype. Keep that
    # full forecast in BF16/FP16 and validate it in bounded chunks instead of
    # materializing a second full-size FP32 tensor plus full-size masks.
    #
    # A finite value already stored in the target dtype is necessarily within
    # that dtype's representable range, so same-dtype forecasts only need a
    # finiteness check. Repair is in-place and only touches chunks containing
    # NaN/Inf values.
    if feature.dtype == dtype:
        flat = feature.reshape(-1)
        element_size = max(1, feature.element_size())
        chunk_elements = max(1, _SANITIZE_CHUNK_BYTES // element_size)
        finite_values = 0
        nonfinite = 0
        bad_chunks: list[tuple[int, int]] = []

        for start in range(0, flat.numel(), chunk_elements):
            stop = min(flat.numel(), start + chunk_elements)
            chunk = flat[start:stop]
            finite = torch.isfinite(chunk)
            if bool(finite.all().item()):
                finite_values += chunk.numel()
            else:
                finite_count = int(finite.sum().item())
                finite_values += finite_count
                nonfinite += chunk.numel() - finite_count
                bad_chunks.append((start, stop))

        if finite_values == 0:
            return None, {"reason": "forecast contains no finite values"}
        if nonfinite == 0:
            return feature, None

        finfo = torch.finfo(dtype)
        for start, stop in bad_chunks:
            torch.nan_to_num_(
                flat[start:stop],
                nan=0.0,
                posinf=finfo.max,
                neginf=finfo.min,
            )
        return feature, {"nonfinite": nonfinite, "below": 0, "above": 0}

    # Compatibility fallback for an unexpected dtype mismatch. The normal H3
    # forecast path above avoids this full-size FP32 conversion entirely.
    fp32 = feature.to(torch.float32)
    finite = torch.isfinite(fp32)
    if not bool(finite.any().item()):
        return None, {"reason": "forecast contains no finite values"}
    info = None
    finfo = torch.finfo(dtype)
    if not bool(finite.all().item()) or bool(((fp32 < finfo.min) | (fp32 > finfo.max)).any().item()):
        info = {
            "nonfinite": int((~finite).sum().item()),
            "below": int((fp32 < finfo.min).sum().item()),
            "above": int((fp32 > finfo.max).sum().item()),
        }
    sanitized = torch.nan_to_num(fp32, nan=0.0, posinf=finfo.max, neginf=finfo.min)
    return sanitized.clamp_(min=finfo.min, max=finfo.max).to(dtype), info


def _execute_actual(
    executor,
    inner: Any,
    runtime: SpectrumH3Runtime,
    run_id: int,
    step_id: int,
    call_id: int,
    layout: Any,
    x,
    timestep,
    context,
    transformer_options,
    minimax_payload,
    kwargs,
    residual_probe=None,
):
    if len(inner.blocks) == 0:
        raise RuntimeError("native MiniMax H3 has no transformer blocks to observe")
    last_index = len(inner.blocks) - 1
    local_options = dict(transformer_options)
    patches_replace = dict(local_options.get("patches_replace") or {})
    dit_replacements = dict(patches_replace.get("dit") or {})
    patches_replace["dit"] = dit_replacements
    local_options["patches_replace"] = patches_replace
    existing = dit_replacements.get(("double_block", last_index))
    first_index = 0
    existing_first = dit_replacements.get(("double_block", first_index))
    observed = False
    actual_target = None
    state_input_target = None

    if runtime.active_state_conditioned_residual and first_index != last_index:
        def capture_state_input(args, replacement_context):
            nonlocal state_input_target
            hidden = args.get("img")
            (aa, _), (_, vb) = target_segments(layout)
            if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[0] < vb:
                raise RuntimeError(
                    "initial MiniMax H3 hidden feature is incompatible with state-conditioned residual forecasting"
                )
            state_input_target = hidden[aa:vb].detach().clone(
                memory_format=torch.contiguous_format
            )
            return (
                existing_first(args, replacement_context)
                if existing_first is not None
                else replacement_context["original_block"](args)
            )

        dit_replacements[("double_block", first_index)] = capture_state_input

    def capture_replacement(args, replacement_context):
        nonlocal actual_target, observed, state_input_target, residual_probe
        output = existing(args, replacement_context) if existing is not None else replacement_context["original_block"](args)
        if not isinstance(output, dict) or "img" not in output or not torch.is_tensor(output["img"]):
            raise RuntimeError("final MiniMax H3 block replacement did not return {'img': tensor}")
        (aa, _), (_, vb) = target_segments(layout)
        hidden = output["img"]
        if hidden.ndim != 2 or hidden.shape[0] < vb:
            raise RuntimeError("final MiniMax H3 hidden feature is incompatible with the packed layout")
        # Native H3 guarantees contiguous [audio | video] target rows at the
        # packed tail. Keep a view here; materializing torch.cat would create a
        # second full target tensor on the GPU before the required CPU archive.
        target = hidden[aa:vb].unsqueeze(0)
        from .backend_history import observe
        resets_before = runtime.stats.backend_history_resets
        observe(runtime, run_id, step_id, local_options,
                local_options.get("attention_backend_preflight_v1"))
        if runtime.stats.backend_history_resets != resets_before:
            residual_probe = None  # discard old-backend shadow/hold evidence
        # The target is a view into the final packed hidden state. Keeping it in
        # this closure pins the entire final-hidden CUDA storage, so retain it
        # only for the optional residual-probe comparison that consumes it
        # immediately after the transformer returns.
        actual_target = target if residual_probe is not None else None
        if runtime.active_state_conditioned_residual:
            try:
                if state_input_target is None:
                    # This only applies to an unexpected one-block native contract.
                    history_target = target.detach().clone(
                        memory_format=torch.contiguous_format
                    )
                    _apply_exact_state_input_embedding_(
                        history_target,
                        inner,
                        x[0],
                        x[1],
                        scale=-1.0,
                    )
                else:
                    # Native H3 updates the packed hidden state in-place. Capture
                    # the exact pre-block target once, then turn that owned buffer
                    # into final_hidden - input_hidden at the last block.
                    state_input_target.neg_().add_(target[0])
                    history_target = state_input_target.unsqueeze(0)
                    state_input_target = None
                runtime.observe_actual(
                    run_id,
                    step_id,
                    call_id,
                    history_target,
                    take_ownership=True,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                runtime.fallback_current_step(
                    run_id,
                    step_id,
                    f"state-conditioned residual actual transform failed: {exc}",
                )
                runtime.observe_actual(run_id, step_id, call_id, target)
        else:
            runtime.observe_actual(run_id, step_id, call_id, target)
        observed = True
        return output

    dit_replacements[("double_block", last_index)] = capture_replacement
    try:
        result = executor(
            x,
            timestep,
            context,
            local_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    except Exception:
        actual_target = None
        raise
    if not observed:
        raise RuntimeError("native MiniMax H3 final transformer block was not executed")
    if residual_probe is not None and actual_target is not None:
        try:
            state = _prepare_output_state(
                inner,
                x[0],
                x[1],
                timestep,
                context,
                transformer_options,
                minimax_payload or {},
                layout,
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
            )
            output_head_started = time.perf_counter()
            try:
                residual_state_scale = (
                    1.0 if runtime.active_state_conditioned_residual else 0.0
                )
                shadow_output = _execute_forecast(
                    inner, residual_probe.shadow, state, x[0], x[1],
                    state_embedding_scale=residual_state_scale,
                )
                hold_output = _execute_forecast(
                    inner, residual_probe.hold, state, x[0], x[1],
                    state_embedding_scale=residual_state_scale,
                )
            finally:
                runtime.record_residual_output_head_seconds(
                    time.perf_counter() - output_head_started
                )
            runtime.record_residual_measurement(
                run_id,
                step_id,
                call_id,
                residual_probe,
                actual_feature=actual_target,
                actual_output=result,
                shadow_output=shadow_output,
                hold_output=hold_output,
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            runtime.disable_experiment(f"residual output-head evaluation failed: {exc}")
        finally:
            # Break the replacement-closure reference immediately. Otherwise a
            # retained patches_replace callback can pin the full final-hidden
            # CUDA storage across later sampler steps.
            actual_target = None
    return result


def _final_layer_uses_pdd_contract(module: Any) -> bool:
    """Detect the native Core FinalLayer contract without probing by execution."""
    final_layer_type = getattr(module, "FinalLayer", None)
    forward = getattr(final_layer_type, "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("unable to inspect native MiniMax H3 FinalLayer.forward contract") from exc
    pdd_parameters = ("sigma", "sample_sigmas", "shifts")
    present = tuple(name in parameters for name in pdd_parameters)
    if all(present):
        return True
    if any(present):
        raise RuntimeError(
            "native MiniMax H3 FinalLayer exposes an incomplete PDD projection contract"
        )
    return False


def _final_layer_project(
    inner: Any,
    module: Any,
    compact: torch.Tensor,
    state: _OutputState,
    video_segment: tuple[int, int, Any],
    audio_segment: tuple[int, int, Any],
    *,
    forward_override: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward = forward_override if forward_override is not None else inner.final_layer
    if _final_layer_uses_pdd_contract(module):
        return forward(
            compact,
            state.t_emb,
            video_segment,
            audio_segment,
            sigma=state.sigma_v,
            sample_sigmas=state.sample_sigmas,
            shifts=(state.shift_v, state.shift_a),
        )
    return forward(
        compact,
        state.t_emb,
        video_segment,
        audio_segment,
    )


def _callable_marker(callable_obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(callable_obj, name, None)
    if value is None:
        value = getattr(getattr(callable_obj, "__func__", None), name, None)
    return default if value is None else value


def _resolve_final_layer_stream_compat(
    inner: Any,
    video_rows: int,
) -> tuple[Any | None, Any | None, bool, int | None]:
    """Recover H3-Optimizations' original FinalLayer for cube-order streaming."""
    forward = inner.final_layer.forward
    if not bool(_callable_marker(forward, "_h3_optimizations_final_layer", False)):
        return None, None, False, None

    signature = _callable_marker(
        forward,
        "_h3_optimizations_final_layer_signature",
        None,
    )
    chunk_rows = None if signature is None else int(signature)
    cube_state = _callable_marker(
        forward,
        "_h3_optimizations_cube_order_state",
        None,
    )
    if cube_state is None:
        # FinalLayer-memory-only patching accepts arbitrary input sizes, so keep
        # the installed wrapper and let it further chunk Spectrum's small slab.
        return None, None, False, chunk_rows

    original = _callable_marker(
        forward,
        "_h3_optimizations_final_layer_original",
        None,
    )
    if not callable(original):
        raise RuntimeError(
            "H3-Optimizations FinalLayer cube-order wrapper has no recoverable original"
        )
    topology, active = cube_state.resolve(int(video_rows))
    if len(topology.forward) != int(video_rows):
        raise RuntimeError("H3 cube-order topology does not match Spectrum video rows")
    return original, topology, not bool(active), chunk_rows


def _slice_mod_selector(
    selector: Any,
    start: int,
    stop: int,
    stream_rows: int,
    *,
    topology: Any | None = None,
) -> Any:
    """Slice per-token modulation selectors while preserving scalar selectors."""
    if (
        torch.is_tensor(selector)
        and selector.ndim > 0
        and int(selector.shape[0]) == int(stream_rows)
    ):
        if topology is not None:
            index = torch.tensor(
                topology.forward[int(start) : int(stop)],
                dtype=torch.long,
                device=selector.device,
            )
            return selector.index_select(0, index)
        return selector[int(start) : int(stop)]
    return selector


def _prepare_exact_state_rows_cpu(
    inner: Any,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact H3 input rows in system RAM for streamed residual reconstruction."""
    module = _native_module(inner)
    common_dit = importlib.import_module("comfy.ldm.common_dit")
    video_cpu = video_x.detach().to(device="cpu", dtype=torch.float32).contiguous()
    audio_cpu = audio_x.detach().to(device="cpu", dtype=torch.float32).contiguous()
    padded_video = common_dit.pad_to_patch_size(video_cpu, tuple(inner.patch_size))
    video_rows = module.patchify_video(
        padded_video,
        tuple(inner.patch_size),
    ).contiguous()
    audio_rows = module.pack_audio(audio_cpu).contiguous()
    if audio_rows.ndim != 2 or video_rows.ndim != 2:
        raise ValueError("native H3 patch helpers returned an unexpected row layout")
    return audio_rows, video_rows


def _copy_cpu_chunk_to_workspace_(
    workspace: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Copy into reusable CUDA storage; pageable CPU sources stay synchronous."""
    count = int(source.shape[0])
    if count > int(workspace.shape[0]):
        raise RuntimeError("Spectrum streaming workspace is smaller than its chunk")
    target = workspace[:count]
    target.copy_(source, non_blocking=bool(source.is_pinned()))
    return target


def _project_cpu_forecast_stream(
    inner: Any,
    module: Any,
    compact_cpu: torch.Tensor,
    state: _OutputState,
    *,
    global_start: int,
    stream_rows: int,
    selector: Any,
    video: bool,
    device: torch.device,
    forward_override: Any | None = None,
    topology: Any | None = None,
    reorder_selector: bool = False,
    chunk_rows_override: int | None = None,
    state_rows_cpu: torch.Tensor | None = None,
    state_projection: Any | None = None,
    state_scale: float = 0.0,
    hidden_workspace: torch.Tensor | None = None,
    state_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project one forecast stream without materializing its hidden state on CUDA."""
    rows = int(stream_rows)
    hidden = int(inner.hidden_size)
    if rows <= 0:
        head = inner.final_layer.video_out if video else inner.final_layer.audio_out
        return torch.empty(
            (0, int(head.out_features)),
            device=device,
            dtype=torch.float32,
        )

    bytes_per_row = max(1, hidden * compact_cpu.element_size())
    chunk_rows = max(1, _FORECAST_HEAD_CHUNK_BYTES // bytes_per_row)
    if chunk_rows_override is not None and int(chunk_rows_override) > 0:
        chunk_rows = min(chunk_rows, int(chunk_rows_override))
    if state_rows_cpu is not None and int(state_rows_cpu.shape[0]) != rows:
        raise ValueError("streamed exact-state rows do not match the target stream")
    if hidden_workspace is not None:
        if (
            hidden_workspace.device != device
            or hidden_workspace.dtype != compact_cpu.dtype
            or int(hidden_workspace.shape[1]) != hidden
        ):
            raise ValueError("Spectrum hidden streaming workspace is incompatible")
        chunk_rows = min(chunk_rows, int(hidden_workspace.shape[0]))
    projected = None

    for local_start in range(0, rows, chunk_rows):
        local_stop = min(rows, local_start + chunk_rows)
        count = local_stop - local_start
        hidden_source = compact_cpu[
            int(global_start) + local_start : int(global_start) + local_stop
        ]
        if hidden_workspace is None:
            hidden_chunk = hidden_source.to(
                device=device,
                dtype=compact_cpu.dtype,
                non_blocking=bool(hidden_source.is_pinned()),
            )
        else:
            hidden_chunk = _copy_cpu_chunk_to_workspace_(
                hidden_workspace,
                hidden_source,
            )
        if state_rows_cpu is not None:
            if state_projection is None:
                raise RuntimeError("streamed exact-state rows have no patch projection")
            if topology is None:
                state_chunk_cpu = state_rows_cpu[local_start:local_stop]
            else:
                state_index = torch.tensor(
                    topology.forward[local_start:local_stop],
                    dtype=torch.long,
                    device=state_rows_cpu.device,
                )
                state_chunk_cpu = state_rows_cpu.index_select(0, state_index)
            if state_workspace is None:
                state_chunk = state_chunk_cpu.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=bool(state_chunk_cpu.is_pinned()),
                )
            else:
                if (
                    state_workspace.device != device
                    or state_workspace.dtype != torch.float32
                    or int(state_workspace.shape[1]) != int(state_chunk_cpu.shape[1])
                ):
                    raise ValueError(
                        "Spectrum exact-state streaming workspace is incompatible"
                    )
                state_chunk = _copy_cpu_chunk_to_workspace_(
                    state_workspace,
                    state_chunk_cpu,
                )
            embedded = state_projection(state_chunk).to(hidden_chunk.dtype)
            hidden_chunk.add_(embedded, alpha=float(state_scale))
            del state_chunk, embedded

        row_selector = _slice_mod_selector(
            selector,
            local_start,
            local_stop,
            rows,
            topology=(topology if reorder_selector else None),
        )
        # FinalLayer is row-local. Give the inactive stream an empty range with
        # a scalar selector so no full-stream modulation tensor is broadcast
        # into the empty slice.
        if video:
            video_segment = (0, count, row_selector)
            audio_segment = (count, count, 0)
        else:
            video_segment = (count, count, 0)
            audio_segment = (0, count, row_selector)

        video_chunk, audio_chunk = _final_layer_project(
            inner,
            module,
            hidden_chunk,
            state,
            video_segment,
            audio_segment,
            forward_override=forward_override,
        )
        selected = video_chunk if video else audio_chunk
        if projected is None:
            projected = selected.new_empty((rows, *selected.shape[1:]))
        if video and topology is not None:
            # topology.forward maps cube-major row -> raster row. Restore each
            # projected slab directly, avoiding another full output allocation.
            restore_index = torch.tensor(
                topology.forward[local_start:local_stop],
                dtype=torch.long,
                device=selected.device,
            )
            projected.index_copy_(0, restore_index, selected)
        else:
            projected[local_start:local_stop].copy_(selected)
        del hidden_source, hidden_chunk, video_chunk, audio_chunk, selected

    if projected is None:
        raise RuntimeError("streamed Spectrum FinalLayer projection produced no rows")
    return projected


def _execute_forecast(
    inner: Any,
    predicted: torch.Tensor,
    state: _OutputState,
    video_x: torch.Tensor,
    audio_x: torch.Tensor,
    *,
    state_embedding_scale: float = 0.0,
):
    module = _native_module(inner)
    (aa, ab), (va, vb) = target_segments(state.layout)
    audio_rows = ab - aa
    video_rows = vb - va
    compact = predicted[0]
    if compact.shape != (audio_rows + video_rows, inner.hidden_size):
        raise RuntimeError("forecasted MiniMax H3 target feature has an invalid compact shape")

    # Keep the complete hidden forecast in system RAM and transfer only bounded
    # row slabs. This path also handles offline replay, exact-state residual
    # reconstruction and H3-Optimizations cube-order FinalLayer compatibility.
    if compact.device.type == "cpu" and video_x.device.type == "cuda":
        forward_override, video_topology, reorder_selector, h3_chunk_rows = (
            _resolve_final_layer_stream_compat(inner, video_rows)
        )
        state_audio_rows = None
        state_video_rows = None
        if float(state_embedding_scale) != 0.0:
            state_audio_rows, state_video_rows = _prepare_exact_state_rows_cpu(
                inner,
                video_x,
                audio_x,
            )
            if (
                int(state_audio_rows.shape[0]) != audio_rows
                or int(state_video_rows.shape[0]) != video_rows
            ):
                raise ValueError(
                    "state-conditioned residual input rows do not match forecast topology"
                )

        # Allocate the CUDA staging tensors once for this forecast and reuse
        # them for every audio/video slab. Keeping them local prevents Spectrum
        # from reserving VRAM across the following actual transformer call.
        bytes_per_hidden_row = max(1, int(inner.hidden_size) * compact.element_size())
        workspace_rows = max(
            1,
            _FORECAST_HEAD_CHUNK_BYTES // bytes_per_hidden_row,
        )
        if h3_chunk_rows is not None and int(h3_chunk_rows) > 0:
            workspace_rows = min(workspace_rows, int(h3_chunk_rows))
        workspace_rows = min(workspace_rows, max(audio_rows, video_rows))
        hidden_workspace = torch.empty(
            (workspace_rows, int(inner.hidden_size)),
            device=video_x.device,
            dtype=compact.dtype,
        )
        audio_state_workspace = None
        video_state_workspace = None
        if state_audio_rows is not None:
            audio_state_workspace = torch.empty(
                (min(workspace_rows, audio_rows), int(state_audio_rows.shape[1])),
                device=video_x.device,
                dtype=torch.float32,
            )
            video_state_workspace = torch.empty(
                (min(workspace_rows, video_rows), int(state_video_rows.shape[1])),
                device=video_x.device,
                dtype=torch.float32,
            )

        audio_projected = _project_cpu_forecast_stream(
            inner,
            module,
            compact,
            state,
            global_start=0,
            stream_rows=audio_rows,
            selector=state.audio_timestep_row,
            video=False,
            device=video_x.device,
            forward_override=forward_override,
            chunk_rows_override=h3_chunk_rows,
            state_rows_cpu=state_audio_rows,
            state_projection=inner.audio_patch_proj,
            state_scale=state_embedding_scale,
            hidden_workspace=hidden_workspace,
            state_workspace=audio_state_workspace,
        )
        video_projected = _project_cpu_forecast_stream(
            inner,
            module,
            compact,
            state,
            global_start=audio_rows,
            stream_rows=video_rows,
            selector=state.video_timestep_row,
            video=True,
            device=video_x.device,
            forward_override=forward_override,
            topology=video_topology,
            reorder_selector=reorder_selector,
            chunk_rows_override=h3_chunk_rows,
            state_rows_cpu=state_video_rows,
            state_projection=inner.video_patch_proj,
            state_scale=state_embedding_scale,
            hidden_workspace=hidden_workspace,
            state_workspace=video_state_workspace,
        )
        del hidden_workspace
        del audio_state_workspace, video_state_workspace
        del state_audio_rows, state_video_rows
    else:
        if float(state_embedding_scale) != 0.0:
            _apply_exact_state_input_embedding_(
                predicted,
                inner,
                video_x,
                audio_x,
                scale=float(state_embedding_scale),
            )
        audio_segment = (0, audio_rows, state.audio_timestep_row)
        video_segment = (audio_rows, audio_rows + video_rows, state.video_timestep_row)
        video_projected, audio_projected = _final_layer_project(
            inner,
            module,
            compact,
            state,
            video_segment,
            audio_segment,
        )

    latent_t, latent_h, latent_w = state.padded_video_shape
    video_out = module.unpatchify_video(
        video_projected,
        latent_t,
        latent_h // inner.patch_size[1],
        latent_w // inner.patch_size[2],
        inner.latents_dim,
        inner.patch_size,
    )
    original_t, original_h, original_w = state.original_video_shape
    video_out = video_out[:, :, :original_t, :original_h, :original_w]
    audio_out = -module.unpack_audio(audio_projected).to(audio_x.dtype)
    # Older cores return the audio velocity scaled by d(sigma_a)/d(sigma_v) from
    # _forward; newer ones carry the audio on the video schedule and undo the
    # scale in forward(), outside this wrapper, so _forward stays unscaled.
    slope_fn = getattr(module, "time_shift_slope", None)
    if slope_fn is not None:
        audio_out = audio_out * slope_fn(state.sigma_v, state.shift_v, state.shift_a).to(audio_out.dtype)
    return [-video_out.to(video_x.dtype), audio_out]


def diffusion_model_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    runtime = options.get(RUNTIME_KEY)
    run_id = options.get(RUN_ID_KEY)
    step_id = options.get(STEP_ID_KEY)
    if not isinstance(runtime, SpectrumH3Runtime) or run_id is None or step_id is None:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    inner = executor.class_obj
    if not is_native_minimax_h3(inner):
        runtime.fallback_current_step(
            int(run_id),
            int(step_id),
            "diffusion model is not ComfyUI's native MiniMax H3",
        )
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        runtime.fallback_current_step(int(run_id), int(step_id), "native H3 input is not a video/audio latent pair")
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    video_x, audio_x = x
    if video_x.shape[0] != 1 or audio_x.shape[0] != 1:
        runtime.fallback_current_step(int(run_id), int(step_id), "native MiniMax H3 batch size is not one")
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    payload = minimax_payload or {}
    layout = _resolve_layout(inner, context, video_x, audio_x, payload)
    (aa, ab), (va, vb) = target_segments(layout)
    labels = branch_labels(options)
    expected_shape = (1, (ab - aa) + (vb - va), int(inner.hidden_size))
    topology = topology_signature(inner, video_x, audio_x, context, layout, options, payload)
    from .backend_history import prepare
    options, backend_policy = prepare(runtime, int(run_id), int(step_id), options, layout, inner)
    if backend_policy is not None:
        options["attention_backend_preflight_v1"] = backend_policy
    call_id, actual = runtime.begin_model_call(
        int(run_id),
        int(step_id),
        topology=topology,
        labels=labels,
        expected_shape=expected_shape,
    )
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 model call run_id=%s step=%s call=%s path=%s target_audio=%s target_video=%s topology=%s",
            run_id,
            step_id,
            call_id,
            "actual" if actual else "forecast",
            ab - aa,
            vb - va,
            topology,
        )

    if actual:
        residual_prediction_device = (
            torch.device("cpu")
            if video_x.device.type == "cuda"
            else video_x.device
        )
        residual_probe = runtime.prepare_residual_probe(
            int(run_id),
            int(step_id),
            call_id,
            device=residual_prediction_device,
            dtype=context.dtype,
        )
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
            residual_probe,
        )

    if kwargs.get("denoise_mask") is not None:
        module = _native_module(inner)
        if not callable(getattr(module, "mask_row_values", None)):
            reason = "native MiniMax H3 core lacks mask_row_values for masked forecasting"
            runtime.fallback_current_step(int(run_id), int(step_id), reason)
            return _execute_actual(
                executor,
                inner,
                runtime,
                int(run_id),
                int(step_id),
                call_id,
                layout,
                x,
                timestep,
                context,
                options,
                minimax_payload,
                kwargs,
            )

    # Keep every complete Spectrum prediction out of CUDA: causal forecasts,
    # state-conditioned residuals, offline replay and bootstrap/model-aware
    # forecasts all use the same bounded output-head streamer below.
    prediction_device = (
        torch.device("cpu") if video_x.device.type == "cuda" else video_x.device
    )

    predicted = runtime.predict(
        int(run_id),
        int(step_id),
        call_id,
        device=prediction_device,
        dtype=context.dtype,
    )
    if predicted is None:
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
        )

    sanitized, event = _sanitize_prediction(predicted, context.dtype)
    if sanitized is None:
        runtime.fallback_current_step(int(run_id), int(step_id), event["reason"] if event else "forecast sanitization failed")
        return _execute_actual(
            executor,
            inner,
            runtime,
            int(run_id),
            int(step_id),
            call_id,
            layout,
            x,
            timestep,
            context,
            options,
            minimax_payload,
            kwargs,
        )
    if event is not None and runtime.config.debug:
        LOG.warning("Spectrum H3 forecast sanitized run_id=%s step=%s event=%s", run_id, step_id, event)
    try:
        state = _prepare_output_state(
            inner,
            video_x,
            audio_x,
            timestep,
            context,
            options,
            payload,
            layout,
            denoise_mask=kwargs.get("denoise_mask"),
            audio_denoise_mask=kwargs.get("audio_denoise_mask"),
        )
        output = _execute_forecast(
            inner,
            sanitized,
            state,
            video_x,
            audio_x,
            state_embedding_scale=(
                1.0 if runtime.active_state_conditioned_residual else 0.0
            ),
        )
        del state
        del sanitized
        del predicted
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        if runtime.offline_phase == "replay":
            raise OfflineReplayAbort(f"offline replay output-head evaluation failed: {exc}") from exc
        raise
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 forecast complete run_id=%s step=%s stage=%s geometry=%s "
            "chunks=%s history=%s",
            run_id,
            step_id,
            runtime.active_stage_index,
            "state_conditioned_residual" if runtime.active_state_conditioned_residual else "absolute_hidden",
            runtime.last_prediction_chunk_count,
            runtime.prediction_history_length,
        )
    return output


def install_h3_wrapper(model: Any) -> None:
    import comfy.patcher_extension

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    if not model.get_wrappers(wrapper_type, WRAPPER_KEY):
        model.add_wrapper_with_key(wrapper_type, WRAPPER_KEY, diffusion_model_wrapper)
