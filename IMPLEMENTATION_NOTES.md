# MiniMax H3 integration notes

Source review date: 2026-08-07

Reviewed native ComfyUI commits: `e377e263049f9338b4d12a3dd417b36ae62948ff` and `0dd9b154a1654fc699dcdc3af066c7cce096045a`

Reviewed Spectrum paper: arXiv `2603.01623v1`

Reviewed official Spectrum implementation commit: `11f317a87352e2c67daa2fac5a971cf04233d7d1`

The separate `ComfyUI-Spectrum-Proper` repository was inspected only for ComfyUI wrapper and clone-lifetime lessons. This repository does not import it, depend on it, or share code or runtime state with it.

## Native execution path

1. `comfy.samplers.CFGGuider.sample` packs the nested MiniMax H3 video/audio latent pair for sampler integration.
2. `CFGGuider.outer_sample` prepares the cloned `ModelPatcher`, then `CFGGuider.inner_sample` copies model options and writes the supplied sigma sequence to `transformer_options["sample_sigmas"]`.
3. A stock k-diffusion sampler calls `KSamplerX0Inpaint`, then `CFGGuider.outer_predict_noise` invokes `PREDICT_NOISE` wrappers around `CFGGuider.predict_noise`.
4. `sampling_function` and `calc_cond_batch` form conditional/unconditional model calls. Each call receives copied/merged transformer options containing `cond_or_uncond`, `uuids`, the current `sigmas`, the full `sample_sigmas`, user patches, replacement patches, hooks, and model-management data.
5. `comfy.model_base.MiniMaxH3._apply_model` converts the flat sampler latent back to the native pair `[video, audio]`, maps sampler sigma through `model_sampling.timestep`, copies transformer options, and calls `MiniMaxH3Model.forward`.
6. `MiniMaxH3Model.forward` invokes ComfyUI `DIFFUSION_MODEL` wrappers around `MiniMaxH3Model._forward`. On cores with `ModelSamplingAV` the sampler carries the audio stream scaled onto the video schedule; `forward` removes that scale before the wrappers run and restores it after, so wrappers always observe the stream's own latent and velocity.
7. `_forward` pads video to the model patch geometry, resolves or constructs `PackedLayout`, derives video and audio timesteps from the current video sigma and both sigma shifts, builds modality/timestep modulation segments, embeds target and conditioning rows, builds RoPE, and runs every transformer block. Per-block replacement patches and the native prefetch queue are applied inside this loop.
8. The packed sequence is `[text | optional keyframe/reference segments | target audio | target video]`. `PackedLayout.segments` proves that the target audio and target video spans are the final two contiguous segments, in that order.
9. Immediately after the final transformer block, the packed hidden tensor contains the desired forecast target. Only the target audio and target video rows are cached, as a compact tensor ordered `[target audio rows | target video rows]`. Text, keyframe, image-reference, video-reference, and audio-reference rows are excluded.
10. `FinalLayer.forward` consumes the current hidden rows plus the current exact timestep embeddings. It independently normalizes/modulates target video and target audio rows, then executes the checkpoint's FP32 output heads.
11. Native reconstruction unpatchifies video and unpacks stereo channel-major audio. The audio velocity convention is core-dependent and is detected from the presence of `time_shift_slope` in `comfy.ldm.minimax.model`: cores that expose it expect `_forward` to apply the video-to-audio sigma-map derivative and return `[-video_velocity, -audio_slope * audio_velocity]`; cores that removed it perform the schedule conversion in `forward` instead, so `_forward` returns the unscaled `[-video_velocity, -audio_velocity]`. `BaseModel._apply_model` packs this native pair again for the sampler and applies the native denoised conversion.

## Integration invariant

The acceleration boundary is the post-final-block hidden feature immediately before `FinalLayer`. Actual steps run the native `_forward` unchanged, with a call-local wrapper around the existing final-block replacement solely to observe its output. Forecast steps compute only current layout and output-head timestep state, predict the compact target feature, remap its audio/video segments to the compact tensor, then call the native final layer and native reconstruction helpers.

This preserves:

- current video and audio timestep conditioning;
- sigma-shift mapping, and whichever audio velocity convention the installed core uses;
- output-head weights and FP32 islands;
- video unpatchification and audio unpacking;
- native return structure;
- existing transformer replacement patches on actual steps;
- native prefetch/model-management behavior on actual steps;
- clone-local runtime selection through call-local transformer options.

## Sampler contract

Solver-step IDs are assigned by a `PREDICT_NOISE` wrapper inside an `OUTER_SAMPLE` run transaction. Forecasting is allowlisted only for deterministic native `sample_euler`, `sample_res_multistep`, and `sample_res_multistep_cfg_pp`, whose reviewed implementations perform exactly one `predict_noise` call per solver iteration. Euler requires one completed actual H3 evaluation after every forecast. RES stores each current denoised result for the following second-order update. The actual step after a forecast consumes that forecast-derived value, then replaces `old_denoised` with its native result; one completed actual evaluation therefore clears the retained forecast before forecasting resumes. RES also enforces a three-step actual tail. These sampler floors are applied at run time and cannot be weakened by an older saved workflow. Ancestral variants remain native because they inject noise between model evaluations. Other samplers remain native and report a debug fallback reason.

Coordinates are derived from the actual supplied sigma sequence. Evaluated sigma values are affinely normalized between the run's evaluated minimum and maximum into `[-1, 1]`; no fixed step count is assumed.

Native EasyCache and LazyCache may terminate a diffusion-model wrapper chain without an H3 call. Their shared `transformer_options["easycache"]` holder is therefore detected before Spectrum opens a run transaction. Spectrum remains inactive for that run and the cache owns the acceleration path.

## Forecast memory model

The forecaster stores at most `max_history` detached model-dtype snapshots in the configured history storage: system RAM by default, or the producing model device when the opt-in VRAM mode is selected. It solves only for history weights:

`w(t*) = phi(t*) (Phi^T Phi + lambda I)^-1 Phi^T`

Spectral and two-point linear weights are combined before feature streaming. Prediction reads one bounded chunk from one history snapshot at a time, accumulates that chunk in FP32 on the output device, and writes the final model-dtype feature. No persistent full-feature FP32 right-hand side or coefficient tensor is created.

Native H3 lays out target rows as one contiguous `[audio | video]` packed tail. Actual capture archives that tail without materializing an audio/video concatenation. System-RAM mode copies the view directly to CPU. VRAM mode must clone it into compact owned device storage, because retaining the view would pin the complete final-block hidden tensor. When one model call contains the complete canonical branch set, the archived tensor transfers directly into forecaster history; split conditional calls retain the transactional canonicalization path and assemble rows only after all calls complete. Debug summaries expose the storage location and wall-clock archive, history-update, and forecast-prediction counters. Device-to-host archiving can synchronize outstanding CUDA work, while device cloning can be asynchronously enqueued.
