# ComfyUI Spectrum MiniMax H3

Spectrum-style spectral feature forecasting for ComfyUI's native MiniMax H3 audio-video model.

This custom node reduces expensive H3 transformer evaluations during sampling. It fits a Chebyshev ridge model to actual post-transformer hidden features and forecasts those features on selected future solver steps. The current-step native MiniMax H3 output heads, video reconstruction, audio reconstruction, sigma mapping, and return structure still execute on every step.

This repository is independent from [ComfyUI-Spectrum-Proper](https://github.com/xmarre/ComfyUI-Spectrum-Proper), which remains a dedicated FLUX implementation.

## Current default and output fidelity

Spectrum is an approximate accelerator. Forecasted steps change the denoising trajectory, so its output is neither lossless nor bit-identical to native MiniMax H3, even with an otherwise identical prompt, seed, model, sampler, and workflow.

Since v0.2.1, the standard audio-fidelity path is:

```text
offline_smoothing_replay = true
blend_weight = 0.50
audio_blend_weight = 0.00
```

The first pass captures a causal trajectory with local-only prediction for both modalities. A transformer-free second pass then applies the configured video blend using past and future actual anchors. No later joint H3 transformer call can feed the replayed video change back into audio, and the zero audio weight prevents direct spectral mixing of audio rows.

This default was introduced after earlier single-pass Spectrum releases were found to reduce overall audio fidelity, clarity, naturalness, and stability. Reports included generated speech and reference-conditioned audio; speech tripping, doubled syllables, and stuttering were the clearest reproducible symptoms of the wider degradation. Matched runs on the affected seed isolated two paths: direct spectral audio blending and indirect video-to-audio feedback through later joint transformer evaluations. The current offline path produced clean audio on that seed while retaining the preferred image result. This validates the correction for the reproduced case; broader checkpoints, prompts, samplers, and reference inputs still need exact-seed validation.

> [!IMPORTANT]
> Workflows saved with v0.2.0 may retain `offline_smoothing_replay=false`. Turn it on once after updating. Workflows created before v0.2.0 did not store this input and receive the current default automatically. Keep `audio_blend_weight=0` unless intentionally testing spectral audio blending.

Visual and trajectory differences remain possible with the current default. Exact-seed A/B tests have shown changes in motion, pose, timing, gaze, and action paths, plus occasional malformed eyes, fingers, faces, limbs, or other articulated details during demanding motion. The cause and frequency of these visual changes have not been isolated across model precision, resolution, prompt complexity, sampler, early native-step count, bootstrap behavior, or total steps.

For quality-critical work, compare the same prompt and seed with Spectrum enabled and disabled. For an audio issue on v0.2.1 or newer, first verify `offline_smoothing_replay=true` and `audio_blend_weight=0`. Increasing `degree`, `warmup_steps`, or total steps changes the forecast schedule and remains an optional A/B experiment, not a requirement of the corrected default. Include the exact version, checkpoint, generation mode, prompt, seed, sampler, scheduler, resolution, duration, step count, node settings, and audio reference in new reports.

Use Spectrum when the speed benefit is worth possible output differences. Disable it when maximum fidelity to the native MiniMax H3 trajectory is required.

## Supported native path

The integration targets `comfy.ldm.minimax.model.MiniMaxH3Model` in native ComfyUI. It supports native text-to-video/audio (`t2va`), first/last-frame-to-video/audio (`fl2va`), and reference-to-video/audio (`ref2va`) workflows. It requires the MiniMax H3 and packed-latent sampler APIs introduced by ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff`, including the `latent_shapes` argument on `outer_sample`. Older ComfyUI revisions are unsupported. Native-equivalence coverage includes that original integration and ComfyUI commit `00d02f2854892ee5b9808bc2f6348b972017886a`, used for the v0.2.2 test run, including the `ModelSamplingAV` audio-schedule contract introduced in `bdcb886a4705a03cf40f4a7226de9fc7c059fc90` and reference-conditioned target forecasting. Required H3 attributes are checked when the node is applied, and replacement output shape is checked on actual steps so incompatible native changes fail with an explicit contract error.

The forecast target is the packed hidden feature immediately after the final H3 transformer block and before `FinalLayer`, ordered as:

```text
[target audio rows | target video rows]
```

Text rows and all keyframe/reference-only rows are excluded from history. Actual steps stay on the native `_forward` implementation. A call-local final-block replacement observes its returned hidden state while preserving an existing replacement patch. Forecast steps skip every transformer block, RoPE construction, conditioning projections, reference embedding, and per-block prefetch, then run the native final layer with freshly computed audio and video timestep embeddings.

Reference conditioning can add substantial preprocessing and end-to-end work outside the forecasted H3 transformer calls. If a native `ref2va` run appears unaffected, enable `debug`: a working Spectrum run reports nonzero `forecast_steps`; on a single-branch run, its actual transformer-call count also falls below the solver-step count. A zero-forecast run logs the exact sampler, cache, wrapper-bypass, label, topology, or prediction fallback reason. Similar-looking output alone does not indicate that Spectrum was inactive.

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git
```

Restart ComfyUI. The node appears under `sampling/spectrum` as **Spectrum Apply MiniMax H3**.

The node adds no third-party Python dependency. It uses PyTorch and ComfyUI modules already present in a normal ComfyUI installation.

### Updating

Use v0.2.1 or newer for the corrected default audio path. Use v0.2.2 or newer for live two-pass progress reporting. Update a Git clone with:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3
git pull --ff-only
```

Restart ComfyUI after updating. A workflow saved with v0.2.0 may still store `offline_smoothing_replay=false`; enable it once in that node.

## Workflow placement

Recommended order:

```text
MiniMax H3 model loader
-> LoRA and other model patches
-> MiniMax H3 Sigma Shift
-> Spectrum Apply MiniMax H3
-> guider and sampler
```

The node accepts and returns `MODEL`. Disabled mode returns the original model object unchanged. Enabled mode clones the model and rejects anything other than the exact native MiniMax H3 model type with a precise error.

## Parameters

| Parameter | Current default | Meaning |
|---|---:|---|
| `enabled` | `true` | Enables the clone-local Spectrum runtime. |
| `blend_weight` | `0.50` | Configured video spectral share. Offline replay validates and may attenuate it independently at each forecast. |
| `degree` | `1` | Maximum Chebyshev polynomial degree. At least `degree + 1` actual points are required. |
| `ridge_lambda` | `0.10` | Ridge regularization applied to the small Gram matrix. |
| `window_size` | `2.0` | Initial adaptive interval. |
| `flex_window` | `0.75` | Amount added to the interval after a scheduled post-warmup actual step. |
| `warmup_steps` | `1` | Initial solver steps forced to native transformer evaluation. Values above `1` disable the one-point bootstrap. |
| `tail_actual_steps` | `1` | Requested final native tail. Deterministic RES enforces a sampler-safe minimum of `3`. |
| `max_history` | `8` | Maximum model-dtype actual feature snapshots retained. |
| `debug` | `false` | Enables concise run, step, topology, fallback, sanitization, chunk, and teardown logs. |
| `history_storage` | `system_ram` | Stores history in `system_ram`, or in `vram` to avoid transfer overhead when sufficient accelerator memory is free. |
| `bootstrap_first_forecast` | `true` | Experimental one-point hold for `degree=1` and `warmup_steps<=1`. Incompatible node settings disable it with a console warning. |
| `anchor_residual_feedback` | `false` | Experimental video-scored actual-refresh guard. It never injects a hidden residual. Disable offline replay before enabling it. |
| `selective_rollback_correction` | `false` | Experimental thresholded, budgeted rollback for the exact deterministic Euler sampler contract. Disable offline replay before enabling it. |
| `offline_smoothing_replay` | `true` | Standard v0.2.1+ audio-fidelity path: local-only causal capture followed by cross-validated, transformer-free bidirectional replay. |
| `audio_blend_weight` | `0.00` | Configured audio spectral share. Zero keeps replayed audio on local interpolation and prevents direct spectral mixing of audio rows. |

Every value is validated. `max_history` must be at least `degree + 1`.

### Why the audio default changed

MiniMax H3 packs generated audio and video into one transformer sequence while conditioning their rows on different shifted timestep schedules. Spectrum revisions before v0.2.0 applied one `blend_weight` to the complete packed target. In two matched runs with `blend_weight=0.5`, both single-pass Spectrum and the first offline prototype produced an audio defect on the tested seed; the corresponding `blend_weight=0` runs were clean. The offline prototype measured a worst held-out spectral/local error ratio of `3.095` for audio and still retained a `0.162-0.309` spectral share. Proportional attenuation alone therefore did not establish perceptual safety for audio.

The runtime now predicts both contiguous modalities into one packed output buffer with separate history weights. `blend_weight` remains the video control for saved-workflow compatibility, and `audio_blend_weight` defaults to `0`. A nonzero audio value remains available for controlled experiments. H3's separate audio sigma shift is a plausible contributor to the poorer measured spectral fit. The demonstrated implementation fault was the modality-agnostic blend policy across two distinct trajectories.

A subsequent full-H3 comparison showed a second failure path. Single-pass `video=0.5, audio=0` retained the broader audio degradation and its speech-tripping symptom. Single-pass `video=0, audio=0` produced clean audio with a less preferred image. A video forecast changes the solver state, and the following actual H3 call jointly attends over audio and video, allowing later audio anchors to inherit the video trajectory change. Offline replay captures both modalities with causal weights fixed at zero and applies the configured per-modality weights during transformer-free replay.

On the same affected seed, revised offline replay with `video=0.5, audio=0` restored clean, stable audio and retained the preferred image result. Disabling offline replay with the same weights reproduced the degraded audio and speech stutter. The matched A/B validates the isolated capture/replay mechanism for that case: local-only capture preserves the clean audio anchor trajectory, and replay-only video blending cannot feed through another joint transformer call. Universal audio safety across seeds, checkpoints, samplers, prompts, and reference inputs has not been established. Single-pass Spectrum remains susceptible to this indirect coupling path.

When `enabled=True`, the three trajectory modes are mutually exclusive. Enabling more than one raises an error that lists every conflicting setting. Turning offline replay off explicitly retains the single-pass comparison path.

## Default trajectory correction and retained experiments

Offline smoothing replay is the standard default-on H3 path because it removed the reproduced causal video-to-audio degradation in the matched test. The two repository-specific residual/rollback experiments remain default-off. Real-checkpoint tests at 0.65 MP, 8 seconds, 20-step Euler found degraded speech with the original audiovisual anchor-feedback injection and excessive rollback work under the original `score > 1` trigger. The safeguards below retain those modes for continued research outside the default execution path.

| Setting | Status/default | Sampler support | Passes | Behavior when unsupported |
|---|---|---|---:|---|
| `anchor_residual_feedback` | Experimental / off | Euler, RES multistep, RES multistep CFG++ | 1 | Ordinary Spectrum/native fallback rules remain active. |
| `selective_rollback_correction` | Experimental / off | Exact deterministic `sample_euler`, with `s_churn=0` | 1, with local replay on a trigger | RES, CFG++, churned, ancestral, unknown, intercepted, and multi-GPU paths log once and run ordinary Spectrum. |
| `offline_smoothing_replay` | Standard / on | Euler, RES multistep, RES multistep CFG++ | 2 | Unsupported samplers run one valid native pass. An incomplete first-pass archive returns the valid local-only first-pass result. |

### Shared anchor residual

The first two experiments evaluate a shadow Spectrum forecast at a completed actual anchor using only the causal history that existed before that anchor. They also evaluate a zero-order hold of the latest previous actual hidden feature. Both candidates run through the current anchor's native `FinalLayer`, video reconstruction, audio reconstruction, and sigma-dependent processing. Video and audio are reduced independently in bounded FP32 chunks:

```text
E_forecast = RMS(actual output - shadow output)
E_hold     = RMS(actual output - held output)
score      = E_forecast / max(E_hold, scale-aware epsilon)
```

Video and audio scores remain separate. Anchor feedback uses only the video score. Rollback uses the maximum finite video/audio/branch score. A score at or below `1` means the shadow forecast is no worse than the epsilon-floored hold baseline at that later actual coordinate. This comparison does not reveal the native hidden feature at the earlier forecast coordinate; it measures a new actual anchor after the trajectory has already advanced.

Missing, duplicate, incomplete, reordered-unmappable, or changed branch labels/topology disable only the experimental behavior for that run. Nonfinite scores do the same. Ordinary Spectrum or the native fallback remains usable. Debug logs report every measured anchor's video, audio, policy score, and action. Summaries separate shadow/hold output-head time from residual reduction time and report policy maxima, terminal probe skips, speculative/discarded work, refresh/rollback suppression, and offline archive/replay costs.

### Anchor residual feedback

This is a forward scheduling guard. It never revises a completed latent step or adds an anchor residual to another coordinate. The original implementation corrected the packed `[audio | video]` hidden feature using the worse modality's score. Real H3 tests then found both a speech-timing regression and, in another run, a slight image regression. Those failures invalidate the assumption that a hidden error vector measured at anchor `t_j` remains a useful correction direction at `t_{j+1}`. The revised policy retains the measurement and removes hidden-state injection entirely.

- For `video_score >= 1.5`, the next logical step is forced actual with reason `anchor residual feedback refresh`.
- For `video_score < 1.5`, the ordinary Spectrum schedule is left unchanged, even if the diagnostic audio score is larger.
- A run performs at most three feedback refreshes. Once the budget is exhausted, later probes are skipped.

A feedback probe is skipped when only forced-tail steps remain because no later forecast can be replaced. This mode retains no hidden residual and performs no correction arithmetic. It spends one additional transformer call for each accepted refresh, so its value depends on an observable quality improvement over ordinary Spectrum.

### Selective rollback correction

The Euler implementation owns a run-local sampler loop through ComfyUI's `SAMPLER_SAMPLE` wrapper. Before a forecast it checkpoints the pre-forecast latent, logical index, runtime scheduler, adaptive window, refresh counters, history references, statistics, and callback/progress position. If the immediately following actual anchor has an aggregate video/audio score of at least `1.5`, and the run has used fewer than three rollback corrections, it:

1. discards that forecast-influenced anchor result;
2. restores the pre-forecast latent and runtime checkpoint;
3. recomputes the previous forecasted interval as an actual transformer step;
4. advances from that corrected result;
5. recomputes the current anchor as actual at the corrected latent; and
6. continues from the corrected trajectory.

The replayed interval cannot request another rollback. A run performs at most three corrections; once that budget is exhausted, later rollback probes are skipped. Scores below `1.5` are logged and accepted without replay. Speculative calls and the discarded actual call remain included in compute counters. Accepted callbacks and previews occur once per logical step. Cancellation and exceptions propagate through the normal ComfyUI path, and run teardown releases the checkpoint.

The threshold and budget are deliberately fixed internal safeguards in this experimental PR, preserving the three-toggle public interface. In the first 20-step real-checkpoint test, the earlier `score > 1` policy rolled back 7 of 8 evaluated forecasts and executed 25 physical transformer calls, exceeding the 20-call native workload. The revised policy can request at most three rollbacks in one run and therefore prevents that observed seven-rollback failure mode. It is still a quality experiment, and the maximum-call result depends on the underlying Spectrum schedule.

The `run_selective_rollback_euler` sampler mirror is integration-tested against ComfyUI commit `00d02f2854892ee5b9808bc2f6348b972017886a`. Compatibility review must re-check `KSamplerX0Inpaint`, `sampler.inpaint_options`, `model_sampling.noise_scaling`, `sampler.max_denoise`, `sampling.to_d`, and `sampling.trange` whenever the corresponding `KSAMPLER.sample` internals change.

Current deterministic RES stores solver-local `old_denoised`, `old_sigma_down`, and, for CFG++, unconditional denoised state inside `res_multistep`. Those values are outside the public `PREDICT_NOISE` transaction. This branch does not claim RES rollback support: selecting rollback with either RES variant logs one warning before sampler mutation and executes ordinary Spectrum.

### Offline smoothing replay

The first pass runs the ordinary Spectrum schedule with both causal blend weights fixed at zero and the external callback suppressed. Audio and video therefore use the one-point hold or causal two-point predictor during capture, independent of the configured replay weights. Every completed actual anchor is retained independently of causal `max_history` eviction. Anchors use the selected `history_storage`: system-RAM history and the offline archive share immutable CPU tensors, while `vram` retains the full archive on the producing device and shares the most recent tensors with causal history. The wrapper also records the exact initial noise, latent, sigma sequence, seed, sampler, schedule decisions, labels, topology, and input shapes needed to restart the same deterministic sampler.

For a first-pass forecast step, the offline smoother combines:

- a spectral prediction fitted over all retained actual anchors; and
- linear interpolation between the nearest earlier and later actual anchors.

Before replay, every interior actual anchor is withheld in turn. A deterministic sample of up to 16,384 hidden values per conditional branch and modality measures the leave-one-anchor-out spectral error against the error of interpolation between the adjacent actual anchors. Audio and video are validated separately. For each missing step, the worse score at the two bracketing anchors limits the spectral share:

```text
validation_score        = RMS(held-out spectral error) / max(RMS(local interpolation error), epsilon)
effective_video_blend   = blend_weight       / max(1, video_validation_score)
effective_audio_blend   = audio_blend_weight / max(1, audio_validation_score)
```

Each configured modality weight remains an upper bound. A spectral fit that validates at least as well as local interpolation retains that modality's share; a worse fit is attenuated in direct proportion to its error ratio. Audio and video use their own validation score and history weights inside one streamed packed prediction buffer, so an unreliable audio fit neither enters the default audio result nor suppresses the video blend. Spectral weights receive a minimal affine correction so they sum to one, preventing ridge regularization from moving a constant hidden trajectory.

First-pass actual coordinates reuse their stored feature exactly, and every smoothed forecast requires a future actual anchor. The replay restarts from cloned original inputs, invokes zero H3 transformer blocks, runs the current replay step's native output heads and reconstruction, and advances the same deterministic solver. H3's `FinalLayer` normalizes and projects the audio and video row slices independently, so replayed video features do not enter audio through joint attention or cross-row output-head mixing. ComfyUI progress covers capture plus replay; the normal sampler bar follows the compute-heavy capture and the transformer-free replay does not create a second terminal bar. External callbacks/previews still run during the accepted replay pass only. Interruption checks remain active in both passes.

This is an anchor-reuse approximation. The stored actual anchors were evaluated on the first-pass trajectory, while replay generally follows a different trajectory. Future anchors improve the hidden-feature interpolation but do not make those anchors native-equivalent at the replay latent. The method is not lossless, fully corrected, or guaranteed to improve quality.

Offline memory includes cloned initial sampling inputs plus every actual hidden anchor on the selected history device. With `max_history=8` and the observed 11-anchor schedule, shared ownership retains 11 unique feature tensors rather than an eight-entry causal history plus a separate eleven-entry archive. At the supplied 0.65 MP / 8-second shape, this changes explicit `vram` storage from approximately 2.90 GiB VRAM plus 3.89 GiB system RAM to approximately 3.89 GiB VRAM total for history/archive, excluding allocator overhead and input clones. System-RAM storage remains approximately 3.89 GiB and avoids duplicate anchor copies.

Replay adds a second sampler pass and output-head work while eliminating H3 transformer calls only in that second pass. Debug summaries separately report configured replay weights, effective causal capture weights, archive time, smoother-build and validation time, retained archive size and device, hypothetical full-schedule size, held-out sample/anchor/stream counts, maximum audio/video validation scores, per-modality effective blend ranges and attenuation/local-only counts, archived-anchor replay steps, smoothed replay steps, and the smoother's real history/chunk counters. If the first pass is unsupported, intercepted, disabled, incomplete, or changes topology, replay is skipped and the valid local-only first-pass result is returned with one warning.

Across three supplied 0.65 MP / 8-second runs of the earlier fixed-blend implementation, the transformer-free replay added 8.49%, 8.63%, and 8.95% to its corresponding first-pass sampler time while retaining 11 first-pass H3 calls. A later VRAM-resident local-only replay completed in `0.441 s`, while the matched single-pass Spectrum run used the same 11-call schedule. Global `blend_weight=0.5` degraded audio in both paths; global `blend_weight=0` produced clean audio in the two matched runs. Direct modality splitting removed direct audio mixing and left the indirect joint-trajectory defect in single-pass `video=0.5, audio=0`. Revised isolated capture/replay at `video=0.5, audio=0` restored the affected seed's audio fidelity and stability, including the speech-stutter symptom. The evidence establishes the mechanism for that case and leaves broader quality and stability unproven.

The default audio-fidelity configuration is:

```text
offline_smoothing_replay = true
blend_weight = 0.50
audio_blend_weight = 0.00
```

### Current default (performance-oriented)

```text
blend_weight = 0.50
audio_blend_weight = 0.00
degree = 1
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.75
warmup_steps = 1
tail_actual_steps = 1
max_history = 8
history_storage = system_ram
bootstrap_first_forecast = true
offline_smoothing_replay = true
anchor_residual_feedback = false
selective_rollback_correction = false
```

### Conservative schedule for A/B testing

```text
blend_weight = 0.50
audio_blend_weight = 0.00
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.75
warmup_steps = 5
tail_actual_steps = 1
max_history = 8
history_storage = system_ram
bootstrap_first_forecast = false
offline_smoothing_replay = true
anchor_residual_feedback = false
selective_rollback_correction = false
```

The current defaults prioritize throughput and have not been established as universally quality-safe. Existing workflows retain their saved input values. The conservative schedule changes the forecast degree, warmup, and bootstrap behavior: it keeps the first five solver steps native, waits for the five actual history points required by degree 4, and does not use the one-point hold. It reduces the speed benefit and is not established as consistently higher quality. Disabling the bootstrap also changes the denoising trajectory and has produced a less preferred visual result in testing.

Before offline replay became the default, one user reported reference-conditioned audio distortion with the aggressive single-pass schedule. Increasing `degree` and `warmup_steps` helped, and a 30-step run with those increased settings produced clean audio on that setup. This remains a historical single-setup result. It does not establish a 30-step requirement for the current default audio path.

## Adaptive schedule

Warmup and final-tail steps are actual. After warmup, with current interval `W`, a step is actual when:

```text
(consecutive_forecasts + 1) mod floor(W) == 0
```

After a successfully completed scheduled actual step, `W` increases by `flex_window`. A fallback actual step does not increase it. Forecasting also waits until at least `max(2, degree + 1)` actual history points exist.

Schedule counts depend on the sampler safeguards and whether the experimental one-point bootstrap is enabled. CFG can execute separate conditional and unconditional H3 transformer calls on each actual solver step. End-to-end wall-clock speedup depends on output-head cost, CPU transfers, model offload, references, CFG branching, latent size, and hardware.

### Experimental one-point bootstrap

`bootstrap_first_forecast=true` (the current performance-oriented default) enables an H3-specific zero-order hold for solver step 1. It requires:

```text
degree = 1
warmup_steps <= 1
```

After actual step 0, the bootstrap reuses that step's packed final-transformer-block hidden feature as the prediction for step 1. The current step still computes its native H3 timestep conditioning, runs `FinalLayer`, performs the video and audio projections and reconstruction, and applies the current sigma-dependent audio processing. It does not copy the previous step's final video or audio output.

When the ComfyUI node receives `degree != 1` or `warmup_steps > 1`, it disables only `bootstrap_first_forecast` for that execution and logs the supplied values. The requested degree and warmup remain unchanged, and normal history-based forecasting continues. Direct `SpectrumH3Config` callers still receive a validation error for an incompatible enabled bootstrap so configuration mistakes outside the node are not silently accepted.

This bootstrap is separate from polynomial regression. Ordinary degree-1 forecasting still requires at least two actual history entries, no factorization is attempted with one entry, and a bootstrap result is never inserted into actual history. Consequently, step 2 is actual because history still contains only step 0; after step 2, ordinary degree-1 forecasts can proceed.

With Euler, `window_size=2.0`, `flex_window=0.75`, `tail_actual_steps=1`, and `max_consecutive_forecasts=1`, the intended schedules are:

| Steps | Schedule | Actual indices | Forecast indices | Totals |
|---:|---|---|---|---|
| 17 | `A F A F A F A F A F A F A F A F A` | `0, 2, 4, 6, 8, 10, 12, 14, 16` | `1, 3, 5, 7, 9, 11, 13, 15` | 9 actual / 8 forecast |
| 20 | `A F A F A F A F A F A F A F A F A F A A` | `0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19` | `1, 3, 5, 7, 9, 11, 13, 15, 17` | 11 actual / 9 forecast |

The second actual step at the end of the 20-step schedule preserves the configured native tail after the mandatory post-forecast refresh. Force-actual mode, unsupported or disabled forecasting, warmup, the final actual tail, sampler refresh requirements, and transactional fallbacks all retain precedence over the bootstrap.

This option changes the denoising trajectory and is experimental. Validate video and audio with exact-seed Spectrum-on/Spectrum-off comparisons for the intended prompt, checkpoint, sampler, resolution, duration, and branch topology. It is not lossless, output-equivalent, or established as universally safe.

## Supported samplers

Forecasting is currently allowlisted for:

- Euler (`sample_euler`)
- RES multistep (`sample_res_multistep`)
- RES multistep CFG++ (`sample_res_multistep_cfg_pp`)

The reviewed implementations make one `predict_noise` call per solver iteration. Euler feeds each approximate denoised result into the latent used by the next evaluation, so it requires one completed actual H3 evaluation after every forecast. RES multistep stores each current denoised result as `old_denoised` for the following second-order update. The actual evaluation immediately after a forecast still consumes forecast-derived history, then replaces `old_denoised` with its native result before another forecast is allowed. This prevents any RES update from combining two forecasted denoised results. RES also keeps its final three solver steps native; this tail floor applies even when a saved workflow supplies a smaller `tail_actual_steps` value. Ancestral samplers execute native MiniMax H3 because injected noise invalidates the smooth deterministic feature trajectory used by the forecaster. Debug mode logs the exact fallback, tail, or post-forecast refresh reason. Multi-GPU parallel sampling also remains native because distributed forecast-row transactions are not yet validated.

Native EasyCache and LazyCache must not accelerate the same model branch as Spectrum. Either cache can return an approximate diffusion result without invoking MiniMax H3, so Spectrum cannot capture the actual post-transformer feature required by its solver-step transaction. If both are attached, Spectrum now logs one warning and remains inactive for that run while the cache continues normally. Use one of these accelerators on a model branch.

## Memory design

The implementation uses the history-weight form of Chebyshev ridge regression:

```text
w(t*) = phi(t*) (Phi^T Phi + lambda I)^-1 Phi^T
H_hat(t*) = w(t*) H
```

Spectral and linear history weights are combined before reading feature history. Persistent large tensors are limited to `max_history` detached model-dtype snapshots in the selected `history_storage`. Design, Gram, Cholesky, and history-weight tensors remain small FP32 CPU matrices. Prediction streams one bounded slice from one history snapshot at a time, accumulates that slice in FP32 on the prediction device, then writes model dtype. There is no persistent full-feature FP32 regression right-hand side or coefficient tensor.

History storage cost is approximately:

```text
branch_count * max_history * (target_audio_rows + target_video_rows)
* hidden_width * model_dtype_bytes
```

In the supplied 0.5 MP workflow, the effective single-branch topology had 27,075-27,702 target rows, hidden width 5,376, and 16-bit history. At `max_history=8`, it retained 2,273.5-2,324.9 MiB (about 2.22-2.27 GiB). A two-branch topology at the same shape would use roughly twice that amount. At the native 1344x768, 124-frame example, the reviewed layout has about 37,710 target rows; eight conditional/unconditional snapshots can approach 6.1 GiB. Reference tokens do not enter the cached target, while longer duration and larger target geometry increase the cost. Lower `max_history` is valid only while it remains at least `degree + 1`.

With `history_storage=system_ram`, forecast VRAM includes one model-dtype target feature for the current model call plus a bounded FP32 accumulation chunk. Actual steps copy each new snapshot to CPU, and forecasts stream the retained snapshots back to the prediction device. These transfers can reduce the theoretical speedup.

With `history_storage=vram`, the same model-dtype history remains on the device that produced it. This avoids the device-to-host archive and repeated host-to-device forecast reads. The captured target is cloned into compact owned storage; retaining its native view would keep the complete final-block hidden tensor alive. The mode needs the full history allocation plus transient headroom for the current snapshot, prediction result, FP32 chunk, allocator fragmentation, and native H3 execution. At the native example above, use it only with materially more than 6.1 GiB of VRAM free at the native generation peak. An explicit VRAM selection can raise an out-of-memory error when that headroom is unavailable.

Debug run summaries report the selected storage and resolved history device together with archive, history-update, and forecast-prediction wall time. CPU archiving can synchronize preceding CUDA work, while GPU cloning can be asynchronously enqueued, so the component counters diagnose the runtime path rather than serving as isolated kernel benchmarks. End-to-end wall time and peak allocated VRAM are the authoritative comparison.

### Measured VRAM-history results

Three supplied full-checkpoint, 20-step Euler A/B pairs at approximately 0.5 MP compared otherwise identical `system_ram` and `vram` runs:

| Pair | System RAM | VRAM | VRAM difference |
|---|---:|---:|---:|
| 1 | 112.43 s | 105.55 s | -6.1% |
| 2 | 115.60 s | 116.08 s | +0.4% |
| 3 | 109.80 s | 107.26 s | -2.3% |
| Mean | 112.61 s | 109.63 s | -2.6% |

The VRAM runs used about 2.22-2.27 GiB more peak memory, matching the retained history reported by Spectrum. The timing benefit was small and variable, so VRAM history should be treated as an optional optimization for systems with spare VRAM rather than a guaranteed speedup. OS-level GPU monitors can hide the live-allocation increase when PyTorch satisfies it from an already-reserved CUDA memory pool.

## Fallback and transaction behavior

The native path is used when forecasting is unsupported or cannot be proven safe. Reasons include sampler incompatibility, missing branch labels, topology changes, audio/video target count changes, hidden-width changes, duplicate or reordered-unmappable labels, nonfinite schedules, prediction shape failures, and unusable forecasts.

Split conditional calls are assigned by ComfyUI's `cond_or_uncond` and UUID labels. Row allocation is transactional. If correspondence becomes incomplete after an earlier subcall forecast, the entire `predict_noise` attempt is discarded and rerun as an actual step. Exceptions abort the active step without advancing scheduler state, preserve the original traceback, and outer-run teardown releases all history.

If a downstream model or cache patch returns a successful `predict_noise` result without reaching the native MiniMax H3 wrapper, Spectrum accepts that result as a passthrough, disables itself for the rest of the run, and releases its forecast history. One warning identifies the bypass. This preserves the other patch's execution path without letting Spectrum continue from an unobserved solver step.

Model wrappers are registered on the cloned `ModelPatcher`. A clone callback creates a new runtime for every downstream clone. The shared inner H3 module stores no Spectrum state and is never monkey-patched.

## Validation status

Automated tests cover:

- direct coefficient, history-weight, chunked, blended, and row-subset equivalence;
- FP32, FP16, and BF16 features;
- history eviction, repeated coordinates, zero ridge, bounded Cholesky jitter, and factorization reuse;
- absence of persistent full-feature FP32 RHS/coefficient storage;
- warmup, final tail, adaptive counts, fallback accounting, abort rollback, and teardown;
- split, reordered, missing, and duplicate branch labels;
- target audio/video segment ordering and sanitization;
- model detection and clone runtime isolation;
- exact native versus wrapped forced-actual video/audio output on a deterministic tiny native H3 fixture;
- proof that a forecast fixture invokes zero H3 transformer blocks;
- refresh-only anchor feedback with video-only policy scoring, a `1.5` threshold, a three-refresh budget, and no retained/injected hidden residual;
- terminal feedback-probe elimination while preserving the final rollback validation probe;
- rollback threshold and three-correction budget enforcement;
- single-buffer modality-specific online and offline prediction with an audio-local default;
- separate residual output-head timing, offline archive/build timing, per-modality cross-validation/effective-blend reporting, replay anchor/smoothed counts, and replay smoother history/chunk reporting;
- continuous two-pass ComfyUI progress, capture progress callbacks, replay-only previews and external callback side effects, and clean progress completion on recoverable replay fallbacks;
- downstream `predict_noise` passthroughs that never reach the native H3 wrapper, including one-warning disablement and retained-history release.

The v0.2.2 suite passes 173 tests against attached ComfyUI source at commit `00d02f2854892ee5b9808bc2f6348b972017886a`; two CUDA-only tests are skipped in the CPU test environment.

A community compatibility report confirmed that revision `dc6291525112cb4246f864738e5bb4e2b85446da` ran without source changes on Windows 11 with a Radeon AI PRO R9700 32 GB, PyTorch 2.9.1 + ROCm 7.2.1, and ComfyUI 0.30.0. In the reported 20-step RES multistep, 864x480, 107-frame `system_ram` workflow, the expected 14 actual and 6 forecasted evaluations reduced warm elapsed time from 212.73 s to 160.97 s (24.33% lower time; about 1.32x throughput). This validates only that exact configuration; other AMD GPUs, ROCm builds, workflows, and quality cases remain unverified. See [issue #6](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/issues/6).

No full MiniMax H3 checkpoint is available in the automated environment. Supplied real-checkpoint A/B runs validate the 0.5 MP VRAM allocation and show a small, variable timing benefit. A pre-v0.2.1 exact-seed test with the pruned BF16 checkpoint at approximately 0.8 MP showed no obvious visible or audible quality decrease. Other comparisons and user reports found trajectory deviations, malformed rapidly moving details, distorted anatomy, additional limbs, and degraded generated or reference-conditioned audio on some setups. Matched tests isolated two reproduced audio failure paths, and the current default removed both on the affected seed. The older reference-audio workaround is recorded under the [conservative schedule](#conservative-schedule-for-ab-testing) as historical evidence, with no implication that the current path requires 30 steps. Other resolutions, durations, CFG topologies, reference modes, hardware, decoded-video metrics, audio metrics, and audiovisual synchronization remain unverified. Spectrum must not be treated as lossless or output-identical to native sampling.

## Tests

Forecaster smoke test in an environment that already has PyTorch:

```bash
python tests/smoke_forecaster.py
```

Full suite against a current ComfyUI checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
PYTHONPATH=/path/to/ComfyUI \
python -m pytest -q
```

## Validation boundaries

The offline mode was promoted because the affected same-seed A/B isolated the remaining failure path: single-pass `video=0.5, audio=0` reproduced the broader loss of audio fidelity and stability, while offline capture/replay with the same weights restored clean audio and retained the preferred image result. Broader checkpoint, sampler, prompt, and conditioning coverage remains valuable. Before either retained experiment is considered for promotion, test 20-step Euler and RES multistep, plus RES CFG++ where applicable, at approximately 0.65 MP, 8 seconds, and 24 fps. Keep prompt, seed, checkpoint, resolution, duration, sampler, scheduler, images, and audio references identical across:

- native Spectrum-off;
- the default offline path;
- explicit single-pass mode with all three settings false; and
- each retained experimental setting enabled separately with offline replay disabled.

Inspect video quality, generated audio, reference-audio distortion, audiovisual synchronization, total wall time, actual/discarded/replayed transformer calls, peak VRAM, and peak system RAM. Cancel during the first pass, residual evaluation, rollback replay, and offline replay. Run at least ten consecutive generations with Sol-Attn enabled and load old saved workflows to detect retained state, memory growth, deadlocks, callback duplication, WSL wedges, and compatibility regressions.

For the modality split specifically, repeat affected exact seeds with `video/audio` weights `0/0`, `0.5/0`, and `0.5/0.5` in single-pass Spectrum and offline replay. Current evidence establishes that global `0.5/0.5` degraded audio and global `0/0` produced clean audio on one seed. On that same seed, revised offline replay at `0.5/0` restored clean, stable audio and retained the preferred image result, while single-pass `0.5/0` reproduced the degradation and speech-stutter symptom. Broader seeds, speech styles, ambient/transient audio, reference inputs, BF16/INT8 checkpoints, and RES variants remain required.

## Repository layout

```text
ComfyUI-Spectrum-MiniMax-H3/
|-- __init__.py
|-- nodes.py
|-- pyproject.toml
|-- LICENSE
|-- README.md
|-- IMPLEMENTATION_NOTES.md
|-- comfyui_spectrum_h3/
|   |-- __init__.py
|   |-- config.py
|   |-- experiments.py
|   |-- forecast.py
|   |-- nodes.py
|   |-- runtime.py
|   |-- rollback.py
|   |-- sampling.py
|   `-- minimax_h3.py
`-- tests/
```

## Credits

- Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon for [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) and the [official Spectrum implementation](https://github.com/hanjq17/Spectrum).
- The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) maintainers for native MiniMax H3, model patching, sampler wrappers, packed latent support, and model-management infrastructure.

## License

GPL-3.0-or-later. The implementation in this repository is standalone. Spectrum's published mathematics and MIT-licensed official implementation were reviewed as primary references; no source file from the official implementation is vendored.
