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

> [!NOTE]
> For constrained GPUs, keep both `history_storage=system_ram` and `offline_archive_storage=system_ram`. The bounded causal history and the full replay archive have separate storage controls. Selecting `vram` for the replay archive retains every actual anchor on the producing device until replay finishes and can consume several GiB independently of `max_history`.

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

Use v0.2.1 or newer for the corrected default audio path. Use v0.2.2 or newer for live two-pass progress reporting. Use v0.2.4 or newer for live KJNodes MiniMax H3 TAE previews during offline replay. Use v0.2.5 or newer for bounded default replay VRAM and MiniMax H3 Turbo sampler support. Update a Git clone with:

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

### Live previews with MiniMax H3 TAE

[Kijai's MiniMax H3 TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE) currently requires KJNodes' `Model Preview Override`. With offline smoothing enabled, Spectrum keeps that observational wrapper inside its two-pass sampler wrapper regardless of whether the preview node appears before or after Spectrum in the model chain. The KJ preview widget therefore updates during the compute-heavy capture pass and again during replay. Capture frames show the provisional local-only trajectory; replay frames show the accepted smoothed trajectory.

Other external sampler callbacks remain replay-only. Spectrum does not invoke arbitrary callback side effects twice merely to obtain a preview.

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
| `history_storage` | `system_ram` | Stores the causal history, capped by `max_history`, in `system_ram` or `vram`. |
| `bootstrap_first_forecast` | `true` | Experimental one-point hold for `degree=1` and `warmup_steps<=1`. Incompatible node settings disable it with a console warning. |
| `anchor_residual_feedback` | `false` | Experimental video-scored actual-refresh guard. It never injects a hidden residual. Disable offline replay before enabling it. |
| `selective_rollback_correction` | `false` | Experimental thresholded, budgeted rollback for the exact deterministic Euler sampler contract. Disable offline replay before enabling it. |
| `offline_smoothing_replay` | `true` | Standard v0.2.1+ audio-fidelity path: local-only causal capture followed by cross-validated, transformer-free bidirectional replay. |
| `audio_blend_weight` | `0.00` | Configured audio spectral share. Zero keeps replayed audio on local interpolation and prevents direct spectral mixing of audio rows. |
| `offline_archive_storage` | `system_ram` | Stores every actual anchor retained until offline replay completes. This archive is not capped by `max_history`; `vram` is an explicit speed/memory tradeoff. |
| `model_aware_mode` | `off` | Experimental shared model/LoRA prior. `schedule` adds risk-based actual anchors; `schedule_confidence` also adapts fitting and blend; `full` adds bounded correction. |
| `model_aware_risk_threshold` | `0.65` | A prospective legacy forecast becomes an actual evaluation when the combined live/model risk reaches this value. Existing hard sampler, warmup, history, and tail rules still take precedence. |

## Experimental model-aware forecasting

`model_aware_mode` is opt-in and defaults to `off`, so old workflows retain the legacy schedule, fitting, blending, and replay behavior. The modes share one controller and progressively enable:

- `schedule`: evaluates a prospective legacy forecast's risk and may convert it to an actual transformer evaluation. It never converts a required actual step into a forecast.
- `schedule_confidence`: also selects a bounded scalar ridge value, a history-valid polynomial degree, and modality-specific spectral share for that forecast.
- `full`: additionally applies a bounded reduced-order correction in the span of the last observed feature change. The correction coefficient comes from counterfactual errors at completed actual anchors and is scaled by the effective model's audio/video output-head sensitivity. It does not compute a Jacobian, JVP, or extra denoiser forward.

### Model and LoRA profile

Before a generation, Spectrum lazily builds a compact scalar profile of the effective `ModelPatcher`. It samples at most 4,096 values from each of eight selected matrices in the final H3 block and `FinalLayer`, estimates projection gain, and reads active patch metadata. For ordinary LoRA updates it evaluates the relative Frobenius magnitude of the composed low-rank update directly from the factors. It does not materialize a full `B @ A` update. Large factor sets use a conservative norm bound; unsupported patch forms lower profile confidence and increase uncertainty instead of being reinterpreted as LoRA.

The final block is emphasized because Spectrum caches the feature immediately after that block; the current native H3 `FinalLayer` consumes that feature directly. Earlier-block patches contribute at a lower weight. This is a structural prior, not a claim that weight spectra are Spectrum's temporal Chebyshev spectrum.

Profiles are kept in a bounded 16-entry process LRU keyed by ComfyUI's clone lineage UUID, patch-set UUID, H3 architecture signature, and bypass-injection adapter metadata including strength and factor identity. Clones with the same effective patch state reuse a profile; `add_patches`, removal/reload, or changed bypass adapters produce a different key. Cached records retain only scalars and strings—no model, patch, CPU weight, or GPU tensor references.

### Live evidence, replay, and samplers

At actual anchors, Spectrum samples at most 4,096 feature values per branch and modality to measure the previous counterfactual forecast/hold ratio, curvature, fit condition, and residual projection. This trajectory evidence gradually calibrates the static prior: a strongly patched but smooth trajectory can regain confidence, while a nominal base model with abrupt observed behavior loses it. The fitting path uses only a small Chebyshev design/Gram system; it never introduces a per-feature regularization matrix.

Offline capture stores each forecast's selected degree, ridge value, audio/video blends, and correction gains as scalar decision state. Replay uses those exact decisions after dynamically inserted anchors, preserving step alignment and the default `audio_blend_weight=0` fix. No tensor from the model profile is archived.

Euler, deterministic RES/CFG++, MiniMax-H3 Turbo, and native `er_sde` retain their existing hard one-forecast horizon and refresh rules. On `er_sde`, evidence is generation-local and tied to the current seeded trajectory; it is never reused across runs. Offline replay still requires the existing seeded native ER-SDE components and rejects custom noise samplers/scalers.

### Overhead, debugging, and limitations

Debug mode reports profile cache/build time, retained and temporary workspace estimates, patch coverage, sensitivity, per-anchor evidence, per-forecast risk/confidence/fitting/correction choices, adaptive extra NFEs, and model-informed versus generic correction error ratios. It logs scalar aggregates only.

The automated CPU fixture verifies bounded construction and per-step behavior but does not contain a full MiniMax-H3 checkpoint. Therefore this repository does **not** yet claim that static weight/LoRA statistics correlate with real-checkpoint forecast error, that adaptive fitting improves quality at equal NFE, or that the model-scaled correction beats generic residual feedback at equal wall-clock time. `full` should be treated as a research ablation, not a quality preset. Until the fixed-seed matrix in [MODEL_AWARE_BENCHMARK.md](MODEL_AWARE_BENCHMARK.md) has real-checkpoint results, the recommended production setting remains `off`; `schedule` is the most conservative starting point for controlled experiments.

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
| `offline_smoothing_replay` | Standard / on | Euler, native ER-SDE, Larryvrh MiniMax H3 Turbo, RES multistep, RES multistep CFG++ | 2 | Unsupported or replay-unsafe sampler configurations run one valid native pass. An incomplete first-pass archive returns the valid local-only first-pass result. |

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

The first pass runs the ordinary Spectrum schedule with both causal blend weights fixed at zero and the external callback suppressed. Audio and video therefore use the one-point hold or causal two-point predictor during capture, independent of the configured replay weights. Every completed actual anchor is retained independently of causal `max_history` eviction. `history_storage` controls only the bounded causal history. `offline_archive_storage` independently controls the full replay archive and defaults to system RAM. When both select the same resolved device, the archive and causal forecaster share immutable owned tensors while an anchor remains in both. When they differ, finalization keeps one compact copy on each selected device. The wrapper also records the exact initial noise, latent, sigma sequence, seed, sampler, schedule decisions, labels, topology, and input shapes needed to restart the same deterministic sampler.

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

First-pass actual coordinates reuse their stored feature exactly, and every smoothed forecast requires a future actual anchor. The replay restarts from cloned original inputs, invokes zero H3 transformer blocks, runs the current replay step's native output heads and reconstruction, and advances the same deterministic solver. H3's `FinalLayer` normalizes and projects the audio and video row slices independently, so replayed video features do not enter audio through joint attention or cross-row output-head mixing. ComfyUI progress covers capture plus replay; the normal sampler bar follows the compute-heavy capture and the transformer-free replay does not create a second terminal bar. Ordinary external sampler callbacks and their previews run during the accepted replay pass only. KJNodes' `Model Preview Override` is the deliberate exception: its preview callback runs during capture and replay. Interruption checks remain active in both passes.

This is an anchor-reuse approximation. The stored actual anchors were evaluated on the first-pass trajectory, while replay generally follows a different trajectory. Future anchors improve the hidden-feature interpolation but do not make those anchors native-equivalent at the replay latent. The method is not lossless, fully corrected, or guaranteed to improve quality.

Offline memory includes cloned initial sampling inputs plus every actual hidden anchor on the selected archive device. With `max_history=8` and the observed 11-anchor schedule at the supplied 0.65 MP / 8-second shape, the archive was approximately 3.89 GiB and the bounded eight-entry causal history was approximately 2.90 GiB. With both storage controls on system RAM, shared ownership retains approximately 3.89 GiB in system RAM. With both explicitly on VRAM, it retains approximately 3.89 GiB in VRAM. With `history_storage=vram` and the default `offline_archive_storage=system_ram`, it retains approximately 2.90 GiB in VRAM plus 3.89 GiB in system RAM. These figures exclude allocator overhead, current-step tensors, prediction buffers, and replay-input clones.

Replay adds a second sampler pass and output-head work while eliminating H3 transformer calls only in that second pass. Debug summaries separately report configured replay weights, effective causal capture weights, both storage selections, archive time, smoother-build and validation time, retained archive size and resolved device, hypothetical full-schedule size, held-out sample/anchor/stream counts, maximum audio/video validation scores, per-modality effective blend ranges and attenuation/local-only counts, archived-anchor replay steps, smoothed replay steps, and the smoother's real history/chunk counters. If the first pass is unsupported, intercepted, disabled, incomplete, or changes topology, replay is skipped and the valid local-only first-pass result is returned with one warning.

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
offline_archive_storage = system_ram
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
offline_archive_storage = system_ram
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
- native ER-SDE (`sample_er_sde`, KSampler name `er_sde`)
- Larryvrh MiniMax H3 Turbo (`_turbo_sampler`)
- RES multistep (`sample_res_multistep`)
- RES multistep CFG++ (`sample_res_multistep_cfg_pp`)

The reviewed implementations make one `predict_noise` call per solver iteration. Euler feeds each approximate denoised result into the latent used by the next evaluation, so it requires one completed actual H3 evaluation after every forecast. ER-SDE also makes exactly one model call per outer iteration: `max_stage` reuses the current and previous denoised results for its higher-stage finite differences and does not add model evaluations. Its native default noise sampler is recreated from the same workflow seed for each offline pass, draws once after each nonterminal deterministic update when effective `s_noise > 0`, and does not draw on the final sigma-zero step. Custom `noise_sampler` or `noise_scaler` callables may carry mutable state across invocations, so offline replay fails closed to one native pass when either override is supplied. ER-SDE uses the same conservative limit of one consecutive forecast followed by one completed actual refresh and has no additional forced tail.

Larryvrh's reviewed MiniMax H3 Turbo sampler follows the same deterministic single-call contract and refresh policy. RES multistep stores each current denoised result as `old_denoised` for the following second-order update. The actual evaluation immediately after a forecast still consumes forecast-derived history, then replaces `old_denoised` with its native result before another forecast is allowed. This prevents any RES update from combining two forecasted denoised results. RES also keeps its final three solver steps native; this tail floor applies even when a saved workflow supplies a smaller `tail_actual_steps` value. Other unreviewed ancestral samplers execute native MiniMax H3. Debug mode logs the exact fallback, tail, or post-forecast refresh reason. Multi-GPU parallel sampling also remains native because distributed forecast-row transactions are not yet validated.

ER-SDE support applies to ordinary Spectrum and the standard offline smoothing replay path. The default-off `anchor_residual_feedback` and `selective_rollback_correction` experiments remain restricted to their separately reviewed sampler contracts.

Native EasyCache and LazyCache must not accelerate the same model branch as Spectrum. Either cache can return an approximate diffusion result without invoking MiniMax H3, so Spectrum cannot capture the actual post-transformer feature required by its solver-step transaction. If both are attached, Spectrum now logs one warning and remains inactive for that run while the cache continues normally. Use one of these accelerators on a model branch.

## Memory design

The implementation uses the history-weight form of Chebyshev ridge regression:

```text
w(t*) = phi(t*) (Phi^T Phi + lambda I)^-1 Phi^T
H_hat(t*) = w(t*) H
```

Spectral and linear history weights are combined before reading feature history. Outside the offline archive, persistent large tensors are limited to `max_history` detached model-dtype snapshots in the selected `history_storage`. Design, Gram, Cholesky, and history-weight tensors remain small FP32 CPU matrices. Prediction streams one bounded slice from one history snapshot at a time, accumulates that slice in FP32 on the prediction device, then writes model dtype. There is no persistent full-feature FP32 regression right-hand side or coefficient tensor.

History storage cost is approximately:

```text
branch_count * max_history * (target_audio_rows + target_video_rows)
* hidden_width * model_dtype_bytes
```

In the supplied 0.5 MP workflow, the effective single-branch topology had 27,075-27,702 target rows, hidden width 5,376, and 16-bit history. At `max_history=8`, it retained 2,273.5-2,324.9 MiB (about 2.22-2.27 GiB). A two-branch topology at the same shape would use roughly twice that amount. At the native 1344x768, 124-frame example, the reviewed layout has about 37,710 target rows; eight conditional/unconditional snapshots can approach 6.1 GiB. Reference tokens do not enter the cached target, while longer duration and larger target geometry increase the cost. Lower `max_history` is valid only while it remains at least `degree + 1`.

With `history_storage=system_ram`, forecast VRAM includes one model-dtype target feature for the current model call plus a bounded FP32 accumulation chunk. Actual steps copy each new snapshot to CPU, and forecasts stream the retained snapshots back to the prediction device. These transfers can reduce the theoretical speedup.

With `history_storage=vram`, the bounded model-dtype causal history remains on the device that produced it. This avoids the device-to-host archive and repeated host-to-device forecast reads. The captured target is cloned into compact owned storage; retaining its native view would keep the complete final-block hidden tensor alive. The mode needs the full bounded history allocation plus transient headroom for the current snapshot, prediction result, FP32 chunk, allocator fragmentation, and native H3 execution. At the native example above, use it only with materially more than 6.1 GiB of VRAM free at the native generation peak. An explicit VRAM selection can raise an out-of-memory error when that headroom is unavailable. Offline replay does not extend this causal allocation past `max_history` unless `offline_archive_storage=vram` is also selected.

Debug run summaries report both storage selections and the replay predictor's resolved history device together with archive, history-update, and forecast-prediction wall time. CPU archiving can synchronize preceding CUDA work, while GPU cloning can be asynchronously enqueued, so the component counters diagnose the runtime path rather than serving as isolated kernel benchmarks. End-to-end wall time and peak allocated VRAM are the authoritative comparison.

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
- independent causal-history and full replay-archive storage, including mixed CPU/CUDA device placement;
- absence of persistent full-feature FP32 RHS/coefficient storage;
- warmup, final tail, adaptive counts, fallback accounting, abort rollback, and teardown;
- split, reordered, missing, and duplicate branch labels;
- target audio/video segment ordering and sanitization;
- model detection and clone runtime isolation;
- exact native versus wrapped forced-actual video/audio output on a deterministic tiny native H3 fixture;
- proof that a forecast fixture invokes zero H3 transformer blocks;
- exact MiniMax H3 Turbo and native ER-SDE recognition, prefix rejection, conservative refresh policies, seeded ER-SDE replay guards, and complete offline capture/replay coverage;
- refresh-only anchor feedback with video-only policy scoring, a `1.5` threshold, a three-refresh budget, and no retained/injected hidden residual;
- terminal feedback-probe elimination while preserving the final rollback validation probe;
- rollback threshold and three-correction budget enforcement;
- single-buffer modality-specific online and offline prediction with an audio-local default;
- separate residual output-head timing, offline archive/build timing, per-modality cross-validation/effective-blend reporting, replay anchor/smoothed counts, and replay smoother history/chunk reporting;
- continuous two-pass ComfyUI progress, capture progress callbacks, capture-and-replay KJ preview updates, replay-only ordinary external callback side effects and previews, and clean progress completion on recoverable replay fallbacks;
- downstream `predict_noise` passthroughs that never reach the native H3 wrapper, including one-warning disablement and retained-history release.
- scalar-only model/LoRA profile construction for base, single, stacked, differently weighted, zero-strength, and unknown patches; clone reuse, patch UUID invalidation, bounded cache lifetime, and no retained model references;
- model-aware risk calibration, bounded adaptive ridge/degree/blend and correction, sampled anchor evidence, and risk-only insertion of actual evaluations without relaxing sampler constraints.

The model-aware development suite passes 214 tests against the attached current ComfyUI source in the CPU test environment; 11 native/CUDA cases are skipped where optional runtime dependencies or CUDA are unavailable. GitHub Actions remains the authoritative multi-revision check after this branch is published.

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
|   |-- model_aware.py
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
