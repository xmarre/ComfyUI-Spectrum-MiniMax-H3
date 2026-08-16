# ComfyUI Spectrum MiniMax H3

Training-free feature forecasting for ComfyUI's native **MiniMax H3 audio-video model**.

Spectrum reduces the number of expensive H3 transformer evaluations during sampling. Actual steps run native MiniMax H3 and retain the packed target hidden state after the final transformer block. Forecast steps predict that state from previous actual anchors, skip the H3 transformer blocks for that step, and continue through the native output/sampler path.

Spectrum is an **approximate accelerator**. Forecasted steps change the denoising trajectory, so output can differ from native H3 even with the same seed and workflow.

## v0.2.11: native ER-SDE quality fix

v0.2.11 fixes the characteristic high-frequency / "confetti" corruption on Spectrum forecast steps with native ER-SDE.

The problem was not simply ER-SDE's random increment. A skipped H3 hidden/velocity forecast was being reconstructed against ER-SDE's current stochastic latent as if it were a valid current-state denoised/x0 estimate. ER-SDE then consumed that inconsistent value in its own solver update and higher-order derivative history.

Spectrum now keeps ER-SDE's native stochastic latent trajectory unchanged while reconstructing skipped **solver-facing denoised/x0 values in ER-SDE solver space** from exact actual denoised anchors:

- the first causal forecast after one actual anchor uses the latest exact actual denoised value;
- later causal forecasts use bounded linear dense output in native ER-SDE lambda coordinate from the two latest exact actual anchors;
- degenerate or excessive extrapolation falls back to the latest exact actual anchor;
- the native ER-SDE random stream, `s_noise`, stage-2/stage-3 history, and latent `x` trajectory remain intact;
- no additional H3 transformer/denoiser NFE is added.

Real MiniMax H3 validation removed the forecast confetti completely and produced a modest but repeatable improvement in fine/low-resolution visual structure without an observed loss of motion/action or audio quality.

The release also hardens post-run teardown: generic-correction research persistence/evaluation is no longer allowed to block a completed sampler result from reaching VAE decode or video saving.

## Installation

### ComfyUI Manager / Registry

Install **ComfyUI-Spectrum-MiniMax-H3** through ComfyUI Manager / the Comfy Registry, then restart ComfyUI.

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git
```

To update an existing Git install:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3
git pull --ff-only
```

Restart ComfyUI after updating.

## Quick start

Recommended model chain:

```text
MiniMax H3 model loader
-> LoRA / model patches
-> MiniMax H3 Sigma Shift
-> Spectrum Apply MiniMax H3
-> guider / sampler
```

The current Python defaults are:

```text
enabled = true
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
model_aware_mode = off
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = false
model_aware_replay_generic_correction = false
generic_correction_mode = coordinate_rls
generic_correction_attenuation = no_attenuation
generic_correction_limiter = hard_clip
generic_correction_limit = 0.40
anchor_residual_feedback = false
selective_rollback_correction = false
```

A normal 20-step run commonly resolves to **11 actual H3 transformer evaluations + 9 forecasts** when no fallback or sampler-specific safeguard adds an actual step.

For quality-critical work, compare Spectrum enabled/disabled with the exact same seed, prompt, checkpoint, references, sampler, schedule, resolution, duration and LoRAs.

## Recommended configurations

### General/default path

Keep the defaults unless you have a reason to change them.

`offline_smoothing_replay=true` uses a compute-heavy causal capture pass followed by a transformer-free replay. It was introduced to preserve audio fidelity after matched H3 testing showed speech/stutter regressions from direct audio spectral mixing and from later joint H3 evaluations. The default keeps `audio_blend_weight=0.0`.

### Native ER-SDE quality path

For native ER-SDE, the currently preferred quality-oriented single-pass setup is:

```text
model_aware_mode = full
offline_smoothing_replay = false
model_aware_trust_shrinkage = false
```

The validated full-mode generic correction is:

```text
generic_correction_mode = coordinate_rls
generic_correction_attenuation = no_attenuation
generic_correction_limiter = hard_clip
generic_correction_limit = 0.40
```

The ER-SDE solver-space dense-output correction added in v0.2.11 is automatic; it does not require a new node setting.

The exact legacy generic-correction reproduction remains:

```text
generic_correction_mode = legacy
generic_correction_attenuation = mode_default
generic_correction_limiter = rational
generic_correction_limit = 0.25
```

### Few-step / acceleration LoRAs

Turbo/LightX2V-style LoRAs can be combined with Spectrum. They can reduce sampling time further, but maintainer testing found noticeably larger changes in composition, motion and fine-detail quality than the normal 20-step path. Treat this as a separate quality/speed tradeoff rather than a free additional acceleration.

## How Spectrum works

MiniMax H3 packs target audio and video rows into the final-transformer hidden state. Spectrum forecasts only the target portion:

```text
[target audio rows | target video rows]
```

Text rows and keyframe/reference-only rows are not stored in forecast history.

On an **actual step** Spectrum lets native H3 run normally and captures the target hidden state. On a **forecast step** it predicts that hidden state from retained actual anchors and skips the expensive H3 transformer blocks. The current native output head/reconstruction path still runs.

For ordinary deterministic/single-call samplers this hidden-state forecast is used directly. Native ER-SDE needs an additional solver bridge because its stochastic latent state and higher-order denoised history make `x - sigma * forecast_velocity` an invalid skipped solver observation. v0.2.11 therefore substitutes a bounded denoised-space dense output before ER-SDE consumes the result.

## Supported native path

Spectrum targets native ComfyUI `comfy.ldm.minimax.model.MiniMaxH3Model` and the packed MiniMax H3 sampler path.

Supported native generation layouts include:

- text-to-video/audio (`t2va`)
- first/last-frame-to-video/audio (`fl2va`)
- reference-to-video/audio (`ref2va`)

The minimum reviewed packed-H3 integration contract starts at ComfyUI commit [`e377e263`](https://github.com/Comfy-Org/ComfyUI/commit/e377e263049f9338b4d12a3dd417b36ae62948ff). Older revisions are unsupported.

## Supported samplers

Forecasting is fail-closed and allowlisted for reviewed single-call sampler contracts.

| Sampler | Function | Spectrum policy |
|---|---|---|
| Euler | `sample_euler` | At most one forecast before an actual refresh. |
| Native ER-SDE | `sample_er_sde` | Same conservative cadence plus stochastic-state tracking and solver-space dense output. |
| MiniMax H3 Turbo | `_turbo_sampler` | Reviewed deterministic single-call contract. |
| RES multistep | `sample_res_multistep` | Conservative cadence with protected native tail. |
| RES multistep CFG++ | `sample_res_multistep_cfg_pp` | Same RES safeguards. |

Unknown or changed sampler contracts fall back to native execution rather than guessing.

### ER-SDE compatibility details

The v0.2.11 path keeps the exact native ER-SDE sampler implementation and RNG stream. Spectrum tracks the native stochastic increment only to maintain state ownership and replay compatibility; it does not disable or rescale ER-SDE noise.

Custom/unreviewed `noise_sampler` or `noise_scaler` implementations fail closed to native behavior where their contract cannot be proven.

ComfyUI-TiledDiffusion's current `KSAMPLER.sample(*args, **kwargs)` passthrough monkeypatch is supported through a narrow semantic validator that recursively verifies its stored native delegate. Arbitrary variadic sampler wrappers are not accepted automatically.

## Scheduling

Warmup and tail constraints are always actual. Reviewed samplers also limit the causal forecast horizon and require actual refreshes.

The default degree-1 bootstrap can reuse step 0 as a one-point hold for step 1. Step 2 then runs actual before normal two-anchor degree-1 forecasting begins.

Typical 20-step Euler/ER-SDE single-pass cadence is approximately:

```text
A F A F A F A F A F A F A F A F A F A A
```

That is 11 actual evaluations and 9 forecasts. Fallbacks, model-aware scheduling, replay, RES tail rules, force-actual conditions and saved workflow settings can change the exact schedule.

## Model-aware modes

`model_aware_mode` controls additional scheduling/correction logic:

- `off` — compatibility/default path;
- `schedule` — model-informed scheduling without confidence gating;
- `schedule_confidence` — scheduling with confidence/risk gating;
- `full` — scheduling plus the validated scalar generic correction.

The shipping full-mode generic controller uses signed coordinate transport with scalar RLS and a hard `±0.40` gain bound. More experimental reliability/regional controller options remain available for research/reproduction but are not the production default.

## Deterministic external activation patches

Spectrum can consume a versioned, pure-data compatibility contract from recognized MiniMax H3 patches that deterministically modify transformer activations. The first supported producer is the MiniMax H3 node in ComfyUI-DiffAid-Patches. A valid descriptor participates in the model-aware structural profile and its canonical fingerprint is part of the profile cache key, so patched and unpatched models cannot alias the same cached profile. Recognized runtime activation modulation is reported separately from LoRA/model-parameter patches.

For a nonzero external patch with a partial hard sigma window (`sigma_ramp=0`), Spectrum compares the exact normalized-sigma active/inactive state supplied by the producer against the last successfully completed solver step. If the current call is the first call in a new hard modulation regime and it was scheduled as a forecast, Spectrum promotes that **current step** to one actual H3 evaluation so the forecast history immediately receives an anchor from the new regime. The state is committed only after successful step finalization; abort/retry/rollback do not advance it early. Multiple contracts crossing on the same step still require only one actual evaluation.

A full `[0,1]` window has no interior transition and adds no compatibility NFE. Smooth `sigma_ramp>0` modulation remains continuous and does not force refreshes solely because its gain changes. The hard-boundary guard remains active when `model_aware_mode=off` because it is a forecast-correctness rule rather than an optional scheduling heuristic.

This mechanism is independent of the native ER-SDE stochastic-state compensation described above. Diff-Aid is deterministic activation modulation; Spectrum does not reuse ER-SDE stochastic compensation for it. Spectrum also does **not** post-hoc multiply or otherwise scale the forecasted `[audio | video]` target feature to imitate Diff-Aid. The external modulation occurs before selected transformer blocks and is transformed nonlinearly by the remaining network, so the compatible action at a declared discontinuity is to acquire a real anchor.

## Offline smoothing replay

With `offline_smoothing_replay=true`, Spectrum performs two sampler passes:

1. a capture pass that gathers exact causal anchors;
2. a transformer-free replay pass that applies the accepted bidirectional smoothing trajectory.

The second pass does **not** double H3 transformer NFEs. It reuses the first-pass archive.

Because callback side effects must not run twice, ordinary sampler callbacks are replay-only. This affects live preview timing.

## Live preview

The explicitly supported capture-pass preview integration is **KJNodes Model Preview Override**, typically used with Kijai's MiniMax H3 TAE. Spectrum recognizes the `kj_preview_override` wrapper as observational and keeps it active during capture/replay.

Built-in ComfyUI preview callbacks, ComfyUI-bleh Better Previews, VHS Preview and other callback-based preview implementations are not currently guaranteed to update during the capture pass.

For native ER-SDE single-pass operation, the v0.2.11 dense-output fix also means forecast previews no longer receive the solver-inconsistent confetti-corrupted denoised reconstruction seen in earlier versions.

## Attention and other acceleration nodes

Spectrum does not replace ComfyUI's attention backend.

- actual steps use the attention backend selected by ComfyUI;
- forecast steps skip the H3 transformer blocks, so they make no transformer attention call.

CK / Comfy Kitchen attention works in maintainer testing. [Issue #41](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/issues/41) remains open for a reported second-generation freeze on some CK + Spectrum systems; update ComfyUI past the upstream H3 CK peak-VRAM fix before treating that report as a Spectrum-only failure.

Do **not** run EasyCache or LazyCache on the same model branch as Spectrum. Those caches can bypass the native H3 observation Spectrum needs. Spectrum detects the active cache and remains inactive for that run.

Multi-GPU parallel sampling remains native because distributed forecast-row transactions have not been validated.

## Memory and storage

`history_storage` controls the bounded causal forecast history. `offline_archive_storage` controls replay anchors when offline replay is enabled.

Use `system_ram` for both unless you have a specific reason to keep history in VRAM. Large H3 hidden histories can consume multiple GiB at high resolution/duration.

v0.2.11 also changes post-run diagnostics so generic-correction report persistence/evaluation runs only after core Spectrum history is released and cannot synchronously block a completed sampler result. Debug teardown logs expose calibration-export and runtime-release boundaries when diagnosing WSL/CUDA shutdown stalls.

No speculative `torch.cuda.empty_cache()` or forced device synchronization is performed during normal teardown.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `enabled` | `true` | Enable Spectrum on the cloned model. |
| `blend_weight` | `0.50` | Video spectral/replay blend ceiling. |
| `audio_blend_weight` | `0.00` | Audio spectral share; zero is the current safe default. |
| `degree` | `1` | Maximum Chebyshev forecast degree. |
| `ridge_lambda` | `0.10` | Ridge regularization for the small forecast fit. |
| `window_size` | `2.0` | Initial adaptive scheduling interval. |
| `flex_window` | `0.75` | Interval growth after a completed scheduled actual step. |
| `warmup_steps` | `1` | Initial actual solver steps. |
| `tail_actual_steps` | `1` | Requested final actual tail; sampler-specific rules can enforce more. |
| `max_history` | `8` | Maximum retained causal actual-feature snapshots. |
| `history_storage` | `system_ram` | Causal history storage: `system_ram` or `vram`. |
| `debug` | `false` | Detailed schedule, storage, ER-SDE and teardown diagnostics. |
| `bootstrap_first_forecast` | `true` | Degree-1 step-1 one-point bootstrap when compatible. |
| `offline_smoothing_replay` | `true` | Two-pass capture + transformer-free replay path. |
| `offline_archive_storage` | `system_ram` | Offline replay archive storage. |
| `model_aware_mode` | `off` | `off`, `schedule`, `schedule_confidence`, or `full`. |
| `model_aware_risk_threshold` | `0.65` | Risk threshold for converting a forecast into an actual step. |
| `model_aware_trust_shrinkage` | `false` | Research/reproduction switch; not promoted. |
| `model_aware_replay_generic_correction` | `false` | Research/legacy replay transfer switch. |
| `generic_correction_mode` | `coordinate_rls` | Validated full-mode scalar controller. |
| `generic_correction_attenuation` | `no_attenuation` | Validated coordinate/RLS attenuation policy. |
| `generic_correction_limiter` | `hard_clip` | Validated gain limiter. |
| `generic_correction_limit` | `0.40` | Validated symmetric gain bound. |
| `anchor_residual_feedback` | `false` | Experimental actual-refresh guard; single-pass only. |
| `selective_rollback_correction` | `false` | Experimental deterministic-Euler rollback path; single-pass only. |

## Research / objective media nodes

The package also contains research-only nodes under `sampling/spectrum/research` for bounded R/A/B decoded-media evaluation and reset/staging helpers. They do nothing unless explicitly added to a workflow.

The sequential objective capture path is bounded and failure-contained so recoverable research/evaluation errors do not abort unrelated output nodes. True host/CUDA OOM conditions are still allowed to propagate normally.

See [OBJECTIVE_MEDIA_BENCHMARK.md](OBJECTIVE_MEDIA_BENCHMARK.md) for the current R/A/B workflow, metrics, provenance grouping and verdict rules.

## Debugging

Enable `debug=true` before reporting a sampler problem. Useful messages include:

```text
Spectrum H3 run start ...
Spectrum H3 external patch profile provider=... instance=... kind=... strength=... blocks=... final_block=... sigma_window=... sigma_ramp=...
Spectrum H3 step ... decision=actual|forecast ...
Spectrum H3 external patch transition step=... transitions=... action=force_actual|already_actual
Spectrum H3 ER-SDE stochastic tracking active ...
Spectrum H3 ER-SDE dense anchor ...
Spectrum H3 ER-SDE dense output ...
Spectrum H3 run summary ... external_patch_transitions=... external_patch_forced_actuals=... external_patch_contract_failures=...
Spectrum H3 teardown transition ...
Spectrum H3 run teardown ...
```

If Spectrum encounters an unreviewed sampler/wrapper contract it should log the reason and run native rather than silently applying an unsafe approximation. Malformed declared external compatibility metadata fails safe to all-actual sampling for that run rather than aborting an otherwise valid generation.

For bug reports, include:

- ComfyUI version/commit;
- Spectrum version/commit;
- sampler and scheduler;
- total steps;
- relevant Spectrum settings;
- model/checkpoint and LoRAs;
- resolution/frame count;
- the Spectrum debug section from run start through teardown.

## Compatibility notes for saved workflows

ComfyUI serializes node widget values. Updating Spectrum does not rewrite existing serialized workflow widget values to current defaults.

In particular, workflows saved on older releases retain the serialized values they already contain for `offline_smoothing_replay`, model-aware options, or generic-correction settings. Compare the saved node against the defaults above when reproducing behavior across versions.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
