# ComfyUI Spectrum MiniMax H3

Training-free Spectrum-style feature forecasting for ComfyUI's native **MiniMax H3 audio-video model**.

Spectrum reduces the number of expensive H3 transformer evaluations during sampling. Actual steps run native MiniMax H3 and capture the packed target hidden state after the final transformer block. Forecast steps predict that hidden state from previous actual anchors with a Chebyshev ridge model, skip the H3 transformer blocks, then run the current native `FinalLayer`, video/audio reconstruction, sigma handling, and sampler update.

Spectrum is an **approximate accelerator**. Forecasted steps change the denoising trajectory. Outputs can differ from native H3 in motion, timing, pose, anatomy, facial behavior, audio, and synchronization even with the same seed and workflow.

## Quick start

For new workflows, the current compatibility-safe defaults are:

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
model_aware_trust_shrinkage = false
model_aware_replay_generic_correction = false
generic_correction_mode = legacy
generic_correction_attenuation = mode_default
generic_correction_limiter = rational
generic_correction_limit = 0.25
anchor_residual_feedback = false
selective_rollback_correction = false
```

With the current defaults, a 20-step **Euler** run normally produces **11 actual transformer evaluations and 9 forecasts** when no fallback or model-aware scheduling rule adds an actual step. Other reviewed samplers can impose sampler-specific tail or replay safeguards that change the count.

For quality-critical work, run an exact-seed A/B with Spectrum enabled and disabled for the checkpoint, sampler, resolution, duration, prompt, references, and LoRAs you intend to use.

## Current recommendations

### General use

Keep the defaults above. The default two-pass `offline_smoothing_replay` path was introduced after matched MiniMax H3 tests isolated two audio-degradation paths in the earlier single-pass design:

- direct spectral mixing of audio rows;
- later joint H3 transformer evaluations inheriting a video-forecast trajectory change.

The current default captures a local-only first pass, keeps `audio_blend_weight=0`, then performs a transformer-free replay that applies the accepted video smoothing. This removed the reproduced speech/stutter failure on the affected seed and retained the preferred video result.

The result remains approximate. Broader audio and visual behavior depends on the model, prompt, sampler, references, resolution, duration, precision, and step count.

### Native ER-SDE quality testing

Current controlled native ER-SDE testing found the best temporal facial behavior among the compared model-aware configurations with:

```text
model_aware_mode = full
model_aware_trust_shrinkage = false
offline_smoothing_replay = false
```

In the tested 20-step run, `full` kept the same 11-actual / 9-forecast schedule as `schedule_confidence`, reduced the measured hidden forecast error, and removed a recurring false eye-motion artifact. Earlier same-seed ER-SDE testing also found a pronunciation case that improved with `full`.

This recommendation is scoped to the tested native ER-SDE setup. No equivalent quality ranking has been established for Euler, RES/RES CFG++, Turbo/LightX2V, or other samplers. The global defaults therefore remain `model_aware_mode=off` and `offline_smoothing_replay=true`.

### Speed-up / few-step LoRAs

Turbo/LightX2V-style LoRAs can be combined with Spectrum at runtime. Maintainer testing found a large additional speed gain and lower visual fidelity than the normal 20-step Spectrum path, including smoother/plasticky detail and larger changes in composition, action, and motion.

A measured 8-second LightX2V 8-step run at roughly 0.9 MP used:

```text
Sampler: ER-SDE
Total steps: 8
Actual transformer evaluations: 5
Forecast steps: 3
Spectrum first-pass wall time: 91.256 s
```

A representative 8-second plain-Spectrum run from the earlier roughly 0.7 MP / 20-step family used:

```text
Total steps: 20
Actual transformer evaluations: 11
Forecast steps: 9
Spectrum first-pass wall time: 157.793 s
```

That comparison is about **1.73x faster** in first-pass sampling time for the LightX2V + Spectrum run. It is not a controlled benchmark pair: the resolutions differ and the 20-step candidate is representative of a rerun cluster. Similar visual degradation was observed with the acceleration LoRA without Spectrum.

For quality-critical generation, the normal-step path remains the safer starting point.

## Supported native path

Spectrum targets native ComfyUI `comfy.ldm.minimax.model.MiniMaxH3Model` and the packed MiniMax H3 sampler path.

Supported native generation layouts:

- text-to-video/audio (`t2va`)
- first/last-frame-to-video/audio (`fl2va`)
- reference-to-video/audio (`ref2va`)

The minimum native integration contract is the MiniMax H3 packed-latent API introduced by ComfyUI commit [`e377e263`](https://github.com/Comfy-Org/ComfyUI/commit/e377e263049f9338b4d12a3dd417b36ae62948ff). Older ComfyUI revisions are unsupported.

Spectrum forecasts only the target portion of the final-transformer-block hidden state:

```text
[target audio rows | target video rows]
```

Text rows and keyframe/reference-only rows stay outside forecast history.

Actual steps continue through native H3 `_forward`. Forecast steps skip the transformer blocks, RoPE construction, conditioning projections, reference embedding, and per-block prefetch for that step, then execute the native current-step output path.

## Supported samplers

Forecasting is allowlisted for these reviewed single-call sampler contracts:

| Sampler | Function | Spectrum policy |
|---|---|---|
| Euler | `sample_euler` | At most one forecast, then a completed actual refresh. |
| Native ER-SDE | `sample_er_sde` | Same one-forecast/one-refresh rule; native seeded replay is supported for reviewed native scaler closures. |
| MiniMax H3 Turbo | `_turbo_sampler` | Larryvrh's reviewed deterministic single-call contract; same conservative refresh rule. |
| RES multistep | `sample_res_multistep` | One-forecast/one-refresh rule and a protected three-step native tail. |
| RES multistep CFG++ | `sample_res_multistep_cfg_pp` | Same RES safeguards. |

Unknown or unreviewed samplers fail closed to native execution.

For ER-SDE offline replay, an explicitly supplied custom `noise_sampler`, arbitrary/stateful `noise_scaler`, or unknown future scaler closure is replay-unsafe and falls back to one native pass. The reviewed native `SamplerER_SDE` scaler closures are accepted. Offline replay also promotes the ER-SDE penultimate step only when the normal schedule would otherwise forecast it, preserving an exact terminal replay anchor without imposing a blanket two-step tail.

Multi-GPU parallel sampling remains native because distributed forecast-row transactions have not been validated.

### Other cache/acceleration nodes

Do not run **EasyCache or LazyCache on the same model branch as Spectrum**. Those caches can return an approximate result without entering the native H3 wrapper, which prevents Spectrum from observing the actual feature required by its transaction. Spectrum detects the active cache and remains inactive for that run.

Other downstream patches that intercept `predict_noise` are handled fail-closed: Spectrum accepts the downstream result, disables itself for the rest of that run, and releases retained forecast history.

## Attention backends

Spectrum does not replace ComfyUI's attention implementation.

- On an **actual** Spectrum step, native MiniMax H3 runs with the attention backend selected by ComfyUI.
- On a **forecast** step, the H3 transformer blocks are skipped, so no transformer attention call is made for that step.

CK / Comfy Kitchen attention works in maintainer testing, including consecutive H3 generations. An open report in [issue #41](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/issues/41) describes a second-generation freeze on two systems when CK and Spectrum are combined. ComfyUI's initial CK integration also had a real H3 peak-VRAM regression that was fixed upstream in commit [`62b3c94`](https://github.com/Comfy-Org/ComfyUI/commit/62b3c94bd45154f6486c7abf1b9efcacee96ea69). Update ComfyUI past that fix before diagnosing a CK + Spectrum memory problem.

If the second generation stalls at `0/N`, enable `debug=true`, keep both Spectrum storage settings on `system_ram`, record VRAM before/after the first run, and test once with `offline_smoothing_replay=false`. Those results distinguish replay lifetime from a broader attention/backend peak-memory condition.

## Installation

### ComfyUI Manager

Install **ComfyUI-Spectrum-MiniMax-H3** through ComfyUI Manager / the Comfy Registry, then restart ComfyUI.

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git
```

Restart ComfyUI. The node appears under:

```text
sampling/spectrum -> Spectrum Apply MiniMax H3
```

The node adds no third-party Python dependency. It uses PyTorch and ComfyUI modules already present in a normal ComfyUI installation.

### Updating a Git install

```bash
cd ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3
git pull --ff-only
```

Restart ComfyUI after updating.

Workflows saved with v0.2.0 may retain `offline_smoothing_replay=false`. Enable it once if you want the current default replay path. Workflows created before v0.2.0 did not serialize that input and receive the current Python default.

## Workflow placement

Recommended model chain:

```text
MiniMax H3 model loader
-> LoRA / model patches
-> MiniMax H3 Sigma Shift
-> Spectrum Apply MiniMax H3
-> guider / sampler
```

Spectrum accepts and returns `MODEL`. Disabled mode returns the original model object unchanged. Enabled mode clones the model and installs clone-local sampler/H3 wrappers.

### Live preview support

With `offline_smoothing_replay=true`, Spectrum performs:

1. a compute-heavy capture pass;
2. a transformer-free replay pass.

Ordinary sampler callbacks are intentionally replay-only. Invoking arbitrary callbacks during capture and replay could duplicate callback side effects. ComfyUI's built-in preview and callback-based preview nodes can therefore appear only near the end because replay is fast.

The **only capture-pass live-preview integration currently supported explicitly is KJNodes' `Model Preview Override`**, used with Kijai's MiniMax H3 TAE. Spectrum recognizes KJNodes' `kj_preview_override` wrapper as observational and keeps it inside the two-pass wrapper. It updates during both capture and replay and works whether the preview override appears before or after Spectrum in the model chain.

Built-in ComfyUI previews, ComfyUI-bleh Better Previews, VHS Preview, and other callback-based preview implementations are not currently supported for capture-pass live preview.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `enabled` | `true` | Enables Spectrum for the cloned model. |
| `blend_weight` | `0.50` | Maximum direct spectral share for video. Offline replay validates and can attenuate it per forecast. |
| `degree` | `1` | Maximum Chebyshev degree. Normal polynomial forecasting needs at least `degree + 1` actual anchors. |
| `ridge_lambda` | `0.10` | Ridge regularization for the small Chebyshev Gram system. |
| `window_size` | `2.0` | Initial adaptive schedule interval. |
| `flex_window` | `0.75` | Amount added after a successfully completed scheduled actual step. |
| `warmup_steps` | `1` | Initial native solver steps. Values above 1 disable the one-point bootstrap at the node boundary. |
| `tail_actual_steps` | `1` | Requested final native tail. RES enforces 3; ER-SDE replay can promote a forecasted penultimate step. |
| `max_history` | `8` | Maximum causal actual-feature snapshots. Must be at least `degree + 1`. |
| `debug` | `false` | Enables run/schedule/fallback/storage/model-aware diagnostics. |
| `history_storage` | `system_ram` | Storage for the bounded causal history: `system_ram` or `vram`. |
| `bootstrap_first_forecast` | `true` | Degree-1 one-point hold for step 1 when `warmup_steps <= 1`. Incompatible UI settings disable only this option and log a warning. |
| `offline_smoothing_replay` | `true` | Default two-pass audio-fidelity path: local-only capture plus transformer-free bidirectional replay. |
| `audio_blend_weight` | `0.00` | Direct spectral share for audio. Zero keeps replay audio on local interpolation. |
| `offline_archive_storage` | `system_ram` | Storage for every actual replay anchor. The archive is not capped by `max_history`. |
| `model_aware_mode` | `off` | `off`, `schedule`, `schedule_confidence`, or `full`. See below. |
| `model_aware_risk_threshold` | `0.65` | Threshold used by model-aware scheduling to convert a risky prospective forecast into an actual evaluation. |
| `model_aware_trust_shrinkage` | `false` | Research/reproduction switch. The completed perceptual gate did not support promotion. |
| `model_aware_replay_generic_correction` | `false` | Legacy/research replay transfer of the causal generic correction. Keep disabled for normal use. |
| `generic_correction_mode` | `legacy` | Advanced `full` controller: exact legacy baseline, coordinate/RLS, coordinate/RLS/reliability, or topology-safe regional VIDEO. New modes are experimental. |
| `generic_correction_attenuation` | `mode_default` | Orthogonal advanced attenuation policy. `mode_default` preserves every existing mode exactly; explicit policies reproduce evaluator candidates. |
| `generic_correction_limiter` | `rational` | Generic gain limiter: validated `rational` baseline or experimental `hard_clip`/`tanh`. |
| `generic_correction_limit` | `0.25` | Symmetric limiter scale. Keep `0.25` for the validated baseline. |
| `anchor_residual_feedback` | `false` | Experimental video-scored actual-refresh guard. Requires single-pass operation. |
| `selective_rollback_correction` | `false` | Experimental deterministic-Euler rollback path. Requires single-pass operation. |

## Adaptive scheduling

Warmup and final-tail constraints are actual. After warmup, the schedule gradually increases its interval using `flex_window`. Reviewed samplers also impose a maximum forecast horizon of one logical step and require an actual refresh after each forecast.

The degree-1 one-point bootstrap lets step 1 reuse the actual step-0 packed target hidden state as a zero-order prediction. It does not add a forecast to actual history. Step 2 therefore runs actual, after which normal degree-1 regression can start.

Typical default **Euler** schedules:

| Total steps | Actual | Forecast | Typical indices |
|---:|---:|---:|---|
| 17 | 9 | 8 | Alternating `A/F` through the final actual step. |
| 20 | 11 | 9 | `A F` through step 17, then steps 18 and 19 actual. |

Sampler safeguards, fallbacks, model-aware scheduling, force-actual conditions, branch topology, and saved settings can change these counts. Use `debug=true` when verifying whether Spectrum is active.

## Offline smoothing replay

The default replay path separates trajectory capture from accepted smoothing.

During capture:

- the ordinary Spectrum actual/forecast schedule runs;
- causal video and audio blend weights are forced to zero;
- every actual target hidden anchor is retained for replay;
- normal arbitrary callbacks remain suppressed;
- KJNodes' supported preview override can still observe the pass.

During replay:

- the sampler restarts from cloned initial inputs;
- actual coordinates reuse stored anchors;
- forecast coordinates combine bracketing interpolation with an all-anchor Chebyshev prediction;
- leave-one-anchor-out validation independently attenuates the video and audio spectral shares;
- no H3 transformer block runs;
- the native current-step output heads/reconstruction and sampler update still run.

`audio_blend_weight=0` makes audio local-only during replay. `blend_weight=0.5` remains an upper bound for video; validation can lower it at individual forecasts.

Replay is still an anchor-reuse approximation. Stored anchors came from the capture trajectory, and the replay trajectory can diverge from it. Future anchors improve interpolation without making replay equivalent to native H3 at the new latent.

## Memory

Spectrum keeps two separate storage lifetimes:

- `history_storage`: bounded causal history, capped by `max_history`;
- `offline_archive_storage`: every actual anchor retained until replay completes.

Both default to `system_ram`. This is the recommended setting for constrained GPUs.

Selecting `vram` avoids repeated CPU/GPU transfers and can reduce forecast overhead. It also retains large H3 hidden states on the GPU. The all-anchor replay archive can consume several GiB independently of `max_history`.

A measured 0.65 MP / 8-second, 11-anchor run retained roughly:

```text
full replay archive: ~3.89 GiB
8-entry causal history: ~2.90 GiB
```

These figures exclude current-step tensors, prediction buffers, replay-input clones, allocator fragmentation, and normal H3 peak allocations. Keep `offline_archive_storage=system_ram` unless you have ample VRAM headroom and have measured the complete generation peak.

## Model-aware modes

`model_aware_mode` is opt-in and defaults to `off`.

| Mode | Behavior |
|---|---|
| `off` | Normal Spectrum scheduling and fitting. |
| `schedule` | Builds a compact model/patch profile and may turn a risky prospective forecast into an actual evaluation. |
| `schedule_confidence` | Adds adaptive ridge, usable degree, and modality-specific blend confidence. |
| `full` | Adds the bounded generic latest-delta causal residual correction to `schedule_confidence`. |

The profile samples bounded statistics from selected final H3 block / `FinalLayer` matrices and active patch metadata. Normal LoRA factors are measured without materializing a full dense update. Cached profiles retain scalar/metadata summaries and no model or GPU tensor references.

The surviving `full` correction is generic trajectory information:

```text
d = h[-1] - h[-2]
r = h_actual - h_pred_uncorrected
g_raw = <r, d> / <d, d>
```

The gain is confidence-scaled and bounded before it is applied to the latest-delta direction. The correction itself adds no transformer evaluation. The scheduling component can still convert risky forecast steps into actual evaluations.

`generic_correction_mode=legacy`, `generic_correction_attenuation=mode_default`,
`generic_correction_limiter=rational`, and
`generic_correction_limit=0.25` preserve this validated path. The selectable
coordinate/RLS, correction-reliability, and coarse temporal VIDEO controllers are
experimental live A/B paths. With `debug=true`, full single-pass runs also emit
scalar-only exact quadratic calibration blocks for the shared CPU evaluator.
When `offline_smoothing_replay=false`, Spectrum automatically persists each
completed compatible block, rejects repeated traces/seeds, evaluates its compatible
whole-run group, prints a concise console recommendation, and refreshes detailed
Markdown and JSON reports. No log capture or manual evaluator command is required.
Recommendations include the exact attenuation policy and are emitted only when
the selected offline candidate is reproducible by live settings. Numerically
equivalent RLS candidates are reported as a tie; the canonical live `lambda=0.90`
is retained instead of exposing a new lambda control without evidence.
See
[GENERIC_CORRECTION_RESEARCH.md](GENERIC_CORRECTION_RESEARCH.md) for their causal
contract, topology proof, offline evaluation discipline, and promotion gate.

The earlier model-specific Feature-3 correction families did not provide material improvement and were retired from normal runtime. The experiment record is preserved in [MODEL_AWARE_BENCHMARK.md](MODEL_AWARE_BENCHMARK.md) and [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

### Trust shrinkage and replay calibration research

`model_aware_trust_shrinkage=false` is the supported setting. Hidden-feature improvements were observed during development, followed by a completed perceptual A/B gate that did not show a reliable user-facing benefit.

`model_aware_replay_generic_correction=false` is also the supported setting. Replay traces rejected transplanting the causal latest-delta scalar onto the different future-bracket replay direction.

The repository retains scalar-only replay calibration export and a CPU evaluator for future replay research. No affine, disagreement, coordinate, validation-penalty, tree, neural, AutoML, or other additional replay controller is applied to replay at runtime. Causal generic-correction experiments remain strictly separate. See [FORECAST_TRUST_BENCHMARK.md](FORECAST_TRUST_BENCHMARK.md) and [GENERIC_CORRECTION_RESEARCH.md](GENERIC_CORRECTION_RESEARCH.md).

## Experimental trajectory controls

The following options remain default-off research paths:

- `anchor_residual_feedback`: measures forecast-versus-hold output error at an actual anchor and can force the next step actual when the video score reaches the fixed threshold. It never injects the measured hidden residual.
- `selective_rollback_correction`: deterministic-Euler-only bounded rollback/recompute after a forecast is shown to have produced a high error score at the following actual anchor.

Disable `offline_smoothing_replay` before enabling either experiment. These modes have narrower sampler contracts and can spend additional transformer evaluations.

## Troubleshooting

### Spectrum appears to do nothing

Enable:

```text
debug = true
```

A working run reports nonzero `forecast_steps`. On a single-branch run, `actual_transformer_calls` should also be below the solver-step count when forecasts were accepted.

A zero-forecast run logs the exact sampler, cache, wrapper-bypass, branch-label, topology, prediction, or safety fallback reason.

Reference-heavy `ref2va` workflows can spend substantial time in preprocessing and other work outside the forecasted H3 transformer calls, so end-to-end wall-clock savings can look smaller than the reduction in transformer evaluations.

### No live preview during sampling

This is expected with the default replay path unless you use **KJNodes -> Model Preview Override** with Kijai's MiniMax H3 TAE. Ordinary callbacks are replay-only. See [Live preview support](#live-preview-support).

### Audio sounds worse

Start from:

```text
offline_smoothing_replay = true
audio_blend_weight = 0.00
model_aware_mode = off
```

Then run the same seed with Spectrum disabled. Include the exact Spectrum version, checkpoint, quantization/precision, sampler, scheduler/shift, steps, resolution, duration, prompt, references, LoRAs, and complete node settings in a bug report.

### CK attention freezes on the next generation

Confirm ComfyUI contains upstream H3 CK memory fix `62b3c94`, keep both Spectrum storage controls on `system_ram`, enable `debug`, and compare one run with `offline_smoothing_replay=false`. See [issue #41](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/issues/41).

### Out of memory with replay

Keep:

```text
history_storage = system_ram
offline_archive_storage = system_ram
```

The replay archive is independent from `max_history`. Choosing `offline_archive_storage=vram` retains every actual anchor until replay teardown.

## Validation and limits

The repository's CI exercises the reviewed MiniMax H3 contract across multiple pinned ComfyUI revisions. Coverage includes forced-actual native equivalence, transformer-free forecast execution, sampler recognition and safety guards, split/reordered conditional labels, audio/video row segmentation, replay storage and callbacks, ER-SDE seeded replay, model-aware profile lifetime, generic correction, and replay-calibration tooling.

Real-checkpoint validation remains essential for quality claims. Current evidence includes exact-seed MiniMax H3 runs across Euler and native ER-SDE, reference-conditioned audio investigations, replay A/B tests, Turbo/LightX2V experiments, and community AMD/ROCm testing. These results cover specific configurations and do not establish universal fidelity.

For the detailed current implementation contract, see:

- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- [MODEL_AWARE_BENCHMARK.md](MODEL_AWARE_BENCHMARK.md)
- [FORECAST_TRUST_BENCHMARK.md](FORECAST_TRUST_BENCHMARK.md)
- [RELEASE_NOTES.md](RELEASE_NOTES.md)

## Tests

Forecaster smoke test in an environment with PyTorch:

```bash
python tests/smoke_forecaster.py
```

Full suite against a ComfyUI checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
PYTHONPATH=/path/to/ComfyUI \
python -m pytest -q
```

## Credits

- Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon for [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) and the [official Spectrum implementation](https://github.com/hanjq17/Spectrum).
- The [ComfyUI](https://github.com/Comfy-Org/ComfyUI) maintainers for native MiniMax H3, sampler wrappers, model patching, packed latent support, and model-management infrastructure.

This repository is independent from [ComfyUI-Spectrum-Proper](https://github.com/xmarre/ComfyUI-Spectrum-Proper), which remains the dedicated FLUX implementation.

## License

GPL-3.0-or-later. The implementation in this repository is standalone. Spectrum's published mathematics and official implementation were reviewed as primary references; no source file from the official implementation is vendored.
