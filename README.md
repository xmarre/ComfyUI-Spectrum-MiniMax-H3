# ComfyUI Spectrum MiniMax H3

Training-free feature forecasting for ComfyUI's native **MiniMax H3 audio-video model**.

Spectrum reduces the number of expensive H3 transformer evaluations during sampling. Actual steps run native MiniMax H3 and retain the packed target hidden state after the final transformer block. Forecast steps predict that state from previous actual anchors, skip the H3 transformer blocks for that step, and continue through the native output/sampler path.

Spectrum is an **approximate accelerator**. Forecasted steps change the denoising trajectory, so output can differ from native H3 even with the same seed and workflow.

## Recent releases

Full release details are kept in [RELEASE_NOTES.md](RELEASE_NOTES.md) and the GitHub release pages.

### v0.2.22 — Native SEEDS-2/3 and SA-Solver support

- Adds reviewed native Spectrum support for stochastic SEEDS-2/3 using exact-current-state input reconstruction plus transformer-residual forecasting, with the outer stochastic SEEDS stage kept exact.
- Adds reviewed native SA-Solver PEC support with actual-only persistent Adams history and causal solver-space dense output so forecast error cannot recursively contaminate the solver.
- H3 Continuum continuation chunks use latest-exact SA dense output on every forecast coordinate; active PECE corrector configurations remain fail-closed/native.
- Final real-media validation passed with DiffAid, Untwist-RoPE and H3 Continuum: SEEDS-2 reached 11/8 on the initial chunk and 12/7 on the Continuum chunk; SA-Solver reached 11/8 on both chunks, all with zero fallbacks and clean decoded output.

### v0.2.18 — RefDelta Solver compatibility

- Added a versioned, fail-closed interop contract for MiniMax H3 RefDelta Solver v0.2.0+.
- Keeps Spectrum forecasts out of RefDelta's actual-anchor risk/correction history while retaining them in ER-SDE solver history.
- Tracks RefDelta's exact post-gate stochastic increment for skipped-state compensation and offline replay.

### v0.2.16 — Untwist compatibility and post-run isolation

- Added Spectrum compatibility for the MiniMax H3 integration in [**ComfyUI-Untwisting-RoPE**](https://github.com/xmarre/ComfyUI-Untwisting-RoPE), including distinct cache identity, hard-boundary anchoring, and stacking with Diff-Aid.
- Moved optional generic-correction post-run analysis into an isolated subprocess and hardened crash, timeout, and process cleanup so diagnostic failures cannot terminate a completed generation.

### v0.2.15 — H3 Continuum interoperability

- Added H3 Continuum actual-prefix interoperability across normal sampling, fallback, and offline-smoothing capture without adding a hard dependency.
- Fixed the native ER-SDE first-forecast corruption exposed by consecutive Continuum prefix actuals.

### v0.2.14 — Native ER-SDE offline replay safety

- Hardened native ER-SDE offline replay around the reviewed KJNodes Model Preview Override callback by bypassing only its replay-time preview wrapper.
- Added callback, noise-sampler, and replay-finalization diagnostics for hangs and hard wedges.

### v0.2.12 — Diff-Aid compatibility

- Added ComfyUI-DiffAid-Patches v1.0.6+ compatibility with separate external-patch identity, telemetry, cache separation, and hard-window transition anchoring.
- Preserved the normal 11-actual / 9-forecast schedule in validated 20-step native ER-SDE tests without model-aware extra NFEs.

### v0.2.11 — Native ER-SDE quality fix

- Fixed native ER-SDE forecast "confetti" corruption with solver-space denoised interpolation while preserving the native stochastic trajectory and RNG stream.
- Hardened post-run teardown so optional generic-correction research work cannot block a completed generation from reaching decode/save.

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

External H3 patch nodes that publish Spectrum compatibility metadata should be placed before Spectrum on the model chain. See [Deterministic external patches](#deterministic-external-patches) below.

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

Forecasting is fail-closed and allowlisted for reviewed sampler contracts.

| Sampler | Function | Spectrum policy |
|---|---|---|
| Euler | `sample_euler` | At most one forecast before an actual refresh. |
| Native ER-SDE | `sample_er_sde` | Same conservative cadence plus stochastic-state tracking and solver-space dense output. |
| RefDelta ER-SDE | `sample_refdelta_er_sde` | Requires RefDelta Solver v0.2.0+; preserves actual-only evidence and transfers the exact adaptive stochastic increment. |
| MiniMax H3 Turbo | `_turbo_sampler` | Reviewed deterministic single-call contract. |
| RES multistep | `sample_res_multistep` | Conservative cadence with protected native tail. |
| RES multistep CFG++ | `sample_res_multistep_cfg_pp` | Same RES safeguards. |
| SEEDS-2 | `sample_seeds_2` | Stochastic outer stage stays exact; internal stage uses exact-current-state + transformer-residual forecasting with shared interleaved history. |
| SEEDS-3 | `sample_seeds_3` | Same state-conditioned residual architecture; more conservative because two internal stages occur between exact outer anchors. |
| SA-Solver PEC | `sample_sa_solver` | Actual-only persistent Adams history plus causal solver-space dense output; isolated forecasts re-anchor on the next exact H3 evaluation. |
| SA-Solver PECE | `sample_sa_solver` / `sample_sa_solver_pece` | Accepted only when no live corrected-state H3 evaluation is required (`corrector_order=0`); active PECE correctors remain native/fail-closed. |

Unknown or changed sampler contracts fall back to native execution rather than guessing.

For SEEDS-2/3, every denoiser stage is a real MiniMax H3 model-evaluation opportunity and therefore a Spectrum logical model call. A terminal-zero schedule contains `2N - 1` H3 evaluations for SEEDS-2 and `3N - 2` for SEEDS-3 when the outer sampler has `N` sigma intervals.

Native stochastic SEEDS is not treated like an ordinary one-call sampler. Its intermediate stages are evaluated after correlated stochastic increments have changed the latent, so Spectrum leaves the native SEEDS noise process untouched and forecasts only the expensive transformer response:

```text
final H3 target hidden = exact current-state input embedding
                       + forecasted transformer residual
```

On every stochastic SEEDS call, the current-state target audio/video input embedding is rebuilt from the exact latent produced by the native solver. Actual H3 calls capture `final_hidden - input_hidden`; eligible internal-stage forecasts predict only that transformer residual and add it back to the exact current-state embedding before the native H3 final layer.

The stochastic policy is deliberately asymmetric:

- **stage 0 is always actual**;
- one-point bootstrap is disabled;
- stochastic SEEDS uses one shared residual history over the true interleaved model-call coordinates;
- exact outer-stage evaluations refresh that shared history for the following internal-stage forecast;
- native noise draws, stochastic increments, stage equations, callback ordering, and RNG order are unchanged;
- hard external-patch transitions, Continuum prefix requirements, warmup/readiness, final-tail rules, and transactional fallbacks remain authoritative actual boundaries;
- the generic model-aware scheduler veto is advisory for eligible stochastic internal stages, while risk/confidence telemetry, fit/blend adaptation, and generic correction remain active;
- stochastic SEEDS never uses offline smoothing replay. A replay request falls back to one causal state-conditioned Spectrum pass instead of reconstructing the noise-conditioned stage trajectory.

Keeping stage 0 exact is a tested correctness boundary. Real H3 validation showed that forecasting the callback-visible outer stage produced the delayed heavy-noise pattern; preserving that stage exactly removed the failure while still allowing the internal transformer evaluations to be skipped.

SEEDS-2 requires `0 < r < 1`. The `r=1` exponential-Heun endpoint creates duplicate timestep coordinates for different solver states and therefore remains native/fail-closed. SEEDS-3 requires `0 < r_1 < r_2 < 1`.

Final real-media SEEDS-2 validation used the production stacked workflow with DiffAid, Untwist-RoPE and H3 Continuum:

- initial chunk: **11 actual / 8 forecast**, **0 fallbacks**;
- Continuum chunk: **12 actual / 7 forecast**, **0 fallbacks**;
- stage 0 remained exact throughout;
- no recurrence of the delayed heavy-noise artifact.

SEEDS-3 uses the same reviewed state-conditioned residual architecture but remains more conservative because two internal stages occur between exact outer anchors.

### SA-Solver compatibility details

Spectrum supports the reviewed native Stochastic Adams `sample_sa_solver` PEC topology. Active PECE configurations that require a second corrected-state H3 evaluation remain fail-closed to untouched native execution because they need a separate predictor/corrector stage identity.

The central SA invariant is:

```text
persistent Adams history = exact H3 denoisers only
forecasted denoiser      = solver-local / ephemeral only
next exact H3 call       = re-anchor Spectrum + Adams history
```

Native SA normally appends every denoised result to its Adams history. Doing that with an approximate Spectrum denoiser makes the forecast error recursive. Spectrum therefore maintains persistent Adams evidence from exact H3 evaluations only. A forecast can participate in the current solver interval, but it is not promoted into persistent solver history.

For stochastic SA, the raw hidden-feature forecast is also not used directly at the SA numerical boundary during the active stochastic interval. Native SA can add a fresh stochastic increment to `x_pred`; although Spectrum can rebuild the exact input embedding of that state, a skipped transformer does not reproduce the transformer's nonlinear response to the new random realization. The final adapter therefore supplies **causal solver-space dense output built only from exact H3 anchors**:

- one trustworthy actual anchor -> latest-exact hold;
- consecutive actual anchors -> latest-exact hold;
- otherwise -> bounded two-anchor extrapolation in SA lambda space;
- invalid, reversed, nonfinite, or overlong geometry -> latest-exact hold.

The native SA tau schedule, stochastic noise draw, predictor/corrector equations, coefficient routine, callback order, sigma schedule, final denoising semantics, and RNG order remain untouched. Spectrum only substitutes the skipped H3 denoiser value that SA consumes.

Stochastic SA uses isolated forecasts with `max_consecutive_forecasts=1`. The next exact H3 call re-anchors both Spectrum and the actual-only Adams history. The generic model-aware scheduler veto is advisory on this sampler-specific stochastic path so the controller cannot silently collapse the intended NFE reduction; telemetry, adaptive fit/blend, and generic correction remain active.

H3 Continuum continuation chunks use a stricter variant: **every forecast coordinate** uses a latest-exact solver-space hold, including coordinates outside the active stochastic-tau interval. Spectrum still executes its cheap hidden-feature forecast for transaction accounting and telemetry, but that raw denoised value is ignored by SA for the entire continuation chunk.

The validated 19-call production cadence is **11 actual / 8 forecast**. Final real-media validation with DiffAid, Untwist-RoPE and H3 Continuum completed at 11/8 in both the initial and continuation chunks with zero fallbacks. The earlier delayed heavy-noise corruption and the later Continuum-only whole-frame shake/flashing were both absent in the final decoded output.

Deterministic SA keeps the ordinary conservative behavior because the stochastic solver-space adaptation is not needed. Offline smoothing replay is intentionally disabled for SA-Solver: replaying changed denoiser values would change Stochastic Adams history even with a reproducible random stream, so reviewed SA runs use one causal Spectrum pass.

Supported tau configuration is the native default or the exact reviewed closure produced by native `get_tau_interval_func`. Arbitrary tau callables fail closed and are never invoked speculatively during validation. Custom callable `noise_sampler` values are accepted without inspection or pre-drawing. `simple_order_2` changes the Adams coefficient calculation but not the reviewed model-call topology.

### ER-SDE compatibility details

The v0.2.11 path keeps the exact native ER-SDE sampler implementation and RNG stream. Spectrum tracks the native stochastic increment only to maintain state ownership and replay compatibility; it does not disable or rescale ER-SDE noise.

Custom/unreviewed `noise_sampler` or `noise_scaler` implementations fail closed to native behavior where their contract cannot be proven.

RefDelta Solver v0.2.0+ is admitted through a separate API-v1 contract. Spectrum marks actual versus forecast results; RefDelta uses only actual outputs for risk and trajectory correction. RefDelta then publishes its final risk-gated stochastic tensor back to Spectrum, which uses that exact tensor for forecast-state compensation and seeded offline replay. Contract/version drift, custom stochastic callbacks, and malformed bridge state fail closed.
Spectrum's CI pins the exact reviewed RefDelta v0.2.0 implementation while exercising this contract across its ComfyUI compatibility matrix.

ComfyUI-TiledDiffusion's current `KSAMPLER.sample(*args, **kwargs)` passthrough monkeypatch is supported through a narrow semantic validator that recursively verifies its stored native delegate. Arbitrary variadic sampler wrappers are not accepted automatically.

## Scheduling

Warmup and tail constraints are always actual. Reviewed samplers also limit the causal forecast horizon and require actual refreshes.

For ordinary one-lane samplers, the default degree-1 bootstrap can reuse step 0 as a one-point hold for step 1. Step 2 then runs actual before normal two-anchor degree-1 forecasting begins. State-conditioned stochastic samplers deliberately suppress that bootstrap. Stochastic SEEDS uses its exact outer stages as fresh anchors for the shared residual history; SA-Solver uses its sampler-specific stochastic interval and Adams-memory safeguards described above.

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

## Deterministic external patches

Spectrum consumes versioned pure-data compatibility metadata from reviewed MiniMax H3 patches that deterministically change transformer execution. These contracts are runtime metadata only; Spectrum does not import or hard-depend on the producing custom nodes.

Two patch families are currently recognized:

- **ComfyUI-DiffAid-Patches v1.0.6+** — `text_activation_modulation`, using `spectrum_h3_external_patch_contracts` plus `spectrum_h3_external_patch_runtime`.
- [**ComfyUI-Untwisting-RoPE**](https://github.com/xmarre/ComfyUI-Untwisting-RoPE) MiniMax H3 integration — `visual_reference_attention_modulation`, using `spectrum_h3_visual_reference_patch_profiles` plus `spectrum_h3_visual_reference_patch_runtime`.

The external descriptor contributes runtime identity and cache fingerprinting so patched/unpatched models and behaviorally different profiles cannot alias the same cached Spectrum model profile. Diff-Aid and Untwist retain distinct `kind` values when stacked; debug/model-aware telemetry can therefore report, for example:

```text
external_patch_kinds=text_activation_modulation,visual_reference_attention_modulation
```

Recognized deterministic activation/attention modulation is not treated as a LoRA/model-parameter patch for Spectrum's calibrated model-aware `patch_risk` prior. Diff-Aid's raw structural magnitude remains available as separate `external_patch_runtime_perturbation` / `external_patch_final_perturbation` telemetry, while normal online trajectory evidence continues to participate in scheduling.

### Hard-boundary transaction guard

For a producer-declared hard temporal window, Spectrum compares the current regime against the last successfully completed solver step. If the current call is the first call in a new hard regime and it was scheduled as a forecast, Spectrum promotes that **current step** to one actual H3 evaluation so forecast history immediately receives an anchor from the new regime.

The state is committed only after successful step finalization; abort/retry/rollback do not advance it early. Multiple external contracts crossing on the same solver step still require only one actual evaluation. A transition landing on an already-actual step adds no NFE.

Diff-Aid supplies normalized sigma directly. A full `[0,1]` window has no interior transition and adds no compatibility NFE. Smooth Diff-Aid modulation with `sigma_ramp>0` remains continuous and does not force refreshes solely because its gain changes.

Untwist supplies actual sampler-schedule progress. Spectrum converts it with:

```text
normalized_sigma = 1 - schedule_progress
```

The Untwist producer's progress window is inclusive at both endpoints, and Spectrum's external descriptor uses inclusive sigma boundaries as well. `end_percent=0.90` therefore remains active at progress exactly `0.90`; the first call with progress greater than `0.90` is the hard end transition.

The Untwist runtime `active` field is validated as part of the declared runtime shape but is intentionally not the temporal transaction-state input. The producer's value also includes per-call reference-range availability and mapping validity. Temporal regime ownership remains the declared static hard window plus exact schedule progress, preventing reference-selection state from advancing or suppressing the committed hard-boundary regime.

The hard-boundary guard remains active when `model_aware_mode=off` because it is a forecast-correctness rule rather than an optional model-aware scheduling heuristic.

Spectrum does **not** post-hoc multiply or otherwise scale forecasted `[audio | video]` target features to imitate either external patch. These interventions occur inside transformer execution and are transformed nonlinearly by later layers, so a real anchor at a declared discontinuity is the compatible action.

### Workflow order

Place the external patch node before Spectrum on the model chain.

Diff-Aid:

```text
Load Diffusion Model
-> MiniMax H3 Diff-Aid Sparse Patch
-> Spectrum Apply MiniMax H3
-> guider / scheduler
```

Untwisting RoPE:

```text
Load Diffusion Model
-> MiniMax H3 Untwist RoPE
-> Spectrum Apply MiniMax H3
-> guider / scheduler
```

Stacked:

```text
Load Diffusion Model
-> MiniMax H3 Diff-Aid Sparse Patch
-> MiniMax H3 Untwist RoPE
-> Spectrum Apply MiniMax H3
-> guider / scheduler
```

In current real Diff-Aid testing, a five-block H3 patch (`1,13,25,37,50`) at strength `0.5` preserved Spectrum's 11-actual / 9-forecast ER-SDE schedule for both `sigma_end=1.0` and `sigma_end=0.95`, with zero model-aware extra NFEs. `sigma_end=0.95` also preserved the intended shot cut in the tested multi-shot prompt while retaining Diff-Aid's prompt-adherence enhancement.

## Offline smoothing replay

With `offline_smoothing_replay=true`, Spectrum performs two sampler passes:

1. a capture pass that gathers exact causal anchors;
2. a transformer-free replay pass that applies the accepted bidirectional smoothing trajectory.

The second pass does **not** double H3 transformer NFEs. It reuses the first-pass archive.

Because callback side effects must not run twice, ordinary sampler callbacks are replay-only. This affects live preview timing.

## Live preview

The explicitly supported capture-pass preview integration is **KJNodes Model Preview Override**, typically used with Kijai's MiniMax H3 TAE. Spectrum recognizes the `kj_preview_override` wrapper as observational.

With native ER-SDE plus `offline_smoothing_replay=true`, v0.2.14 keeps KJ preview active during the first-pass capture and bypasses the KJ preview wrapper during transformer-free replay while preserving Spectrum's underlying replay callback/progress semantics. Other supported paths retain their existing preview behavior.

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

Core post-run ordering is:

```text
calibration export -> core Spectrum runtime/VRAM release -> optional research dispatch
```

Optional generic-correction evaluation/report work is dispatched only after core runtime/history state has been released. In v0.2.16 it runs in an isolated child process rather than in a Python thread inside ComfyUI. The parent keeps only a bounded watcher/lifetime boundary; one research worker can be active at a time, timeout cleanup is bounded, and native child fatal signals are reported without invalidating the completed sampler result.

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

Separate optional generic-correction post-run analysis uses the v0.2.16 isolated subprocess boundary described above, so a native crash in that optional diagnostic worker is contained outside the ComfyUI generation process.

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
Spectrum H3 offline transition ... event=er_sde_replay_preview_bypass ...
Spectrum H3 offline transition ... event=er_sde_callback_begin|er_sde_callback_end ...
Spectrum H3 ER-SDE replay boundary event=noise_sampler_begin|noise_sampler_end ...
Spectrum H3 run summary ... external_patch_kinds=... external_patch_transitions=... external_patch_forced_actuals=... external_patch_contract_failures=...
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
