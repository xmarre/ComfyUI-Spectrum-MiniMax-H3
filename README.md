# ComfyUI Spectrum MiniMax H3

A native ComfyUI MiniMax H3 implementation of **Spectrum**, the training-free spectral diffusion feature forecasting method introduced by **Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon** in [*Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*](https://arxiv.org/abs/2603.01623).

**Original Spectrum:** [Paper](https://arxiv.org/abs/2603.01623) · [Project page](https://hanjq17.github.io/Spectrum/) · [Official implementation](https://github.com/hanjq17/Spectrum)

This repository adapts Spectrum to ComfyUI's native **MiniMax H3 audio-video model** and extends the integration around MiniMax H3's packed audio/video representation, ComfyUI sampler contracts, stochastic and multistage samplers, replay, and the other H3-specific compatibility paths documented below. The implementation in this repository is standalone; the Spectrum paper and official implementation are the primary upstream references for the spectral forecasting method.

The core forecasting approach comes from Spectrum: denoiser features are treated as functions over diffusion time and approximated with Chebyshev polynomial bases whose coefficients are fitted online with ridge regression, allowing selected future feature states to be forecast without a full denoiser evaluation.

Spectrum reduces the number of expensive H3 transformer evaluations during sampling. Actual steps run native MiniMax H3 and retain the packed target hidden state after the final transformer block. Forecast steps predict that state from previous actual anchors, skip the H3 transformer blocks for that step, and continue through the native output/sampler path.

Spectrum is an **approximate accelerator**. Forecasted steps change the denoising trajectory, so output can differ from native H3 even with the same seed and workflow.

## Recent releases

Full release details are kept in [RELEASE_NOTES.md](RELEASE_NOTES.md) and the GitHub release pages.

### v0.2.23 — Active SA-Solver PECE + RefDelta multi-backend composition

- Adds full active SA-Solver PECE acceleration with explicit predicted/corrected phase ownership and actual-only persistent endpoints.
- Makes **`balanced` the default PECE forecast policy** after matched production testing; `max_speed` remains the higher-speed option and `stable_start` the more conservative option.
- Adds Spectrum composition for RefDelta Solver v0.6.0 SEEDS-2/3, SA-Solver PEC, and SA-Solver PECE backends while preserving RefDelta's actual-only evidence ownership.
- Final production testing with DiffAid, Untwist-RoPE and H3 Continuum produced acceptable decoded media with both balanced and max-speed PECE; balanced was perceptually preferred in the tested workflow.
- Clarifies that sampler `steps` are outer sigma intervals, while SEEDS-2/3 and active PECE expose additional logical H3 model-call opportunities.

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

## Step counts vs. H3 model-call opportunities

The ComfyUI **steps** value is the number of outer sigma intervals. It is **not**
necessarily the number of logical MiniMax-H3 model-call opportunities. Multi-stage
solvers and active SA-Solver PECE can expose multiple H3 calls inside one outer interval.

For the usual terminal-zero schedule, with `N = len(sigmas) - 1` outer steps:

| Sampler topology | H3 model-call opportunities | 10 outer steps | 19 outer steps |
| --- | ---: | ---: | ---: |
| Ordinary one-call samplers (for example Euler, RES multistep, ER-SDE) | `N` | 10 | 19 |
| SA-Solver PEC / inactive PECE (`use_pece = false` or `corrector_order = 0`) | `N` | 10 | 19 |
| SEEDS-2 | `2N - 1` | 19 | 37 |
| SEEDS-3 | `3N - 2` | 28 | 55 |
| Active SA-Solver PECE (`use_pece = true`, `corrector_order > 0`) | `2N - 1` | 19 | 37 |

The final SEEDS interval terminates directly at sigma zero, so it does not expose its
internal stage calls; that is why the usual counts are `2N - 1` and `3N - 2` rather than
`2N` and `3N`. The active-PECE `2N - 1` formula additionally assumes
`corrector_order > 0`; with `corrector_order = 0`, PECE reduces to `N` logical H3
model-call opportunities.

Spectrum's `actual / forecast` accounting is over these **logical H3 model-call
opportunities**, not over the UI step count. For example, a clean 10-outer-step
SA-Solver PECE run has 19 logical H3 calls: `balanced` is nominally **11 actual / 8
forecast**, while `max_speed` is **10 / 9**. Continuum prefixes, hard external-patch
transitions, warmup, fallbacks, and other force-actual conditions can change that split,
but they do not change the underlying outer-step count.

This matters when comparing speed across samplers: a 10-step SEEDS-2 or PECE run is not
an NFE-equivalent comparison with a 10-step Euler/RES/ER-SDE run.

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
sa_pece_forecast_policy = balanced
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
| RefDelta SEEDS-2 | `sample_refdelta_seeds_2` | Same Spectrum SEEDS stage policy; RefDelta keeps actual-only outer trajectory evidence and reuses one stochastic gate across the native correlated SEEDS noise segments. |
| SEEDS-3 | `sample_seeds_3` | Same state-conditioned residual architecture; more conservative because two internal stages occur between exact outer anchors. |
| RefDelta SEEDS-3 | `sample_refdelta_seeds_3` | Same Spectrum SEEDS-3 stage policy with RefDelta outer-trajectory correction and one frozen gate across all three native stochastic segments. |
| SA-Solver PEC | `sample_sa_solver` | Actual-only persistent Adams history plus causal solver-space dense output; isolated forecasts re-anchor on the next exact H3 evaluation. |
| RefDelta SA-Solver PEC | `sample_refdelta_sa_solver` | RefDelta trajectory/stochastic control composed around Spectrum's actual-only isolated PEC adapter. |
| SA-Solver PECE | `sample_sa_solver` / `sample_sa_solver_pece` | Active correctors use explicit predicted/corrected phases; P0 plus exact corrected endpoints own shared persistent history, while later predicted phases are solver-space forecasts unless a correctness boundary promotes them. |
| RefDelta SA-Solver PECE | `sample_refdelta_sa_solver_pece` | Same Spectrum active-PECE endpoint topology and user-selected PECE forecast policy; RefDelta keeps P0/C_i actual endpoint evidence and uses the corrected endpoint to own trajectory/stochastic state. |

Unknown or changed sampler contracts fall back to native execution rather than guessing.

### RefDelta sampler-family composition

The RefDelta Stability Sampler can select `er_sde`, `seeds_2`, `seeds_3`,
`sa_solver`, or `sa_solver_pece`. Spectrum recognizes the corresponding RefDelta
sampler functions through a separate versioned backend contract. The existing ER-SDE exact-increment contract remains
unchanged.

For RefDelta SEEDS, Spectrum still owns the reviewed state-conditioned internal-stage
forecast topology. RefDelta observes/corrects the outer denoiser trajectory only. Its
stochastic controller resolves one gate for the outer interval and applies that same gate
to each native SEEDS noise segment, so Spectrum forecasting does not alter the correlated
noise construction.

For RefDelta SA-Solver PEC, Spectrum's isolated-history PEC adapter remains the solver
owner: only actual denoisers enter persistent Adams history and forecast denoisers are
ephemeral current-interval inputs.

For RefDelta SA-Solver PECE, the same active-PECE adapter used by native
`sample_sa_solver_pece` remains authoritative. Spectrum owns predicted/corrected phase
scheduling, solver-space forecast substitution, Continuum protection, external hard
transitions, and `sa_pece_forecast_policy`. RefDelta wraps the model/noise path around
that adapter and mirrors native endpoint replacement for its own evidence:

```text
persistent RefDelta evidence = P0, C1, C2, ...
ephemeral RefDelta PECE calls = P1, P2, P3, ...
```

A predicted P_i may be an actual H3 evaluation or a Spectrum forecast, but it never becomes
a second same-coordinate RefDelta history anchor. Every persistent PECE endpoint must be
actual; a forecasted C_i fails closed.

The backend bridge classifies each completed Spectrum logical model call as actual or
forecast. RefDelta uses only actual outer calls as trajectory anchors. Offline smoothing
replay is intentionally disabled for RefDelta SEEDS/SA composition because the backend
evidence contract is defined on the live causal trajectory.

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

Spectrum supports native Stochastic Adams PEC and active PECE. The original SA-Solver method is formulated in data-prediction space: preceding data-prediction evaluations form the predictor stencil, the predicted endpoint is evaluated, and that endpoint extends the current corrector stencil. Current ComfyUI PECE adds another evaluation after correction. At every corrected outer interval, ComfyUI therefore has two real H3 opportunities at the same sigma and different latent states:

```text
predicted phase: model(x_pred, sigma_i)   -> callback-visible
corrected phase: model(x_corr, sigma_i)   -> callback-invisible
```

With `N = len(sigmas) - 1`, ordinary PEC and PECE with `corrector_order=0` have `N` H3 opportunities and `N` callbacks. Active PECE has `N` predicted opportunities, `N - 1` corrected opportunities, `2N - 1` total H3 opportunities, and still only `N` callbacks.

The central SA invariant is:

```text
persistent Adams history = exact H3 denoisers only
forecasted denoiser      = solver-local / ephemeral only
next exact H3 call       = re-anchor Spectrum + Adams history
```

Native SA normally appends every denoised result to its Adams history. Doing that with an approximate Spectrum denoiser makes the forecast error recursive. Spectrum therefore maintains persistent Adams evidence from exact H3 evaluations only. A forecast can participate in the current solver interval, but it is not promoted into persistent solver history.

Active PECE uses an endpoint-owned policy that mirrors native ComfyUI PECE:

- predicted and corrected calls retain distinct logical identities; no epsilon is added to sigma and no fake solver coordinate is created;
- outer step 0 has no corrected evaluation, so P0 is the first exact persistent endpoint;
- for every later outer step, the exact corrected evaluation C_i replaces P_i as native PECE's persistent Adams endpoint and is also the Spectrum feature-history owner for that coordinate;
- P_i for i > 0 is never retained in Spectrum persistent feature history, even when a hard boundary promotes that predicted call to exact, because C_i supersedes it at the same outer coordinate;
- the shared feature history therefore contains exactly P0, C1, C2, ... rather than two different latent states at the same scalar coordinate;
- every corrected phase remains an actual H3 evaluation and every persistent Adams entry remains an actual H3 observation;
- a corrected-phase forecast and a both-phases-forecast pair remain prohibited by construction.

This removes the old predicted-lane refresh debt without pretending that a corrected hidden state is the predicted hidden state. The feature-history ownership follows the solver's endpoint replacement semantics instead: after a forecasted P_i is consumed by the current corrector, exact C_i arrives before the next predictor and becomes the next trustworthy endpoint anchor.

For **every active-PECE predicted forecast**, the raw hidden-feature forecast is transaction-only and is not used as SA's denoised value. The adapter supplies causal solver-space dense output built only from exact persistent PECE endpoints:

- P1, with only P0 available -> latest-exact hold;
- with two trustworthy endpoints -> bounded two-anchor secant extrapolation in SA lambda space, including consecutive C endpoints;
- invalid, reversed, nonfinite, or excessive extrapolation geometry -> latest-exact hold;
- H3 Continuum -> latest-exact hold on every forecast coordinate.

The native SA tau schedule, stochastic noise draw, predictor/corrector equations, coefficient routine, callback order, sigma schedule, final denoising semantics, and RNG order remain untouched. A forecasted predicted endpoint may affect only its current corrector; it is never inserted into persistent Adams history.

A declared hard external-patch transition still promotes the affected predicted phase to an exact H3 evaluation before the corrector consumes it. After that exact corrected endpoint lands, the PECE dense-output interpolation anchors restart from the post-transition endpoint so the secant bridge does not extrapolate across a transformer discontinuity. Native Adams history is **not** reset or rewritten.

H3 Continuum's protected prefix remains defined in native outer-step coordinates and still protects both predicted and corrected phases inside each protected outer interval. Outside the prefix, corrected phases are exact and each predicted forecast uses the latest exact corrected endpoint as its solver-space hold. Spectrum still executes its cheap hidden-feature forecast for accounting/model-wrapper continuity, but that raw forecast is ignored by SA.

Because every forecasted predicted phase is followed by an exact corrected H3 call, active PECE no longer needs a separate predicted-phase H3 refresh before forecasting the next outer interval. The final corrected endpoint stays exact while the final predicted phase may forecast unless another correctness boundary requires it to be actual.

### Active-PECE forecast policy

`sa_pece_forecast_policy` exposes the early quality/speed tradeoff instead of hiding one preferred cadence inside the sampler integration. All three policies use the **same** corrected-endpoint ownership, actual-only Adams persistence, solver-space bridge, Continuum handling, and hard-transition rules. They differ only in how many initial PECE outer coordinates are protected from predicted-phase forecasting:

| Policy | Protected predicted phases | Clean 10-outer topology | Intended use |
|---|---:|---:|---|
| `max_speed` | P0 only | **10 actual / 9 forecast** | Maximum endpoint-bridge acceleration; trades one additional exact early anchor for speed. |
| `balanced` **(default)** | P0, P1 | **11 actual / 8 forecast** | Released quality/speed default after matched production testing. |
| `stable_start` | P0, P1, P2 | **12 actual / 7 forecast** | Stronger early stabilization at the cost of one more H3 evaluation. |

The ordinary `warmup_steps` setting can only make these policies more conservative: a larger user warmup extends the exact prefix beyond the selected policy minimum. It never weakens Continuum or external-patch safety boundaries.

The selector was added after real 10-outer production testing of the `max_speed` endpoint bridge. With the custom `phase_offset_uniform` scheduler, the tested DiffAid + Untwist + runtime-LoRA workflow completed at **12 actual / 7 forecast** for the initial chunk and **13 actual / 6 forecast** for the H3 Continuum chunk, both with **0 fallbacks**. The decoded final video was reported as good, but the earliest callback previews moved substantially between scenes before settling, including a changed first-frame composition after a few early steps.

Final matched production testing with the dedicated RefDelta SA-Solver scheduler showed that both `max_speed` and `balanced` remained structurally clean and produced acceptable decoded output. The `balanced` run was perceptually better in the tested workflow, while the exact `simple_control` scheduler was slightly preferred over the bounded Adams-conditioned scheduler variant. The early coarse/colored-shape previews were also observed with other MiniMax-H3 samplers and are not treated as an SA-PECE correctness failure. `balanced` is therefore the released default; `max_speed` remains the explicit higher-speed option and `stable_start` the more conservative option.

For reference, with no additional hard boundary, Continuum prefix, fallback, force-actual override, or larger user warmup:

- `max_speed`, **10 outer intervals** -> **10 actual / 9 forecast**;
- `balanced`, **10 outer intervals** -> **11 actual / 8 forecast**;
- `stable_start`, **10 outer intervals** -> **12 actual / 7 forecast**;
- `max_speed`, **19 outer intervals** -> **19 actual / 18 forecast**.

These are nominal topology counts, not promises that every production stack will hit them. H3 Continuum's protected prefix and hard external transitions such as DiffAid/Untwist remain authoritative and can promote specific predicted forecasts to actual. Those safety NFEs are intentionally not traded away merely to preserve a headline ratio.

The earlier real-media PECE validation at **16/3** for 10 outer intervals and **29/8** for 19 outer intervals belongs to the previous conservative predicted-lane-refresh implementation. It remains useful evidence for the phase isolation and exact-corrected persistence design, but it is not treated as validation of the endpoint-bridge selector modes.

The validated 19-call ordinary PEC production cadence remains **11 actual / 8 forecast**. Final PEC real-media validation with DiffAid, Untwist-RoPE and H3 Continuum completed at 11/8 in both the initial and continuation chunks with zero fallbacks.

Deterministic PEC keeps the ordinary conservative behavior because the stochastic solver-space adaptation is not needed. Active PECE retains the phase-specific predicted/exact-corrected topology even when tau or `s_noise` disables stochastic injection. Offline smoothing replay is intentionally disabled for SA-Solver: replaying changed denoiser values would change Stochastic Adams history even with a reproducible random stream, so reviewed SA runs use one causal Spectrum pass.

Supported tau configuration is the native default or the exact reviewed closure produced by native `get_tau_interval_func`. Arbitrary tau callables fail closed and are never invoked speculatively during validation. Custom callable `noise_sampler` values are accepted without inspection or pre-drawing. `simple_order_2` changes the Adams coefficient calculation but not the reviewed model-call topology.

### ER-SDE compatibility details

The v0.2.11 path keeps the exact native ER-SDE sampler implementation and RNG stream. Spectrum tracks the native stochastic increment only to maintain state ownership and replay compatibility; it does not disable or rescale ER-SDE noise.

Custom/unreviewed `noise_sampler` or `noise_scaler` implementations fail closed to native behavior where their contract cannot be proven.

RefDelta Solver v0.2.0+ is admitted through a separate API-v1 contract. Spectrum marks actual versus forecast results; RefDelta uses only actual outputs for risk and trajectory correction. RefDelta then publishes its final risk-gated stochastic tensor back to Spectrum, which uses that exact tensor for forecast-state compensation and seeded offline replay. Contract/version drift, custom stochastic callbacks, and malformed bridge state fail closed.
Spectrum's CI pins the exact reviewed RefDelta v0.2.0 implementation while exercising this contract across its ComfyUI compatibility matrix.

ComfyUI-TiledDiffusion's current `KSAMPLER.sample(*args, **kwargs)` passthrough monkeypatch is supported through a narrow semantic validator that recursively verifies its stored native delegate. Arbitrary variadic sampler wrappers are not accepted automatically.

## Scheduling

Warmup and tail constraints are always actual. Reviewed samplers also limit the causal forecast horizon and require actual refreshes.

For ordinary one-lane samplers, the default degree-1 bootstrap can reuse step 0 as a one-point hold for step 1. Step 2 then runs actual before normal two-anchor degree-1 forecasting begins. State-conditioned stochastic SEEDS deliberately suppresses that bootstrap and uses exact outer stages as fresh anchors for its shared residual history. Active PECE uses the separate `sa_pece_forecast_policy` selector. The released default `balanced` keeps P0/P1 exact; `max_speed` permits the solver-owned P1 one-anchor hold for one additional forecast, while `stable_start` also keeps P2 exact.

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

There is one versioned, fail-closed exception for active SA-Solver PECE. A terminal predicted phase may remain forecasted when every hard transition on that call explicitly declares the reviewed terminal-PECE capability and the runtime topology proves that the next logical call is the same outer step's final corrected phase, is part of persistent endpoint history, and is guaranteed actual. The corrected evaluation still executes and confirms the new external-patch regime. Missing capability metadata, PEC/non-PECE sampling, an interior transition, an unknown topology, a missing corrector, or a stacked transition without the capability retains the ordinary actual promotion.

The exception does not treat the corrector as erasing predictor error: the predictor still affects the state entering correction. Matched decoded-media A/B testing found no discernible quality regression from deferring the qualified terminal predictor, and subsequent two-chunk Continuum runs with the corrected even/even H3 refine geometry exercised the deferred path cleanly under heavier 6-step refinement. The exact corrected endpoint restarts Spectrum's PECE dense-output history, while native Adams history and SA-Solver sigma/tau/noise/RNG ordering remain unchanged.

Diff-Aid supplies normalized sigma directly. A full `[0,1]` window has no interior transition and adds no compatibility NFE. Smooth Diff-Aid modulation with `sigma_ramp>0` remains continuous and does not force refreshes solely because its gain changes.

Untwist supplies actual sampler-schedule progress. Spectrum converts it with:

```text
normalized_sigma = 1 - schedule_progress
```

The Untwist producer's progress window is inclusive at both endpoints, and Spectrum's external descriptor uses inclusive sigma boundaries as well. `end_percent=0.90` therefore remains active at progress exactly `0.90`; the first call with progress greater than `0.90` is the hard end transition.

The Untwist runtime `active` field is validated as part of the declared runtime shape but is intentionally not the temporal transaction-state input. The producer's value also includes per-call reference-range availability and mapping validity. Temporal regime ownership remains the declared static hard window plus exact schedule progress, preventing reference-selection state from advancing or suppressing the committed hard-boundary regime. Untwist visual-profile schema v2 adds the required boolean `terminal_pece_exact_corrector_safe`; schema v1 and invalid/mismatched schema pairs remain conservative.

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
Spectrum H3 external patch transition step=... transitions=... action=force_actual|already_actual|allow_terminal_pece_forecast reason=...
Spectrum H3 external patch terminal PECE exact corrector confirmed ...
Spectrum H3 ER-SDE stochastic tracking active ...
Spectrum H3 ER-SDE dense anchor ...
Spectrum H3 ER-SDE dense output ...
Spectrum H3 offline transition ... event=er_sde_replay_preview_bypass ...
Spectrum H3 offline transition ... event=er_sde_callback_begin|er_sde_callback_end ...
Spectrum H3 ER-SDE replay boundary event=noise_sampler_begin|noise_sampler_end ...
Spectrum H3 run summary ... external_patch_kinds=... external_patch_transitions=... external_patch_forced_actuals=... external_patch_terminal_pece_deferred=... external_patch_terminal_pece_confirmed=... external_patch_terminal_pece_failed_safe=... external_patch_contract_failures=...
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

## Credits

### Spectrum

This project is based on **Spectrum — Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration**, created by:

- **Jiaqi Han**
- **Juntong Shi**
- **Puheng Li**
- **Haotian Ye**
- **Qiushan Guo**
- **Stefano Ermon**

Original work:

- **Paper:** [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)
- **Project page:** [hanjq17.github.io/Spectrum](https://hanjq17.github.io/Spectrum/)
- **Official implementation:** [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum)

Spectrum introduced the training-free spectral feature-forecasting approach used as the foundation of this project: modeling denoiser feature trajectories with Chebyshev polynomial bases, fitting their coefficients with ridge regression, and forecasting future diffusion-step features to avoid selected full network evaluations.

If you use this project in research, please also cite the original Spectrum work:

```bibtex
@article{han2026adaptive,
  title={Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration},
  author={Han, Jiaqi and Shi, Juntong and Li, Puheng and Ye, Haotian and Guo, Qiushan and Ermon, Stefano},
  journal={arXiv preprint arXiv:2603.01623},
  year={2026}
}
```

### ComfyUI

Thanks to the [ComfyUI](https://github.com/Comfy-Org/ComfyUI) maintainers and contributors for the native MiniMax H3 implementation, model-patching infrastructure, sampler APIs, packed latent support, model management, and the surrounding execution framework this integration builds on.

This repository is an independent ComfyUI/MiniMax H3 implementation and integration. No source file from the official Spectrum implementation is vendored here.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
