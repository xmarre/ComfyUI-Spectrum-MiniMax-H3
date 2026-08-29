# Spectrum MiniMax H3 v0.2.22

v0.2.22 adds reviewed native Spectrum support for ComfyUI **SEEDS-2, SEEDS-3, and SA-Solver**, including the stochastic MiniMax H3 paths that required sampler-specific state handling rather than ordinary one-call forecasting.

## Native SEEDS-2 / SEEDS-3 support

Spectrum now understands the native multistage SEEDS solver geometry instead of treating each H3 model call as an independent diffusion step.

For stochastic SEEDS:

- native stochastic increments, noise draws, stage equations, and callback ordering remain untouched;
- Spectrum forecasts the H3 **transformer residual** while rebuilding the exact current-state target input embedding from the native solver latent;
- stochastic stages share one residual history over the true interleaved model-call coordinates;
- the outer SEEDS stage remains exact, while supported internal stages may be forecast;
- one-point bootstrap remains disabled;
- hard external-patch transitions, Continuum prefix requirements, warmup/readiness, and final-tail exactness remain authoritative;
- the ordinary model-aware scheduler veto is advisory on the sampler-specific stochastic internal-stage path, while model-aware telemetry, adaptive fit/blend, and generic correction remain active;
- stochastic SEEDS does not use offline smoothing replay because replaying the model-call sequence independently of the native noise-conditioned stage trajectory would break solver geometry.

The final real SEEDS-2 validation completed cleanly with DiffAid, Untwist-RoPE, and H3 Continuum:

- initial chunk: **11 actual / 8 forecast**, **0 fallbacks**;
- Continuum chunk: **12 actual / 7 forecast**, **0 fallbacks**;
- stage 0 stayed exact throughout;
- the previously observed delayed heavy-noise pattern did not recur.

SEEDS-3 is supported under the same reviewed state-conditioned residual architecture, but remains more conservative because its two consecutive internal stages do not have an exact outer-stage anchor between them.

## Native SA-Solver support

Spectrum now supports native SA-Solver without allowing approximate H3 denoisers to become recursive persistent Adams observations.

The stochastic SA integration uses:

- **actual-only persistent Adams history**;
- forecast values only as bounded solver-local/ephemeral inputs;
- causal solver-space dense output built from exact H3 anchors during stochastic intervals;
- exact native tau/noise draws, predictor/corrector equations, callback ordering, sigma schedule, and final denoising semantics;
- a one-forecast maximum streak with an exact H3 re-anchor after each skipped transformer call;
- fail-closed/native behavior for unsupported active PECE corrector configurations.

For H3 Continuum continuation chunks, every SA forecast coordinate uses a latest-exact solver-space hold and ignores the raw Spectrum hidden-feature denoised value at the SA boundary. This removes the remaining continuation-specific instability while preserving the 11/8 NFE target.

Final real-media SA validation completed with:

- initial chunk: **11 actual / 8 forecast**, **0 fallbacks**;
- Continuum chunk: **11 actual / 8 forecast**, **0 fallbacks**;
- all eight Continuum forecast coordinates using the reviewed all-forecast latest-exact isolation path;
- no recurrence of the delayed heavy-noise corruption;
- no recurrence of the later whole-frame vertical shake/flashing artifact.

## Validation and compatibility

PR #86 is the reviewed SEEDS-2/3 implementation and passed Actions **#495** across all seven supported ComfyUI/Python lanes. PR #87 is the stacked SA-Solver implementation and passed Actions **#511** on its final one-commit head.

Both PRs were additionally validated with real MiniMax H3 media using the production workflow with DiffAid, Untwist-RoPE, H3 Continuum, runtime LoRA hooks, and the current PDD-capable ComfyUI H3 path.

Existing Euler, RES multistep, ER-SDE, RefDelta, Continuum, generic-correction defaults, trust-shrinkage default-off behavior, and ordinary sampler scheduling remain unchanged outside the new sampler-specific paths.

---

# Spectrum MiniMax H3 v0.2.21

v0.2.21 restores Spectrum forecast execution with current ComfyUI MiniMax H3 after the PDD LoRA update changed the native output-head contract.

## ComfyUI 0.34 PDD final-layer compatibility

- Fixes `FinalLayer.forward() missing 3 required positional arguments: 'sigma', 'sample_sigmas', and 'shifts'` on the first eligible Spectrum forecast with ComfyUI 0.34.0 and later PDD-capable H3 cores.
- Spectrum now detects the reviewed native FinalLayer contract and forwards the same current sigma, exact sampler sigma schedule, and video/audio sigma shifts used by native H3.
- Reviewed older ComfyUI revisions retain the existing four-argument FinalLayer path.
- An incomplete future PDD argument contract fails explicitly instead of guessing.
- No H3 Continuum-side workaround is required; Continuum exposed the stale Spectrum forecast output-head call.

## Validation

PR #84 adds focused regression coverage for the PDD projection call and adds ComfyUI commit `2504e68d4d9dedb514e172692f13436623f25aed` to the compatibility matrix with its required `comfy-kitchen` and `comfy-aimdo` versions. All seven ComfyUI/Python lanes pass, including the historical legacy FinalLayer contracts. CodeRabbit reported no actionable code finding.

PR #84 supersedes #83; thanks to @bun-dev for independently identifying the same Core contract break and proposing the compatible call shape.

---

## v0.2.20

v0.2.20 fixes the RefDelta API-v1 step-provenance handoff after a tracked ER-SDE model call.

## RefDelta tracked-step provenance

- Fixes `RefDelta requested a model-result classification for the wrong step` immediately after the first successful Spectrum/RefDelta model evaluation.
- The real runtime trace showed Spectrum's ER-SDE stochastic tracker had already consumed and logged step 0 while the separate RefDelta bridge-local descriptor had not been mirrored reliably through the surrounding ComfyUI/Continuum model-options path.
- For stochastic RefDelta runs, provenance is now recorded directly from the exact `ERSDEStepDescriptor` that the ER-SDE stochastic tracker successfully consumes. RefDelta therefore uses the same authoritative step classification that Spectrum used for stochastic-state ownership.
- The existing post-model bridge update remains a redundant consistency path instead of the sole source of RefDelta provenance.
- Deterministic `s_noise=0` RefDelta runs keep their direct bridge descriptor path because no stochastic tracker exists in that mode.
- Stale-step and external-increment source validation remain strict; diagnostics now include requested/source and observed step IDs.
- No model-path, Continuum, RefDelta Solver, prompt, sampler-setting, or workflow changes are required.

## Validation

PR #78 adds regression coverage for the exact failure mode: the tracker consumes step 0 successfully without a separate bridge update, and RefDelta must still classify the same step correctly. Coverage also verifies stale-step rejection, external-increment source checking, replay provenance, and deterministic no-tracker behavior.

The complete six-lane Spectrum ComfyUI/Python matrix passes, including Ruff, compileall, focused external compatibility suites, and native MiniMax H3 fixtures. CodeRabbit reported no actionable merge-blocking finding and rated the runtime change minimal risk.

---

## v0.2.19

v0.2.19 fixes RefDelta Solver discovery in the actual ComfyUI custom-node loading layout.

## RefDelta custom-node namespace discovery

- Fixes the runtime failure `RefDelta interop API is unavailable: No module named 'comfyui_refdelta_solver'` that could occur even though the RefDelta sampler itself was already loaded and usable by ComfyUI.
- Spectrum now recognizes the already-loaded RefDelta implementation when ComfyUI has placed it under a package-relative custom-node namespace instead of exposing `comfyui_refdelta_solver` as a top-level package.
- Canonical RefDelta imports reuse the exact live config class, sampler function, and interop contract objects instead of importing the same files a second time under a different module identity.
- The existing strict RefDelta API-v1 checks remain unchanged: function identity, config provenance, interop version, option allowlist, stochastic ownership, and wrapper ordering still fail closed on drift.
- No `PYTHONPATH` changes, pip installation, duplicated model path, or workflow changes are required.

## Validation

PR #76 adds a regression for the exact nested ComfyUI package layout that produced the failure. The complete six-lane Spectrum ComfyUI/Python matrix passes, including Ruff, compileall, focused external compatibility tests, and native MiniMax H3 fixtures. CodeRabbit reported no actionable findings on the substantive compatibility change.

This patch changes discovery only. RefDelta solver behavior, Spectrum forecasting policy, stochastic compensation, Continuum interoperability, Diff-Aid handling, Untwisting RoPE handling, and offline replay semantics remain unchanged.

---

## v0.2.18

v0.2.18 adds explicit compatibility with MiniMax H3 RefDelta Solver v0.2.0+.

## RefDelta API v1

- `sample_refdelta_er_sde` is admitted only when the installed function, config type, option set, and versioned interop marker match the reviewed contract.
- Spectrum passes actual/forecast/replay provenance to RefDelta. Forecasted denoised values continue through ER-SDE solver history but cannot enter RefDelta's raw-model risk or correction evidence.
- RefDelta publishes the exact stochastic increment after its risk and endpoint gates. Spectrum retains that tensor for the next skipped-state compensation instead of reconstructing an ungated native increment.
- Native-equivalence RefDelta configurations continue through the reviewed native ER-SDE ownership path.
- Deterministic `s_noise=0` RefDelta runs still receive actual/forecast provenance without allocating a stochastic tracker.
- Offline replay preserves source-actual provenance and aborts safely to the completed first pass if interop state becomes inconsistent.

## Validation and failure policy

The test matrix covers the shared API contract, exact gated-increment transfer, missing-publication rejection, actual/forecast/replay classification, native ComfyUI contracts, Python 3.12/3.13, Ruff, compileall, and wheel construction. Unreviewed RefDelta versions, options, stochastic callbacks, wrapper ordering, or bridge state disable forecasting or fail explicitly rather than silently changing stochastic ownership.
The cross-repository jobs pin the exact RefDelta API-v1 commit reviewed for this release.
Release validation covers six Spectrum matrix jobs and the four-job RefDelta native fixture matrix.

Existing native ER-SDE, Euler, RES multistep, Turbo, Continuum, Diff-Aid, Untwisting RoPE, masked H3, refinement, model-aware, and offline-replay behavior remains unchanged.

---

## v0.2.17

v0.2.17 completes the current H3 Continuum interoperability work: native masked continuation can remain forecast-capable where the installed ComfyUI core exposes the required per-token H3 mask helper, and the learned-latent sampler-2 refinement path can use Spectrum without inheriting sampler-1's Continuum actual-prefix policy.

## Native Masked H3 forecasting

Spectrum now reconstructs native MiniMax H3 FinalLayer modulation for mixed VIDEO/AUDIO denoise masks instead of disabling forecasting for the entire masked continuation chunk.

- Per-row VIDEO and AUDIO timestep selections follow native H3 mask semantics.
- Scalar fully-generating paths retain the existing fast path.
- Residual/shadow output-head evaluation uses the same reconstruction.
- Per-row timestep index tensors are placed on the target latent device, avoiding CPU/CUDA index-device mismatches in FinalLayer implementations that use `index_select`.
- On older reviewed ComfyUI cores that do not expose `mask_row_values`, a masked forecast fails closed to one native H3 transformer evaluation instead of raising during output-head reconstruction.
- Malformed audio mask layouts continue to fail explicitly.

## Short learned-latent refinement

The integrated MiniMax H3 latent upscaler/refiner supplies an explicit `h3_refinement` API-v1 marker on a clone of Continuum's exact per-chunk MODEL. Spectrum validates that contract and lets its sampler-2 actual-prefix policy override the generation-only Continuum prefix carried by sampler 1.

Normal Continuum generation still preserves its two-step actual prefix. A valid three-step sampler-2 refinement can therefore use:

```text
actual -> forecast -> actual
```

with the normal warmup/final-tail safety rules.

Spectrum continues to honor genuine external-patch hard transitions. DiffAid v1.0.7 removes the artificial partial-denoise transition at its source by evaluating marked refinement against the full H3 sigma reference; Spectrum does not bypass a real model-function transition.

## Coordinated runtime validation

The complete real CUDA path was validated with:

- H3 Continuum exact `refine_state` handoff;
- MiniMax H3 learned latent upscale + internal sampler-2 refinement;
- DiffAid marked-refinement sigma semantics;
- Untwisting RoPE external-patch metadata;
- native ER-SDE;
- Spectrum enabled on both the main Continuum generation and sampler 2.

A three-step high-resolution refinement produced `2 actual + 1 forecast` per refined chunk, with no inherited Continuum-prefix force-actual and no artificial DiffAid middle-step transition. The resulting media quality was user-validated as impeccable.

The tested 0.7 MP native -> 1.75x learned-upscale workflow reduced the Refine node from roughly 302.5 s with three native refinement NFEs to roughly 212.7 s with the middle NFE forecast, while preserving the tested output quality.

## Validation and compatibility

The PR test matrix covers Python 3.10-3.13, the forecaster smoke test, Ruff/compileall, focused H3/external compatibility suites, and native MiniMax H3 fixtures across the reviewed ComfyUI revisions.

This release is coordinated with:

- ComfyUI-DiffAid-Patches v1.0.7;
- H3 Continuum v3.4.1;
- the integrated MiniMax H3 Latent Upscaler + Refine release.

Existing Spectrum defaults, generic-correction defaults, normal 19-step Continuum forecasting policy, ER-SDE stochastic ownership, offline-replay policy, and real external-patch transition barriers remain unchanged.
