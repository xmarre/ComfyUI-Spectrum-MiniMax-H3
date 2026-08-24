# Spectrum MiniMax H3 v0.2.18

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
