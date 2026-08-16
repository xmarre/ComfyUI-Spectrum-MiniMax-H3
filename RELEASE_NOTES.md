# Spectrum MiniMax H3 v0.2.15

v0.2.15 adds first-class interoperability with H3 Continuum's continuation sampler contract and closes the native ER-SDE solver-space edge case exposed by Continuum's two-step actual prefix.

## H3 Continuum actual-prefix interoperability

Spectrum can now consume H3 Continuum's optional versioned runtime request from:

```text
transformer_options["h3_continuum"]
```

For the reviewed API v1 contract, an active request can specify an initial number of solver steps that must remain real H3 transformer evaluations. Spectrum treats that request as a run-local scheduling constraint rather than as a special sampler implementation.

The integration is deliberately narrow:

- no hard dependency on H3 Continuum is added;
- missing, inactive, malformed, wrong-type, negative, or unknown-API metadata leaves ordinary Spectrum behavior unchanged;
- the requested prefix is clamped to the available solver-step count;
- the prefix applies to ordinary runs, single-pass fallback, and the offline-smoothing first pass;
- offline replay is explicitly prefix-free because it reuses the already-captured anchor schedule;
- each continuation sampling call accepts and logs its request independently, with no prefix state leaking into later chunks.

The original interop design and runtime validation were contributed by **@ukr8b3g-cmyk** in PR #52 and reconciled onto current Spectrum main in PR #61 after the contributor branch became stale against the later ER-SDE, Diff-Aid, Python 3.13, and replay-safety work.

## Diff-Aid coexistence

Continuum's actual-prefix request composes with the v0.2.12 external-patch compatibility layer rather than bypassing it.

A dedicated regression places a Diff-Aid hard-sigma transition on a step already protected by the Continuum prefix and verifies that:

- the step remains one actual H3 evaluation;
- the external transition is observed exactly once;
- Diff-Aid does not request an additional compatibility NFE for an already-actual step;
- the two-step prefix still accounts for exactly two transformer calls;
- external-patch run state is released normally at teardown.

## Native ER-SDE first forecast after an actual prefix

Continuum commonly produces a continuation schedule beginning like:

```text
actual 0  (Continuum prefix)
actual 1  (Continuum prefix)
forecast 2
actual 3
forecast 4
```

The v0.2.11 ER-SDE solver-space bridge handled the original one-anchor bootstrap and later two-anchor lambda-space extrapolation, but deliberately rejected two *consecutive* actual anchors as extrapolation history. That made the first post-prefix forecast fall back to the older direct pending-`q` correction path.

Real Model Preview Override output reproduced the characteristic high-frequency/confetti corruption exactly on that first forecast.

v0.2.15 closes that gap:

- any causal forecast immediately following an exact actual anchor is eligible for solver-space handling;
- when the retained anchors are consecutive warmup/prefix actuals, the first forecast uses the newest exact solver-space denoised anchor as a hold;
- later forecasts retain the existing bounded native-ER-SDE lambda-space extrapolation once the anchor geometry supports it;
- unexpectedly missing extrapolation coordinates fall safe to the newest exact actual hold rather than reintroducing the noisy causal forecast;
- offline replay deliberately retains its separate exact pending-`q` compensation path so replay-smoothed hidden features remain observable to the sampler;
- the native ER-SDE RNG stream, stochastic latent ownership, `s_noise`, stages 1/2/3, and no-extra-NFE contract are unchanged.

The seeded replay regression now protects the actual invariant — byte-identical native stochastic draws — without incorrectly requiring the causal first-pass and replay denoised trajectories to be identical when their solver-facing policies intentionally differ.

## Runtime validation

The reconciled Continuum interop in PR #61 passed the complete repository CI matrix and was then validated on restarted ComfyUI 0.33.0 with the current H3 continuation path:

- the initial chunk ran without a Continuum prefix request;
- continuation chunks accepted `H3 Continuum API v1, actual prefix=2` exactly once each;
- steps 0 and 1 were actual evaluations with reason `H3 Continuum actual prefix`;
- step 2 returned to normal Spectrum forecast policy;
- no duplicate prefix NFEs or cross-chunk state leakage occurred;
- current-Core `video_audio` continuation references (`ref_audio` + `ref_img`) were exercised;
- Diff-Aid coexistence completed with zero external-patch contract failures;
- sampling, VAE decode, Continuum assembly, downstream processing, and final video output completed.

The ER-SDE prefix fix in PR #62 passed the full five-lane matrix — four reviewed ComfyUI revisions on Python 3.12 plus the current Python 3.13 lane — including forecaster smoke, Ruff/compileall, focused compatibility suites, and the full native MiniMax H3 tests. CodeRabbit's final review reported no actionable findings.

A final real Continuum + native ER-SDE runtime gate used **2 × 5-second chunks** with Model Preview Override. The previously reproducible third-step / Spectrum `step=2` confetti was gone.

## Compatibility and release scope

This release preserves the v0.2.14 native ER-SDE offline-replay/KJ preview protection, v0.2.13 Python 3.13 provenance normalization, v0.2.12 Diff-Aid interoperability, and the existing model-aware/generic-correction defaults.

Spectrum only consumes Continuum's declared API v1 metadata; it does not vendor or patch H3 Continuum itself. The installed Continuum version must independently support the installed ComfyUI H3 runtime.

There is no change to the normal Spectrum defaults, ordinary non-Continuum schedules, native ER-SDE random stream, or the standard transformer NFE budget outside the explicit Continuum prefix request.
