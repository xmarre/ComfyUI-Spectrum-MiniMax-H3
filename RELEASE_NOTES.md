# Spectrum MiniMax H3 v0.2.7

Adds opt-in model-aware forecasting controls, retains the useful generic residual correction discovered during that work, and fixes the ER-SDE odd-step terminal replay artifact found during final real-generation validation.

## Experimental model-aware forecasting

A new `model_aware_mode` input is available and defaults to `off`, so existing workflows keep legacy Spectrum behavior unless explicitly enabled.

- `off`: legacy Spectrum scheduling, fitting, blending, and replay.
- `schedule`: adds a compact model/patch risk prior that may turn a prospective forecast into an actual transformer evaluation.
- `schedule_confidence`: also adapts fit confidence, ridge/degree selection, and modality-specific blend.
- `full`: adds the retained bounded generic latest-delta hidden-residual correction on top of scheduling and confidence.

`model_aware_risk_threshold` defaults to `0.65`. Existing hard sampler, warmup, history, refresh, and tail rules still take precedence.

The profile is deliberately compact. It retains scalar sensitivities and patch metadata rather than detached output-head tensors, uses a bounded process cache, and does not add transformer NFEs by itself.

## Feature-3 result: model-specific correction retired

The model-informed correction experiments were completed rather than shipped speculatively. Exact/diagonal head metrics, K=2 trajectory corrections, transformed directions, previous hidden residual persistence, and static/current-FinalLayer adjoint variants did not produce a material model-specific improvement over the generic baseline in the real base-H3 gates.

Those model-specific correction paths are therefore retired from normal runtime. `full` applies only the generic latest-delta correction:

```text
d = h[-1] - h[-2]
r = h_actual - h_pred_uncorrected
g_raw = <r, d> / <d, d>
h_corrected = h_pred + bounded(g_raw) * d
```

In the final same-seed correction gate, that generic correction improved the aggregate hidden forecast-ratio metric by about 6.2% for audio and 5.5% for video. This is a generic trajectory-correction result, not evidence of a successful model-specific Feature 3 and not a claim of an equivalent perceptual-quality percentage.

The complete experimental record remains in `MODEL_AWARE_BENCHMARK.md`.

## ER-SDE terminal replay protection

Final validation exposed a step-count parity failure in native `er_sde` with offline replay: a 25-step run could leave the penultimate logical step forecasted while the final step was actual. Replay then had to reconstruct the penultimate feature across the last wide, nonlinear anchor interval, producing severe visible artifacts in the reproduced case. The failure also reproduced with `model_aware_mode=off`, isolating it from the model-aware features.

ER-SDE now enforces a minimum two-logical-step actual tail. On the failing 25-step schedule the end changes from:

```text
22 actual
23 forecast
24 actual
```

to:

```text
22 actual
23 actual
24 actual
```

The protected 25-step real run completed artifact-free with `model_aware_mode=full`, 14 actual transformer calls and 11 forecast steps. Schedules that already end with two actual logical steps do not gain another forced evaluation; the reproduced odd-step case costs one additional actual NFE.

Euler, MiniMax H3 Turbo, and deterministic RES/CFG++ sampler policies are unchanged by this ER-SDE-specific guard. RES continues to enforce its existing three-step protected tail.

## Runtime and telemetry cleanup

- Removes retired exact/diagonal/K=2 experiment payload from the normal shipping path.
- Keeps the compact scalar model/patch profile and bounded cache.
- Keeps generic correction telemetry explicit and marks retired model-informed Feature-3 paths as retired.
- Preserves offline replay decisions without archiving model-profile tensors.
- Keeps model-aware failures fail-safe: risky/invalid model-aware construction falls back to actual work rather than aborting sampling.

## Compatibility

- `model_aware_mode` defaults to `off`; existing workflows remain on legacy behavior unless changed.
- Existing offline smoothing, audio default (`audio_blend_weight=0`), Euler, Turbo, RES multistep, RES CFG++, native ER-SDE seeded replay, previews, storage controls, and native fallback behavior remain supported.
- Custom replay-unsafe ER-SDE stochastic components continue to fail closed to one valid native pass.

## Validation

- Final real-generation 25-step native ER-SDE gate: artifact-free after protecting the penultimate logical step.
- The reproduced 25-step first pass changed from 13 actual / 12 forecast to 14 actual / 11 forecast exactly as intended.
- 20-step and 32-step ER-SDE outputs used during diagnosis were artifact-free; focused tests pin the two-actual-tail invariant across 20, 25, and 32 steps.
- GitHub Actions run #163 passed the four reviewed ComfyUI revisions, including wheel build, forecaster smoke, scoped Ruff, `compileall`, and full pytest.
- Representative matrix result: 235 passed, 4 skipped; the skips are existing CUDA-only tests on CPU runners.
- All current CodeRabbit review threads are resolved.
