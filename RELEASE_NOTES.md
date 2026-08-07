# Spectrum MiniMax H3 v0.1.7

Adds an experimental one-point bootstrap forecast for degree-1 schedules and adopts the validated degree-1 setup as the preliminary default for new node instances.

## Added

- Add optional `bootstrap_first_forecast`, disabled by default, which can forecast solver step 1 from the actual step-0 packed H3 hidden feature.
- Route the explicit one-point `[1.0]` hold through the existing bounded prediction engine while preserving current-step timestep conditioning, output heads, branch mapping, shape checks, retry behavior, and accounting.
- Keep ordinary degree-1 regression unchanged: it still requires two actual history samples, never factorizes a one-point system, and never inserts forecasts into actual history.

## Preliminary defaults

- Set `degree=1` for new node instances.
- Set `warmup_steps=1` for new node instances.
- Keep `tail_actual_steps=1` as the requested native tail. Deterministic RES still enforces its sampler-safe three-step minimum.
- Keep `bootstrap_first_forecast=false`; enable it explicitly to use the new step-1 hold.

Existing workflows retain their saved input values.

## Full-checkpoint validation

A same-seed 20-step Euler run with `degree=1`, `warmup_steps=1`, `tail_actual_steps=1`, `bootstrap_first_forecast=true`, `window_size=2.0`, and `flex_window=0.75` completed with:

```text
A F A F A F A F A F A F A F A F A F A A
```

- 11 actual transformer calls and 9 forecasts.
- Zero fallbacks.
- 177.80 s sampler time, down from 324.98 s native: 45.29% reduction and 1.83x sampler throughput.
- 200.32 s full-prompt time, down from 340.59 s native: 41.19% reduction. The Spectrum run also included 5.65 s of upstream `MiniMaxH3ImageToVideo` work, so the full-prompt comparison is less controlled than the sampler measurement.
- 0.141 s total forecast prediction time and 3221.5 MiB of history in VRAM.

The bootstrap run was 6.52 s faster than an earlier non-bootstrap Spectrum run with the same 11 actual calls and 9 forecasts. That 3.54% difference is treated as run variance and/or sigma-position cost, since bootstrap did not reduce the transformer-call count at 20 steps. Odd step counts such as 17 or 19 are where the shifted schedule can remove one additional actual call.

## Automated validation

- Exact 17-step and 20-step Euler bootstrap schedules.
- Whole-step abort, retry, fallback, reordered-label, topology, shape, final-tail, refresh, and accounting invariants.
- One-point prediction exactness with no regression factorization or history mutation.
- Native H3 proof that the bootstrap skips transformer blocks while running the current step's output path.
- 106 tests passed with one CUDA-only VRAM/system-RAM parity test skipped on current ComfyUI `2340099d`, original H3 integration commit `e377e263`, and compatibility commit `0dd9b154`.
