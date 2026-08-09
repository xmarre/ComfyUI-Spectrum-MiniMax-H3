# Spectrum MiniMax H3 v0.2.4

Restores useful live MiniMax H3 TAE previews while the default offline smoothing replay path is running.

## Live preview repair

- Restores KJNodes `Model Preview Override` updates during the compute-heavy capture pass and the accepted replay pass.
- Supports either ordering of `Model Preview Override` and Spectrum in the model patch chain.
- Capture previews show the provisional local-only trajectory. Replay previews update the widget with the accepted smoothed trajectory.
- Keeps ordinary external sampler callbacks and their previews replay-only, so unrelated callback side effects still run once per logical step.
- Preserves the existing wrapper order when offline smoothing is disabled.

## Root cause

Offline replay intentionally replaces the external callback during capture with a progress-only callback. When KJNodes' preview wrapper enclosed Spectrum's two-pass wrapper, it received callbacks only during the transformer-free replay, which usually completed too quickly to act as a useful live preview. Spectrum now places that observational wrapper immediately inside its own wrapper for offline runs, giving it one callback stream in each pass.

## Compatibility

This release does not change node inputs, saved workflows, forecast schedules, audio handling, progress accounting, or generated tensors. Kijai's MiniMax H3 TAE still requires KJNodes `Model Preview Override`.

## Validation

- A live user test of the PR confirmed that MiniMax H3 TAE previews work again.
- Regression tests cover both wrapper orders and verify that single-pass ordering remains unchanged.
- The full CPU suite passes with 175 tests; the two CUDA-only tests are skipped.
- All three GitHub Actions ComfyUI compatibility jobs and the forecaster smoke test pass.
