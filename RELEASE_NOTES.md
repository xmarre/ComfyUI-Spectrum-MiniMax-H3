# Spectrum MiniMax H3 v0.2.6

Adds proper native ComfyUI `er_sde` sampler support to Spectrum's ordinary path and the default offline smoothing replay path.

## Native ER-SDE integration

- Recognizes ComfyUI's exact native `sample_er_sde` implementation behind the `er_sde` KSampler name.
- Preserves the one-to-one mapping between Spectrum logical steps and ER-SDE model evaluations.
- Applies a conservative schedule of at most one consecutive forecast followed by one completed actual refresh.
- Keeps ER-SDE's solver-local denoised history, stage updates, callbacks, final sigma-zero behavior, and stochastic update order inside the native sampler loop.
- Supports the standard `offline_smoothing_replay=true` path with deterministic reconstruction of the native seeded noise sequence.
- Uses no additional forced actual tail beyond the configured user tail.

## Replay safety

ER-SDE's default noise sampler is recreated from the same ComfyUI seed on each sampler invocation. This allows capture and replay to reproduce the same stochastic innovations from identical inputs.

Custom `noise_sampler` or `noise_scaler` callables may retain mutable state that Spectrum cannot safely reconstruct. When either override is present, offline replay fails closed to one native pass. Ordinary single-pass Spectrum remains available because the one-model-call topology is unchanged.

The default-off `anchor_residual_feedback` and `selective_rollback_correction` experiments remain limited to their separately reviewed sampler contracts.

## Compatibility

Existing workflows retain their saved values. Euler, MiniMax H3 Turbo, RES multistep, RES multistep CFG++, offline smoothing, audio handling, progress reporting, and native fallback behavior are unchanged.

## Validation

- Confirmed in a real MiniMax H3 ER-SDE generation with the default offline smoothing replay path.
- 116 CPU tests passed; three CUDA-only tests were skipped in the CPU environment.
- Native sampler-contract tests passed against all three older pinned ComfyUI revisions.
- The four-version GitHub Actions matrix passed, including reviewed current ComfyUI master `27bca654eb9a70237d93f56a6ea336ab55f8925d`.
- CodeRabbit reported no actionable review findings.
