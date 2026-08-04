# Spectrum MiniMax H3 v0.1.3

Stabilizes deterministic Euler and RES multistep audio-video forecasting and adds Comfy Registry publishing metadata.

## Changed

- Require one completed actual H3 evaluation after every RES multistep forecast so its retained `old_denoised` state is native before forecasting resumes.
- Enforce a three-step actual tail for deterministic RES, including saved workflows that request a smaller tail.
- Require one completed actual H3 evaluation after every Euler forecast to prevent late forecast streaks from accumulating temporal audio/video errors on short schedules.
- Keep ancestral Euler and RES variants on the native path because their injected noise breaks the forecaster's smooth deterministic trajectory assumption.
- Keep `tail_actual_steps=1` as the configurable Euler default while applying the sampler-specific RES floor internally.
- Add the xmarre Comfy Registry publisher metadata and publishing workflow.

## Highlights

- Forecasts selected post-transformer H3 features while preserving the native current-step output heads, reconstruction, sigma mapping, and return structure.
- Keeps runtime and history state isolated per model clone and rolls incomplete split-branch transactions back to a complete native step.
- Supports deterministic Euler and RES multistep sampling with sampler-specific post-forecast refresh and tail policies, plus explicit native fallback for stochastic samplers, unsupported samplers, incompatible topology, invalid forecasts, and multi-GPU parallel sampling.
- Bounds retained history on CPU and streams forecast accumulation in chunks to avoid persistent full-feature FP32 coefficient or right-hand-side tensors.
- Avoids redundant full-target GPU concatenation and CPU restacking on native single-call actual steps, and reports history/forecast timing counters in debug summaries.
- Leaves the separate FLUX-focused ComfyUI-Spectrum-Proper repository unchanged.

## Validation

- The local suite passes against native ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff`.
- Automated tests cover forecasting mathematics, scheduling, rollback, clone isolation, the actual ComfyUI loader shape, native-path equivalence, and zero transformer-block execution on forecast steps.
- Sampler contract tests verify one model call per deterministic solver step, RES's current/previous-denoised recurrence, ancestral noise injection, rollback-safe refresh state, and sampler-specific forecast spacing and tails.

## Current limits

The supplied full-checkpoint RES A/B indicates that a three-step actual tail removes the remaining slight artifacts. That measured 20-step run completed in `1:50` with 14 actual and 6 forecast steps, compared with the supplied `2:06` native baseline. The automated environment cannot decode a full MiniMax H3 generation, so broader prompt coverage and audiovisual quality remain real-generation validation items; timings can also vary with model warmup and GPU state.
