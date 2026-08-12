# Model-aware forecasting benchmark protocol

This matrix is intentionally separate from implementation-unit results. Do not promote `schedule`, `schedule_confidence`, or `full` from experimental status until the same-seed real-checkpoint comparisons below are recorded.

## Fixed inputs

Record the exact ComfyUI commit, node commit, MiniMax-H3 checkpoint and precision, LoRA files and strengths, prompt, reference media hashes, prespecified seeds, sampler, scheduler, steps, resolution, frame count, CFG, Spectrum base controls, device, PyTorch build, and warm-up procedure. Run at least one base-model configuration and one materially strong LoRA configuration. Add a stacked-LoRA configuration when resources permit.

Use at least five independent seeds for each complete comparison matrix, paired across every compared variant. Report every paired result plus the median paired delta and a bootstrap 95% confidence interval. Establish timing repeatability with at least three control repetitions after warm-up; a timing delta is inconclusive when it is smaller than the larger of 3% or twice the control's median absolute deviation. Treat forecast-error deltas smaller than 2% as inconclusive. Perceptual video, audio, and synchronization results require blinded per-seed judgments and remain descriptive.

Use the same inputs for:

| ID | Spectrum configuration | Purpose |
|---|---|---|
| A | Legacy Spectrum, `model_aware_mode=off` | Current accelerator control |
| B | `schedule` | Scheduling contribution |
| C | `schedule_confidence` | Adaptive fitting/blend contribution |
| D | `full` | Model-scaled correction contribution |
| E | Spectrum disabled | Native reference |

For equal-NFE comparison, create three separate legacy controls: A-B matches B's actual transformer-call count, A-C matches C's count, and A-D matches D's count for the same seed. Record the actual step IDs as well as the total count for every pair. If the legacy controls cannot express an exact target count, mark that pair's equal-NFE cell unavailable. Do not reuse one A run for variants with different NFE totals. For equal-wall-clock comparison, choose the closest completed run for each pair without changing prompt, seed, model, or sampler, and report the residual timing mismatch.

## Measurements

Capture the debug summary plus:

- actual transformer NFEs, forecast steps, adaptive extra NFEs, discarded/replayed calls;
- sampler wall time and total generation wall time;
- profile miss/hit time, temporary workspace estimate, retained profile bytes, and model-aware per-step overhead;
- peak allocated/reserved VRAM and peak process RSS;
- sampled counterfactual forecast/hold error at actual anchors by audio/video stream;
- correction magnitude and model-informed versus generic-correction error ratios;
- decoded same-seed video, audio, and synchronization assessments using a blinded ordering where practical.

Repeat each timing after one warm-up and include a second generation with the identical effective patch set to measure cache benefit. Run both cache measurements in the same process with the same clone lineage, patch-set identity and strengths, H3 architecture, and bypass-adapter metadata. Do not clear or evict the process-local profile cache between them. Report every seed, repetition, failure, and cancellation.

## Required ablations

Run, or explicitly mark unavailable:

1. trajectory only (neutral profile prior);
2. weight prior only (before live calibration);
3. trajectory plus correct profile;
4. trajectory plus a deliberately mismatched profile;
5. model-scaled correction versus the logged generic residual correction baseline;
6. base model, strong single LoRA, and stacked LoRAs;
7. Euler, deterministic RES/CFG++ where applicable, Turbo, and native seeded `er_sde`.

The current public node has no neutral-profile, mismatched-profile, profile-injection, or calibration-freeze control. Consequently, ablations 1-4 are unavailable for end-to-end real-checkpoint runs in this branch. Unit tests can inject profiles directly into `ModelAwareController`, but that does not constitute the required sampling harness. Keep these cells unavailable until a benchmark-only harness can inject a selected immutable profile before `start_run`, optionally replace it with a neutral or deliberately mismatched profile, and freeze `observe_anchor` calibration for the full run without changing ordinary node behavior.

Evaluate each hypothesis from its paired per-seed deltas. Reject the weight-prior hypothesis when the upper bound of the paired 95% confidence interval fails to exceed the 2% forecast-error tolerance against both neutral and mismatched profiles at matched NFE. Reject the model-scaled-correction hypothesis by the same rule against generic correction after applying the timing tolerance above. If the required ablation cells are unavailable or the interval crosses the tolerance boundary, report the hypothesis as untested or inconclusive. A higher NFE count alone is not evidence of a better scheduler.

## Result status

No full MiniMax-H3 checkpoint is present in the automated environment used to implement this branch. Real-checkpoint quality, equal-NFE, equal-wall-clock, VRAM, and end-to-end timing cells are therefore pending and must not be inferred from unit fixtures. The branch remains experimental until those cells are populated.
