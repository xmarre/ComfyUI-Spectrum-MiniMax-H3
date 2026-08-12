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
| D | `full` | Exact linear `FinalLayer` head-space correction contribution |
| E | Spectrum disabled | Native reference |

For equal-NFE comparison, create three separate legacy controls: A-B matches B's actual transformer-call count, A-C matches C's count, and A-D matches D's count for the same seed. Record the actual step IDs as well as the total count for every pair. If the legacy controls cannot express an exact target count, mark that pair's equal-NFE cell unavailable. Do not reuse one A run for variants with different NFE totals. For equal-wall-clock comparison, choose the closest completed run for each pair without changing prompt, seed, model, or sampler, and report the residual timing mismatch.

## Measurements

Capture the debug summary plus:

- actual transformer NFEs, forecast steps, adaptive extra NFEs, discarded/replayed calls;
- sampler wall time and total generation wall time;
- profile miss/hit time, temporary workspace estimate, retained profile bytes, and model-aware per-step overhead;
- peak allocated/reserved VRAM and peak process RSS;
- sampled counterfactual forecast/hold error and curvature at every actual anchor, separately for audio and video;
- the explicit aggregate rule and value consumed by the scheduler (`max(audio, video)` in this revision);
- generic Euclidean, Gram-diagonal, and exact linear-head-space projections for each stream;
- raw and smoothly bounded generic/diagonal/exact gains, applied trust-mixed gain, bound-active flags, exact-versus-generic deltas, and exact-versus-diagonal deltas;
- per-stream applied, pure generic, pure diagonal, and pure exact error ratios in ordinary feature RMS and exact linear-head RMS;
- per-stream online exact-candidate trust, eligible comparison count, exact wins, and exact losses;
- anchor-evidence timing split into temporal-weight fitting, sample/index selection, one-time head materialization, exact-head projection, scalar device transfer, reductions/error calculations, and fit-condition calculation;
- retained generic-sample and exact-head evidence bytes, device-materialized head bytes, exact-row temporary workspace, and the separately configured full-history/archive memory;
- causal correction time and offline-replay correction-weight construction time/application count as separate fields;
- decoded same-seed video, audio, and synchronization assessments using a blinded ordering where practical.

Repeat each timing after one warm-up and include a second generation with the identical effective patch set to measure cache benefit. Run both cache measurements in the same process with the same clone lineage, patch-set identity and strengths, H3 architecture, and bypass-adapter metadata. Do not clear or evict the process-local profile cache between them. Report every seed, repetition, failure, and cancellation.

When a runtime/bypass LoRA loader is upstream, run one mode that executes the loader and then change only `model_aware_mode` so ComfyUI reuses the cached MODEL output. The second profile must retain the same patch/adaptor identity, active patch and key counts, recognized/unknown counts, perturbation, and cache key. A zero-patch profile after cached reuse invalidates the model-aware comparison.

## Pilot observations

The first real 8-step seeded ER-SDE pilot used one workflow/configuration/seed and produced three anchor updates. `schedule` and `schedule_confidence` retained the same 5-actual/3-forecast schedule, confidence mode changed ridge/blend values, and `full` applied correction. At every measured anchor, the model-scaled and generic corrected ratios were identical. The learned projections were large and negative, and both gain paths reached the existing `-0.25` bound, so the clamp erased their pre-clamp difference. This does not disprove the model-sensitivity hypothesis; that pilot could not observe its incremental effect after saturation.

A later 20-step `off` then `full` pilot reported risk up to `0.575384` (below the unchanged `0.65` threshold), nine saturated `-0.25` learned corrections, identical model/generic corrected ratios, and about 3.98 seconds of anchor-evidence work for nine anchors. About 2.756 seconds of the reported 2.904-second model-aware evidence total was device transfer. Its `full` profile also incorrectly reported zero patches after ComfyUI reused a cached output from `RuntimeBypassDoraPowerLoraLoader`, so its weight-prior result is invalid. That exposed a profiler lifetime bug: the loader persists effective adapters as bypass-hook objects in `ModelPatcher.injections`, while the old profiler recognized only manager objects exposing an `.adapters` dictionary. The corrected benchmark must verify persistent hook-derived profile identity before interpreting model-aware results.

The first channel-resolved correction architecture retained the generic Euclidean projection as control and used the normalized hidden-channel diagonal of the corresponding native H3 `FinalLayer` head Gram matrix. It added a monotonic rational ±0.25 soft bound and trajectory-calibrated trust. Evidence samples remained in a bounded device-local history and only reduced scalars returned to CPU.

A clean paired 20-step base-H3 `schedule_confidence` / `full` run then held the complete 11-actual / 9-forecast schedule, step IDs, risk/confidence/ridge/blend evolution, and raw anchor ratios fixed. Feature 3 reduced the mean audio ratio from `1.794656` to `1.688162` (5.93%) and video from `1.382575` to `1.309622` (5.28%), establishing that residual correction was useful at matched NFE. Almost all of that came from the generic candidate. Audio generic/diagonal/applied means were `1.688749` / `1.687578` / `1.688162`; video was `1.308225` / `1.311046` / `1.309622`. The diagonal produced only about 0.069% pure audio improvement and was about 0.216% worse for video, with 5/3 audio wins/losses and 1/7 video wins/losses. This is a real negative result for the diagonal approximation, not for generic correction.

The current experiment therefore keeps generic and diagonal candidates independently auditable and makes the full linear output-head operator primary. It samples complete hidden rows and computes `<rW^T,dW^T>/<dW^T,dW^T>` without materializing `W^T W`. The result is exact for the linear head and remains explicitly incomplete for timestep RMSNorm/AdaLN.

## Immediate base-H3 Feature 3 gate

Use base H3 with no active LoRA for the next correction benchmark. Run `schedule_confidence` and `full` with the same seed and unchanged workflow. Require the same actual/forecast step IDs and total transformer NFE count; stop the comparison if the schedules differ. Do not change `model_aware_risk_threshold` or add refreshes for this gate.

For every eligible actual anchor and separately for audio/video, record:

1. generic, diagonal, and exact raw gains; all smoothly bounded candidates; exact-versus-generic and exact-versus-diagonal post-bound deltas; trust; and final applied gain;
2. pure generic, diagonal, exact, and applied counterfactual error in ordinary feature RMS and exact linear-head RMS;
3. cumulative eligible exact-versus-generic comparisons, exact wins, and exact losses;
4. exact trust before the applied forecast and ending trust after the observed anchor;
5. total model-aware evidence time, one-time head materialization, exact-head projection time, scalar transfer, reductions, retained generic/exact evidence bytes, head device bytes, temporary workspace, and the `full - schedule_confidence` wall-clock delta;
6. decoded video, audio, and synchronization comparison between paired C/D outputs.

The exact candidate is supported only when `model_aware_head_metric_available=True`. A zero post-bound exact/generic delta with unequal raw candidates, hidden-feature CPU transfer in the exact path, a changed C/D NFE schedule, or a missing head operator invalidates the run. A falling trust value and zero applied exact delta are valid negative results when the pure exact candidate loses. Compare exact directly with diagonal to determine whether cross-hidden-channel Gram terms added useful information.

## Required ablations

Run, or explicitly mark unavailable:

1. trajectory only (neutral profile prior);
2. weight prior only (before live calibration);
3. trajectory plus correct profile;
4. trajectory plus a deliberately mismatched profile;
5. pure exact linear-head candidate, Gram-diagonal ablation, applied trust-mixed correction, and generic Euclidean baseline;
6. base model, strong single LoRA, and stacked LoRAs;
7. Euler, deterministic RES/CFG++ where applicable, Turbo, and native seeded `er_sde`.

The current public node has no neutral-profile, mismatched-profile, profile-injection, or calibration-freeze control. Consequently, ablations 1-4 are unavailable for end-to-end real-checkpoint runs in this branch. Unit tests can inject profiles directly into `ModelAwareController`, but that does not constitute the required sampling harness. Keep these cells unavailable until a benchmark-only harness can inject a selected immutable profile before `start_run`, optionally replace it with a neutral or deliberately mismatched profile, and freeze `observe_anchor` calibration for the full run without changing ordinary node behavior.

Evaluate each hypothesis from its paired per-seed deltas. Reject the weight-prior hypothesis when the upper bound of the paired 95% confidence interval fails to exceed the 2% forecast-error tolerance against both neutral and mismatched profiles at matched NFE. Reject the exact-head correction hypothesis by the same rule against both generic and diagonal candidates after applying the timing tolerance above. Report exact-versus-generic and exact-versus-diagonal post-bound deltas, pure-candidate win rates, ending trust, and applied-gain delta even when the perceptual result is inconclusive. If the required ablation cells are unavailable or the interval crosses the tolerance boundary, report the hypothesis as untested or inconclusive. A higher NFE count alone is not evidence of a better scheduler.

## Result status

No full MiniMax-H3 checkpoint is present in the automated environment used to implement this branch. Real-checkpoint quality, equal-NFE, equal-wall-clock, VRAM, and end-to-end timing cells are therefore pending and must not be inferred from unit fixtures. The branch remains experimental until those cells are populated.
