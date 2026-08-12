# Model-aware forecasting benchmark protocol

This document records the experimental evidence for `model_aware_mode` and the gates used before a model-informed correction is allowed to affect generated output. The branch remains experimental. Real-checkpoint results take precedence over synthetic unit fixtures.

## Fixed benchmark rules

Record the exact ComfyUI commit, node commit, MiniMax-H3 checkpoint and precision, LoRA state, prompt, reference-media hashes, seed, sampler, scheduler, steps, resolution, frame count, CFG, Spectrum controls, device, PyTorch build, storage mode, and warm-up procedure.

For a mechanical same-seed comparison, every compared run must retain the same actual/forecast step IDs, total transformer NFE count, Feature-2 risk/confidence values, adaptive ridge, and adaptive blend decisions. A mismatch invalidates the correction comparison.

Treat forecast-error deltas below 2% as sub-material. Win/loss counts are supporting evidence only. A 7/7 or 8/8 win count with a ~0.1% error delta is still a sub-material result.

## Mode boundaries

| Mode | Enabled experiment layers | Feature-3 correction work |
|---|---|---|
| `off` | legacy Spectrum | none |
| `schedule` | Feature 1 model/profile scheduling prior | none |
| `schedule_confidence` | Feature 1 + Feature 2 trajectory confidence/adaptive fit | none |
| `full` | Feature 1 + Feature 2 + Feature 3 | scalar applied correction plus telemetry-only model-direction candidates |

`schedule_confidence` must report zero K=2 solve work, zero model-direction construction, zero exact-head correction projection, and zero correction applications. `full` must not change the scheduler/NFE policy merely because Feature 3 is enabled.

## Feature-3 experiment history

### Gram-diagonal scalar metric

The first model-specific correction approximated the `FinalLayer` head metric with

```text
S = diag(W^T W)
```

and changed the scalar fit while continuing to apply the correction along the latest trajectory delta `d`.

A clean paired base-H3 run established that the generic residual correction itself was useful at matched NFE. The model-diagonal increment was small and the video result was qualitatively poor: approximately 5/3 audio wins/losses and 1/7 video wins/losses, with only ~0.069% pure audio improvement and ~0.216% video degradation relative to generic correction.

### Exact static output-head scalar metric

The next experiment retained the complete cross-hidden-channel geometry of each static native output head without constructing `W^T W`:

```text
R = r @ W.T
D = d @ W.T
g_exact = <R, D> / <D, D>
```

One clean real run produced 8/0 exact-head wins for audio and 8/0 for video while the diagonal video candidate remained 1/7. The mean exact-head scalar advantage over generic scalar was only approximately 0.22% for audio and 0.06% for video.

**Full output-head cross-channel geometry fixed the diagonal approximation's qualitative error, but scalar gain changes alone remained sub-material.**

This closed the question of repeatedly inventing new scalar weighting metrics.

### K=2 recent-trajectory subspace

The next hypothesis changed correction rank while keeping the same causal trajectory family:

```text
d0 = h[-1] - h[-2]
d1 = h[-2] - h[-3]
```

Generic K=2 solved the Euclidean two-direction least-squares problem. Exact K=2 solved the identical span in static output-head space.

The latest clean base-H3 `full -> schedule_confidence -> full` gate used 20 steps, ER-SDE, 11 actual steps, 9 forecast steps, 11 transformer calls, zero extra NFEs, and identical Feature-2 schedule/risk/confidence behavior.

Mean ordinary feature-RMS ratios were:

| Stream | generic scalar | exact scalar | scalar applied | generic K=2 | exact K=2 | applied K=2 |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 1.641362 | 1.640962 | 1.641162 | 1.645173 | 1.644676 | 1.644899 |
| Video | 1.277945 | 1.277594 | 1.277769 | 1.281283 | 1.281104 | 1.281193 |

Generic K=2 versus generic scalar produced 0/7 audio wins/losses and approximately 0.23% worse mean error. Video produced 1/6 and approximately 0.26% worse mean error. Exact K=2 versus generic K=2 was only approximately +0.03% for audio and +0.01% for video. Exact K=2 versus exact scalar produced 1/6 for both streams. The trust-mixed K=2 path was also slightly worse than the retained scalar applied path.

**The second recent trajectory direction did not improve Feature 3. Generic K=2 was consistently worse than the generic scalar correction, and exact-head weighting did not rescue it. Therefore increasing temporal trajectory rank is not the next direction of work.**

K=2 remains documented as a rejected hypothesis. Normal generation no longer computes its 2x2 Gram systems or applies its replay stencil. Small mathematical helpers/tests may remain as historical verification.

## K=2 timing finding

Warm real runs attributed roughly 2.17-2.24 seconds to `model_aware_subspace_solve_s`, including approximately 2.173 seconds in `schedule_confidence` where the K=2 correction was never applied.

The implementation review found that the timer enclosed tiny CUDA reductions/eigensystem/solve operations. CUDA work is asynchronous, so a small scalar operation that introduces a host dependency can inherit synchronization cost from earlier queued work. This makes the reported solve timer an attribution boundary rather than evidence that the literal 2x2 arithmetic takes hundreds of milliseconds.

The required fix is architectural: rejected K=2 runtime work is removed from the normal path, and `schedule_confidence` never enters Feature-3 correction construction. No `torch.cuda.synchronize()` is added to normal inference merely to improve timing attribution.

## Current Feature-3 hypothesis: transform the direction

**Use the actual model geometry to transform the correction direction itself rather than merely reweighting a scalar objective or adding more historical trajectory directions.**

The current applied `full` correction is restored to the retained scalar hierarchy:

1. generic Euclidean scalar;
2. Gram-diagonal scalar historical ablation;
3. exact static-head scalar;
4. exact-versus-generic trust-mixed scalar applied correction.

The two new direction candidates are telemetry-only.

### Static output-head direction

For the latest causal trajectory delta `d` and native stream head `W`:

```text
m_W = W^T W d
    = (d @ W.T) @ W
```

The production implementation never constructs the 5376 x 5376 Gram matrix. This candidate isolates the value of changing the hidden-space direction using complete static head geometry.

### Full native FinalLayer local direction

Let `F_t(h)` be the stream-specific native `FinalLayer` mapping at the target timestep and `J_t = dF_t/dh`. The second candidate is

```text
m_F = J_t^T J_t d
```

`J_t` is never materialized. The implementation uses an analytic FinalLayer-only JVP followed by an analytic VJP. It never differentiates through a transformer block and never adds a denoiser NFE.

For native H3:

```text
y = W * (RMSNorm(h) * (1 + scale_t) + shift_t) + b
```

The local geometry therefore includes exact RMSNorm, timestep AdaLN multiplicative scale, and the native FP32 audio/video output head.

## Causality and calibration

At completed actual anchor N, the counterfactual candidate is evaluated from the uncorrected spectral prediction that was constructible from anchors `< N`, the latest causal trajectory delta, and the target timestep's FinalLayer state. The future actual hidden at N is used only as scoring/training evidence after candidate construction.

The coefficient applied to candidate N is an EWMA learned from earlier completed anchors only. The instantaneous diagnostic

```text
alpha_N = <r_N, m_N> / <m_N, m_N>
```

is scored at N and then updates the calibration for later candidates. It never retroactively modifies candidate N.

The sampled current anchor is appended to the direction-evidence history only after all candidates for that anchor have been scored.

## Correction budget

For candidate direction `m` and coefficient `alpha`, form `c_raw = alpha * m` and measure

```text
q = ||c_raw|| / max(||d||, eps)
```

in the same sampled generic hidden-space geometry. Apply the same rational radial soft limit corresponding to 0.25 used by the scalar correction. A large `W^T W` or `J^T J` norm cannot grant a larger physical correction budget.

When `m == d`, the radial rule reduces to the existing scalar rational-bound semantics within numerical tolerance.

## Evidence and memory policy

The first directional screen uses deterministic complete-row sampling only. Sampled hidden rows remain on the feature device. Only reduced scalar telemetry returns to CPU.

The global profile cache retains detached CPU output-head tensors and diagonals only. It never retains model/module/GPU references. GPU head materialization is `full`-only and is released at run end/disable. The CPU audio/video heads are approximately 2.6 MiB total for base H3 (`32 x 5376` and `96 x 5376`, FP32), which is retained because this has simpler and safer cache lifetime semantics than a cache that holds live model references.

Direction evidence is bounded by `max_history` and a small row sample. Debug output reports retained direction evidence bytes, per-run device head bytes, and peak sampled workspace.

## Telemetry required for the next gate

For each eligible stream/anchor, preserve the existing scalar ratios and report static/full model-direction eligibility, direction norm ratio, coefficient used, radial scale, bounded correction norm ratio, bound activity, ordinary-RMS ratio, static-head ratio, full FinalLayer/output-space ratio where available, and relative advantages against generic scalar, exact scalar, and the other model-direction candidate.

Run summaries must include eligible/fallback counts, mean/max ratios, mean/max relative advantage magnitude, wins/losses, direction computation time, JVP/VJP enqueue timing, scalar-transfer time, retained evidence bytes, temporary workspace, head materialization bytes/time, K=2 runtime work equal to zero, and no-extra-transformer-NFE confirmation.

Timing fields that do not introduce a synchronization boundary are labeled as enqueue timing. No explicit synchronization is added for measurement.

## Next real benchmark gate

Use one saved base-H3 workflow and seed, with no active LoRA:

1. `full`
2. `schedule_confidence`
3. `full`

Keep the checkpoint/precision, prompt and references, seed, native `sample_er_sde`, scheduler, 20 steps, resolution, frame count, CFG, all Spectrum controls, storage modes, and debug setting identical. Enable debug logging.

Reject the comparison if actual/forecast IDs, total transformer NFE, or Feature-2 risk/confidence/ridge/blend decisions differ.

Report separately:

- existing scalar correction: `schedule_confidence` raw versus scalar applied;
- static direction: `W^T W d` candidate versus generic scalar and exact-head scalar;
- full direction: `J_t^T J_t d` candidate versus generic scalar, exact-head scalar, and `W^T W d`;
- runtime: `schedule_confidence` overhead after K=2 removal, warm `full` overhead, sampled direction/JVP/VJP work, evidence memory/workspace, and head materialization.

Use 2% forecast-error improvement as the materiality gate. If both transformed directions remain at approximately 0.1% scale or are worse, stop this line of Feature-3 work and reassess from first principles. Do not proceed automatically to K=3, larger trajectory bases, PCA/SVD, learned regressors, transformer JVPs, full Jacobians, or another scalar metric.

## Result status

Automated tests validate mathematical equivalence, analytic derivatives, causal ordering, mode boundaries, rollback state, and no explicit Gram/Jacobian materialization. The automated environment does not contain the full MiniMax-H3 checkpoint. Real-checkpoint quality, VRAM, and production wall-clock conclusions remain pending the gate above.
