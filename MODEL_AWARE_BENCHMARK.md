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

A clean base-H3 `full -> schedule_confidence -> full` gate used 20 steps, ER-SDE, 11 actual steps, 9 forecast steps, 11 transformer calls, zero extra NFEs, and identical Feature-2 schedule/risk/confidence behavior.

Mean ordinary feature-RMS ratios were:

| Stream | generic scalar | exact scalar | scalar applied | generic K=2 | exact K=2 | applied K=2 |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 1.641362 | 1.640962 | 1.641162 | 1.645173 | 1.644676 | 1.644899 |
| Video | 1.277945 | 1.277594 | 1.277769 | 1.281283 | 1.281104 | 1.281193 |

Generic K=2 versus generic scalar produced 0/7 audio wins/losses and approximately 0.23% worse mean error. Video produced 1/6 and approximately 0.26% worse mean error. Exact K=2 versus generic K=2 was only approximately +0.03% for audio and +0.01% for video. Exact K=2 versus exact scalar produced 1/6 for both streams. The trust-mixed K=2 path was also slightly worse than the retained scalar applied path.

**The second recent trajectory direction did not improve Feature 3. Generic K=2 was consistently worse than the generic scalar correction, and exact-head weighting did not rescue it. Therefore increasing temporal trajectory rank is not the next direction of work.**

K=2 remains documented as a rejected hypothesis. Normal generation no longer computes its 2x2 Gram systems or applies its replay stencil. Mathematical helpers/tests remain as historical verification only.

## K=2 runtime-retirement result

The first post-retirement real gate verified the intended mode boundary. In `schedule_confidence`:

```text
model_aware_subspace_gram_s=0
model_aware_subspace_solve_s=0
model_aware_subspace_workspace_bytes=0
model_aware_head_materialized_bytes=0
model_aware_exact_head_projection_calls=0
model_aware_correction_s=0
model_aware_offline_replay_correction_applications=0
feature3_direction_evidence_bytes=0
feature3_direction_workspace_bytes=0
feature3_direction_compute_s=0
feature3_static_enqueue_s=0
feature3_full_enqueue_s=0
feature3_jvp_enqueue_s=0
feature3_vjp_enqueue_s=0
```

The old ~2.17 s K=2 solve attribution in `schedule_confidence` is therefore gone. The remaining Feature-2 scalar-transfer timing is a separate existing synchronization/attribution issue and is outside the current Feature-3 correction revision.

Current debug summaries explicitly identify K=2 as retired. Legacy K=2 field names that remain for compatibility are marked `retired_k2_inactive`; zero-filled historical counters remain zero.

## Retained applied scalar correction

The generated `full` output continues to use the retained one-direction scalar correction only:

```text
d = h[-1] - h[-2]
```

The first post-retirement real gate confirmed that this path remained healthy:

| Stream | raw forecast | generic scalar | exact-head scalar | scalar applied |
|---|---:|---:|---:|---:|
| Audio | 1.762326 | 1.661408 | 1.659099 | 1.660249 |
| Video | 1.367852 | 1.296997 | 1.296834 | 1.296916 |

This is roughly a 5.8% audio and 5.2% video improvement over the raw forecast. The exact-head scalar increment remains small: about 0.14% for audio and 0.01% for video versus generic scalar.

The model-transformed directions below remain counterfactual telemetry only. Offline smoothing replay remains scalar-only.

## Current Feature-3 hypothesis: transform the direction

**Use the actual model geometry to transform the correction direction itself rather than merely reweighting a scalar objective or adding more historical trajectory directions.**

### Static output-head direction

For the latest causal trajectory delta `d` and native stream head `W`:

```text
m_W_raw = W^T W d
        = (d @ W.T) @ W
```

The production implementation never constructs the 5376 x 5376 Gram matrix.

### Full native FinalLayer local direction

Let `F_t(h)` be the stream-specific native `FinalLayer` mapping at the target timestep and `J_t = dF_t/dh`:

```text
m_F_raw = J_t^T J_t d
```

`J_t` is never materialized. The implementation uses an analytic FinalLayer-only JVP followed by an analytic VJP. It never differentiates through a transformer block and never adds a denoiser NFE.

For native H3:

```text
y = W * (RMSNorm(h) * (1 + scale_t) + shift_t) + b
```

The local geometry therefore includes exact RMSNorm, target-timestep AdaLN multiplicative scale, and the native FP32 audio/video output head.

## First transformed-direction screen: diagnostic, not a valid rejection

A clean real `full -> schedule_confidence -> full` gate used the same saved base-H3 workflow and seed, 20 steps, native ER-SDE, no active LoRA, 11 actual steps, 9 forecast steps, 11 actual transformer calls, zero extra transformer NFEs, and identical Feature-1/2 schedule/risk/confidence/ridge/blend behavior.

The initial unnormalized screen reported:

| Stream | static `W^T W d` ratio | static vs generic | full `J^T J d` ratio | full vs generic |
|---|---:|---:|---:|---:|
| Audio | 1.691852 | 0/8, mean advantage -0.070682 | 1.694189 | 0/8, mean advantage -0.072172 |
| Video | 1.374858 | 0/8, mean advantage -0.060498 | 1.375616 | 0/8, mean advantage -0.061091 |

Static raw direction norm ratios were not pathological: approximately 1.875676 for audio and 2.595429 for video. This initial static result is therefore evidence that the direction may genuinely be poor, but the static candidate is rerun under the same scale-invariant calibration as the full candidate before final rejection.

The full candidate cannot be rejected from this run. Later anchors reported `full_dir_norm_ratio=0.000000` at six-decimal precision while simultaneously reporting finite coefficients around -0.86 to -1.36, `full_bounded_norm_ratio=0.000000`, `full_radial_scale` in roughly the `1e-4` to `7e-4` range, and `full_bound_active=True`.

That exposed two coupled experimental confounds.

### Radial-bound numerical bug

The old implementation formed

```text
q_bounded = q / (1 + q/L)
scale = q_bounded / clamp_min(q, eps)
```

For `0 < q < eps`, `q_bounded ~= q` but the second expression becomes approximately `q/eps`, spuriously suppressing a correction that is already tiny. A rational upper bound must instead use the algebraically equivalent stable scale directly:

```text
scale = 1 / (1 + q/L)
q_bounded = q * scale
```

so `q -> 0` implies `scale -> 1`.

### Arbitrary operator-scale confound

The raw transformed directions have operator-dependent magnitude. Raw least squares is scale-invariant because, for `m' = c m`, the fitted coefficient changes by `1/c`. The experiment then soft-limited the instantaneous coefficient around the existing direction-alpha prior, destroying that invariance. For an extremely small `J_t^T J_t d`, an enormous coefficient may be required solely to cancel arbitrary operator scale, so the alpha limiter was testing orientation plus operator magnitude instead of orientation itself.

The first `J_t^T J_t d` screen is therefore retained as a diagnostic failed experiment-design run, not as an efficacy gate.

## Delta-equivalent direction normalization

Every finite, nonzero transformed direction is now normalized before coefficient fitting, EWMA calibration, radial budgeting, and candidate scoring:

```text
m_normalized = m_raw * ||d|| / ||m_raw||
```

The scale factor is positive, so orientation and sign are preserved exactly. For eligible candidates:

```text
||m_normalized|| ~= ||d||
```

The experiment now separates three quantities:

1. raw operator scale `||m_raw|| / ||d||`;
2. normalized model-derived orientation in delta-equivalent units;
3. final correction magnitude after coefficient calibration and the 0.25 radial budget.

The required positive-scale invariant is:

```text
normalize(c * m, d) ~= normalize(m, d),  c > 0
```

and, given the same calibration state:

```text
correction(c * m) ~= correction(m)
```

within floating-point tolerance. Negative scaling is not identified as equivalent; sign remains part of the direction.

The existing `_DIRECTION_ALPHA_LIMIT` is unchanged. After normalization, a coefficient such as `alpha=0.1` again means approximately a 0.1-delta-sized raw correction independently of whether the direction came from `W^T W d` or `J_t^T J_t d`.

## Stable correction budget

For normalized direction `m`:

```text
c_raw = alpha * m
q_raw = ||c_raw|| / ||d||
radial_scale = 1 / (1 + q_raw / 0.25)
q_bounded = q_raw * radial_scale
c_bounded = radial_scale * c_raw
```

Required limits:

- `q_raw -> 0`: `radial_scale -> 1`, `q_bounded ~= q_raw`;
- `q_raw = 1`: `q_bounded = 0.2`;
- `q_raw -> infinity`: `q_bounded -> 0.25`;
- the bound never increases correction magnitude;
- `m=d` preserves the existing scalar trust-budget interpretation.

Zero directions, non-finite tensors, invalid limits, non-finite normalization scales/results, and reference deltas below the existing finite-resolution threshold fail closed. Finite representable transformed directions are not rejected merely because their raw operator magnitude is below machine epsilon relative to `d`; that is precisely the case normalization must handle.

## Causality and calibration

At completed actual anchor N, the counterfactual candidate is evaluated from the uncorrected spectral prediction constructible from anchors `< N`, the latest causal trajectory delta, and the target timestep's FinalLayer state. The future actual hidden at N is used only as scoring/training evidence after candidate construction.

The coefficient applied to candidate N is an EWMA learned from earlier completed anchors only. The instantaneous diagnostic is now fit in normalized units:

```text
alpha_N = <r_N, m_normalized,N> / <m_normalized,N, m_normalized,N>
```

It updates calibration only for later candidates. The sampled current anchor is appended to direction history only after all candidates for that anchor have been scored.

## Native FinalLayer magnitude audit

The implementation was rechecked against native ComfyUI H3 before normalization was accepted as the remedy:

- native `FinalLayer.norm` RMSNorm weight and `eps` are captured from the actual module;
- native default `final_norm_eps` is `1e-5`;
- target video/audio modulation tags are 0/2 and are converted back to the native timestep row correctly;
- native semantics are exactly `RMSNorm(h) * (1 + scale[row]) + shift[row]`;
- video/audio output heads are the native FP32 heads;
- the JVP casts the modulated hidden to FP32 before the output projection, and the VJP mirrors the native dtype transition;
- the reference point is the uncorrected `predicted_native` hidden for the target coordinate;
- the RMSNorm derivative contains the required single hidden-dimension mean terms only;
- the JVP/VJP and `J^T J d` continue to match PyTorch autograd/explicit-Jacobian fixtures.

No second implementation error explaining the small full-direction magnitude was found in this audit. The previous real log printed the raw ratio only to six decimals, so its exact magnitude cannot be reconstructed after the fact. The corrected telemetry reports raw `J_t^T J_t d` norms and ratios in scientific notation on the next real run. The small scale is currently treated as plausible native local geometry, not as proof of a bug.

## Telemetry

Per eligible static/full candidate, debug logs now report separately:

```text
raw_direction_norm
reference_delta_norm
raw_direction_norm_ratio
normalized_direction_norm_ratio
instantaneous_alpha_raw
alpha_used
raw_correction_norm_ratio
bounded_correction_norm_ratio
radial_scale
bound_active
```

Raw-scale fields use scientific notation. Existing ordinary RMS, static-head RMS, FinalLayer/output-space RMS, wins/losses, relative advantages, timing, retained evidence bytes, workspace, head materialization, and no-extra-NFE telemetry remain.

Run summaries report raw direction-ratio min/mean/max, normalized direction-ratio min/mean/max, instantaneous-alpha min/mean/max, used-alpha mean/max-absolute, raw-correction mean/max, bounded-correction mean/max, and radial-scale min/mean.

Active labels are:

```text
feature3_applied_correction=scalar_latest_delta
feature3_direction_screen=static_head_and_full_final_layer_normalized
feature3_direction_units=delta_equivalent_norm
feature3_k2_runtime=retired
```

## Evidence and memory policy

The direction screen remains deterministic complete-row sampling capped at 32 rows per stream. Sampled hidden rows stay on the feature device. Only reduced scalar telemetry is transferred to CPU.

The global profile cache retains detached CPU output-head tensors/diagonals only and no model/module/GPU references. Per-run GPU heads are released at run end/disable. Base-H3 CPU output heads remain approximately 2.6 MiB total (`32 x 5376` and `96 x 5376`, FP32).

Normalization adds only sampled-vector norm/scaling work and one normalized sampled direction tensor per static/full candidate while it is being evaluated. It does not construct a full-resolution transformed direction, transfer full hidden tensors to CPU, add explicit CUDA synchronization, or add a transformer/denoiser NFE.

## Automated verification requirements

The complete suite must continue to cover the previous Feature-3 invariants, including analytic RMSNorm JVP/VJP, FinalLayer JVP/VJP, `J^T J d` versus explicit Jacobian, no hidden-width Gram materialization, off-diagonal `W` behavior, strict causal predicted-hidden reference, previous-anchor-only coefficient chronology, snapshot/rollback, no model/module retention, `schedule_confidence` zero Feature-3 work, scalar-only applied/replay paths, and no extra transformer calls.

The normalization revision additionally tests:

- tiny positive `q << eps` does not suppress radial scale;
- exact rational radial behavior from `q=1e-12` through `q=100`;
- normalized direction norm matches `||d||`;
- positive direction scaling invariance from `1e-12` through `1e6`;
- final bounded correction invariance under the same positive scalings;
- a synthetic tiny-scale `J^T J d` remains eligible, normalizes to delta-equivalent units, and does not receive a spuriously tiny radial scale;
- zero/non-finite directions, tiny reference deltas, and invalid bounds fail closed.

## Next real benchmark gate

Use the exact same saved base-H3 workflow and fixed seed as the diagnostic run, with no active LoRA:

1. `full`
2. `schedule_confidence`
3. `full`

Keep checkpoint/precision, prompt/references, seed, native `sample_er_sde`, scheduler, 20 steps, resolution, frame count, CFG, every Spectrum control, storage mode, and debug setting identical. Enable debug logging.

Reject the comparison if actual/forecast step IDs, transformer NFE, or Feature-2 risk/confidence/ridge/blend decisions differ.

Report separately:

### A. Scalar control

```text
raw
generic scalar
exact scalar
scalar applied
```

### B. Static direction after scale normalization

Report raw `W^T W d` norm ratio, normalized norm ratio, alpha used, raw correction norm ratio, bounded norm ratio, radial scale, and normalized candidate versus generic/exact scalar.

### C. Full FinalLayer direction after scale normalization

Report raw `J_t^T J_t d` norm ratio in scientific notation, normalized norm ratio, alpha used, raw correction norm ratio, bounded norm ratio, radial scale, and normalized candidate versus generic scalar, exact scalar, and normalized static candidate.

### D. Sanity gate

Eligible static/full candidates must show normalized direction norm ratio approximately 1. A tiny raw `J_t^T J_t d` magnitude must no longer produce `radial_scale ~= 1e-4` unless the normalized correction itself genuinely exceeds the trust budget.

Only after this corrected rerun is the 2% materiality stopping rule applied. If normalized `W^T W d` and normalized `J_t^T J_t d` remain materially worse than scalar correction or improve only at approximately 0.1% scale, this model-transformed-direction family has received a fair test and should be rejected. At that point Feature 3 is reassessed from first principles rather than automatically escalating to K=3, PCA/SVD, learned regressors, transformer JVPs, full Jacobians, or another scalar metric.
