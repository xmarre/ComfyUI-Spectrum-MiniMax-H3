# Model-aware forecasting benchmark protocol

This document records the experimental evidence for `model_aware_mode` and the gates used before a model-informed correction is allowed to affect generated output. The branch remains experimental. Real-checkpoint results take precedence over synthetic fixtures.

## Fixed benchmark rules

Record the exact ComfyUI commit, node commit, MiniMax-H3 checkpoint and precision, LoRA state, prompt, reference-media hashes, seed, sampler, scheduler, steps, resolution, frame count, CFG, Spectrum controls, device, PyTorch build, storage mode, and warm-up procedure.

A mechanical same-seed correction comparison is valid only when the compared runs retain the same actual/forecast step IDs, total transformer NFE count, Feature-2 risk/confidence values, adaptive ridge, and adaptive blend decisions. Any mismatch invalidates the correction comparison.

Treat forecast-error deltas below **2%** as sub-material. Win/loss counts are supporting evidence only.

## Mode boundaries

| Mode | Enabled experiment layers | Feature-3 correction work |
|---|---|---|
| `off` | legacy Spectrum | none |
| `schedule` | Feature 1 model/profile scheduling prior | none |
| `schedule_confidence` | Feature 1 + Feature 2 trajectory confidence/adaptive fit | none |
| `full` | Feature 1 + Feature 2 + Feature 3 | retained scalar correction applied; any new hypothesis telemetry-only until gated |

`schedule_confidence` must report zero K=2 solve work, zero model-direction work, zero exact-head correction projection, and zero correction applications. `full` must not alter scheduler/NFE policy merely because Feature 3 is enabled.

# Feature-3 experimental history

## 1. Gram-diagonal scalar metric

The first model-specific scalar metric approximated the FinalLayer head metric with

```text
S = diag(W^T W)
```

while applying the correction along the latest temporal trajectory delta `d`.

A clean base-H3 run showed that the generic temporal residual correction itself was useful, but the diagonal model increment was small and qualitatively unreliable. The diagonal experiment produced approximately 5/3 audio wins/losses and 1/7 video wins/losses, with roughly +0.069% audio and -0.216% video versus the generic scalar correction.

Conclusion: the diagonal approximation discarded important cross-hidden-channel geometry.

## 2. Exact static-head scalar metric

The next experiment kept complete static output-head geometry without constructing `W^T W`:

```text
R = r @ W.T
D = d @ W.T
g_exact = <R, D> / <D, D>
```

One clean real run produced 8/0 exact-head wins for both audio and video while the diagonal video candidate remained 1/7. The mean exact-head scalar advantage over generic scalar was only approximately 0.22% for audio and 0.06% for video.

**Full output-head cross-channel geometry fixed the diagonal approximation's qualitative error, but scalar gain changes alone remained sub-material.**

This closed the question of repeatedly inventing new scalar weighting metrics.

## 3. K=2 recent-trajectory subspace

The next hypothesis expanded the correction span:

```text
d0 = h[-1] - h[-2]
d1 = h[-2] - h[-3]
```

Generic K=2 solved a Euclidean two-direction least-squares problem. Exact K=2 solved the same span in static output-head space.

A clean 20-step base-H3 ER-SDE `full -> schedule_confidence -> full` gate used 11 actual steps, 9 forecast steps, 11 transformer calls, zero extra NFEs, and identical Feature-2 scheduling behavior.

| Stream | generic scalar | exact scalar | scalar applied | generic K=2 | exact K=2 | applied K=2 |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 1.641362 | 1.640962 | 1.641162 | 1.645173 | 1.644676 | 1.644899 |
| Video | 1.277945 | 1.277594 | 1.277769 | 1.281283 | 1.281104 | 1.281193 |

Generic K=2 was approximately 0.23% worse than generic scalar for audio (0/7) and 0.26% worse for video (1/6). Exact-head weighting changed K=2 only approximately +0.03% for audio and +0.01% for video. Exact K=2 versus exact scalar was 1/6 for both streams.

**The second recent trajectory direction did not improve Feature 3. Generic K=2 was consistently worse than the generic scalar correction, and exact-head weighting did not rescue it. Therefore increasing temporal trajectory rank is not the next direction of work.**

K=2 is retired from normal runtime. Historical math/tests remain only where useful for regression/documentation.

### K=2 runtime-retirement result

The first post-retirement real gate verified that `schedule_confidence` now reports:

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
```

The old approximately 2.17 s K=2 solve attribution in `schedule_confidence` disappeared. The literal 2x2 solve was not plausibly the underlying cost; the old timing boundary included asynchronous CUDA work and host dependency attribution. No `torch.cuda.synchronize()` was added merely for measurement.

# 4. Transformed temporal trajectory directions

The next family changed the correction **direction** rather than the scalar gain or recent temporal rank.

Static candidate:

```text
m_W_raw = W^T W d
```

Full local FinalLayer candidate:

```text
m_F_raw = J_t^T J_t d
```

where `J_t = dF_t/dh` for native H3 FinalLayer at the target timestep. `J_t` was never materialized; the experiment used analytic FinalLayer-only JVP/VJP and no extra transformer NFE.

## Initial unnormalized diagnostic run

The first real screen reported approximately:

| Stream | static ratio | static vs generic | full ratio | full vs generic |
|---|---:|---:|---:|---:|
| Audio | 1.691852 | 0/8, -0.070682 | 1.694189 | 0/8, -0.072172 |
| Video | 1.374858 | 0/8, -0.060498 | 1.375616 | 0/8, -0.061091 |

Static raw direction norm ratios were approximately 1.875676 audio and 2.595429 video. Full `J_t^T J_t d` was printed as `0.000000` at six decimals while finite alpha values and pathological `1e-4`-class radial scales were reported.

This run was retained as a **diagnostic failed experiment-design run**, not an efficacy rejection.

### Radial numerical bug found

The old implementation effectively used:

```text
q_bounded = q / (1 + q/L)
radial_scale = q_bounded / clamp_min(q, eps)
```

For `0 < q < eps`, this incorrectly produced approximately `q/eps` instead of the correct `radial_scale -> 1` limit.

The stable implementation is:

```text
radial_scale = 1 / (1 + q/L)
q_bounded = q * radial_scale
```

### Arbitrary operator-scale confound found

Raw `W^T W d` and `J_t^T J_t d` have arbitrary operator-dependent magnitudes. Least-squares alpha is scale-invariant only before the alpha soft limiter. The limiter therefore mixed orientation with operator magnitude.

The corrected experiment normalized every finite nonzero model direction to delta-equivalent norm before alpha fitting, EWMA calibration, radial budgeting, and scoring:

```text
m_normalized = m_raw * ||d|| / ||m_raw||
```

Positive rescaling of a direction then leaves the normalized candidate/final correction invariant within floating-point tolerance.

# Corrected normalized transformed-direction gate: FINAL RESULT

The corrected real benchmark ran exactly:

```text
full
schedule_confidence
full
```

with one fixed saved base-H3 workflow and seed, no active LoRA, native `sample_er_sde`, 20 steps, and otherwise identical settings.

All three first passes used:

```text
actual:   0,2,4,6,8,10,12,14,16,18,19
forecast: 1,3,5,7,9,11,13,15,17
actual transformer calls = 11
forecast steps           = 9
extra transformer NFE    = 0
risk max                 = 0.568715
confidence min           = 0.431285
```

Degree, ridge, stream blends, and actual/forecast decisions were identical. The efficacy comparison is mechanically valid.

## Normalization/radial sanity gate passed

Every eligible static/full candidate reported:

```text
normalized_direction_norm_ratio ~= 1.0
```

Raw model-operator scales were:

| Stream | static `||W^T W d||/||d||` mean | full `||J^T J d||/||d||` mean |
|---|---:|---:|
| Audio | 1.968860e+00 | 8.879409e-09 |
| Video | 2.728169e+00 | 1.992571e-08 |

The extremely small full-FinalLayer direction magnitude was therefore genuine native local geometry, not just six-decimal logging loss.

Corrected full-direction radial scales were:

```text
Audio: min 0.784770, mean 0.864360
Video: min 0.929963, mean 0.951673
```

The old `1e-4` suppression disappeared. The normalization/radial fix is validated.

## Normalized static `W^T W d`: rejected

| Stream | candidate mean | vs generic scalar | mean advantage | vs exact scalar | mean advantage |
|---|---:|---:|---:|---:|---:|
| Audio | 1.653988 | 0/8 | -0.071455 | 0/8 | -0.073525 |
| Video | 1.394504 | 0/8 | -0.064070 | 0/8 | -0.064525 |

The approximately 6-7% regression remains after removing operator magnitude, alpha-scale, K=2 conditioning, and radial-bound confounds. The orientation itself is poor.

## Normalized full `J_t^T J_t d`: rejected

| Stream | candidate mean | vs generic scalar | mean advantage | vs exact scalar | mean advantage |
|---|---:|---:|---:|---:|---:|
| Audio | 1.655349 | 0/8 | -0.072343 | 0/8 | -0.074414 |
| Video | 1.395260 | 0/8 | -0.064658 | 0/8 | -0.065114 |

It also failed to improve the static model direction:

```text
Audio full vs static: 0/8, mean advantage -0.000827
Video full vs static: 0/8, mean advantage -0.000551
```

Native RMSNorm, target-timestep AdaLN, and exact output-head local geometry therefore do not rescue the transformed temporal direction.

**The scale-normalization sanity gate passed. Both transformed directions were evaluated in delta-equivalent units. Static `W^T W d` and full `J_t^T J_t d` then lost every eligible comparison against the scalar baseline on both streams by approximately 6-7%. The transformed-trajectory-direction family is therefore rejected.**

Normal runtime no longer computes this rejected WtW/JtJ screen. Its row history, direction-alpha state, geometry capture, per-anchor transformed logging, JVP/VJP calls, direction workspace and evidence buffers are removed. Pure math helpers/tests remain only as historical regression fixtures.

Do not continue this family with K=3/K4, larger recent-delta bases, PCA/SVD, another WtW/JtJ variant, alternate normalization, transformer JVP/Jacobians, learned regressors, a looser correction bound, or alpha retuning intended only to rescue this mechanism.

# Retained scalar baseline

The latest corrected real `full` scalar summaries are:

| Stream | raw forecast | generic scalar | exact-head scalar | scalar applied |
|---|---:|---:|---:|---:|
| Audio | 1.721166 | 1.615845 | 1.612975 | 1.614404 |
| Video | 1.378964 | 1.300207 | 1.299703 | 1.299954 |

The generic temporal residual correction improves forecast error by approximately 6.2% audio and 5.7% video. That remains worthwhile.

The model-specific increment is much smaller:

```text
exact-head scalar vs generic:
    audio ~0.18%
    video ~0.04%

trust-mixed applied vs generic:
    audio ~0.09%
    video ~0.02%
```

**The retained scalar correction is useful, but its model-specific exact-head increment remains sub-material and does not satisfy the intended 2% Feature-3 materiality target.**

Do not describe the approximately 6% overall correction improvement as a model-specific Feature-3 improvement. Almost all of it is the generic temporal residual correction.

**Overall Feature-3 status: generic scalar residual correction remains useful (~6% audio / ~5-6% video in the latest run), while the model-specific scalar increment remains sub-material and all tested model-transformed trajectory directions failed. Feature 3 remains scientifically unresolved.**

# First-principles reassessment: previous-error adjoint

The rejected family always asked model geometry to transform the latest temporal trajectory direction `d`. A distinct hypothesis is to start from a **strictly observed previous forecast error** instead.

At completed actual anchor `k`:

```text
r_k = h_actual,k - h_pred_uncorrected,k

e_k = F_k(h_actual,k) - F_k(h_pred_uncorrected,k)
```

For a later target `t`, screen three directions:

```text
A. generic previous residual
   m_R = r_previous

B. static model adjoint
   m_W = W^T e_previous

C. current local FinalLayer adjoint
   m_J = J_t^T e_previous
```

This is not `J_t^T J_t d`: its source signal is an observed prior forecast error rather than another transformation of temporal trajectory motion.

## Source-grounded feasibility review

### Causality

GO. At observer/finalization for actual anchor `N`, the forecaster still contains only anchors `< N`; `forecaster.update(N)` occurs after model-aware anchor observation. The uncorrected predicted hidden for `N` can therefore be reconstructed solely from prior history and the same causal weights used by the scalar evidence path.

The target timestep and native FinalLayer geometry depend only on current timestep/static model state. The implementation captures this state **before actual N executes**, so candidate construction cannot depend on actual N.

Sequence:

```text
1. previous residual state from anchors < N already exists
2. reconstruct uncorrected prediction for N from history < N
3. construct m_R, m_W, m_J and apply only alpha learned from earlier anchors
4. score fixed candidates against actual N
5. compute instantaneous alpha diagnostics from N's residual
6. compute/store r_N and e_N for future anchors only
```

No same-anchor hindsight is permitted.

### Row/output correspondence

GO within one generation. Spectrum already rejects changes in target audio/video row count, hidden width, packed topology, and conditional branch identity. Native H3 target audio/video segments remain contiguous and stable. The new experiment additionally fixes deterministic complete-row sample indices per stream and fails closed if row count/branch count changes.

Native output dimensions are stable because the same stream heads remain in use:

```text
audio: 32
video: 96
```

### Computing `e_k`

GO. No new transformer call or full output archive is required. The existing full hidden history already stores each actual final-block target feature. For at most 32 deterministic complete rows per stream, the screen reconstructs the causal predicted hidden, then evaluates the native sampled FinalLayer difference:

```text
F_k(h_actual,k) - F_k(h_pred_uncorrected,k)
```

Only sampled residual tensors are retained.

The pre-existing residual-feedback probe cannot be reused as the authoritative source because it is optional, only runs on particular actual refreshes, and evaluates its own shadow/hold policy. The new screen instead reuses the forecaster history/weights/head infrastructure and retains only the small sampled residual state actually required.

### Cross-timestep interpretation

GO as a telemetry hypothesis, not as an identity. `e_previous` lives in a stable stream output coordinate basis (same sampled token rows and output channels). Applying `J_t^T` asks how that persisted output-space error would map back into the current target hidden tangent space.

Changing timestep/AdaLN means `e_previous` is not the exact current output error. That is the hypothesis being tested: short-horizon residual persistence plus current target adjoint transport. The current `J_t` explicitly incorporates current RMSNorm and target-timestep AdaLN scale. The static `W^T e_previous` ablation isolates the value of the output head from the additional current local normalization/modulation geometry.

### Critical generic baseline

The model-specific claim is deliberately harder than “beats the uncorrected forecast.” The primary attribution baseline is:

```text
m_R = r_previous
```

All three candidates use the same deterministic rows, same delta-equivalent normalization, same previous-anchor-only alpha calibration, and same 0.25 rational radial budget.

A model-adjoint candidate must beat the previous-hidden-residual baseline materially. Otherwise any improvement is simply residual persistence rather than model geometry.

### Memory

At the maximum 32 sampled rows per stream, FP32 persistent previous-error state for base H3 is:

```text
hidden residuals:
    2 streams * 32 rows * 5376 hidden * 4 bytes
    = 1,376,256 bytes

output residuals:
    audio: 32 * 32 * 4   = 4,096 bytes
    video: 32 * 96 * 4   = 12,288 bytes

persistent residual payload:
    1,392,640 bytes
    = 1.328125 MiB
```

Row indices add at most 512 bytes for two 32-element int64 index vectors. Current timestep geometry is transient and small; the existing `full` scalar path already owns the per-run FP32 audio/video head materialization.

No second full hidden history is retained. When history storage is system RAM, only selected rows are transferred for the telemetry calculation.

### Compute

At 32 sampled rows per stream and hidden width 5376:

```text
W^T e_previous:
    audio  = 32 * 32 * 5376  = 5.505M MAC
    video  = 32 * 96 * 5376  = 16.515M MAC
    total  = 22.020M MAC ~= 44 MFLOP
```

`J_t^T e_previous` has the same head backprojection order plus O(hidden) RMSNorm/AdaLN VJP work. Constructing the exact sampled current output residual adds another sampled FinalLayer projection of the same order. Candidate scoring remains sub-GFLOP-scale and is substantially cheaper than one 50-block MiniMax-H3 denoiser evaluation.

No production-GPU wall-clock claim is made until the real gate is run.

### Normalization/budget

GO. Each finite nonzero direction is normalized to the latest temporal-delta norm:

```text
m_normalized = m * ||d|| / ||m||
```

The already validated 0.25 rational radial budget is reused unchanged. `_DIRECTION_ALPHA_LIMIT` is unchanged. Positive direction rescaling does not affect the final candidate.

### ER-SDE and offline replay

GO. The screen runs only after/beside normal model evaluation and does not touch ER-SDE generator/noise-sampler/derivative state. Extra transformer NFE remains zero.

The experiment is telemetry-only. Offline replay continues to use the retained scalar correction and does not archive/apply previous-error candidates. The screen is skipped during replay; first-pass observations may train/score telemetry without affecting generated output.

## GO decision and isolated implementation

The source review satisfies the causality, correspondence, cost, attribution, sampler and replay requirements, so one telemetry-only screening experiment is implemented.

Active `full` output remains exactly the retained scalar latest-delta correction. The rejected transformed-trajectory screen remains runtime-retired.

The new screen reports, per stream:

- previous-hidden-residual eligibility/ratio/alpha/bounded correction;
- static `W^T e_previous` eligibility/ratio and static-head ratio;
- local `J_t^T e_previous` eligibility/ratio and current FinalLayer output-space ratio;
- raw/normalized direction ratios, radial scale and bound activity;
- residual vs generic scalar;
- static vs previous residual and generic scalar;
- full adjoint vs previous residual, static adjoint and generic scalar;
- wins/losses, mean paired relative advantage and max absolute advantage;
- evidence bytes, sampled workspace, compute/geometry/VJP timing;
- explicit zero extra transformer NFE.

No candidate is eligible for efficacy scoring until it has a coefficient calibrated from an earlier completed anchor.

# Next real gate

Run one mechanical gate only:

```text
1. full
2. schedule_confidence
3. full
```

Use the same saved base-H3 workflow and fixed seed, no active LoRA, native `sample_er_sde`, 20 steps, identical checkpoint/precision, scheduler, prompt/references, resolution, frames, CFG, Spectrum settings, storage mode, and debug setting.

Reject the comparison if actual/forecast step IDs, transformer NFE, Feature-2 risk/confidence, adaptive ridge or stream blends differ.

Verify first:

```text
feature3_transformed_trajectory_runtime=retired
feature3_direction_evidence_bytes=0
feature3_direction_workspace_bytes=0
feature3_jvp_enqueue_s=0
feature3_vjp_enqueue_s=0
feature3_error_extra_transformer_nfe=0
```

Then evaluate separately:

1. retained scalar control remains unchanged;
2. `m_R = r_previous` versus generic scalar;
3. `m_W = W^T e_previous` versus `m_R` and generic scalar;
4. `m_J = J_t^T e_previous` versus `m_R`, `m_W`, and generic scalar;
5. eligible normalized direction ratios are approximately 1;
6. evidence/workspace/compute remain sampled and much cheaper than a denoiser call.

The **2% materiality gate remains mandatory for model-specific value**. A model adjoint must materially beat the previous-hidden-residual baseline, not merely the raw forecast. Only after such a result should multi-seed validation be considered.
