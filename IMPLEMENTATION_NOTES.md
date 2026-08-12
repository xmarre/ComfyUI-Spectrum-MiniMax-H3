# MiniMax H3 integration notes

Source review date: 2026-08-12

Reviewed native ComfyUI revisions used by the PR matrix:

- `e377e263049f9338b4d12a3dd417b36ae62948ff`
- `0dd9b154a1654fc699dcdc3af066c7cce096045a`
- `5599a05fea715cb2aff11f30f5b06e16d0dfa0c4`
- `27bca654eb9a70237d93f56a6ea336ab55f8925d`

## Native execution boundary

MiniMax-H3 sampling enters through the normal ComfyUI guider/sampler `PREDICT_NOISE` path. Native `MiniMaxH3Model._forward` resolves the packed layout, separate audio/video timestep state, modulation segments and embeddings, executes the transformer stack, then sends the final packed hidden to `FinalLayer`.

The target packed tail is contiguous:

```text
[target audio rows | target video rows]
```

Spectrum captures only that target tail immediately after the final transformer block and before `FinalLayer`. Forecast steps predict the same pre-`FinalLayer` target hidden and then use the current native output/reconstruction path.

## Native FinalLayer contract

The reviewed base architecture has hidden width 5376 and FP32 stream heads:

```text
audio_out: 5376 -> 32
video_out: 5376 -> 96
```

Effective stream mapping:

```text
n(h)   = RMSNorm(h)
z_t(h) = n(h) * (1 + scale_t) + shift_t
F_t(h) = W z_t(h) + b
```

The output-head island is FP32. Audio/video target timestep rows remain stable for the target packed segments within one generation.

## Forecast history and row correspondence

The normal forecaster stores at most `max_history` detached compact target features in configured history storage. Branch identity, topology, target row count and hidden width are validated before forecasting. Any change fails to actual execution rather than silently remapping rows.

This stability also supports sampled Feature-3 experiments: deterministic complete-row indices may be fixed for a stream and reused across actual anchors. The current telemetry screens cap the total selected complete rows at 32 per stream across conditional branches.

## Model profile lifetime

The model profile stores immutable metadata plus detached CPU copies of native audio/video output-head weights and their Gram diagonals. It retains no live model/module/GPU reference.

Base-H3 detached FP32 head payload:

```text
(32 + 96) * 5376 * 4
= 2,752,512 bytes
= 2.625 MiB
```

GPU head materialization is `full`-only and is released at run end/disable.

## Mode boundaries

```text
off
    legacy Spectrum

schedule
    Feature 1 profile/scheduling prior only
    no trajectory/correction evidence

schedule_confidence
    Feature 1 + Feature 2 trajectory confidence/adaptive fitting
    generic risk evidence only
    no exact-head correction projection
    no K=2
    no Feature-3 candidate screen
    no correction application

full
    Feature 1 + Feature 2 + Feature 3
    retained scalar latest-delta correction applied
    current new hypotheses telemetry-only until real gated
```

## Retained applied scalar correction

The generated `full` output continues to use only the existing one-dimensional causal correction along:

```text
d = h[-1] - h[-2]
```

The historical scalar measurements remain independently observable:

```text
g_generic = <r,d>/<d,d>
g_diag    = <r,Sd>/<d,Sd>,  S=diag(W^T W)
g_exact   = <rW^T,dW^T>/<dW^T,dW^T>
```

The actual applied scalar is the existing trust mixture between generic and exact-head scalar gains under the unchanged rational 0.25 correction limit. K=2 coefficients are explicitly ignored by forecast application.

The latest real gate showed approximately 6.2% audio / 5.7% video improvement from the generic temporal residual correction, while the model-specific exact-head increment remained only approximately 0.18% audio / 0.04% video and the trust-mixed applied increment approximately 0.09% / 0.02%. The generic benefit must not be mislabeled as model-specific Feature-3 efficacy.

## Retired K=2 runtime

The rejected K=2 experiment used the two most recent temporal deltas. Real evidence showed it was consistently worse than the scalar baseline and exact-head weighting did not rescue it. Normal runtime no longer performs K=2 Gram/solve work or replay stencils. Historical helpers/tests may remain.

## Retired transformed-trajectory runtime

Two further hypotheses transformed the latest temporal delta with model geometry:

```text
static: m_W = W^T W d
full:   m_F = J_t^T J_t d
```

The full path used analytic native FinalLayer JVP/VJP. The initial screen exposed a radial `q < eps` bug and an arbitrary operator-scale confound. Both were fixed by stable rational scaling and delta-equivalent direction normalization.

The corrected real gate then established:

```text
normalized ||m||/||d|| ~= 1 for all eligible candidates
raw ||J^T J d||/||d|| ~= 1e-8
static W^T W d: 0/8 vs scalar on both streams, ~6-7% worse
full J^T J d:   0/8 vs scalar on both streams, ~6-7% worse
full did not improve static
```

The transformed-trajectory family is therefore closed. Normal runtime no longer captures geometry, retains row history, calibrates direction alpha, computes WtW/JtJ, invokes the rejected JVP/VJP, or logs per-anchor telemetry for that family.

Debug summaries intentionally report its old runtime fields as zero and mark the family/K=2 as retired.

Pure helpers for WtW, RMSNorm JVP/VJP, FinalLayer JVP/VJP, normalization and radial budgeting remain as mathematical regression fixtures because the current previous-error experiment reuses the VJP and normalization primitives but not the rejected JtJ mechanism.

# Current isolated Feature-3 hypothesis: previous-error adjoint

At a completed actual anchor `k`, define the causal prediction errors:

```text
r_k = h_actual,k - h_pred_uncorrected,k

e_k = F_k(h_actual,k) - F_k(h_pred_uncorrected,k)
```

For a later target `t`, the telemetry-only screen compares:

```text
m_R = r_previous
m_W = W^T e_previous
m_J = J_t^T e_previous
```

This explicitly separates generic residual persistence (`m_R`) from static model geometry (`m_W`) and current local FinalLayer geometry (`m_J`). A model-specific claim requires a model adjoint to beat `m_R`, not merely the uncorrected forecast.

## Strict causal ordering

The target FinalLayer state depends only on target timestep/static model state and is captured before actual N executes. The model-aware observer runs before `forecaster.update(N)`, so history still contains only anchors `< N`.

For anchor N:

```text
1. retrieve previous error state from anchors < N
2. reconstruct uncorrected predicted hidden for N from history < N
3. construct/normalize m_R, m_W, m_J
4. apply alpha learned only from earlier completed anchors
5. score those fixed candidates against actual N
6. compute instantaneous alpha diagnostics from N
7. compute/store r_N and e_N for future anchors only
```

Actual N never enters candidate construction or alpha used at N.

## Native output residual

For selected complete rows, current `e_N` is computed exactly through the native sampled FinalLayer difference:

```text
F_N(h_actual,N) - F_N(h_pred_uncorrected,N)
```

Shift and output bias cancel. RMSNorm, target AdaLN scale and FP32 head projection are preserved. No transformer block is re-executed.

## Static/current adjoints

Static:

```text
m_W = e_previous @ W
```

Current local:

```text
m_J = J_t^T e_previous
```

`J_t^T e_previous` uses the analytic FinalLayer VJP only. The previous output residual lives in the stable output coordinate basis for the same sampled target rows/output channels. Reusing it across timesteps is a testable residual-persistence hypothesis, not assumed truth; current target RMSNorm/AdaLN geometry is applied through `J_t`.

## Shared normalization and trust budget

Every finite nonzero candidate is normalized before coefficient fitting/scoring:

```text
m_normalized = m * ||d|| / ||m||
```

The same validated rational budget remains:

```text
q_raw       = ||alpha * m_normalized|| / ||d||
radial      = 1 / (1 + q_raw / 0.25)
q_bounded   = q_raw * radial
```

`_DIRECTION_ALPHA_LIMIT` is unchanged. Positive scaling of a raw direction does not change the final candidate.

## Evidence/memory

No second full hidden history is allocated. The screen reuses normal forecaster history and keeps only the most recent sampled residual state.

Maximum persistent FP32 state at 32 complete rows per stream:

```text
hidden residuals = 1,376,256 bytes
output residuals =    16,384 bytes
subtotal         = 1,392,640 bytes = 1.328125 MiB
row indices      <= 512 bytes
```

When normal history is in system RAM, only selected rows are copied to the feature device for the screen.

## Compute

Maximum static head backprojection work at 32 rows per stream:

```text
audio W^T e = 5.505M MAC
video W^T e = 16.515M MAC
combined    = 22.020M MAC ~= 44 MFLOP
```

The current local VJP has the same output-head backprojection order plus O(hidden) RMSNorm/AdaLN operations. Exact sampled output-residual construction/scoring remains sub-GFLOP order. This is far below a 50-block H3 denoiser call and adds zero transformer NFE.

## ER-SDE / replay ownership

The telemetry does not modify ER-SDE generator, noise-sampler or derivative state. No denoiser call is added.

Offline replay remains scalar-only and deterministic. Previous-error telemetry is skipped during replay; first-pass observations may collect/score the experiment but do not alter replay weights/output.

## Active runtime labels

Normal `full` summary should show both facts clearly:

```text
feature3_applied_correction=scalar_latest_delta
feature3_transformed_trajectory_runtime=retired
feature3_direction_evidence_bytes=0
feature3_direction_workspace_bytes=0
feature3_jvp_enqueue_s=0
feature3_vjp_enqueue_s=0

feature3_previous_error_screen=residual_vs_static_adjoint_vs_local_adjoint
feature3_previous_error_applied=false
feature3_error_extra_transformer_nfe=0
```

The new screen remains counterfactual until a model adjoint materially beats the previous-hidden-residual baseline in a mechanically matched real gate.
