# MiniMax H3 integration notes

Source review date: 2026-08-12

Reviewed native ComfyUI revisions used by the PR matrix:

- `e377e263049f9338b4d12a3dd417b36ae62948ff`
- `0dd9b154a1654fc699dcdc3af066c7cce096045a`
- `5599a05fea715cb2aff11f30f5b06e16d0dfa0c4`
- `27bca654eb9a70237d93f56a6ea336ab55f8925d`

Reviewed Spectrum paper: arXiv `2603.01623v1`.

Reviewed official Spectrum implementation commit: `11f317a87352e2c67daa2fac5a971cf04233d7d1`.

## Native execution boundary

MiniMax-H3 sampling enters ComfyUI through the normal `CFGGuider` / sampler / `PREDICT_NOISE` path. Native `MiniMaxH3Model._forward` pads the video, resolves `PackedLayout`, constructs separate audio/video timestep state, builds modulation segments and embeddings, executes every transformer block, then sends the final packed hidden to `FinalLayer`.

The packed target tail is contiguous and ordered:

```text
[target audio rows | target video rows]
```

Spectrum captures only that compact target tail immediately after the final transformer block and immediately before native `FinalLayer`. Text, keyframe/reference, image-reference, video-reference, and audio-reference rows are not placed in forecast history.

Actual steps retain the native H3 transformer execution. Forecast steps predict only the compact pre-`FinalLayer` target hidden and then call the current native output-head/reconstruction path. This preserves current timestep conditioning, current sigma shifts, native audio schedule conversion, FP32 output heads, video unpatchification, audio unpacking, and the installed core's native return convention.

## Native FinalLayer contract

The reviewed native class has hidden width 5376 and independent FP32 output heads:

```text
audio_out: 5376 -> 32
video_out: 5376 -> 96
```

The effective stream mapping is:

```text
n(h) = RMSNorm(h)
z_t(h) = n(h) * (1 + scale_t) + shift_t
F_t(h) = W z_t(h) + b
```

The exact source uses `1.0 + scale[...]` for both streams and converts the modulated hidden to `torch.float32` before the output linear layer. The timestep row is selected from the native modulation segments: video uses the row whose modulation tag is `0`, audio uses the row whose modulation tag is `2`; the embedded timestep row is `row // 3`.

`operations.RMSNorm` delegates to PyTorch RMSNorm with the module's exact `eps` and effective cast weight. Feature-3 geometry capture uses ComfyUI's `cast_bias_weight` / `uncast_bias_weight` contract so the path remains compatible with the oldest pinned revision and does not retain a module reference after the native block call.

## RMSNorm analytic derivative

For one hidden row `x`, tangent `v`, RMSNorm weight `w`, hidden width `H`,

```text
s = mean(x^2) + eps
q = s^(-1/2)
y = w * x * q
```

The analytic JVP used in normal inference is:

```text
J_RMS(x) v = w * [q v - x q^3 mean(x v)]
```

For output cotangent `g`, define `p = g * w`. The analytic VJP is:

```text
J_RMS(x)^T g = q p - x q^3 mean(x p)
```

Tests compare both expressions against PyTorch autograd on small CPU tensors. Autograd is not used during generation.

## Full FinalLayer analytic JVP/VJP

For `a_t = 1 + scale_t`, the FinalLayer JVP is:

```text
v_norm = J_RMS(h) v
v_mod  = a_t * v_norm
J_t v  = v_mod @ W.T
```

The output projection is evaluated in FP32, matching native H3.

For output cotangent `u`, the VJP is:

```text
g_mod = u @ W
g_norm = a_t * g_mod
J_t^T u = J_RMS(h)^T g_norm
```

The implementation follows the dtype transition induced by the native FP32 output-head cast. Shift and output bias do not appear in the derivative. Tests verify the complete JVP, complete VJP, and `J_t^T J_t d` against explicit/autograd references on small tensors.

No full Jacobian is materialized. No transformer JVP/VJP is performed.

## Forecast history and storage

The normal forecaster stores at most `max_history` detached compact target features in the configured history storage. System-RAM remains the default; VRAM history remains opt-in. Prediction solves only for temporal weights and streams history chunks into an FP32 accumulator on the requested output device.

The published Spectrum baseline remains last-block spectral forecasting. Model-aware work is layered around the same hidden target; it does not replace the Chebyshev forecaster.

## Model profile lifetime

The process-local model profile stores immutable metadata plus detached CPU copies of the two native output-head weights and their normalized Gram diagonals. It stores no live model/module reference and no GPU tensor.

The static base-H3 head payload is approximately:

```text
(32 * 5376 + 96 * 5376) * 4 bytes ~= 2.625 MiB
```

Keeping this detached CPU payload avoids introducing model lifetime references into the global profile cache. GPU materialization of output-head tensors is `full`-only and is cleared when the model-aware run ends or is disabled.

LoRA/patch identity remains part of the profile/cache key and scheduling prior. This Feature-3 revision does not add new LoRA behavior.

## Model-aware mode boundaries

The intended ownership is strict:

```text
off
    legacy Spectrum

schedule
    Feature 1 profile/scheduling prior only
    no trajectory evidence allocation
    no correction fitting or application

schedule_confidence
    Feature 1 + Feature 2 trajectory confidence/adaptive fitting
    bounded generic risk evidence only
    no exact-head correction projection
    no K=2 solve
    no model-direction construction
    no correction application

full
    Feature 1 + Feature 2 + Feature 3
    retained scalar correction is applied
    model-transformed directions are telemetry-only
```

The runtime's existing head materializer already returns no GPU head payload unless mode is `full`. The Feature-3 revision also gates generic correction evidence and directional row history by mode.

## Retained scalar correction architecture

The active `full` correction is one-dimensional again. Let the latest causal trajectory direction be:

```text
d = h[-1] - h[-2]
```

The historical candidate hierarchy remains independently measurable:

```text
g_generic  = <r, d> / <d, d>

g_diag     = <r, S d> / <d, S d>
S          = diag(W^T W)

g_exact    = <r W^T, d W^T> / <d W^T, d W^T>
```

The active scalar gain is the existing trust interpolation between generic and exact static-head gains after the existing rational correction limit. The Gram-diagonal metric remains an ablation only.

Forecast application now explicitly ignores the rejected K=2 coefficient fields even if an old internal/test decision object contains them.

## Rejected K=2 trajectory subspace

The completed K=2 experiment used:

```text
d0 = h[-1] - h[-2]
d1 = h[-2] - h[-3]
```

and compared Euclidean and exact-head 2x2 solves over the same span. Real base-H3 evidence showed generic K=2 was consistently worse than generic scalar and exact-head weighting changed the K=2 result only at approximately 0.01-0.03% mean scale.

Normal generation therefore no longer calls the K=2 Gram/solve path, no longer applies K=2 coefficients, and no longer produces a K=2 offline-replay stencil. Focused K=2 mathematical utilities/tests may remain to preserve the experimental record and compatibility with historical in-memory structures.

The old ~2.17-2.24 second `model_aware_subspace_solve_s` attribution was not evidence of expensive literal 2x2 arithmetic. The timed path included CUDA reductions/eigen/solve operations, and a host dependency can absorb synchronization from previously queued work. The architectural correction is to remove the rejected runtime work. No explicit CUDA synchronization is added merely to make timers look cleaner.

## Feature-3 direction screen

The new experiment changes the hidden-space correction direction while leaving generated output controlled by the retained scalar path.

### Static native head direction

For stream head `W`:

```text
m_W = W^T W d
```

Production computes:

```text
u   = d @ W.T
m_W = u @ W
```

This preserves every cross-hidden-channel term without constructing a `5376 x 5376` Gram matrix.

### Full native FinalLayer direction

At target timestep `t`:

```text
J_t = dF_t / dh
m_F = J_t^T J_t d
```

The implementation performs the analytic FinalLayer-only JVP described above, followed by the analytic VJP. The work is limited to sampled hidden rows, RMSNorm/AdaLN elementwise operations, and the 5376x32 or 5376x96 output-head multiplies.

No transformer block is re-executed. Transformer NFE count is unchanged.

## Causal reference state

The full direction is evaluated around the uncorrected sampled spectral prediction for the target coordinate:

```text
h_ref = h_pred_uncorrected(t)
```

This prediction is reconstructed only from actual anchors that already existed before the current actual anchor. The target timestep/AdaLN state is derived from the native timestep embedding/modulation segment and contains no future hidden information.

During a completed actual anchor N:

1. build the counterfactual spectral prediction from history `< N`;
2. read the coefficient calibration that existed before N;
3. build `d`, `m_W`, and `m_F` from causal history and target timestep geometry;
4. score all candidates against actual N;
5. compute instantaneous `<r_N,m_N>/<m_N,m_N>` diagnostics;
6. update calibration for future anchors;
7. append actual N to the sampled direction-evidence history.

This ordering prevents same-anchor hindsight. Snapshot/rollback includes directional calibration, sampled row history/indices, and counters, so speculative state cannot leak across rollback.

## Direction coefficient calibration

Each model-transformed direction has a separate bounded EWMA scalar amplitude. At anchor N the candidate uses the EWMA learned from prior anchors, scaled by the same current Feature-2 confidence convention used elsewhere in the controller. The current anchor's instantaneous projection is training evidence for later candidates only.

A zero, tiny, nonfinite, or shape-incompatible direction is marked ineligible and contributes no model-direction candidate. Scalar correction remains available.

## Direction correction budget

For candidate `m` and prior coefficient `alpha`:

```text
c_raw = alpha * m
q_raw = ||c_raw|| / max(||d||, eps)
q_bounded = q_raw / (1 + q_raw / 0.25)
```

The candidate vector is radially scaled so its sampled hidden-space norm ratio equals `q_bounded`. This is the same physical 0.25 rational soft budget used by the scalar correction. It prevents `W^T W` or `J^T J` amplification from receiving a larger correction magnitude.

When `m == d`, the rule reduces to the existing scalar rational semantics.

## Direction evidence and memory

Directional screening uses deterministic complete-row sampling with a default cap of 32 target rows per stream. The selected hidden rows remain on the producing feature device. The history is bounded by `max_history`.

The full hidden tensor is never transferred to CPU for the experiment. Reduced scalar telemetry is transferred after device-local candidate evaluation.

Temporary work includes the sampled prediction, residual, delta, static direction/candidate, and full direction/candidate plus output-head intermediates. Debug telemetry reports the peak sampled workspace and retained direction evidence bytes separately from the ordinary forecaster history/archive.

## Telemetry timing semantics

The direction evaluator reports:

- total direction computation span;
- static-direction enqueue time;
- full FinalLayer direction enqueue time;
- JVP enqueue time;
- VJP enqueue time;
- reduced scalar transfer time;
- head materialization time/bytes;
- retained sampled evidence bytes and workspace.

Small CUDA operation timers that do not synchronize are explicitly labeled `enqueue_s`; they must not be interpreted as isolated GPU kernel duration. Normal inference adds no `torch.cuda.synchronize()` for telemetry.

The complete direction span includes the reduced scalar transfer that closes the dependency chain for the sampled candidate work, making that aggregate timing the useful wall-clock contribution to inspect in the real benchmark.

## Offline smoothing replay

Offline smoothing remains default-on behavior for MiniMax-H3. The new model-direction candidates are telemetry-only and never alter replay output.

The first pass archives only the retained scalar Feature-3 decision. Replay reconstructs the same scalar correction decision using the causal anchor relationship already supported by the offline smoother. The rejected K=2 replay stencil is inactive in normal generation because the runtime no longer produces an eligible K=2 decision.

Offline replay still performs no transformer calls during replay. The new telemetry does not change the archived latent/sampler random process.

## ER-SDE contract

Native `sample_er_sde` remains supported through its reviewed one-model-call-per-outer-step contract. Solver-local denoised derivative history and stochastic noise draws remain owned by the native sampler.

Feature-3 direction screening observes cached hidden trajectory/model geometry only. It does not mutate ER-SDE's generator, noise sampler, `old_denoised`, derivative history, or stage logic. It introduces zero additional denoiser/transformer evaluations.

The existing offline replay restriction for custom mutable ER-SDE noise samplers/scalers remains unchanged.

## Rollback and mutation safety

Directional state is part of `ModelAwareController.snapshot()` / `restore()`. History entries are detached tensors and are replaced/appended rather than mutated in place. No model reference is captured in rollback state.

Run end, forecasting disable, and model-aware disable clear pending FinalLayer geometry, sampled direction evidence, row-index tensors, and device-materialized output heads. Global profile CPU tensors remain process-local immutable cache payload.

## Verification expectations

The test suite covers:

- scalar `full` application after K=2 retirement;
- no correction/K=2/model-direction work in `schedule_confidence`;
- scheduling parity for equal trajectory evidence;
- unchanged transformer NFE counters;
- `(d @ W.T) @ W` equivalence to an explicit small `W.T @ W` reference;
- no production 5376x5376 Gram materialization;
- cross-channel behavior distinct from the diagonal approximation;
- analytic RMSNorm JVP/VJP versus autograd;
- analytic full FinalLayer JVP/VJP versus autograd;
- `J^T J d` versus an explicit/autograd small reference;
- exact native `1 + scale` and FP32-head source contract;
- causal predicted-hidden reference and post-score history append;
- prior-anchor-only coefficient calibration;
- 0.25 radial budget and `m == d` scalar equivalence;
- safe zero/nonfinite direction fallback;
- controller snapshot/restore;
- system-RAM/VRAM history behavior inherited from the complete suite;
- no model/module references in retained Feature-3 state.

The cross-ComfyUI GitHub Actions matrix remains the required final compatibility gate.
