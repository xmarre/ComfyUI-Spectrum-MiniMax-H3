# Model-aware forecasting benchmark record

This document is the experimental record for PR #39. Real MiniMax-H3 checkpoint evidence takes precedence over synthetic fixtures. The original Feature-3 objective was to find a materially useful **model-informed** forecast correction. That objective is now closed as a negative result for the tested base MiniMax-H3 configuration. The useful generic latest-delta residual correction is retained independently.

## Mechanical validity rule

A same-seed correction comparison is valid only when the compared runs retain the same actual/forecast step IDs, transformer NFE count, Feature-2 risk/confidence evolution, adaptive ridge, degree, and stream blends. The final gates used the same saved workflow, base MiniMax-H3, native `sample_er_sde`, 20 steps, fixed seed, no active LoRA, same checkpoint/precision, scheduler, prompt/references, resolution, frame count, CFG, Spectrum settings, storage mode, and debug enabled.

The final gate used:

```text
full
schedule_confidence
full
```

and all three first passes retained:

```text
actual:   0,2,4,6,8,10,12,14,16,18,19
forecast: 1,3,5,7,9,11,13,15,17
actual transformer calls = 11
forecast steps           = 9
extra transformer NFE    = 0
risk max                 = 0.550956
confidence min           = 0.449044
```

The profile reported `patches=0`, `recognized_lora=0`, `perturbation=0`, as expected for the no-LoRA base-H3 gate.

## Experiment history

### 1. Scalar head metrics

The first Feature-3 family kept the latest temporal delta as the correction direction and changed only the scalar metric used to fit its amplitude:

- generic Euclidean residual projection;
- normalized Gram-diagonal output-head metric;
- exact static output-head metric;
- trust mixture between generic and exact-head scalar gains.

The generic scalar correction repeatedly improved forecast error by about 5-6%, but the **increment caused by model-specific exact-head geometry** stayed far below the required 2% materiality threshold. The latest completed gate reported:

| Stream | raw forecast | generic scalar | exact scalar | trust-mixed applied |
|---|---:|---:|---:|---:|
| Audio | 1.685420 | 1.581040 | 1.575096 | 1.578044 |
| Video | 1.353503 | 1.279524 | 1.280589 | 1.280054 |

Generic latest-delta correction improves raw forecast by approximately 6.2% audio and 5.5% video. On this gate exact-vs-generic was only about +0.38% audio and -0.08% video; the trust-mixed applied increment was only about +0.19% audio and -0.04% video. Earlier gates likewise produced approximately 0.01-0.2% class model-specific increments.

**Conclusion:** retain the useful generic scalar correction; retire the exact-head/diagonal correction machinery from the applied path.

### 2. K=2 causal trajectory span

The rank-expansion experiment used the two most recent causal trajectory differences instead of the single latest delta. Generic K=2 was consistently worse than the generic scalar correction, and exact-head weighting did not rescue it.

> The second recent trajectory direction did not improve Feature 3. Generic K=2 was consistently worse than the generic scalar correction, and exact-head weighting did not rescue it. Therefore increasing temporal trajectory rank is not the next direction of work.

K=2 was removed from normal runtime. Its historical solve attribution of roughly 2.17-2.24 s included asynchronous CUDA work and was not interpreted as literal 2x2 arithmetic cost.

### 3. Transformed latest-delta directions

Two model-transformed directions were screened without applying them to generated output:

```text
m_W = W^T W d
m_F = J_t^T J_t d
```

The first unnormalized run exposed a radial-bound numerical bug and an operator-scale confound. The old radial implementation effectively divided `q_bounded` by `clamp_min(q, eps)`, causing tiny positive `q` to receive an erroneous additional suppression. The stable rule is:

```text
radial_scale = 1 / (1 + q / 0.25)
q_bounded    = q * radial_scale
```

The experiment was then corrected by normalizing every finite nonzero transformed direction to delta-equivalent norm before amplitude fitting, calibration, radial budgeting, and scoring:

```text
m_normalized = m_raw * ||d|| / ||m_raw||
```

The corrected real gate passed the numerical sanity checks. Eligible normalized direction ratios were approximately 1. Raw operator-scale means were:

| Stream | `||W^T W d|| / ||d||` | `||J^T J d|| / ||d||` |
|---|---:|---:|
| Audio | 1.968860e+00 | 8.879409e-09 |
| Video | 2.728169e+00 | 1.992571e-08 |

The extremely small full-FinalLayer scale was therefore real local geometry, not merely six-decimal logging loss. The corrected full radial scales were sane: audio min 0.784770 / mean 0.864360; video min 0.929963 / mean 0.951673.

Final normalized efficacy:

| Stream | static candidate | static vs generic | full candidate | full vs generic |
|---|---:|---:|---:|---:|
| Audio | 1.653988 | 0/8, -0.071455 | 1.655349 | 0/8, -0.072343 |
| Video | 1.394504 | 0/8, -0.064070 | 1.395260 | 0/8, -0.064658 |

Full-vs-static was also 0/8 on both streams, with mean advantage -0.000827 audio and -0.000551 video.

**Conclusion:** the scale-normalization sanity gate passed, yet both transformed directions lost every eligible comparison against the scalar baseline by roughly 6-7%. `W^T W d` and `J_t^T J_t d` are rejected. Full FinalLayer geometry did not improve the static output-head transform.

### 4. Previous observed forecast error

A final, causally distinct hypothesis used the strictly observed previous forecast error rather than transforming the current latest delta:

```text
r_k = h_actual,k - h_pred_uncorrected,k
e_k = F_k(h_actual,k) - F_k(h_pred_uncorrected,k)

m_R = r_previous
m_W = W^T e_previous
m_J = J_t^T e_previous
```

A source-level feasibility review established that previous residual state was available before anchor N, deterministic complete-row correspondence was stable within a run, target FinalLayer state depended only on current timestep/static-model state, the experiment could remain sampled to at most 32 complete rows per stream, offline replay could remain untouched, and no transformer NFE was required. The experiment was therefore allowed one telemetry-only real gate.

The gate was mechanically valid. `schedule_confidence` performed zero previous-error work. Each residual/static/full candidate on both streams showed the expected causal pattern of `eligible=7`, `fallback=2`, zero failures, and zero extra transformer NFE. Every eligible candidate normalized to approximately unit delta-equivalent norm.

Representative raw direction ratios were:

| Stream | `r_previous` | `W^T e_previous` | `J_t^T e_previous` |
|---|---:|---:|---:|
| Audio | 1.661760e+00 | 2.323667e-04 | 1.991956e-08 |
| Video | 1.392982e+00 | 2.254621e-04 | 2.845391e-08 |

There is therefore no remaining operator-scale explanation for the efficacy result.

#### Final previous-error table

| Stream | Comparison | Wins/losses | Mean relative advantage |
|---|---|---:|---:|
| Audio | `r_previous` vs generic scalar | 0/7 | -3.8938% |
| Audio | `W^T e_previous` vs `r_previous` | 0/7 | -3.0698% |
| Audio | `J_t^T e_previous` vs `r_previous` | 0/7 | -3.1027% |
| Audio | `W^T e_previous` vs generic scalar | 0/7 | -7.0795% |
| Audio | `J_t^T e_previous` vs generic scalar | 0/7 | -7.1138% |
| Audio | `J_t^T e_previous` vs `W^T e_previous` | 0/7 | -0.0319% |
| Video | `r_previous` vs generic scalar | 1/6 | -1.3247% |
| Video | `W^T e_previous` vs `r_previous` | 0/7 | -4.9102% |
| Video | `J_t^T e_previous` vs `r_previous` | 0/7 | -4.9392% |
| Video | `W^T e_previous` vs generic scalar | 0/7 | -6.2697% |
| Video | `J_t^T e_previous` vs generic scalar | 0/7 | -6.2991% |
| Video | `J_t^T e_previous` vs `W^T e_previous` | 0/7 | -0.0276% |

Candidate means were:

```text
Audio:
    r_previous           1.577665
    W^T e_previous       1.626384
    J_t^T e_previous     1.626890

Video:
    r_previous           1.326834
    W^T e_previous       1.392176
    J_t^T e_previous     1.392558
```

**The generic one-anchor previous hidden residual is not a superior correction source, and projecting previous output error back through either the static output head or the complete current FinalLayer adjoint makes forecast error materially worse. The previous-error-adjoint family is therefore rejected.**

## Final Feature-3 conclusion

**Across exact scalar head metrics, higher temporal rank, transformed trajectory directions, previous hidden residual persistence, static output adjoints, and complete local FinalLayer adjoints, no model-informed correction mechanism produced a material >=2% improvement over the appropriate generic baseline. Feature 3's original model-informed correction objective is therefore closed as a negative experimental result for the tested base MiniMax-H3 configuration.**

This closes mechanism exploration in PR #39. The result does not justify K=3/K4, larger residual histories, residual PCA/SVD, alternate normalizations, transformer JVP/VJP/Jacobians, Hessian approximations, learned correction models, output-space residual forecasting, looser correction bounds, alpha retuning, per-timestep alpha tables, or hand-tuned stream exceptions.

**The generic latest-delta residual correction remains useful and is retained independently of the failed model-informed Feature-3 objective.** Its roughly 5-6% improvement must not be described as model-informed.

## Final shipping architecture after cleanup

Public serialized modes remain unchanged:

```text
off
    legacy Spectrum

schedule
    Feature 1 model/patch-profile scheduling prior only

schedule_confidence
    Feature 1 + Feature 2 confidence/adaptive fitting
    no forecast correction

full
    Feature 1 + Feature 2
    + bounded generic latest-delta scalar residual correction
    no model-specific Feature-3 correction
```

The surviving generic correction uses the existing causal projection chronology and existing rational 0.25 trust region. The applied gain is exactly the generic scalar gain; exact-head trust mixing and K=2 are not applied.

Rejected runtime is removed rather than preserved for intermediate-PR compatibility. The persistent profile retains compact scalar sensitivities/patch metadata but no full audio/video output-head copies or Gram diagonals. Full exact-head head materialization/projection/evidence is disabled. Offline smoothing replay consumes the same stored generic scalar correction decision as the first-pass `full` path and still adds no transformer call.

Normal runtime identifies the final state with:

```text
feature3_model_informed_correction=retired_no_material_gain
feature3_applied_correction=generic_scalar_latest_delta
feature3_k2_runtime=retired
feature3_transformed_trajectory_runtime=retired
feature3_previous_error_runtime=retired
feature3_direction_evidence_bytes=0
feature3_direction_workspace_bytes=0
feature3_error_evidence_bytes=0
feature3_error_workspace_bytes=0
feature3_extra_transformer_nfe=0
```

## Final user gate

One final real run is required only to verify the cleaned shipping architecture, not to discover another Feature-3 mechanism:

```text
1. full
2. schedule_confidence
3. full
```

Use the same saved base-H3 workflow and fixed seed, native `sample_er_sde`, 20 steps, no active LoRA, identical checkpoint/precision, scheduler, prompt/references, resolution, frame count, CFG, Spectrum settings, storage mode, and debug enabled.

Verify:

- the same 11/9 actual/forecast schedule;
- unchanged Feature-2 risk/confidence/ridge/degree/blend behavior;
- zero extra transformer NFE;
- K=2/transformed-direction/previous-error runtime remains retired;
- exact-head materialization/projection/evidence remains zero;
- `full` reports `feature3_applied_correction=generic_scalar_latest_delta`;
- replay uses the same generic scalar correction decision;
- compact profile memory is substantially below the old ~2.8 MiB head-retaining profile;
- no correctness or ER-SDE replay regression.

If that mechanical gate passes, PR #39 is scientifically finished.

## Subsequent generic-scalar research pass

PR #39's negative conclusion remains in force for model-informed direction
families. A later research pass does not reopen K=2, transformed directions,
previous-error directions, output-head geometry, or replay transfer. It builds a
predeclared controller family around the retained generic causal latest delta:

- exact legacy behavior remains available for reproduction;
- signed sampler-coordinate transport is separately scorable;
- exponentially weighted least-squares estimates scalar magnitude from accumulated
  `B=<r,d>` and `C=<d,d>` support;
- correction-specific reliability attenuates only the generic correction;
- topology-proven coarse temporal VIDEO regions remain local latest-delta scalars;
- rational, hard-clip, and tanh limiters are evaluated from exact quadratic moments;
- a CPU evaluator enforces whole-run separation and reports hidden-space results
  without promoting a perceptual claim.

All candidate paths add zero transformer NFEs. Their implementation and validation
contract are documented in [GENERIC_CORRECTION_RESEARCH.md](GENERIC_CORRECTION_RESEARCH.md).
PR #51 subsequently completed three-run hidden-space generalization and three
decoded-media native-reference triads, promoting
`coordinate_rls + no_attenuation + hard_clip + 0.40` as the generic-correction
default used when `model_aware_mode=full`. This later result does not alter PR
#39's negative conclusion for model-informed direction families.
