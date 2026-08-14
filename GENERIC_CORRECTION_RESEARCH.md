# Generic causal correction research pass

This document defines the experimental controller family built around the one
generic correction geometry that has shown both hidden-space and perceptual value:

```text
residual  r = actual - predicted
direction d = latest_exact - previous_exact
candidate   = predicted + g*d
```

The current validated production baseline remains `generic_correction_mode=legacy`.
The additional modes are live experimental paths and exact offline-scoring targets.
They have not yet been perceptually validated.

## Evidence and claim boundary

In a controlled native ER-SDE 20-step comparison, `schedule_confidence` and
`full` both used 11 actual steps, 9 forecast steps, 11 transformer calls, and
zero model-aware extra NFEs. The retained generic correction changed measured
hidden forecast ratios approximately as follows:

```text
audio: 1.777636 -> 1.670690   (~6.02% hidden-error reduction)
video: 1.313055 -> 1.250087   (~4.80% hidden-error reduction)
```

The same test cycle included a recurring false eye-motion artifact that was absent
with `full`, plus an earlier native ER-SDE pronunciation case improved by `full`.
Those results motivate better scalar estimation along the latest-delta direction.
The percentages are hidden-feature error reductions for particular traces. They
are not perceptual-quality percentages and do not establish results for Euler,
RES/RES CFG++, Turbo/LightX2V, or other samplers.

## Live modes

| `generic_correction_mode` | Geometry and controller | Status |
|---|---|---|
| `legacy` | Untransported latest delta, existing projection EWMA, existing general-confidence/horizon scaling | Validated baseline and default |
| `coordinate_rls` | Signed coordinate-transported latest delta, exponentially weighted least-squares scalar, general-confidence scaling | Experimental |
| `coordinate_rls_reliability` | Coordinate/RLS plus correction-specific reliability | Experimental |
| `regional` | Audio global; VIDEO split into four topology-proven temporal bands with regularized regional RLS/reliability gains | Experimental; global fallback when topology is unproven |

All modes retain the same sampler schedule. They add no transformer evaluation.

The default limiter is the legacy rational soft bound:

```text
bounded(g, L) = g / (1 + abs(g)/L)
L = 0.25
```

`hard_clip` and `tanh`, along with an advanced limit control, are available for
controlled A/B generation. No alternative limiter is promoted by this PR.

## Coordinate transport

The latest delta contains the spacing between its two exact anchors. For target
coordinate `x_t`, latest coordinate `x_1`, and previous coordinate `x_0`, the
experimental direction is:

```text
d_target = (latest_exact - previous_exact) * (x_t - x_1) / (x_1 - x_0)
```

The signed ratio supports increasing and decreasing schedules and nonuniform
spacing. Duplicate, tiny, or nonfinite anchor spacing falls back to the legacy
unscaled direction. The real normalized sampler coordinate is used; step index
is not substituted.

The limiter bounds the dimensionless gain applied to `d_target`. It does not
re-bound the algebraically equivalent coefficient on the untransported anchor
delta; doing so would make the candidate geometry depend on which representation
is used and would break parity with the exact coordinate-direction moments.

## Recursive scalar estimator

For each exact causal observation, the runtime records:

```text
B_t = mean(r_t * d_t)
C_t = mean(d_t^2)

B_acc = lambda * B_acc + B_t
C_acc = lambda * C_acc + C_t
g_hat = B_acc / C_acc
```

The live canonical forgetting factor is `lambda=0.90`. The CPU evaluator includes
the predeclared family `0.75, 0.90, 0.97, 1.0`. Energy-weighted sufficient
statistics prevent a nearly stationary direction from dominating through a noisy
per-anchor division. State is separate for audio, global video, and each temporal
video region; it resets per run and participates in rollback snapshots.

## Correction-specific reliability

Reliability only attenuates the generic correction magnitude. It never shrinks
the whole forecast toward an anchor. The bounded score combines causal history of:

- absolute residual/direction alignment;
- stability of successive signed oracle coefficients (negative gains remain valid);
- realized normalized advantage of the previously predicted correction;
- nondegenerate direction support and estimator age.

Telemetry exposes reliability, accumulated direction energy, effective estimator
age, raw gain, scaled gain, and applied gain.

## Topology-safe temporal VIDEO regions

Native MiniMax H3 patchification is explicitly:

```text
[B,C,T,H,W] -> [B,T,H,W,C,pt,ph,pw] -> rows
```

For the supported batch-size-one path, each temporal token is therefore a
contiguous row block. Regional mode validates `video_padded`, `patch_size`, and
`target_video_rows` from the runtime topology signature before creating temporal
bands. Neighboring gains are smoothed, and weak regions shrink toward the global
video estimate. An inconsistent or unavailable mapping falls back to the global
coordinate/RLS/reliability controller. No per-token gain field is retained.

## Exact calibration export

With `debug=true`, single-pass `model_aware_mode=full` emits:

```text
SPECTRUM_GENERIC_CORRECTION_CALIBRATION_JSON={...}
```

At each eligible exact anchor and stream/region, the runtime stores scalar moments:

```text
A = mean(r^2)
B = mean(r*d)
C = mean(d^2)

MSE(g) = A - 2*g*B + g^2*C
g_oracle = B/C
```

Both legacy-direction and coordinate-direction moments are included. The export
also carries the exact runtime ratio denominator, causal pre-target gains and
reliability, anchor IDs/coordinates, sampler/schedule metadata, topology/config
fingerprints, seed when available, package/source provenance, and candidate gains.
It serializes no hidden tensor or token payload.

The sign convention is fixed: `residual=actual-predicted` and
`direction=latest-previous`, so the quadratic cross term is `-2*g*B`.
Target-derived moments, oracle gains, and errors are labels. Controller decisions
for that target are frozen before those labels are read.

## Automatic research store and shared CPU evaluator

An eligible completed run is one with:

```text
debug = true
model_aware_mode = full
offline_smoothing_replay = false
```

After the run completes successfully, Spectrum stores the versioned scalar block
under:

```text
ComfyUI/user/__cache/spectrum_h3/generic_correction/v1/
  runs/<trace-fingerprint>.json
  reports/<compatibility-group>.md
  reports/<compatibility-group>.json
  corrupt/...
```

The exact `user` root follows ComfyUI's configured user directory. Writes use a
same-directory temporary file, flush it, and atomically replace the destination.
The store retains at most 12 valid runs per compatibility group and 96 valid runs
globally, plus 24 compatibility-group report pairs and 16 quarantined corrupt
files. Cleanup is deterministic by filesystem
modification time and filename. Close ComfyUI and delete this exact `v1` directory
to clear the research state safely; Spectrum recreates it on the next eligible run.

Every new compatible independent run immediately refreshes its group report and
prints the validation level, compact VIDEO/AUDIO rankings, baseline context, and
strongest live configuration for a later manual perceptual A/B. One run is labeled
development-only, two runs preliminary whole-run leave-one-out, and three or more
whole-run leave-one-out generalization. Settings and defaults are never changed
automatically.

Compatibility includes source schema, package/source provenance, the full sampler
schedule fingerprint, sampler name, step count, topology fingerprint, and the base
configuration after deliberately excluding only debug and the candidate mode /
limiter / limit controls. Exact trace duplicates never count twice. Within one
group, a repeated known seed is also treated as the same evidence identity; when a
seed is unavailable, the trace fingerprint is the conservative identity.

The runtime and forensic CLI execute the same implementation in
`comfyui_spectrum_h3/generic_correction_evaluator.py`. The CLI remains available
for reproduction:

Run:

```text
python tools/analyze_generic_correction.py comfyui.log --json
```

The evaluator accepts raw blocks, complete ComfyUI logs, and multiple independent
runs. It scores the uncorrected forecast, exact legacy behavior, legacy and
coordinate-direction oracles, the RLS forgetting-factor family, four attenuation
policies, three limiter families, three predeclared limits, and global versus
regional VIDEO.

Reported metrics include mean normalized hidden error, improvement over
uncorrected and legacy, oracle headroom and captured headroom, target wins/losses,
and worst regression, separately for audio and video. Runtime metric parity is:
per-target normalized ratio followed by arithmetic mean.

Validation discipline is whole-run only:

```text
1 run:  development only / non-confirmatory
2 runs: leave-one-run-out / preliminary
3+ runs: whole-run leave-one-run-out generalization
```

Incompatible source/config/schedule/sampler/topology groups are reported separately,
and duplicate trace or seeded-run identities are rejected. Target rows are never
randomized across folds. Evaluation uses only persisted CPU scalars, retains no GPU
tensor, makes no transformer call, and cannot change the sampler schedule. Store,
evaluation, or report failures produce a warning after the completed generation;
invalid files are skipped and moved into the bounded `corrupt` directory so later
runs remain analyzable.

## Replay separation

These controllers belong to the causal latest-exact trajectory geometry. No new
coordinate scalar, RLS state, reliability value, regional gain, or limiter choice
is transplanted onto future-bracket offline replay.

```text
model_aware_replay_generic_correction = false
```

remains the supported default. Offline smoothing replay remains a separate
compatibility/audio path, and its transformer-free second pass is unchanged.

## Promotion gate

The new modes remain experimental until independent real generations establish:

- identical actual/forecast step IDs and transformer-call counts;
- `model_aware_extra_nfes=0`;
- repeatable hidden-space improvement over exact legacy on held-out runs;
- perceptual non-regression across the chosen audio/video comparisons;
- acceptable correction overhead and memory behavior.

No hidden-space result alone promotes a new default.
