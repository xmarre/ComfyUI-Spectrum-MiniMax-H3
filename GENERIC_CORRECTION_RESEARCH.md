# Generic causal correction research pass

This document defines the experimental controller family built around the one
generic correction geometry that has shown both hidden-space and perceptual value:

```text
residual  r = actual - predicted
direction d = latest_exact - previous_exact
candidate   = predicted + g*d
```

The validated generic-correction default used by the opt-in
`model_aware_mode=full` path is:

```text
coordinate_rls + no_attenuation + hard_clip + 0.40
canonical RLS lambda = 0.90
```

The global `model_aware_mode` default remains `off`. The exact
`legacy + mode_default + rational + 0.25` path remains selectable for
reproduction and ablation.

## Evidence and claim boundary

In a controlled native ER-SDE 20-step comparison, `schedule_confidence` and
`full` both used 11 actual steps, 9 forecast steps, 11 transformer calls, and
zero model-aware extra NFEs. The retained generic correction changed measured
hidden forecast ratios approximately as follows:

```text
audio: 1.777636 -> 1.670690   (~6.02% hidden-error reduction)
video: 1.313055 -> 1.250087   (~4.80% hidden-error reduction)
```

The final hidden-space validation contains three independent runs in each of two
compatible live-trajectory groups. Both groups select the exact promoted family:

```text
group A: +15.7788% vs exact legacy, 48 wins / 0 losses, worst regression 0
group B: +15.7757% vs exact legacy, 48 wins / 0 losses, worst regression 0
```

Whole-run leave-one-out generalization is used; target rows are never randomized
across folds. The percentages are normalized hidden-feature error reductions,
not perceptual-quality percentages.

Separate decoded-media validation used three independent native-reference R/A/B
triads with native H3, ER-SDE, 20 steps, 512x768, 192 frames, eight seconds, and
24 fps. Verdicts were two candidate-favored, one mixed, and zero legacy-favored.
Broad VIDEO MS-SSIM, PSNR, and temporal metrics favored the candidate on all
three seeds. Audio spectral evidence was generally candidate-favored; one seed
showed weaker phase-sensitive correlation/SI-SDR diagnostics with zero detected
bounded lag. Supporting manual comparisons found no repeatable candidate audio
regression and a small candidate advantage in rapid fine motion.

This evidence does not establish results for Euler, RES/RES CFG++,
Turbo/LightX2V, other step counts, resolutions, prompts, or LoRAs.

## Live modes

| `generic_correction_mode` | Geometry and controller | Status |
|---|---|---|
| `legacy` | Untransported latest delta, existing projection EWMA, existing general-confidence/horizon scaling | Exact reproduction/ablation path |
| `coordinate_rls` | Signed coordinate-transported latest delta with exponentially weighted least-squares scalar | Validated default with `no_attenuation + hard_clip + 0.40` |
| `coordinate_rls_reliability` | Coordinate/RLS plus correction-specific reliability | Experimental |
| `regional` | Audio global; VIDEO split into four topology-proven temporal bands with regularized regional RLS/reliability gains | Experimental; global fallback when topology is unproven |

All modes retain the same sampler schedule. They add no transformer evaluation.

The validated coordinate/RLS default uses a hard bound:

```text
bounded(g, 0.40) = clamp(g, -0.40, 0.40)
```

The exact legacy reproduction configuration uses the previous rational soft
bound:

```text
bounded(g, L) = g / (1 + abs(g)/L)
L = 0.25
```

`tanh` and other mode/attenuation combinations remain controlled research paths.

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

## Orthogonal attenuation policy

`generic_correction_attenuation=no_attenuation` is the validated coordinate/RLS
default. `mode_default` remains saved-workflow compatible and preserves each
mode's former numerical behavior: `coordinate_rls` uses general forecast
confidence, while `coordinate_rls_reliability` and `regional` use combined
general-confidence and correction-reliability attenuation. Research runs can
select `general_confidence`,
`correction_reliability`, or `combined_conservative` explicitly. Reports record
both the requested selector and the resolved policy actually used.

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

Native H3 audio calibration also records diagnostic-only `audio_start`,
`audio_middle`, and `audio_end` moments. H3 packs `[B,C,2,T]` channel-major as
`[left t0..T-1, right t0..T-1]`, so each band combines the corresponding time
range from both stereo channels rather than slicing one contiguous third of all
rows. The verified native latent rate is 40 Hz: clips at least three seconds
use one-second (40-token) start/end windows and a middle remainder; shorter
supported clips use deterministic non-empty temporal thirds. The aggregate
AUDIO moments are reconstructed exactly from these bands. No raw tensor is
persisted and no transformer call is added.

Coordinate transport only rescales the same one-dimensional latest-delta
direction. It does not create a new correction subspace or increase oracle
expressiveness; legacy and coordinate oracle headroom are identical when the
scalar is freely refit per target. Transport exists to make a history-estimated
coefficient better behaved across nonuniform solver-coordinate spacing.

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
strongest live-reproducible hidden-space configuration. Reports distinguish that
ranking from the repository's separately decoded/perceptually validated runtime
default. One run is labeled development-only, two runs preliminary whole-run
leave-one-out, and three or more whole-run leave-one-out generalization. Research
machinery never changes live settings automatically.

Compatibility includes source schema, package/source provenance, the full sampler
schedule fingerprint, sampler name, step count, topology fingerprint, and the
executed base configuration, including correction mode, attenuation, limiter,
and limit. Only debug is excluded. Missing attenuation in an older saved workflow
normalizes to the compatibility-safe `mode_default`. Exact trace duplicates never count twice. Within one
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
Candidates using noncanonical offline RLS lambdas remain clearly labeled
offline-only. Step-for-step equivalent candidates form an explicit tie group;
the live recommendation uses canonical `lambda=0.90` when it belongs to that
group and otherwise emits no unreproducible live configuration.

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

The PR #51 promotion gate required:

- identical actual/forecast step IDs and transformer-call counts;
- `model_aware_extra_nfes=0`;
- repeatable hidden-space improvement over exact legacy on held-out runs;
- perceptual non-regression across the chosen audio/video comparisons;
- acceptable correction overhead and memory behavior.

The gate passed for the exact `coordinate_rls + no_attenuation + hard_clip +
0.40` family:

- controlled 20-step ER-SDE legacy/candidate runs preserved 11 actual steps,
  9 forecasts, 11 transformer calls, zero extra model-aware NFEs, and zero
  fallbacks;
- two independent three-run hidden-space groups selected the same candidate with
  about 15.78% improvement, 48/0 target wins/losses, and no worst regression;
- three decoded-media triads produced 2 candidate-favored / 1 mixed / 0
  legacy-favored verdicts, with supporting perceptual non-regression;
- observed candidate and legacy sampler wall times were approximately 174.66 s
  and 174.78 s in the same session;
- runtime inspection confirms scalar/small bounded controller state, no persistent
  per-token gain field, debug-only scalar calibration, and no schedule/NFE owner.

Hidden-space evidence identified the candidate. Separate decoded-media and
perceptual evidence closed the promotion gate.
