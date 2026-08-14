# MiniMax H3 model-aware integration notes

Source review date: 2026-08-12

Reviewed native ComfyUI revisions used by the PR matrix:

- `e377e263049f9338b4d12a3dd417b36ae62948ff`
- `0dd9b154a1654fc699dcdc3af066c7cce096045a`
- `5599a05fea715cb2aff11f30f5b06e16d0dfa0c4`
- `27bca654eb9a70237d93f56a6ea336ab55f8925d`

The experimental history and negative Feature-3 result are preserved in `MODEL_AWARE_BENCHMARK.md`. This file describes the final runtime architecture after that experiment was closed.

## Native execution boundary

MiniMax-H3 sampling enters through the normal ComfyUI guider/sampler `PREDICT_NOISE` path. Actual Spectrum steps execute native H3 transformer blocks and capture only the packed target hidden immediately after the final transformer block and immediately before native `FinalLayer`:

```text
[target audio rows | target video rows]
```

Text, keyframe/reference, image-reference, video-reference, and audio-reference rows are excluded from forecast history.

Forecast steps predict only this compact pre-`FinalLayer` target hidden. The current native FinalLayer and reconstruction path then execute normally, preserving current timestep conditioning, native audio schedule conversion, output heads, video unpatchification, audio unpacking, and return conventions.

No final shipping model-aware feature adds a transformer/denoiser evaluation.

## Model/patch profile

Before a generation, Spectrum lazily constructs a bounded process-local profile of the effective `ModelPatcher`. The profile exists to support Feature 1 scheduling and Feature 2 confidence/adaptive fitting.

The profile retains compact metadata and scalar statistics only:

- base-model identity and patch identity;
- patch counts/coverage;
- recognized LoRA and unknown-patch counts;
- profile confidence;
- aggregate model sensitivity;
- audio/video scalar sensitivities;
- patch and final-block perturbation estimates;
- forecast-risk prior;
- build/memory telemetry.

The scalar stream sensitivities are computed during profile construction from sampled native model weights, including the audio/video output heads. The output-head matrices are **not retained** afterward.

The rejected Feature-3 experiments previously caused the profile to retain detached CPU FP32 copies of `FinalLayer.audio_out` / `video_out` plus Gram diagonals. Those tensors were approximately 2.6 MiB on real H3 and are no longer required by Feature 1 or Feature 2. Final runtime profile construction therefore discards them after deriving the compact scalar sensitivity information.

The process LRU remains keyed by clone lineage, patch identity, H3 architecture signature, and bypass-injection adapter metadata. Cached entries retain no model/module/GPU reference.

## Public mode semantics

The serialized mode values remain unchanged for workflow compatibility:

```text
off
    legacy Spectrum behavior

schedule
    Feature 1 only:
    model/patch risk prior may convert a prospective forecast to an actual

schedule_confidence
    Feature 1 + Feature 2:
    adaptive confidence, degree, ridge and stream blend
    no forecast correction

full
    Feature 1 + Feature 2
    + bounded generic latest-delta residual correction
    no model-specific Feature-3 correction
```

A required actual step is never converted into a forecast by the model-aware controller. Existing sampler/warmup/history/tail constraints remain authoritative.

## Surviving generic scalar correction

The useful correction is independent of the failed model-informed Feature-3 objective.

At a completed actual anchor, Spectrum reconstructs the uncorrected spectral prediction from history available before that anchor and forms:

```text
d = h[-1] - h[-2]
r = h_actual - h_pred_uncorrected
```

The generic scalar projection is:

```text
g_raw = <r, d> / <d, d>
```

The existing controller confidence convention is preserved, and the scalar is passed through the existing rational 0.25 trust region. In the controller's existing scalar notation this is equivalent to the bounded generic gain:

```text
g = g_raw_scaled / (1 + |g_raw_scaled| / 0.25)
```

The exact legacy implementation remains selectable and is the default. The
subsequent generic-scalar research pass adds coordinate-transported/RLS,
correction-reliability, and topology-proven regional VIDEO paths behind the
separate `generic_correction_mode` selector. These paths reuse only the causal
latest-delta direction, add no transformer evaluation, and do not revive retired
model-specific Feature-3 geometry. Exact scalar calibration and the shared CPU evaluator
are described in [GENERIC_CORRECTION_RESEARCH.md](GENERIC_CORRECTION_RESEARCH.md).

The orthogonal `generic_correction_attenuation` selector defaults to
`mode_default`, preserving all previous mode numerics and saved workflows. It
also allows exact live reproduction of the evaluator's four attenuation
families without multiplying mode names. The canonical RLS forgetting factor
remains `0.90`; equivalent clipped candidates are reported as ties rather than
promoting the first sorted lambda.

`full` applies this gain to the latest-delta correction exactly. The retired exact-head and Gram-diagonal candidates no longer alter the applied value:

```text
applied_gain == generic_gain
```

K=2 coefficient vectors are not applied. The current anchor is used to train future generic projection state only after its counterfactual prediction has been formed, preserving the existing causal chronology.

## Feature-3 negative result and runtime retirement

The following investigated mechanisms are not part of normal runtime:

- exact static output-head trust-mixed correction;
- Gram-diagonal correction ablation;
- K=2 causal trajectory correction;
- `W^T W d` transformed trajectory direction;
- `J_t^T J_t d` transformed trajectory direction;
- previous-hidden-residual persistence as a correction source;
- `W^T e_previous` output-error adjoint;
- `J_t^T e_previous` current-FinalLayer adjoint.

The large experimental modules, FinalLayer geometry hooks, sampled previous-error state, previous-error alpha state, transformed-direction row histories, normalization runtime, JVP/VJP runtime, and associated snapshot/logging/workspace extensions were removed from the production package.

Debug summary marks the scientific/runtime state explicitly:

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

## Generic evidence storage

Feature 2 and the legacy generic scalar correction continue to use the existing deterministic sampled hidden evidence. The advanced RLS/reliability candidates reduce exact full-stream `A/B/C` moments through bounded chunks and retain only scalars. Calibration export is enabled only for debug full single-pass runs and serializes no hidden payload.

Diagnostic native-audio calibration uses the proven channel-major stereo layout
and combines matching left/right time slices into one-second start/end bands at
the native 40 Hz latent rate (or deterministic thirds for short clips), plus a
middle band. Aggregate AUDIO moments are reconstructed from the three bands.
This changes neither live audio geometry nor transformer NFE.

Exact-head projected-row history is no longer populated by normal runtime. Full head materialization, exact-head evidence storage, exact-head projection calls, exact-head temporary workspace, and exact-head projection timing are therefore zero in normal `full`.

The regular forecast history still obeys `max_history` and `history_storage`. No second full hidden history is added by the final model-aware architecture.

## Offline smoothing replay

Offline replay remains a separate compatibility/audio path. The supported default
`model_aware_replay_generic_correction=false` does not transplant the causal
latest-delta scalar onto the different future-bracket replay direction. The
legacy `true` setting remains available only for regression/scientific
reproduction. No coordinate/RLS, reliability, regional, or limiter-calibration
state is archived or applied by replay. Replay remains transformer-free and
preserves the existing ER-SDE seeded replay ownership rules.

For native `sample_er_sde`, Spectrum still requires the native seeded noise components for replay and does not mutate generator/noise-sampler/solver-derivative state.

## Performance/lifetime invariants

Final model-aware runtime must preserve:

- zero extra transformer NFE;
- no full output-head tensor retained in the profile;
- no generated full hidden tensor transfer to CPU for model-aware correction;
- no explicit `torch.cuda.synchronize()` for timing;
- no previous-error or transformed-direction workspace;
- no exact-head materialization/projection in `full`;
- process-profile cache entries with no model/module/GPU reference;
- no Feature-3 correction work in `schedule` or `schedule_confidence`.

The final real mechanical gate described in `MODEL_AWARE_BENCHMARK.md` is intended to verify these invariants and measure the simplified `full` overhead. It is not a new Feature-3 discovery experiment.
