# Objective decoded-media benchmark

This research tool compares two accelerated MiniMax H3 outputs against the same
full-compute native result after video and audio decoding:

```text
R = native MiniMax H3, Spectrum bypassed
A = accelerated legacy Spectrum
B = accelerated correction candidate
```

Native H3 is the full-reference target for this acceleration-preservation
question. Hidden-feature forecast accuracy and decoded-media fidelity remain
separate evidence layers.

## Recommended workflow: one side-branch node, three normal runs

For ordinary testing, use only:

```text
Spectrum H3 Objective Media Capture (Sequential - Recommended)
```

Add one capture node as a side branch from the same decoded outputs that already
feed the normal video-save/combine path:

```text
decoded IMAGE ──┬──> Video Combine / normal save path
                └──> Objective Media Capture (Sequential - Recommended)

decoded AUDIO ──┬──> Video Combine / normal save path
                └──> Objective Media Capture (Sequential - Recommended)
```

The capture node does not replace Video Combine and does not output media for
Video Combine. It is a measurement side branch only.

Run the same normal workflow three times. Keep `benchmark_id`,
`generation_seed`, FPS, steps, `compatibility_tag`, prompt, model, resolution,
decoders, and all other non-Spectrum generation settings unchanged. Change the
capture role and Spectrum configuration for each run:

```text
run 1: role = R - native reference
run 2: role = A - legacy Spectrum
run 3: role = B - candidate
```

The order does not matter. Each incomplete role is moved to CPU RAM and retained
only in the current ComfyUI Python process. When the third compatible role
arrives, the node automatically runs the objective evaluator, writes bounded
JSON/Markdown reports, prints the summary, and releases all raw R/A/B tensors.
Raw video/audio is never written to disk by the evaluator.

The node is an `OUTPUT_NODE`, so it runs as part of each ordinary queued
generation. No three-branch generation graph is required.

### Sequential node controls

- `role`: `R - native reference`, `A - legacy Spectrum`, or `B - candidate`.
- `fps`: actual output FPS; keep fixed across the triad.
- `benchmark_id`: one unique identifier for the triad; keep fixed across R/A/B.
- `generation_seed`: the actual generation seed as a decimal **STRING**. It is
  deliberately not an INT seed widget, so ComfyUI does not attach the
  seed `control after generate` randomizer to this benchmark metadata field.
- `steps`: native sampler step count, normally 20 for the current ER-SDE gate.
- `compatibility_tag`: short user assertion identifying the unchanged
  model/weights/precision/scheduler/conditioning/VAE/audio-decoder setup. Keep it
  fixed across compatible seeds and change it when those generation settings
  change.
- `frame_chunk_size`: CPU evaluator chunk size.
- `reset_before_capture`: clear an incomplete triad with this benchmark ID
  before storing the current role.

The recommended sequential node intentionally has no multiline
`provenance_json` field. It creates valid R/A/B provenance automatically from
`steps`, `compatibility_tag`, and the fixed controller-role definitions. The
result records that compatibility as user-asserted rather than pretending the
capture node can inspect every upstream model-loader setting.

A duplicate role for the same incomplete benchmark is rejected unless
`reset_before_capture=true`. R/A/B captures for one benchmark must have matching
FPS, generation seed, steps, compatibility tag, frame chunk size, decoded video
topology, and audio presence/channel topology.

Incomplete capture memory is bounded to three benchmark IDs and 12 GiB total.
Old incomplete benchmark IDs may be evicted with a warning to stay inside the
bound. `Spectrum H3 Objective Media Capture Reset` can release one incomplete
benchmark or all captures explicitly. Restarting ComfyUI also clears incomplete
captures.

## Exact R/A/B Spectrum settings

All three roles must otherwise use the same input and generation settings.

```text
R native
  Spectrum bypassed; every transformer step is native

A legacy
  model_aware_mode = full
  generic_correction_mode = legacy
  generic_correction_attenuation = mode_default
  generic_correction_limiter = rational
  generic_correction_limit = 0.25
  offline_smoothing_replay = false
  model_aware_trust_shrinkage = false
  model_aware_replay_generic_correction = false

B candidate
  model_aware_mode = full
  generic_correction_mode = coordinate_rls
  generic_correction_attenuation = no_attenuation
  generic_correction_limiter = hard_clip
  generic_correction_limit = 0.40
  offline_smoothing_replay = false
  model_aware_trust_shrinkage = false
  model_aware_replay_generic_correction = false
```

The evaluator rejects frame-count, resolution, channel-count, and material
audio-duration mismatches. Audio is deterministically resampled to the native
reference rate only when sample rates differ. Channel-count mismatches are
rejected; the core evaluator never silently downmixes.

## One-shot alternatives

The research category retains the original one-shot nodes for workflows that
intentionally produce all three roles in the same ComfyUI execution:

```text
Spectrum H3 Objective Media Stage (One-Shot)
Spectrum H3 Objective Quality Compare (One-Shot)
Spectrum H3 Objective Quality Compare (Staged One-Shot)
```

The staged one-shot form requires three separate Media Stage nodes:

```text
R IMAGE/AUDIO -> Media Stage R -> reference_media ┐
A IMAGE/AUDIO -> Media Stage A -> legacy_media    ├-> Quality Compare (Staged One-Shot)
B IMAGE/AUDIO -> Media Stage B -> candidate_media ┘
```

`staged_media` is only an internal CPU bundle for that one-shot comparator. It
never connects to Video Combine. The original IMAGE/AUDIO outputs can fan out to
both Video Combine and the research nodes.

For normal R/A/B generation in three separate queue executions, use the
sequential capture node instead.

## VIDEO metrics

All metrics compare A to R and B to R.

| Metric | Direction | Purpose |
|---|---:|---|
| SSIM | higher | Single-scale local luminance/contrast/structure diagnostic. |
| MS-SSIM | higher | Primary multiscale structural-fidelity signal. |
| PSNR dB | higher | Pixel-error diagnostic only. |
| Temporal derivative error | lower | Three-scale error on frame-to-frame changes; exposes freezes, jitter, overshoot, and wrong motion. |
| Global Laplacian detail error | lower | High-frequency spatial-detail difference. |
| Motion-weighted detail error | lower | Detail error weighted by native-reference motion energy; emphasizes small moving-detail failures. |

Video reports include aggregate, percentile, worst-frame, per-frame, and
worst-window values. Paired uncertainty uses deterministic temporal block
bootstrap rather than treating frames as independent samples.

## AUDIO metrics

Audio is scored on decoded waveforms before AAC/container muxing.

| Metric | Direction | Purpose |
|---|---:|---|
| Multi-resolution log-STFT error | lower | Log-magnitude distance at FFT sizes 256/512/1024/2048. |
| Windowed spectral error | lower | 0.5-second local spectral distance with start/middle/end and worst-window summaries. |
| Normalized correlation | higher | Raw time-domain shape diagnostic. |
| SI-SDR dB | higher | Scale-invariant reference-fidelity diagnostic. |
| Bounded lag | diagnostic | ±20 ms correlation search; never changes primary unaligned scores. |

## Verdict rule

The evaluator does not collapse the panel into an undocumented weighted score.
Primary metrics are:

```text
VIDEO: MS-SSIM, temporal derivative error, motion-weighted detail error
AUDIO when present: multi-resolution STFT error
```

A role is favored only when at least one primary improves by 1% or more, no
primary regresses by more than 2%, and the worst-frame/worst-window/lag
Guardrails do not regress by more than 5%. The verdict is
`candidate_favored`, `legacy_favored`, or `mixed_or_inconclusive` and all raw
metrics remain visible.

## Optional backends

The core evaluator depends only on PyTorch and never downloads weights or uses
the network.

- LPIPS package availability is reported, but pretrained weights are not fetched
  automatically.
- FFmpeg/libvmaf availability is reported; VMAF is not required for the in-memory
  core evaluator.
- ViSQOL availability is reported; it is not silently applied because its modes
  require sample-rate/downmix transformations.

Missing optional backends cannot prevent node registration or core evaluation.

## Reports and retention

Only JSON and Markdown reports are persisted under ComfyUI's internal user
cache:

```text
__cache/spectrum_h3/objective_media/v1/
```

Writes are atomic. Raw media is never persisted. Incomplete sequential captures
exist only in bounded process CPU RAM. Compatible independent triads refresh an
aggregate report; duplicate benchmark-ID/seed identities are not counted twice,
and incompatible compatibility groups remain separate.
