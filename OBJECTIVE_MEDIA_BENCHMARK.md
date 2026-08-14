# Objective decoded-media benchmark

This research tool compares two accelerated MiniMax H3 outputs against the same
full-compute native result after video and audio decoding:

```text
R = native MiniMax H3, Spectrum bypassed
A = accelerated legacy Spectrum
B = accelerated correction candidate
```

The native result is a full-reference target for the acceleration question. It
is not a claim that the native sample is ground-truth reality. Hidden-feature
forecast accuracy and decoded-media fidelity remain separate evidence layers.

## Recommended node: sequential capture

For ordinary real testing, use:

```text
Spectrum H3 Objective Media Capture (Sequential - Recommended)
```

Add **one** capture node as a side branch from the same decoded IMAGE/AUDIO that
already feeds the normal video-save/combine path:

```text
decoded IMAGE ──┬──> Video Combine / normal save path
                └──> Objective Media Capture (Sequential)

decoded AUDIO ──┬──> Video Combine / normal save path
                └──> Objective Media Capture (Sequential)
```

The capture node does not replace Video Combine and does not output media for
Video Combine. It is an objective-measurement side branch only.

Run the normal workflow three times with the same `benchmark_id`, seed, FPS,
provenance, and generation inputs. Change only the benchmark role and the
Spectrum configuration appropriate to that role:

```text
run 1: role = R - native reference
run 2: role = A - legacy Spectrum
run 3: role = B - candidate
```

The order does not matter. Each incomplete role is moved to CPU RAM and retained
only in the current ComfyUI Python process. When the third compatible role
arrives, the node automatically calls the same objective evaluator used by the
one-shot nodes, writes JSON/Markdown reports, prints the summary, and releases
all raw R/A/B tensors immediately. Raw video/audio is never written to disk.

Incomplete capture memory is bounded to three benchmark IDs and 12 GiB total.
Old incomplete benchmark IDs may be evicted with a warning to stay inside the
bound. `Spectrum H3 Objective Media Capture Reset` can release one incomplete
benchmark or all incomplete captures explicitly. Restarting ComfyUI also clears
incomplete captures.

A duplicate role for the same benchmark is rejected. Set
`reset_before_capture=true` on the next capture to restart that triad cleanly.
R/A/B captures for one benchmark must keep the same FPS, seed,
`provenance_json`, chunk size, decoded video topology, and audio-presence/channel
topology.

The capture node is an `OUTPUT_NODE`, so it runs as part of each ordinary queued
generation. No three-branch generation graph is required.

## One-shot nodes

The research category also retains the original one-shot alternatives:

```text
Spectrum H3 Objective Media Stage (One-Shot)
Spectrum H3 Objective Quality Compare (One-Shot)
Spectrum H3 Objective Quality Compare (Staged One-Shot)
```

These are for workflows that deliberately produce R, A, and B in the **same
ComfyUI execution**.

The direct one-shot compare accepts three decoded `IMAGE` batches and,
optionally, three decoded `AUDIO` values.

The staged one-shot form requires **three separate Media Stage nodes**:

```text
R IMAGE/AUDIO -> Media Stage R -> reference_media ┐
A IMAGE/AUDIO -> Media Stage A -> legacy_media    ├-> Quality Compare (Staged One-Shot)
B IMAGE/AUDIO -> Media Stage B -> candidate_media ┘
```

A `staged_media` output is only an internal CPU bundle for the staged compare.
It never connects to Video Combine. The original decoded IMAGE/AUDIO can fan out
normally to both Video Combine and the research stage.

For normal single-generation workflows where R, A, and B are produced in three
separate queue executions, use the sequential capture node instead.

Current ComfyUI uses `[frames, H, W, C]` IMAGE tensors and AUDIO dictionaries
with `waveform: [B, C, samples]` plus an integer `sample_rate`.

Neither the sequential nor one-shot path enters the Spectrum forecasting hot
path. When unused, the evaluator does no work, adds no transformer call, retains
no tensor, and changes no model output. Evaluation is CPU-only and video is
processed in bounded frame chunks.

## Required same-input protocol

R, A, and B must use the same seed, prompt/multimodal description, references,
model and weights, precision, resolution, frame count, FPS, sampler, scheduler,
step count, conditioning, video VAE, audio decoder, LoRAs, and all remaining
generation settings.

Use these Spectrum roles:

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
audio-duration mismatches. Candidate audio is deterministically resampled to the
native-reference rate only when sample rates differ. Channel-count mismatches
are rejected; the core evaluator never silently downmixes.

## Provenance

`provenance_json` records compatibility grouping and the R/A/B role definitions.
The node now ships with a **valid** default for the current R/A/B controller
family so the evaluator does not fail merely because placeholder fields are
empty. The default marks model/precision/decoder details as
`same-workflow`/`user-unverified-same-workflow`.

For scientifically meaningful aggregation across multiple benchmark IDs/seeds,
replace those placeholders with the exact model weights, precision, sampler,
scheduler, conditioning, VAE/decoder, and remaining generation settings. A
single triad can still be evaluated with the valid default, but its aggregate
provenance should be treated as user-unverified.

## VIDEO panel

All video metrics compare A to R and B to R. IMAGE values are checked against
ComfyUI's decoded `[0, 1]` range and the first three color channels are scored.

| Metric | Direction | Purpose |
|---|---:|---|
| SSIM | higher | Single-scale local luminance/contrast/structure diagnostic. |
| MS-SSIM | higher | Primary multiscale structural-fidelity signal. |
| PSNR dB | higher | Pixel-error diagnostic; never the sole decision signal. Exact equality is capped at 120 dB for valid JSON. |
| Temporal derivative error | lower | Three-scale L1 difference between `R[t]-R[t-1]` and `X[t]-X[t-1]`; exposes freezes, jitter, overshoot, and wrong motion. |
| Global Laplacian detail error | lower | High-frequency spatial-detail difference. |
| Motion-weighted detail error | lower | Laplacian error weighted by normalized native-reference motion energy; emphasizes small moving-detail failures without a hand/face detector. |

Every metric stores mean, median, p05, p95, worst decile, exact worst frame,
per-frame values, and a worst five-frame window. Paired uncertainty uses a
deterministic circular temporal block bootstrap. Individual frames are never
randomized as independent observations.

## AUDIO panel

The evaluator scores decoded waveform tensors before AAC or container muxing.
Primary metrics preserve native timing.

| Metric | Direction | Purpose |
|---|---:|---|
| Multi-resolution log-STFT error | lower | Mean log-magnitude distance at FFT sizes 256, 512, 1024, and 2048. |
| Windowed spectral error | lower | 0.5-second time-local spectral distance with start/middle/end and worst-window summaries. |
| Normalized correlation | higher | Stable raw time-domain shape diagnostic. |
| SI-SDR dB | higher | Scale-invariant reference-fidelity diagnostic; exact equality is capped at 120 dB. |
| Bounded lag | diagnostic | Correlation search limited to ±20 ms. It reports lag and an aligned correlation diagnostic; it never changes the primary scores. |

The dependency-free resampler is recorded as
`torch_linear_align_corners_false`. Same-decoder benchmark triads should normally
have identical sample rates, so resampling should not occur in the main H3 gate.

## Verdict rule

The evaluator does not produce a weighted magic score. It applies a symmetric
Pareto-style dominance rule to explicit primary metrics:

```text
VIDEO: MS-SSIM, temporal derivative error, motion-weighted detail error
AUDIO when present: multi-resolution STFT error
```

A role is favored only when:

1. at least one primary metric improves by 1% or more;
2. no primary metric regresses by more than 2%;
3. worst-frame MS-SSIM and, when present, worst-window spectral error and
   absolute bounded lag do not regress by more than 5%.

The result is `candidate_favored`, `legacy_favored`, or
`mixed_or_inconclusive`. All raw metrics remain visible. Similarity metrics are
reported as higher-is-better and distance/error metrics as lower-is-better.

## Optional learned/external backends

The core result depends only on PyTorch and never downloads a model or accesses
the network.

- LPIPS is detected but is not instantiated automatically. The official
  implementation constructs a pretrained feature trunk by default, which can
  request weights through TorchVision. An absent or unconfigured LPIPS package
  cannot prevent node registration or core evaluation.
- FFmpeg/libvmaf availability is detected. VMAF is not run on the in-memory
  tensor panel because its normal integration requires an FFmpeg build with
  `libvmaf` and a raw/encoded video conversion. It remains a forensic signal,
  not an authority for generative-trajectory fidelity.
- The ViSQOL binary is detected but is not run. Official general-audio mode
  requires 48 kHz and downmixes to mono; speech mode requires 16 kHz. Applying
  those transformations automatically would change the core tensor comparison.

The JSON report records availability and the reason each optional backend was
not executed. Missing LPIPS, libvmaf, or ViSQOL never fails the core evaluator.
Official references:

- <https://github.com/richzhang/PerceptualSimilarity>
- <https://github.com/Netflix/vmaf>
- <https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md>
- <https://github.com/google/visqol>

## Reports, retention, and compatible aggregation

Only JSON and Markdown reports are persisted under ComfyUI's internal user
cache:

```text
__cache/spectrum_h3/objective_media/v1/
```

Writes are temporary-file + fsync + atomic replace. Raw video and audio are
never persisted. Retention is bounded to 24 triads per compatibility group and
24 aggregate groups. Removing the `objective_media/v1` directory safely clears
persisted evaluator reports. Incomplete sequential raw captures are process-RAM
only and are cleared by the reset node or a ComfyUI restart.

Compatible independent triads refresh an aggregate report containing per-seed
verdicts, mean/median advantages, wins/losses/ties, worst regressions, and a
deterministic whole-triad bootstrap interval when at least two cases exist.
Duplicate benchmark-ID/seed pairs are not counted twice. Incompatible settings
produce separate groups. No row-randomized cross-validation is performed.

## Smallest real three-way workflow

Use one normal generation workflow, not three simultaneous generation branches.

1. Branch decoded IMAGE and AUDIO to both the existing Video Combine/save node
   and one `Spectrum H3 Objective Media Capture (Sequential - Recommended)`.
2. Set FPS, one unique `benchmark_id`, the fixed seed, and the provenance once.
3. Run native R with Spectrum bypassed and set capture role to
   `R - native reference`.
4. Configure legacy A exactly as above, change only capture role to
   `A - legacy Spectrum`, and queue again.
5. Configure candidate B exactly as above, change only capture role to
   `B - candidate`, and queue again.
6. On the third captured role, the node automatically emits the objective
   summary and JSON/Markdown report paths, then releases all raw R/A/B media.

The normal videos remain saved through Video Combine on every run. The research
capture is only a side branch.

Repeat with independent seeds when another triad is needed; compatible aggregate
reports refresh automatically. Manual visual/listening notes remain optional
supporting observations.
