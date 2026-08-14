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

## Nodes and execution boundary

The research category contains:

```text
Spectrum H3 Objective Media Stage
Spectrum H3 Objective Quality Compare
Spectrum H3 Objective Quality Compare (Staged)
```

The direct compare node accepts three decoded `IMAGE` batches and, optionally,
three decoded `AUDIO` values. Current ComfyUI uses `[frames, H, W, C]` IMAGE
tensors and AUDIO dictionaries with `waveform: [B, C, samples]` plus an integer
`sample_rate`.

The staged workflow is recommended for a full-resolution triad. Put one media
stage after each branch's video/audio decoders. A stage moves that branch's
decoded tensors to CPU immediately and writes nothing to disk. The staged
compare receives those three CPU objects. This avoids retaining all three
decoded outputs in VRAM while preserving their decoded tensor values.

Neither node enters the Spectrum forecasting path. When unused, the evaluator
does no work, adds no transformer call, retains no tensor, and changes no model
output. Evaluation is CPU-only and video is processed in bounded frame chunks.

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

The compare node rejects frame-count, resolution, channel-count, and material
audio-duration mismatches. Candidate audio is deterministically resampled to the
native-reference rate only when sample rates differ. Channel-count mismatches
are rejected; the core evaluator never silently downmixes.

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
all evaluator state.

`provenance_json` is mandatory. Its `compatibility` object must contain model,
model weights, precision, sampler, scheduler, steps, conditioning, video VAE,
audio decoder, and remaining generation settings. Non-empty R/A/B role records
are also required. Computed topology (frames, resolution, FPS, audio rate,
channels, and length) joins that signature automatically.

Compatible independent triads refresh an aggregate report containing per-seed
verdicts, mean/median advantages, wins/losses/ties, worst regressions, and a
deterministic whole-triad bootstrap interval when at least two cases exist.
Duplicate benchmark-ID/seed pairs are not counted twice. Incompatible settings
produce separate groups. No row-randomized cross-validation is performed.

## Smallest real three-way workflow

1. Start from one known-good native MiniMax H3 workflow and one ordinary new
   seed. Keep the same model loader, conditioning, latent/noise construction,
   sigma shift, sampler, scheduler, steps, VAE decoders, and references for all
   three branches.
2. R bypasses Spectrum. A and B each receive their own `Spectrum Apply MiniMax
   H3` clone configured exactly as listed in the same-input protocol.
3. Decode video with the same VAE and audio with the same audio VAE on all three
   branches.
4. Connect each decoded video/audio pair to one `Spectrum H3 Objective Media
   Stage` node.
5. Connect the R, A, and B stage outputs to `Spectrum H3 Objective Quality
   Compare (Staged)`.
6. Set FPS to 24, use a unique benchmark ID, enter the seed, and fill one
   `provenance_json` object. Queue the workflow once.
7. Read the emitted summary and Markdown path. The sibling JSON contains the
   bounded per-frame/per-window arrays and block-bootstrap intervals. Repeat
   with independent seeds; the compatible aggregate refreshes automatically.

This workflow makes the objective report the first decoded-output gate. Manual
visual/listening notes remain optional supporting observations.
