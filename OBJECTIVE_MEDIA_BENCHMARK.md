# Objective decoded-media benchmark

This research benchmark compares two accelerated MiniMax H3 outputs against the same full-compute native decoded result:

```text
R = native MiniMax H3, Spectrum bypassed
A = accelerated legacy Spectrum
B = accelerated correction candidate
```

The benchmark operates on decoded IMAGE/AUDIO before video encoding or audio muxing. Native H3 is the full-reference target for the acceleration-preservation question.

## PR #51 validation result

The predeclared candidate was evaluated on three independent controlled triads
using native MiniMax H3, ER-SDE, 20 steps, 512x768, 192 frames, 24 fps, eight
seconds, and stereo 32 kHz audio. Results were two `candidate_favored`, one
`mixed_or_inconclusive`, and zero `legacy_favored` verdicts. VIDEO MS-SSIM,
PSNR, and temporal fidelity favored the candidate on all three seeds. Motion
detail favored it on two seeds and legacy on one; the small worst-frame losses
remained well inside the declared guardrail.

Audio MR-STFT was generally candidate-favored. One seed had weaker raw
correlation and SI-SDR diagnostics with zero detected bounded lag. Those
phase-sensitive diagnostics were not part of the predeclared verdict gate and
did not produce a repeatable perceptual regression. This evidence supports the
generic-correction default for the tested full-mode ER-SDE setup. It does not
establish the same ranking for other samplers, step counts, resolutions,
prompts, LoRAs, or acceleration schedules.

## Recommended workflow

Use:

```text
Spectrum H3 Objective Media Capture (Sequential - Bounded)
```

Keep the existing Video Combine exactly as it is and fan the same decoded outputs to the capture node:

```text
fixed seed INT ──────┬────> generation seed input(s)
                     └────> Objective Media Capture [generation_seed]

                     ┌────> Video Combine [images]
decoded IMAGE ───────┤
                     └────> Objective Media Capture [video]

                     ┌────> Video Combine [audio]
decoded AUDIO ───────┤
                     └────> Objective Media Capture [audio]
```

`Video Combine [images]` and `Video Combine [audio]` are sockets on the same existing Video Combine node.

Run the same workflow three times with one `benchmark_id` and the same linked seed:

```text
R: role = R - native reference
   Spectrum bypassed

A: role = A - legacy Spectrum
   model_aware_mode = full
   generic_correction_mode = legacy
   generic_correction_attenuation = mode_default
   generic_correction_limiter = rational
   generic_correction_limit = 0.25
   offline_smoothing_replay = false
   model_aware_trust_shrinkage = false
   model_aware_replay_generic_correction = false

B: role = B - candidate
   model_aware_mode = full
   generic_correction_mode = coordinate_rls
   generic_correction_attenuation = no_attenuation
   generic_correction_limiter = hard_clip
   generic_correction_limit = 0.40
   offline_smoothing_replay = false
   model_aware_trust_shrinkage = false
   model_aware_replay_generic_correction = false
```

Keep prompt/conditioning, model and weights, precision, resolution, duration/frame count, FPS, sampler/scheduler, steps, VAE/audio decoder, and other non-Spectrum generation settings identical.

## Sequential capture memory/performance design

The original sequential implementation retained the complete decoded R/A/B IMAGE tensors in process CPU RAM and then ran the full-resolution dense structural evaluator synchronously when the third role arrived. That design was unsafe for real H3 video sizes: it could hold several GiB of decoded float media while starting a CPU- and memory-bandwidth-heavy SSIM/MS-SSIM pass.

The recommended sequential node no longer retains full decoded video.

At the end of each role it immediately constructs a deterministic bounded analysis representation:

- source frame count is preserved;
- source topology/resolution is recorded and still used for compatibility checks;
- RGB is deterministically downscaled only when necessary to at most 393,216 analysis pixels per frame;
- retained analysis video is CPU `float16`;
- staging is performed in small frame chunks;
- AUDIO is retained on CPU as float32;
- full-resolution decoded IMAGE is released with the normal ComfyUI execution after the capture node returns;
- raw video/audio is never persisted to disk by the benchmark.

Incomplete sequential state is bounded to 2 benchmark IDs and 4 GiB total analysis RAM. Old incomplete IDs may be evicted with a warning if required by the bound. Restarting ComfyUI clears incomplete captures. `Spectrum H3 Objective Media Capture Reset` can clear one pending benchmark or all pending captures explicitly.

When the third compatible role arrives, the node evaluates only the bounded analysis surfaces. The sequential video profile is versioned as:

```text
sequential_bounded_luma_block_ssim_v1
```

It uses bounded luma block-SSIM/MS-SSIM, RGB PSNR as a diagnostic, luma temporal-derivative error, global luma Laplacian detail error, and native-motion-weighted luma detail error. The same documented comparison/verdict/temporal-bootstrap machinery is reused. AUDIO retains the existing MR-STFT, normalized-correlation, SI-SDR and bounded-lag panel.

The evaluator prints explicit VIDEO/AUDIO start/end timings so any future slow point is visible in the console instead of appearing as a silent post-generation stall.

Bounded-profile reports are compatibility-isolated from the older full-resolution objective profile. They must not be numerically mixed as if they were the same metric implementation.

## Controls

- `role`: R, A, or B.
- `fps`: actual output FPS; identical across the triad.
- `benchmark_id`: unique ID for one triad; identical across R/A/B.
- `generation_seed`: required link-only INT. Connect the exact same fixed seed source that drives generation. The capture node owns no independent seed widget and no `control_after_generate` state.
- `steps`: generation step count.
- `compatibility_tag`: short identifier for the unchanged model/precision/scheduler/conditioning/decoder setup.
- `frame_chunk_size`: bounded evaluator frame chunk size.
- `reset_before_capture`: restart this incomplete benchmark before storing the current role.

The third compatible role automatically evaluates, writes report files, prints the summary, and releases all retained R/A/B analysis tensors in a `finally` path.

## Reports

Reports are stored under ComfyUI's internal cache:

```text
__cache/spectrum_h3/objective_media/v1/
```

A completed triad produces:

```text
report_json_path
report_markdown_path
aggregate_json_path
aggregate_markdown_path
```

Only JSON/Markdown metrics and metadata are persisted. Raw media is never persisted.

Comparison rows retain raw legacy/candidate values, absolute candidate delta,
metric direction, metric role, and the existing decision-relative fraction.
Human reports use correlation-point deltas for normalized correlation, dB deltas
for PSNR/SI-SDR, and millisecond deltas for bounded lag. These three diagnostics
are visibly separated from verdict primary and guardrail rows. Existing v1 JSON
reports remain valid; the additive fields are derived while loading/rendering,
so completed triads do not need to be regenerated.

## One-shot full-media alternatives

These remain research alternatives for a workflow that intentionally has all three roles available in one execution:

```text
Spectrum H3 Objective Media Stage (One-Shot / Full Media)
Spectrum H3 Objective Quality Compare (One-Shot / Full Media)
Spectrum H3 Objective Quality Compare (Staged One-Shot / Full Media)
```

They retain/evaluate full decoded media and are not the recommended path for normal sequential H3 testing.

## Decision rule

The benchmark does not collapse results into an undocumented weighted score. Primary metrics remain VIDEO structural/temporal/motion-detail fidelity and, when audio is present, MR-STFT error. The existing 1% material-improvement, 2% primary-regression, and 5% worst-case guardrail rules produce `candidate_favored`, `legacy_favored`, or `mixed_or_inconclusive`.

Hidden-feature generic-correction reports and decoded-media reports remain separate evidence layers. A hidden-space improvement is not reported as the same thing as a decoded perceptual-quality improvement.
