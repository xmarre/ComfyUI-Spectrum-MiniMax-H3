# Spectrum MiniMax H3 v0.2.9

v0.2.9 promotes the validated generic-correction controller for the opt-in full causal path, adds bounded decoded-media research and reporting, hardens the offline first-pass transition, and preserves native behavior for per-token denoise masks.

## Validated generic-correction default

When `model_aware_mode = full` uses causal generic correction, the validated controller is now:

```text
generic_correction_mode = coordinate_rls
generic_correction_attenuation = no_attenuation
generic_correction_limiter = hard_clip
generic_correction_limit = 0.40
RLS lambda = 0.90
```

Two independent three-run hidden-space compatibility groups selected the same candidate with approximately 15.78% lower normalized reconstruction error than exact legacy, 48 wins / 0 losses per group, and zero worst regression. Three controlled decoded native-H3 R/A/B triads then produced two candidate-favored verdicts, one mixed verdict, and no legacy-favored verdict. Broad VIDEO MS-SSIM, PSNR, and temporal-derivative fidelity favored the candidate in all three triads. AUDIO spectral metrics were generally candidate-favored; normalized correlation and SI-SDR remain phase-sensitive diagnostics outside the predeclared verdict gate.

The decoded validation used native MiniMax H3, ER-SDE, 20 steps, 512x768, 192 frames, 24 fps, eight seconds, and three fixed seeds. Candidate and legacy used identical actual/forecast schedules and transformer-call budgets, with zero model-aware extra NFEs. Their observed sampler wall times were approximately 174.66 and 174.78 seconds. These results do not establish the same ranking for other samplers, step counts, resolutions, prompts, LoRAs, or model variants. The hidden-space percentage is reconstruction-ranking evidence, not a perceptual-quality percentage.

The exact legacy configuration remains available for reproduction and saved workflows:

```text
generic_correction_mode = legacy
generic_correction_attenuation = mode_default
generic_correction_limiter = rational
generic_correction_limit = 0.25
```

The global defaults remain compatibility-safe:

```text
model_aware_mode = off
model_aware_trust_shrinkage = false
model_aware_replay_generic_correction = false
offline_smoothing_replay = true
```

Widget ordering and explicit saved values remain stable.

## Objective decoded-media benchmark and reporting

The recommended sequential R/A/B benchmark now reduces each decoded VIDEO immediately to a deterministic CPU `float16` analysis surface capped at 393,216 pixels per frame and retains AUDIO on CPU as `float32`. It keeps at most two pending benchmark IDs within a 4 GiB analysis-memory limit, preserves source topology in compatibility checks, rejects duplicate/incompatible roles and seeds, releases completed triads on success or failure, and never persists raw media. The original one-shot nodes remain available as explicitly labeled Full Media research alternatives.

Objective comparison rows now expose raw legacy/candidate values, absolute candidate deltas, metric direction, verdict role, display units, and the existing decision-relative advantage where the predeclared gate requires it. Human-facing diagnostics use native units:

- normalized correlation: correlation-point delta;
- SI-SDR and PSNR: dB delta;
- bounded lag: milliseconds.

Diagnostics are visibly separated from verdict-primary and guardrail metrics in Markdown, aggregate output, and the ComfyUI console summary. Existing schema-v1 case JSON remains authoritative and can be normalized, rendered, and aggregated without regenerating the three collected triads. The verdict implementation and thresholds are unchanged.

## Offline replay transition hardening

The completed first-pass executor now returns and tears down before smoother construction. Transition telemetry covers the final actual observation, archive record/finalization, executor return, archive completion, smoother setup, and replay setup. The causal forecaster no longer retains a redundant terminal anchor after the archive becomes its long-lived owner. Completion/setup failures preserve the valid completed first-pass result.

This addresses the observed end-of-first-pass stall boundary without attributing the historical freeze to a replay pass that had not started.

## Native fallback for per-token denoise masks

Spectrum detects the `denoise_mask` and `audio_denoise_mask` arguments introduced by ComfyUI's MiniMax H3 per-token masking support. When either mask is active, Spectrum disables feature forecasting for that run and delegates the call unchanged to the native MiniMax H3 executor before registering a forecast model call.

Forecast reconstruction currently has one timestep-modulation row per target stream. Per-token masking can assign different timestep rows to individual VIDEO or AUDIO tokens, so forecasting that call would apply incorrect output-head modulation. The guard preserves the exact mask object and native semantics. It is inert when the arguments are absent or `None`, including on older reviewed ComfyUI revisions.

## Runtime and research boundaries

The promoted controller retains scalar/small-matrix state only. It adds no transformer call, sampler step, schedule ownership, persistent per-token gain field, or production hidden-tensor retention. Regional correction remains a separate research mode. Calibration captures scalar moments only, remains debug-only, and reports post-generation persistence failures as warnings.

Automatic hidden-space reports distinguish candidate ranking evidence from the separate decoded/perceptual evidence used for promotion. Research reports do not mutate live settings.

## Validation

The release branch is validated by the repository's pinned four-revision ComfyUI matrix. Each job builds the wheel, runs the native-H3 forecaster smoke test, scoped Ruff, `compileall`, and the full pytest suite. Coverage includes promoted defaults, exact legacy reproduction, controller reset/rollback, schedule and NFE invariants, objective schema-v1 normalization, native-unit diagnostic rendering, bounded sequential cleanup/resources, offline replay transition ownership, regional research summaries, malformed research data, video-only calibration groups, node registration, and independent VIDEO/AUDIO mask passthrough.
