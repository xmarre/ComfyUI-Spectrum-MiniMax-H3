# Spectrum MiniMax H3 v0.2.11

v0.2.11 fixes native ER-SDE forecast-state corruption, removes the characteristic high-frequency/confetti artifact from Spectrum ER-SDE forecast steps, hardens sampler-wrapper compatibility, and prevents optional post-run research from blocking an already-completed generation.

## Native ER-SDE solver-space fix

Earlier Spectrum ER-SDE forecast steps predicted H3 hidden/velocity state without evaluating the current stochastic latent, then reconstructed denoised/x0 against ER-SDE's current `x`. Direct subtraction of the pending stochastic increment corrected one real mismatch term but did not remove the visible artifact in real MiniMax H3 testing.

The stronger mismatch was solver-space consistency: ER-SDE consumes and differentiates denoised/x0 values in its update/history, while the skipped H3 hidden forecast was not a valid current-state denoised observation.

v0.2.11 therefore keeps native ER-SDE's stochastic latent trajectory unchanged and reconstructs skipped solver-facing denoised/x0 values from exact actual denoised anchors:

- the first causal forecast after one exact actual anchor uses the latest exact actual denoised value;
- later causal forecasts use bounded linear dense output in native ER-SDE lambda coordinate from the two latest exact actual anchors;
- degenerate, reversed, non-finite, or excessively long extrapolation falls back to the latest exact actual anchor;
- native `s_noise`, RNG draws, stochastic latent updates, stage-2/stage-3 history and terminal behavior remain intact;
- there are zero additional H3 transformer/denoiser NFEs.

The dense output is substituted before native ER-SDE's callback, integration update, derivative history and `old_denoised` assignment.

## Real MiniMax H3 validation

The solver-space candidate was tested on real native MiniMax H3 ER-SDE generations after the simpler exact-q candidate failed visually.

Observed result:

- forecast confetti/noise disappeared completely;
- final fine/low-resolution visual structure improved modestly but visibly across repeated generations;
- action/motion remained comparable in the tested prompts;
- audio remained comparable;
- the normal 20-step Spectrum budget remained 11 actual H3 evaluations + 9 forecasts;
- no additional H3 transformer NFE was introduced.

A later T2V rerun from an older saved workflow also completed normally after the teardown hardening below.

## Native stochastic ownership and replay safety

Spectrum still tracks ER-SDE's exact additive stochastic increment so ownership is explicit across actual, forecast and replay paths. The stochastic increment remains part of native ER-SDE's latent `x`; Spectrum does not disable, globally attenuate or rescale native ER-SDE noise.

Replay and compatibility paths retain exact stochastic compensation where solver-space dense output is not the reviewed causal path.

`s_noise=0` remains on the untouched native path without stochastic tracker allocation.

## TiledDiffusion sampler-wrapper compatibility

Real runtime testing exposed ComfyUI-TiledDiffusion's variadic `KSAMPLER_sample(*args, **kwargs)` monkeypatch above ComfyUI's native `KSAMPLER.sample`.

The ER-SDE preflight now supports that audited passthrough structure without broadly trusting arbitrary wrappers. The semantic validator:

- verifies exact `*args/**kwargs` delegation;
- validates the stored native target recursively;
- verifies the reviewed TiledDiffusion `model_options` / `sigmas` bookkeeping shape;
- rejects changed argument forwarding, alias mutation, cycles and unrelated state mutation;
- still fails closed to untouched native sampling when the wrapper contract cannot be proven.

## Post-run teardown safety

A real saved-workflow run later hard-wedged WSL after all ER-SDE solver steps and the Spectrum run summary had completed, before `Spectrum H3 run teardown` or downstream VAE/video nodes appeared.

The old generic-correction hook performed optional persistence/evaluation synchronously inside `SpectrumH3Runtime.end_run()`. A filesystem/evaluator stall could therefore block a valid completed sampler result from reaching downstream nodes.

v0.2.11 changes the boundary:

- calibration-block export remains synchronous and bounded;
- core Spectrum runtime/history release remains synchronous;
- optional generic-correction persistence/evaluation is dispatched only after runtime release;
- the research worker is daemonized and bounded to one active job;
- if a previous research job is still active, later diagnostic jobs are skipped rather than accumulating threads;
- the research worker receives only a tensor-free calibration block and cannot retain Spectrum feature-history VRAM;
- teardown debug breadcrumbs identify calibration export, runtime release and research dispatch boundaries.

No speculative `torch.cuda.empty_cache()`, forced CUDA synchronization, or history offload was added.

## README refresh

The README has been rewritten around the current production behavior and now documents:

- the v0.2.11 ER-SDE solver-space path;
- current Python defaults and recommended ER-SDE quality settings;
- supported samplers and fail-closed behavior;
- TiledDiffusion compatibility;
- model-aware/generic-correction defaults;
- offline replay and live-preview behavior;
- attention/cache compatibility;
- memory/storage guidance and teardown diagnostics;
- research nodes and current debugging information.

## Compatibility

This release does not change the global Python defaults for:

- `offline_smoothing_replay=true`;
- `model_aware_mode=off`;
- `audio_blend_weight=0.0`;
- the validated full-mode `coordinate_rls + no_attenuation + hard_clip + 0.40` generic correction.

Saved ComfyUI workflows can retain serialized values from older releases; updating the custom node does not automatically rewrite those widgets.

Unknown or unreviewed sampler/noise/wrapper contracts continue to fail closed to native execution.

## Validation

The release is covered by the repository's four-revision pinned ComfyUI matrix. It builds the wheel, runs the native-H3 forecaster smoke test, scoped Ruff, `compileall`, and the full pytest suite on every reviewed revision.

Focused v0.2.11 coverage includes ER-SDE stochastic ownership, first/terminal boundaries, seeded RNG/replay parity, stages 1/2/3, solver-space bootstrap hold and lambda extrapolation, extrapolation guards, all-actual parity, replay isolation, TiledDiffusion wrapper validation, bounded denoised-anchor lifetime, post-run release/research ordering, bounded background research dispatch and failure cleanup.

The package and source-checkout calibration provenance versions are advanced to `0.2.11`.
