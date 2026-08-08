# Spectrum MiniMax H3 v0.2.2

Fixes ComfyUI progress reporting for the default offline smoothing replay path.

## Progress reporting

- The compute-heavy capture pass now reports live progress to ComfyUI instead of running with no visible progress bar.
- Capture and replay share one continuous two-pass range. For a 20-step sampler run, capture reports steps 1–20 and replay completes steps 21–40.
- The normal terminal sampler bar follows capture. The transformer-free replay suppresses its own terminal bar, avoiding a second bar that appears and completes almost instantly.
- External callbacks and previews remain replay-only, so accepted output side effects still occur once per logical sampler step.
- If capture cannot be replayed or replay aborts recoverably, the combined progress range is completed before the valid first-pass result is returned.

## Compatibility

This release changes progress reporting only. Node inputs, saved workflows, sampling schedules, generated-result handling, and the v0.2.1 audio-quality defaults are unchanged.

## Validation

- Confirmed in a live ComfyUI GPU generation: progress is visible during capture and completes through replay.
- 173 tests pass against ComfyUI commit `00d02f2`; the two CUDA-only tests are skipped in the CPU test environment.
- Tests cover successful capture/replay progress, replay-only callback forwarding, incomplete capture, recoverable replay abort, result retention, and runtime cleanup.
