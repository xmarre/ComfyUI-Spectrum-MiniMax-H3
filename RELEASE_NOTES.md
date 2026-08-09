# Spectrum MiniMax H3 v0.2.5

Reduces offline replay GPU memory pressure and enables Spectrum forecasting for Larryvrh's MiniMax H3 Turbo sampler.

## Offline replay memory fix

- Separates the causal forecast history from the full offline replay archive.
- Keeps `history_storage` bounded by `max_history`, including when it is set to `vram`.
- Adds `offline_archive_storage`, defaulting to `system_ram` for new and existing workflows.
- Retains `offline_archive_storage=vram` as an explicit option for systems with sufficient headroom.
- Reports both configured storage locations and the resolved archive device in debug summaries.

### Root cause

Before offline replay, `history_storage=vram` retained at most `max_history` feature snapshots. Offline replay reused that setting for every actual anchor until its second pass completed. The meaning of the existing option therefore changed from a bounded causal history to an all-anchor CUDA archive. One recorded 20-step run retained about 4.3 GiB for that archive, enough to cause allocator pressure, severe slowdown, or an out-of-memory error on a 12 GB GPU.

The archive now has its own storage policy. Existing workflows do not contain the new trailing input, so they receive the safe `system_ram` archive default automatically.

## MiniMax H3 Turbo sampler support

- Recognizes Larryvrh's exact `_turbo_sampler` contract.
- Applies the existing conservative Euler safeguards: at most one consecutive forecast and one required actual refresh after a forecast.
- Covers the standard offline capture and transformer-free replay path.
- Keeps similarly named and otherwise unknown samplers on the native bypass.

The reviewed Turbo implementation makes one denoiser call per scheduler step and does not inject ancestral noise, churn, or additional model evaluations.

## Compatibility

The new archive selector is a trailing optional node input, preserving existing saved workflows. Existing Euler and RES schedules, offline smoothing, audio handling, progress reporting, and generated-result processing are unchanged. Workflows that intentionally want the former all-anchor CUDA behavior can select `offline_archive_storage=vram`.

## Validation

- The revised default archive behavior was confirmed in a real H3 generation after the reported VRAM regression.
- The combined CPU suite passes 183 tests; four CUDA-only test cases are skipped in the CPU environment.
- Mixed-storage tests cover causal history on CUDA with the archive on CPU and the inverse placement.
- Turbo tests cover exact-name recognition, prefix rejection, policy constraints, callbacks, and complete offline capture/replay.
- Both feature PRs passed the three-version ComfyUI GitHub Actions matrix and CodeRabbit review.
