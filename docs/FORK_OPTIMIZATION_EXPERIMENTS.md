# Forecast-path validation and streamed FinalLayer chunk settings

This branch/PR deliberately contains **only** the optional streamed FinalLayer
chunk budget and the exact-weight identity/latest hold fast path. Prompt authoring
is reviewed independently on a separate feature branch.

## Compatibility and numerical invariants

- Spectrum scheduling, sampler equations, audio blending and default settings
  remain unchanged.
- Single-anchor segments with exactly one nonzero weight equal to 1.0, and with
  matching history/output dtypes, copy directly to the requested destination.
- Weighted segments and dtype conversions continue through the prior FP32
  accumulation order. Row reordering and CPU-history to CUDA output are covered
  by regression tests for FP16, BF16 and FP32.
- CUDA test cases skip on CPU-only runners, rather than producing false GPU-pass
  claims. Tests can be run on a CUDA-equipped ComfyUI environment with
  `python -m pytest -q tests/test_identity_hold_fastpath.py tests/test_forecast_head_budget.py`.

## Optional streamed-head tuning

The default work budget remains **16 MiB**; supported values are 16, 32, 64,
128 MiB. Invalid settings warn and fall back to 16 MiB. Set before starting
ComfyUI; Windows example:

```bat
set SPECTRUM_H3_FORECAST_HEAD_CHUNK_MIB=64
```

The variable affects only the audited CPU-history streamed FinalLayer path, not
the forecast scheduler, GPU-resident history, unrecognized FinalLayer fallbacks,
or any entire-model execution. It is read once at module import/startup.

## Matched CUDA acceptance benchmark (required before merging)

**No production GPU speed or quality results are claimed by these CPU-only tests.**
Use the same model weights, ComfyUI revision, MiniMax H3 native workflow,
seed, prompt, input/reference frames, 124/192 frame count, resolution,
dtype, sampler, step count, attention backend, Spectrum config and replay mode
for each run.

1. Fully restart ComfyUI with 16 MiB (control). Warm up the GPU once.
2. Run at least three identical prompts/seeds, preserving generated media and
   console/debug receipts. Time the full sampler with CUDA synchronization.
3. Instrument `_project_cpu_forecast_stream` using CUDA events or surrounding
   `torch.cuda.synchronize()` and `time.perf_counter()`. Report the
   synchronized forecast-head latency in milliseconds.
4. Use `torch.cuda.reset_peak_memory_stats()` before each timed sampler call,
   and record `torch.cuda.max_memory_allocated()` in MiB afterwards.
5. Record actual and forecast NFEs from Spectrum debug receipts, including
   fallback/replay passes. Check that both counts and sampler topology match.
6. Repeat at 64 MiB (or 32 MiB if 64 MiB OOMs), with no other change.
7. Compare exact outputs or a justified numerical tolerance, and inspect
   decoded video/audio for regressions. Report the measured timings and peaks
   without assuming that larger chunks are necessarily faster.

| Chunk | Head latency, synchronized ms | End-to-end sampler s | Peak CUDA MiB | Actual NFE | Forecast NFE | Output regression |
|---|---|---|---|---|---|---|
| 16 MiB | pending CUDA run | pending | pending | pending | pending | pending |
| 64 MiB | pending CUDA run | pending | pending | pending | pending | pending |

The existing CI is CPU-only, so CUDA parity and GPU benchmark measurements
must be supplied from a CUDA runner/workstation separately.
