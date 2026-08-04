# Spectrum MiniMax H3 v0.1.4

Adds optional device-resident history storage for systems with spare VRAM.

## Added

- Add `history_storage=system_ram|vram` to the Spectrum node, defaulting to the existing system-RAM behavior.
- Keep model-dtype history on the producing device in VRAM mode, avoiding actual-step device-to-host copies and repeated forecast-time host-to-device reads.
- Clone the target view into compact owned device storage so cached history does not retain the complete final-block hidden tensor.
- Report the configured storage and resolved history device in debug summaries.

## Highlights

- System-RAM mode remains the default and preserves existing saved-workflow behavior.
- VRAM mode preserves the forecasting math, model dtype, FP32 accumulation order, scheduler, and fallback semantics.
- History is still bounded by `max_history` and released at run teardown.

## Memory guidance

- The supplied 0.5 MP single-branch workflow retained about 2.22-2.27 GiB at `max_history=8`; a two-branch topology at the same shape would use roughly twice that amount.
- At the native 1344x768, 124-frame example, eight two-branch snapshots approach 6.1 GiB.
- VRAM mode also needs headroom for native model execution, the current snapshot, the prediction result, an FP32 accumulation chunk, and allocator fragmentation.

## Measured 0.5 MP A/B results

Three supplied full-checkpoint Euler pairs measured total times of 112.43/105.55 s, 115.60/116.08 s, and 109.80/107.26 s for system RAM/VRAM respectively. The combined means were 112.61 s and 109.63 s, a 2.6% advantage for VRAM mode. Individual pairs ranged from 6.1% faster to 0.4% slower, so the performance benefit is workload- and system-dependent.

## Validation

- Automated tests cover setting validation, default compatibility, compact owned storage, storage-device tracking, and CPU/CUDA prediction equivalence.
- Existing native-path equivalence, transaction, sampler-contract, scheduler, and bounded-memory tests remain in place.

## Current limits

The supplied 0.5 MP tests verify the expected additional allocation and show a small, variable timing benefit. Other resolutions, durations, CFG topologies, reference modes, and hardware remain unverified. Selecting VRAM storage with insufficient headroom can raise an out-of-memory error.
