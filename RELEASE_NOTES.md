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

- At the supplied 0.5 MP layout, one two-branch history point is about 568 MiB: roughly 2.78 GiB at `max_history=5` or 4.44 GiB at `max_history=8`.
- At the native 1344x768, 124-frame example, eight two-branch snapshots approach 6.1 GiB.
- VRAM mode also needs headroom for native model execution, the current snapshot, the prediction result, an FP32 accumulation chunk, and allocator fragmentation.

## Validation

- Automated tests cover setting validation, default compatibility, compact owned storage, storage-device tracking, and CPU/CUDA prediction equivalence.
- Existing native-path equivalence, transaction, sampler-contract, scheduler, and bounded-memory tests remain in place.

## Current limits

End-to-end speedup and real-checkpoint peak VRAM remain hardware-dependent validation items. Selecting VRAM storage with insufficient headroom can raise an out-of-memory error.
