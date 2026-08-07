# Spectrum MiniMax H3 v0.1.6

Restores Spectrum on current ComfyUI after the native MiniMax H3 `ModelSamplingAV` update and makes native cache conflicts fail safe.

## Fixed

- Remove `time_shift_slope` from the unconditional native-helper requirement after ComfyUI removed it in `bdcb886a`.
- Match both native MiniMax H3 audio-velocity contracts: derivative-scaled output on the original core and raw `_forward` output when `ModelSamplingAV` performs the schedule conversion outside Spectrum's wrapper boundary.
- Detect native EasyCache or LazyCache on the same model branch before opening a Spectrum transaction. Spectrum remains inactive for that run and the cache continues normally, avoiding `Spectrum H3 solver step completed without an H3 model call`.

## Validation

- Add an exact reconstruction test that feeds a captured native final-block feature through the forecast output path and compares both video and audio velocity with native `_forward`.
- Run native-equivalence CI against the original H3 integration commit and current ComfyUI commit `0dd9b154a1654fc699dcdc3af066c7cce096045a`.
- Add regression coverage proving EasyCache/LazyCache presence bypasses Spectrum without starting or advancing runtime state.
- Validate the full CPU suite locally on both sides of the upstream `ModelSamplingAV` change: 80 passed and one CUDA-only equivalence test skipped on each core.

## Documentation

- Document the supported native contract range and the mutual exclusion between Spectrum and native EasyCache/LazyCache on one model branch.

## Scope

The automated native tests use tiny CPU fixtures. Contributor testing reported four successful full MiniMax H3 generations with the ComfyUI compatibility patch on an AMD Radeon AI PRO R9700 / ROCm system. Exact-seed full-checkpoint A/B validation of the new ComfyUI audio path remains outstanding.
