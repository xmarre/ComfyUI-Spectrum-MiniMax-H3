# Spectrum MiniMax H3 v0.2.9

## Native fallback for per-token denoise masks

Spectrum now detects the `denoise_mask` and `audio_denoise_mask` model arguments introduced by ComfyUI's MiniMax H3 per-token masking support. When either mask is active, Spectrum evaluates the run through the native MiniMax H3 path and disables feature forecasting for that run.

Forecast output reconstruction currently has one timestep-modulation row per target stream. Per-token masking can assign different timestep rows to individual video or audio tokens, so applying the forecast output head would use incorrect modulation for masked tokens. Native fallback preserves the intended masked-generation semantics. The guard is inert on ComfyUI revisions that do not supply these arguments and on unmasked runs.

## Validation

Regression coverage exercises video and audio mask arguments independently, verifies that the exact mask object reaches the native executor, and confirms that the runtime records an actual passthrough step with forecasting disabled. The existing four-revision ComfyUI compatibility matrix continues to cover unmasked native MiniMax H3 generation.
