# Spectrum MiniMax H3 v0.1.8

Corrects the preliminary one-point-bootstrap configuration introduced in v0.1.7.

## Fixed

- Set `bootstrap_first_forecast=true` for new node instances, alongside the existing preliminary defaults `degree=1`, `warmup_steps=1`, and `tail_actual_steps=1`.
- Align the runtime configuration default, ComfyUI input metadata, and `apply()` fallback so every new or unserialized node path receives the same setting.
- Keep the explicit degree-4 aggressive preset at `bootstrap_first_forecast=false`, because the one-point bootstrap is valid only for degree 1.
- Add regression coverage for all four preliminary defaults and for validation of the aggressive preset.

Existing workflows retain serialized input values. Deterministic RES continues to enforce its sampler-safe three-step minimum.
