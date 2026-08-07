# Spectrum MiniMax H3 v0.1.9

Prevents incompatible one-point-bootstrap settings from turning a valid ComfyUI workflow into an execution-time error.

## Fixed

- When `bootstrap_first_forecast` is enabled with `warmup_steps > 1`, keep the requested warmup and automatically disable only the one-point bootstrap for that execution.
- Apply the same normalization when `degree != 1`, since the bootstrap is defined only for degree 1.
- Emit a console warning with the supplied degree and warmup values whenever normalization occurs.
- Add inline ComfyUI tooltips for `degree`, `warmup_steps`, and `bootstrap_first_forecast`, and document the effective behavior in the README.
- Preserve strict `SpectrumH3Config` validation for direct programmatic callers outside the ComfyUI node boundary.

Compatible settings and explicitly disabled bootstrap configurations are unchanged. Normal history-based forecasting continues with the requested degree and warmup values.
