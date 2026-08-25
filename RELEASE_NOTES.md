# Spectrum MiniMax H3 v0.2.20

v0.2.20 fixes the RefDelta API-v1 step-provenance handoff after a tracked ER-SDE model call.

## RefDelta tracked-step provenance

- Fixes `RefDelta requested a model-result classification for the wrong step` immediately after the first successful Spectrum/RefDelta model evaluation.
- The real runtime trace showed Spectrum's ER-SDE stochastic tracker had already consumed and logged step 0 while the separate RefDelta bridge-local descriptor had not been mirrored reliably through the surrounding ComfyUI/Continuum model-options path.
- For stochastic RefDelta runs, provenance is now recorded directly from the exact `ERSDEStepDescriptor` that the ER-SDE stochastic tracker successfully consumes. RefDelta therefore uses the same authoritative step classification that Spectrum used for stochastic-state ownership.
- The existing post-model bridge update remains a redundant consistency path instead of the sole source of RefDelta provenance.
- Deterministic `s_noise=0` RefDelta runs keep their direct bridge descriptor path because no stochastic tracker exists in that mode.
- Stale-step and external-increment source validation remain strict; diagnostics now include requested/source and observed step IDs.
- No model-path, Continuum, RefDelta Solver, prompt, sampler-setting, or workflow changes are required.

## Validation

PR #78 adds regression coverage for the exact failure mode: the tracker consumes step 0 successfully without a separate bridge update, and RefDelta must still classify the same step correctly. Coverage also verifies stale-step rejection, external-increment source checking, replay provenance, and deterministic no-tracker behavior.

The complete six-lane Spectrum ComfyUI/Python matrix passes, including Ruff, compileall, focused external compatibility suites, and native MiniMax H3 fixtures. CodeRabbit reported no actionable merge-blocking finding and rated the runtime change minimal risk.
