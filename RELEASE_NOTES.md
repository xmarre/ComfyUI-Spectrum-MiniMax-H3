# Spectrum MiniMax H3 v0.2.3

Restores the ComfyUI Manager installation path and brings the README in line with the current v0.2.1+ audio behavior.

## Comfy Registry and Manager installation

- Adds a root `.comfyignore` so Comfy Registry packages retain the complete runtime node while excluding `.github/` workflows and `tests/`.
- Earlier Registry archives included development-only files that read test environment variables, launch an isolated test subprocess, and read release metadata. The Registry scanner flagged those files even though ComfyUI never imports or executes them.
- Publishing v0.2.3 with those files excluded gives Manager a clean Registry version to resolve through its default channel once Registry scanning completes.

## Audio documentation

- Documents the current default audio path: `offline_smoothing_replay=true`, `blend_weight=0.5`, and `audio_blend_weight=0`.
- Separates reproduced pre-v0.2.1 single-pass audio failures from the current offline capture/replay behavior.
- Clarifies that workflows saved with v0.2.0 may retain `offline_smoothing_replay=false` and need it re-enabled once.
- Records increased `degree`, increased `warmup_steps`, and the reported clean 30-step run as historical single-setup evidence, not requirements for the current default.
- Updates the documented ComfyUI validation target and v0.2.2 progress-reporting coverage.

## Compatibility

Runtime node code, inputs, saved-workflow compatibility, sampling schedules, and generated-result handling are unchanged. This release changes Registry packaging and documentation only.

## Validation

- Verified the Registry archive excludes every `.github/` and `tests/` path while retaining every runtime Python module, `pyproject.toml`, README, and license file.
- Verified the excluded paths cover every scanner finding reported for v0.2.1.
- The documentation branch passes Markdown fence and internal-anchor validation.
- The test matrix passes, including 173 tests against ComfyUI commit `00d02f2854892ee5b9808bc2f6348b972017886a`; two CUDA-only tests are skipped in the CPU environment.
