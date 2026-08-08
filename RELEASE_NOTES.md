# Spectrum MiniMax H3 v0.2.3

Restores installation through ComfyUI Manager's default channel.

## Registry packaging

- Adds a root `.comfyignore` so Comfy Registry packages contain the runtime node and exclude `.github/` workflows and `tests/`.
- Earlier Registry versions were automatically flagged because the published archives included development-only code that reads the test environment, launches an isolated test subprocess, and reads the package version in the release workflow.
- The runtime node contains none of those flagged operations; they were packaging false positives from files that are never imported or executed by ComfyUI.

## Compatibility

This release changes Registry packaging only. Node code, inputs, saved workflows, sampling behavior, generated results, and the v0.2.2 progress reporting fix are unchanged.

## Validation

- Verified the Registry archive file set excludes `.github/` and `tests/` while retaining every runtime Python module, `pyproject.toml`, README, and license file.
- Verified the excluded files cover every finding reported by the Comfy Registry scanner for v0.2.1.
