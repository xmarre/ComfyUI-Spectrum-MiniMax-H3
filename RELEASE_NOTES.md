# Spectrum MiniMax H3 v0.2.13

v0.2.13 fixes a false native ER-SDE compatibility rejection on Python 3.13. Spectrum could disable ER-SDE and fall back to an untouched native sampler with:

```text
native sample_er_sde implementation is not a reviewed revision
```

The affected ComfyUI `sample_er_sde` implementation had not actually changed. The failure came from Spectrum's source-provenance guard hashing a decorator-stripped Python AST with the runtime default representation from `ast.dump()`.

## Python 3.13 ER-SDE compatibility fix

Python 3.13 changed the default AST dump representation so empty fields are omitted unless `show_empty=True`. Spectrum's reviewed ER-SDE source digests were generated from the full-field representation used by Python 3.12 and older runtimes.

That meant identical native source could produce a different digest solely because ComfyUI was running under Python 3.13. The same interpreter-dependent mismatch affected both the reviewed `sample_er_sde` implementation and native `default_noise_sampler` provenance.

v0.2.13 normalizes the AST representation before hashing:

- Python runtimes that support `show_empty` explicitly use `ast.dump(..., show_empty=True)`;
- older Python versions retain the existing compatible dump path;
- the existing reviewed digest constants remain unchanged;
- genuinely changed or unreviewed ER-SDE source still fails closed.

This keeps the source-contract safety invariant while removing the interpreter-version false rejection.

## Scope

This release does not change ER-SDE solver math, stochastic compensation, solver-space dense output, forecast cadence, offline replay ownership, `s_noise`, `max_stage`, sampler options, or scheduler behavior.

The issue was reproduced independently of scheduler choice. The reported ComfyUI v0.30.0 and v0.33.0 ER-SDE implementations use the same relevant solver body, so changing to the beta scheduler cannot resolve this provenance failure.

## Validation

The fix was validated by reproducing the Python 3.13 digest difference and confirming that normalized full-field AST dumping restores the existing reviewed digests exactly:

- `sample_er_sde`: `55b76bd3a76d44fbd363de39f2ab3ea672c78de9f001f47168b47ec6ff6d2447`
- `default_noise_sampler`: `11cfe81f36f0b43e96c12eff32a4f074f35227a53ca116e837bf268b6383f9ad`

CI now includes a dedicated Python 3.13 job against the reviewed current ComfyUI revision in addition to the existing Python 3.12 multi-revision matrix. The full PR #58 matrix passed before merge.

## Compatibility

- Fixes issue #57 for Python 3.13 environments using reviewed native ComfyUI ER-SDE.
- Python 3.10, 3.11 and 3.12 behavior remains unchanged.
- Python 3.13 is now explicitly included in package classifiers and CI coverage.
- Native fallback behavior remains fail-closed for genuinely unreviewed ER-SDE implementations.
