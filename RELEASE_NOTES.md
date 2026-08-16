# Spectrum MiniMax H3 v0.2.12

v0.2.12 adds coordinated MiniMax H3 interoperability with ComfyUI-DiffAid-Patches v1.0.6. Spectrum now understands Diff-Aid's deterministic text-activation modulation as a versioned runtime contract, protects hard sigma-window regime changes with a real anchor when necessary, and keeps the normal Spectrum acceleration budget instead of misclassifying Diff-Aid modulation strength as parameter-space model risk.

## Diff-Aid H3 interoperability

The supported workflow order is:

```text
Load Diffusion Model
-> MiniMax H3 Diff-Aid Sparse Patch
-> Spectrum Apply MiniMax H3
-> guider / scheduler
```

ComfyUI-DiffAid-Patches v1.0.6 publishes a pure-data descriptor for each enabled, nonzero H3 sparse patch and exposes the exact per-call normalized sigma already used by Diff-Aid's own timestep wrapper. Spectrum consumes that metadata without depending on Diff-Aid internals.

The v1 contract describes the producer, patch kind, MiniMax H3 architecture, unique patch instance, resolved 0-based block indices, model block count, strength, sigma window/ramp, token weighting, conditional-only mode and text-modulation scope. Runtime entries identify the same instance and carry the normalized sigma for the current model call.

The contract contains no model objects or retained tensors.

## Hard sigma-window correctness

For a partial hard window (`sigma_ramp=0`), Spectrum compares the producer's exact active/inactive state with the last successfully finalized solver step.

If a patch crosses a hard on/off boundary on a step that Spectrum had scheduled as a forecast, that current step is promoted to one actual H3 evaluation so forecast history immediately receives an anchor from the new modulation regime.

The transition state is committed only after successful step finalization. Retry, abort and rollback paths therefore cannot advance the external-patch regime early. Multiple contracts crossing on the same solver step still require at most one promoted actual evaluation.

A full `[0,1]` hard window has no interior transition and adds no compatibility NFE. Smooth `sigma_ramp>0` modulation remains continuous and is not converted into forced refreshes solely because its gain changes.

The hard-boundary guard is independent of `model_aware_mode`; it remains active even when model-aware scheduling is off because it is a forecast-correctness rule.

## Model-aware risk separation

Real testing exposed an important calibration error during development: the first consumer implementation accumulated Diff-Aid's activation-modulation strength across affected blocks and fed that raw structural magnitude into Spectrum's parameter-space model-aware patch prior.

A five-block H3 patch at strength `0.5` produced raw structural telemetry of `2.112114`, saturated `patch_risk=1.0`, and turned a normal 20-step ER-SDE run into 16 actual + 4 forecast steps with 10 unnecessary model-aware NFEs.

v0.2.12 fixes the semantics at the profile source:

- recognized external activation patches still contribute to patch/runtime counts and cache identity;
- their raw structural magnitude remains available as `external_patch_runtime_perturbation` telemetry;
- they do not inflate the calibrated parameter-space `patch_perturbation`, `final_block_perturbation`, profile confidence or forecast-risk prior;
- ordinary online Spectrum trajectory evidence remains free to request actual evaluations when the observed trajectory itself becomes risky.

This keeps the safety mechanisms that are calibrated to Spectrum's trajectory while removing a unit/meaning mismatch between Diff-Aid activation strength and parameter-patch perturbation.

## Real MiniMax H3 validation

The final producer/consumer pair was validated in restarted ComfyUI on native MiniMax H3 with ER-SDE, 20 steps, `model_aware_mode=full`, using the five-block Diff-Aid H3 patch `1,13,25,37,50` at strength `0.5`.

Three matched runs produced:

| Run | Diff-Aid window | Actual / forecast | Sampler time | Compatibility result |
|---|---|---:|---:|---|
| Spectrum control | Diff-Aid inactive | 11 / 9 | 187.236 s | no external contract |
| Diff-Aid full window | `[0.0, 1.0]`, ramp 0 | 11 / 9 | 189.001 s | 0 transitions, 0 forced actuals |
| Diff-Aid partial window | `[0.0, 0.95]`, ramp 0 | 11 / 9 | 184.946 s | 1 transition, 0 forced actuals |

The partial-window run detected the exact off→on boundary at normalized sigma `0.947368` on solver step 8. That step was already an actual refresh, so the transition required no additional H3 transformer call.

All three runs retained the normal 11 actual + 9 forecast budget and reported `model_aware_extra_nfes=0`. The small runtime differences are within normal run-to-run variance; the compatibility path itself did not introduce a meaningful sampling penalty.

In the tested multi-shot prompt, Diff-Aid retained its stronger prompt-following/visual prompt representation. Full-window activation changed the intended shot transition into a more continuous shot, while `sigma_end=0.95` restored the intended cut between the first and later shots while retaining the prompt-adherence enhancement. This is an empirical workflow result rather than a universal model default.

## Fail-safe behavior

Declared compatibility metadata is validated strictly. Malformed or inconsistent external metadata does not abort an otherwise valid generation; Spectrum fails safe to all-actual sampling for that run and records a contract failure.

Validation covers schema/provider/instance identity, architecture/kind/scope, block mappings, model block count, scalar ranges, runtime instance ordering and finite normalized sigma values. Runtime metadata is validated without mutating producer-owned dictionaries.

Unknown external model modifications that do not publish the reviewed contract remain outside this compatibility path.

## Transaction and teardown hardening

The compatibility layer preserves Spectrum's existing transaction order around model execution, ER-SDE consumption, retries, finalization and abort handling. A dedicated transaction-parity tripwire compares the critical native and compatibility operation ordering so future sampling edits cannot silently diverge.

The compatibility end-run wrapper also preserves the underlying post-run safety hook as a distinct callable rather than rebinding it, keeping the teardown invariant auditable.

Offline replay does not re-promote hard transitions already handled during the causal capture pass.

## Compatibility requirements

- Spectrum MiniMax H3: **v0.2.12 or newer**
- ComfyUI-DiffAid-Patches: **v1.0.6 or newer** for the producer half of this contract
- Native MiniMax H3 packed-model path as already required by Spectrum

Older Diff-Aid releases do not publish the compatibility descriptor/runtime sigma metadata and therefore do not provide this coordinated path.

This release does not change the global Python defaults for `offline_smoothing_replay`, `model_aware_mode`, audio blend, or the validated full-mode `coordinate_rls + no_attenuation + hard_clip + 0.40` generic correction.

## Validation

The final PR head is covered by Spectrum's four-revision pinned ComfyUI matrix. The workflow builds the wheel, runs the native-H3 forecaster smoke test, scoped Ruff, `compileall`, focused external-compatibility suites, transaction-order parity tests and the native-H3 test suite on every reviewed ComfyUI revision.

The companion Diff-Aid v1.0.6 producer PR is independently green, including its focused/full compatibility tests and CodeRabbit review.

The package and source-checkout calibration provenance versions are advanced to `0.2.12`.
