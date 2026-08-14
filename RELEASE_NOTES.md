# Spectrum MiniMax H3 v0.2.10

v0.2.10 hardens the Objective Sequential Capture research path against host/CUDA memory pressure and makes recoverable capture failures non-fatal to the surrounding ComfyUI workflow. The bounded analysis transform, objective R/A/B semantics, report schema, forecasting behavior, and production Spectrum defaults are unchanged.

## Memory-safe Objective Sequential Capture

Sequential capture now performs validation and memory admission before allocating the retained analysis destination. The preflight accounts for the decoded source media already live, the new bounded VIDEO/AUDIO destinations, existing pending captures, bounded staging workspaces, and explicit host/CUDA safety margins.

The retained research format remains the same:

- VIDEO is reduced deterministically to the bounded CPU `float16` analysis surface capped at 393,216 pixels per frame;
- AUDIO is retained as an independent CPU `float32` waveform;
- at most two incomplete benchmark IDs and 4 GiB of retained analysis media are kept;
- raw decoded input tensors are never stored in `_PENDING_CAPTURES`.

Video staging derives its chunk size from a conservative 64 MiB workspace target, capped at four frames. The approximately 0.7 MP, 192-frame float32 case that exceeded that workspace target at four frames now stages three frames at a time. Audio uses bounded sample chunks as well.

Finite/range validation no longer materializes a boolean tensor for every source element. It reduces each chunk with `aminmax`, checks only the resulting scalar bounds, clamps the owned work buffer in place, and releases intermediate representations before the next chunk allocation.

Preflight, staging, and pending insertion are serialized under the pending-state lock so concurrent capture nodes cannot both admit themselves from stale accounting.

## Lower bounded-evaluator peak memory

The bounded VIDEO evaluator now processes legacy and candidate chunks sequentially against one reference chunk instead of materializing both comparison chunks together. Intermediate float32 tensors are released after each comparison.

This reduces the Spectrum-owned evaluator live set without changing the existing bounded metric profile or verdict calculations.

## Recoverable failures no longer abort unrelated output nodes

Objective Sequential Capture now converts recoverable capture/evaluation failures into its status output instead of raising through the ComfyUI graph. This includes validation errors, incompatible or duplicate R/A/B roles, preflight rejection, staging failures, report/evaluator failures, and unexpected non-resource-exhaustion exceptions. Earlier accepted R/A roles remain intact when a later role is rejected before completion, while a completed triad is always released after evaluation succeeds or fails.

This lets unrelated output nodes, including video saving/combine nodes on the same execution, continue when the research capture itself cannot complete.

Actual resource-exhaustion exceptions are deliberately not swallowed. `MemoryError`, CUDA OOM, and non-objective exceptions reporting an out-of-memory condition still propagate so ComfyUI/PyTorch can perform their normal OOM recovery behavior.

## Capture diagnostics and ownership

Major capture boundaries now emit timestamped diagnostics with elapsed time and available memory telemetry where supported. Logged stages include input receipt, preflight, destination allocation, bounded chunk validation/resize/transfer/copy, pending insertion, evaluation, and report persistence.

Host telemetry uses `psutil` when available with platform fallbacks; CUDA captures also report allocator/free-memory state when available. If host telemetry is unavailable, a conservative absolute incremental guard remains in force.

Source ownership is explicit: only newly allocated reduced VIDEO and copied AUDIO tensors enter the pending store. Tests cover source release, reset, eviction, completed-triad teardown, staging failure, and evaluation failure.

## Compatibility

This release does not alter:

- the production feature forecaster or sampler schedules;
- model-aware controller defaults;
- offline smoothing replay behavior;
- the validated `coordinate_rls + no_attenuation + hard_clip + 0.40` full-mode generic-correction default;
- exact legacy generic-correction reproduction;
- Objective Media schema-v1 verdict thresholds or existing collected reports.

The package and source-checkout calibration provenance versions are advanced to `0.2.10`.

## Remaining resource boundary

The preflight is conservative admission control, not an atomic reservation of host or device memory. A real OOM can still occur if system/WSL commit availability changes after preflight, another process allocates concurrently, allocator/backend scratch exceeds the estimate, or an accelerator does not expose usable free-memory telemetry. Those true resource failures remain visible and are not converted into a successful capture status.

## Validation

The PR is covered by the repository's pinned four-revision ComfyUI matrix. The matrix builds the wheel, runs the native-H3 forecaster smoke test, scoped Ruff, `compileall`, and the full pytest suite. Focused sequential-capture coverage includes deterministic staging parity across float16/bfloat16/float32/float64 inputs, bounded workspace accounting, preflight rejection before destination allocation, missing-telemetry fallback, duplicate/topology checks before staging, non-fatal recoverable errors, OOM propagation, source ownership, reset/eviction release, completed-triad teardown, and optional CUDA staging.
