# Spectrum MiniMax H3 v0.2.14

v0.2.14 hardens native ER-SDE offline smoothing replay around a reproduced WSL hard-wedge boundary. In the captured failing run, Spectrum completed the entire first pass, released causal history, constructed and validated the offline smoother, entered transformer-free replay, and then stopped producing output after replay step 13. That trace rules out the previously suspected post-run teardown boundary and narrows the protection target to work occurring inside replay.

## Native ER-SDE replay preview protection

KJNodes Model Preview Override performs observational preview work from the sampler callback, including GPU preview/decode work and a blocking GPU-to-CPU transfer before its asynchronous encoder handoff. Offline replay can drive that callback much faster than the transformer-backed capture pass while the full replay archive is still live.

For **native ER-SDE offline replay only**, Spectrum now recognizes the reviewed KJNodes nested callback provenance and bypasses the KJ preview wrapper while forwarding directly to the underlying Spectrum replay callback. This preserves Spectrum's replay progress and external callback semantics without performing KJ's preview decode/copy during the fast replay.

The guard is deliberately narrow:

- KJNodes preview remains active during the expensive first-pass capture;
- native ER-SDE single-pass behavior is unchanged;
- other sampler and replay callback paths are unchanged;
- arbitrary callback wrappers are never unwrapped;
- the KJ callback is accepted only when its module, qualified name, source-file suffix and `original_callback` closure shape match the reviewed implementation.

This release does **not** claim that the historical WSL wedge has been proven to originate inside KJNodes. The callback is the strongest narrowed protection target from the reproduced trace, and the additional diagnostics below make a subsequent real run classify the remaining boundary precisely if the wedge recurs.

## Replay boundary diagnostics

With `debug=true`, native ER-SDE replay now records boundary breadcrumbs around:

- sampler callback begin/end;
- stochastic noise-sampler begin/end;
- successful replay step finalization.

These breadcrumbs do not reduce latent tensors and do not introduce an explicit CUDA synchronization.

The replay `finalize_step_end` breadcrumb is installed outside the v0.2.12 external-patch finalizer. It therefore means the effective finalization chain has completed, including Diff-Aid compatibility bookkeeping, rather than reporting completion before an outer finalizer has finished.

## Existing compatibility preserved

The conflict resolution for PR #55 was performed against v0.2.13 rather than taking the older v0.2.11 branch state. v0.2.14 therefore retains:

- v0.2.12 Diff-Aid H3 forecast compatibility and hard-transition protection;
- v0.2.13 Python 3.13 native ER-SDE source-provenance normalization;
- the v0.2.11 ER-SDE stochastic-state and solver-space dense-output correction.

There is no change to native ER-SDE solver math, RNG ownership, `s_noise`, `max_stage`, forecast cadence, scheduler behavior, or the normal actual/forecast NFE budget.

## Live preview behavior

For offline smoothing replay with native ER-SDE, KJNodes Model Preview Override now provides the useful first-pass preview and is intentionally skipped during the transformer-free replay. For paths outside that exact combination, the existing preview behavior remains unchanged.

## Validation

The reconciled PR #55 head passed the complete repository test matrix before merge:

- four reviewed ComfyUI revisions on Python 3.12;
- the current reviewed ComfyUI revision on Python 3.13;
- forecaster smoke test;
- Ruff and `compileall`;
- focused external-patch compatibility suites;
- native MiniMax H3 test suite.

Dedicated regression coverage verifies strict KJ callback recognition/rejection, callback begin/end behavior including exception propagation, debug-off inertness, and the required wrapper ordering that keeps replay finalization telemetry outside external-patch bookkeeping.
