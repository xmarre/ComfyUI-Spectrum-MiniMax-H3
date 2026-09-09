# Spectrum MiniMax H3 v0.2.26

v0.2.26 adds a provider-generic numerical-attention history contract so Spectrum can keep forecasts only across backend routes whose next-call identity and actual execution receipts are provably compatible. The contract was then validated in the released ComfyUI-Sol-H3 v0.1.0 production stack.

## Numerical backend history and receipts

Spectrum now consumes `attention_backend_history_v1` providers before a model call and `attention_backend_receipts_v1` after actual execution.

- A provider preflight callback returns a stable hashable identity for the next numerical route, or `None` when it cannot prove that route.
- Actual providers append hashable receipts and qualify them through `accept_receipts(...)` before Spectrum retains the resulting H3 anchor as forecast-safe.
- Policy or receipt transitions clear incompatible forecaster/controller evidence, invalidate incompatible offline state and require a fresh actual anchor.
- Provider removal is also a numerical transition rather than silently inheriting the old backend history.
- Same-step conflicting receipts are executed but are not retained as one mixed numerical anchor.
- Backend identity participates in rollback snapshots, and a late receipt reset discards residual-probe evidence belonging to the previous numerical backend.

This is intentionally provider-generic. Spectrum does not contain a Sol-H3, Untwist, Diff-Aid or Flow allowlist; each producer has to prove its own routing semantics.

## Fail-closed metadata behavior

Backend-history callbacks are metadata/qualification hooks rather than transformer computation. A failing preflight callback or failing `accept_receipts(...)` therefore makes the affected call actual-only instead of aborting the sampling run. CUDA out-of-memory remains a hard resource failure and is propagated.

The first provider identity can be established without invalidating an otherwise empty history/archive only before backend-dependent evidence exists. A provider appearing later, provider removal, or a genuine policy/receipt change retains the full conservative reset path.

Core Block Sparse Attention still has no predictive backend-history contract and therefore remains actual-only under this mechanism.

## Sol-H3 production validation

The primary integration is released as [ComfyUI-Sol-H3 v0.1.0](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.0). On the production RTX PRO 6000 Blackwell stack with VDN, Untwist, Diff-Aid and Flow, the final schedule was:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

The matched Sol-bypassed control uses `13 actual + 5 forecast`. The one additional Sol-enabled actual is the intentional first low-stage `dense -> sol` numerical-backend transition and remains a safety anchor rather than being hidden by the history contract.

The same production run confirmed direct VDN API-v3 and Flow mixed-grid routing with 1:1 requested/kernel Q-row accounting and zero square-Q expansion. This release makes no large Sol-Attn speed claim from whole-workflow timing: those runs differ in actual-NFE count, sparse routing/content and cache state, so the backend-history result is the restored, correctly qualified forecast schedule rather than an uncontrolled percentage comparison.

## Validation

- PR #104 remains one implementation commit on top of v0.2.25 main before release metadata is applied.
- Its reviewed matrix passed all nine ComfyUI/Python lanes, including the current Comfy Compiler/Aimdo lane.
- CodeRabbit's substantive findings were fixed and resolved: provider metadata failures are actual-only rather than fatal, and initial provider registration no longer destroys an empty offline capture.
- The released Sol-H3 native-interop suite exercises Spectrum with Sol-H3, VDN, Untwist, Diff-Aid and Flow.
- Final real SM120 production execution established the 18 logical / 14 actual / 4 forecast schedule above.

Existing sampler equations, PECE/SEEDS/SA-Solver ownership, Continuum prefix semantics, external-patch exactness rules, generic correction, offline replay policy and the v0.2.25 Comfy Compiler compatibility boundary are unchanged outside numerical-backend history qualification.

---

Previous release notes through v0.2.25 are preserved verbatim in `docs/RELEASE_NOTES_v0.2.25_AND_EARLIER.md`.
