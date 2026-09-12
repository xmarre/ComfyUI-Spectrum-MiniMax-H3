# Spectrum MiniMax H3 v0.2.27

v0.2.27 combines the unreleased Core BlockSparseAttention forecast-recovery work from PR #106 with PR #107's forecast streaming and CUDA target-lifetime fix. The release restores the reviewed Mixed-Grid production schedule while substantially reducing Spectrum's forecast-head VRAM pressure.

## Streamed H3 forecast projection and retained-target VRAM fix

PR #107, contributed by @Pizzawookiee, removes two independent sources of excessive CUDA memory use.

Spectrum can now keep a predicted H3 hidden state in system RAM and stream it through the audited MiniMax H3 FinalLayer using bounded reusable CUDA workspaces instead of materializing the complete forecast hidden representation on the GPU.

- The default forecast-head workspace budget is 16 MiB.
- Audio and video rows are projected independently while preserving the native packed layout and timestep selectors.
- State-conditioned residual reconstruction uses the same bounded strategy instead of reintroducing a full hidden-sized CUDA allocation.
- Native FinalLayer streaming is accepted only when the current bound method matches the reviewed MiniMax H3 implementation.
- H3-Optimizations is accepted only through its reviewed source/marker contract, including cube-order selector/state remapping and output restoration.
- Unknown or foreign FinalLayer patches keep the historical one-shot monolithic projection path instead of being slab-called under an unproven contract.

A separate actual-step lifetime bug is also fixed. The final hidden target view is now retained on CUDA only while an active residual probe actually needs it, and the reference is cleared after comparison and on executor failure. Ordinary actual steps therefore no longer pin successive full final-hidden storages through the callback closure.

### Measured memory and timing

The contributor supplied matched 0.1 MP / 15 s / 20-step CUDA comparisons for both stock/native token order and H3-Optimizations sparse/cube order.

| Path | Incremental forecast-head CUDA peak | Whole-run wall time |
| --- | ---: | ---: |
| Stock/native | 253.78 -> 84.44 MiB (**-66.7%**) | 435.28 -> 448.14 s (**+3.0%**) |
| H3-Optimizations cube order | 253.68 -> 84.43 MiB (**-66.7%**) | 315.85 -> 309.37 s (**-2.1%**) |

The streamed FinalLayer itself is materially slower in that small benchmark (roughly 6-7x forecast-head latency), so this release does **not** claim a forecast-head speedup. The observed benefit is lower CUDA memory pressure; whole-workflow timing in the supplied matched runs remained approximately flat.

Both post-change runs completed the expected 20/20 logical calls as **11 actual / 9 forecast / 0 fallbacks**. Decoded-output comparisons were also supplied (stock PSNR 21.38 dB / SSIM 0.709; cube-order PSNR 20.48 dB / SSIM 0.733). These are real-media quality checks, not strict numerical or bitwise parity claims.

## Core BlockSparseAttention forecast recovery

PR #106 adds a fail-closed Spectrum contract for reviewed ComfyUI core `BlockSparseAttention`, including Flow Mixed-Grid execution, and recovers forecasts that were previously lost at the audited `h3_chunked_sparse_cold -> h3_chunked_sparse_primed` transition.

The carry is intentionally narrow. Spectrum retains history across that transition only when the predecessor/current calls are adjacent in the same run and the reviewed BSA owner, source, patch generation, sequence layout, UUIDs, numerical identity and exact calibration tensor ownership all remain proven. Dense-to-sparse, reverse transitions, changed owners/layout/settings/source and every unproven route still reset to an actual H3 evaluation.

The scheduler also gains a transactional deferred one-point bootstrap entitlement for ordinary degree-1 runs when `bootstrap_first_forecast` is enabled. The entitlement can survive exact-prefix/backend-veto steps but is consumed only by a successfully finalized forecast; state-conditioned residual, separate-stage history and offline replay remain excluded.

Real production validation on the RTX PRO 6000 Blackwell stack with VDN, DiffAid, Untwist, core BSA, Spectrum, Continuum and Progressive Mixed-Grid Flow reached:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

Compared with the safe 17-actual / 1-forecast control, the later Mixed-Grid sampler measured **179.745 s -> 132.611 s (-26.2%)**, and the Mixed-Grid high stage measured **82.362 s -> 51.141 s (-37.9%)**. Controlled visual inspection found no apparent quality degradation and no clear winner between the safe control and recovered 14A/4F output.

## Compatibility and validation

- PR #106's final implementation was validated across the full 10-job matrix and by real SM120 production execution.
- PR #107's final head passed the full CI matrix after review, and the squash-merged `main` commit `27d178d3e9b22cdabd3b48fec4d9f13f617cc62e` subsequently passed push run #672 across the repository test workflow.
- The #107 rebase preserves the existing #106 BSA/backend-history ordering and actual-execution receipt completion contract.
- Regression coverage now includes streamed-vs-monolithic row/selector behavior, H3-Optimizations cube-order restoration, state-conditioned reconstruction/fallback, bounded sanitization, source gating, unknown-wrapper one-shot behavior, core-BSA ownership/history proofs and cold-to-primed recovery.
- CUDA out-of-memory remains a hard resource failure; unsupported or unproven external contracts continue to fail closed to actual execution rather than silently weakening the safety boundary.

Existing sampler equations, PECE/SEEDS/SA-Solver ownership, Continuum prefix semantics, external-patch exactness rules, generic correction, offline replay policy and the v0.2.26 numerical-backend history contract remain unchanged outside the specific BSA-recovery and H3 forecast-memory paths above.

---

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
