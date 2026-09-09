# Numerical attention backend history

Production companion: [ComfyUI-Sol-H3 v0.1.0](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.0).

`attention_backend_history_v1` maps provider keys to callbacks accepting `layout`, `options`, and `model` keyword arguments. They return an immutable preflight identity, or None for routing that cannot be predicted. Providers also expose `accept_receipts(receipts)` to qualify actual routing as forecastable. Actual implementations append hashable receipts to the per-call `attention_backend_receipts_v1` list before final-block return.

Provider callbacks are metadata/qualification hooks rather than part of the transformer computation. If a preflight callback or `accept_receipts` fails, Spectrum treats that provider as unprovable and executes actual-only for the affected routing instead of aborting the sampling run. CUDA out-of-memory remains a hard resource failure and is propagated.

The consumer reads preflight before `begin_model_call` and receipts before `observe_actual`. Policy/receipt changes clear stage banks and controllers, invalidate incompatible offline archives and require fresh actuals. Backend identity participates in rollback snapshots. Same-step conflicting routing is executed but not retained as one mixed numerical anchor. Pending old-backend residual probes are discarded when a late receipt resets history.

The first provider identity in a run may be established without a reset only while no backend-dependent evidence exists yet: no retained forecaster history/topology, no current-step model evidence, and no recorded offline archive steps/anchors. This avoids invalidating an otherwise empty offline-capture archive at step 0. A provider appearing after evidence exists, provider removal, or a genuine identity/receipt transition still performs the full conservative reset.

Opaque policies execute actual-only for the affected call. Core Block Sparse Attention currently does not publish this interface, so its compatibility path is actual-only. Offline replay is unavailable when its archive spans incompatible backends; this must not be described as a forecasting speedup. Provider removal also invalidates inherited history.

## Production evidence

The released Sol-H3 v0.1.0 stack on RTX PRO 6000 Blackwell exercised this contract together with VDN API v3, Untwist preprocessing, Diff-Aid and Flow mixed-grid routing. The final schedule was `18` logical calls, `14` actual transformer NFEs and `4` Spectrum forecasts (`low 8/2`, `high 4/2`, `probe 2/0`). The Sol-bypassed control uses `13 actual + 5 forecast`; the additional Sol actual is the intentional first `dense -> sol` numerical-backend transition.

This confirms that the contract restores forecasting once the active backend route is provable without weakening real numerical transitions. Whole-workflow Sol timing is not treated as a controlled speed A/B because actual-NFE count, sparse routing/content and cache state differ between those runs.

Validation includes the full Spectrum CPU/current-Comfy matrix, cross-package Sol-H3 native-interop coverage for replacement/provider composition and repeated scopes, and the final real SM120 production run described above.
