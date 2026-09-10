# Core-BSA pool transition investigation

Status: investigation checkpoint only; no production implementation change. This file belongs to the development mirror and must not be consolidated into the implementation PR.

## Revisions and evidence

Spectrum PR #106 was re-fetched at 041b1de91779e59f9997d0b5ff5091c6929e37dc, one commit over be95adecec0b85c80d0c9fc5dd8d07386d50aaee. Its original head is preserved on checkpoint/core-bsa-before-pool-transition-20260910. Flow PR #26 remains unchanged at 860f308cc4f84f920d0c1facd4ece028584a02a0.

Read the complete downloaded metrics_00343_.json, metrics_00317_.json and the relevant lines of Pasted text(20260910-030334).txt. Pasted text(20260910-025048).txt is downloaded but not used for conclusions.

The metrics confirm:
- 00343: 18 logical, 17 actual, 1 forecast; low 9A/1F, high 6A/0F, probe 2A.
- 00317: 18 logical, 14 actual, 4 forecasts; low 8A/2F, high 4A/2F, probe 2A.

The completed production log confirms 16 audited safe preflights and 15 accepted actual calls with receipts=50 expected=50. Five preflights have flow_mixed=True and seq_len=43557. Sparse stages retain two dense blocks and use 48 chunked sparse blocks. Do not summarize the first route entry alone: it is dense even during sparse stages.

The counters establish the NFE difference; they do not by themselves establish identical seeds, settings, or decoded quality.

## Pool semantics from exact comfy-kitchen v0.2.33

Tag commit: e9ea99cf2f0af1d0c49c04690d4153a91c2b8668.

Primary sources:
- https://github.com/Comfy-Org/comfy-kitchen/blob/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668/comfy_kitchen/backends/cuda/__init__.py
- https://github.com/Comfy-Org/comfy-kitchen/blob/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668/comfy_kitchen/backends/cuda/sage_attention/sol_attn_producer.cu
- https://github.com/Comfy-Org/comfy-kitchen/blob/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh
- https://github.com/Comfy-Org/comfy-kitchen/blob/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu
- https://github.com/Comfy-Org/comfy-kitchen/blob/e9ea99cf2f0af1d0c49c04690d4153a91c2b8668/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu

sol_attn_chunked cold execution traverses the QKV producer twice. The first traversal collects uncentered post-RoPE K sums and V absolute maxima; the second quantizes using those current-input statistics. Primed execution traverses once and uses the supplied statistics from the previous actual call.

K centering changes the quantized carrier. In exact arithmetic subtracting one shared key vector leaves softmax unchanged, but this identity does not eliminate INT8 rounding error or prove equality of the sparse implementation.

V scaling is directly numerical:
    s = max(1e-8, 1.1 * previous_absmax / 127)
    q = clamp(round_to_nearest_even(v / s), -127, 127)
The core consumes the same supplied scale when using the quantized carriers. Values beyond the previous maximum's 10% headroom can clip. The next statistics are collected from current unquantized inputs, independently of the previous scales.

These are calibration statistics, not an output cache and not merely an allocation optimization. Their continuous updates already occur within the primed identity. Consequently numerical dependence alone neither proves a hard cold/primed boundary necessary nor proves history carry safe.

An accepted cold call initializes a plausible successor calibration state from its own input. That supports investigating a narrowly validated successor transition. It does not prove the accuracy of forecasting a different timestep. Even matched-input cold/reused-stat comparisons can differ slightly because Python and CUDA reduction orders for K means differ.

Skipping a transformer also skips its statistics refresh. The subsequent actual then uses older calibration data. Validation must measure the subsequent actual as well as the proposed forecast. A same-input cold/primed comparison alone cannot establish this multi-step effect.

The cold producer's second QKV traversal is real existing work but is not a second complete H3 transformer NFE.

## Spectrum source trace and additional scheduler blocker

core_bsa_compat.probe retains BOTH route_specs and pool_ownership in its identity. Cold to primed changes route names and replaces missing entries with tensor lifetime generations. Renaming the routes alone does not remove this boundary.

runtime.prepare_backend_history clears history on identity change. runtime.observe_backend_history separately clears history on receipt change. A transition design must address both deliberately while continuing to verify original raw receipts.

In runtime.py's existing decision chain, bootstrap_first_forecast requires:
- non-state-conditioned residual mode;
- degree == 1;
- policy_step_id == 1;
- exactly one history entry.

The solver-owned bootstrap also requires policy_step_id == 1.

Dense to sparse remains a required hard boundary under the task's safety constraints. Low step 2 therefore leaves only one sparse anchor. At low step 3, retaining that anchor across cold to primed still cannot satisfy either bootstrap clause. Normal fitting requires additional history.

Executed the actual AST-extracted scheduler if/elif chain from the unmodified runtime.py with one retained history entry:
- three-step stage, prefix 1, step 1: forecast, one-point bootstrap forecast;
- five-step stage, prefix 0, step 3: actual, insufficient actual history;
- five-step stage, prefix 2, step 3: actual, insufficient actual history.

This is targeted execution of the existing decision code with scalar fixtures, not a full runtime or CUDA test.

Conditional on an otherwise successful cold/primed carry implementation, unchanged scheduling can recover the two high-stage forecasts but cannot recover the missing low-stage forecast. The resulting upper bound from this change alone is 15A/3F, not 14A/4F.

A generic bootstrap-after-every-backend-reset rule would expose step 3 in BOTH low stages, potentially producing 13A/5F. The requested 14A/4F therefore also requires a defensible scheduling rule that explains the different low-stage choices. Do not encode arbitrary stage/count exceptions to match the target.

## Next bounded diagnostic and remaining gate

Before changing production:
1. At the candidate transition, retain cold-call calibration and capture actual primed execution on matched inputs. Compare cold versus supplied calibration to separate quantization differences from timestep differences.
2. Measure the proposed hidden forecast against the actual next-timestep hidden output.
3. Measure clipping/range headroom at the following actual when the intervening pool refresh is skipped. Preserve and restore owner pool state around diagnostic alternatives; diagnostics must not alter the control trajectory.
4. Resolve low-stage scheduling explicitly, preserving dense-to-sparse boundaries, Continuum prefixes, final tails, and existing quality settings.
5. Only then implement a narrowly source-gated transition with raw receipt validation and owner/layout/UUID/settings invalidation tests, and run the controlled production gate.

No NVIDIA runtime is exposed in this workspace (nvidia-smi unavailable); default Python also lacks torch. No CUDA experiment or decoded quality comparison was performed. No claim of recovered NFEs or completed fix is supported. Neither PR was changed or merged.
