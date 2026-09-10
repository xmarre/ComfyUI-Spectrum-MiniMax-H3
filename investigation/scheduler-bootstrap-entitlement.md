# Deferred first-forecast bootstrap investigation

Status: proposed scheduling rule for a development mirror. No production scheduler changes or numerical safety claim. This note addresses single-stage, non-state-conditioned runtime paths; solver-owned bootstrap and independent multistage histories require separate analysis.

## Existing behavior

`SpectrumH3Runtime.begin_step` orders forced execution, feedback refresh, Continuum prefix, sampler prefix, warmup, final tail, sampler exact steps/stages and stage refresh before bootstrap. Configured bootstrap requires degree one, non-state-conditioned residuals, exactly one retained actual anchor and `policy_step_id == 1`. Normal fitting follows bootstrap and requires `min_fit_points`.

`sampling._continuum_actual_prefix` parses the explicit interop contract. `sampling._continuum_prefix_for_phase` preserves that prefix for sampling and first-pass execution; only offline replay receives zero. The prefix is an execution requirement, not scheduler credit.

`SpectrumH3Runtime._reset_backend_history` clears feature histories, model-aware controllers, branch topology, consecutive forecasts and feedback state. It forces the current step actual. `prepare_backend_history` applies policy identity transitions; `observe_backend_history` independently applies receipt transitions. Consequently a cold-to-primed carry design must address both transitions while preserving raw receipt checks.

`finalize_step` advances the adaptive window only for an eligible `adaptive_recompute` actual. Prefix and insufficient-history actuals do not advance it. Resetting feature history does not reset the adaptive window. Changing the window cannot make a one-anchor normal fit ready: the insufficient-history branch is evaluated first.

After a required dense-to-sparse boundary, the first sparse actual supplies one sparse anchor. At the next step, preserving that anchor across a validated cold-to-primed successor would still fail the current bootstrap check whenever the policy step is greater than one.

## Candidate rule

Treat configured one-point bootstrap as permission to initialize the **first forecasting episode of a run**, deferred while actual execution is mandatory:

1. Initialize one unused bootstrap entitlement at run start, only when the existing bootstrap configuration is valid.
2. Keep all existing forced-execution, prefix, tail, sampler and feedback precedence checks.
3. After those checks, allow a one-point hold when the entitlement is unused, exactly one compatible actual anchor exists, and the current path is the supported single-stage non-state-conditioned path. Require that anchor to belong to the current numerical backend segment and immediately preceding completed logical step. A current backend transition can still override the proposed forecast to actual.
4. Retire the entitlement when **any forecast successfully commits**, including a normal fitted forecast. A later history reset never replenishes it.
5. Keep post-forecast refresh and model-aware rejection rules effective. The entitlement is not a waiver for unsafe backend state, failed branch validation or incompatible layouts.

This replaces the global step-one restriction for the scoped path with a lifecycle condition. It does not use desired NFE totals, resolution, native/mixed flags, stage length or chunk index. The current solver-owned bootstrap remains unchanged.

The rule deliberately extends the existing scheduling policy. Permission for one-point hold at the start of a run does not prove equal accuracy later in denoising. The proposed forecast must be evaluated against the actual hidden output, with subsequent stale-calibration execution measured separately.

## Conditional schedule derivation

For the observed five-call low-stage structure, assume a hard dense-to-sparse boundary at logical step 2, a protected final step 4, no additional forced work, and numerically accepted sparse cold-to-primed carry:

| Path | Step 0 | Step 1 | Step 2 | Step 3 | Step 4 |
| --- | --- | --- | --- | --- | --- |
| No two-step Continuum prefix | Actual anchor | Existing bootstrap; retire entitlement | Actual sparse boundary; clear dense history | Actual; entitlement already retired and only one sparse anchor | Actual tail |
| Two-step Continuum prefix | Required actual | Required actual; entitlement unused | Actual sparse boundary; clear dense history | Deferred bootstrap from sparse anchor; retire entitlement | Actual tail |

Thus exactly one additional low-stage forecast becomes eligible in this pair of paths. The two paths differ because only one has already executed its first forecast. This is a conditional source-level derivation, not a measured production schedule or guarantee of 14 actual / 4 forecast calls. Additional model-aware, interop or backend decisions can still require actual execution.

The corresponding three-call high-stage sequence can retain its existing step-one bootstrap if the first anchor's cold-to-primed successor is accepted. Its final tail stays actual. This is separate from the low-stage scheduler extension.

## Rejected shortcuts and counterexamples

- Bootstrap after every history reset would add a second bootstrap to the first low path, changing more than the missing forecast and allowing repeated history resets to repeatedly authorize one-point holds.
- Granting entitlement based on a Continuum flag hardcodes the motivating caller. The first-forecast lifecycle also explains an equivalent mandatory prefix supplied through another supported contract.
- Retiring only after bootstrap leaves permission available after an ordinary fitted forecast. A later reset could then introduce an unrelated one-point forecast; retiring on any committed forecast prevents this.
- Consuming permission at `begin_step` loses it on backend veto, incomplete conditional branches, retry or aborted execution. Consumption must occur after `finalize_step` validates the complete forecast transaction.
- Reusing dense history to fit the first sparse forecast violates the numerical-backend boundary. Deferred bootstrap uses only the sparse anchor.
- Increasing adaptive window credit cannot overcome the earlier history-readiness check and changes unrelated schedule decisions.
- A long prefix that supplies two compatible anchors does not need one-point bootstrap. Normal fitting proceeds; its first committed forecast retires the entitlement.
- A prefix extending into the tail produces no deferred bootstrap. An entitlement can expire unused.

## Implementation and transaction constraints

Store explicit run-local lifecycle state rather than deriving it from cumulative telemetry. `RuntimeRollbackSnapshot`, `create_rollback_snapshot` and `restore_rollback_snapshot` must preserve that state so discarded speculative forecasts do not permanently consume permission. `abort_step` and `prepare_actual_retry` must not consume it. `start_run` resets it; backend resets do not. Offline replay must not create new online bootstrap opportunities.

If independent stage histories are later supported, define ownership per history lane and snapshot all lane state explicitly. Do not silently grant a global entitlement separately to every solver stage. Maintain current solver-owned state-conditioned bootstrap until its residual contract is independently extended and tested.

## Focused test matrix

Use the full runtime with actual branch transactions and backend prepare/receipt observation. Initial `begin_step` decisions alone are insufficient because backend checks can change the effective mode.

| Case | Required result |
| --- | --- |
| Five-call low path, no prefix, hard boundary at 2 | Only existing bootstrap at 1; no new bootstrap at 3 |
| Same path, explicit two-step prefix | Prefix 0 and 1 actual; boundary 2 actual; deferred bootstrap at 3; tail 4 actual |
| Same paths with cold-to-primed carry unavailable | Backend veto remains actual; no unsafe forecast |
| Repeated backend transitions after first committed forecast | No renewed entitlement |
| Two-anchor prefix without backend transition | Ordinary fit; first fitted forecast retires entitlement |
| Prefix/tail overlap, force-actual, warmup, sampler exact steps and refresh | Existing precedence preserved; permission can remain unused |
| Bootstrap disabled or incompatible degree/configuration | Existing behavior preserved |
| Missing, stale, wrong-layout or wrong-backend anchor | No deferred bootstrap |
| Model-aware veto and unsafe receipt | Actual execution; entitlement not spent |
| Incomplete branch set, retry and abort | No entitlement consumption before successful forecast commit |
| Rollback before first forecast | Restore unused entitlement and original schedule |
| Rollback after first forecast | Restore retired entitlement |
| New run and offline replay | New online run initializes permission; replay does not consume or grant online permission |
| Existing state-conditioned and multistage suites | Unchanged behavior for paths outside the initial scope |

CPU transaction tests can establish these structural properties. Matched production shadow measurements and decoded output comparison are still needed to judge the numerical and perceptual consequences of the newly eligible hold.
