# CUDA evidence: metrics_00364 / schema-v4 BSA transition probe

Production CUDA run supplied 2026-09-10 on RTX PRO 6000 Blackwell. The run retained the unchanged safe schedule (18 logical / 17 actual / 1 forecast) while exercising the schema-v4 one-file-per-generation diagnostic.

Observed structural results:

- one generation-scoped JSONL captured four Spectrum runtime segments;
- four cold->primed candidates were accepted with predecessor pool continuity;
- all four adjacent following-actual measurements completed, including both high-grid stages whose semantic identity changed;
- every measured shadow restored BSA pool and RNG state;
- every diagnostic transformer/forecast counter delta was zero;
- all four run segments ended with `errors=0` and `pending_next_actual=false`;
- no skipped diagnostic candidates were reported.

Attention relative-L2 ranges across sampled sparse blocks (2 / 26 / 49):

| segment | cold->primed | stale-calibration next actual |
| --- | --- | --- |
| native low | 0.0116 / 0.0152 / 0.0857 | 0.0114 / 0.0153 / 0.0781 |
| first high | 0.0106 / 0.0149 / 0.0647 | 0.0100 / 0.0148 / 0.0681 |
| Mixed-Grid low | 0.0105 / 0.0148 / 0.1317 | 0.0104 / 0.0149 / 0.1366 |
| second high | 0.0100 / 0.0147 / 0.0898 | 0.0097 / 0.0147 / 0.0876 |

The one-point-hold diagnostic remains deliberately conservative/raw: video relative-L2 versus oracle actual hidden target was 0.5969, 0.3475, 0.6773 and 0.4451 across the four candidate segments. These are not final blended-output errors and are not treated as direct quality thresholds.

Interpretation for implementation work:

The structural question is now answered: cold->primed is a real, adjacent calibration-state transition with exact pool ownership continuity in production, including the two high-grid stages. A skipped actual necessarily leaves stale BSA calibration for the next exact call; the measured consequence is finite and bounded in this run but non-zero, especially at the final sampled block. Any production history carry therefore must be one-way, source/audit gated and must not pretend that the BSA calibration refresh occurred.

This evidence justifies implementing the previously derived one-shot deferred-bootstrap scheduler entitlement together with the narrow one-way cold->primed history carry as an experimental production candidate for direct CUDA/visual validation. It does not, by itself, establish final video-quality equivalence; that requires a real 14A/4F production run.
