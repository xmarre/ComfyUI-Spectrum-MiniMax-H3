# Native RES GPU investigation

`Spectrum RES Investigation` measures the native deterministic `res_multistep`
path. It supplies a controlled intervention for forecast-related visual noise;
it does not establish a quality fix or change ordinary Spectrum defaults.
`res_multistep_cfg_pp` and ancestral RES have different contracts and are outside
this diagnostic node's scope. Their ordinary Spectrum support is unchanged.

## Workflow connection

Apply the PR overlay with the ComfyUI patcher, restart ComfyUI, and connect:

`Spectrum Apply MiniMax H3` → `Spectrum RES Investigation` → the existing guider/sampler.

Keep the existing model, references, prompts, LoRAs, seed, scheduler, sigma shift,
resolution, duration, CFG, attention backend, and video/audio save nodes. Change
only the settings listed in the test matrix. Use native `res_multistep` with 25
steps and `selected_step=23` (zero-based). Set `debug=true` on Spectrum Apply.

| Investigation mode | Behavior |
| --- | --- |
| `trace` | Records the ordinary trajectory. Adds no model evaluations and preserves outputs. |
| `replace_selected` | Attempts the selected forecast, then evaluates the actual model at the same input and sigma. Returns the actual denoised output and commits its exact hidden feature through the existing retry transaction. |
| `all_actual` | Forces actual model calls and disables offline replay on the cloned model. Provides a single-pass exact control. |

The diagnostic node preserves all other Spectrum settings. It clones the model
binding and runtime; the incoming model is unaffected. For normal use, connect
the Spectrum Apply output directly to the guider without the investigation node.

## First GPU comparison

Use the affected workflow's existing `offline_smoothing_replay` setting for B,
C, and D. Give each run its corresponding label.

| Run | Mode | `tail_actual_steps` | Purpose |
| --- | --- | ---: | --- |
| A: `native_control` | `all_actual` | 1 | Exact trajectory and decoded media reference. |
| B: `spectrum_tail1` | `trace` | 1 | Reproduce the forecast-containing tail. |
| C: `replace_step23` | `replace_selected` | 1 | Measure the selected same-state denoised error and replace that forecast with an exact evaluation. |
| D: `spectrum_tail3` | `trace` | 3 | Compare the reported successful tail topology. |

B must report `selected_status="selected_forecast_traced"`. C must report
`selected_status="replaced_with_actual"` and `replacement_count=1` in its causal
or capture report. If step 23 is already actual because of another scheduling
condition, C reports `selected_step_was_actual`; that run did not exercise the
intervention. Do not infer a forecast-quality result from it.

If B reproduces the original 13-actual/12-forecast schedule, C should have
14 actual steps and 11 accepted forecasts. The attempted forecast at step 23 is
discarded and separately counted in `discarded_forecast_model_calls`. Counts
refer to the capture/causal pass; offline replay has its own report and counts.
CFG branch batching can change model-subcall counts without changing solver-step
counts.

Compare B and C first. Their input and denoised hashes through step 22, and their
step-23 input hashes, should match. If they do not, inspect model/backend state
or execution nondeterminism before attributing their final difference to the
intervention. Inspect all decoded outputs for pulsing, motion, fine detail, and
audio; a small tensor error alone does not establish good media quality.

## Offline replay interpretation

Offline mode produces separate `offline_first_pass` and `offline_replay` reports.
The intervention runs only during capture. Its exact feature becomes an
additional replay anchor; the replay phase is observational and reports
`replay_observation_only`. A change in final replay media can therefore come
from both the capture trajectory and the changed smoothing anchor set.

If offline mode is affected, repeat B and C with
`offline_smoothing_replay=false`, keeping everything else fixed. This isolates
the live native RES trajectory. A and this single-pass B can also be compared
directly. For the 25-step penultimate intervention, the remaining step is already
actual. When replacing an earlier forecast, the new exact anchor can change
later Spectrum decisions as well as the solver trajectory; inspect the recorded
decision sequence before treating it as a single-output perturbation.

## Reports and checkpoints

Each completed pass writes a uniquely named JSON report and safetensors file to
`ComfyUI/output/spectrum_h3_res_investigation/`, or the corresponding directory
under a configured ComfyUI output root. The console prints the exact JSON path.
No environment-variable setup is needed.

The JSON contains the real sigma array/dtype, native RES source digest, seed,
Spectrum configuration, phase, decisions, coordinates, per-stream input/output
hashes and statistics, and native update coefficients. The selected replacement
also records full-output RMSE, relative L2, maximum absolute error, and cosine
for video and audio separately. A byte-level input witness verifies that both
model calls leave the selected latent unchanged. These measurements are after CFG and H3 output
mapping, before KSampler's output inpaint mask. Coefficients describe native
unmasked deterministic RES; they are not a bound on nonlinear propagation, and
masking can reduce the error delivered to the integrator.

The safetensors file holds full input/denoised checkpoints at step 0 and around
the selected step, the attempted forecast and exact replacement when measured,
and final returned latent streams. Checkpoints are copied to CPU, bounded to
these selected states, and released after report writing. Their size depends on
latent resolution/duration; `checkpoint_tensor_bytes` reports the retained size.
The report does not retain transformer hidden histories beyond Spectrum's normal
history/archive lifetime. Diagnostics add synchronization and file output, so
their wall time is not an acceleration benchmark.

`completed=true` and `valid_trace=true` identify a complete observed native H3
transaction sequence. Failed runs are marked incomplete. A report-writing failure
is logged explicitly and does not discard otherwise completed media.

To compare two JSON/safetensors pairs, the repository supplies
`tools/analyze_res_investigation.py`. It checks matching seed, sigma schedule and
dtype, latent shape, masking status, and native RES source digest, then compares
full checkpoint tensors per stream. It also reports whether the trajectories
were identical through the selected input. It does not evaluate decoded video
or audio quality. For example, from the custom-node directory:

```bash
python tools/analyze_res_investigation.py /path/to/spectrum_tail1_single_pass_REPORT.json /path/to/replace_step23_single_pass_REPORT.json --output /path/to/res_comparison.json
```

Return the JSON reports, corresponding safetensors files, saved workflow,
console log, and decoded outputs together. The workflow carries model, prompt,
reference, and external-patch identity that the numerical report does not capture.

## What the experiment can establish

- Matched B/C prefixes plus a measured same-state error identify the denoised
  difference introduced at the selected forecast.
- Differences at the following input and final latent measure persistence along
  those observed trajectories. They do not demonstrate a general error bound.
- Improvement in C's decoded output supports the selected forecast's causal
  contribution under that fixed configuration.
- A good result with a three-step tail does not identify why the forecast failed.
- Native RES coefficients, synthetic tests, and CPU output parity do not qualify
  H3 audiovisual quality. A runtime correction needs checkpoint GPU evidence.
