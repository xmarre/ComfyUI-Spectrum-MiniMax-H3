# RES4LYF SDE GPU validation

Spectrum accelerates the named RES4LYF beta samplers `res_2m`, `res_3m`,
`res_2s`, `res_3s`, `res_5s` and `res_6s`, plus their ODE variants. The SDE
path substitutes causal denoised-space forecasts built from exact H3 anchors.
Its decoded audio and video quality requires real H3 validation. CPU tests of
native RES4LYF with a synthetic denoiser establish call topology, noise ownership
and all-actual numerical parity, including shifted AV noise; they do not establish
speech quality or visual quality when H3 calls are skipped.

Support does not require a particular RES4LYF source revision. CPU coverage
includes RES4LYF 1.2.0 on ComfyUI 0.38.0 and the earlier September fixture.
Spectrum checks native API bindings and live stochastic-state ownership.
Record the installed ComfyUI, Spectrum and RES4LYF revisions from the
ComfyUI Patcher, including the Spectrum PR overlay revision, with the results.

## Matched comparison

Use a normal single-chunk H3 workflow with audible dialogue and visible motion.
Keep the model, VAE, prompt, conditioning, references, LoRAs, scheduler, CFG,
denoise, resolution, duration, frame rate, precision and other patches fixed
within each pair. Fix the seed and its post-generation control; disable random
or incrementing seeds. Keep every exposed SDE noise seed fixed as well.
Preserve RES4LYF's `SDE noise seed` log lines when emitted and verify that the
same noise seed was used within each pair; the named wrappers can derive it
from `torch.initial_seed()`.

Select the named sampler through ComfyUI's sampler selection. The generic
`rk_beta` sampler and its configurable guides or RK overrides are not the
reviewed named-wrapper contract. Use the same outer step count within each
pair; 19 or 20 steps is suitable for the initial comparison. Logical H3 call
counts differ between multistep and fixed-stage samplers.

Run these four outputs:

| Sampler | Native control | Spectrum comparison |
| --- | --- | --- |
| `res_2m` | Spectrum node `enabled=false` | Spectrum node `enabled=true` |
| `res_3s` | Spectrum node `enabled=false` | Spectrum node `enabled=true` |

Use these Spectrum settings for the enabled runs:

| Setting | Value |
| --- | --- |
| `degree` | `1` |
| `warmup_steps` | `1` |
| `tail_actual_steps` | `3` |
| `bootstrap_first_forecast` | `false` |
| `offline_smoothing_replay` | `false` |
| `model_aware_mode` | `off` |
| `anchor_residual_feedback` | `false` |
| `selective_rollback_correction` | `false` |
| `debug` | `true` |

Keep the other Spectrum settings fixed and save the workflow. RES4LYF also
enforces exact startup, recurrence refreshes and an exact final outer interval.
A KSampler denoise mask or multi-GPU parallel calls keep SDE runs all-actual;
the initial acceleration comparison needs a workflow without those conditions.

The SDE bridge forecasts both packed streams. `audio_blend_weight=0` does not
make skipped-call audio native. Preserve that setting in the workflow, but
assess the resulting audio directly.

## Verify that acceleration was exercised

Save the complete console log for each enabled run. Check all of the following:

- `Spectrum H3 RES4LYF stochastic tracking active` appears.
- `Spectrum H3 RES4LYF dense output` appears for completed forecasts.
- `Spectrum H3 RES4LYF stochastic tracking finished` reports `invalid_reason=None`.
- The final `Spectrum H3 run summary` reports `forecast_steps>0`,
  `forecast_calls>0` and `disabled=False`. Record `actual_steps`,
  `actual_transformer_calls`, `fallbacks`, logical call count and `wall_s`.

A clean tracking-finished line alone does not prove that any H3 calls were
skipped. An all-actual run cannot validate accelerated media quality. Preserve
fallback reasons when a run does not meet the checks above.

RES4LYF can record one final noise transition without a following H3 call:
the native terminal denoise consumes that landing. The CPU topology tests cover
this terminal behavior; `transitions_recorded` can exceed
`transitions_consumed` by one on a successful run.

## Assess the outputs

Compare each Spectrum output with its same-sampler native control:

- Audio: complete dialogue, timing, intelligibility, stutter, missing or extra
  syllables, noise, bursts and level changes. Listen to the full clip.
- Video: motion, temporal stability, flicker, detail, artifacts and preview
  corruption. Inspect the full decoded clip and the forecast previews.
- Performance: sampling time, actual transformer evaluations, forecast count
  and peak VRAM. Keep model loading and decoding time separate when available.

Save the decoded audio/video outputs, workflows and logs. Bitwise equality is
not expected from accelerated SDE forecasting. A successful queue, clean
tracking or a speed gain does not establish quality acceptance.

If an enabled output regresses, run an all-actual Spectrum control with the
same sampler and seed: set `warmup_steps=64` and keep
`bootstrap_first_forecast=false` for the 19/20-step workflow. Verify
`forecast_steps=0` and `forecast_calls=0`. Compare it with the native control
and, if necessary, a repeated native run to establish the stack's own
reproducibility. This distinguishes tracked native execution from errors caused
by skipped H3 evaluations.

After both single-chunk pairs pass, test the intended production workflow.
H3 Continuum continuation chunks use a latest-exact hold for every SDE forecast;
single-chunk acceptance does not establish multi-chunk quality. Acceptance of
`res_2m` and `res_3s` also does not establish quality for every other named wrapper.
