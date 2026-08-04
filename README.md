# ComfyUI Spectrum MiniMax H3

Spectrum-style spectral feature forecasting for ComfyUI's native MiniMax H3 audio-video model.

This custom node reduces expensive H3 transformer evaluations during sampling. It fits a Chebyshev ridge model to actual post-transformer hidden features and forecasts those features on selected future solver steps. The current-step native MiniMax H3 output heads, video reconstruction, audio reconstruction, sigma mapping, and return structure still execute on every step.

This repository is independent from [ComfyUI-Spectrum-Proper](https://github.com/xmarre/ComfyUI-Spectrum-Proper), which remains a dedicated FLUX implementation.

## Supported native path

The integration targets `comfy.ldm.minimax.model.MiniMaxH3Model` in native ComfyUI. It requires the MiniMax H3 and packed-latent sampler APIs present at ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff` from August 3, 2026, including the `latent_shapes` argument on `outer_sample`. Older ComfyUI revisions are unsupported. Development and native-equivalence tests are pinned to that commit. Later revisions are unverified; required H3 attributes are checked when the node is applied, and replacement output shape is checked on actual steps so incompatible native changes fail with an explicit contract error.

The forecast target is the packed hidden feature immediately after the final H3 transformer block and before `FinalLayer`, ordered as:

```text
[target audio rows | target video rows]
```

Text rows and all keyframe/reference-only rows are excluded from history. Actual steps stay on the native `_forward` implementation. A call-local final-block replacement observes its returned hidden state while preserving an existing replacement patch. Forecast steps skip every transformer block, RoPE construction, conditioning projections, reference embedding, and per-block prefetch, then run the native final layer with freshly computed audio and video timestep embeddings.

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git
```

Restart ComfyUI. The node appears under `sampling/spectrum` as **Spectrum Apply MiniMax H3**.

The node adds no third-party Python dependency. It uses PyTorch and ComfyUI modules already present in a normal ComfyUI installation.

## Workflow placement

Recommended order:

```text
MiniMax H3 model loader
-> LoRA and other model patches
-> MiniMax H3 Sigma Shift
-> Spectrum Apply MiniMax H3
-> guider and sampler
```

The node accepts and returns `MODEL`. Disabled mode returns the original model object unchanged. Enabled mode clones the model and rejects anything other than the exact native MiniMax H3 model type with a precise error.

## Parameters

| Parameter | Conservative default | Meaning |
|---|---:|---|
| `enabled` | `true` | Enables the clone-local Spectrum runtime. |
| `blend_weight` | `0.50` | Spectral share of the prediction. The remainder is a two-point local linear forecast. |
| `degree` | `4` | Maximum Chebyshev polynomial degree. At least `degree + 1` actual points are required. |
| `ridge_lambda` | `0.10` | Ridge regularization applied to the small Gram matrix. |
| `window_size` | `2.0` | Initial adaptive interval. |
| `flex_window` | `0.75` | Amount added to the interval after a scheduled post-warmup actual step. |
| `warmup_steps` | `5` | Initial solver steps forced to native transformer evaluation. |
| `tail_actual_steps` | `1` | Requested final native tail. Deterministic RES enforces a sampler-safe minimum of `3`. |
| `max_history` | `8` | Maximum model-dtype actual feature snapshots retained. |
| `debug` | `false` | Enables concise run, step, topology, fallback, sanitization, chunk, and teardown logs. |
| `history_storage` | `system_ram` | Stores history in `system_ram`, or in `vram` to avoid transfer overhead when sufficient accelerator memory is free. |

Every value is validated. `max_history` must be at least `degree + 1`.

### Conservative preset

```text
blend_weight = 0.50
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 0.75
warmup_steps = 5
tail_actual_steps = 1
max_history = 8
history_storage = system_ram
```

### Provisional aggressive preset

```text
blend_weight = 0.75
degree = 4
ridge_lambda = 0.10
window_size = 2.0
flex_window = 3.0
warmup_steps = 5
tail_actual_steps = 1
max_history = 8
history_storage = system_ram
```

Both presets remain provisional pending broader prompt, sampler, and quality coverage.

## Adaptive schedule

Warmup and final-tail steps are actual. After warmup, with current interval `W`, a step is actual when:

```text
(consecutive_forecasts + 1) mod floor(W) == 0
```

After a successfully completed scheduled actual step, `W` increases by `flex_window`. A fallback actual step does not increase it. Forecasting also waits until at least `max(2, degree + 1)` actual history points exist.

For a 20-step run with the conservative settings, the sampler-aware scheduler currently produces:

| Sampler | Actual H3 solver steps | Forecasted solver steps | Forecast indices | Transformer-step reduction |
|---|---:|---:|---|---:|
| Euler | 13 | 7 | `5, 7, 9, 11, 13, 15, 17` | 35% |
| RES multistep / CFG++ | 14 | 6 | `5, 7, 9, 11, 13, 15` | 30% |

These counts are solver-step counts. CFG can execute separate conditional and unconditional H3 transformer calls on each actual solver step. End-to-end wall-clock speedup depends on output-head cost, CPU transfers, model offload, references, CFG branching, latent size, and hardware.

## Supported samplers

Forecasting is currently allowlisted for:

- Euler (`sample_euler`)
- RES multistep (`sample_res_multistep`)
- RES multistep CFG++ (`sample_res_multistep_cfg_pp`)

The reviewed implementations make one `predict_noise` call per solver iteration. Euler feeds each approximate denoised result into the latent used by the next evaluation, so it requires one completed actual H3 evaluation after every forecast. RES multistep stores each current denoised result as `old_denoised` for the following second-order update. The actual evaluation immediately after a forecast still consumes forecast-derived history, then replaces `old_denoised` with its native result before another forecast is allowed. This prevents any RES update from combining two forecasted denoised results. RES also keeps its final three solver steps native; this tail floor applies even when a saved workflow supplies a smaller `tail_actual_steps` value. Ancestral samplers execute native MiniMax H3 because injected noise invalidates the smooth deterministic feature trajectory used by the forecaster. Debug mode logs the exact fallback, tail, or post-forecast refresh reason. Multi-GPU parallel sampling also remains native because distributed forecast-row transactions are not yet validated.

## Memory design

The implementation uses the history-weight form of Chebyshev ridge regression:

```text
w(t*) = phi(t*) (Phi^T Phi + lambda I)^-1 Phi^T
H_hat(t*) = w(t*) H
```

Spectral and linear history weights are combined before reading feature history. Persistent large tensors are limited to `max_history` detached model-dtype snapshots in the selected `history_storage`. Design, Gram, Cholesky, and history-weight tensors remain small FP32 CPU matrices. Prediction streams one bounded slice from one history snapshot at a time, accumulates that slice in FP32 on the prediction device, then writes model dtype. There is no persistent full-feature FP32 regression right-hand side or coefficient tensor.

History storage cost is approximately:

```text
branch_count * max_history * (target_audio_rows + target_video_rows)
* hidden_width * model_dtype_bytes
```

At the native 1344x768, 124-frame example, the reviewed layout has about 37,710 target rows. With hidden width 5,376 and BF16/FP16 history, one snapshot is roughly 387 MiB per branch. Eight conditional/unconditional snapshots can therefore approach 6.1 GiB in the selected storage. Reference tokens do not enter the cached target, while longer duration and larger target geometry increase the cost. Lower `max_history` is valid only while it remains at least `degree + 1`.

With `history_storage=system_ram`, forecast VRAM includes one model-dtype target feature for the current model call plus a bounded FP32 accumulation chunk. Actual steps copy each new snapshot to CPU, and forecasts stream the retained snapshots back to the prediction device. These transfers can reduce the theoretical speedup.

With `history_storage=vram`, the same model-dtype history remains on the device that produced it. This avoids the device-to-host archive and repeated host-to-device forecast reads. The captured target is cloned into compact owned storage; retaining its native view would keep the complete final-block hidden tensor alive. The mode needs the full history allocation plus transient headroom for the current snapshot, prediction result, FP32 chunk, allocator fragmentation, and native H3 execution. At the native example above, use it only with materially more than 6.1 GiB of VRAM free at the native generation peak. An explicit VRAM selection can raise an out-of-memory error when that headroom is unavailable.

Debug run summaries report the selected storage and resolved history device together with archive, history-update, and forecast-prediction wall time. CPU archiving can synchronize preceding CUDA work, while GPU cloning can be asynchronously enqueued, so the component counters diagnose the runtime path rather than serving as isolated kernel benchmarks. End-to-end wall time and peak allocated VRAM are the authoritative comparison.

## Fallback and transaction behavior

The native path is used when forecasting is unsupported or cannot be proven safe. Reasons include sampler incompatibility, missing branch labels, topology changes, audio/video target count changes, hidden-width changes, duplicate or reordered-unmappable labels, nonfinite schedules, prediction shape failures, and unusable forecasts.

Split conditional calls are assigned by ComfyUI's `cond_or_uncond` and UUID labels. Row allocation is transactional. If correspondence becomes incomplete after an earlier subcall forecast, the entire `predict_noise` attempt is discarded and rerun as an actual step. Exceptions abort the active step without advancing scheduler state, preserve the original traceback, and outer-run teardown releases all history.

Model wrappers are registered on the cloned `ModelPatcher`. A clone callback creates a new runtime for every downstream clone. The shared inner H3 module stores no Spectrum state and is never monkey-patched.

## Validation status

Automated tests cover:

- direct coefficient, history-weight, chunked, blended, and row-subset equivalence;
- FP32, FP16, and BF16 features;
- history eviction, repeated coordinates, zero ridge, bounded Cholesky jitter, and factorization reuse;
- absence of persistent full-feature FP32 RHS/coefficient storage;
- warmup, final tail, adaptive counts, fallback accounting, abort rollback, and teardown;
- split, reordered, missing, and duplicate branch labels;
- target audio/video segment ordering and sanitization;
- model detection and clone runtime isolation;
- exact native versus wrapped forced-actual video/audio output on a deterministic tiny native H3 fixture;
- proof that a forecast fixture invokes zero H3 transformer blocks.

No full MiniMax H3 checkpoint is available in the automated environment. Real text-to-video/audio generation, reference modes, long-duration memory behavior, wall-clock speedup, VRAM/RSS peaks, decoded video metrics, audio metrics, and audiovisual synchronization remain unverified by the automated suite. No claim of lossless quality is made.

## Tests

Forecaster smoke test in an environment that already has PyTorch:

```bash
python tests/smoke_forecaster.py
```

Full suite against a current ComfyUI checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
PYTHONPATH=/path/to/ComfyUI \
python -m pytest -q
```

## Repository layout

```text
ComfyUI-Spectrum-MiniMax-H3/
|-- __init__.py
|-- nodes.py
|-- pyproject.toml
|-- LICENSE
|-- README.md
|-- IMPLEMENTATION_NOTES.md
|-- comfyui_spectrum_h3/
|   |-- __init__.py
|   |-- config.py
|   |-- forecast.py
|   |-- nodes.py
|   |-- runtime.py
|   |-- sampling.py
|   `-- minimax_h3.py
`-- tests/
```

## Credits

- Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon for [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623) and the [official Spectrum implementation](https://github.com/hanjq17/Spectrum).
- The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) maintainers for native MiniMax H3, model patching, sampler wrappers, packed latent support, and model-management infrastructure.

## License

GPL-3.0-or-later. The implementation in this repository is standalone. Spectrum's published mathematics and MIT-licensed official implementation were reviewed as primary references; no source file from the official implementation is vendored.
