# Fork experiments: prompt adherence and forecast throughput

Optional experiments; no measured H3 quality or speed gain is claimed yet.
The upstream Spectrum scheduler, sampler equations and default forecast-head
chunk-size are unchanged.

## Spectrum H3 Prompt Composer

Place the STRING output before the existing H3 prompt encoder. No LLM or
additional transformer inference is required.

- Supports native base workflows t2va, i2va, fl2va and l2va.
- Uses the official integrated_multimodal_description, overall_soundscape and
  non_diegetic_music fields and prescribed image-alignment headers.
- Plain descriptions become one [Shot 1]; multi-shot descriptions must use
  [Shot 2] At 00:03.500, ... and so forth.
- Authored words, scene, dialogue, motion and camera instructions are preserved.
  This tool does not invent character attributes or rewrite semantics.
- strict_validation rejects nonconsecutive shot labels, missing/out-of-order
  cut times and frame counts inconsistent with the common 17k+5 convention.
  Otherwise warnings are returned on the validation_report STRING output.
- effective_frame_count must match the **actual** generated frame count, after
  any upstream snapping. Final-frame time = (frame_count-1)/24 seconds.
- ref2va uses a different reference-analysis contract and is not supported yet.

Example scene: [Shot 1] A goblin faces a locked vault. The camera tracks right.
[Shot 2] At 00:03.500, the camera cuts to a close-up of the lock opening.

Source: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md

## Optional streamed-head performance tuning

The default streamed FinalLayer work budget remains 16 MiB. Before starting
ComfyUI, optionally set an environment variable, for example on Windows:

    set SPECTRUM_H3_FORECAST_HEAD_CHUNK_MIB=64

Accepted sizes: 16, 32, 64, 128 MiB. Invalid values fall back to 16 MiB with
a warning. Restart ComfyUI after changing the variable. This modifies only
the reviewed CPU-history streamed FinalLayer path, not whole-model sampling,
the unknown-FinalLayer monolithic fallback, or GPU-resident history.
Larger slabs may reduce overhead, or cause OOM/slower inference. Do not assume
larger is faster.

## Quality and speed acceptance procedure

1. Fix model, prompt, references, seed, resolution, frame count, sampler,
   attention backend, and Spectrum settings.
2. Compare 16 MiB control with 32/64/128 MiB experimental runs.
3. Measure end-to-end sampler time with debug disabled; benchmark synchronized
   CUDA forecast-head latency separately.
4. Require identical actual/forecast calls, no new fallbacks, acceptable CUDA
   peaks and decoded audiovisual non-regression.
5. Separately compare native H3, unmodified Spectrum and the new structured
   prompt path for action, identity, timing and speech adherence.

CPU tests cover parser/contract behavior. They do not establish real H3 GPU
speed, output parity or prompt-adherence improvements.
