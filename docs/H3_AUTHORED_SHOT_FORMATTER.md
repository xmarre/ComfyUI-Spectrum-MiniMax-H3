# H3 authored shot formatter — independent proposal

This **deterministic formatter** accepts user-authored shots in T2VA, I2VA,
FL2VA and L2VA modes. It does **not** generate shot plans, detect shots,
analyze images or improve semantic adherence by itself. No extra inference is
performed. `ref2va` has a distinct reference-analysis contract and remains
out of scope.

## Duration convention

The effective AV duration is `frame_count / 24`, matching native ComfyUI H3
(`temporal_shape`), **not** the presentation time of the last frame
(`(frame_count - 1) / 24`). The FL2VA/L2VA alignment header uses effective
AV duration formatted to two decimals. The diagnostics separately show both:

| Actual snapped frames | Effective AV duration | Last-frame PTS | Alignment header |
|---|---|---|---|
| 124 | 5.166667 s | 5.125 s | 5.17-second mark |
| 192 | 8.000000 s | 7.958333 s | 8.00-second mark |

The user must supply the **actual** snapped frame count from the native H3
workflow, not an un-snapped requested length. This node does not read connected
first/last images, native latent tensors, or effective frame counts. Its `OK`
means **TEXT/SHOT SYNTAX ONLY**, never verified reference-image identity or
physical image alignment.

## Soundscape

The default `overall_soundscape` is empty, **not `N/A`**.
When omitted, the formatter emits a visible missing-description placeholder
and a warning. `complete_silence=True` explicitly emits
`overall_soundscape: N/A`; conflicting sound and silence instructions are
reported. `non_diegetic_music: N/A` remains the valid no-score default.

## Duration diagnostics and experimental control

Native H3 frames follow the 17k+5 grid at 24 fps. Standard trained duration
is approximately 124–362 frames (about 5–15 seconds). A grid-valid 22-frame
request is still outside the trained range. The node warns, and strict mode
requires explicit `allow_experimental_duration=True` for such use.
This does not imply that untested durations are supported or quality-verified.

Source references:
- Native code: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
- MiniMax prompt guide: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md

Use `python -m pytest -q tests/test_prompt_tools.py` for syntactic regression
checks. There are **no measured prompt-adherence gains** in this proposal.
