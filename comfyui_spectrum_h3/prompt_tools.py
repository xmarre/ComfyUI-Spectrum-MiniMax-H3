"""Deterministic H3 base-workflow prompt composition; no model calls."""
from __future__ import annotations

import re

_MODES = ("t2va", "i2va", "fl2va", "l2va")
_SHOTS = re.compile(r"\[Shot\s+(\d+)\]", re.IGNORECASE)
_CUT = re.compile(r"^\s*At\s+(\d{2}):(\d{2})\.(\d{3})\s*,", re.IGNORECASE)
_FIELDS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")


def _check_shots(description: str, duration: float) -> tuple[int, list[str]]:
    matches = list(_SHOTS.finditer(description))
    if not matches:
        return 1, []
    issues: list[str] = []
    indices = [int(m.group(1)) for m in matches]
    if indices != list(range(1, len(indices) + 1)):
        issues.append("Shots must be numbered consecutively starting with [Shot 1].")
    previous_time = 0.0
    for i, match in enumerate(matches):
        section = description[match.end():matches[i + 1].start() if i + 1 < len(matches) else None]
        cut = _CUT.match(section)
        if i == 0:
            if cut is not None:
                issues.append("[Shot 1] must not have a cut timestamp.")
            continue
        if cut is None:
            issues.append(f"[Shot {indices[i]}] needs 'At MM:SS.SSS, ...' immediately after the label.")
            continue
        minute, second, ms = map(int, cut.groups())
        cut_time = minute * 60 + second + ms / 1000
        if second >= 60 or cut_time <= previous_time or cut_time >= duration:
            issues.append(f"[Shot {indices[i]}] cut time must increase and remain inside the clip.")
        previous_time = cut_time
    return indices[-1], issues


def compose_h3_prompt(
    scene_and_shots: str,
    mode: str = "t2va",
    frame_count: int = 124,
    overall_soundscape: str = "N/A",
    non_diegetic_music: str = "N/A",
    strict_validation: bool = False,
) -> tuple[str, str]:
    """Return structured prompt and diagnostics; supplied frames must be effective frames."""
    if mode not in _MODES:
        raise ValueError(f"Unsupported H3 base workflow mode: {mode!r}")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 5:
        raise ValueError("frame_count must be an integer >= 5")
    if not isinstance(scene_and_shots, str) or not scene_and_shots.strip():
        raise ValueError("Scene and shots cannot be empty")
    if not isinstance(strict_validation, bool):
        raise TypeError("strict_validation must be a boolean")
    description = scene_and_shots.strip()
    if any(label in description.lower() for label in _FIELDS):
        raise ValueError("Provide just the scene and shots, not an already structured H3 prompt")
    duration = (frame_count - 1) / 24.0
    shot_number, issues = _check_shots(description, duration)
    if not _SHOTS.search(description):
        description = "[Shot 1] " + description
    if (frame_count - 5) % 17:
        issues.append(
            "Frame count is not 17k+5; supply the effective output length "
            "from the H3 workflow so final-frame alignment is accurate."
        )
    if strict_validation and issues:
        raise ValueError("Invalid H3 shot plan: " + " ".join(issues))

    header = ""
    end = f"{duration:.2f}"
    if mode == "i2va":
        header = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    elif mode == "fl2va":
        header = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {shot_number}) aligns with the {end}-second mark of the target video."
        )
    elif mode == "l2va":
        header = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {shot_number}]) aligns with the "
            f"{end}-second mark of the target video."
        )

    sound = (overall_soundscape or "").strip() or "N/A"
    music = (non_diegetic_music or "").strip() or "N/A"
    body = (
        f"integrated_multimodal_description: {description}\n\n"
        f"overall_soundscape: {sound}\n\n"
        f"non_diegetic_music: {music}"
    )
    prompt = f"{header}\n\n{body}" if header else body
    diagnostics = (
        f"{'OK' if not issues else 'WARN'}: H3 {mode.upper()} | {frame_count} effective frames "
        f"| {duration:.3f}s | {shot_number} shot(s)"
    )
    if issues:
        diagnostics += "\n" + "\n".join(f"- {issue}" for issue in issues)
    return prompt, diagnostics


class SpectrumH3PromptComposer:
    """Optional H3 prompt authoring helper."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_and_shots": (
                    "STRING",
                    {"multiline": True, "default": "Cinematic wide shot of a forest at dawn. The camera slowly tracks forward."},
                ),
                "mode": (list(_MODES), {"default": "t2va"}),
                "effective_frame_count": (
                    "INT",
                    {"default": 124, "min": 5, "max": 2048, "step": 1},
                ),
            },
            "optional": {
                "overall_soundscape": ("STRING", {"multiline": True, "default": "N/A"}),
                "non_diegetic_music": ("STRING", {"multiline": True, "default": "N/A"}),
                "strict_validation": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "validation_report")
    FUNCTION = "compose"
    CATEGORY = "sampling/spectrum"

    def compose(
        self,
        scene_and_shots,
        mode,
        effective_frame_count,
        overall_soundscape="N/A",
        non_diegetic_music="N/A",
        strict_validation=False,
    ):
        return compose_h3_prompt(
            scene_and_shots=scene_and_shots,
            mode=mode,
            frame_count=effective_frame_count,
            overall_soundscape=overall_soundscape,
            non_diegetic_music=non_diegetic_music,
            strict_validation=strict_validation,
        )
