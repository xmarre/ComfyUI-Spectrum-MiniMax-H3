"""Deterministic formatter for authored H3 shot instructions, not semantic planning."""
from __future__ import annotations

import re

_MODES = ("t2va", "i2va", "fl2va", "l2va")
_SHOTS = re.compile(r"\[Shot\s+(\d+)\]", re.IGNORECASE)
_CUT = re.compile(r"^\s*At\s+(\d{2}):(\d{2})\.(\d{3})\s*,", re.IGNORECASE)
_FIELDS = ("integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:")
_FPS = 24
_TRAINED_MIN_FRAMES = 124
_TRAINED_MAX_FRAMES = 362


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
    overall_soundscape: str = "",
    non_diegetic_music: str = "N/A",
    strict_validation: bool = False,
    complete_silence: bool = False,
    allow_experimental_duration: bool = False,
) -> tuple[str, str]:
    """Format *authored* shots using native H3 AV duration = frame_count / 24.

    Frame_count is user-supplied: this function cannot inspect connected images,
    conditioning, or the actual native snapped latent/frame count.
    """
    if mode not in _MODES:
        raise ValueError(f"Unsupported H3 base workflow mode: {mode!r}")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 5:
        raise ValueError("frame_count must be an integer >= 5")
    if not isinstance(scene_and_shots, str) or not scene_and_shots.strip():
        raise ValueError("Scene and shots cannot be empty")
    if not isinstance(strict_validation, bool):
        raise TypeError("strict_validation must be a boolean")
    if not isinstance(complete_silence, bool) or not isinstance(allow_experimental_duration, bool):
        raise TypeError("silence and experimental-duration flags must be booleans")
    description = scene_and_shots.strip()
    if any(label in description.lower() for label in _FIELDS):
        raise ValueError("Provide just the scene and shots, not an already structured H3 prompt")

    duration = frame_count / _FPS  # Native ComfyUI effective AV duration.
    last_frame_pts = (frame_count - 1) / _FPS  # Presentation timestamp, not AV duration.
    shot_number, issues = _check_shots(description, duration)
    notices: list[str] = []
    if not _SHOTS.search(description):
        description = "[Shot 1] " + description
    if (frame_count - 5) % 17:
        issues.append(
            "Frame count is not 17k+5; supply the actual snapped output frame count."
        )
    if not _TRAINED_MIN_FRAMES <= frame_count <= _TRAINED_MAX_FRAMES:
        if allow_experimental_duration:
            notices.append(
                "EXPERIMENTAL frame count outside the usual 124–362 trained range; "
                "user acknowledged an untested duration."
            )
        else:
            issues.append(
                "Frame count outside the usual 124–362 trained range; "
                "set allow_experimental_duration=True for an intentional experiment."
            )

    sound = (overall_soundscape or "").strip()
    if complete_silence:
        if sound and sound.upper() != "N/A":
            issues.append(
                "Complete silence conflicts with the supplied soundscape description."
            )
        sound = "N/A"
    elif not sound or sound.upper() == "N/A":
        issues.append(
            "overall_soundscape is missing. Describe the ambient sounds, or "
            "enable complete_silence for an explicit silent video (N/A)."
        )
        sound = "[MISSING: describe the ambient and physical sounds]"
    music = (non_diegetic_music or "").strip() or "N/A"

    if strict_validation and issues:
        raise ValueError("Invalid H3 authored shot formatting: " + " ".join(issues))

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

    body = (
        f"integrated_multimodal_description: {description}\n\n"
        f"overall_soundscape: {sound}\n\n"
        f"non_diegetic_music: {music}"
    )
    result = f"{header}\n\n{body}" if header else body
    prefix = "WARN" if issues or notices else "OK"
    diagnostics = (
        f"{prefix}: TEXT/SHOT SYNTAX ONLY — H3 {mode.upper()} | {frame_count} authored effective frames "
        f"| AV duration {duration:.3f}s | last-frame PTS {last_frame_pts:.3f}s "
        f"| {shot_number} shot(s). Connected images, latent snapping and alignment were NOT inspected."
    )
    if issues or notices:
        diagnostics += "\n" + "\n".join(f"- {message}" for message in issues + notices)
    return result, diagnostics


class H3AuthoredShotFormatter:
    """Optional deterministic formatting of user-authored H3 base-workflow shots."""

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
                    {"default": 124, "min": 5, "max": 3600, "step": 1},
                ),
            },
            "optional": {
                "overall_soundscape": ("STRING", {"multiline": True, "default": ""}),
                "non_diegetic_music": ("STRING", {"multiline": True, "default": "N/A"}),
                "complete_silence": ("BOOLEAN", {"default": False}),
                "strict_validation": ("BOOLEAN", {"default": False}),
                "allow_experimental_duration": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "syntax_report")
    FUNCTION = "compose"
    CATEGORY = "prompt/H3"

    def compose(
        self,
        scene_and_shots,
        mode,
        effective_frame_count,
        overall_soundscape="",
        non_diegetic_music="N/A",
        complete_silence=False,
        strict_validation=False,
        allow_experimental_duration=False,
    ):
        return compose_h3_prompt(
            scene_and_shots=scene_and_shots,
            mode=mode,
            frame_count=effective_frame_count,
            overall_soundscape=overall_soundscape,
            non_diegetic_music=non_diegetic_music,
            complete_silence=complete_silence,
            strict_validation=strict_validation,
            allow_experimental_duration=allow_experimental_duration,
        )
