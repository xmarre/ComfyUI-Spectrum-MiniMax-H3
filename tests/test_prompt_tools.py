from __future__ import annotations

import pytest

from comfyui_spectrum_h3.prompt_tools import H3AuthoredShotFormatter, compose_h3_prompt


def test_plain_t2va_preserves_authored_words_and_audio_fields():
    prompt, report = compose_h3_prompt(
        "A slim goblin moves left without turning.",
        overall_soundscape="Quiet footsteps.",
    )
    assert prompt == (
        "integrated_multimodal_description: [Shot 1] A slim goblin moves left without turning.\n\n"
        "overall_soundscape: Quiet footsteps.\n\n"
        "non_diegetic_music: N/A"
    )
    assert report.startswith("OK: TEXT/SHOT SYNTAX ONLY")
    assert "NOT inspected" in report


@pytest.mark.parametrize("mode,frames,expected", [
    ("i2va", 124, "For the target video, at 0.00 seconds"),
    ("fl2va", 124, "Picture 2 (from Shot 2) aligns with the 5.17-second mark"),
    ("l2va", 124, "<Picture 1> (from [Shot 2]) aligns with the 5.17-second mark"),
    ("fl2va", 192, "Picture 2 (from Shot 2) aligns with the 8.00-second mark"),
    ("l2va", 192, "<Picture 1> (from [Shot 2]) aligns with the 8.00-second mark"),
])
def test_native_effective_av_duration_and_alignment_header(mode, frames, expected):
    description = "[Shot 1] A man walks. [Shot 2] At 00:02.000, cut to a doorway."
    prompt, report = compose_h3_prompt(
        description, mode=mode, frame_count=frames,
        overall_soundscape="Footsteps echo softly."
    )
    assert expected in prompt
    assert description in prompt
    assert "2 shot(s)" in report
    if frames == 124:
        assert "last-frame PTS 5.125s" in report
    if frames == 192:
        assert "last-frame PTS 7.958s" in report


def test_missing_soundscape_never_silently_becomes_silence():
    prompt, report = compose_h3_prompt("A bird glides.")
    assert "overall_soundscape: [MISSING:" in prompt
    assert "WARN:" in report
    assert "complete_silence" in report
    with pytest.raises(ValueError, match="overall_soundscape"):
        compose_h3_prompt("A bird glides.", strict_validation=True)


def test_explicit_silence_required_for_na():
    prompt, report = compose_h3_prompt("An empty stage.", complete_silence=True)
    assert "overall_soundscape: N/A" in prompt
    assert report.startswith("OK:")
    prompt, report = compose_h3_prompt("An empty stage.", overall_soundscape="N/A")
    assert "overall_soundscape: N/A" not in prompt
    assert report.startswith("WARN:")
    with pytest.raises(ValueError, match="conflicts"):
        compose_h3_prompt(
            "An empty stage.", overall_soundscape="Rain.", complete_silence=True,
            strict_validation=True
        )


def test_duration_range_warns_even_for_17k_plus_five_count_22():
    prompt, report = compose_h3_prompt(
        "A short movement.", frame_count=22, overall_soundscape="Light footsteps."
    )
    assert "[Shot 1]" in prompt
    assert "outside the usual 124" in report
    with pytest.raises(ValueError, match="experimental"):
        compose_h3_prompt(
            "A short movement.", frame_count=22, overall_soundscape="Light footsteps.",
            strict_validation=True
        )
    _, report = compose_h3_prompt(
        "A short movement.", frame_count=22, overall_soundscape="Light footsteps.",
        strict_validation=True, allow_experimental_duration=True
    )
    assert "EXPERIMENTAL" in report


def test_out_of_order_and_missing_cut_report_syntax_errors():
    text = "[Shot 1] Start. [Shot 2] At 00:04.000, cut. [Shot 3] At 00:03.000, cut."
    prompt, report = compose_h3_prompt(text, overall_soundscape="Rain outside.")
    assert text in prompt
    assert "cut time" in report
    with pytest.raises(ValueError, match="cut time"):
        compose_h3_prompt(text, overall_soundscape="Rain outside.", strict_validation=True)
    _, report = compose_h3_prompt(
        "[Shot 1] Beginning. [Shot 2] Another scene.",
        overall_soundscape="Light wind."
    )
    assert "needs 'At MM:SS.SSS" in report


def test_frame_grid_mismatch_not_silently_snapped():
    _, report = compose_h3_prompt("A pan.", frame_count=125, overall_soundscape="Traffic.")
    assert "not 17k+5" in report
    with pytest.raises(ValueError, match="snapped"):
        compose_h3_prompt(
            "A pan.", frame_count=125, overall_soundscape="Traffic.",
            strict_validation=True
        )


def test_invalid_inputs_and_metadata():
    for description in ("", "overall_soundscape: extra", "integrated_multimodal_description: test"):
        with pytest.raises(ValueError):
            compose_h3_prompt(description)
    with pytest.raises(ValueError, match="Unsupported"):
        compose_h3_prompt("Scene.", mode="ref2va")
    with pytest.raises(ValueError, match="frame_count"):
        compose_h3_prompt("Scene.", frame_count=0)
    node = H3AuthoredShotFormatter()
    assert node.RETURN_TYPES == ("STRING", "STRING")
    assert node.INPUT_TYPES()["required"]["mode"][0] == ["t2va", "i2va", "fl2va", "l2va"]
    prompt, report = node.compose("A bird glides.", "t2va", 124, complete_silence=True)
    assert "[Shot 1] A bird glides." in prompt
    assert report.startswith("OK:")
