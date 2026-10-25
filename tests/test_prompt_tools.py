from __future__ import annotations

import pytest

from comfyui_spectrum_h3.prompt_tools import SpectrumH3PromptComposer, compose_h3_prompt


def test_plain_t2va_preserves_words_and_audio_fields():
    prompt, report = compose_h3_prompt(
        "A slim goblin moves left without turning.",
        overall_soundscape="Quiet footsteps.",
        non_diegetic_music="N/A",
    )
    assert prompt == (
        "integrated_multimodal_description: [Shot 1] A slim goblin moves left without turning.\n\n"
        "overall_soundscape: Quiet footsteps.\n\n"
        "non_diegetic_music: N/A"
    )
    assert report.startswith("OK:")


@pytest.mark.parametrize("mode,expected", [
    ("i2va", "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."),
    ("fl2va", "Picture 2 (from Shot 2) aligns with the 5.12-second mark"),
    ("l2va", "<Picture 1> (from [Shot 2]) aligns with the 5.12-second mark"),
])
def test_alignment_and_last_shot(mode, expected):
    description = "[Shot 1] A man walks. [Shot 2] At 00:02.000, the camera cuts to a doorway."
    prompt, report = compose_h3_prompt(description, mode=mode)
    assert expected in prompt
    assert description in prompt
    assert "2 shot(s)" in report


def test_out_of_order_cuts_strict_and_warning():
    text = "[Shot 1] Start. [Shot 2] At 00:04.000, cut. [Shot 3] At 00:03.000, cut."
    prompt, report = compose_h3_prompt(text)
    assert text in prompt
    assert "WARN:" in report
    with pytest.raises(ValueError, match="cut time"):
        compose_h3_prompt(text, strict_validation=True)


def test_missing_cut_is_flagged():
    _, report = compose_h3_prompt("[Shot 1] Beginning. [Shot 2] Another scene.")
    assert "needs 'At MM:SS.SSS" in report


def test_bad_frame_count_not_silently_snapped():
    _, report = compose_h3_prompt("The camera pans.", frame_count=125)
    assert "not 17k+5" in report
    with pytest.raises(ValueError, match="effective"):
        compose_h3_prompt("The camera pans.", frame_count=125, strict_validation=True)


def test_invalid_input_rejected_and_no_duplicate_fields():
    for description in ("", "overall_soundscape: extra", "integrated_multimodal_description: test"):
        with pytest.raises(ValueError):
            compose_h3_prompt(description)
    with pytest.raises(ValueError, match="Unsupported"):
        compose_h3_prompt("Scene.", mode="ref2va")
    with pytest.raises(ValueError, match="frame_count"):
        compose_h3_prompt("Scene.", frame_count=0)


def test_node_metadata_and_invocation():
    node = SpectrumH3PromptComposer()
    assert node.RETURN_TYPES == ("STRING", "STRING")
    assert node.INPUT_TYPES()["required"]["mode"][0] == ["t2va", "i2va", "fl2va", "l2va"]
    prompt, report = node.compose("A bird glides.", "t2va", 124)
    assert "[Shot 1] A bird glides." in prompt
    assert report.startswith("OK:")
