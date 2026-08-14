from comfyui_spectrum_h3 import NODE_CLASS_MAPPINGS
from comfyui_spectrum_h3.objective_media_nodes import _parse_generation_seed


def test_recommended_sequential_capture_consumes_linked_int_seed():
    capture_cls = NODE_CLASS_MAPPINGS["SpectrumH3ObjectiveSequentialCapture"]
    generation_seed = capture_cls.INPUT_TYPES()["required"]["generation_seed"]

    assert generation_seed[0] == "INT"
    options = generation_seed[1]
    assert options["forceInput"] is True
    assert options["min"] == 0
    assert options["max"] == 0xFFFFFFFFFFFFFFFF
    assert "control_after_generate" not in options
    assert "default" not in options


def test_linked_int_seed_is_accepted_without_string_copying():
    assert _parse_generation_seed(123456789) == 123456789
