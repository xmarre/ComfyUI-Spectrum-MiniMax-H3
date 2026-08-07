from __future__ import annotations

import inspect

import pytest

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("degree", 4.5),
        ("warmup_steps", 4.5),
        ("tail_actual_steps", 4.5),
        ("max_history", 8.5),
    ],
)
def test_count_fields_reject_fractional_values(field, value):
    with pytest.raises(ValueError, match="integer"):
        SpectrumH3Config(**{field: value}).validate()


def test_history_storage_rejects_unknown_locations():
    with pytest.raises(ValueError, match="history_storage"):
        SpectrumH3Config(history_storage="automatic").validate()


def test_bootstrap_first_forecast_defaults_to_false():
    assert SpectrumH3Config().bootstrap_first_forecast is False


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_bootstrap_first_forecast_rejects_non_boolean_values(value):
    with pytest.raises(TypeError, match="bootstrap_first_forecast must be a boolean"):
        SpectrumH3Config(bootstrap_first_forecast=value).validate()


def test_bootstrap_first_forecast_requires_degree_one():
    with pytest.raises(ValueError, match="bootstrap_first_forecast requires degree == 1"):
        SpectrumH3Config(bootstrap_first_forecast=True, degree=2).validate()


def test_bootstrap_first_forecast_requires_at_most_one_warmup_step():
    with pytest.raises(ValueError, match="bootstrap_first_forecast requires warmup_steps <= 1"):
        SpectrumH3Config(
            bootstrap_first_forecast=True,
            degree=1,
            warmup_steps=2,
        ).validate()


def test_bootstrap_first_forecast_accepts_degree_one_and_one_warmup_step():
    config = SpectrumH3Config(
        bootstrap_first_forecast=True,
        degree=1,
        warmup_steps=1,
    ).validate()

    assert config.bootstrap_first_forecast is True


def test_node_exposes_bootstrap_as_an_optional_boolean():
    optional = SpectrumApplyMiniMaxH3.INPUT_TYPES()["optional"]
    apply_parameters = inspect.signature(SpectrumApplyMiniMaxH3.apply).parameters

    assert optional["bootstrap_first_forecast"] == ("BOOLEAN", {"default": False})
    assert apply_parameters["bootstrap_first_forecast"].default is False
