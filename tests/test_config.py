from __future__ import annotations

import inspect
import logging

import pytest

from comfyui_spectrum_h3 import nodes as nodes_module
from comfyui_spectrum_h3.config import AGGRESSIVE_PRESET, SpectrumH3Config
from comfyui_spectrum_h3.nodes import (
    SpectrumApplyMiniMaxH3,
    _effective_bootstrap_first_forecast,
)


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


def test_offline_archive_storage_rejects_unknown_locations():
    with pytest.raises(ValueError, match="offline_archive_storage"):
        SpectrumH3Config(offline_archive_storage="automatic").validate()


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_audio_blend_weight_requires_a_finite_unit_interval(value):
    with pytest.raises(ValueError, match="audio_blend_weight"):
        SpectrumH3Config(audio_blend_weight=value).validate()


@pytest.mark.parametrize("value", ["", "confidence", "FULL", None])
def test_model_aware_mode_rejects_unknown_values(value):
    with pytest.raises((TypeError, ValueError), match="model_aware_mode"):
        SpectrumH3Config(model_aware_mode=value).validate()


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_model_aware_threshold_requires_a_finite_unit_interval(value):
    with pytest.raises(ValueError, match="model_aware_risk_threshold"):
        SpectrumH3Config(model_aware_risk_threshold=value).validate()


def test_preliminary_scheduler_defaults():
    config = SpectrumH3Config()
    required = SpectrumApplyMiniMaxH3.INPUT_TYPES()["required"]
    optional = SpectrumApplyMiniMaxH3.INPUT_TYPES()["optional"]
    apply_parameters = inspect.signature(SpectrumApplyMiniMaxH3.apply).parameters

    assert config.degree == 1
    assert config.warmup_steps == 1
    assert config.tail_actual_steps == 1
    assert config.bootstrap_first_forecast is True
    assert config.blend_weight == 0.5
    assert config.audio_blend_weight == 0.0
    assert config.offline_smoothing_replay is True
    assert config.offline_archive_storage == "system_ram"
    assert config.model_aware_mode == "off"
    assert config.model_aware_risk_threshold == 0.65
    assert config.generic_correction_mode == "legacy"
    assert config.generic_correction_limiter == "rational"
    assert config.generic_correction_limit == 0.25
    assert required["degree"][1]["default"] == 1
    assert required["warmup_steps"][1]["default"] == 1
    assert required["tail_actual_steps"][1]["default"] == 1
    assert optional["bootstrap_first_forecast"][0] == "BOOLEAN"
    assert optional["bootstrap_first_forecast"][1]["default"] is True
    assert "degree=1" in optional["bootstrap_first_forecast"][1]["tooltip"]
    assert "warmup_steps<=1" in optional["bootstrap_first_forecast"][1]["tooltip"]
    assert "disable bootstrap_first_forecast" in required["degree"][1]["tooltip"]
    assert "disable bootstrap_first_forecast" in required["warmup_steps"][1]["tooltip"]
    assert apply_parameters["bootstrap_first_forecast"].default is True
    assert optional["audio_blend_weight"][0] == "FLOAT"
    assert optional["audio_blend_weight"][1]["default"] == 0.0
    assert apply_parameters["audio_blend_weight"].default == 0.0
    assert optional["offline_smoothing_replay"][0] == "BOOLEAN"
    assert optional["offline_smoothing_replay"][1]["default"] is True
    assert apply_parameters["offline_smoothing_replay"].default is True
    assert optional["offline_archive_storage"][0] == ["system_ram", "vram"]
    assert optional["offline_archive_storage"][1]["default"] == "system_ram"
    assert "not capped by max_history" in optional["offline_archive_storage"][1]["tooltip"]
    assert apply_parameters["offline_archive_storage"].default == "system_ram"
    assert optional["model_aware_mode"][0] == [
        "off",
        "schedule",
        "schedule_confidence",
        "full",
    ]
    assert optional["model_aware_mode"][1]["default"] == "off"
    assert apply_parameters["model_aware_mode"].default == "off"
    assert optional["generic_correction_mode"][0] == [
        "legacy",
        "coordinate_rls",
        "coordinate_rls_reliability",
        "regional",
    ]
    assert optional["generic_correction_mode"][1]["default"] == "legacy"
    assert apply_parameters["generic_correction_mode"].default == "legacy"
    assert optional["generic_correction_limiter"][1]["default"] == "rational"
    assert apply_parameters["generic_correction_limiter"].default == "rational"
    assert optional["generic_correction_limit"][1]["default"] == 0.25
    assert apply_parameters["generic_correction_limit"].default == 0.25
    for name in ("anchor_residual_feedback", "selective_rollback_correction"):
        assert getattr(config, name) is False
        assert optional[name][0] == "BOOLEAN"
        assert optional[name][1]["default"] is False
        assert apply_parameters[name].default is False


@pytest.mark.parametrize("value", ["", "rls", "REGIONAL", None])
def test_generic_correction_mode_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="generic_correction_mode"):
        SpectrumH3Config(generic_correction_mode=value).validate()


@pytest.mark.parametrize("value", ["", "clip", "softsign", None])
def test_generic_correction_limiter_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="generic_correction_limiter"):
        SpectrumH3Config(generic_correction_limiter=value).validate()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_generic_correction_limit_requires_positive_finite_unit_range(value):
    with pytest.raises(ValueError, match="generic_correction_limit"):
        SpectrumH3Config(generic_correction_limit=value).validate()


@pytest.mark.parametrize(
    "name",
    (
        "anchor_residual_feedback",
        "selective_rollback_correction",
        "offline_smoothing_replay",
    ),
)
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_trajectory_settings_require_strict_booleans(name, value):
    with pytest.raises(TypeError, match=rf"{name} must be a boolean"):
        SpectrumH3Config(**{name: value})


@pytest.mark.parametrize(
    "enabled",
    (
        ("anchor_residual_feedback", "selective_rollback_correction"),
        ("anchor_residual_feedback", "offline_smoothing_replay"),
        ("selective_rollback_correction", "offline_smoothing_replay"),
        (
            "anchor_residual_feedback",
            "selective_rollback_correction",
            "offline_smoothing_replay",
        ),
    ),
)
def test_trajectory_settings_are_mutually_exclusive_on_construction(enabled):
    values = {name: True for name in enabled}
    with pytest.raises(ValueError) as error:
        SpectrumH3Config(**values)
    for name in enabled:
        assert name in str(error.value)


@pytest.mark.parametrize(
    "name",
    ("anchor_residual_feedback", "selective_rollback_correction"),
)
def test_research_modes_require_explicitly_disabling_default_offline_replay(name):
    with pytest.raises(ValueError) as error:
        SpectrumH3Config(**{name: True})
    assert name in str(error.value)
    assert "offline_smoothing_replay" in str(error.value)

    config = SpectrumH3Config(
        **{name: True},
        offline_smoothing_replay=False,
    ).validate()
    assert getattr(config, name) is True
    assert config.offline_smoothing_replay is False


def test_disabled_config_allows_irrelevant_trajectory_conflicts():
    config = SpectrumH3Config(
        enabled=False,
        anchor_residual_feedback=True,
        selective_rollback_correction=True,
        offline_smoothing_replay=True,
    )
    config.validate()


def test_disabled_node_returns_original_before_trajectory_validation():
    model = object()
    (result,) = SpectrumApplyMiniMaxH3().apply(
        model,
        False,
        0.5,
        1,
        0.1,
        2.0,
        0.75,
        1,
        1,
        8,
        False,
        anchor_residual_feedback=True,
        selective_rollback_correction=True,
    )
    assert result is model


def test_aggressive_preset_explicitly_disables_degree_one_bootstrap():
    assert AGGRESSIVE_PRESET.degree == 4
    assert AGGRESSIVE_PRESET.bootstrap_first_forecast is False
    assert AGGRESSIVE_PRESET.offline_smoothing_replay is True
    AGGRESSIVE_PRESET.validate()


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


@pytest.mark.parametrize(
    ("degree", "warmup_steps"),
    [
        (2, 1),
        (1, 2),
        (2, 2),
    ],
)
def test_node_disables_incompatible_bootstrap_settings(degree, warmup_steps, caplog):
    with caplog.at_level(logging.WARNING, logger="comfyui_spectrum_h3.nodes"):
        effective = _effective_bootstrap_first_forecast(
            requested=True,
            degree=degree,
            warmup_steps=warmup_steps,
        )

    assert effective is False
    assert "Disabling bootstrap_first_forecast" in caplog.text

    assert f"degree={degree}" in caplog.text
    assert f"warmup_steps={warmup_steps}" in caplog.text


def test_node_keeps_compatible_bootstrap_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="comfyui_spectrum_h3.nodes"):
        effective = _effective_bootstrap_first_forecast(
            requested=True,
            degree=1,
            warmup_steps=1,
        )

    assert effective is True
    assert not caplog.records


def test_node_keeps_explicitly_disabled_bootstrap_without_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="comfyui_spectrum_h3.nodes"):
        effective = _effective_bootstrap_first_forecast(
            requested=False,
            degree=4,
            warmup_steps=5,
        )

    assert effective is False
    assert not caplog.records


def test_apply_normalizes_reported_warmup_conflict(monkeypatch, caplog):
    captured = {}

    class FakeModel:
        def clone(self):
            return FakeModel()

    class FakeRuntime:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setattr(nodes_module, "require_native_minimax_h3", lambda model: None)
    monkeypatch.setattr(nodes_module, "SpectrumH3Runtime", FakeRuntime)
    monkeypatch.setattr(nodes_module, "install_sampler_wrappers", lambda model, runtime: None)
    monkeypatch.setattr(nodes_module, "install_h3_wrapper", lambda model: None)

    with caplog.at_level(logging.WARNING, logger="comfyui_spectrum_h3.nodes"):
        (patched,) = SpectrumApplyMiniMaxH3().apply(
            FakeModel(),
            True,
            0.50,
            1,
            0.10,
            2.0,
            0.75,
            2,
            1,
            8,
            False,
            bootstrap_first_forecast=True,
            audio_blend_weight=0.25,
            offline_archive_storage="vram",
        )

    assert isinstance(patched, FakeModel)
    assert captured["config"].degree == 1
    assert captured["config"].warmup_steps == 2
    assert captured["config"].bootstrap_first_forecast is False
    assert captured["config"].audio_blend_weight == 0.25
    assert captured["config"].offline_archive_storage == "vram"
    assert "Disabling bootstrap_first_forecast" in caplog.text
