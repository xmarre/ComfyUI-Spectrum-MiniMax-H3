from __future__ import annotations

import comfyui_spectrum_h3.minimax_h3 as minimax_h3


def test_forecast_head_budget_keeps_conservative_default():
    mib = 1024 * 1024
    assert minimax_h3._forecast_head_chunk_budget_from_env(None) == 16 * mib
    assert minimax_h3._forecast_head_chunk_budget_from_env("") == 16 * mib


def test_forecast_head_budget_supported_sizes():
    mib = 1024 * 1024
    for value in ("16", "32", "64", "128"):
        assert minimax_h3._forecast_head_chunk_budget_from_env(value) == int(value) * mib


def test_forecast_head_budget_rejects_invalid_values():
    default = 16 * 1024 * 1024
    for value in ("0", "-1", "15", "1024", "abc", "2.0"):
        assert minimax_h3._forecast_head_chunk_budget_from_env(value) == default
