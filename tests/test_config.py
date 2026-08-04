from __future__ import annotations

import pytest

from comfyui_spectrum_h3.config import SpectrumH3Config


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
