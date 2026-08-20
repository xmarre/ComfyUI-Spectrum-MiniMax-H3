from __future__ import annotations

import math
from typing import Any

REFINEMENT_REQUEST_KEY = "h3_refinement"
REFINEMENT_API = 1

_INSTALL_MARKER_ATTR = "_spectrum_h3_refinement_compat_install_marker"
_INSTALL_MARKER_VALUE = f"spectrum-h3-refinement:v{REFINEMENT_API}"
_ORIGINAL_CONTINUUM_ACTUAL_PREFIX = None
_INSTALLED = False


def _refinement_request(model_options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a validated low-sigma refinement contract, otherwise None.

    Invalid metadata deliberately fails open to the existing Continuum policy:
    it must never suppress Continuum's native prefix unless the producer supplied
    a complete, reviewed refinement contract.
    """
    transformer_options = (model_options or {}).get("transformer_options")
    if not isinstance(transformer_options, dict):
        return None
    request = transformer_options.get(REFINEMENT_REQUEST_KEY)
    if not isinstance(request, dict):
        return None

    api = request.get("api")
    active = request.get("active")
    prefix = request.get("min_actual_prefix_steps")
    sigma_reference = request.get("sigma_reference")
    if type(api) is not int or type(active) is not bool or type(prefix) is not int:
        return None
    if api != REFINEMENT_API or active is not True or prefix < 0:
        return None
    if isinstance(sigma_reference, bool) or not isinstance(sigma_reference, (int, float)):
        return None
    sigma_reference = float(sigma_reference)
    if not math.isfinite(sigma_reference) or sigma_reference <= 0.0:
        return None

    return {
        "api": api,
        "active": active,
        "min_actual_prefix_steps": prefix,
        "sigma_reference": sigma_reference,
    }


def _continuum_actual_prefix_with_refinement(model_options: dict[str, Any] | None) -> int:
    request = _refinement_request(model_options)
    if request is not None:
        return int(request["min_actual_prefix_steps"])
    if _ORIGINAL_CONTINUUM_ACTUAL_PREFIX is None:
        raise RuntimeError("Spectrum H3 refinement compatibility lost the original Continuum prefix resolver")
    return _ORIGINAL_CONTINUUM_ACTUAL_PREFIX(model_options)


def _is_installed_replacement(value: Any) -> bool:
    return getattr(value, _INSTALL_MARKER_ATTR, None) == _INSTALL_MARKER_VALUE


def install_refinement_compat() -> None:
    """Let explicit sampler-2 metadata override sampler-1 Continuum prefix policy."""
    global _INSTALLED, _ORIGINAL_CONTINUUM_ACTUAL_PREFIX

    from . import sampling

    current = sampling._continuum_actual_prefix
    if _is_installed_replacement(current):
        _INSTALLED = True
        return

    _ORIGINAL_CONTINUUM_ACTUAL_PREFIX = current
    setattr(
        _continuum_actual_prefix_with_refinement,
        _INSTALL_MARKER_ATTR,
        _INSTALL_MARKER_VALUE,
    )
    sampling._continuum_actual_prefix = _continuum_actual_prefix_with_refinement
    _INSTALLED = True


__all__ = [
    "REFINEMENT_API",
    "REFINEMENT_REQUEST_KEY",
    "install_refinement_compat",
]
