from __future__ import annotations

from typing import Any

from . import external_patch_compat as compat


_INSTALLED = False
_ORIGINAL_CLEAR_MODEL_PROFILE_CACHE = None


def get_model_forecastability_profile_fail_safe(model_patcher: Any) -> Any:
    """Keep malformed declared metadata from aborting model-aware setup.

    Runtime setup parses the same declaration before sampling and then disables
    forecasting for that run. Returning the ordinary base profile here avoids a
    premature exception while preserving the required all-actual fail-safe.
    """

    try:
        return compat.get_model_forecastability_profile_with_external_patches(
            model_patcher
        )
    except compat.ExternalPatchContractError:
        assert compat._ORIGINAL_PROFILE_LOOKUP is not None
        return compat._ORIGINAL_PROFILE_LOOKUP(model_patcher)


def _clear_all_model_profile_caches() -> None:
    assert _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE is not None
    _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE()
    compat.clear_external_profile_cache()


def install_external_patch_hardening() -> None:
    global _INSTALLED
    global _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE
    if _INSTALLED:
        return
    if not compat._INSTALLED or compat._ORIGINAL_PROFILE_LOOKUP is None:
        raise RuntimeError(
            "external patch hardening requires install_external_patch_compat() first"
        )

    from . import model_aware
    from . import sampling

    _ORIGINAL_CLEAR_MODEL_PROFILE_CACHE = model_aware.clear_model_profile_cache

    model_aware.get_model_forecastability_profile = (
        get_model_forecastability_profile_fail_safe
    )
    sampling.get_model_forecastability_profile = (
        get_model_forecastability_profile_fail_safe
    )
    model_aware.clear_model_profile_cache = _clear_all_model_profile_caches
    _INSTALLED = True


__all__ = [
    "get_model_forecastability_profile_fail_safe",
    "install_external_patch_hardening",
]