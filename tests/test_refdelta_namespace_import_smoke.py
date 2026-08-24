from __future__ import annotations

import importlib
import sys
from types import ModuleType

from comfyui_spectrum_h3.refdelta_interop import REFDELTA_INTEROP_CONTRACT


def test_nested_refdelta_modules_keep_object_identity_through_canonical_alias():
    nested_package = "custom_nodes.synthetic_refdelta_smoke.comfyui_refdelta_solver"
    canonical = "comfyui_refdelta_solver"
    canonical_names = (
        canonical,
        f"{canonical}.config",
        f"{canonical}.sampler",
        f"{canonical}.spectrum_interop",
    )
    nested_names = (
        nested_package,
        f"{nested_package}.config",
        f"{nested_package}.sampler",
        f"{nested_package}.spectrum_interop",
    )
    previous = {name: sys.modules.get(name) for name in canonical_names}

    package_module = ModuleType(nested_names[0])
    package_module.__path__ = []
    config_module = ModuleType(nested_names[1])
    sampler_module = ModuleType(nested_names[2])
    interop_module = ModuleType(nested_names[3])

    class Config:
        pass

    def sampler_function():
        return None

    config_module.RefDeltaSamplerConfig = Config
    sampler_module.sample_refdelta_er_sde = sampler_function
    interop_module.SPECTRUM_INTEROP_CONTRACT = REFDELTA_INTEROP_CONTRACT

    try:
        for name in canonical_names:
            sys.modules.pop(name, None)
        for name, module in zip(
            nested_names,
            (package_module, config_module, sampler_module, interop_module),
            strict=True,
        ):
            sys.modules[name] = module

        imported_config = importlib.import_module(f"{canonical}.config")
        imported_sampler = importlib.import_module(f"{canonical}.sampler")
        imported_interop = importlib.import_module(f"{canonical}.spectrum_interop")

        assert imported_config.RefDeltaSamplerConfig is Config
        assert imported_sampler.sample_refdelta_er_sde is sampler_function
        assert imported_interop.SPECTRUM_INTEROP_CONTRACT is REFDELTA_INTEROP_CONTRACT
    finally:
        for name in canonical_names:
            sys.modules.pop(name, None)
        for name in nested_names:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module
