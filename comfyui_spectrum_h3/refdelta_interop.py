from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import ClassVar

import torch

from .er_sde_stochastic import ERSDEStepDescriptor, ERSDEStochasticTracker


REFDELTA_BRIDGE_KEY = "spectrum_h3_refdelta_bridge"
REFDELTA_INTEROP_CONTRACT = (
    "comfyui-refdelta-spectrum",
    1,
    "actual-anchor-history",
    "exact-gated-stochastic-increment",
)
REFDELTA_BACKEND_INTEROP_CONTRACT = (
    "comfyui-refdelta-spectrum-backend",
    1,
    "actual-anchor-history",
    "native-solver-noise-geometry",
)
_REFDELTA_CANONICAL_PACKAGE = "comfyui_refdelta_solver"
_REFDELTA_ALIAS_METADATA = frozenset(
    {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__path__",
        "__file__",
        "__cached__",
    }
)
_REFDELTA_TRACKER_BRIDGE_MARKER = "__spectrum_refdelta_bridge__"


class RefDeltaInteropError(RuntimeError):
    """The reviewed Spectrum/RefDelta interop state became inconsistent."""


class _RefDeltaNamespaceAliasLoader(importlib.abc.Loader):
    """Expose an already-loaded ComfyUI RefDelta module under its canonical name."""

    def __init__(self, target: ModuleType) -> None:
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        target = self.target
        for name, value in vars(target).items():
            if name not in _REFDELTA_ALIAS_METADATA:
                module.__dict__[name] = value
        module.__doc__ = getattr(target, "__doc__", None)
        target_file = getattr(target, "__file__", None)
        if target_file is not None:
            module.__file__ = target_file


class _RefDeltaNamespaceAliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ComfyUI's package-relative RefDelta namespace without re-importing it."""

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != _REFDELTA_CANONICAL_PACKAGE and not fullname.startswith(
            f"{_REFDELTA_CANONICAL_PACKAGE}."
        ):
            return None

        suffix = fullname[len(_REFDELTA_CANONICAL_PACKAGE) :]
        nested_suffix = f".{_REFDELTA_CANONICAL_PACKAGE}{suffix}"
        candidates: dict[int, ModuleType] = {}
        for module_name, module in tuple(sys.modules.items()):
            if (
                module is not None
                and module_name != fullname
                and module_name.endswith(nested_suffix)
            ):
                candidates[id(module)] = module
        if len(candidates) != 1:
            return None

        target_module = next(iter(candidates.values()))
        return importlib.util.spec_from_loader(
            fullname,
            _RefDeltaNamespaceAliasLoader(target_module),
            is_package=hasattr(target_module, "__path__"),
        )


def _install_refdelta_namespace_alias_finder() -> None:
    """Prefer the live ComfyUI-loaded RefDelta package when it has a nested namespace."""
    if any(isinstance(finder, _RefDeltaNamespaceAliasFinder) for finder in sys.meta_path):
        return
    # ComfyUI imports custom nodes as package-relative modules. Put this narrow finder
    # before PathFinder so the sampler currently wired into the workflow remains the
    # provenance source even if another top-level RefDelta checkout is importable.
    sys.meta_path.insert(0, _RefDeltaNamespaceAliasFinder())


_install_refdelta_namespace_alias_finder()


@dataclass(slots=True)
class RefDeltaBackendInteropBridge:
    """Classify RefDelta SEEDS/SA model calls from the completed Spectrum step."""

    runtime: object
    api_version: ClassVar[int] = 1
    interop_contract: ClassVar[tuple[str, int, str, str]] = (
        REFDELTA_BACKEND_INTEROP_CONTRACT
    )

    def model_result_is_actual(self, step_id: int) -> bool:
        observed_step = getattr(self.runtime, "last_completed_step_id", None)
        if observed_step != int(step_id):
            raise RefDeltaInteropError(
                "RefDelta backend requested a classification for the wrong Spectrum "
                f"step (requested={int(step_id)}, observed={observed_step})"
            )
        mode = getattr(self.runtime, "last_completed_mode", None)
        if mode == "actual":
            return True
        if mode == "forecast":
            return False
        raise RefDeltaInteropError(
            f"unreviewed Spectrum RefDelta backend step mode {mode!r}"
        )

    def clear(self) -> None:
        return None


@dataclass(slots=True)
class RefDeltaInteropBridge:
    run_id: int
    tracker: ERSDEStochasticTracker | None
    api_version: ClassVar[int] = 1
    interop_contract: ClassVar[tuple[str, int, str, str]] = (
        REFDELTA_INTEROP_CONTRACT
    )
    _descriptor: ERSDEStepDescriptor | None = None

    def __post_init__(self) -> None:
        """Observe the exact descriptor successfully consumed by stochastic tracking.

        ComfyUI can reconstruct model-option dictionaries between the sampler and
        PREDICT_NOISE layers. RefDelta keeps the bridge from the sampler's extra_args,
        while Spectrum's stochastic tracker is the object that actually consumes the
        post-model descriptor. Bind this bridge to that tracker instance so provenance
        follows the successful consume operation instead of depending on a second
        model-options lookup finding the bridge object after the model call.
        """
        tracker = self.tracker
        if tracker is None:
            return
        consume = tracker.consume
        consume_function = getattr(consume, "__func__", consume)
        owner = getattr(consume_function, _REFDELTA_TRACKER_BRIDGE_MARKER, None)
        if owner is not None and owner is not self:
            raise RefDeltaInteropError(
                "ER-SDE tracker is already bound to another Spectrum RefDelta bridge"
            )

        @wraps(consume)
        def consume_with_refdelta_provenance(
            denoised: torch.Tensor,
            descriptor: ERSDEStepDescriptor,
        ) -> torch.Tensor:
            result = consume(denoised, descriptor)
            self.note_model_result(descriptor)
            return result

        setattr(
            consume_with_refdelta_provenance,
            _REFDELTA_TRACKER_BRIDGE_MARKER,
            self,
        )
        tracker.consume = consume_with_refdelta_provenance

    def note_model_result(self, descriptor: ERSDEStepDescriptor) -> None:
        if descriptor.run_id != self.run_id:
            raise RefDeltaInteropError("stale Spectrum RefDelta run descriptor")
        self._descriptor = descriptor

    def model_result_is_actual(self, step_id: int) -> bool:
        descriptor = self._descriptor
        if descriptor is None or descriptor.step_id != int(step_id):
            current = "none" if descriptor is None else str(descriptor.step_id)
            raise RefDeltaInteropError(
                "RefDelta requested a model-result classification for the wrong step "
                f"(requested={int(step_id)}, observed={current})"
            )
        if descriptor.mode == "actual":
            return True
        if descriptor.mode == "forecast":
            return False
        if descriptor.mode == "replay" and descriptor.replay_source_actual is not None:
            return descriptor.replay_source_actual
        raise RefDeltaInteropError(
            f"unreviewed Spectrum RefDelta step mode {descriptor.mode!r}"
        )

    def publish_stochastic_increment(
        self,
        source_step_id: int,
        increment: torch.Tensor,
    ) -> None:
        descriptor = self._descriptor
        if descriptor is None or descriptor.step_id != int(source_step_id):
            current = "none" if descriptor is None else str(descriptor.step_id)
            raise RefDeltaInteropError(
                "RefDelta published a stochastic increment for the wrong step "
                f"(source={int(source_step_id)}, observed={current})"
            )
        if self.tracker is None:
            raise RefDeltaInteropError(
                "RefDelta published a stochastic increment for a deterministic run"
            )
        self.tracker.publish_external_increment(source_step_id, increment)

    def clear(self) -> None:
        self._descriptor = None

    @property
    def is_replay_step(self) -> bool:
        return self._descriptor is not None and self._descriptor.mode == "replay"


__all__ = [
    "REFDELTA_BACKEND_INTEROP_CONTRACT",
    "REFDELTA_BRIDGE_KEY",
    "REFDELTA_INTEROP_CONTRACT",
    "RefDeltaBackendInteropBridge",
    "RefDeltaInteropBridge",
    "RefDeltaInteropError",
]
