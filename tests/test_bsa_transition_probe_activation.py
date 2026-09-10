import sys
from types import ModuleType

from comfyui_spectrum_h3 import bsa_transition_probe as probe


def test_diagnostic_does_not_auto_enable_without_cuda(monkeypatch):
    monkeypatch.delenv(probe.ENV, raising=False)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    assert probe._diagnostic_directory(automatic=True) is None


def test_diagnostic_auto_writes_below_comfyui_output_on_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv(probe.ENV, raising=False)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    assert probe._diagnostic_directory(automatic=True) == (
        tmp_path / probe.AUTO_OUTPUT_SUBDIR
    )


def test_explicit_directory_overrides_automatic_location(monkeypatch, tmp_path):
    target = tmp_path / "custom-diagnostics"
    monkeypatch.setenv(probe.ENV, str(target))
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    assert probe._diagnostic_directory(automatic=False) == target


def test_explicit_disable_beats_automatic_cuda_activation(monkeypatch):
    monkeypatch.setenv(probe.ENV, "off")
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    assert probe._diagnostic_directory(automatic=True) is None
