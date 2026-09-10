from pathlib import Path

from comfyui_spectrum_h3 import bsa_transition_probe as probe


def test_transition_probe_is_not_automatic_after_cuda_evidence(monkeypatch):
    monkeypatch.delenv(probe.ENV, raising=False)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    assert probe._diagnostic_directory(automatic=True) is None


def test_transition_probe_explicit_environment_override_still_works(monkeypatch, tmp_path):
    monkeypatch.setenv(probe.ENV, str(tmp_path))
    assert probe._diagnostic_directory(automatic=True) == Path(tmp_path)
    assert probe._diagnostic_directory(automatic=False) == Path(tmp_path)


def test_transition_probe_explicit_disable_still_wins(monkeypatch):
    monkeypatch.setenv(probe.ENV, "off")
    assert probe._diagnostic_directory(automatic=True) is None
