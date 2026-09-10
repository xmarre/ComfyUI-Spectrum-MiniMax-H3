from comfyui_spectrum_h3 import bsa_transition_probe as probe


def test_diagnostic_does_not_auto_enable_without_cuda(monkeypatch):
    monkeypatch.delenv(probe.ENV, raising=False)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    assert probe._diagnostic_directory(automatic=True) is None


def test_diagnostic_no_longer_auto_enables_on_cuda_after_evidence(monkeypatch):
    monkeypatch.delenv(probe.ENV, raising=False)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    assert probe._diagnostic_directory(automatic=True) is None


def test_explicit_directory_overrides_automatic_location(monkeypatch, tmp_path):
    target = tmp_path / "custom-diagnostics"
    monkeypatch.setenv(probe.ENV, str(target))
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: False)
    assert probe._diagnostic_directory(automatic=False) == target
    assert probe._diagnostic_directory(automatic=True) == target


def test_explicit_disable_beats_automatic_cuda_activation(monkeypatch):
    monkeypatch.setenv(probe.ENV, "off")
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    assert probe._diagnostic_directory(automatic=True) is None
