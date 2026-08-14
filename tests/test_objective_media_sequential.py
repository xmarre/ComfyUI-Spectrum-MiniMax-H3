from __future__ import annotations

import json

import pytest
import torch

import comfyui_spectrum_h3.objective_media_nodes as nodes
from comfyui_spectrum_h3.objective_media import ObjectiveMediaError


def _video(frames: int = 4, height: int = 8, width: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.rand((frames, height, width, 3), generator=generator)


def _audio(sample_rate: int = 8000) -> dict[str, object]:
    waveform = torch.linspace(-0.5, 0.5, sample_rate // 10)[None, None, :]
    return {"waveform": waveform, "sample_rate": sample_rate}


@pytest.fixture(autouse=True)
def _clear_pending():
    nodes.clear_pending_objective_media()
    yield
    nodes.clear_pending_objective_media()


def _capture(node, role: str, *, benchmark_id: str = "seq-1", video=None, reset=False):
    return node.capture(
        _video() if video is None else video,
        role,
        24.0,
        benchmark_id,
        123,
        nodes.DEFAULT_PROVENANCE_JSON,
        4,
        reset_before_capture=reset,
        audio=_audio(),
    )


def test_sequential_capture_three_runs_auto_evaluates_and_releases(monkeypatch):
    calls = []

    def fake_evaluate(reference_video, legacy_video, candidate_video, **kwargs):
        calls.append((reference_video, legacy_video, candidate_video, kwargs))
        return ("done", "report.json", "report.md", "aggregate.json", "aggregate.md")

    monkeypatch.setattr(nodes, "_evaluate_and_persist", fake_evaluate)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()

    first = _capture(node, "R - native reference")
    second = _capture(node, "A - legacy Spectrum")
    third = _capture(node, "B - candidate")

    assert "pending=A,B" in first[0]
    assert "pending=B" in second[0]
    assert third[0] == "done"
    assert len(calls) == 1
    assert calls[0][3]["benchmark_id"] == "seq-1"
    assert calls[0][3]["seed"] == 123
    assert calls[0][3]["reference_audio"]["waveform"].device.type == "cpu"
    assert calls[0][0].device.type == "cpu"
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_sequential_capture_role_order_is_independent(monkeypatch):
    seen = []

    def fake_evaluate(reference_video, legacy_video, candidate_video, **kwargs):
        seen.append((reference_video.mean().item(), legacy_video.mean().item(), candidate_video.mean().item()))
        return ("done", "", "", "", "")

    monkeypatch.setattr(nodes, "_evaluate_and_persist", fake_evaluate)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    base = _video()
    node.capture(base + 0.2, "B - candidate", 24.0, "order", 123, nodes.DEFAULT_PROVENANCE_JSON, 4, False, _audio())
    node.capture(base, "R - native reference", 24.0, "order", 123, nodes.DEFAULT_PROVENANCE_JSON, 4, False, _audio())
    node.capture(base + 0.1, "A - legacy Spectrum", 24.0, "order", 123, nodes.DEFAULT_PROVENANCE_JSON, 4, False, _audio())

    assert len(seen) == 1
    assert seen[0][0] < seen[0][1] < seen[0][2]


def test_duplicate_role_is_rejected_and_reset_before_capture_restarts():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    with pytest.raises(ObjectiveMediaError, match="already contains role R"):
        _capture(node, "R - native reference")

    _capture(node, "A - legacy Spectrum", reset=True)
    state = nodes.pending_objective_media_state()
    assert state["benchmarks"]["seq-1"]["roles"] == ["A"]


def test_incompatible_topology_is_rejected_without_destroying_existing_capture():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    with pytest.raises(ObjectiveMediaError, match="matching decoded video/audio topology"):
        _capture(node, "A - legacy Spectrum", video=_video(width=9))
    state = nodes.pending_objective_media_state()
    assert state["benchmarks"]["seq-1"]["roles"] == ["R"]


def test_completion_failure_releases_all_raw_media(monkeypatch):
    def fail(*args, **kwargs):
        raise ObjectiveMediaError("synthetic evaluation failure")

    monkeypatch.setattr(nodes, "_evaluate_and_persist", fail)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    _capture(node, "A - legacy Spectrum")
    with pytest.raises(ObjectiveMediaError, match="synthetic evaluation failure"):
        _capture(node, "B - candidate")
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_pending_benchmark_count_is_bounded_by_eviction(monkeypatch):
    monkeypatch.setattr(nodes, "MAX_PENDING_BENCHMARKS", 1)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference", benchmark_id="old")
    _capture(node, "R - native reference", benchmark_id="new")
    state = nodes.pending_objective_media_state()
    assert set(state["benchmarks"]) == {"new"}


def test_single_capture_over_ram_bound_is_rejected(monkeypatch):
    monkeypatch.setattr(nodes, "MAX_PENDING_BYTES", 1)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    with pytest.raises(ObjectiveMediaError, match="exceeds the sequential capture RAM limit"):
        _capture(node, "R - native reference")
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_capture_is_cpu_only_and_does_not_persist_raw_media(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    pending = nodes._PENDING_CAPTURES["seq-1"]["roles"]["R"]
    assert pending["video"].device.type == "cpu"
    assert pending["audio"]["waveform"].device.type == "cpu"
    assert list(tmp_path.rglob("*")) == []


def test_capture_reset_node_releases_one_or_all_benchmarks():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference", benchmark_id="one")
    _capture(node, "R - native reference", benchmark_id="two")
    reset = nodes.SpectrumH3ObjectiveCaptureReset()
    (summary,) = reset.clear("one", "benchmark")
    assert "one" in summary
    assert set(nodes.pending_objective_media_state()["benchmarks"]) == {"two"}
    reset.clear("unused", "all")
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_default_provenance_is_nonempty_and_node_is_registered():
    provenance = json.loads(nodes.DEFAULT_PROVENANCE_JSON)
    for field in (
        "model",
        "model_weights",
        "precision",
        "sampler",
        "scheduler",
        "conditioning",
        "video_vae",
        "audio_decoder",
    ):
        assert provenance["compatibility"][field]
    assert provenance["compatibility"]["steps"] == 20
    assert provenance["R"] and provenance["A"] and provenance["B"]
    assert (
        nodes.NODE_CLASS_MAPPINGS["SpectrumH3ObjectiveSequentialCapture"]
        is nodes.SpectrumH3ObjectiveSequentialCapture
    )
    assert nodes.SpectrumH3ObjectiveSequentialCapture.OUTPUT_NODE is True
