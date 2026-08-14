from __future__ import annotations

import gc
import json
import weakref

import pytest
import torch

import comfyui_spectrum_h3.objective_media_nodes as nodes
from comfyui_spectrum_h3.objective_media import ObjectiveMediaError


def _video(
    frames: int = 4,
    height: int = 8,
    width: int = 8,
    *,
    offset: float = 0.0,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return (
        torch.rand((frames, height, width, 3), generator=generator) * 0.7
        + offset
    ).clamp(0.0, 1.0)


def _audio(sample_rate: int = 8000) -> dict[str, object]:
    waveform = torch.linspace(-0.5, 0.5, sample_rate // 10)[None, None, :]
    return {"waveform": waveform, "sample_rate": sample_rate}


@pytest.fixture(autouse=True)
def _clear_pending():
    nodes.clear_pending_objective_media()
    yield
    nodes.clear_pending_objective_media()


def _capture(
    node,
    role: str,
    *,
    benchmark_id: str = "seq-1",
    video=None,
    reset=False,
):
    return node.capture(
        _video() if video is None else video,
        role,
        24.0,
        benchmark_id,
        123,
        20,
        "fixture-workflow",
        4,
        reset_before_capture=reset,
        audio=_audio(),
    )


def test_sequential_capture_three_runs_auto_evaluates_and_releases(monkeypatch):
    calls = []

    def fake_evaluate(reference_video, legacy_video, candidate_video, **kwargs):
        calls.append((reference_video, legacy_video, candidate_video, kwargs))
        return (
            "done",
            "report.json",
            "report.md",
            "aggregate.json",
            "aggregate.md",
        )

    monkeypatch.setattr(nodes, "_evaluate_and_persist_sequential", fake_evaluate)
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
    assert calls[0][3]["provenance"]["compatibility"]["steps"] == 20
    assert (
        calls[0][3]["provenance"]["compatibility"]["generation_settings"][
            "compatibility_tag"
        ]
        == "fixture-workflow"
    )
    assert calls[0][3]["reference_audio"]["waveform"].device.type == "cpu"
    assert calls[0][0].device.type == "cpu"
    assert calls[0][0].dtype == torch.float16
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_sequential_capture_role_order_is_independent(monkeypatch):
    seen = []

    def fake_evaluate(reference_video, legacy_video, candidate_video, **kwargs):
        seen.append(
            (
                reference_video.float().mean().item(),
                legacy_video.float().mean().item(),
                candidate_video.float().mean().item(),
            )
        )
        return ("done", "", "", "", "")

    monkeypatch.setattr(nodes, "_evaluate_and_persist_sequential", fake_evaluate)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    base = _video()
    node.capture(
        base + 0.2,
        "B - candidate",
        24.0,
        "order",
        123,
        20,
        "fixture",
        4,
        False,
        _audio(),
    )
    node.capture(
        base,
        "R - native reference",
        24.0,
        "order",
        123,
        20,
        "fixture",
        4,
        False,
        _audio(),
    )
    node.capture(
        base + 0.1,
        "A - legacy Spectrum",
        24.0,
        "order",
        123,
        20,
        "fixture",
        4,
        False,
        _audio(),
    )

    assert len(seen) == 1
    assert seen[0][0] < seen[0][1] < seen[0][2]


def test_sequential_capture_retains_bounded_float16_analysis_not_full_media(
    monkeypatch,
):
    monkeypatch.setattr(nodes, "SEQUENTIAL_MAX_ANALYSIS_PIXELS", 64)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    source = _video(frames=3, height=16, width=16)
    _capture(node, "R - native reference", video=source)

    pending = nodes._PENDING_CAPTURES["seq-1"]
    staged = pending["roles"]["R"]["video"]
    metadata = pending["source_video_metadata"]

    assert staged.device.type == "cpu"
    assert staged.dtype == torch.float16
    assert staged.shape[0] == source.shape[0]
    assert staged.shape[-1] == 3
    assert staged.shape[1] * staged.shape[2] <= 64
    assert metadata["height"] == 16
    assert metadata["width"] == 16
    assert metadata["analysis_height"] == staged.shape[1]
    assert metadata["analysis_width"] == staged.shape[2]
    assert (
        staged.numel() * staged.element_size()
        < source.numel() * source.element_size()
    )


def test_duplicate_role_is_reported_nonfatally_and_reset_restarts():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    duplicate = _capture(node, "R - native reference")

    assert "skipped without aborting the workflow" in duplicate[0]
    assert "already contains role R" in duplicate[0]
    assert nodes.pending_objective_media_state()["benchmarks"]["seq-1"][
        "roles"
    ] == ["R"]

    _capture(node, "A - legacy Spectrum", reset=True)
    state = nodes.pending_objective_media_state()
    assert state["benchmarks"]["seq-1"]["roles"] == ["A"]


def test_incompatible_source_topology_is_rejected_without_destroying_existing_capture():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    result = _capture(node, "A - legacy Spectrum", video=_video(width=9))

    assert "skipped without aborting the workflow" in result[0]
    assert "matching decoded source" in result[0]
    state = nodes.pending_objective_media_state()
    assert state["benchmarks"]["seq-1"]["roles"] == ["R"]


def test_completion_failure_releases_bounded_analysis_media(monkeypatch):
    def fail(*args, **kwargs):
        raise ObjectiveMediaError("synthetic evaluation failure")

    monkeypatch.setattr(nodes, "_evaluate_and_persist_sequential", fail)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    _capture(node, "A - legacy Spectrum")
    result = _capture(node, "B - candidate")

    assert "synthetic evaluation failure" in result[0]
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
    result = _capture(node, "R - native reference")

    assert "exceeds the sequential RAM limit" in result[0]
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_capture_is_cpu_only_and_does_not_persist_raw_media(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    pending = nodes._PENDING_CAPTURES["seq-1"]["roles"]["R"]
    assert pending["video"].device.type == "cpu"
    assert pending["video"].dtype == torch.float16
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


def test_sequential_schema_requires_linked_int_seed_and_has_no_json_blob():
    required = nodes.SpectrumH3ObjectiveSequentialCapture.INPUT_TYPES()["required"]
    generation_seed = required["generation_seed"]
    assert generation_seed[0] == "INT"
    assert generation_seed[1]["forceInput"] is True
    assert "default" not in generation_seed[1]
    assert "control_after_generate" not in generation_seed[1]
    assert "seed" not in required
    assert "provenance_json" not in required
    assert required["compatibility_tag"][0] == "STRING"


def test_generation_seed_validation_is_strict():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    for invalid in ("not-a-seed", -1, 2**64):
        result = node.capture(
            _video(),
            "R - native reference",
            24.0,
            f"bad-{invalid}",
            invalid,
            20,
            "fixture",
            4,
            False,
            _audio(),
        )
        assert "generation_seed" in result[0]
        assert "skipped without aborting the workflow" in result[0]


@pytest.mark.parametrize(
    ("height", "width", "dtype", "pixel_cap"),
    (
        (7, 9, torch.float16, 256),
        (9, 11, torch.bfloat16, 64),
        (17, 19, torch.float32, 64),
        (13, 21, torch.float64, 80),
    ),
)
def test_sequential_staging_is_bounded_deterministic_and_dtype_independent(
    monkeypatch,
    height,
    width,
    dtype,
    pixel_cap,
):
    monkeypatch.setattr(nodes, "SEQUENTIAL_MAX_ANALYSIS_PIXELS", pixel_cap)
    source = _video(frames=5, height=height, width=width).to(dtype=dtype)
    original_source = source.clone()
    audio = _audio()

    first, first_metadata = nodes._stage_media_sequential(source, audio)
    second, second_metadata = nodes._stage_media_sequential(source, audio)
    target_height, target_width = nodes._bounded_analysis_size(height, width)

    assert first["video"].shape == (5, target_height, target_width, 3)
    assert first["video"].dtype == torch.float16
    assert first["video"].device.type == "cpu"
    assert target_height * target_width <= pixel_cap
    assert torch.equal(first["video"], second["video"])
    assert torch.equal(source, original_source)
    assert first_metadata["analysis_height"] == target_height
    assert first_metadata["analysis_width"] == target_width
    assert first_metadata["analysis_bytes"] == second_metadata["analysis_bytes"]
    assert first["audio"]["waveform"].dtype == torch.float32
    assert first["audio"]["waveform"].data_ptr() != audio["waveform"].data_ptr()


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_sequential_staging_matches_previous_numeric_transform(monkeypatch, dtype):
    monkeypatch.setattr(nodes, "SEQUENTIAL_MAX_ANALYSIS_PIXELS", 64)
    source = _video(frames=5, height=17, width=19).to(dtype=dtype)
    target_height, target_width = nodes._bounded_analysis_size(17, 19)
    expected = torch.empty((5, target_height, target_width, 3), dtype=torch.float16)
    for start in range(0, 5, nodes.SEQUENTIAL_STAGE_CHUNK_FRAMES):
        end = min(5, start + nodes.SEQUENTIAL_STAGE_CHUNK_FRAMES)
        nchw = source[start:end].clamp(0.0, 1.0).movedim(-1, 1)
        resized = torch.nn.functional.interpolate(
            nchw.to(dtype=torch.float32),
            size=(target_height, target_width),
            mode="area",
        )
        expected[start:end].copy_(resized.movedim(1, -1).to(torch.float16))

    staged, _ = nodes._stage_media_sequential(source)
    assert torch.equal(staged["video"], expected)


def test_sequential_validation_never_materializes_source_sized_isfinite(
    monkeypatch,
):
    original = nodes.torch.isfinite
    checked_sizes = []

    def tracked(value):
        checked_sizes.append(value.numel())
        return original(value)

    monkeypatch.setattr(nodes.torch, "isfinite", tracked)
    nodes._stage_media_sequential(_video(frames=5, height=17, width=19), _audio())

    assert checked_sizes
    assert max(checked_sizes) == 2


def test_memory_estimate_scales_with_media_not_total_video_workspace():
    common = {
        "video_dtype": torch.float32,
        "source_device_type": "cpu",
    }
    small = nodes._memory_estimate_for_shapes((192, 608, 800, 3), **common)
    medium = nodes._memory_estimate_for_shapes((192, 640, 864, 3), **common)
    large = nodes._memory_estimate_for_shapes((192, 736, 960, 3), **common)
    longer = nodes._memory_estimate_for_shapes((384, 736, 960, 3), **common)
    with_pending = nodes._memory_estimate_for_shapes(
        (192, 736, 960, 3),
        **common,
        existing_pending_bytes=123_456,
    )
    with_audio = nodes._memory_estimate_for_shapes(
        (192, 736, 960, 3),
        **common,
        audio_shape=(1, 2, 384_000),
        audio_dtype=torch.float32,
        audio_device_type="cpu",
    )
    cuda_source = nodes._memory_estimate_for_shapes(
        (192, 736, 960, 3),
        torch.float32,
        "cuda",
    )

    assert small["source_tensor_bytes"] < medium["source_tensor_bytes"]
    assert medium["source_tensor_bytes"] < large["source_tensor_bytes"]
    assert small["estimated_host_live_bytes"] < medium["estimated_host_live_bytes"]
    assert medium["estimated_host_live_bytes"] < large["estimated_host_live_bytes"]
    assert longer["source_tensor_bytes"] == 2 * large["source_tensor_bytes"]
    assert longer["retained_video_bytes"] == 2 * large["retained_video_bytes"]
    assert (
        longer["staging_workspace_estimate"]
        == large["staging_workspace_estimate"]
    )
    assert (
        with_pending["estimated_host_live_bytes"]
        - large["estimated_host_live_bytes"]
        == 123_456
    )
    assert with_audio["source_audio_bytes"] > 0
    assert with_audio["retained_analysis_bytes"] > large["retained_analysis_bytes"]
    assert large["chunk_frames"] <= nodes.SEQUENTIAL_STAGE_CHUNK_FRAMES
    assert large["staging_workspace_estimate"] < large["source_tensor_bytes"]
    assert cuda_source["cpu_transfer_workspace_bytes"] > 0
    assert cuda_source["device_staging_workspace_bytes"] > 0
    assert cuda_source["uses_cuda"] == 1


def test_unsafe_host_preflight_rejects_before_destination_allocation(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_memory_snapshot",
        lambda device: {
            "rss_bytes": 1_000_000,
            "available_host_bytes": 1,
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
        },
    )

    def must_not_stage(*args, **kwargs):
        raise AssertionError("staging was reached")

    monkeypatch.setattr(nodes, "_stage_media_sequential", must_not_stage)
    result = _capture(
        nodes.SpectrumH3ObjectiveSequentialCapture(),
        "R - native reference",
    )

    assert "aborted before staging" in result[0]
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_missing_memory_telemetry_is_nonfatal(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "_memory_snapshot",
        lambda device: {
            "rss_bytes": None,
            "available_host_bytes": None,
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
        },
    )
    result = _capture(
        nodes.SpectrumH3ObjectiveSequentialCapture(),
        "R - native reference",
    )
    assert "stored R" in result[0]


def test_missing_memory_telemetry_uses_conservative_absolute_guard(monkeypatch):
    monkeypatch.setattr(nodes, "SEQUENTIAL_UNMEASURED_INCREMENTAL_LIMIT_BYTES", 1)
    monkeypatch.setattr(
        nodes,
        "_memory_snapshot",
        lambda device: {
            "rss_bytes": None,
            "available_host_bytes": None,
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
        },
    )

    def must_not_stage(*args, **kwargs):
        raise AssertionError("staging was reached")

    monkeypatch.setattr(nodes, "_stage_media_sequential", must_not_stage)
    result = _capture(
        nodes.SpectrumH3ObjectiveSequentialCapture(),
        "R - native reference",
    )

    assert "telemetry is unavailable" in result[0]


def test_duplicate_and_topology_failures_happen_before_staging(monkeypatch):
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")

    def must_not_stage(*args, **kwargs):
        raise AssertionError("staging was reached")

    monkeypatch.setattr(nodes, "_stage_media_sequential", must_not_stage)
    duplicate = _capture(node, "R - native reference")
    topology = _capture(node, "A - legacy Spectrum", video=_video(width=9))

    assert "already contains role R" in duplicate[0]
    assert "matching decoded source" in topology[0]
    assert nodes.pending_objective_media_state()["benchmarks"]["seq-1"][
        "roles"
    ] == ["R"]


def test_staging_failure_preserves_accepted_roles(monkeypatch):
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")

    def fail(*args, **kwargs):
        raise ObjectiveMediaError("synthetic staging failure")

    monkeypatch.setattr(nodes, "_stage_media_sequential", fail)
    result = _capture(node, "A - legacy Spectrum")

    assert "synthetic staging failure" in result[0]
    state = nodes.pending_objective_media_state()
    assert state["benchmarks"]["seq-1"]["roles"] == ["R"]


def test_unexpected_capture_failure_is_logged_and_nonfatal(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic unexpected failure")

    monkeypatch.setattr(nodes, "_source_topology", fail)
    with caplog.at_level("ERROR"):
        result = _capture(
            nodes.SpectrumH3ObjectiveSequentialCapture(),
            "R - native reference",
        )

    assert "synthetic unexpected failure" in result[0]
    assert "skipped without aborting the workflow" in result[0]
    assert "other output nodes can continue" in caplog.text
    assert nodes.pending_objective_media_state()["benchmark_count"] == 0


def test_capture_does_not_swallow_execution_interrupts(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(nodes, "_source_topology", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _capture(
            nodes.SpectrumH3ObjectiveSequentialCapture(),
            "R - native reference",
        )


def test_pending_capture_does_not_retain_source_tensor_objects():
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    source_video = _video()
    source_audio = _audio()
    video_reference = weakref.ref(source_video)
    audio_reference = weakref.ref(source_audio["waveform"])
    node.capture(
        source_video,
        "R",
        24.0,
        "ownership",
        123,
        20,
        "fixture",
        4,
        audio=source_audio,
    )
    del source_video, source_audio
    gc.collect()

    assert video_reference() is None
    assert audio_reference() is None
    pending = nodes._PENDING_CAPTURES["ownership"]["roles"]["R"]
    assert pending["video"].dtype == torch.float16
    assert pending["audio"]["waveform"].dtype == torch.float32


def test_reset_and_eviction_release_retained_tensors(monkeypatch):
    monkeypatch.setattr(nodes, "MAX_PENDING_BENCHMARKS", 1)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference", benchmark_id="old")
    old_video = weakref.ref(nodes._PENDING_CAPTURES["old"]["roles"]["R"]["video"])
    _capture(node, "R - native reference", benchmark_id="new")
    gc.collect()
    assert old_video() is None

    new_video = weakref.ref(nodes._PENDING_CAPTURES["new"]["roles"]["R"]["video"])
    nodes.clear_pending_objective_media("new")
    gc.collect()
    assert new_video() is None


def test_completed_triad_releases_all_retained_media(monkeypatch):
    observed = []

    def fake_evaluate(reference_video, legacy_video, candidate_video, **kwargs):
        observed.extend(
            weakref.ref(value)
            for value in (reference_video, legacy_video, candidate_video)
        )
        return ("done", "", "", "", "")

    monkeypatch.setattr(nodes, "_evaluate_and_persist_sequential", fake_evaluate)
    node = nodes.SpectrumH3ObjectiveSequentialCapture()
    _capture(node, "R - native reference")
    _capture(node, "A - legacy Spectrum")
    _capture(node, "B - candidate")
    gc.collect()

    assert nodes.pending_objective_media_state()["benchmark_count"] == 0
    assert all(reference() is None for reference in observed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_staging_is_optional_and_returns_cpu_analysis():
    source = _video(frames=3, height=17, width=19).cuda()
    staged, metadata = nodes._stage_media_sequential(source)
    assert staged["video"].device.type == "cpu"
    assert staged["video"].dtype == torch.float16
    assert metadata["device"].startswith("cuda")
