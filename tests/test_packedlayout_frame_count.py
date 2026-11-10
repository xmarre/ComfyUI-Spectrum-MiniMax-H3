"""Regression coverage for native H3 PackedLayout constructor compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3 import minimax_h3


class _CurrentPackedLayout:
    """Matches ComfyUI revisions whose constructor omits frame_count."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        self.keyframes = keyframes
        self.refs = refs


class _LegacyPackedLayout(_CurrentPackedLayout):
    """Matches ComfyUI revisions that need frame_count for layout rebuilding."""

    def __init__(
        self, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=None, refs=None, frame_count=None,
    ):
        super().__init__(text_len, latent_t, latent_h, latent_w, audio_t, keyframes, refs)
        self.frame_count = frame_count


@pytest.mark.parametrize("layout_type", [_CurrentPackedLayout, _LegacyPackedLayout])
@pytest.mark.parametrize("cached_layout", [None, SimpleNamespace(signature=(5, 2, 6, 6, 4))])
def test_resolve_layout_rebuild_uses_native_constructor_contract(monkeypatch, layout_type, cached_layout):
    monkeypatch.setattr(
        minimax_h3,
        "_native_module",
        lambda _inner: SimpleNamespace(PackedLayout=layout_type),
    )
    inner = SimpleNamespace(patch_size=(2, 2, 2))
    context = SimpleNamespace(shape=(1, 5, 8))
    video = SimpleNamespace(shape=(1, 8, 3, 5, 6))
    audio = SimpleNamespace(shape=(1, 2, 2, 4))
    keyframes = [{"resolved_frame_index": 0}]
    refs = [{"kind": "image"}]
    payload = {
        "layout": cached_layout,
        "keyframes": keyframes,
        "refs": refs,
        "frame_count": 11,
    }

    result = minimax_h3._resolve_layout(inner, context, video, audio, payload)

    assert isinstance(result, layout_type)
    assert result.signature == (5, 4, 6, 6, 4)
    assert result.keyframes is keyframes
    assert result.refs is refs
    if layout_type is _LegacyPackedLayout:
        assert result.frame_count == 11
    else:
        assert not hasattr(result, "frame_count")
    assert payload["layout"] is cached_layout


def test_resolve_layout_reuses_matching_cached_layout(monkeypatch):
    class NoRebuildAllowed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("matching cached layout must not be rebuilt")

    monkeypatch.setattr(
        minimax_h3,
        "_native_module",
        lambda _inner: SimpleNamespace(PackedLayout=NoRebuildAllowed),
    )
    inner = SimpleNamespace(patch_size=(2, 2, 2))
    cached = SimpleNamespace(signature=(5, 4, 6, 6, 4))

    result = minimax_h3._resolve_layout(
        inner,
        SimpleNamespace(shape=(1, 5, 8)),
        SimpleNamespace(shape=(1, 8, 3, 5, 6)),
        SimpleNamespace(shape=(1, 2, 2, 4)),
        {"layout": cached, "frame_count": 11},
    )

    assert result is cached


def test_resolve_layout_matches_native_comfyui_on_stale_cache(monkeypatch):
    try:
        from comfy.ldm.minimax import model as native
    except Exception as exc:
        pytest.skip(f"native ComfyUI MiniMax H3 is unavailable: {exc}")

    monkeypatch.setattr(minimax_h3, "_native_module", lambda _inner: native)
    inner = SimpleNamespace(patch_size=(1, 2, 2))
    context = SimpleNamespace(shape=(1, 5, 8))
    video = SimpleNamespace(shape=(1, 8, 1, 4, 4))
    audio = SimpleNamespace(shape=(1, 2, 2, 3))
    payload = {
        "layout": SimpleNamespace(signature=(5, 2, 4, 4, 3)),
        "frame_count": 1,
    }

    rebuilt = minimax_h3._resolve_layout(inner, context, video, audio, payload)
    expected_kwargs = {"keyframes": None, "refs": None}
    import inspect

    if "frame_count" in inspect.signature(native.PackedLayout.__init__).parameters:
        expected_kwargs["frame_count"] = 1
    expected = native.PackedLayout(5, 1, 4, 4, 3, **expected_kwargs)

    assert rebuilt.signature == expected.signature
    assert rebuilt.segments == expected.segments
    assert rebuilt.seq_len == expected.seq_len
    import torch

    torch.testing.assert_close(rebuilt.position_ids, expected.position_ids, rtol=0, atol=0)
