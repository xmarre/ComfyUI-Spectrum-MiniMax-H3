from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class TemporalRegion:
    region_id: str
    start_row: int
    end_row: int
    start_temporal_token: int
    end_temporal_token: int


def topology_map(topology: tuple[Any, ...]) -> dict[str, Any]:
    return {
        str(entry[0]): entry[1]
        for entry in topology
        if isinstance(entry, tuple) and len(entry) == 2
    }


def temporal_video_regions(
    topology: tuple[Any, ...],
    *,
    video_start_row: int,
    video_end_row: int,
    requested_regions: int = 4,
) -> tuple[TemporalRegion, ...] | None:
    """Map contiguous native H3 video rows into proven temporal bands.

    Native MiniMax H3 patchifies `[B,C,T,H,W]` into `B,T,H,W,...` before
    flattening. With batch size one, every temporal token is therefore one
    contiguous block of `(padded_h/ph)*(padded_w/pw)` rows. The topology
    signature carries every quantity used below; an inconsistent signature
    fails closed rather than guessing a flattening order.
    """
    values = topology_map(topology)
    padded = values.get("video_padded")
    video_shape = values.get("video_shape")
    patch_size = values.get("patch_size")
    target_video_rows = values.get("target_video_rows")
    if (
        not isinstance(padded, (tuple, list))
        or len(padded) != 3
        or not isinstance(video_shape, (tuple, list))
        or len(video_shape) != 5
        or not isinstance(patch_size, (tuple, list))
        or len(patch_size) != 3
        or not isinstance(target_video_rows, int)
    ):
        return None
    try:
        batch = int(video_shape[0])
        temporal, height, width = (int(item) for item in padded)
        pt, ph, pw = (int(item) for item in patch_size)
        start = int(video_start_row)
        end = int(video_end_row)
        requested = int(requested_regions)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        batch != 1
        or min(temporal, height, width, pt, ph, pw, requested) <= 0
        or temporal % pt
        or height % ph
        or width % pw
        or end <= start
    ):
        return None
    temporal_tokens = temporal // pt
    rows_per_temporal_token = (height // ph) * (width // pw)
    expected_rows = temporal_tokens * rows_per_temporal_token
    if target_video_rows != expected_rows or end - start != expected_rows:
        return None
    region_count = min(requested, temporal_tokens)
    if region_count < 2:
        return None
    regions: list[TemporalRegion] = []
    for index in range(region_count):
        start_t = index * temporal_tokens // region_count
        end_t = (index + 1) * temporal_tokens // region_count
        if end_t <= start_t:
            return None
        start_row = start + start_t * rows_per_temporal_token
        end_row = start + end_t * rows_per_temporal_token
        regions.append(
            TemporalRegion(
                region_id=f"t{index}",
                start_row=start_row,
                end_row=end_row,
                start_temporal_token=start_t,
                end_temporal_token=end_t,
            )
        )
    if regions[0].start_row != start or regions[-1].end_row != end:
        return None
    if any(left.end_row != right.start_row for left, right in pairwise(regions)):
        return None
    return tuple(regions)


__all__ = ["TemporalRegion", "temporal_video_regions", "topology_map"]
