"""Pure planning helpers for segmented Best Face long-video generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentPlan:
    """One independently generated shot on the source-audio timeline."""

    index: int
    start_time: float
    source_duration: float
    num_frames: int
    output_duration: float


def snap_frame_count(duration: float, frame_rate: float) -> int:
    """Return the smallest LTX-valid ``8k + 1`` count covering duration."""
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be greater than zero")
    latent_intervals = math.ceil(max(0.0, duration * frame_rate - 1.0) / 8.0)
    return max(9, latent_intervals * 8 + 1)


def build_segment_plan(
    audio_duration: float,
    *,
    max_segment_seconds: float,
    frame_rate: float,
) -> list[SegmentPlan]:
    """Split a timeline into short clips without cumulative audio drift."""
    if audio_duration <= 0:
        raise ValueError("audio_duration must be greater than zero")
    if max_segment_seconds <= 0:
        raise ValueError("max_segment_seconds must be greater than zero")

    # Split the source timeline evenly.  Filling max-sized shots greedily can
    # leave a tiny final shot (for example 16.5s -> 8s + 8s + 0.5s).  Very
    # short audio conditionings are unreliable and often produce a frozen
    # mouth, so distribute the same audio over equally useful shots instead.
    segment_count = max(1, math.ceil(audio_duration / max_segment_seconds))
    target_duration = audio_duration / segment_count
    plans: list[SegmentPlan] = []
    start = 0.0

    for index in range(segment_count):
        # Compute the final boundary from the total duration rather than by
        # repeated addition, avoiding cumulative floating-point drift.
        end = audio_duration if index == segment_count - 1 else target_duration * (index + 1)
        source_duration = end - start
        frames = snap_frame_count(source_duration, frame_rate)
        output_duration = frames / frame_rate
        plans.append(
            SegmentPlan(
                index=index,
                start_time=start,
                source_duration=source_duration,
                num_frames=frames,
                output_duration=output_duration,
            )
        )
        start = end

    return plans


def stable_config_hash(value: Any, length: int = 12) -> str:
    """Return a short deterministic digest for resumable segment filenames."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def serialise_plan(plans: list[SegmentPlan]) -> list[dict[str, Any]]:
    return [asdict(plan) for plan in plans]


def concat_file_line(path: str | Path) -> str:
    """Quote a path for ffmpeg's concat demuxer."""
    absolute = str(Path(path).expanduser().resolve())
    return "file '" + absolute.replace("'", "'\\''") + "'"


__all__ = [
    "SegmentPlan",
    "build_segment_plan",
    "concat_file_line",
    "serialise_plan",
    "snap_frame_count",
    "stable_config_hash",
]
