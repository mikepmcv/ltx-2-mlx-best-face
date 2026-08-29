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

    standard_frames = snap_frame_count(max_segment_seconds, frame_rate)
    standard_duration = standard_frames / frame_rate
    plans: list[SegmentPlan] = []
    start = 0.0
    epsilon = 1e-7

    while start < audio_duration - epsilon:
        remaining = audio_duration - start
        source_duration = min(standard_duration, remaining)
        frames = (
            standard_frames
            if remaining >= standard_duration - epsilon
            else snap_frame_count(source_duration, frame_rate)
        )
        output_duration = frames / frame_rate
        plans.append(
            SegmentPlan(
                index=len(plans),
                start_time=start,
                source_duration=source_duration,
                num_frames=frames,
                output_duration=output_duration,
            )
        )
        start += source_duration

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
