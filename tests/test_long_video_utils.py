from pathlib import Path

import pytest

from ltx_pipelines_mlx.long_video_utils import (
    build_segment_plan,
    concat_file_line,
    snap_frame_count,
    stable_config_hash,
)


def test_snap_frame_count_is_valid_and_covers_duration():
    frames = snap_frame_count(8.0, 24.0)
    assert frames == 193
    assert (frames - 1) % 8 == 0
    assert frames / 24.0 >= 8.0


def test_plan_has_no_cumulative_source_audio_gap():
    plans = build_segment_plan(600.0, max_segment_seconds=8.0, frame_rate=24.0)
    assert len(plans) == 75
    assert sum(plan.source_duration for plan in plans) == pytest.approx(600.0)
    for previous, current in zip(plans, plans[1:]):
        assert current.start_time == pytest.approx(
            previous.start_time + previous.source_duration
        )
    assert all((plan.num_frames - 1) % 8 == 0 for plan in plans)


def test_segments_are_balanced_and_only_frame_padded():
    plans = build_segment_plan(10.0, max_segment_seconds=8.0, frame_rate=24.0)
    assert len(plans) == 2
    assert plans[0].source_duration == pytest.approx(5.0)
    assert plans[1].source_duration == pytest.approx(5.0)
    assert all(plan.output_duration >= plan.source_duration for plan in plans)
    assert sum(plan.source_duration for plan in plans) == pytest.approx(10.0)


def test_short_tail_is_balanced_across_segments():
    plans = build_segment_plan(16.5, max_segment_seconds=8.0, frame_rate=24.0)
    assert len(plans) == 3
    assert [plan.source_duration for plan in plans] == pytest.approx([5.5, 5.5, 5.5])
    assert all(plan.source_duration <= 8.0 for plan in plans)


def test_config_hash_is_order_independent():
    assert stable_config_hash({"a": 1, "b": 2}) == stable_config_hash(
        {"b": 2, "a": 1}
    )


def test_concat_quote_escapes_apostrophes(tmp_path: Path):
    line = concat_file_line(tmp_path / "Mike's clip.mp4")
    assert line.startswith("file '")
    assert "'\\''" in line


@pytest.mark.parametrize("duration,rate", [(0, 24), (-1, 24), (1, 0)])
def test_invalid_frame_inputs(duration: float, rate: float):
    with pytest.raises(ValueError):
        snap_frame_count(duration, rate)
