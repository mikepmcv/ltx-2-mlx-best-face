from pathlib import Path

import pytest

from ltx_pipelines_mlx.upscale_video import build_command, build_video_filter


def test_crop_filter_fills_exact_portrait_frame() -> None:
    result = build_video_filter(width=1080, height=1920, fit="crop", sharpen=0.2)

    assert "force_original_aspect_ratio=increase" in result
    assert "crop=1080:1920" in result
    assert "unsharp=5:5:0.200" in result


def test_zero_sharpen_omits_unsharp_filter() -> None:
    result = build_video_filter(width=1920, height=1080, fit="pad", sharpen=0.0)

    assert "force_original_aspect_ratio=decrease" in result
    assert "pad=1920:1080" in result
    assert "unsharp" not in result


def test_invalid_sharpen_is_rejected() -> None:
    with pytest.raises(ValueError, match="sharpen"):
        build_video_filter(width=1080, height=1920, fit="crop", sharpen=2.0)


def test_command_copies_audio_and_preserves_frame_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ltx_pipelines_mlx.upscale_video.find_ffmpeg", lambda: "/ffmpeg")

    command = build_command(
        source=Path("input.mp4"),
        destination=Path("output.mp4"),
        width=1080,
        height=1920,
        fit="crop",
        sharpen=0.2,
        crf=16,
        preset="slow",
    )

    assert command[0] == "/ffmpeg"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "-r" not in command
    assert command[-1] == "output.mp4"
