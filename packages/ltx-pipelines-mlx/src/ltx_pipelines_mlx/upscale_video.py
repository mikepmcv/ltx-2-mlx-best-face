"""Temporally stable delivery upscale for completed LTX videos.

This is deliberately separate from generation. It performs one high-quality
FFmpeg resize, applies optional mild sharpening, and copies the existing audio
stream without re-encoding it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


def build_video_filter(*, width: int, height: int, fit: str, sharpen: float) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    if fit not in {"crop", "pad", "stretch"}:
        raise ValueError(f"unsupported fit mode: {fit}")
    if not 0.0 <= sharpen <= 1.5:
        raise ValueError("sharpen must be between 0.0 and 1.5")

    if fit == "crop":
        filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={width}:{height}",
        ]
    elif fit == "pad":
        filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        ]
    else:
        filters = [f"scale={width}:{height}:flags=lanczos"]

    if sharpen > 0:
        filters.append(f"unsharp=5:5:{sharpen:.3f}:5:5:0.0")
    return ",".join(filters)


def build_command(
    *,
    source: Path,
    destination: Path,
    width: int,
    height: int,
    fit: str,
    sharpen: float,
    crf: int,
    preset: str,
) -> list[str]:
    return [
        find_ffmpeg(),
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        build_video_filter(width=width, height=height, fit=fit, sharpen=sharpen),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.upscale_video",
        description="Upscale a completed LTX video without altering its timing or audio.",
    )
    parser.add_argument("--input", "-i", required=True, help="Completed source video")
    parser.add_argument("--output", "-o", required=True, help="Upscaled MP4 destination")
    parser.add_argument("--width", "-W", type=int, default=1080)
    parser.add_argument("--height", "-H", type=int, default=1920)
    parser.add_argument(
        "--fit",
        choices=["crop", "pad", "stretch"],
        default="crop",
        help="crop fills the exact output frame; pad preserves every source pixel; stretch ignores aspect ratio",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.20,
        help="Mild luma sharpening from 0.0 to 1.5 (default: 0.20)",
    )
    parser.add_argument("--crf", type=int, default=16, help="H.264 quality; lower is higher quality (default: 16)")
    parser.add_argument(
        "--preset",
        choices=["medium", "slow", "slower"],
        default="slow",
        help="Encoding efficiency/speed tradeoff (default: slow)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"input video does not exist: {source}")
    if source == destination:
        raise ValueError("input and output must be different files")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        source=source,
        destination=destination,
        width=args.width,
        height=args.height,
        fit=args.fit,
        sharpen=args.sharpen,
        crf=args.crf,
        preset=args.preset,
    )
    print(
        f"Upscaling {source.name} to {args.width}x{args.height} "
        f"(fit={args.fit}, sharpen={args.sharpen:.2f}, audio=copy)"
    )
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg upscale failed with exit code {result.returncode}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
