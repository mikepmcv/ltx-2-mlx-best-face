"""Resumable Best Face + locked-TTS orchestration for long talking videos.

The model still renders short shots. This module keeps identity and scene
conditioning fresh on every shot, strips every generated audio stream, and
muxes only PCM copied from the supplied TTS recording into the final video.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ltx_core_mlx.components.patchifiers import snap_output_dimensions
from ltx_core_mlx.utils.ffmpeg import find_ffmpeg

from .best_face import (
    DEFAULT_GEMMA,
    DEFAULT_MODEL,
    OFFICIAL_BASE_FACE_STRENGTH,
    OFFICIAL_SPATIAL_UPSCALER_FILE,
    BestFacePipeline,
    _default_best_face_spec,
)
from .long_video_utils import (
    SegmentPlan,
    build_segment_plan,
    concat_file_line,
    serialise_plan,
    stable_config_hash,
)


SCENE_LOCK = (
    "The camera, background geometry, furniture, exposure, white balance, "
    "light direction, light intensity, and color temperature remain constant. "
    "No camera drift, zoom, exposure breathing, relighting, time-of-day change, "
    "moving furniture, or morphing background."
)


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{label} failed:\n{detail[-4000:]}")


def _find_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required alongside ffmpeg")
    return ffprobe


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            _find_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not read audio duration: {result.stderr.strip()}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe returned an invalid audio duration") from exc
    if duration <= 0:
        raise RuntimeError("Input audio is empty")
    return duration


def _extract_padded_pcm(
    *,
    source: Path,
    plan: SegmentPlan,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-ss",
            f"{plan.start_time:.9f}",
            "-t",
            f"{plan.source_duration:.9f}",
            "-i",
            str(source),
            "-af",
            f"apad=pad_dur={plan.output_duration:.9f},atrim=0:{plan.output_duration:.9f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        label=f"extracting audio segment {plan.index}",
    )


def _normalise_video(source: Path, destination: Path, *, frame_rate: float) -> None:
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-r",
            f"{frame_rate:g}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        label="stripping model audio",
    )


def _lock_background(
    *,
    generated: Path,
    background: Path,
    foreground_mask: Path,
    destination: Path,
    width: int,
    height: int,
    frame_rate: float,
    mask_feather: float,
) -> None:
    blur = f",gblur=sigma={mask_feather:g}" if mask_feather > 0 else ""
    graph = (
        f"[1:v]scale={width}:{height},format=yuv420p[bg];"
        f"[2:v]scale={width}:{height},format=gray{blur}[mask];"
        "[bg][0:v][mask]maskedmerge,format=yuv420p[outv]"
    )
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-i",
            str(generated),
            "-loop",
            "1",
            "-i",
            str(background),
            "-loop",
            "1",
            "-i",
            str(foreground_mask),
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-an",
            "-r",
            f"{frame_rate:g}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(destination),
        ],
        label="locking the background",
    )


def _write_concat_list(paths: list[Path], destination: Path) -> None:
    destination.write_text(
        "\n".join(concat_file_line(path) for path in paths) + "\n",
        encoding="utf-8",
    )


def _assemble(
    *,
    visual_segments: list[Path],
    audio_segments: list[Path],
    work_dir: Path,
    output: Path,
) -> None:
    video_list = work_dir / "video-concat.txt"
    audio_list = work_dir / "audio-concat.txt"
    silent_video = work_dir / "assembled-video.mp4"
    assembled_audio = work_dir / "assembled-original-tts.wav"
    _write_concat_list(visual_segments, video_list)
    _write_concat_list(audio_segments, audio_list)

    _run(
        [
            find_ffmpeg(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(video_list),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            str(silent_video),
        ],
        label="concatenating video segments",
    )
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(audio_list),
            "-c:a",
            "pcm_s16le",
            str(assembled_audio),
        ],
        label="concatenating original TTS segments",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_ffmpeg(),
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(assembled_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ],
        label="muxing the untouched TTS track",
    )


def _load_overrides(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("segments", [])
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("Shot manifest must be a list or contain a segments list")
    overrides: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            raise ValueError("Each shot override requires an integer index")
        index = int(row["index"])
        if index in overrides:
            raise ValueError(f"Duplicate shot override index {index}")
        overrides[index] = row
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.long_video",
        description="Create resumable long Best Face videos from a supplied TTS recording.",
    )
    parser.add_argument("--audio", required=True, help="Complete TTS/narration audio")
    parser.add_argument("--reference", required=True, help="Best Face identity reference")
    parser.add_argument("--first-frame", required=True, help="Master presenter + scene frame")
    parser.add_argument("--prompt", required=True, help="Presenter/action prompt")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--foreground-mask", default=None, help="White=generated presenter; black=locked scene")
    parser.add_argument("--mask-feather", type=float, default=6.0)
    parser.add_argument("--shot-manifest", default=None, help="Optional JSON per-segment prompt/frame/mask overrides")
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--limit-seconds", type=float, default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--best-face-lora", default=None)
    parser.add_argument("--best-face-strength", type=float, default=1.0)
    parser.add_argument("--base-face-lora", default=None)
    parser.add_argument("--base-face-strength", type=float, default=OFFICIAL_BASE_FACE_STRENGTH)
    parser.add_argument("--spatial-upscaler", default=None)
    parser.add_argument("--extra-lora", action="append", nargs=2, default=[], metavar=("PATH", "STRENGTH"))
    parser.add_argument("--character-sheet", action="store_true")
    parser.add_argument("--height", "-H", type=int, default=1024)
    parser.add_argument("--width", "-W", type=int, default=576)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--quality", choices=["standard", "fast", "ultrafast"], default="fast")
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--first-frame-strength", type=float, default=1.0)
    parser.add_argument("--keyframe-layout-blur", type=float, default=32.0)
    parser.add_argument("--low-memory", action="store_true", help="Reload large components per shot; slower but uses less RAM")
    parser.add_argument("--no-scene-lock-prompt", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audio = Path(args.audio).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    first_frame = Path(args.first_frame).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    foreground_mask = Path(args.foreground_mask).expanduser().resolve() if args.foreground_mask else None
    shot_manifest = Path(args.shot_manifest).expanduser().resolve() if args.shot_manifest else None

    for label, path in (("audio", audio), ("reference", reference), ("first frame", first_frame)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if foreground_mask is not None and not foreground_mask.is_file():
        raise FileNotFoundError(f"foreground mask does not exist: {foreground_mask}")

    duration = _probe_duration(audio)
    if args.limit_seconds is not None:
        if args.limit_seconds <= 0:
            raise ValueError("limit-seconds must be greater than zero")
        duration = min(duration, args.limit_seconds)
    args.height, args.width = snap_output_dimensions(
        args.height,
        args.width,
        two_stage=True,
    )
    plans = build_segment_plan(
        duration,
        max_segment_seconds=args.segment_seconds,
        frame_rate=args.frame_rate,
    )
    overrides = _load_overrides(shot_manifest)

    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output.with_suffix(output.suffix + ".work")
    )
    audio_dir = work_dir / "audio"
    raw_dir = work_dir / "raw"
    visual_dir = work_dir / "visual"
    for directory in (audio_dir, raw_dir, visual_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - len(plans) - 1)

    lora_spec = args.best_face_lora or _default_best_face_spec(args.character_sheet)
    base_face_lora = args.base_face_lora
    if args.character_sheet and base_face_lora is None:
        base_face_lora = _default_best_face_spec(False)
    spatial_upscaler = args.spatial_upscaler
    if args.character_sheet and spatial_upscaler is None:
        spatial_upscaler = OFFICIAL_SPATIAL_UPSCALER_FILE
    resize_mode = "native_resolution" if args.character_sheet else "match_target"
    extra_loras = [(path, float(strength)) for path, strength in args.extra_lora]

    def fingerprint(path: Path | None) -> dict | None:
        if path is None:
            return None
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    run_config = {
        "audio": fingerprint(audio),
        "reference": fingerprint(reference),
        "first_frame": fingerprint(first_frame),
        "foreground_mask": fingerprint(foreground_mask),
        "shot_manifest": fingerprint(shot_manifest),
        "prompt": args.prompt,
        "model": args.model,
        "gemma": args.gemma,
        "lora": lora_spec,
        "base_face_lora": base_face_lora,
        "extra_loras": extra_loras,
        "height": args.height,
        "width": args.width,
        "frame_rate": args.frame_rate,
        "quality": args.quality,
        "reference_scale": args.reference_scale,
        "first_frame_strength": args.first_frame_strength,
        "mask_feather": args.mask_feather,
        "seed": args.seed,
        "plans": serialise_plan(plans),
    }
    run_hash = stable_config_hash(run_config)
    (work_dir / f"run-{run_hash}.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Best Face long video: {duration:.1f}s, {len(plans)} shots, run {run_hash}")
    print("Original TTS will be locked for lip-sync; all model-generated audio will be discarded.")
    if foreground_mask is None:
        print("Background: anchored by the master first frame (supply --foreground-mask for pixel locking).")
    else:
        print("Background: pixel-locked outside the supplied foreground mask.")

    pipe: BestFacePipeline | None = None
    visuals: list[Path] = []
    audio_chunks: list[Path] = []
    started = time.time()

    for plan in plans:
        override = overrides.get(plan.index, {})
        segment_prompt = str(override.get("prompt", args.prompt)).strip()
        if not args.no_scene_lock_prompt:
            segment_prompt = f"{segment_prompt} {SCENE_LOCK}"
        segment_frame = Path(override.get("first_frame", first_frame)).expanduser().resolve()
        segment_mask_raw = override.get("foreground_mask", foreground_mask)
        segment_mask = Path(segment_mask_raw).expanduser().resolve() if segment_mask_raw else None
        segment_seed = int(override.get("seed", args.seed + plan.index))
        if not segment_frame.is_file():
            raise FileNotFoundError(f"segment {plan.index} first frame does not exist: {segment_frame}")
        if segment_mask is not None and not segment_mask.is_file():
            raise FileNotFoundError(f"segment {plan.index} mask does not exist: {segment_mask}")

        segment_config = {
            **asdict(plan),
            "prompt": segment_prompt,
            "first_frame": str(segment_frame),
            "foreground_mask": str(segment_mask) if segment_mask else None,
            "seed": segment_seed,
            "run": run_hash,
        }
        segment_hash = stable_config_hash(segment_config, length=10)
        stem = f"segment-{plan.index:04d}-{segment_hash}"
        chunk_audio = audio_dir / f"{stem}.wav"
        raw_video = raw_dir / f"{stem}.mp4"
        visual_video = visual_dir / f"{stem}.mp4"

        if not chunk_audio.is_file():
            _extract_padded_pcm(source=audio, plan=plan, destination=chunk_audio)

        if not raw_video.is_file():
            if pipe is None:
                pipe = BestFacePipeline(
                    model_dir=args.model,
                    gemma_model_id=args.gemma,
                    best_face_lora=lora_spec,
                    best_face_strength=args.best_face_strength,
                    base_face_lora=base_face_lora,
                    base_face_strength=args.base_face_strength,
                    spatial_upscaler=spatial_upscaler,
                    extra_loras=extra_loras,
                    low_memory=args.low_memory,
                )
            print(f"[{plan.index + 1}/{len(plans)}] generating {plan.output_duration:.2f}s")
            pipe.generate_and_save_best_face(
                prompt=segment_prompt,
                reference=str(reference),
                output_path=str(raw_video),
                height=args.height,
                width=args.width,
                num_frames=plan.num_frames,
                frame_rate=args.frame_rate,
                seed=segment_seed,
                resize_mode=resize_mode,
                reference_scale=args.reference_scale,
                ugc_fast=args.quality == "fast",
                ugc_ultrafast=args.quality == "ultrafast",
                first_frame=str(segment_frame),
                first_frame_strength=args.first_frame_strength,
                first_frame_mode="appearance",
                keyframe_layout_blur=args.keyframe_layout_blur,
                audio_path=str(chunk_audio),
                audio_start_time=0.0,
                audio_max_duration=plan.output_duration,
            )
        else:
            print(f"[{plan.index + 1}/{len(plans)}] resuming existing shot")

        if not visual_video.is_file():
            if segment_mask is None:
                _normalise_video(raw_video, visual_video, frame_rate=args.frame_rate)
            else:
                _lock_background(
                    generated=raw_video,
                    background=segment_frame,
                    foreground_mask=segment_mask,
                    destination=visual_video,
                    width=args.width,
                    height=args.height,
                    frame_rate=args.frame_rate,
                    mask_feather=args.mask_feather,
                )
        visuals.append(visual_video)
        audio_chunks.append(chunk_audio)

    _assemble(
        visual_segments=visuals,
        audio_segments=audio_chunks,
        work_dir=work_dir,
        output=output,
    )
    summary = {
        **run_config,
        "output": str(output),
        "work_dir": str(work_dir),
        "elapsed_seconds": time.time() - started,
        "segments": [str(path) for path in visuals],
        "audio_mode": "locked_original_tts_only",
        "background_mode": "masked_pixel_lock" if foreground_mask else "first_frame_anchor",
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()


__all__ = ["main"]
