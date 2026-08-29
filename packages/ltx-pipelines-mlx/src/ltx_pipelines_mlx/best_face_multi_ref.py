"""Multi-reference Best Face ID for LTX-2.3 on Apple Silicon.

This experimental variant layers multiple identity images into the existing
Best Face exact/UGC-fast pipeline without changing single-reference behaviour.

Each reference is VAE-encoded independently, then the clean identity tokens and
their RoPE positions are concatenated into one reference-conditioning segment.
All references intentionally share the same source-phase ID because they
describe the same character.

Example:

    uv run python -m ltx_pipelines_mlx.best_face_multi_ref \
      --character-sheet \
      --reference face-closeup.png \
      --reference character-sheet.png \
      --reference smile-teeth.png \
      --prompt "ref_t2v: A woman speaks naturally to the camera." \
      --ugc-fast \
      --frames 145 --frame-rate 24 \
      -H 1024 -W 576 \
      -o multi-ref.mp4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from ltx_core_mlx.conditioning.source_phase import SourcePhaseBlock
from ltx_core_mlx.conditioning.types.reference_video_cond import (
    VideoConditionByReferenceLatent,
)
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_video_positions

from .best_face import (
    DEFAULT_GEMMA,
    DEFAULT_MODEL,
    OFFICIAL_BASE_FACE_STRENGTH,
    OFFICIAL_SPATIAL_UPSCALER_FILE,
    _default_best_face_spec,
    _resolve_generation_settings,
)
from .best_face_exact import (
    OFFICIAL_DISTILLED_LORA_STRENGTH,
    BestFaceExactPipeline,
)
from .utils.media_io import load_image_and_preprocess


class ReferenceSet(str):
    """String-compatible primary reference carrying additional identity images.

    The parent Best Face implementation expects ``reference`` to be a string in
    a few non-conditioning places (notably metadata). Making the set a ``str``
    subclass preserves that behaviour: legacy code sees the first image, while
    this module's conditioning override can access all images.
    """

    references: tuple[str, ...]

    def __new__(cls, references: Sequence[str]) -> "ReferenceSet":
        values = tuple(str(value) for value in references if str(value))
        if not values:
            raise ValueError("At least one identity reference is required")
        obj = str.__new__(cls, values[0])
        obj.references = values
        return obj


def _reference_paths(reference: str | ReferenceSet) -> tuple[str, ...]:
    if isinstance(reference, ReferenceSet):
        return reference.references
    return (str(reference),)


class BestFaceMultiRefExactPipeline(BestFaceExactPipeline):
    """Best Face exact/UGC-fast pipeline with equal-weight multi-image identity.

    Single-reference calls delegate to the parent implementation for parity.
    Multiple references are encoded independently and concatenated as clean
    reference tokens. This intentionally changes only identity conditioning;
    denoising schedules, LoRAs, source-phase behaviour, upscaling, CFG++ and
    UGC-fast settings remain inherited from ``BestFaceExactPipeline``.
    """

    def _build_identity_conditioning(
        self,
        *,
        reference: str | ReferenceSet,
        resize_mode: str,
        target_h: int,
        target_w: int,
        reference_scale: float,
        frame_rate: float,
        num_generation_tokens: int,
        source_id: float,
        phase_scale: float,
        crf: int,
    ) -> tuple[VideoConditionByReferenceLatent, SourcePhaseBlock]:
        references = _reference_paths(reference)

        # Preserve byte-for-byte parent behaviour for the existing workflow.
        if len(references) == 1:
            return super()._build_identity_conditioning(
                reference=references[0],
                resize_mode=resize_mode,
                target_h=target_h,
                target_w=target_w,
                reference_scale=reference_scale,
                frame_rate=frame_rate,
                num_generation_tokens=num_generation_tokens,
                source_id=source_id,
                phase_scale=phase_scale,
                crf=crf,
            )

        assert self.vae_encoder is not None

        token_sets: list[mx.array] = []
        position_sets: list[mx.array] = []

        for path in references:
            ref_h, ref_w, position_scale_h, position_scale_w = (
                self._scaled_reference_geometry(
                    path,
                    resize_mode=resize_mode,
                    target_h=target_h,
                    target_w=target_w,
                    reference_scale=reference_scale,
                )
            )

            ref_pixels = load_image_and_preprocess(path, ref_h, ref_w, crf=crf)
            ref_pixels = ref_pixels[:, :, None, :, :]  # BCHW -> BCFHW
            ref_latent = self.vae_encoder.encode(ref_pixels)

            ref_f = int(ref_latent.shape[2])
            ref_h_lat = int(ref_latent.shape[3])
            ref_w_lat = int(ref_latent.shape[4])
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(
                ref_latent.shape[0], -1, ref_latent.shape[1]
            )
            ref_positions = compute_video_positions(
                ref_f,
                ref_h_lat,
                ref_w_lat,
                frame_rate=frame_rate,
            )
            ref_positions = mx.concatenate(
                [
                    ref_positions[..., :1],
                    ref_positions[..., 1:2] * position_scale_h,
                    ref_positions[..., 2:3] * position_scale_w,
                ],
                axis=-1,
            )
            mx.eval(ref_tokens, ref_positions)
            token_sets.append(ref_tokens)
            position_sets.append(ref_positions)

        # Treat every image as another observation of the same identity source.
        # Keeping the same position grid is deliberate: a close-up, character
        # sheet and smile reference all describe the character at source time 0.
        all_tokens = mx.concatenate(token_sets, axis=1)
        all_positions = mx.concatenate(position_sets, axis=1)
        mx.eval(all_tokens, all_positions)

        condition = VideoConditionByReferenceLatent(
            reference_latent=all_tokens,
            reference_positions=all_positions,
            downscale_factor=1,
            strength=1.0,
        )
        phase_block = SourcePhaseBlock(
            start=num_generation_tokens,
            length=int(all_tokens.shape[1]),
            segment_value=float(source_id) * float(phase_scale),
        )
        return condition, phase_block


def _enrich_metadata(saved_path: str, references: Sequence[str]) -> None:
    """Record all identity inputs while retaining the parent's primary field."""
    metadata_path = Path(f"{saved_path}.json")
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    metadata["references"] = [
        str(Path(path).expanduser().resolve()) for path in references
    ]
    metadata["reference_count"] = len(references)
    metadata["reference_strategy"] = (
        "single_reference_parent_parity"
        if len(references) == 1
        else "concatenated_identity_tokens"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.best_face_multi_ref",
        description=(
            "Native MLX Best Face ID with one or more identity references "
            "(LTX-2.3 exact/UGC-fast recipe)"
        ),
    )
    parser.add_argument(
        "--reference",
        "-i",
        action="append",
        required=True,
        help=(
            "Identity reference image. Repeat for multiple images; order matters "
            "and the first image remains the primary metadata reference."
        ),
    )
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--best-face-lora", default=None)
    parser.add_argument("--best-face-strength", type=float, default=1.0)
    parser.add_argument("--base-face-lora", default=None)
    parser.add_argument(
        "--base-face-strength",
        type=float,
        default=OFFICIAL_BASE_FACE_STRENGTH,
    )
    parser.add_argument("--spatial-upscaler", default=None)
    parser.add_argument("--distilled-lora", default=None)
    parser.add_argument(
        "--distilled-lora-strength",
        type=float,
        default=OFFICIAL_DISTILLED_LORA_STRENGTH,
    )
    parser.add_argument(
        "--extra-lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=[],
    )
    parser.add_argument("--character-sheet", action="store_true")
    parser.add_argument(
        "--resize-mode",
        choices=["match_target", "native_resolution"],
        default=None,
    )
    parser.add_argument("--reference-scale", type=float, default=1.0)
    parser.add_argument("--stage1-reference-scale", type=float, default=None)
    parser.add_argument("--stage2-reference-scale", type=float, default=None)
    parser.add_argument("--fast-refine", action="store_true")

    speed_presets = parser.add_mutually_exclusive_group()
    speed_presets.add_argument("--ugc-fast", action="store_true")
    speed_presets.add_argument("--ugc-ultrafast", action="store_true")

    parser.add_argument("--height", "-H", type=int, default=576)
    parser.add_argument("--width", "-W", type=int, default=768)
    parser.add_argument("--frames", "-f", type=int, default=49)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--source-id", type=float, default=2.0)
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--reference-crf", type=int, default=0)
    parser.add_argument("--seed", "-s", type=int, default=-1)
    parser.add_argument("--no-low-memory", action="store_true")
    parser.add_argument("--first-frame", default=None)
    parser.add_argument("--last-frame", default=None)
    parser.add_argument("--first-frame-strength", type=float, default=1.0)
    parser.add_argument("--last-frame-strength", type=float, default=1.0)
    parser.add_argument(
        "--first-frame-mode",
        choices=["appearance", "layout"],
        default="appearance",
    )
    parser.add_argument(
        "--last-frame-mode",
        choices=["appearance", "layout"],
        default="appearance",
    )
    parser.add_argument("--keyframe-layout-blur", type=float, default=32.0)

    args = parser.parse_args()
    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    references = ReferenceSet(args.reference)
    best_face_lora = args.best_face_lora or _default_best_face_spec(
        args.character_sheet
    )
    base_face_lora = args.base_face_lora
    if args.character_sheet and base_face_lora is None:
        base_face_lora = _default_best_face_spec(False)

    spatial_upscaler = args.spatial_upscaler
    if args.character_sheet and spatial_upscaler is None:
        spatial_upscaler = OFFICIAL_SPATIAL_UPSCALER_FILE

    resize_mode = args.resize_mode or (
        "native_resolution" if args.character_sheet else "match_target"
    )
    effective_settings = _resolve_generation_settings(
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        reference_scale=args.reference_scale,
        stage1_reference_scale=args.stage1_reference_scale,
        stage2_reference_scale=args.stage2_reference_scale,
        fast_refine=args.fast_refine,
        ugc_fast=args.ugc_fast,
        ugc_ultrafast=args.ugc_ultrafast,
    )
    extra_loras = [(path, float(strength)) for path, strength in args.extra_lora]

    pipe = BestFaceMultiRefExactPipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        best_face_lora=best_face_lora,
        best_face_strength=args.best_face_strength,
        base_face_lora=base_face_lora,
        base_face_strength=args.base_face_strength,
        spatial_upscaler=spatial_upscaler,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength=args.distilled_lora_strength,
        extra_loras=extra_loras,
        low_memory=not args.no_low_memory,
    )

    t0 = time.time()
    print("Best Face MLX multi-reference (exact/UGC-fast recipe)")
    print(f"  references: {len(references.references)}")
    for index, path in enumerate(references.references, start=1):
        print(f"    {index}: {path}")
    print(f"  resize mode: {resize_mode}")
    print(
        "  stages: "
        f"{effective_settings[0] or 8}+{effective_settings[1] or 3}, "
        f"reference scales {effective_settings[2]:g}/{effective_settings[3]:g}, "
        f"{'single-pass' if effective_settings[4] else 'CFG++'} refine"
    )
    print(f"  seed: {args.seed}")

    saved_path = pipe.generate_and_save_best_face(
        prompt=args.prompt,
        reference=references,
        output_path=args.output,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        resize_mode=resize_mode,
        reference_scale=args.reference_scale,
        stage1_reference_scale=args.stage1_reference_scale,
        stage2_reference_scale=args.stage2_reference_scale,
        fast_refine=args.fast_refine,
        ugc_fast=args.ugc_fast,
        ugc_ultrafast=args.ugc_ultrafast,
        source_id=args.source_id,
        phase_scale=args.phase_scale,
        reference_crf=args.reference_crf,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        first_frame_strength=args.first_frame_strength,
        last_frame_strength=args.last_frame_strength,
        first_frame_mode=args.first_frame_mode,
        last_frame_mode=args.last_frame_mode,
        keyframe_layout_blur=args.keyframe_layout_blur,
    )
    _enrich_metadata(saved_path, references.references)
    print(f"Saved {Path(saved_path)} in {time.time() - t0:.1f}s")

    if pipe.low_memory:
        pipe.dit = None
        pipe.image_conditioner.free()
        pipe.upsampler = None
        aggressive_cleanup()


if __name__ == "__main__":
    main()


__all__ = [
    "BestFaceMultiRefExactPipeline",
    "ReferenceSet",
]
