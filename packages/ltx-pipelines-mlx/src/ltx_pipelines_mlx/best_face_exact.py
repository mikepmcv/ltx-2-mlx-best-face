"""Parity-first Best Face ID pipeline for LTX-2.3 on Apple Silicon.

This variant mirrors the Best Face author's published fast demo recipe more
closely than :mod:`ltx_pipelines_mlx.best_face`:

- LTX-2.3 dev transformer;
- official LTX-2.3 distilled-1.1 LoRA at strength 0.6;
- base Best Face ID LoRA at strength 0.2 plus character-sheet LoRA at 1.0;
- Euler stage 1 and Euler ancestral CFG++ stage 2 at CFG 1;
- native MLX identity-overlap/source-phase conditioning.

The sibling ``best_face`` module uses the standalone distilled checkpoint as a
faster/convenient shortcut. Use this module first when validating visual parity.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from ltx_core_mlx.utils.memory import aggressive_cleanup

from .best_face import (
    DEFAULT_GEMMA,
    DEFAULT_MODEL,
    OFFICIAL_BASE_FACE_STRENGTH,
    OFFICIAL_SPATIAL_UPSCALER_FILE,
    BestFacePipeline,
    _default_best_face_spec,
    _resolve_adapter_spec,
    _resolve_generation_settings,
)

OFFICIAL_DISTILLED_LORA_STRENGTH = 0.6
OFFICIAL_STAGE2_SIGMAS = [0.85, 0.725, 0.421875, 0.0]
OFFICIAL_NEGATIVE_PROMPT = (
    "pc game, console game, video game, cartoon, childish, ugly, artifacts, "
    "low resolution, blurry, jagged edges"
)


class BestFaceExactPipeline(BestFacePipeline):
    """Best Face with dev + official distilled LoRA, matching the author recipe."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        *,
        gemma_model_id: str = DEFAULT_GEMMA,
        best_face_lora: str,
        best_face_strength: float = 1.0,
        base_face_lora: str | None = None,
        base_face_strength: float = OFFICIAL_BASE_FACE_STRENGTH,
        spatial_upscaler: str | None = None,
        distilled_lora: str | None = None,
        distilled_lora_strength: float = OFFICIAL_DISTILLED_LORA_STRENGTH,
        extra_loras: list[tuple[str, float]] | None = None,
        low_memory: bool = True,
    ):
        super().__init__(
            model_dir=model_dir,
            gemma_model_id=gemma_model_id,
            best_face_lora=best_face_lora,
            best_face_strength=best_face_strength,
            base_face_lora=base_face_lora,
            base_face_strength=base_face_strength,
            spatial_upscaler=spatial_upscaler,
            extra_loras=extra_loras,
            low_memory=low_memory,
        )

        if distilled_lora is None:
            distilled_lora_path = self._resolve_safetensors(
                self.model_dir,
                "ltx-2.3-22b-distilled-lora-384",
            )
            if not distilled_lora_path.exists():
                raise FileNotFoundError(
                    "Official LTX-2.3 distilled LoRA was not found in the model directory. "
                    "Expected ltx-2.3-22b-distilled-lora-384[-1.1].safetensors."
                )
            distilled_lora = str(distilled_lora_path)
        else:
            distilled_lora = _resolve_adapter_spec(distilled_lora)

        # The published recipe fuses the official distillation adapter first,
        # then the identity adapter. Keep that ordering explicit.
        self._pending_loras.insert(0, (distilled_lora, float(distilled_lora_strength)))
        self._best_face_cfg_pp = True
        self._best_face_stage2_sigmas = OFFICIAL_STAGE2_SIGMAS
        self._best_face_negative_prompt = OFFICIAL_NEGATIVE_PROMPT

    def load(self) -> None:
        """Load dev transformer + both LoRAs, then VAE encoder and upsampler."""
        if self._loaded:
            return

        if self.dit is None:
            transformer_path = self.model_dir / "transformer-dev.safetensors"
            if not transformer_path.exists():
                raise FileNotFoundError(
                    f"Dev transformer not found: {transformer_path}. "
                    "Use a model repo that includes LTX-2.3 dev weights."
                )
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)

        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.best_face_exact",
        description=(
            "Native MLX Best Face ID using LTX-2.3 dev + official distilled-1.1 LoRA + Best Face"
        ),
    )
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--reference", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--best-face-lora", default=None)
    parser.add_argument("--best-face-strength", type=float, default=1.0)
    parser.add_argument(
        "--base-face-lora",
        default=None,
        help="Override the base Face-ID LoRA stacked with --character-sheet.",
    )
    parser.add_argument(
        "--base-face-strength", type=float, default=OFFICIAL_BASE_FACE_STRENGTH
    )
    parser.add_argument(
        "--spatial-upscaler",
        default=None,
        help="Select an exact converted MLX spatial-upscaler file.",
    )
    parser.add_argument(
        "--distilled-lora",
        default=None,
        help="Override the official distilled LoRA; accepts local path or repo::filename.",
    )
    parser.add_argument(
        "--distilled-lora-strength", type=float, default=OFFICIAL_DISTILLED_LORA_STRENGTH
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
    parser.add_argument(
        "--reference-scale",
        type=float,
        default=1.0,
        help="Downscale the reference before VAE encoding while preserving its H/W position span.",
    )
    parser.add_argument(
        "--stage1-reference-scale",
        type=float,
        default=None,
        help="Override reference scale for half-resolution Stage 1 only.",
    )
    parser.add_argument(
        "--stage2-reference-scale",
        type=float,
        default=None,
        help="Override reference scale for full-resolution Stage 2 only.",
    )
    parser.add_argument(
        "--fast-refine",
        action="store_true",
        help="Use single-pass Stage 2 refinement instead of official two-pass CFG++.",
    )
    parser.add_argument(
        "--ugc-fast",
        action="store_true",
        help=(
            "Opt-in UGC speed preset: 6+2 steps, Stage 1 reference scale 0.5, "
            "Stage 2 reference scale 1.0, and single-pass refinement. Explicit "
            "stage/scale flags override preset values."
        ),
    )
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
    parser.add_argument(
        "--first-frame",
        default=None,
        help="Optional image guiding the opening composition at pixel frame 0.",
    )
    parser.add_argument(
        "--last-frame",
        default=None,
        help="Optional image guiding the final composition at the last pixel frame.",
    )
    parser.add_argument("--first-frame-strength", type=float, default=1.0)
    parser.add_argument("--last-frame-strength", type=float, default=1.0)
    parser.add_argument(
        "--first-frame-mode",
        choices=["appearance", "layout"],
        default="appearance",
        help="Use layout to retain composition while suppressing face and color appearance.",
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

    best_face_lora = args.best_face_lora or _default_best_face_spec(args.character_sheet)
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
    )
    extra_loras = [(path, float(strength)) for path, strength in args.extra_lora]

    pipe = BestFaceExactPipeline(
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
    print("Best Face MLX (parity recipe: dev + distilled LoRA + identity LoRAs)")
    print(f"  model: {args.model}")
    print(f"  reference: {args.reference}")
    print(f"  resize mode: {resize_mode}")
    print(f"  reference scale: {args.reference_scale:g}")
    print(
        "  stages: "
        f"{effective_settings[0] or 8}+{effective_settings[1] or 3}, "
        f"reference scales {effective_settings[2]:g}/{effective_settings[3]:g}, "
        f"{'single-pass' if effective_settings[4] else 'CFG++'} refine"
    )
    if base_face_lora:
        print(f"  base Face-ID LoRA: {base_face_lora} (strength {args.base_face_strength:g})")
    print(f"  character Face-ID LoRA: {best_face_lora} (strength {args.best_face_strength:g})")
    print(f"  spatial upscaler: {spatial_upscaler or 'auto'}")
    print(f"  source phase: source_id={args.source_id:g}, scale={args.phase_scale:g}")
    print(f"  seed: {args.seed}")
    if args.first_frame:
        print(f"  first frame: {args.first_frame} (strength {args.first_frame_strength:g})")
    if args.last_frame:
        print(f"  last frame: {args.last_frame} (strength {args.last_frame_strength:g})")

    pipe.generate_and_save_best_face(
        prompt=args.prompt,
        reference=args.reference,
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
    print(f"Saved {Path(args.output)} in {time.time() - t0:.1f}s")

    if pipe.low_memory:
        pipe.dit = None
        pipe.image_conditioner.free()
        pipe.upsampler = None
        aggressive_cleanup()


if __name__ == "__main__":
    main()


__all__ = [
    "OFFICIAL_DISTILLED_LORA_STRENGTH",
    "OFFICIAL_NEGATIVE_PROMPT",
    "OFFICIAL_STAGE2_SIGMAS",
    "BestFaceExactPipeline",
]
