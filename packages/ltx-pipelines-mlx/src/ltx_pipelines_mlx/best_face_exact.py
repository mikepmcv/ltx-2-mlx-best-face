"""Parity-first Best Face ID pipeline for LTX-2.3 on Apple Silicon.

This variant mirrors the Best Face author's published fast demo recipe more
closely than :mod:`ltx_pipelines_mlx.best_face`:

- LTX-2.3 dev transformer;
- official LTX-2.3 distilled-1.1 LoRA at strength 1.0;
- Best Face ID LoRA at strength 1.0;
- the same distilled 8-step + 3-step two-stage schedule;
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
    BestFacePipeline,
    _default_best_face_spec,
    _resolve_adapter_spec,
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
        distilled_lora: str | None = None,
        distilled_lora_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        low_memory: bool = True,
    ):
        super().__init__(
            model_dir=model_dir,
            gemma_model_id=gemma_model_id,
            best_face_lora=best_face_lora,
            best_face_strength=best_face_strength,
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
        "--distilled-lora",
        default=None,
        help="Override the official distilled LoRA; accepts local path or repo::filename.",
    )
    parser.add_argument("--distilled-lora-strength", type=float, default=1.0)
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

    args = parser.parse_args()
    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    best_face_lora = args.best_face_lora or _default_best_face_spec(args.character_sheet)
    resize_mode = args.resize_mode or (
        "native_resolution" if args.character_sheet else "match_target"
    )
    extra_loras = [(path, float(strength)) for path, strength in args.extra_lora]

    pipe = BestFaceExactPipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        best_face_lora=best_face_lora,
        best_face_strength=args.best_face_strength,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength=args.distilled_lora_strength,
        extra_loras=extra_loras,
        low_memory=not args.no_low_memory,
    )

    t0 = time.time()
    print("Best Face MLX (parity recipe: dev + distilled LoRA + identity LoRA)")
    print(f"  model: {args.model}")
    print(f"  reference: {args.reference}")
    print(f"  resize mode: {resize_mode}")
    print(f"  reference scale: {args.reference_scale:g}")
    print(f"  source phase: source_id={args.source_id:g}, scale={args.phase_scale:g}")
    print(f"  seed: {args.seed}")

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
        source_id=args.source_id,
        phase_scale=args.phase_scale,
        reference_crf=args.reference_crf,
    )
    print(f"Saved {Path(args.output)} in {time.time() - t0:.1f}s")

    if pipe.low_memory:
        pipe.dit = None
        pipe.image_conditioner.free()
        pipe.upsampler = None
        aggressive_cleanup()


if __name__ == "__main__":
    main()


__all__ = ["BestFaceExactPipeline"]
