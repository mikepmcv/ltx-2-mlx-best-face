"""Native-MLX Best Face ID inference for LTX-2.3.

The default path mirrors the fast recipe used by the Best Face author:
a distilled LTX-2.3 transformer, two-stage 8+3-step generation, and the actual
Best Face identity LoRA. The only new model behavior is the identity-overlap
reference conditioning (clean reference tokens + source-phase/TASS-RoPE).

Run:

    uv run python -m ltx_pipelines_mlx.best_face \
        --prompt "A podcast host speaks naturally to camera..." \
        --reference host.png \
        --frames 49 \
        -H 576 -W 768 \
        -o best-face-test.mp4
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from PIL import Image

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
from ltx_core_mlx.conditioning.source_phase import SourcePhaseBlock, clear_source_phase, install_source_phase
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions

from .distilled import DistilledPipeline
from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .utils.helpers import create_noised_state
from .utils.media_io import load_image_and_preprocess
from .utils.progress import phase
from .utils.samplers import denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- MLX graph materialiser

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"
BEST_FACE_REPO = "Alissonerdx/LTX-Best-Face-ID"
BEST_FACE_FILE = "Best_FaceID_v1.0_LoRA.safetensors"
BEST_FACE_CHARACTER_SHEET_FILE = "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors"


def _resolve_adapter_spec(spec: str) -> str:
    """Resolve ``repo::filename``; pass normal local/HF LoRA specs through."""
    if "::" not in spec:
        return spec
    repo_id, filename = spec.split("::", 1)
    if not repo_id or not filename:
        raise ValueError(f"Invalid adapter spec {spec!r}; expected repo::filename")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _round32(value: int) -> int:
    return max(32, int(round(value / 32.0)) * 32)


class BestFacePipeline(DistilledPipeline):
    """Fast two-stage LTX-2.3 distilled pipeline with Best Face identity lock.

    v1 ports the identity-critical inference mechanism:
    - actual Best Face LoRA weights;
    - reference image encoded by the normal LTX VAE;
    - reference latent appended as separate clean tokens (not I2V frame 0);
    - overlap positions on the target frame-0 RoPE grid;
    - reference timestep 0 via the existing denoise-mask path;
    - source-phase/TASS-RoPE with source_id=2 and phase_scale=1;
    - reference tokens trimmed before upscaling/decoding.

    The optional ArcFace projector is intentionally omitted because the model
    author documents it as marginal. Training-time ArcFace loss is already
    baked into the LoRA weights.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        *,
        gemma_model_id: str = DEFAULT_GEMMA,
        best_face_lora: str,
        best_face_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        low_memory: bool = True,
    ):
        # Block streaming is deliberately off for first parity validation:
        # source-phase is installed on the actual LTXModel instance.
        super().__init__(
            model_dir=model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=False,
            tile_count=None,
        )

        loras: list[tuple[str, float]] = [
            (_resolve_adapter_spec(best_face_lora), float(best_face_strength))
        ]
        for path, strength in extra_loras or []:
            loras.append((_resolve_adapter_spec(path), float(strength)))
        self._pending_loras = loras

    @staticmethod
    def _reference_size(
        reference: str,
        *,
        resize_mode: str,
        target_h: int,
        target_w: int,
    ) -> tuple[int, int]:
        if resize_mode == "match_target":
            return target_h, target_w
        if resize_mode != "native_resolution":
            raise ValueError(
                f"Unsupported resize_mode={resize_mode!r}; use match_target or native_resolution"
            )
        with Image.open(reference) as image:
            src_w, src_h = image.size
        return _round32(src_h), _round32(src_w)

    def _build_identity_conditioning(
        self,
        *,
        reference: str,
        resize_mode: str,
        target_h: int,
        target_w: int,
        frame_rate: float,
        num_generation_tokens: int,
        source_id: float,
        phase_scale: float,
        crf: int,
    ) -> tuple[VideoConditionByReferenceLatent, SourcePhaseBlock]:
        assert self.vae_encoder is not None

        ref_h, ref_w = self._reference_size(
            reference,
            resize_mode=resize_mode,
            target_h=target_h,
            target_w=target_w,
        )

        # Best Face/BFS feeds the resized reference straight to the LTX VAE.
        # Keep CRF=0 by default; nonzero is exposed only for experiments.
        ref_pixels = load_image_and_preprocess(reference, ref_h, ref_w, crf=crf)
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
        _materialize(ref_tokens, ref_positions)

        condition = VideoConditionByReferenceLatent(
            reference_latent=ref_tokens,
            reference_positions=ref_positions,
            downscale_factor=1,
            strength=1.0,
        )
        block = SourcePhaseBlock(
            start=num_generation_tokens,
            length=int(ref_tokens.shape[1]),
            segment_value=float(source_id) * float(phase_scale),
        )
        return condition, block

    def generate_best_face(
        self,
        prompt: str,
        reference: str,
        *,
        height: int = 576,
        width: int = 768,
        num_frames: int = 49,
        frame_rate: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        resize_mode: str = "match_target",
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Generate a Best Face video using LTX's fast distilled 8+3 flow."""
        if not prompt.lstrip().startswith("ref_t2v:"):
            prompt = "ref_t2v: " + prompt.strip()

        # Positive text only: distilled pipeline has no CFG branch.
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        height, width = snap_output_dimensions(height, width, two_stage=True)

        # ---------------- Stage 1: half resolution, normally 8 steps ----------------
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        generation_tokens_1 = F * H_half * W_half
        video_shape = (1, generation_tokens_1, 128)

        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)
        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        identity_1, phase_1 = self._build_identity_conditioning(
            reference=reference,
            resize_mode=resize_mode,
            target_h=H_half * 32,
            target_w=W_half * 32,
            frame_rate=frame_rate,
            num_generation_tokens=generation_tokens_1,
            source_id=source_id,
            phase_scale=phase_scale,
            crf=reference_crf,
        )

        video_state_1 = create_noised_state(
            base_shape=video_shape,
            conditionings=[identity_1],
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state_1 = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)

        install_source_phase(self.dit, [phase_1], theta=float(self.dit.config.rope_theta))
        self._pre_denoise_flush(video_state_1, audio_state_1)
        try:
            output_1 = denoise_loop(
                model=x0_model,
                video_state=video_state_1,
                audio_state=audio_state_1,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
                on_step=self._stepwise_hook(F, H_half, W_half, stage=1),
            )
        finally:
            clear_source_phase(self.dit)

        if self.low_memory:
            aggressive_cleanup()

        # Only generated tokens are spatially upscaled; identity tokens are context.
        gen_tokens_1 = output_1.video_latent[:, :generation_tokens_1, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        # ---------------- Stage 2: full resolution, normally 3 steps ----------------
        H_full = H_half * 2
        W_full = W_half * 2
        generation_tokens_2 = F * H_full * W_full

        identity_2, phase_2 = self._build_identity_conditioning(
            reference=reference,
            resize_mode=resize_mode,
            target_h=H_full * 32,
            target_w=W_full * 32,
            frame_rate=frame_rate,
            num_generation_tokens=generation_tokens_2,
            source_id=source_id,
            phase_scale=phase_scale,
            crf=reference_crf,
        )

        if self.low_memory:
            # Both stage-2 reference tokens and upscaled latent are materialized.
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)
        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=[identity_2],
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F, H_full, W_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        install_source_phase(self.dit, [phase_2], theta=float(self.dit.config.rope_theta))
        self._pre_denoise_flush(video_state_2, audio_state_2)
        try:
            output_2 = denoise_loop(
                model=x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_2,
                on_step=self._stepwise_hook(F, H_full, W_full, stage=2),
            )
        finally:
            clear_source_phase(self.dit)

        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_2 = output_2.video_latent[:, :generation_tokens_2, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)
        _materialize(video_latent, audio_latent)
        return video_latent, audio_latent

    def generate_and_save_best_face(
        self,
        *,
        prompt: str,
        reference: str,
        output_path: str,
        height: int = 576,
        width: int = 768,
        num_frames: int = 49,
        frame_rate: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        resize_mode: str = "match_target",
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = 0,
    ) -> str:
        video_latent, audio_latent = self.generate_best_face(
            prompt=prompt,
            reference=reference,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            resize_mode=resize_mode,
            source_id=source_id,
            phase_scale=phase_scale,
            reference_crf=reference_crf,
        )
        return self._decode_and_save_video(
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
        )


def _default_best_face_spec(character_sheet: bool) -> str:
    filename = BEST_FACE_CHARACTER_SHEET_FILE if character_sheet else BEST_FACE_FILE
    return f"{BEST_FACE_REPO}::{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.best_face",
        description="Native MLX LTX-2.3 Best Face ID (distilled 8+3-step pipeline)",
    )
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--reference", "-i", required=True, help="Identity reference image")
    parser.add_argument("--output", "-o", required=True, help="Output .mp4 path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument(
        "--best-face-lora",
        default=None,
        help="Local path, single-file HF repo, or repo::filename; defaults to Best Face v1.",
    )
    parser.add_argument("--best-face-strength", type=float, default=1.0)
    parser.add_argument(
        "--extra-lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="Additional LoRA to fuse (repeatable).",
    )
    parser.add_argument(
        "--character-sheet",
        action="store_true",
        help="Use the Best Face character-sheet adapter and native reference resolution.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["match_target", "native_resolution"],
        default=None,
    )
    parser.add_argument("--height", "-H", type=int, default=576)
    parser.add_argument("--width", "-W", type=int, default=768)
    parser.add_argument("--frames", "-f", type=int, default=49)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--source-id", type=float, default=2.0)
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument(
        "--reference-crf",
        type=int,
        default=0,
        help="Optional reference-image H.264 CRF; 0 disables it (Best Face parity default).",
    )
    parser.add_argument("--seed", "-s", type=int, default=-1)
    parser.add_argument("--no-low-memory", action="store_true")

    args = parser.parse_args()
    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    lora_spec = args.best_face_lora or _default_best_face_spec(args.character_sheet)
    resize_mode = args.resize_mode or (
        "native_resolution" if args.character_sheet else "match_target"
    )
    extra_loras = [(path, float(strength)) for path, strength in args.extra_lora]

    pipe = BestFacePipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        best_face_lora=lora_spec,
        best_face_strength=args.best_face_strength,
        extra_loras=extra_loras,
        low_memory=not args.no_low_memory,
    )

    t0 = time.time()
    print("Best Face MLX (distilled)")
    print(f"  model: {args.model}")
    print(f"  reference: {args.reference}")
    print(f"  resize mode: {resize_mode}")
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
        source_id=args.source_id,
        phase_scale=args.phase_scale,
        reference_crf=args.reference_crf,
    )
    print(f"Saved {Path(args.output)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


__all__ = ["BestFacePipeline"]
