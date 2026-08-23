"""Native-MLX Best Face ID inference for LTX-2.3.

This is a deliberately isolated pipeline: it reuses the existing LTX-2 MLX
model, VAE, guider, sampler, decoder, and LoRA loader, while adding the
identity-overlap conditioning required by LTX-Best-Face-ID.

Run it directly:

    python -m ltx_pipelines_mlx.best_face \
        --prompt "A podcast host speaks calmly to camera" \
        --reference host.png \
        --frame-rate 24 \
        --frames 49 \
        --output host.mp4

The default adapter is Alissonerdx/LTX-Best-Face-ID's close-up checkpoint.
Use --character-sheet for the character-sheet continuation checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import mlx.core as mx
from huggingface_hub import hf_hub_download
from PIL import Image

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import (
    compute_video_latent_shape,
    snap_output_dimensions,
)
from ltx_core_mlx.conditioning.source_phase import (
    SourcePhaseBlock,
    clear_source_phase,
    install_source_phase,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import (
    VideoConditionByReferenceLatent,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .scheduler import ltx2_schedule
from .ti2vid_one_stage import TI2VidOneStagePipeline
from .ti2vid_two_stages import DEFAULT_CFG_SCALE
from .utils.helpers import create_noised_state
from .utils.media_io import DEFAULT_IMAGE_CRF, load_image_and_preprocess
from .utils.samplers import guided_denoise_loop

_materialize = getattr(mx, "eval")

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"
BEST_FACE_REPO = "Alissonerdx/LTX-Best-Face-ID"
BEST_FACE_FILE = "Best_FaceID_v1.0_LoRA.safetensors"
BEST_FACE_CHARACTER_SHEET_FILE = "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors"


def _resolve_adapter_spec(spec: str) -> str:
    """Resolve ``repo::filename`` or pass through a local/HF LoRA spec.

    ``BasePipeline`` already knows how to resolve a normal local path or an HF
    repo containing exactly one safetensors file. Best Face contains multiple
    adapters, so ``repo::filename`` is supported here to select one explicitly.
    """
    if "::" not in spec:
        return spec
    repo_id, filename = spec.split("::", 1)
    if not repo_id or not filename:
        raise ValueError(f"Invalid adapter spec {spec!r}; expected repo::filename")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _round32(value: int) -> int:
    return max(32, int(round(value / 32.0)) * 32)


class BestFacePipeline(TI2VidOneStagePipeline):
    """LTX-2.3 dev+CFG pipeline with Best Face identity-overlap conditioning.

    The implementation mirrors the identity-critical inference path:

    - encode the reference through the normal LTX video VAE;
    - append the reference latent as separate clean tokens;
    - place those tokens on an overlapping frame-0 T/H/W coordinate grid;
    - give them timestep 0 via the existing denoise mask semantics;
    - apply source-phase/TASS-RoPE only to the reference token range;
    - run the unmodified LTX transformer with the actual Best Face LoRA;
    - slice reference tokens away before VAE decode.

    ArcFace/IdentityProjector is intentionally not part of v1.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        *,
        gemma_model_id: str = DEFAULT_GEMMA,
        dev_transformer: str = "transformer-dev.safetensors",
        best_face_lora: str,
        best_face_strength: float = 1.0,
        extra_loras: list[tuple[str, float]] | None = None,
        low_memory: bool = True,
    ):
        # Best Face v1 keeps normal in-memory MLX execution. Block streaming is
        # intentionally disabled until source-phase has been verified through
        # the wrapper path as well.
        super().__init__(
            model_dir=model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=False,
            dev_transformer=dev_transformer,
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

        ref_pixels = load_image_and_preprocess(reference, ref_h, ref_w, crf=crf)
        ref_pixels = ref_pixels[:, :, None, :, :]  # BCHW -> BCFHW with F=1
        ref_latent = self.vae_encoder.encode(ref_pixels)

        # VideoEncoder returns B,C,F,H,W. Flatten exactly like the standard
        # reference-conditioning path, keeping the latent channel last.
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

        # Materialize before optionally freeing the VAE encoder.
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
        height: int = 480,
        width: int = 704,
        num_frames: int = 49,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        resize_mode: str = "match_target",
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = DEFAULT_IMAGE_CRF,
    ) -> tuple[mx.array, mx.array]:
        """Generate an identity-preserving LTX video from one reference image."""
        if not prompt.lstrip().startswith("ref_t2v:"):
            prompt = "ref_t2v: " + prompt.strip()

        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = (
            self._encode_text_with_negative(prompt)
        )

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        height, width = snap_output_dimensions(height, width, two_stage=False)
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        num_generation_tokens = F * H * W
        video_shape = (1, num_generation_tokens, 128)

        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        identity_condition, phase_block = self._build_identity_conditioning(
            reference=reference,
            resize_mode=resize_mode,
            target_h=H * 32,
            target_w=W * 32,
            frame_rate=frame_rate,
            num_generation_tokens=num_generation_tokens,
            source_id=source_id,
            phase_scale=phase_scale,
            crf=reference_crf,
        )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=[identity_condition],
            spatial_dims=(F, H, W),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H, W),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )

        # Patch only this model instance. The active block can later be replaced
        # for another stage/resolution without re-patching the model code.
        install_source_phase(
            self.dit,
            [phase_block],
            theta=float(self.dit.config.rope_theta),
        )

        sigmas = ltx2_schedule(num_steps, num_tokens=num_generation_tokens)
        x0_model = X0Model(self.dit)

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        video_factory = create_multimodal_guider_factory(
            video_guider_params,
            negative_context=neg_video_embeds,
        )
        audio_factory = create_multimodal_guider_factory(
            audio_guider_params,
            negative_context=neg_audio_embeds,
        )

        self._pre_denoise_flush(video_state, audio_state)
        if self.low_memory:
            # The encoded reference tokens are materialized; the VAE encoder is
            # no longer needed during the expensive transformer loop.
            self.image_conditioner.free()
            aggressive_cleanup()

        try:
            output = guided_denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                video_guider_factory=video_factory,
                audio_guider_factory=audio_factory,
                sigmas=sigmas,
                tap=None,
                on_step=None,
            )
        finally:
            clear_source_phase(self.dit)

        # Reference tokens participate in attention/denoising but never render.
        generated = output.video_latent[:, :num_generation_tokens, :]
        video_latent = self.video_patchifier.unpatchify(generated, (F, H, W))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)
        _materialize(video_latent, audio_latent)
        return video_latent, audio_latent

    def generate_and_save_best_face(
        self,
        *,
        prompt: str,
        reference: str,
        output_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 49,
        frame_rate: float,
        seed: int = 42,
        num_steps: int = 30,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 1.0,
        resize_mode: str = "match_target",
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = DEFAULT_IMAGE_CRF,
    ) -> str:
        video_latent, audio_latent = self.generate_best_face(
            prompt=prompt,
            reference=reference,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            resize_mode=resize_mode,
            source_id=source_id,
            phase_scale=phase_scale,
            reference_crf=reference_crf,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()
        result = self._decode_and_save_video(
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
        )

        if self.low_memory:
            self.vae_decoder = None
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()

        return result


def _default_best_face_spec(character_sheet: bool) -> str:
    filename = BEST_FACE_CHARACTER_SHEET_FILE if character_sheet else BEST_FACE_FILE
    return f"{BEST_FACE_REPO}::{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ltx_pipelines_mlx.best_face",
        description="Native MLX LTX-2.3 Best Face ID inference",
    )
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--reference", "-i", required=True, help="Identity reference image")
    parser.add_argument("--output", "-o", required=True, help="Output .mp4 path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--gemma", default=DEFAULT_GEMMA)
    parser.add_argument("--dev-transformer", default="transformer-dev.safetensors")
    parser.add_argument(
        "--best-face-lora",
        default=None,
        help=(
            "Best Face adapter: local path, single-file HF repo, or repo::filename. "
            "Defaults to the official close-up/character-sheet file."
        ),
    )
    parser.add_argument("--best-face-strength", type=float, default=1.0)
    parser.add_argument(
        "--extra-lora",
        action="append",
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=[],
        help="Additional LoRA to fuse (repeatable), e.g. a distilled LoRA.",
    )
    parser.add_argument(
        "--character-sheet",
        action="store_true",
        help=(
            "Use Best_FaceID_CharacterSheet_v1.0_LoRA and native reference resolution "
            "unless --best-face-lora/--resize-mode overrides it."
        ),
    )
    parser.add_argument(
        "--resize-mode",
        choices=["match_target", "native_resolution"],
        default=None,
    )
    parser.add_argument("--height", "-H", type=int, default=480)
    parser.add_argument("--width", "-W", type=int, default=704)
    parser.add_argument("--frames", "-f", type=int, default=49)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--source-id", type=float, default=2.0)
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--reference-crf", type=int, default=DEFAULT_IMAGE_CRF)
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
        dev_transformer=args.dev_transformer,
        best_face_lora=lora_spec,
        best_face_strength=args.best_face_strength,
        extra_loras=extra_loras,
        low_memory=not args.no_low_memory,
    )

    t0 = time.time()
    print("Best Face MLX")
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
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
        resize_mode=resize_mode,
        source_id=args.source_id,
        phase_scale=args.phase_scale,
        reference_crf=args.reference_crf,
    )
    print(f"Saved {Path(args.output)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


__all__ = ["BestFacePipeline"]
