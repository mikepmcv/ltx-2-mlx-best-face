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
import json
import random
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageEnhance, ImageFilter

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
from ltx_core_mlx.conditioning.source_phase import SourcePhaseBlock, clear_source_phase, install_source_phase
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.audio_vae import encode_audio
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.audio import load_audio
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions

from .distilled import DistilledPipeline
from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .utils.helpers import create_noised_state
from .utils.media_io import load_image_and_preprocess, resize_and_center_crop
from .utils.progress import phase
from .utils.samplers import denoise_loop, euler_ancestral_cfg_pp_denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009 -- MLX graph materialiser

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"
BEST_FACE_REPO = "Alissonerdx/LTX-Best-Face-ID"
BEST_FACE_FILE = "Best_FaceID_v1.0_LoRA.safetensors"
BEST_FACE_CHARACTER_SHEET_FILE = "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors"
OFFICIAL_BASE_FACE_STRENGTH = 0.2
OFFICIAL_SPATIAL_UPSCALER_FILE = "spatial_upscaler_x2_v1_1.safetensors"
UGC_FAST_STAGE1_STEPS = 6
UGC_FAST_STAGE2_STEPS = 2
UGC_FAST_STAGE1_REFERENCE_SCALE = 1.0
UGC_FAST_STAGE2_REFERENCE_SCALE = 1.0
UGC_ULTRAFAST_STAGE1_REFERENCE_SCALE = 0.5


def _sigma_schedule_for_steps(sigmas: list[float], steps: int | None) -> list[float]:
    """Reduce a published schedule while preserving its terminal zero."""
    available_steps = len(sigmas) - 1
    if steps is None or steps == available_steps:
        return sigmas
    if not 1 <= steps <= available_steps:
        raise ValueError(f"steps must be between 1 and {available_steps}")
    # The distilled table spends several early steps close to sigma=1. Keep
    # the initial sigma and the final `steps` values so reduced schedules retain
    # the more consequential low-noise detail passes and always finish at zero.
    return [sigmas[0], *sigmas[-steps:]]


def _resolve_generation_settings(
    *,
    stage1_steps: int | None,
    stage2_steps: int | None,
    reference_scale: float,
    stage1_reference_scale: float | None,
    stage2_reference_scale: float | None,
    fast_refine: bool,
    ugc_fast: bool,
    ugc_ultrafast: bool,
) -> tuple[int | None, int | None, float, float, bool]:
    """Resolve opt-in speed settings without changing parity defaults."""
    if ugc_fast and ugc_ultrafast:
        raise ValueError("ugc_fast and ugc_ultrafast are mutually exclusive")
    if ugc_fast or ugc_ultrafast:
        if stage1_steps is None:
            stage1_steps = UGC_FAST_STAGE1_STEPS
        if stage2_steps is None:
            stage2_steps = UGC_FAST_STAGE2_STEPS
        if stage1_reference_scale is None:
            stage1_reference_scale = (
                UGC_ULTRAFAST_STAGE1_REFERENCE_SCALE
                if ugc_ultrafast
                else UGC_FAST_STAGE1_REFERENCE_SCALE
            )
        if stage2_reference_scale is None:
            stage2_reference_scale = UGC_FAST_STAGE2_REFERENCE_SCALE
        fast_refine = True

    if stage1_reference_scale is None:
        stage1_reference_scale = reference_scale
    if stage2_reference_scale is None:
        stage2_reference_scale = reference_scale
    return (
        stage1_steps,
        stage2_steps,
        stage1_reference_scale,
        stage2_reference_scale,
        fast_refine,
    )


def _resolve_adapter_spec(spec: str) -> str:
    """Resolve ``repo::filename``; pass normal local/HF LoRA specs through."""
    if "::" not in spec:
        return spec
    repo_id, filename = spec.split("::", 1)
    if not repo_id or not filename:
        raise ValueError(f"Invalid adapter spec {spec!r}; expected repo::filename")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _round32(value: float) -> int:
    return max(32, round(value / 32.0) * 32)


def _metadata_path(output_path: str) -> Path:
    return Path(f"{output_path}.json")


def _write_generation_metadata(output_path: str, metadata: dict[str, object]) -> Path:
    """Write reproducibility metadata beside a successfully generated video."""
    path = _metadata_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare_keyframe_image(
    image_path: str,
    height: int,
    width: int,
    *,
    mode: str,
    layout_blur: float,
) -> Image.Image:
    """Prepare an appearance keyframe or a low-frequency layout-only guide."""
    if mode not in {"appearance", "layout"}:
        raise ValueError("keyframe mode must be appearance or layout")
    if layout_blur < 0:
        raise ValueError("keyframe layout blur must be non-negative")

    with Image.open(image_path) as source:
        image = resize_and_center_crop(source.convert("RGB"), height, width)
    if mode == "layout":
        # Retain silhouette, scale, screen position, and room geometry while
        # suppressing face/texture identity and strong color grading. Scale the
        # blur with encoding width so half/full stages see equivalent structure.
        image = ImageEnhance.Color(image).enhance(0.1)
        radius = layout_blur * width / 1024.0
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image


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
        base_face_lora: str | None = None,
        base_face_strength: float = OFFICIAL_BASE_FACE_STRENGTH,
        spatial_upscaler: str | None = None,
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

        loras: list[tuple[str, float]] = []
        if base_face_lora is not None and base_face_strength != 0:
            loras.append((_resolve_adapter_spec(base_face_lora), float(base_face_strength)))
        loras.append((_resolve_adapter_spec(best_face_lora), float(best_face_strength)))
        for path, strength in extra_loras or []:
            loras.append((_resolve_adapter_spec(path), float(strength)))
        self._pending_loras = loras
        self._spatial_upscaler_path = spatial_upscaler

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

    @classmethod
    def _scaled_reference_geometry(
        cls,
        reference: str,
        *,
        resize_mode: str,
        target_h: int,
        target_w: int,
        reference_scale: float,
    ) -> tuple[int, int, float, float]:
        """Return VAE input size and spatial position multipliers.

        The reference is encoded at the scaled size, while the position
        multipliers keep its H/W RoPE coordinates spanning the same area as
        the unscaled reference. Ratios are derived from the rounded sizes so
        non-exact scales still preserve the original span.
        """
        if not 0.0 < reference_scale <= 1.0:
            raise ValueError("reference_scale must be greater than 0 and at most 1")

        original_h, original_w = cls._reference_size(
            reference,
            resize_mode=resize_mode,
            target_h=target_h,
            target_w=target_w,
        )
        encoded_h = _round32(original_h * reference_scale)
        encoded_w = _round32(original_w * reference_scale)
        return (
            encoded_h,
            encoded_w,
            original_h / encoded_h,
            original_w / encoded_w,
        )

    def _build_identity_conditioning(
        self,
        *,
        reference: str,
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
        assert self.vae_encoder is not None

        ref_h, ref_w, position_scale_h, position_scale_w = self._scaled_reference_geometry(
            reference,
            resize_mode=resize_mode,
            target_h=target_h,
            target_w=target_w,
            reference_scale=reference_scale,
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
        ref_positions = mx.concatenate(
            [
                ref_positions[..., :1],
                ref_positions[..., 1:2] * position_scale_h,
                ref_positions[..., 2:3] * position_scale_w,
            ],
            axis=-1,
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

    def _encode_locked_audio_tokens(
        self,
        audio_path: str,
        *,
        num_frames: int,
        frame_rate: float,
        start_time: float,
        max_duration: float | None,
    ) -> mx.array:
        """Encode supplied speech and return clean audio tokens for joint AV denoising."""
        if frame_rate <= 0:
            raise ValueError("frame_rate must be greater than zero")
        clip_duration = num_frames / frame_rate
        read_duration = clip_duration if max_duration is None else min(max_duration, clip_duration)
        if read_duration <= 0:
            raise ValueError("audio_max_duration must be greater than zero")

        self._load_audio_encoder()
        assert self.audio_encoder is not None
        assert self.audio_processor is not None

        audio_data = load_audio(
            audio_path,
            target_sample_rate=16000,
            start_time=start_time,
            max_duration=read_duration,
        )
        if audio_data is None:
            raise ValueError(f"No audio found in {audio_path}")

        target_samples = max(1, round(clip_duration * audio_data.sample_rate))
        waveform = audio_data.waveform[:, :, :target_samples]
        missing_samples = target_samples - int(waveform.shape[-1])
        if missing_samples > 0:
            waveform = mx.pad(waveform, ((0, 0), (0, 0), (0, missing_samples)))

        audio_latent = encode_audio(
            waveform,
            audio_data.sample_rate,
            self.audio_encoder,
            self.audio_processor,
        )
        audio_t = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_latent = audio_latent[:, :, :audio_t, :]
        audio_tokens, _ = self.audio_patchifier.patchify(audio_latent)
        if int(audio_tokens.shape[1]) != audio_t:
            raise ValueError(
                "Encoded audio length does not match the requested video: "
                f"expected {audio_t} tokens, got {audio_tokens.shape[1]}"
            )
        _materialize(audio_tokens)

        if self.low_memory:
            self.audio_conditioner.free()
            aggressive_cleanup()
        return audio_tokens

    def _prepare_original_audio(
        self,
        audio_path: str,
        *,
        num_frames: int,
        frame_rate: float,
        start_time: float,
        max_duration: float | None,
    ) -> str:
        """Create an exact-duration PCM copy of the supplied audio for final muxing."""
        clip_duration = num_frames / frame_rate
        read_duration = clip_duration if max_duration is None else min(max_duration, clip_duration)
        audio_data = load_audio(
            audio_path,
            target_sample_rate=48000,
            start_time=start_time,
            max_duration=read_duration,
        )
        if audio_data is None:
            raise ValueError(f"No audio found in {audio_path}")

        target_samples = max(1, round(clip_duration * audio_data.sample_rate))
        waveform = audio_data.waveform[:, :, :target_samples]
        missing_samples = target_samples - int(waveform.shape[-1])
        if missing_samples > 0:
            waveform = mx.pad(waveform, ((0, 0), (0, 0), (0, missing_samples)))
        _materialize(waveform)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name
        self._save_waveform(waveform, temp_path, sample_rate=audio_data.sample_rate)
        return temp_path

    def _encode_keyframe(
        self,
        image: str,
        height: int,
        width: int,
        *,
        mode: str,
        layout_blur: float,
    ) -> mx.array:
        """VAE-encode one framing keyframe and return its spatial tokens."""
        assert self.vae_encoder is not None
        prepared = _prepare_keyframe_image(
            image,
            height,
            width,
            mode=mode,
            layout_blur=layout_blur,
        )
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        pixels = mx.array(array * 2.0 - 1.0).transpose(2, 0, 1)[None, ...].astype(mx.bfloat16)
        latent = self.vae_encoder.encode(pixels[:, :, None, :, :])
        tokens = latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, latent.shape[1])
        _materialize(tokens)
        return tokens

    @staticmethod
    def _keyframe_specs(
        *,
        first_frame: str | None,
        last_frame: str | None,
        first_frame_strength: float,
        last_frame_strength: float,
        first_frame_mode: str,
        last_frame_mode: str,
        num_frames: int,
    ) -> list[tuple[str, int, float, str]]:
        for name, strength in (
            ("first_frame_strength", first_frame_strength),
            ("last_frame_strength", last_frame_strength),
        ):
            if not 0.0 <= strength <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        for mode in (first_frame_mode, last_frame_mode):
            if mode not in {"appearance", "layout"}:
                raise ValueError("keyframe mode must be appearance or layout")

        specs: list[tuple[str, int, float, str]] = []
        if first_frame is not None:
            specs.append((first_frame, 0, first_frame_strength, first_frame_mode))
        if last_frame is not None:
            specs.append((last_frame, num_frames - 1, last_frame_strength, last_frame_mode))
        return specs

    @staticmethod
    def _build_keyframe_conditionings(
        encoded: list[tuple[mx.array, int, float]],
        *,
        spatial_dims: tuple[int, int, int],
        frame_rate: float,
    ) -> list[VideoConditionByKeyframeIndex]:
        return [
            VideoConditionByKeyframeIndex(
                frame_idx=frame_idx,
                keyframe_latent=tokens,
                spatial_dims=spatial_dims,
                frame_rate=frame_rate,
                strength=strength,
                num_pixel_frames=1,
            )
            for tokens, frame_idx, strength in encoded
        ]

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
        reference_scale: float = 1.0,
        stage1_reference_scale: float | None = None,
        stage2_reference_scale: float | None = None,
        fast_refine: bool = False,
        ugc_fast: bool = False,
        ugc_ultrafast: bool = False,
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = 0,
        first_frame: str | None = None,
        last_frame: str | None = None,
        first_frame_strength: float = 1.0,
        last_frame_strength: float = 1.0,
        first_frame_mode: str = "appearance",
        last_frame_mode: str = "appearance",
        keyframe_layout_blur: float = 32.0,
        audio_path: str | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate a Best Face video using LTX's fast distilled 8+3 flow."""
        if not prompt.lstrip().startswith("ref_t2v:"):
            prompt = "ref_t2v: " + prompt.strip()

        (
            stage1_steps,
            stage2_steps,
            stage1_reference_scale,
            stage2_reference_scale,
            fast_refine,
        ) = _resolve_generation_settings(
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            reference_scale=reference_scale,
            stage1_reference_scale=stage1_reference_scale,
            stage2_reference_scale=stage2_reference_scale,
            fast_refine=fast_refine,
            ugc_fast=ugc_fast,
            ugc_ultrafast=ugc_ultrafast,
        )

        locked_audio_tokens = None
        if audio_path is not None:
            with phase("Encoding locked TTS audio", verbose=self.verbose):
                locked_audio_tokens = self._encode_locked_audio_tokens(
                    audio_path,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    start_time=audio_start_time,
                    max_duration=audio_max_duration,
                )

        # The official character-sheet refine pass uses CFG++ at CFG 1. Its
        # unconditional prediction still controls the sampler direction.
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            _materialize(video_embeds, audio_embeds)
            if getattr(self, "_best_face_cfg_pp", False) and not fast_refine:
                negative_prompt = getattr(self, "_best_face_negative_prompt", "")
                neg_video_embeds, neg_audio_embeds = self._encode_text(negative_prompt)
                _materialize(neg_video_embeds, neg_audio_embeds)
            else:
                neg_video_embeds = neg_audio_embeds = None
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        height, width = snap_output_dimensions(height, width, two_stage=True)
        keyframe_specs = self._keyframe_specs(
            first_frame=first_frame,
            last_frame=last_frame,
            first_frame_strength=first_frame_strength,
            last_frame_strength=last_frame_strength,
            first_frame_mode=first_frame_mode,
            last_frame_mode=last_frame_mode,
            num_frames=num_frames,
        )

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
            reference_scale=stage1_reference_scale,
            frame_rate=frame_rate,
            num_generation_tokens=generation_tokens_1,
            source_id=source_id,
            phase_scale=phase_scale,
            crf=reference_crf,
        )
        encoded_keyframes_1 = [
            (
                self._encode_keyframe(
                    path,
                    H_half * 32,
                    W_half * 32,
                    mode=mode,
                    layout_blur=keyframe_layout_blur,
                ),
                frame_idx,
                strength,
            )
            for path, frame_idx, strength, mode in keyframe_specs
        ]
        keyframes_1 = self._build_keyframe_conditionings(
            encoded_keyframes_1,
            spatial_dims=(F, H_half, W_half),
            frame_rate=frame_rate,
        )

        video_state_1 = create_noised_state(
            base_shape=video_shape,
            conditionings=[identity_1, *keyframes_1],
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        if locked_audio_tokens is None:
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
        else:
            # A zero denoise mask freezes the supplied TTS latent. The video
            # still attends to it, so facial motion follows the recording.
            audio_state_1 = LatentState(
                latent=locked_audio_tokens,
                clean_latent=locked_audio_tokens,
                denoise_mask=mx.zeros(
                    (1, locked_audio_tokens.shape[1], 1), dtype=mx.bfloat16
                ),
                positions=audio_positions,
            )

        sigmas_1 = _sigma_schedule_for_steps(DISTILLED_SIGMAS, stage1_steps)
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
            reference_scale=stage2_reference_scale,
            frame_rate=frame_rate,
            num_generation_tokens=generation_tokens_2,
            source_id=source_id,
            phase_scale=phase_scale,
            crf=reference_crf,
        )
        encoded_keyframes_2 = [
            (
                self._encode_keyframe(
                    path,
                    H_full * 32,
                    W_full * 32,
                    mode=mode,
                    layout_blur=keyframe_layout_blur,
                ),
                frame_idx,
                strength,
            )
            for path, frame_idx, strength, mode in keyframe_specs
        ]
        keyframes_2 = self._build_keyframe_conditionings(
            encoded_keyframes_2,
            spatial_dims=(F, H_full, W_full),
            frame_rate=frame_rate,
        )

        if self.low_memory:
            # Both stage-2 reference tokens and upscaled latent are materialized.
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        stage2_schedule = getattr(self, "_best_face_stage2_sigmas", STAGE_2_SIGMAS)
        sigmas_2 = _sigma_schedule_for_steps(stage2_schedule, stage2_steps)
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)
        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=[identity_2, *keyframes_2],
            spatial_dims=(F, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = output_1.audio_latent
        if locked_audio_tokens is None:
            audio_state_2 = create_noised_state(
                base_shape=audio_tokens_1.shape,
                conditionings=[],
                spatial_dims=(F, H_full, W_full),
                positions=audio_positions,
                seed=seed + 2,
                sigma=start_sigma,
                initial_latent=audio_tokens_1,
            )
        else:
            # Keep the original TTS latent frozen through refinement too.
            audio_state_2 = LatentState(
                latent=locked_audio_tokens,
                clean_latent=locked_audio_tokens,
                denoise_mask=mx.zeros(
                    (1, locked_audio_tokens.shape[1], 1), dtype=mx.bfloat16
                ),
                positions=audio_positions,
            )

        install_source_phase(self.dit, [phase_2], theta=float(self.dit.config.rope_theta))
        self._pre_denoise_flush(video_state_2, audio_state_2)
        try:
            stage2_kwargs = dict(
                model=x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_2,
                on_step=self._stepwise_hook(F, H_full, W_full, stage=2),
            )
            if getattr(self, "_best_face_cfg_pp", False) and not fast_refine:
                assert neg_video_embeds is not None and neg_audio_embeds is not None
                output_2 = euler_ancestral_cfg_pp_denoise_loop(
                    **stage2_kwargs,
                    negative_video_text_embeds=neg_video_embeds,
                    negative_audio_text_embeds=neg_audio_embeds,
                )
            else:
                output_2 = denoise_loop(**stage2_kwargs)
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
        reference_scale: float = 1.0,
        stage1_reference_scale: float | None = None,
        stage2_reference_scale: float | None = None,
        fast_refine: bool = False,
        ugc_fast: bool = False,
        ugc_ultrafast: bool = False,
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_crf: int = 0,
        first_frame: str | None = None,
        last_frame: str | None = None,
        first_frame_strength: float = 1.0,
        last_frame_strength: float = 1.0,
        first_frame_mode: str = "appearance",
        last_frame_mode: str = "appearance",
        keyframe_layout_blur: float = 32.0,
        audio_path: str | None = None,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
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
            reference_scale=reference_scale,
            stage1_reference_scale=stage1_reference_scale,
            stage2_reference_scale=stage2_reference_scale,
            fast_refine=fast_refine,
            ugc_fast=ugc_fast,
            ugc_ultrafast=ugc_ultrafast,
            source_id=source_id,
            phase_scale=phase_scale,
            reference_crf=reference_crf,
            first_frame=first_frame,
            last_frame=last_frame,
            first_frame_strength=first_frame_strength,
            last_frame_strength=last_frame_strength,
            first_frame_mode=first_frame_mode,
            last_frame_mode=last_frame_mode,
            keyframe_layout_blur=keyframe_layout_blur,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
        )
        if audio_path is None:
            saved_path = self._decode_and_save_video(
                video_latent,
                audio_latent,
                output_path,
                frame_rate=frame_rate,
            )
        else:
            # Decode video with an exact-duration PCM copy of the input. The
            # model's audio output is deliberately discarded, guaranteeing
            # that generated speech, ambience, or music cannot reach the file.
            if self.low_memory and self.dit is not None:
                self.dit = None
                self._loaded = False
                aggressive_cleanup()
            self._load_decoders()
            original_audio = self._prepare_original_audio(
                audio_path,
                num_frames=num_frames,
                frame_rate=frame_rate,
                start_time=audio_start_time,
                max_duration=audio_max_duration,
            )
            try:
                with phase("Decoding video + muxing original TTS", verbose=self.verbose):
                    self.video_decoder_block.decode_and_stream(
                        video_latent,
                        output_path,
                        frame_rate=frame_rate,
                        audio_path=original_audio,
                    )
                saved_path = output_path
            finally:
                Path(original_audio).unlink(missing_ok=True)
        effective_height, effective_width = snap_output_dimensions(height, width, two_stage=True)
        (
            effective_stage1_steps,
            effective_stage2_steps,
            effective_stage1_reference_scale,
            effective_stage2_reference_scale,
            effective_fast_refine,
        ) = _resolve_generation_settings(
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            reference_scale=reference_scale,
            stage1_reference_scale=stage1_reference_scale,
            stage2_reference_scale=stage2_reference_scale,
            fast_refine=fast_refine,
            ugc_fast=ugc_fast,
            ugc_ultrafast=ugc_ultrafast,
        )
        _write_generation_metadata(
            saved_path,
            {
                "created_at": datetime.now(UTC).isoformat(),
                "pipeline": type(self).__name__,
                "prompt": prompt,
                "reference": str(Path(reference).expanduser().resolve()),
                "reference_scale": reference_scale,
                "stage1_reference_scale": effective_stage1_reference_scale,
                "stage2_reference_scale": effective_stage2_reference_scale,
                "fast_refine": effective_fast_refine,
                "ugc_fast": ugc_fast,
                "ugc_ultrafast": ugc_ultrafast,
                "resize_mode": resize_mode,
                "reference_crf": reference_crf,
                "first_frame": str(Path(first_frame).expanduser().resolve()) if first_frame else None,
                "last_frame": str(Path(last_frame).expanduser().resolve()) if last_frame else None,
                "first_frame_strength": first_frame_strength,
                "last_frame_strength": last_frame_strength,
                "first_frame_mode": first_frame_mode,
                "last_frame_mode": last_frame_mode,
                "keyframe_layout_blur": keyframe_layout_blur,
                "audio_path": (
                    str(Path(audio_path).expanduser().resolve()) if audio_path else None
                ),
                "audio_start_time": audio_start_time if audio_path else None,
                "audio_max_duration": audio_max_duration if audio_path else None,
                "audio_mode": "locked_input" if audio_path else "generated",
                "height": effective_height,
                "width": effective_width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "seed": seed,
                "stage1_steps": effective_stage1_steps,
                "stage2_steps": effective_stage2_steps,
                "source_id": source_id,
                "phase_scale": phase_scale,
                "model": str(self.model_dir),
                "spatial_upscaler": self._spatial_upscaler_path,
                "loras": [
                    {"path": str(path), "strength": strength}
                    for path, strength in self._pending_loras
                ],
                "output": str(Path(saved_path).expanduser().resolve()),
            },
        )
        return saved_path


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
        help="Use a single conditioned Stage 2 pass instead of two-pass CFG++.",
    )
    speed_presets = parser.add_mutually_exclusive_group()
    speed_presets.add_argument(
        "--ugc-fast",
        action="store_true",
        help=(
            "Balanced UGC preset: 6+2 steps, native references in both stages, "
            "and single-pass refinement. Explicit stage/scale flags override it."
        ),
    )
    speed_presets.add_argument(
        "--ugc-ultrafast",
        action="store_true",
        help=(
            "Maximum-speed UGC preset: same as --ugc-fast but Stage 1 uses a "
            "0.5 reference scale, trading more identity fidelity for speed."
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
    parser.add_argument(
        "--reference-crf",
        type=int,
        default=0,
        help="Optional reference-image H.264 CRF; 0 disables it (Best Face parity default).",
    )
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
    parser.add_argument(
        "--audio",
        default=None,
        help="Optional TTS/audio file. Its latent is frozen for lip-sync and its original PCM is muxed.",
    )
    parser.add_argument("--audio-start", type=float, default=0.0)
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Read at most this many seconds, then pad silence to the video duration.",
    )

    args = parser.parse_args()
    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    lora_spec = args.best_face_lora or _default_best_face_spec(args.character_sheet)
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

    pipe = BestFacePipeline(
        model_dir=args.model,
        gemma_model_id=args.gemma,
        best_face_lora=lora_spec,
        best_face_strength=args.best_face_strength,
        base_face_lora=base_face_lora,
        base_face_strength=args.base_face_strength,
        spatial_upscaler=spatial_upscaler,
        extra_loras=extra_loras,
        low_memory=not args.no_low_memory,
    )

    t0 = time.time()
    print("Best Face MLX (distilled)")
    print(f"  model: {args.model}")
    print(f"  reference: {args.reference}")
    print(f"  resize mode: {resize_mode}")
    print(f"  reference scale: {args.reference_scale:g}")
    print(
        "  stages: "
        f"{effective_settings[0] or 8}+{effective_settings[1] or 3}, "
        f"reference scales {effective_settings[2]:g}/{effective_settings[3]:g}, "
        f"{'single-pass' if effective_settings[4] else 'standard'} refine"
    )
    if base_face_lora:
        print(f"  base Face-ID LoRA: {base_face_lora} (strength {args.base_face_strength:g})")
    print(f"  character Face-ID LoRA: {lora_spec} (strength {args.best_face_strength:g})")
    print(f"  spatial upscaler: {spatial_upscaler or 'auto'}")
    print(f"  source phase: source_id={args.source_id:g}, scale={args.phase_scale:g}")
    print(f"  seed: {args.seed}")
    if args.first_frame:
        print(f"  first frame: {args.first_frame} (strength {args.first_frame_strength:g})")
    if args.last_frame:
        print(f"  last frame: {args.last_frame} (strength {args.last_frame_strength:g})")
    if args.audio:
        print(f"  locked TTS audio: {args.audio} (start {args.audio_start:g}s)")

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
        audio_path=args.audio,
        audio_start_time=args.audio_start,
        audio_max_duration=args.audio_max_duration,
    )
    print(f"Saved {Path(args.output)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


__all__ = ["BestFacePipeline"]
