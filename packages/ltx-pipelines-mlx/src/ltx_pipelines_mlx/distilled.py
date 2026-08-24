"""Distilled two-stage video generation pipeline.

LTX-2.3 keeps the existing deterministic Euler path. LTX-2.5 is detected
from checkpoint architecture metadata and uses its required first-frame
keyframe embedding plus ancestral stage-1 sampling.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions

from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .ti2vid_two_stages import TI2VidTwoStagesPipeline
from .utils.helpers import create_noised_state
from .utils.ltx25_sampler import denoise_loop_v25, first_latent_frame_keyframes_mask
from .utils.progress import phase
from .utils.samplers import denoise_loop

_materialize = getattr(mx, "eval")  # noqa: B009


class DistilledPipeline(TI2VidTwoStagesPipeline):
    """Distilled two-stage T2V/I2V pipeline (half-res -> upscale -> full-res refine)."""

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        tile_count=None,
    ):
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            tile_count=tile_count,
        )

    def load(self) -> None:
        if self._loaded:
            return
        if self.dit is None:
            transformer_path = self.model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = self._resolve_safetensors(self.model_dir, "transformer-distilled")
            self.dit = self._load_transformer_with_optional_streaming(transformer_path)
        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()
        self._loaded = True

    def generate_two_stage(  # type: ignore[override]
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        image: str | None = None,
        images=None,
        prompt_relay=None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        # Prompt Relay is preserved for both 2.3 and 2.5.
        encode_prompt, relay_token_ranges = self._prompt_relay_setup(prompt, prompt_relay)

        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(encode_prompt)
            _materialize(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        num_text_tokens = video_embeds.shape[1]
        relay_mask = self._prompt_relay_mask_builder(prompt_relay, relay_token_ranges, num_text_tokens)

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # This capability is a checkpoint architecture flag introduced by 2.5,
        # so it is safer than guessing from a local directory/repository name.
        is_v25 = bool(getattr(self.dit.config, "use_keyframes_abs_pos_embedding", False))
        if is_v25 and self._tile_count is not None:
            raise ValueError(
                "LTX-2.5 modality tiling is not enabled in this compatibility PR yet; "
                "omit --tile-frames/--tile-spatial. Block streaming (--low-ram) remains supported."
            )

        height, width = snap_output_dimensions(height, width, two_stage=True)
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)
        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        resolved_images = list(images) if images else []
        if image is not None and not resolved_images:
            resolved_images = [ImageConditioningInput(path=image, frame_idx=0, strength=1.0)]
        conditionings_1: list = []
        if resolved_images:
            conditionings_1 = combined_image_conditionings(
                resolved_images,
                enc_h=enc_h_half,
                enc_w=enc_w_half,
                spatial_dims=(F, H_half, W_half),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=conditionings_1,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
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

        stage1_dit = self.dit
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_1 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_half, W_half))
            stage1_dit = TiledLTXModel(self.dit, tiler_1)
        x0_model = X0Model(stage1_dit)

        self._pre_denoise_flush(video_state, audio_state)
        if is_v25:
            keyframes_mask_1 = first_latent_frame_keyframes_mask(
                video_state.latent.shape[1],
                H_half * W_half,
                batch=video_state.latent.shape[0],
            )
            output_1 = denoise_loop_v25(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=list(sigmas_1),
                keyframes_mask=keyframes_mask_1,
                video_cross_attention_mask=relay_mask(F, H_half, W_half, video_state.latent.shape[1]),
                ancestral_eta=1.0,
                ancestral_s_noise=1.0,
                noise_seed=seed + 10000,
                on_step=self._stepwise_hook(F, H_half, W_half, stage=1),
            )
        else:
            output_1 = denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
                video_cross_attention_mask=relay_mask(F, H_half, W_half, video_state.latent.shape[1]),
                on_step=self._stepwise_hook(F, H_half, W_half, stage=1),
            )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_1 = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2
        conditionings_2: list = []
        if resolved_images:
            conditionings_2 = combined_image_conditionings(
                resolved_images,
                enc_h=H_full * 32,
                enc_w=W_full * 32,
                spatial_dims=(F, H_full, W_full),
                video_encoder=self.vae_encoder,
                frame_rate=frame_rate,
            )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None
            aggressive_cleanup()

        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]
        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)
        video_state_2 = create_noised_state(
            base_shape=video_tokens.shape,
            conditionings=conditionings_2,
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

        stage2_x0_model = x0_model
        if self._tile_count is not None:
            from ltx_core_mlx.components.modality_tiling import TiledLTXModel, VideoModalityTiler

            tiler_2 = VideoModalityTiler(self._tile_count, latent_shape=(F, H_full, W_full))
            stage2_x0_model = X0Model(TiledLTXModel(self.dit, tiler_2))

        self._pre_denoise_flush(video_state_2, audio_state_2)
        if is_v25:
            keyframes_mask_2 = first_latent_frame_keyframes_mask(
                video_state_2.latent.shape[1],
                H_full * W_full,
                batch=video_state_2.latent.shape[0],
            )
            output_2 = denoise_loop_v25(
                model=stage2_x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=list(sigmas_2),
                keyframes_mask=keyframes_mask_2,
                video_cross_attention_mask=relay_mask(F, H_full, W_full, video_state_2.latent.shape[1]),
                ancestral_eta=0.0,
                noise_seed=seed + 10000,
                on_step=self._stepwise_hook(F, H_full, W_full, stage=2),
            )
        else:
            output_2 = denoise_loop(
                model=stage2_x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_2,
                video_cross_attention_mask=relay_mask(F, H_full, W_full, video_state_2.latent.shape[1]),
                on_step=self._stepwise_hook(F, H_full, W_full, stage=2),
            )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_2 = output_2.video_latent[:, : F * H_full * W_full, :]
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)
        return video_latent, audio_latent


__all__ = ["DistilledPipeline"]
