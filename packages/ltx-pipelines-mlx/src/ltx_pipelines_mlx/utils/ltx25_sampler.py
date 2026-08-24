"""LTX-2.5 sampler compatibility helpers.

Kept separate from the mature 2.3 samplers so adding 2.5 does not change
existing CFG/STG/res2s behavior for pre-2.5 checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
from mlx_arsenal.diffusion import euler_step
from tqdm import tqdm

from ltx_core_mlx.components.guiders import MultiModalGuiderFactory
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, apply_denoise_mask
from ltx_core_mlx.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup

OnStepFn = Callable[[int, int, mx.array, float], None]


@dataclass
class DenoiseOutput:
    video_latent: mx.array
    audio_latent: mx.array


def first_latent_frame_keyframes_mask(
    total_tokens: int,
    tokens_per_latent_frame: int,
    *,
    batch: int = 1,
) -> mx.array:
    """Mark the generated first latent frame for LTX-2.5's keyframe embedding."""
    if tokens_per_latent_frame <= 0 or tokens_per_latent_frame > total_tokens:
        raise ValueError("tokens_per_latent_frame must be in [1, total_tokens]")
    head = mx.ones((batch, tokens_per_latent_frame, 1), dtype=mx.bfloat16)
    tail = mx.zeros((batch, total_tokens - tokens_per_latent_frame, 1), dtype=mx.bfloat16)
    return mx.concatenate([head, tail], axis=1)


def _uniform(mask: mx.array) -> bool:
    return bool(mx.all(mask == 1.0).item())


def _per_token_timesteps(sigma: float, mask: mx.array) -> mx.array:
    return (mask * sigma).squeeze(-1).astype(mx.float32)


def euler_ancestral_step(
    x: mx.array,
    x0: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array | None,
    *,
    eta: float = 1.0,
    s_noise: float = 1.0,
) -> mx.array:
    """Upstream LTX-2.5 ancestral Euler step in rectified-flow coordinates."""
    if sigma_next == 0:
        return x0.astype(x.dtype)
    if eta > 0 and noise is None:
        raise ValueError("ancestral Euler requires noise when eta > 0")

    xf = x.astype(mx.float32)
    x0f = x0.astype(mx.float32)
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * xf + (1.0 - sigma_down_ratio) * x0f

    if eta > 0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
        renoise_coeff = max(renoise_sq, 0.0) ** 0.5
        x_next = (
            (alpha_next / alpha_down) * x_next
            + noise.astype(mx.float32) * s_noise * renoise_coeff
        )
    return x_next.astype(x.dtype)


def denoise_loop_v25(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    *,
    sigmas: list[float],
    keyframes_mask: mx.array,
    video_cross_attention_mask: mx.array | None = None,
    ancestral_eta: float = 0.0,
    ancestral_s_noise: float = 1.0,
    noise_seed: int = 0,
    show_progress: bool = True,
    on_step: OnStepFn | None = None,
) -> DenoiseOutput:
    """Euler loop with the LTX-2.5 keyframe mask and optional ancestral noise."""
    video_positions = video_state.positions
    audio_positions = audio_state.positions
    video_attention_mask = video_state.attention_mask
    audio_attention_mask = audio_state.attention_mask
    video_x = video_state.latent
    audio_x = audio_state.latent

    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising (LTX-2.5)", disable=not show_progress)
    video_uniform = _uniform(video_state.denoise_mask)
    audio_uniform = _uniform(audio_state.denoise_mask)
    rng_key = mx.random.key(noise_seed)

    for step_idx, (sigma, sigma_next) in enumerate(iterator):
        sigma_arr = mx.array([sigma], dtype=mx.float32)
        batch = video_x.shape[0]
        kwargs: dict = {
            "video_latent": video_x,
            "audio_latent": audio_x,
            "sigma": mx.broadcast_to(sigma_arr, (batch,)),
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "video_attention_mask": video_attention_mask,
            "audio_attention_mask": audio_attention_mask,
            "keyframes_mask": keyframes_mask,
        }
        if video_cross_attention_mask is not None:
            kwargs["video_cross_attention_mask"] = video_cross_attention_mask
        if not video_uniform:
            kwargs["video_timesteps"] = _per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            kwargs["audio_timesteps"] = _per_token_timesteps(sigma, audio_state.denoise_mask)

        video_x0, audio_x0 = model(**kwargs)
        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        if on_step is not None:
            on_step(step_idx, len(steps), video_x0, sigma)

        if ancestral_eta > 0 and sigma_next != 0:
            rng_key, video_key = mx.random.split(rng_key)
            rng_key, audio_key = mx.random.split(rng_key)
            video_noise = mx.random.normal(video_x.shape, key=video_key)
            audio_noise = mx.random.normal(audio_x.shape, key=audio_key)
            video_x = euler_ancestral_step(
                video_x,
                video_x0,
                sigma,
                sigma_next,
                video_noise,
                eta=ancestral_eta,
                s_noise=ancestral_s_noise,
            )
            audio_x = euler_ancestral_step(
                audio_x,
                audio_x0,
                sigma,
                sigma_next,
                audio_noise,
                eta=ancestral_eta,
                s_noise=ancestral_s_noise,
            )
            # Conditioning must remain clean after the re-noise operation.
            video_x = apply_denoise_mask(video_x, video_state.clean_latent, video_state.denoise_mask)
            audio_x = apply_denoise_mask(audio_x, audio_state.clean_latent, audio_state.denoise_mask)
        else:
            video_x = euler_step(video_x, video_x0, sigma, sigma_next)
            audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)

        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()
    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


def guided_denoise_loop_v25(
    model: X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    video_guider_factory: MultiModalGuiderFactory,
    *,
    audio_guider_factory: MultiModalGuiderFactory | None = None,
    sigmas: list[float],
    keyframes_mask: mx.array,
    show_progress: bool = True,
    on_step: OnStepFn | None = None,
) -> DenoiseOutput:
    """LTX-2.5 guided Euler loop used by dev-model A2V stage 1.

    This mirrors the existing guided sampler but keeps sigma in fp32 before
    the DiT timestep embedding and injects the mandatory 2.5 first-frame
    keyframe mask into every cond/uncond/STG/modality pass.
    """
    if audio_guider_factory is None:
        audio_guider_factory = MultiModalGuiderFactory(
            negative_context=None,
            _params_by_sigma=video_guider_factory._params_by_sigma,
        )

    video_x = video_state.latent
    audio_x = audio_state.latent
    steps = list(zip(sigmas[:-1], sigmas[1:]))
    iterator = tqdm(steps, desc="Denoising (LTX-2.5 guided)", disable=not show_progress)
    video_uniform = _uniform(video_state.denoise_mask)
    audio_uniform = _uniform(audio_state.denoise_mask)
    last_video_x0: mx.array | None = None
    last_audio_x0: mx.array | None = None

    for step_idx, (sigma, sigma_next) in enumerate(iterator):
        video_guider = video_guider_factory.build_from_sigma(sigma)
        audio_guider = audio_guider_factory.build_from_sigma(sigma)

        if (
            video_guider.should_skip_step(step_idx)
            and audio_guider.should_skip_step(step_idx)
            and last_video_x0 is not None
            and last_audio_x0 is not None
        ):
            video_x = euler_step(video_x, last_video_x0, sigma, sigma_next)
            audio_x = euler_step(audio_x, last_audio_x0, sigma, sigma_next)
            mx.async_eval(video_x, audio_x)
            continue

        sigma_arr = mx.array([sigma], dtype=mx.float32)
        batch = video_x.shape[0]
        base_kwargs: dict = {
            "video_latent": video_x,
            "audio_latent": audio_x,
            "sigma": mx.broadcast_to(sigma_arr, (batch,)),
            "video_positions": video_state.positions,
            "audio_positions": audio_state.positions,
            "video_attention_mask": video_state.attention_mask,
            "audio_attention_mask": audio_state.attention_mask,
            "keyframes_mask": keyframes_mask,
        }
        if not video_uniform:
            base_kwargs["video_timesteps"] = _per_token_timesteps(sigma, video_state.denoise_mask)
        if not audio_uniform:
            base_kwargs["audio_timesteps"] = _per_token_timesteps(sigma, audio_state.denoise_mask)

        cond_v, cond_a = model(
            **base_kwargs,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
        )

        neg_v: mx.array | float = 0.0
        neg_a: mx.array | float = 0.0
        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            neg_video = (
                video_guider.negative_context
                if video_guider.negative_context is not None
                else video_text_embeds
            )
            neg_audio = (
                audio_guider.negative_context
                if audio_guider.negative_context is not None
                else audio_text_embeds
            )
            neg_v, neg_a = model(
                **base_kwargs,
                video_text_embeds=neg_video,
                audio_text_embeds=neg_audio,
            )

        ptb_v: mx.array | float = 0.0
        ptb_a: mx.array | float = 0.0
        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbations: list[Perturbation] = []
            if video_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=video_guider.params.stg_blocks,
                    )
                )
            if audio_guider.do_perturbed_generation():
                perturbations.append(
                    Perturbation(
                        type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        blocks=audio_guider.params.stg_blocks,
                    )
                )
            config = PerturbationConfig(perturbations=perturbations)
            ptb_v, ptb_a = model(
                **base_kwargs,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
                perturbations=BatchedPerturbationConfig(perturbations=[config] * batch),
            )

        mod_v: mx.array | float = 0.0
        mod_a: mx.array | float = 0.0
        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            config = PerturbationConfig(
                perturbations=[
                    Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                    Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                ]
            )
            mod_v, mod_a = model(
                **base_kwargs,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
                perturbations=BatchedPerturbationConfig(perturbations=[config] * batch),
            )

        if video_guider.should_skip_step(step_idx) and last_video_x0 is not None:
            video_x0 = last_video_x0
        else:
            video_x0 = video_guider.calculate(cond_v, neg_v, ptb_v, mod_v)

        if audio_guider.should_skip_step(step_idx) and last_audio_x0 is not None:
            audio_x0 = last_audio_x0
        else:
            audio_x0 = audio_guider.calculate(cond_a, neg_a, ptb_a, mod_a)

        video_x0 = apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)
        last_video_x0 = video_x0
        last_audio_x0 = audio_x0

        if on_step is not None:
            on_step(step_idx, len(steps), video_x0, sigma)

        video_x = euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = euler_step(audio_x, audio_x0, sigma, sigma_next)
        mx.async_eval(video_x, audio_x)

    aggressive_cleanup()
    return DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


__all__ = [
    "DenoiseOutput",
    "denoise_loop_v25",
    "euler_ancestral_step",
    "first_latent_frame_keyframes_mask",
    "guided_denoise_loop_v25",
]
