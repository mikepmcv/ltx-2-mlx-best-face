"""LTX diffusion transformer for joint audio/video generation on MLX.

Supports both LTX-2.3 and LTX-2.5 checkpoint architecture metadata.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.guidance.perturbations import BatchedPerturbationConfig
from ltx_core_mlx.model.transformer.adaln import AdaLayerNormSingle
from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

_DIT_EVAL_EVERY = int(_os.environ.get("LTX2_DIT_EVAL_EVERY", "8"))
_mx_eval = getattr(mx, "eval")  # noqa: B009


class Modality(Enum):
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class LTXModelConfig:
    """Configuration for LTXModel, populated from checkpoint metadata when available."""

    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_num_heads: int = 32
    audio_num_heads: int = 32
    video_head_dim: int = 128
    audio_head_dim: int = 64
    av_cross_num_heads: int = 32
    av_cross_head_dim: int = 64
    video_patch_channels: int = 128
    audio_patch_channels: int = 128
    ff_mult: float = 4.0
    timestep_embedding_dim: int = 256
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: int = 1
    rope_theta: float = 10000.0
    rope_type: str = "split"
    positional_embedding_max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6
    # LTX-2.5 deltas. Defaults preserve 2.3 behavior.
    ff_bias: bool = True
    audio_ff_bias: bool = True
    use_keyframes_abs_pos_embedding: bool = False

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> LTXModelConfig:
        t = config.get("transformer", config)
        d = cls()
        return cls(
            num_layers=t.get("num_layers", d.num_layers),
            video_dim=t.get("cross_attention_dim", d.video_dim),
            audio_dim=t.get("audio_cross_attention_dim", d.audio_dim),
            video_num_heads=t.get("num_attention_heads", d.video_num_heads),
            audio_num_heads=t.get("audio_num_attention_heads", d.audio_num_heads),
            video_head_dim=t.get("attention_head_dim", d.video_head_dim),
            audio_head_dim=t.get("audio_attention_head_dim", d.audio_head_dim),
            av_cross_num_heads=t.get("audio_num_attention_heads", d.av_cross_num_heads),
            av_cross_head_dim=t.get("audio_attention_head_dim", d.av_cross_head_dim),
            video_patch_channels=t.get("in_channels", d.video_patch_channels),
            audio_patch_channels=t.get("audio_in_channels", d.audio_patch_channels),
            timestep_scale_multiplier=t.get("timestep_scale_multiplier", d.timestep_scale_multiplier),
            av_ca_timestep_scale_multiplier=t.get(
                "av_ca_timestep_scale_multiplier", d.av_ca_timestep_scale_multiplier
            ),
            rope_theta=t.get("positional_embedding_theta", d.rope_theta),
            rope_type=t.get("rope_type", d.rope_type),
            positional_embedding_max_pos=tuple(
                t.get("positional_embedding_max_pos", d.positional_embedding_max_pos)
            ),
            audio_positional_embedding_max_pos=tuple(
                t.get("audio_positional_embedding_max_pos", d.audio_positional_embedding_max_pos)
            ),
            norm_eps=t.get("norm_eps", d.norm_eps),
            ff_bias=t.get("ff_bias", d.ff_bias),
            audio_ff_bias=t.get("audio_ff_bias", d.audio_ff_bias),
            use_keyframes_abs_pos_embedding=t.get(
                "use_keyframes_abs_pos_embedding", d.use_keyframes_abs_pos_embedding
            ),
        )

    @classmethod
    def from_checkpoint_dir(cls, model_dir) -> LTXModelConfig:
        import json
        import sys
        from pathlib import Path

        model_dir = Path(model_dir)
        for name in ("embedded_config.json", "config.json"):
            path = model_dir / name
            if not path.exists():
                continue
            try:
                return cls.from_checkpoint_config(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"warning: failed to read {path}: {exc}; using defaults", file=sys.stderr)
                return cls()
        print(
            f"warning: no transformer config found in {model_dir}; using hardcoded defaults",
            file=sys.stderr,
        )
        return cls()


class LTXModel(nn.Module):
    """Joint audio/video diffusion transformer shared by LTX-2.3 and LTX-2.5."""

    def __init__(self, config: LTXModelConfig | None = None):
        super().__init__()
        self.config = config or LTXModelConfig()
        config = self.config
        vd = config.video_dim
        ad = config.audio_dim
        t_dim = config.timestep_embedding_dim

        self.patchify_proj = nn.Linear(config.video_patch_channels, vd)
        self.audio_patchify_proj = nn.Linear(config.audio_patch_channels, ad)

        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = mx.zeros((1, vd))

        self.proj_out = nn.Linear(vd, config.video_patch_channels)
        self.audio_proj_out = nn.Linear(ad, config.audio_patch_channels)
        self.scale_shift_table = mx.zeros((2, vd))
        self.audio_scale_shift_table = mx.zeros((2, ad))

        self.adaln_single = AdaLayerNormSingle(vd, num_params=9, timestep_dim=t_dim)
        self.audio_adaln_single = AdaLayerNormSingle(ad, num_params=9, timestep_dim=t_dim)
        self.prompt_adaln_single = AdaLayerNormSingle(vd, num_params=2, timestep_dim=t_dim)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(ad, num_params=2, timestep_dim=t_dim)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(vd, num_params=4, timestep_dim=t_dim)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(ad, num_params=4, timestep_dim=t_dim)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(vd, num_params=1, timestep_dim=t_dim)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(ad, num_params=1, timestep_dim=t_dim)

        self.transformer_blocks = [
            BasicAVTransformerBlock(
                video_dim=vd,
                audio_dim=ad,
                video_num_heads=config.video_num_heads,
                audio_num_heads=config.audio_num_heads,
                video_head_dim=config.video_head_dim,
                audio_head_dim=config.audio_head_dim,
                av_cross_num_heads=config.av_cross_num_heads,
                av_cross_head_dim=config.av_cross_head_dim,
                ff_mult=config.ff_mult,
                norm_eps=config.norm_eps,
                ff_bias=config.ff_bias,
                audio_ff_bias=config.audio_ff_bias,
            )
            for _ in range(config.num_layers)
        ]
        self.gradient_checkpointing = False

    def _embed_timestep_scalar(self, timestep: mx.array) -> mx.array:
        t_scaled = timestep * self.config.timestep_scale_multiplier
        return get_timestep_embedding(t_scaled, self.config.timestep_embedding_dim)

    def _embed_timestep_per_token(self, per_token_timesteps: mx.array) -> mx.array:
        B, N = per_token_timesteps.shape
        flat = (per_token_timesteps * self.config.timestep_scale_multiplier).reshape(-1)
        emb = get_timestep_embedding(flat, self.config.timestep_embedding_dim)
        return emb.reshape(B, N, -1)

    def _adaln_per_token(
        self,
        adaln_module: AdaLayerNormSingle,
        t_emb_per_token: mx.array,
    ) -> tuple[mx.array, mx.array]:
        B, N, D = t_emb_per_token.shape
        flat = t_emb_per_token.reshape(B * N, D)
        params, embedded = adaln_module(flat)
        return params.reshape(B, N, -1), embedded.reshape(B, N, -1)

    def compute_gate_signal(
        self,
        video_latent: mx.array,
        audio_latent: mx.array | None,
        timestep: mx.array,
        video_timesteps: mx.array | None = None,
    ) -> mx.array:
        del audio_latent
        video_latent = video_latent.astype(mx.bfloat16)
        # Keep sigma in fp32 through the sinusoidal embedding. Rounding first
        # materially changes high-frequency timestep features in LTX-2.5.
        timestep = timestep.astype(mx.float32)
        video_hidden = self.patchify_proj(video_latent)
        t_emb = self._embed_timestep_scalar(timestep)
        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps.astype(mx.float32))
            video_adaln_emb, _ = self._adaln_per_token(self.adaln_single, vt_emb)
        else:
            video_adaln_emb, _ = self.adaln_single(t_emb)
        return self.transformer_blocks[0].compute_video_normed_sa(video_hidden, video_adaln_emb)

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array | None,
        timestep: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_positions: mx.array | None = None,
        audio_positions: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_cross_attention_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        keyframes_mask: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        block_provider: callable | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        _dt = mx.float32 if _os.environ.get("LTX2_DIT_FP32") else mx.bfloat16
        video_latent = video_latent.astype(_dt)
        run_ax = audio_latent is not None
        audio_latent = audio_latent.astype(_dt) if run_ax else None
        if video_text_embeds is not None:
            video_text_embeds = video_text_embeds.astype(_dt)
        if audio_text_embeds is not None:
            audio_text_embeds = audio_text_embeds.astype(_dt)

        video_hidden = self.patchify_proj(video_latent)
        audio_hidden = self.audio_patchify_proj(audio_latent) if run_ax else None

        kf_emb = getattr(self, "keyframes_abs_pos_embedding", None)
        if kf_emb is not None and keyframes_mask is not None:
            kf = (keyframes_mask > 0).astype(video_hidden.dtype)
            video_hidden = video_hidden + kf * kf_emb.astype(video_hidden.dtype)

        timestep = timestep.astype(mx.float32)
        t_emb = self._embed_timestep_scalar(timestep)
        av_ca_factor = self.config.av_ca_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        t_emb_av_gate = get_timestep_embedding(
            timestep * self.config.timestep_scale_multiplier * av_ca_factor,
            self.config.timestep_embedding_dim,
        )

        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps.astype(mx.float32))
            video_adaln_emb, video_embedded_ts = self._adaln_per_token(self.adaln_single, vt_emb)
            av_ca_video_emb, _ = self._adaln_per_token(self.av_ca_video_scale_shift_adaln_single, vt_emb)
        else:
            video_adaln_emb, video_embedded_ts = self.adaln_single(t_emb)
            av_ca_video_emb, _ = self.av_ca_video_scale_shift_adaln_single(t_emb)
        av_ca_a2v_gate_emb, _ = self.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
        video_prompt_emb, _ = self.prompt_adaln_single(t_emb)

        if run_ax:
            if audio_timesteps is not None:
                at_emb = self._embed_timestep_per_token(audio_timesteps.astype(mx.float32))
                audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
                av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
            else:
                audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
                av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
            av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
            audio_prompt_emb, _ = self.audio_prompt_adaln_single(t_emb)
        else:
            audio_adaln_emb = None
            audio_embedded_ts = None
            av_ca_audio_emb = None
            av_ca_v2a_gate_emb = None
            audio_prompt_emb = None

        video_rope_freqs = None
        audio_rope_freqs = None
        if video_positions is not None:
            video_rope_freqs = self._compute_rope_freqs(
                video_positions, self.config.video_num_heads, self.config.video_head_dim
            )
        if run_ax and audio_positions is not None:
            audio_rope_freqs = self._compute_rope_freqs(
                audio_positions,
                self.config.audio_num_heads,
                self.config.audio_head_dim,
                max_pos_override=list(self.config.audio_positional_embedding_max_pos),
            )

        video_cross_rope_freqs = None
        audio_cross_rope_freqs = None
        cross_pe_max_pos = max(
            self.config.positional_embedding_max_pos[0],
            self.config.audio_positional_embedding_max_pos[0],
        )
        if video_positions is not None:
            video_cross_rope_freqs = self._compute_rope_freqs(
                video_positions[:, :, 0:1],
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )
        if run_ax and audio_positions is not None:
            audio_cross_rope_freqs = self._compute_rope_freqs(
                audio_positions[:, :, 0:1],
                self.config.av_cross_num_heads,
                self.config.av_cross_head_dim,
                max_pos_override=[cross_pe_max_pos],
            )

        block_input_v = video_hidden
        block_input_a = audio_hidden

        if block_stack_override is not None:
            if not run_ax:
                raise ValueError("block_stack_override is not supported on the audio-free forward")
            video_hidden, audio_hidden = block_stack_override(video_hidden, audio_hidden)
        else:
            num_layers = self.config.num_layers if block_provider is not None else len(self.transformer_blocks)
            for block_idx in range(num_layers):
                block = block_provider(block_idx) if block_provider is not None else self.transformer_blocks[block_idx]

                if self.gradient_checkpointing:
                    def _run_block(params, vh, ah, _block=block, _bidx=block_idx):
                        _block.update(params)
                        return _block(
                            video_hidden=vh,
                            audio_hidden=ah,
                            video_adaln_params=video_adaln_emb,
                            audio_adaln_params=audio_adaln_emb,
                            video_prompt_adaln_params=video_prompt_emb,
                            audio_prompt_adaln_params=audio_prompt_emb,
                            av_ca_video_params=av_ca_video_emb,
                            av_ca_audio_params=av_ca_audio_emb,
                            av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                            av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                            video_text_embeds=video_text_embeds,
                            audio_text_embeds=audio_text_embeds,
                            video_rope_freqs=video_rope_freqs,
                            audio_rope_freqs=audio_rope_freqs,
                            video_cross_rope_freqs=video_cross_rope_freqs,
                            audio_cross_rope_freqs=audio_cross_rope_freqs,
                            video_attention_mask=video_attention_mask,
                            audio_attention_mask=audio_attention_mask,
                            video_cross_attention_mask=video_cross_attention_mask,
                            perturbations=perturbations,
                            block_idx=_bidx,
                        )

                    video_hidden, audio_hidden = mx.checkpoint(_run_block)(
                        block.trainable_parameters(), video_hidden, audio_hidden
                    )
                    continue

                video_hidden, audio_hidden = block(
                    video_hidden=video_hidden,
                    audio_hidden=audio_hidden,
                    video_adaln_params=video_adaln_emb,
                    audio_adaln_params=audio_adaln_emb,
                    video_prompt_adaln_params=video_prompt_emb,
                    audio_prompt_adaln_params=audio_prompt_emb,
                    av_ca_video_params=av_ca_video_emb,
                    av_ca_audio_params=av_ca_audio_emb,
                    av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                    av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                    video_text_embeds=video_text_embeds,
                    audio_text_embeds=audio_text_embeds,
                    video_rope_freqs=video_rope_freqs,
                    audio_rope_freqs=audio_rope_freqs,
                    video_cross_rope_freqs=video_cross_rope_freqs,
                    audio_cross_rope_freqs=audio_cross_rope_freqs,
                    video_attention_mask=video_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    video_cross_attention_mask=video_cross_attention_mask,
                    perturbations=perturbations,
                    block_idx=block_idx,
                )
                if block_provider is not None:
                    _mx_eval(video_hidden, audio_hidden)
                elif _DIT_EVAL_EVERY > 0 and (block_idx + 1) % _DIT_EVAL_EVERY == 0:
                    if audio_hidden is not None:
                        _mx_eval(video_hidden, audio_hidden)
                    else:
                        _mx_eval(video_hidden)

        if tap is not None:
            tap(
                video_hidden - block_input_v,
                (audio_hidden - block_input_a) if run_ax else None,
            )

        video_out = self._output_block(video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out)
        audio_out = (
            self._output_block(audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out)
            if run_ax
            else None
        )
        return video_out, audio_out

    def _output_block(
        self,
        x: mx.array,
        embedded_timestep: mx.array,
        scale_shift_table: mx.array,
        proj: nn.Linear,
    ) -> mx.array:
        if embedded_timestep.ndim == 2:
            embedded_timestep = embedded_timestep[:, None, :]
        scale_shift_values = scale_shift_table[None, None, :, :] + embedded_timestep[:, :, None, :]
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = mx.fast.layer_norm(x, weight=None, bias=None, eps=self.config.norm_eps)
        x = x * (1.0 + scale) + shift
        return proj(x)

    def _compute_rope_freqs(
        self,
        positions: mx.array,
        num_heads: int,
        head_dim: int,
        max_pos_override: list[int] | None = None,
    ) -> mx.array:
        from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs

        inner_dim = num_heads * head_dim
        max_pos = (
            max_pos_override
            if max_pos_override is not None
            else list(self.config.positional_embedding_max_pos[: positions.shape[-1]])
        )
        return precompute_rope_freqs(
            positions,
            inner_dim=inner_dim,
            num_heads=num_heads,
            theta=self.config.rope_theta,
            max_pos=max_pos,
            rope_type=self.config.rope_type,
        )


class X0Model(nn.Module):
    """Wrapper converting velocity prediction to x0 prediction."""

    def __init__(self, model: LTXModel):
        super().__init__()
        self.model = model

    def __call__(
        self,
        video_latent: mx.array,
        audio_latent: mx.array | None,
        sigma: mx.array,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        tap: callable | None = None,
        block_stack_override: callable | None = None,
        **kwargs,
    ) -> tuple[mx.array, mx.array | None]:
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            perturbations=perturbations,
            tap=tap,
            block_stack_override=block_stack_override,
            **kwargs,
        )

        if video_timesteps is not None:
            video_sigma = video_timesteps[:, :, None].astype(mx.float32)
        else:
            video_sigma = sigma[:, None, None].astype(mx.float32)
        video_x0 = (video_latent.astype(mx.float32) - video_sigma * video_v.astype(mx.float32)).astype(
            video_latent.dtype
        )

        if audio_v is None or audio_latent is None:
            return video_x0, None
        if audio_timesteps is not None:
            audio_sigma = audio_timesteps[:, :, None].astype(mx.float32)
        else:
            audio_sigma = sigma[:, None, None].astype(mx.float32)
        audio_x0 = (audio_latent.astype(mx.float32) - audio_sigma * audio_v.astype(mx.float32)).astype(
            audio_latent.dtype
        )
        return video_x0, audio_x0
