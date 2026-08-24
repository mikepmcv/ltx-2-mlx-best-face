"""BasicAVTransformerBlock -- joint audio+video transformer block.

Ported from ltx-core/src/ltx_core/model/transformer/transformer.py

Per-block weight keys (under ``transformer_blocks.N``):
    attn1, audio_attn1               -- self-attention (video / audio)
    attn2, audio_attn2               -- text cross-attention (video / audio)
    audio_to_video_attn              -- A->V cross-modal attention
    video_to_audio_attn              -- V->A cross-modal attention
    ff, audio_ff                     -- feed-forward (video / audio)
    scale_shift_table                -- (9, video_dim)  video self-attn AdaLN
    audio_scale_shift_table          -- (9, audio_dim)  audio self-attn AdaLN
    prompt_scale_shift_table         -- (2, video_dim)  video text cross-attn
    audio_prompt_scale_shift_table   -- (2, audio_dim)  audio text cross-attn
    scale_shift_table_a2v_ca_video   -- (5, video_dim)  AV cross-attn video side
    scale_shift_table_a2v_ca_audio   -- (5, audio_dim)  AV cross-attn audio side
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.feed_forward import FeedForward


class BasicAVTransformerBlock(nn.Module):
    """Joint audio+video transformer block with adaptive layer norm.

    Each block processes both video and audio tokens through:
      1. Self-attention (video & audio, with AdaLN from per-block tables + timestep)
      2. Text cross-attention (video & audio)
      3. Audio-video cross-modal attention (bidirectional)
      4. Feed-forward (video & audio)

    Args:
        video_dim: Video hidden dimension (default 4096).
        audio_dim: Audio hidden dimension (default 2048).
        video_num_heads: Number of attention heads for video (default 32).
        audio_num_heads: Number of attention heads for audio (default 32).
        video_head_dim: Per-head dimension for video (default 128).
        audio_head_dim: Per-head dimension for audio (default 64).
        av_cross_num_heads: Heads for cross-modal attention (default 32).
        av_cross_head_dim: Per-head dim for cross-modal attention (default 64).
        ff_mult: Feed-forward expansion factor.
        norm_eps: Epsilon for layer norms.
    """

    def __init__(
        self,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        video_num_heads: int = 32,
        audio_num_heads: int = 32,
        video_head_dim: int = 128,
        audio_head_dim: int = 64,
        av_cross_num_heads: int = 32,
        av_cross_head_dim: int = 64,
        ff_mult: float = 4.0,
        norm_eps: float = 1e-6,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=video_dim,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.audio_attn1 = Attention(
            query_dim=audio_dim,
            num_heads=audio_num_heads,
            head_dim=audio_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.attn2 = Attention(
            query_dim=video_dim,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            use_rope=False,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.audio_attn2 = Attention(
            query_dim=audio_dim,
            num_heads=audio_num_heads,
            head_dim=audio_head_dim,
            use_rope=False,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.audio_to_video_attn = Attention(
            query_dim=video_dim,
            kv_dim=audio_dim,
            out_dim=video_dim,
            num_heads=av_cross_num_heads,
            head_dim=av_cross_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.video_to_audio_attn = Attention(
            query_dim=audio_dim,
            kv_dim=video_dim,
            out_dim=audio_dim,
            num_heads=av_cross_num_heads,
            head_dim=av_cross_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )
        self.ff = FeedForward(video_dim, dim_out=video_dim, mult=ff_mult, bias=ff_bias)
        self.audio_ff = FeedForward(audio_dim, dim_out=audio_dim, mult=ff_mult, bias=audio_ff_bias)

        self.scale_shift_table = mx.zeros((9, video_dim))
        self.audio_scale_shift_table = mx.zeros((9, audio_dim))
        self.prompt_scale_shift_table = mx.zeros((2, video_dim))
        self.audio_prompt_scale_shift_table = mx.zeros((2, audio_dim))
        self.scale_shift_table_a2v_ca_video = mx.zeros((5, video_dim))
        self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio_dim))
        self._norm_eps = norm_eps

    @staticmethod
    def _unpack_adaln(params: mx.array, table: mx.array, num_params: int, dim: int) -> list[mx.array]:
        if params.ndim == 2:
            p = params.reshape(-1, num_params, dim)
            p = p + table[None, :num_params, :]
            return [p[:, i, :][:, None, :] for i in range(num_params)]
        B, N, _ = params.shape
        p = params.reshape(B, N, num_params, dim)
        p = p + table[None, None, :num_params, :]
        return [p[:, :, i, :] for i in range(num_params)]

    def _rms_norm(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight=None, eps=self._norm_eps)

    def compute_video_normed_sa(self, video_hidden: mx.array, video_adaln_params: mx.array) -> mx.array:
        vdim = video_hidden.shape[-1]
        v_shift_sa, v_scale_sa, *_ = self._unpack_adaln(video_adaln_params, self.scale_shift_table, 9, vdim)
        return self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa

    def __call__(
        self,
        video_hidden: mx.array,
        audio_hidden: mx.array | None,
        video_adaln_params: mx.array,
        audio_adaln_params: mx.array,
        video_prompt_adaln_params: mx.array,
        audio_prompt_adaln_params: mx.array,
        av_ca_video_params: mx.array,
        av_ca_audio_params: mx.array,
        av_ca_a2v_gate_params: mx.array,
        av_ca_v2a_gate_params: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_rope_freqs: mx.array | None = None,
        audio_rope_freqs: mx.array | None = None,
        video_cross_rope_freqs: mx.array | None = None,
        audio_cross_rope_freqs: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_cross_attention_mask: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        block_idx: int = 0,
    ) -> tuple[mx.array, mx.array]:
        run_ax = audio_hidden is not None
        vdim = video_hidden.shape[-1]
        adim = audio_hidden.shape[-1] if run_ax else 0

        (v_shift_sa, v_scale_sa, v_gate_sa, v_shift_ff, v_scale_ff, v_gate_ff, v_shift_ca, v_scale_ca, v_gate_ca) = (
            self._unpack_adaln(video_adaln_params, self.scale_shift_table, 9, vdim)
        )
        if run_ax:
            (
                a_shift_sa,
                a_scale_sa,
                a_gate_sa,
                a_shift_ff,
                a_scale_ff,
                a_gate_ff,
                a_shift_ca,
                a_scale_ca,
                a_gate_ca,
            ) = self._unpack_adaln(audio_adaln_params, self.audio_scale_shift_table, 9, adim)

        (av_v_scale_a2v, av_v_shift_a2v, av_v_scale_v2a, av_v_shift_v2a) = self._unpack_adaln(
            av_ca_video_params, self.scale_shift_table_a2v_ca_video, 4, vdim
        )
        if av_ca_a2v_gate_params.ndim == 2:
            av_v_gate_a2v = (av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[4, :])[:, None, :]
        else:
            av_v_gate_a2v = av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[None, None, 4, :]

        if run_ax:
            (av_a_scale_a2v, av_a_shift_a2v, av_a_scale_v2a, av_a_shift_v2a) = self._unpack_adaln(
                av_ca_audio_params, self.scale_shift_table_a2v_ca_audio, 4, adim
            )
            if av_ca_v2a_gate_params.ndim == 2:
                av_a_gate_v2a = (av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[4, :])[:, None, :]
            else:
                av_a_gate_v2a = av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[None, None, 4, :]

        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa
        v_ptb_mask = None
        if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block_idx):
            v_ptb_mask = perturbations.mask_like(
                PerturbationType.SKIP_VIDEO_SELF_ATTN, block_idx, video_hidden[:, :1, :1, None]
            )
        video_sa_out = self.attn1(
            video_normed,
            rope_freqs=video_rope_freqs,
            attention_mask=video_attention_mask,
            perturbation_mask=v_ptb_mask,
        )
        video_hidden = video_hidden + video_sa_out * v_gate_sa

        if run_ax:
            audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_sa) + a_shift_sa
            a_ptb_mask = None
            if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block_idx):
                a_ptb_mask = perturbations.mask_like(
                    PerturbationType.SKIP_AUDIO_SELF_ATTN, block_idx, audio_hidden[:, :1, :1, None]
                )
            audio_sa_out = self.audio_attn1(
                audio_normed,
                rope_freqs=audio_rope_freqs,
                attention_mask=audio_attention_mask,
                perturbation_mask=a_ptb_mask,
            )
            audio_hidden = audio_hidden + audio_sa_out * a_gate_sa

        if video_text_embeds is not None:
            video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ca) + v_shift_ca
            vp_shift, vp_scale = self._unpack_adaln(video_prompt_adaln_params, self.prompt_scale_shift_table, 2, vdim)
            text_scaled = video_text_embeds * (1.0 + vp_scale) + vp_shift
            video_hidden = (
                video_hidden
                + self.attn2(
                    video_normed,
                    encoder_hidden_states=text_scaled,
                    attention_mask=video_cross_attention_mask,
                )
                * v_gate_ca
            )

        if run_ax and audio_text_embeds is not None:
            audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ca) + a_shift_ca
            ap_shift, ap_scale = self._unpack_adaln(
                audio_prompt_adaln_params, self.audio_prompt_scale_shift_table, 2, adim
            )
            text_scaled = audio_text_embeds * (1.0 + ap_scale) + ap_shift
            audio_hidden = audio_hidden + self.audio_attn2(audio_normed, encoder_hidden_states=text_scaled) * a_gate_ca

        video_norm3 = self._rms_norm(video_hidden)
        audio_norm3 = self._rms_norm(audio_hidden) if run_ax else None

        if run_ax:
            video_q_a2v = video_norm3 * (1.0 + av_v_scale_a2v) + av_v_shift_a2v
            audio_kv_a2v = audio_norm3 * (1.0 + av_a_scale_a2v) + av_a_shift_a2v
            a2v_out = (
                self.audio_to_video_attn(
                    video_q_a2v,
                    encoder_hidden_states=audio_kv_a2v,
                    rope_freqs=video_cross_rope_freqs,
                    rope_freqs_k=audio_cross_rope_freqs,
                )
                * av_v_gate_a2v
            )
            if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx):
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx, video_hidden)
                a2v_out = a2v_out * a2v_mask
            video_hidden = video_hidden + a2v_out

            audio_q_v2a = audio_norm3 * (1.0 + av_a_scale_v2a) + av_a_shift_v2a
            video_kv_v2a = video_norm3 * (1.0 + av_v_scale_v2a) + av_v_shift_v2a
            v2a_out = (
                self.video_to_audio_attn(
                    audio_q_v2a,
                    encoder_hidden_states=video_kv_v2a,
                    rope_freqs=audio_cross_rope_freqs,
                    rope_freqs_k=video_cross_rope_freqs,
                )
                * av_a_gate_v2a
            )
            if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx):
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx, audio_hidden)
                v2a_out = v2a_out * v2a_mask
            audio_hidden = audio_hidden + v2a_out

        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ff) + v_shift_ff
        video_hidden = video_hidden + self.ff(video_normed) * v_gate_ff

        if run_ax:
            audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ff) + a_shift_ff
            audio_hidden = audio_hidden + self.audio_ff(audio_normed) * a_gate_ff

        return video_hidden, audio_hidden
