"""Unit coverage for LTX-2.5 architecture/sampler compatibility."""

import numpy as np
import mlx.core as mx
import pytest

from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import Gemma4LanguageModel
from ltx_pipelines_mlx.utils.ltx25_sampler import (
    euler_ancestral_step,
    first_latent_frame_keyframes_mask,
    guided_denoise_loop_v25,
)


def test_ltx25_config_reads_architecture_deltas():
    cfg = LTXModelConfig.from_checkpoint_config(
        {
            "transformer": {
                "ff_bias": False,
                "audio_ff_bias": True,
                "use_keyframes_abs_pos_embedding": True,
                "av_ca_timestep_scale_multiplier": 1000.0,
            }
        }
    )
    assert cfg.ff_bias is False
    assert cfg.audio_ff_bias is True
    assert cfg.use_keyframes_abs_pos_embedding is True
    assert cfg.av_ca_timestep_scale_multiplier == 1000.0


def test_pre25_defaults_stay_backward_compatible():
    cfg = LTXModelConfig.from_checkpoint_config({"transformer": {}})
    assert cfg.ff_bias is True
    assert cfg.audio_ff_bias is True
    assert cfg.use_keyframes_abs_pos_embedding is False


def test_first_frame_keyframe_mask_marks_only_first_frame():
    mask = first_latent_frame_keyframes_mask(12, 4, batch=2)
    arr = np.asarray(mask)
    assert arr.shape == (2, 12, 1)
    assert np.all(arr[:, :4, :] == 1)
    assert np.all(arr[:, 4:, :] == 0)


def test_ancestral_terminal_step_returns_x0():
    x = mx.zeros((1, 2, 3), dtype=mx.bfloat16)
    x0 = mx.ones((1, 2, 3), dtype=mx.bfloat16)
    out = euler_ancestral_step(x, x0, 0.5, 0.0, None)
    assert np.allclose(np.asarray(out), np.asarray(x0))


def test_ancestral_requires_noise_when_enabled():
    x = mx.zeros((1, 2, 3))
    x0 = mx.ones((1, 2, 3))
    with pytest.raises(ValueError, match="requires noise"):
        euler_ancestral_step(x, x0, 0.8, 0.4, None, eta=1.0)


class _FakeTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, _text):
        return list(range(10, 20))


def test_gemma4_keeps_prompt_head_when_truncating():
    model = Gemma4LanguageModel()
    model._tokenizer = _FakeTokenizer()
    tokens, mask = model.tokenize("ignored", max_length=5)
    assert np.asarray(tokens).tolist() == [[2, 10, 11, 12, 13]]
    assert np.asarray(mask).tolist() == [[1, 1, 1, 1, 1]]


class _RecordingX0:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["video_latent"], kwargs["audio_latent"]


def test_ltx25_a2v_guided_path_passes_keyframe_mask_and_fp32_sigma():
    video = mx.zeros((1, 4, 2), dtype=mx.bfloat16)
    audio = mx.zeros((1, 2, 2), dtype=mx.bfloat16)
    video_state = LatentState(
        latent=video,
        clean_latent=mx.zeros_like(video),
        denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
        positions=mx.zeros((1, 4, 3)),
    )
    audio_state = LatentState(
        latent=audio,
        clean_latent=audio,
        denoise_mask=mx.zeros((1, 2, 1), dtype=mx.bfloat16),
        positions=mx.zeros((1, 2, 1)),
    )
    keyframes_mask = first_latent_frame_keyframes_mask(4, 2)
    negative = mx.zeros((1, 1, 2), dtype=mx.bfloat16)
    video_factory = create_multimodal_guider_factory(
        MultiModalGuiderParams(cfg_scale=2.0),
        negative_context=negative,
    )
    audio_factory = create_multimodal_guider_factory(MultiModalGuiderParams())
    model = _RecordingX0()

    guided_denoise_loop_v25(
        model=model,
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((1, 1, 2)),
        audio_text_embeds=mx.zeros((1, 1, 2)),
        video_guider_factory=video_factory,
        audio_guider_factory=audio_factory,
        sigmas=[1.0, 0.0],
        keyframes_mask=keyframes_mask,
        show_progress=False,
    )

    # CFG makes both a conditioned and an unconditional call. Every pass must
    # carry the 2.5 first-frame marker and an unrounded fp32 sigma.
    assert len(model.calls) == 2
    for call in model.calls:
        assert np.array_equal(np.asarray(call["keyframes_mask"]), np.asarray(keyframes_mask))
        assert call["sigma"].dtype == mx.float32
