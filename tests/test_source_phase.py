"""Tests for Best Face source-phase RoPE math."""

import math

import mlx.core as mx

from ltx_core_mlx.conditioning.source_phase import (
    SourcePhaseBlock,
    apply_source_phase,
    rotate_rope_block,
)


def _to_list(x):
    mx.eval(x)
    return x.tolist()


def test_source_phase_only_rotates_reference_range():
    cos = mx.ones((1, 2, 6, 4), dtype=mx.float32)
    sin = mx.zeros((1, 2, 6, 4), dtype=mx.float32)
    rope = (cos, sin, "split")

    block = SourcePhaseBlock(start=4, length=2, segment_value=2.0)
    new_cos, new_sin, rope_type = rotate_rope_block(rope, block)

    assert rope_type == "split"
    assert _to_list(new_cos[:, :, :4, :]) == _to_list(cos[:, :, :4, :])
    assert _to_list(new_sin[:, :, :4, :]) == _to_list(sin[:, :, :4, :])

    d = mx.arange(4).astype(mx.float32)
    phase = 2.0 * mx.exp((-d / 4.0) * math.log(10000.0))
    expected_cos = mx.cos(phase)
    expected_sin = mx.sin(phase)

    mx.eval(new_cos, new_sin, expected_cos, expected_sin)
    assert mx.allclose(new_cos[0, 0, 4, :], expected_cos, atol=1e-6).item()
    assert mx.allclose(new_sin[0, 0, 4, :], expected_sin, atol=1e-6).item()


def test_multiple_source_phase_blocks_are_independent():
    cos = mx.ones((1, 1, 8, 3), dtype=mx.float32)
    sin = mx.zeros((1, 1, 8, 3), dtype=mx.float32)

    blocks = [
        SourcePhaseBlock(start=4, length=2, segment_value=2.0),
        SourcePhaseBlock(start=6, length=2, segment_value=3.0),
    ]
    new_cos, new_sin, _ = apply_source_phase((cos, sin, "split"), blocks)

    d = mx.arange(3).astype(mx.float32)
    rate = mx.exp((-d / 3.0) * math.log(10000.0))
    mx.eval(new_cos, new_sin, rate)

    assert mx.allclose(new_cos[0, 0, 4, :], mx.cos(2.0 * rate), atol=1e-6).item()
    assert mx.allclose(new_sin[0, 0, 4, :], mx.sin(2.0 * rate), atol=1e-6).item()
    assert mx.allclose(new_cos[0, 0, 6, :], mx.cos(3.0 * rate), atol=1e-6).item()
    assert mx.allclose(new_sin[0, 0, 6, :], mx.sin(3.0 * rate), atol=1e-6).item()


def test_zero_segment_is_noop():
    cos = mx.random.normal((1, 1, 5, 4))
    sin = mx.random.normal((1, 1, 5, 4))
    out = rotate_rope_block(
        (cos, sin, "split"),
        SourcePhaseBlock(start=2, length=2, segment_value=0.0),
    )
    assert out[0] is cos
    assert out[1] is sin
