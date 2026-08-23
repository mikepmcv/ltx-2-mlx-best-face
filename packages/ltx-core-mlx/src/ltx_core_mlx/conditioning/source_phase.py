"""Source-phase RoPE support for identity/reference conditioning.

This module implements the inference-time positional tag used by reference
identity adapters such as LTX-Best-Face-ID. Reference tokens are appended to
the video token stream and share the target's spatial/temporal coordinate
grid. A source-specific phase rotation disambiguates those tokens from the
tokens that should actually be generated.

The implementation is intentionally generic: callers provide token ranges and
segment values, so the same machinery can support one or many references.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import mlx.core as mx


@dataclass(frozen=True)
class SourcePhaseBlock:
    """A contiguous reference-token block that receives one source-phase tag."""

    start: int
    length: int
    segment_value: float

    @property
    def end(self) -> int:
        return self.start + self.length


def rotate_rope_block(
    rope_freqs,
    block: SourcePhaseBlock,
    *,
    theta: float = 10000.0,
):
    """Rotate one token range of an LTX split-RoPE ``(cos, sin, type)`` tuple.

    The extra phase is ``segment_value * theta ** (-d / L)`` where ``L`` is
    the per-head frequency count. This matches the source-phase convention
    used by the Best Face ID overlap reference recipe.
    """
    if block.length <= 0 or block.segment_value == 0.0:
        return rope_freqs

    if not isinstance(rope_freqs, (tuple, list)) or len(rope_freqs) < 2:
        raise TypeError("Expected LTX RoPE tuple/list containing cos and sin tensors")

    cos = rope_freqs[0]
    sin = rope_freqs[1]
    rest = tuple(rope_freqs[2:])

    if cos.ndim != 4 or sin.ndim != 4:
        raise ValueError(
            f"Expected per-head LTX RoPE tensors shaped (B,H,N,L); got {cos.shape} and {sin.shape}"
        )
    if cos.shape != sin.shape:
        raise ValueError(f"RoPE cos/sin shape mismatch: {cos.shape} vs {sin.shape}")
    if block.start < 0 or block.end > cos.shape[2]:
        raise ValueError(
            f"Source-phase block [{block.start}:{block.end}] exceeds token count {cos.shape[2]}"
        )

    freq_count = cos.shape[-1]
    d = mx.arange(freq_count).astype(mx.float32)
    phase = float(block.segment_value) * mx.exp((-d / float(freq_count)) * math.log(float(theta)))
    phase_cos = mx.cos(phase).astype(cos.dtype)[None, None, None, :]
    phase_sin = mx.sin(phase).astype(sin.dtype)[None, None, None, :]

    start, end = block.start, block.end
    cos_ref = cos[:, :, start:end, :]
    sin_ref = sin[:, :, start:end, :]
    rotated_cos = cos_ref * phase_cos - sin_ref * phase_sin
    rotated_sin = sin_ref * phase_cos + cos_ref * phase_sin

    new_cos = mx.concatenate(
        [cos[:, :, :start, :], rotated_cos, cos[:, :, end:, :]],
        axis=2,
    )
    new_sin = mx.concatenate(
        [sin[:, :, :start, :], rotated_sin, sin[:, :, end:, :]],
        axis=2,
    )

    return (new_cos, new_sin, *rest)


def apply_source_phase(
    rope_freqs,
    blocks: Iterable[SourcePhaseBlock],
    *,
    theta: float = 10000.0,
):
    """Apply source-phase rotations for one or more reference-token blocks."""
    out = rope_freqs
    for block in blocks:
        out = rotate_rope_block(out, block, theta=theta)
    return out


def _unwrap_ltx_model(model):
    """Return the underlying LTXModel for normal or streaming wrappers."""
    if hasattr(model, "_compute_rope_freqs"):
        return model

    for attr in ("model", "_model", "base_model", "_base_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "_compute_rope_freqs"):
            return candidate

    raise TypeError(
        f"Could not locate an LTX model with _compute_rope_freqs inside {type(model).__name__}"
    )


def install_source_phase(
    model,
    blocks: Iterable[SourcePhaseBlock],
    *,
    theta: float = 10000.0,
):
    """Install/update source-phase RoPE on an LTX model instance.

    Only the 3-axis video self-attention RoPE is modified. Audio RoPE and the
    1-D audio/video cross-modal temporal RoPE remain untouched, matching the
    identity-overlap inference recipe.

    The patch is idempotent. Repeated calls simply replace the active block
    metadata, which is useful when stage 2 changes the video token count.
    """
    target = _unwrap_ltx_model(model)
    block_tuple = tuple(blocks)

    if not hasattr(target, "_source_phase_original_compute_rope_freqs"):
        target._source_phase_original_compute_rope_freqs = target._compute_rope_freqs

        def patched_compute_rope_freqs(
            positions,
            num_heads: int,
            head_dim: int,
            max_pos_override=None,
        ):
            original = target._source_phase_original_compute_rope_freqs
            rope = original(
                positions,
                num_heads,
                head_dim,
                max_pos_override=max_pos_override,
            )

            active = getattr(target, "_source_phase_blocks", ())
            if not active:
                return rope

            # Video self-attention uses 3 axes (T/H/W). Cross-modal RoPE is
            # recomputed from positions[..., 0:1] and must not receive the tag.
            if positions.shape[-1] != 3:
                return rope

            token_count = positions.shape[1]
            if any(block.end > token_count for block in active):
                raise ValueError(
                    f"Source-phase block exceeds video token count {token_count}: {active}"
                )

            return apply_source_phase(
                rope,
                active,
                theta=getattr(target, "_source_phase_theta", 10000.0),
            )

        # Instance attributes are not descriptors, so this closure intentionally
        # omits ``self`` and behaves like the bound method it replaces.
        target._compute_rope_freqs = patched_compute_rope_freqs

    target._source_phase_blocks = block_tuple
    target._source_phase_theta = float(theta)
    return target


def clear_source_phase(model) -> None:
    """Disable source-phase tagging without destroying the installed patch."""
    target = _unwrap_ltx_model(model)
    target._source_phase_blocks = ()


__all__ = [
    "SourcePhaseBlock",
    "apply_source_phase",
    "clear_source_phase",
    "install_source_phase",
    "rotate_rope_block",
]
