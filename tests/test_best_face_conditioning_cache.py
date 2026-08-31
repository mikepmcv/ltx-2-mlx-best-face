from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from PIL import Image

import ltx_pipelines_mlx.best_face as best_face
from ltx_pipelines_mlx.best_face import BestFacePipeline


class _FakeVAEEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, pixels):
        self.calls += 1
        return mx.zeros((1, 128, 1, 2, 2), dtype=mx.bfloat16)


def test_identity_conditioning_reuses_materialized_reference(monkeypatch, tmp_path: Path):
    reference = tmp_path / "character.png"
    Image.new("RGB", (64, 64)).save(reference)
    monkeypatch.setattr(
        best_face,
        "load_image_and_preprocess",
        lambda *args, **kwargs: mx.zeros((1, 3, 64, 64), dtype=mx.float32),
    )

    pipe = BestFacePipeline.__new__(BestFacePipeline)
    pipe.low_memory = False
    pipe.image_conditioner = SimpleNamespace(_encoder=None)
    pipe.vae_encoder = _FakeVAEEncoder()
    pipe._identity_conditioning_cache = {}

    first_condition, first_phase = pipe._build_identity_conditioning(
        reference=str(reference),
        resize_mode="native_resolution",
        target_h=64,
        target_w=64,
        reference_scale=1.0,
        frame_rate=24.0,
        num_generation_tokens=100,
        source_id=2.0,
        phase_scale=1.0,
        crf=0,
    )
    second_condition, second_phase = pipe._build_identity_conditioning(
        reference=str(reference),
        resize_mode="native_resolution",
        target_h=64,
        target_w=64,
        reference_scale=1.0,
        frame_rate=24.0,
        num_generation_tokens=200,
        source_id=2.0,
        phase_scale=1.0,
        crf=0,
    )

    assert pipe.vae_encoder.calls == 1
    assert first_condition.reference_latent is second_condition.reference_latent
    assert first_phase.start == 100
    assert second_phase.start == 200


def test_low_memory_does_not_retain_identity_cache(monkeypatch, tmp_path: Path):
    reference = tmp_path / "character.png"
    Image.new("RGB", (64, 64)).save(reference)
    monkeypatch.setattr(
        best_face,
        "load_image_and_preprocess",
        lambda *args, **kwargs: mx.zeros((1, 3, 64, 64), dtype=mx.float32),
    )

    pipe = BestFacePipeline.__new__(BestFacePipeline)
    pipe.low_memory = True
    pipe.image_conditioner = SimpleNamespace(_encoder=None)
    pipe.vae_encoder = _FakeVAEEncoder()
    pipe._identity_conditioning_cache = {}

    for _ in range(2):
        pipe._build_identity_conditioning(
            reference=str(reference),
            resize_mode="native_resolution",
            target_h=64,
            target_w=64,
            reference_scale=1.0,
            frame_rate=24.0,
            num_generation_tokens=100,
            source_id=2.0,
            phase_scale=1.0,
            crf=0,
        )

    assert pipe.vae_encoder.calls == 2
    assert pipe._identity_conditioning_cache == {}
