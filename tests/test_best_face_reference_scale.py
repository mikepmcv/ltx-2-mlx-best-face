import inspect
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_pipelines_mlx.best_face import (
    OFFICIAL_BASE_FACE_STRENGTH,
    OFFICIAL_SPATIAL_UPSCALER_FILE,
    BestFacePipeline,
    _metadata_path,
    _prepare_keyframe_image,
    _resolve_generation_settings,
    _sigma_schedule_for_steps,
    _write_generation_metadata,
)
from ltx_pipelines_mlx.best_face_exact import (
    OFFICIAL_DISTILLED_LORA_STRENGTH,
    OFFICIAL_NEGATIVE_PROMPT,
    OFFICIAL_STAGE2_SIGMAS,
    BestFaceExactPipeline,
)
from ltx_pipelines_mlx.utils.helpers import create_noised_state


def test_reference_scale_halves_vae_size_and_preserves_position_span(tmp_path: Path):
    reference = tmp_path / "sheet.png"
    Image.new("RGB", (1536, 1024)).save(reference)

    geometry = BestFacePipeline._scaled_reference_geometry(
        str(reference),
        resize_mode="native_resolution",
        target_h=576,
        target_w=768,
        reference_scale=0.5,
    )

    assert geometry == (512, 768, 2.0, 2.0)


def test_reference_scale_accounts_for_dimension_rounding(tmp_path: Path):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (1000, 700)).save(reference)

    encoded_h, encoded_w, scale_h, scale_w = BestFacePipeline._scaled_reference_geometry(
        str(reference),
        resize_mode="native_resolution",
        target_h=576,
        target_w=768,
        reference_scale=0.5,
    )

    assert (encoded_h, encoded_w) == (352, 512)
    assert encoded_h * scale_h == 704
    assert encoded_w * scale_w == 992


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.01])
def test_reference_scale_rejects_invalid_values(tmp_path: Path, scale: float):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (64, 64)).save(reference)

    with pytest.raises(ValueError, match="reference_scale"):
        BestFacePipeline._scaled_reference_geometry(
            str(reference),
            resize_mode="native_resolution",
            target_h=576,
            target_w=768,
            reference_scale=scale,
        )


def test_generation_settings_preserve_existing_defaults():
    assert _resolve_generation_settings(
        stage1_steps=None,
        stage2_steps=None,
        reference_scale=1.0,
        stage1_reference_scale=None,
        stage2_reference_scale=None,
        fast_refine=False,
        ugc_fast=False,
    ) == (None, None, 1.0, 1.0, False)


def test_ugc_fast_resolves_speed_preset():
    assert _resolve_generation_settings(
        stage1_steps=None,
        stage2_steps=None,
        reference_scale=1.0,
        stage1_reference_scale=None,
        stage2_reference_scale=None,
        fast_refine=False,
        ugc_fast=True,
    ) == (6, 2, 0.5, 1.0, True)


def test_ugc_fast_allows_explicit_stage_and_scale_overrides():
    assert _resolve_generation_settings(
        stage1_steps=7,
        stage2_steps=3,
        reference_scale=0.75,
        stage1_reference_scale=0.625,
        stage2_reference_scale=0.875,
        fast_refine=False,
        ugc_fast=True,
    ) == (7, 3, 0.625, 0.875, True)


def test_reduced_sigma_schedule_keeps_terminal_zero_and_detail_steps():
    schedule = [1.0, 0.99, 0.98, 0.9, 0.7, 0.4, 0.0]

    assert _sigma_schedule_for_steps(schedule, 4) == [1.0, 0.9, 0.7, 0.4, 0.0]


def test_full_sigma_schedule_is_unchanged():
    schedule = [1.0, 0.7, 0.0]

    assert _sigma_schedule_for_steps(schedule, None) is schedule
    assert _sigma_schedule_for_steps(schedule, 2) is schedule


@pytest.mark.parametrize("steps", [0, 9])
def test_sigma_schedule_rejects_invalid_step_count(steps: int):
    with pytest.raises(ValueError, match="steps must be between"):
        _sigma_schedule_for_steps([1.0, 0.7, 0.0], steps)


def test_keyframe_specs_use_first_and_last_pixel_frames():
    specs = BestFacePipeline._keyframe_specs(
        first_frame="opening.png",
        last_frame="ending.png",
        first_frame_strength=1.0,
        last_frame_strength=0.75,
        first_frame_mode="layout",
        last_frame_mode="appearance",
        num_frames=145,
    )

    assert specs == [
        ("opening.png", 0, 1.0, "layout"),
        ("ending.png", 144, 0.75, "appearance"),
    ]


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_keyframe_specs_reject_invalid_strength(strength: float):
    with pytest.raises(ValueError, match="strength"):
        BestFacePipeline._keyframe_specs(
            first_frame="opening.png",
            last_frame=None,
            first_frame_strength=strength,
            last_frame_strength=1.0,
            first_frame_mode="appearance",
            last_frame_mode="appearance",
            num_frames=49,
        )


def test_layout_keyframe_suppresses_texture_and_color(tmp_path: Path):
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[:, :32, 0] = 255
    source[:, 32:, 2] = 255
    path = tmp_path / "layout.png"
    Image.fromarray(source).save(path)

    appearance = np.asarray(
        _prepare_keyframe_image(str(path), 64, 64, mode="appearance", layout_blur=16)
    ).astype(np.float32)
    layout = np.asarray(
        _prepare_keyframe_image(str(path), 64, 64, mode="layout", layout_blur=16)
    ).astype(np.float32)

    assert layout.std() < appearance.std()
    assert np.abs(layout[..., 0] - layout[..., 2]).mean() < np.abs(
        appearance[..., 0] - appearance[..., 2]
    ).mean()


def test_generation_metadata_is_written_beside_output(tmp_path: Path):
    output = tmp_path / "clip.mp4"
    metadata_path = _write_generation_metadata(
        str(output),
        {"seed": 123, "prompt": "test"},
    )

    assert metadata_path == _metadata_path(str(output))
    assert metadata_path.name == "clip.mp4.json"
    assert metadata_path.read_text(encoding="utf-8") == '{\n  "prompt": "test",\n  "seed": 123\n}\n'


def test_legacy_noise_respects_partial_strength_on_appended_keyframe():
    keyframe = VideoConditionByKeyframeIndex(
        frame_idx=0,
        keyframe_latent=mx.ones((1, 1, 2)),
        spatial_dims=(1, 1, 1),
        frame_rate=24.0,
        strength=0.0,
    )

    state = create_noised_state(
        base_shape=(1, 1, 2),
        conditionings=[keyframe],
        spatial_dims=(1, 1, 1),
        positions=mx.zeros((1, 1, 3)),
        seed=7,
        sigma=1.0,
        legacy_scalar_blend=True,
    )

    assert mx.array_equal(state.clean_latent[:, 1:, :], mx.ones((1, 1, 2))).item()
    assert not mx.array_equal(state.latent[:, 1:, :], state.clean_latent[:, 1:, :]).item()


def test_legacy_noise_preserves_full_strength_appended_reference():
    keyframe = VideoConditionByKeyframeIndex(
        frame_idx=0,
        keyframe_latent=mx.ones((1, 1, 2)),
        spatial_dims=(1, 1, 1),
        frame_rate=24.0,
        strength=1.0,
    )

    state = create_noised_state(
        base_shape=(1, 1, 2),
        conditionings=[keyframe],
        spatial_dims=(1, 1, 1),
        positions=mx.zeros((1, 1, 3)),
        seed=7,
        sigma=1.0,
        legacy_scalar_blend=True,
    )

    assert mx.array_equal(state.latent[:, 1:, :], state.clean_latent[:, 1:, :]).item()


def test_best_face_exact_uses_published_character_sheet_defaults():
    signature = inspect.signature(BestFaceExactPipeline.__init__)

    assert signature.parameters["distilled_lora_strength"].default == 0.6
    assert signature.parameters["base_face_strength"].default == 0.2
    assert OFFICIAL_BASE_FACE_STRENGTH == 0.2
    assert OFFICIAL_SPATIAL_UPSCALER_FILE == "spatial_upscaler_x2_v1_1.safetensors"
    assert OFFICIAL_DISTILLED_LORA_STRENGTH == 0.6
    assert OFFICIAL_STAGE2_SIGMAS == [0.85, 0.725, 0.421875, 0.0]
    assert OFFICIAL_NEGATIVE_PROMPT == (
        "pc game, console game, video game, cartoon, childish, ugly, artifacts, "
        "low resolution, blurry, jagged edges"
    )


def test_best_face_exact_orders_official_character_sheet_loras(tmp_path: Path):
    pipe = BestFaceExactPipeline(
        model_dir=str(tmp_path),
        best_face_lora="character.safetensors",
        best_face_strength=1.0,
        base_face_lora="base.safetensors",
        base_face_strength=0.2,
        distilled_lora="distilled.safetensors",
        distilled_lora_strength=0.6,
        spatial_upscaler="spatial_upscaler_x2_v1_1.safetensors",
    )

    assert pipe._pending_loras == [
        ("distilled.safetensors", 0.6),
        ("base.safetensors", 0.2),
        ("character.safetensors", 1.0),
    ]
    assert pipe._spatial_upscaler_path == "spatial_upscaler_x2_v1_1.safetensors"
