from pathlib import Path

import pytest
from PIL import Image

from ltx_pipelines_mlx.best_face import (
    BestFacePipeline,
    _metadata_path,
    _write_generation_metadata,
)


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


def test_keyframe_specs_use_first_and_last_pixel_frames():
    specs = BestFacePipeline._keyframe_specs(
        first_frame="opening.png",
        last_frame="ending.png",
        first_frame_strength=1.0,
        last_frame_strength=0.75,
        num_frames=145,
    )

    assert specs == [("opening.png", 0, 1.0), ("ending.png", 144, 0.75)]


@pytest.mark.parametrize("strength", [-0.01, 1.01])
def test_keyframe_specs_reject_invalid_strength(strength: float):
    with pytest.raises(ValueError, match="strength"):
        BestFacePipeline._keyframe_specs(
            first_frame="opening.png",
            last_frame=None,
            first_frame_strength=strength,
            last_frame_strength=1.0,
            num_frames=49,
        )


def test_generation_metadata_is_written_beside_output(tmp_path: Path):
    output = tmp_path / "clip.mp4"
    metadata_path = _write_generation_metadata(
        str(output),
        {"seed": 123, "prompt": "test"},
    )

    assert metadata_path == _metadata_path(str(output))
    assert metadata_path.name == "clip.mp4.json"
    assert metadata_path.read_text(encoding="utf-8") == '{\n  "prompt": "test",\n  "seed": 123\n}\n'
