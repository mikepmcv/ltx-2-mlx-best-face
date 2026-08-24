from pathlib import Path

import pytest
from PIL import Image

from ltx_pipelines_mlx.best_face import BestFacePipeline


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
