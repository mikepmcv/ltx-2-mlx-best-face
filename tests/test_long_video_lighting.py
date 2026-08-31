from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ltx_pipelines_mlx.long_video import (
    _measure_brightness_correction,
    _stable_luma,
)


def _solid(path: Path, value: int) -> Path:
    Image.new("RGB", (64, 64), (value, value, value)).save(path)
    return path


def test_brightness_correction_matches_master_and_clamps(tmp_path: Path):
    master = _solid(tmp_path / "master.png", 130)
    generated = _solid(tmp_path / "generated.png", 100)

    correction = _measure_brightness_correction(
        master_frame=master,
        generated_frame=generated,
        foreground_mask=None,
        maximum=0.08,
    )

    assert correction == pytest.approx(0.08)


def test_border_measurement_ignores_central_presenter(tmp_path: Path):
    first = Image.new("RGB", (64, 64), (100, 100, 100))
    ImageDraw.Draw(first).rectangle((16, 8, 48, 63), fill=(240, 240, 240))
    second = Image.new("RGB", (64, 64), (100, 100, 100))
    ImageDraw.Draw(second).rectangle((16, 8, 48, 63), fill=(20, 20, 20))
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first.save(first_path)
    second.save(second_path)

    assert _stable_luma(first_path, foreground_mask=None) == pytest.approx(
        _stable_luma(second_path, foreground_mask=None)
    )


def test_mask_measurement_uses_only_black_background(tmp_path: Path):
    image = Image.new("RGB", (64, 64), (80, 80, 80))
    ImageDraw.Draw(image).rectangle((16, 8, 48, 63), fill=(230, 230, 230))
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((16, 8, 48, 63), fill=255)
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    image.save(image_path)
    mask.save(mask_path)

    assert _stable_luma(image_path, foreground_mask=mask_path) == pytest.approx(
        80.0, abs=1.0
    )
