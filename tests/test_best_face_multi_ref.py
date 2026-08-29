from __future__ import annotations

import pytest

from ltx_pipelines_mlx.best_face_multi_ref import ReferenceSet, _reference_paths


def test_reference_set_is_string_compatible() -> None:
    refs = ReferenceSet(["face.png", "sheet.png", "teeth.png"])

    assert isinstance(refs, str)
    assert str(refs) == "face.png"
    assert refs.references == ("face.png", "sheet.png", "teeth.png")


def test_reference_set_requires_at_least_one_image() -> None:
    with pytest.raises(ValueError, match="At least one identity reference"):
        ReferenceSet([])


def test_reference_paths_preserves_single_reference_compatibility() -> None:
    assert _reference_paths("face.png") == ("face.png",)


def test_reference_paths_returns_all_multi_refs_in_order() -> None:
    refs = ReferenceSet(["face.png", "sheet.png", "teeth.png"])
    assert _reference_paths(refs) == ("face.png", "sheet.png", "teeth.png")
