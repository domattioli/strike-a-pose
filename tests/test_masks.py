"""Tests of person masks: provided silhouettes, the status rules, and the optional SAM 2 backend."""

import sys
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.real.masks import (
    MaskBackend,
    MaskStatus,
    ProvidedMasks,
    Sam2Masks,
    mask_status,
    segment_subject,
)

SIDE = 100  # every test mask is 100 by 100 pixels, so 1 pixel is 0.01 percent of the image


class StubMasks:
    """A backend that returns one fixed mask for every subject, and records the box it was given."""

    name = "stub"

    def __init__(self, mask: np.ndarray | None) -> None:
        self._mask = mask
        self.boxes: list[object] = []

    def segment(self, subject_id, image, box):
        self.boxes.append(box)
        return self._mask


def block(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    """Set a rectangle of the mask to True, in place."""
    mask[top : top + height, left : left + width] = True


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubMasks(None), MaskBackend)
    assert isinstance(ProvidedMasks({}), MaskBackend)


def test_provided_masks_return_the_supplied_silhouette_and_none_for_unknown_subjects() -> None:
    silhouette = np.zeros((SIDE, SIDE), dtype=np.uint8)
    block(silhouette, 10, 10, 60, 40)
    backend = ProvidedMasks({"front_0001": silhouette})
    image = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)

    mask = backend.segment("front_0001", image, None)
    assert mask is not None
    assert mask.dtype == bool
    assert np.array_equal(mask, silhouette.astype(bool))
    assert backend.segment("unknown", image, None) is None
    assert backend.name == "provided"


def test_one_large_blob_is_ok() -> None:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 10, 10, 60, 50)  # 3000 pixels, 30 percent of the image
    assert mask_status(mask) is MaskStatus.OK


def test_no_mask_is_skipped_no_mask() -> None:
    assert mask_status(None) is MaskStatus.SKIPPED_NO_MASK


def test_empty_mask_is_skipped_no_mask() -> None:
    # One rule set, real/common.py: an all-zero mask means no person was found.
    assert mask_status(np.zeros((SIDE, SIDE), dtype=bool)) is MaskStatus.SKIPPED_NO_MASK


def test_mask_below_the_area_fraction_is_skipped_unusable() -> None:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 0, 0, 10, 10)  # 100 pixels, 1 percent of the image, below 2 percent
    assert mask_status(mask) is MaskStatus.SKIPPED_UNUSABLE
    block(mask, 0, 10, 10, 20)  # 300 pixels, 3 percent of the image, now enough
    assert mask_status(mask) is MaskStatus.OK


def test_two_equal_blobs_are_skipped_multi_person() -> None:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 5, 5, 40, 40)
    block(mask, 55, 55, 40, 40)
    assert mask_status(mask) is MaskStatus.SKIPPED_MULTI_PERSON


def test_two_person_mask_is_multi_person_even_when_it_also_fails_usability() -> None:
    # The largest component holds 60 of 100 pixels, below 90 percent, so usability fails too.
    # The status must still be the multi-person one, because that rule runs first.
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 0, 0, 10, 6)  # 60 pixels
    block(mask, 50, 50, 10, 4)  # 40 pixels
    assert mask_status(mask) is MaskStatus.SKIPPED_MULTI_PERSON


def test_a_small_speck_beside_a_person_keeps_the_mask_ok() -> None:
    # A 5-pixel speck is 2.4 percent of the 205-pixel mask: below the 10 percent person share.
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 10, 10, 40, 5)  # 200 pixels, 2 percent of the image
    block(mask, 80, 80, 1, 5)  # 5 pixels
    assert mask_status(mask) is MaskStatus.OK


def test_largest_component_below_ninety_percent_is_skipped_unusable() -> None:
    # The mask holds 200 pixels (2 percent of the image). Two specks of 18 and 12 pixels stay below
    # the 10 percent person share (20 pixels), but the largest component holds only 170 of 200.
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 10, 10, 17, 10)  # 170 pixels
    block(mask, 60, 60, 1, 18)  # 18 pixels
    block(mask, 80, 80, 1, 12)  # 12 pixels
    assert mask_status(mask) is MaskStatus.SKIPPED_UNUSABLE


def test_thresholds_are_taken_from_the_arguments() -> None:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    block(mask, 0, 0, 10, 10)  # 1 percent of the image
    assert mask_status(mask, min_area_fraction=0.01) is MaskStatus.OK


def test_mask_that_is_not_two_dimensional_is_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        mask_status(np.ones((SIDE, SIDE, 3), dtype=bool))


def test_segment_subject_returns_the_mask_and_its_status() -> None:
    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    block(mask, 10, 10, 60, 50)
    image = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    box = (10.0, 10.0, 60.0, 70.0)
    backend = StubMasks(mask)

    result = segment_subject(backend, "photo_7", image, box)
    assert result.status is MaskStatus.OK
    assert result.mask is not None
    assert result.mask.dtype == bool
    assert backend.boxes == [box]


def test_segment_subject_with_no_mask_has_no_mask_and_the_no_mask_status() -> None:
    image = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    result = segment_subject(StubMasks(None), "photo_8", image, None)
    assert result.mask is None
    assert result.status is MaskStatus.SKIPPED_NO_MASK


def test_sam2_is_skipped_when_the_package_is_absent() -> None:
    pytest.importorskip("sam2")
    # The package is present here. A missing checkpoint is then reported before any model loads.
    with pytest.raises(MissingAssetError, match="real.sam2.checkpoint"):
        Sam2Masks("<root>/sam2/sam2.1_hiera_base_plus.pt", None)


def test_missing_sam2_package_raises_an_import_error_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A None entry in sys.modules makes every import of the package fail, present or not.
    monkeypatch.setitem(sys.modules, "sam2", None)
    checkpoint = tmp_path / "sam2.1_hiera_base_plus.pt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ImportError, match=r"\[real\]"):
        Sam2Masks(str(checkpoint), None)
