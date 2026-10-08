"""Smoke tests for real/common.py: padding, area resize, threshold, mask status, nominal cameras."""

import numpy as np
import pytest

from strike_a_pose.config import load_config
from strike_a_pose.real.common import (
    MASK_STATUSES,
    mask_status,
    nominal_camera,
    pad_to_square,
    prepare_mask,
)


@pytest.fixture
def real_config(tiny_config_path):
    """The resolved real and camera sections of configs/tiny.yaml."""
    return load_config(tiny_config_path)


def _blob(size, top, left, height, width):
    """A filled rectangle of ones inside a size by size mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top : top + height, left : left + width] = 1
    return mask


def test_pad_to_square_centers_the_shorter_side():
    image = np.ones((2, 4), dtype=np.uint8)
    square = pad_to_square(image)
    assert square.shape == (4, 4)
    assert square[1:3].sum() == 8
    assert square[0].sum() == 0 and square[3].sum() == 0
    assert square.dtype == np.uint8


def test_pad_to_square_rejects_a_non_image():
    with pytest.raises(ValueError):
        pad_to_square(np.zeros((2, 2, 2)))


def test_prepare_mask_resizes_pads_and_thresholds(real_config):
    image_size = real_config["camera"]["image_size"]
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[:, :40] = 255  # the left half is foreground; the padded square keeps it on the left
    prepared = prepare_mask(mask, image_size)
    assert prepared.shape == (image_size, image_size)
    assert prepared.dtype == np.uint8
    assert set(np.unique(prepared)) <= {0, 1}
    # The 40 by 80 foreground sits in rows 20 to 59 of the 80 by 80 square and in columns 0 to 39.
    # Resized to image_size pixels, it covers rows 16 to 47 and columns 0 to 31 of the output.
    assert prepared[18:45, :28].all()
    assert not prepared[:14].any()
    assert not prepared[:, 36:].any()


def test_prepare_mask_thresholds_at_one_half():
    # Each 2 by 2 block of the 4 by 4 mask becomes one output pixel with the value of its share.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1  # share 1.0: kept
    mask[0, 2] = 1
    mask[0, 3] = 1  # share 0.5: kept, because the threshold is inclusive
    mask[2, 0] = 1  # share 0.25: dropped
    prepared = prepare_mask(mask, 2)
    assert prepared.tolist() == [[1, 1], [0, 0]]


def test_prepare_mask_accepts_boolean_masks():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    prepared = prepare_mask(mask, 8)
    assert prepared.sum() == 16


def test_prepare_mask_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        prepare_mask(np.full((4, 4), 300, dtype=np.int32), 4)


def test_no_mask_is_skipped_no_mask(real_config):
    assert mask_status(None, real_config["real"]) == "skipped_no_mask"
    empty = np.zeros((64, 64), dtype=np.uint8)
    assert mask_status(empty, real_config["real"]) == "skipped_no_mask"


def test_a_large_single_person_mask_is_ok(real_config):
    mask = _blob(64, 8, 20, 48, 24)
    assert mask_status(mask, real_config["real"]) == "ok"


def test_a_tiny_blob_is_unusable(real_config):
    mask = _blob(64, 10, 10, 2, 2)  # 4 pixels, far below 2 percent of 4096
    assert mask_status(mask, real_config["real"]) == "skipped_unusable"


def test_a_two_person_mask_is_skipped_multi_person_not_unusable(real_config):
    # Two equal blobs: each holds 50 percent of the mask area, so the largest component is 50
    # percent of the mask, which fails the 90 percent usability rule. The multi-person rule must
    # decide first, so the status is skipped_multi_person.
    mask = _blob(64, 4, 4, 56, 20) + _blob(64, 4, 40, 56, 20)
    assert mask_status(mask, real_config["real"]) == "skipped_multi_person"


def test_a_small_second_component_below_ten_percent_is_not_multi_person(real_config):
    main = _blob(64, 4, 20, 56, 24)  # 1344 pixels
    speck = _blob(64, 0, 0, 2, 2)  # 4 pixels, far below 10 percent of the mask area
    assert mask_status(main + speck, real_config["real"]) == "ok"


def test_status_values_are_the_documented_set(real_config):
    statuses = {
        mask_status(None, real_config["real"]),
        mask_status(_blob(64, 8, 20, 48, 24), real_config["real"]),
        mask_status(_blob(64, 10, 10, 2, 2), real_config["real"]),
        mask_status(_blob(64, 4, 4, 56, 20) + _blob(64, 4, 40, 56, 20), real_config["real"]),
    }
    assert statuses == set(MASK_STATUSES)


def test_mask_status_rejects_non_binary_masks(real_config):
    with pytest.raises(ValueError):
        mask_status(np.full((4, 4), 2, dtype=np.uint8), real_config["real"])


def test_nominal_camera_sits_at_the_configured_distance_and_height(real_config):
    real = real_config["real"]
    camera = real_config["camera"]
    K, rotation, translation = nominal_camera(real, camera, 0.0)
    assert K.shape == (3, 3)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    position = -rotation.T @ translation
    nominal = real["nominal_camera"]
    assert position[0] == pytest.approx(0.0, abs=1e-9)
    assert position[2] == pytest.approx(nominal["distance_m"], abs=1e-9)
    assert position[1] == pytest.approx(nominal["height_m"], abs=1e-9)


def test_nominal_camera_looks_at_the_lookat_height(real_config):
    real = real_config["real"]
    camera = real_config["camera"]
    K, rotation, translation = nominal_camera(real, camera, 90.0)
    position = -rotation.T @ translation
    target = np.array([0.0, real["nominal_camera"]["lookat_height_m"], 0.0])
    forward = rotation[2]  # the optical axis is the third camera axis
    expected = (target - position) / np.linalg.norm(target - position)
    assert np.allclose(forward, expected, atol=1e-9)
    # Azimuth 90 degrees puts the camera on the +x side.
    assert position[0] == pytest.approx(real["nominal_camera"]["distance_m"], abs=1e-9)
    assert position[2] == pytest.approx(0.0, abs=1e-9)


def test_nominal_camera_intrinsics_follow_the_camera_section(real_config):
    real = real_config["real"]
    camera = real_config["camera"]
    K, _rotation, _translation = nominal_camera(real, camera, 0.0)
    assert K[0, 0] == pytest.approx(camera["focal_px"])
    assert K[0, 2] == pytest.approx((camera["image_size"] - 1) / 2.0)


def test_nominal_camera_rejects_a_zero_distance(real_config):
    zero_distance = {"distance_m": 0.0, "height_m": 1.2, "lookat_height_m": 0.9}
    real = {**real_config["real"], "nominal_camera": zero_distance}
    with pytest.raises(ValueError):
        nominal_camera(real, real_config["camera"], 0.0)
