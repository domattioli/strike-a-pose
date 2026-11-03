"""Normalization helpers (spec 002 FR-004, FR-005; contract: sample-contract.md)."""

import numpy as np
import pytest
import torch

from strike_a_pose.data.transforms import (
    add_root,
    denormalize_keypoints_2d,
    image_to_tensor,
    mask_to_tensor,
    normalize_keypoints_2d,
    root_relative,
    tensor_to_image,
)


def test_image_round_trip_exact_for_all_uint8_values():
    img = np.arange(256, dtype=np.uint8).repeat(3).reshape(16, 48)
    t = image_to_tensor(img)
    assert t.dtype == torch.float32 and t.shape == (1, 16, 48)
    assert float(t.min()) == 0.0 and float(t.max()) == 1.0
    assert np.array_equal(tensor_to_image(t), img)
    assert np.array_equal(tensor_to_image(t[0]), img)


def test_tensor_to_image_clamps():
    t = torch.tensor([[[-0.5, 0.5, 1.5]]])
    assert tensor_to_image(t).tolist() == [[0, 128, 255]]


@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.float32), np.zeros((4, 4, 3), np.uint8)])
def test_image_to_tensor_rejects_non_grey_uint8(bad):
    with pytest.raises(ValueError):
        image_to_tensor(bad)


def test_mask_to_tensor():
    m = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    t = mask_to_tensor(m)
    assert t.dtype == torch.bool and t.shape == (1, 2, 2)
    assert t[0].tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_root_relative_round_trip_batched(lib):
    rng = np.random.default_rng(0)
    j = rng.normal(size=(5, 24, 3)).astype(np.float32)
    if lib == "torch":
        j = torch.from_numpy(j)
    rel = root_relative(j)
    assert np.abs(np.asarray(rel[:, 0])).max() == 0.0
    back = add_root(rel, j[:, 0])
    assert np.allclose(np.asarray(back), np.asarray(j), atol=1e-6)
    assert type(rel) is type(j)


def test_root_relative_other_root():
    j = np.arange(24 * 3, dtype=np.float64).reshape(24, 3)
    rel = root_relative(j, root=12)
    assert np.all(rel[12] == 0)
    assert np.allclose(add_root(rel, j[12]), j)


def test_normalize_keypoints_2d_reference_values():
    w, h = 192, 256
    kp = np.array([[0.0, 0.0], [w, h], [w / 2, h / 2]])
    n = normalize_keypoints_2d(kp, w, h)
    assert np.allclose(n, [[-1, -1], [1, 1], [0, 0]])


@pytest.mark.parametrize("lib", ["numpy", "torch"])
def test_normalize_keypoints_2d_round_trip(lib):
    kp = np.random.default_rng(1).uniform(0, 256, size=(3, 4, 24, 2))
    if lib == "torch":
        kp = torch.from_numpy(kp)
    back = denormalize_keypoints_2d(normalize_keypoints_2d(kp, 192, 256), 192, 256)
    assert np.allclose(np.asarray(back), np.asarray(kp), atol=1e-9)
    assert type(back) is type(kp)
