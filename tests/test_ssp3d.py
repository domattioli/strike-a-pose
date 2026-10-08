"""Tests for real/ssp3d.py: labels, key logging, silhouettes, and stand-in ground truth."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.body.base import canonical_mesh
from strike_a_pose.body.standin import NUM_BETAS, StandInBody
from strike_a_pose.measure import measure_mesh
from strike_a_pose.real import ssp3d
from strike_a_pose.real.ssp3d import (
    Ssp3dError,
    ground_truth_measurements,
    load_labels,
    load_silhouette,
    normalise_gender,
)

STEP_CM = 1.0
COUNT = 4


@pytest.fixture(scope="module")
def stand_in() -> StandInBody:
    return StandInBody(radial_segments=8, cap_rings=2)


def _write_dataset(root: Path, **overrides: np.ndarray) -> Path:
    rng = np.random.default_rng(0)
    fields = {
        "fnames": np.array([f"img_{i:03d}.png" for i in range(COUNT)]),
        "shapes": rng.normal(0.0, 0.5, size=(COUNT, NUM_BETAS)),
        "poses": rng.normal(0.0, 0.1, size=(COUNT, 72)),
        "genders": np.array(["m", "f", "m", "f"]),
        "bbox_whs": rng.uniform(10, 50, size=(COUNT, 4)),
    }
    fields.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "labels.npz", **fields)
    return root


def test_load_labels_reads_fields_and_logs_keys(tmp_path, caplog):
    root = _write_dataset(tmp_path / "ssp3d")
    with caplog.at_level(logging.INFO, logger=ssp3d.__name__):
        labels = load_labels(root)
    assert len(labels) == COUNT
    assert labels.filenames[2] == "img_002.png"
    assert labels.genders == ("male", "female", "male", "female")
    assert labels.betas.shape == (COUNT, NUM_BETAS)
    assert labels.boxes is not None and labels.boxes.shape == (COUNT, 4)
    logged = " ".join(record.getMessage() for record in caplog.records)
    for key in ("fnames", "shapes", "poses", "genders", "bbox_whs"):
        assert key in logged


def test_load_labels_accepts_alternative_key_names_and_no_boxes(tmp_path):
    root = tmp_path / "ssp3d"
    root.mkdir()
    np.savez(
        root / "labels.npz",
        filenames=np.array(["a.png", "b.png"]),
        betas=np.zeros((2, NUM_BETAS)),
        gender=np.array([b"f", b"m"]),
    )
    labels = load_labels(root)
    assert labels.genders == ("female", "male")
    assert labels.boxes is None


def test_missing_labels_names_the_configuration_key(tmp_path):
    with pytest.raises(MissingAssetError, match="real.ssp3d.path"):
        load_labels(tmp_path / "absent")


def test_missing_field_lists_the_keys_present(tmp_path):
    root = tmp_path / "ssp3d"
    root.mkdir()
    np.savez(root / "labels.npz", fnames=np.array(["a.png"]), unrelated=np.zeros(3))
    with pytest.raises(Ssp3dError, match="unrelated"):
        load_labels(root)


def test_length_mismatch_is_an_error(tmp_path):
    root = _write_dataset(tmp_path / "ssp3d", genders=np.array(["m", "f"]))
    with pytest.raises(Ssp3dError, match="disagree"):
        load_labels(root)


def test_normalise_gender():
    assert normalise_gender("M") == "male"
    assert normalise_gender(b"f") == "female"
    with pytest.raises(Ssp3dError):
        normalise_gender("x")


def test_load_silhouette_returns_binary_mask_or_none(tmp_path):
    root = _write_dataset(tmp_path / "ssp3d")
    (root / "silhouettes").mkdir()
    image = np.zeros((20, 10), dtype=np.uint8)
    image[5:15, 3:7] = 255
    cv2.imwrite(str(root / "silhouettes" / "img_001.png"), image)
    mask = load_silhouette(root, "img_001.png")
    assert mask is not None
    assert mask.shape == (20, 10)
    assert set(np.unique(mask)) == {0, 1}
    assert int(mask.sum()) == 40
    assert load_silhouette(root, "img_000.png") is None


def test_ground_truth_matches_direct_measurement_per_gender(tmp_path, stand_in):
    labels = load_labels(_write_dataset(tmp_path / "ssp3d"))
    calls: list[str] = []

    def factory(gender: str) -> StandInBody:
        calls.append(gender)
        return stand_in

    truth = ground_truth_measurements(labels, factory, STEP_CM)
    assert truth.shape == (COUNT, 5)
    assert sorted(calls) == ["female", "male"]
    assert np.isfinite(truth).all()
    for row in range(COUNT):
        vertices, joints = canonical_mesh(stand_in, labels.betas[row])
        expected = measure_mesh(vertices, stand_in.faces, stand_in.part_ids, joints, STEP_CM)
        np.testing.assert_allclose(truth[row], expected.numpy(), rtol=0, atol=1e-9)
    assert 100.0 < truth[:, 0].min() and truth[:, 0].max() < 250.0


def test_ground_truth_indices_select_and_order_rows(tmp_path, stand_in):
    labels = load_labels(_write_dataset(tmp_path / "ssp3d"))
    everyone = ground_truth_measurements(labels, lambda gender: stand_in, STEP_CM)
    chosen = ground_truth_measurements(labels, lambda gender: stand_in, STEP_CM, indices=[3, 0])
    np.testing.assert_allclose(chosen, everyone[[3, 0]], rtol=0, atol=1e-9)
