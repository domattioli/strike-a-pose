"""Tests for predict.py: cell files, exact monotonicity, linearization, floor, and refusals."""

import json
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from strike_a_pose import predict as predict_module
from strike_a_pose.body.standin import StandInBody
from strike_a_pose.checkpoint import DONE_MARKER_NAME, read_done_marker
from strike_a_pose.config import load_config
from strike_a_pose.data.dataset import ShapeDataset
from strike_a_pose.data.generate import data_directory, generate_dataset
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.model.vae import ShapeVAE
from strike_a_pose.predict import (
    PredictionRefusedError,
    linearized_measurements,
    predict_directory,
    run_predict,
    summarize_samples,
)
from strike_a_pose.train import train_model

# One epoch on small_config data. The wider focal length keeps most bodies in frame.
RUN_OVERRIDES: dict[str, object] = {
    "train.batch_size": 8,
    "data.min_unflagged": 0,
    "camera.focal_px": 48,
    "evaluate.views": [1, 2, 4],
    "evaluate.noise_deg": [0, 5],
}
SPLITS = ("cal", "test")
FLOOR_CM = 0.001  # calibrate.spread_floor_cm in configs/tiny.yaml


class Trained(NamedTuple):
    """A generated and trained run: its configuration and its output folder."""

    config: dict[str, Any]
    out: Path


def make_config(overrides: dict[str, object]) -> dict[str, Any]:
    from tests.conftest import SMALL_CONFIG_OVERRIDES, TINY_CONFIG

    merged = {**SMALL_CONFIG_OVERRIDES, **RUN_OVERRIDES, **overrides}
    return load_config(TINY_CONFIG, [f"{key}={json.dumps(value)}" for key, value in merged.items()])


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    """Run on one thread, so these tests stay fast while other jobs share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Trained:
    """Generate small data and train one epoch once; every test predicts from this run."""
    config = make_config({})
    out = tmp_path_factory.mktemp("predicted")
    generate_dataset(config, out)
    train_model(config, out)
    return Trained(config, out)


@pytest.fixture(scope="module")
def predicted(trained) -> Path:
    """Run the exact prediction of the three view counts and two noise levels once."""
    result = run_predict(trained.config, trained.out)
    assert result.completed
    return predict_directory(trained.out)


def read_cell(directory: Path, split: str, views: int, noise: int) -> dict[str, np.ndarray]:
    with np.load(directory / f"{split}_v{views}_n{noise}.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_one_file_per_split_and_cell(trained, predicted) -> None:
    names = sorted(path.name for path in predicted.glob("*.npz"))
    expected = sorted(f"{s}_v{v}_n{n}.npz" for s in SPLITS for v in (1, 2, 4) for n in (0, 5))
    assert names == expected
    marker = read_done_marker(predicted)
    assert marker is not None and marker.stage == "predict"
    assert set(marker.inputs) == {"generate", "train"}
    assert (predicted / DONE_MARKER_NAME).is_file()


def test_cell_file_contents(trained, predicted) -> None:
    config = trained.config
    split = DataSplit(config["data"]["n_train"], config["data"]["n_cal"], config["data"]["n_test"])
    k = config["predict"]["n_samples"]
    for name in SPLITS:
        cell = read_cell(predicted, name, 2, 5)
        bodies = split.range_of(name)
        ids = cell["body_id"]
        assert str(cell["split"]) == name
        assert int(cell["views"]) == 2 and float(cell["noise_deg"]) == 5.0
        assert ids.dtype.kind == "i" and len(set(ids.tolist())) == len(ids) > 0
        assert ids.min() >= bodies.start and ids.max() < bodies.stop
        n = len(ids)
        for key in ("m_true", "m_median", "spread"):
            assert cell[key].shape == (n, 5)
            assert np.isfinite(cell[key]).all()
        assert cell["latent_var_mean"].shape == (n,) and (cell["latent_var_mean"] > 0).all()
        assert cell["samples"].shape == (n, k, 5) and cell["samples"].dtype == np.float32
        assert (cell["spread"] >= FLOOR_CM).all()
        assert np.allclose(
            cell["m_median"], np.median(cell["samples"].astype(float), axis=1), atol=1e-3
        )
        # The ground truth is the measurement stored by the data stage, in cm and sensible.
        assert 100.0 < cell["m_true"][:, 0].min() and cell["m_true"][:, 0].max() < 260.0


def test_bodies_match_across_cells_and_ground_truth_is_cell_independent(predicted) -> None:
    first = read_cell(predicted, "test", 1, 0)
    last = read_cell(predicted, "test", 4, 5)
    assert np.array_equal(first["body_id"], last["body_id"])
    assert np.array_equal(first["m_true"], last["m_true"])


def test_latent_variance_never_increases_with_view_count_exactly(predicted) -> None:
    for name in SPLITS:
        for noise in (0, 5):
            v1, v2, v4 = (
                read_cell(predicted, name, views, noise)["latent_var_mean"] for views in (1, 2, 4)
            )
            assert np.all(v2 <= v1)
            assert np.all(v4 <= v2)
            assert np.all(v4 < v1)  # views do add information


def test_each_body_is_encoded_once_per_noise_level(trained, monkeypatch) -> None:
    encoded = []
    real_encode = ShapeVAE.encode

    def counting_encode(self, views):
        encoded.append(int(views.view_mask.shape[0]) * int(views.view_mask[0].sum()))
        return real_encode(self, views)

    monkeypatch.setattr(ShapeVAE, "encode", counting_encode)
    run_predict(trained.config, trained.out)
    config = trained.config
    data_dir = data_directory(trained.out)
    split = DataSplit(config["data"]["n_train"], config["data"]["n_cal"], config["data"]["n_test"])
    bodies = sum(
        len(ShapeDataset.for_cell(config, data_dir, split.range_of(name), views=4, noise_deg=0.0))
        for name in SPLITS
    )
    # Four camera slots per body and noise level (4 encodings, not 1 + 2 + 4 = 7 per level).
    assert sum(encoded) == 4 * bodies * 2


def test_resume_skips_a_finished_stage_and_reuses_cell_files(trained) -> None:
    out = trained.out
    predict_dir = predict_directory(out)
    run_predict(trained.config, out)
    skipped = run_predict(trained.config, out, resume=True)
    assert skipped.skipped and skipped.completed

    target = predict_dir / "test_v2_n5.npz"
    before = read_cell(predict_dir, "test", 2, 5)
    target.unlink()
    (predict_dir / DONE_MARKER_NAME).unlink()
    result = run_predict(trained.config, out, resume=True)
    assert result.cells_written == ("test_v2_n5.npz",)
    assert len(result.cells_reused) == 11
    after = read_cell(predict_dir, "test", 2, 5)
    for key in ("body_id", "m_median", "spread", "latent_var_mean", "samples"):
        assert np.array_equal(before[key], after[key]), key


def test_spread_is_floored(trained, tmp_path) -> None:
    config = make_config(
        {
            "calibrate.spread_floor_cm": 1000.0,
            "evaluate.views": [1],
            "evaluate.noise_deg": [0],
        }
    )
    out = tmp_path / "floored"
    shutil.copytree(trained.out, out, ignore=shutil.ignore_patterns("predict"))
    run_predict(config, out)
    cell = read_cell(predict_directory(out), "cal", 1, 0)
    assert (cell["spread"] == 1000.0).all()


def test_summarize_samples_scaled_mad_and_floor() -> None:
    rng = np.random.default_rng(3)
    draws = rng.normal(loc=50.0, scale=2.0, size=(3, 4001, 5))
    median, spread = summarize_samples(draws, FLOOR_CM)
    assert np.allclose(median, 50.0, atol=0.3)
    assert np.allclose(spread, 2.0, atol=0.2)  # 1.4826 * MAD estimates the standard deviation
    constant = np.full((2, 6, 5), 7.0)
    median, spread = summarize_samples(constant, FLOOR_CM)
    assert (median == 7.0).all() and (spread == FLOOR_CM).all()


def stand_in_measurer():
    return predict_module._MeshMeasurer(StandInBody(), 0.5)


def test_linearized_equals_exact_at_the_median_sample_and_near_it_one_unit_away() -> None:
    measure = stand_in_measurer()
    center = np.zeros(10)
    identity = np.eye(10)
    # Per coefficient the samples are center - 1, center, center + 1 in that coefficient and the
    # center in the others, so the median sample is the center and the others are one unit away.
    samples = np.concatenate([center[None], center + identity, center - identity])[None]
    approx = linearized_measurements(measure, samples)[0]
    exact = measure(samples[0])
    assert np.array_equal(approx[0], exact[0])  # exactly equal at the median sample
    relative = np.abs(approx - exact) / np.abs(exact)
    # One unit up is the step of the forward difference. The hip is the max over several slices, so
    # one unit down may leave the straight line by a little more than for the other measurements.
    assert relative[1:11].max() < 0.01
    assert relative[11:, [0, 1, 2, 4]].max() < 0.01
    assert relative[11:, 3].max() < 0.02


def test_linearized_mode_measures_eleven_meshes_per_body_and_cell(trained, tmp_path, monkeypatch):
    counts = {"meshes": 0}
    real_measure = predict_module.measure_batch

    def counting_measure(vertices, *args, **kwargs):
        counts["meshes"] += len(vertices)
        return real_measure(vertices, *args, **kwargs)

    monkeypatch.setattr(predict_module, "measure_batch", counting_measure)
    config = make_config(
        {"predict.measure_mode": "linearized", "evaluate.views": [4], "evaluate.noise_deg": [0]}
    )
    out = tmp_path / "linearized"
    shutil.copytree(trained.out, out, ignore=shutil.ignore_patterns("predict"))
    run_predict(config, out)
    rows = sum(len(read_cell(predict_directory(out), name, 4, 0)["body_id"]) for name in SPLITS)
    assert counts["meshes"] == 11 * rows

    counts["meshes"] = 0
    exact_config = make_config({"evaluate.views": [4], "evaluate.noise_deg": [0]})
    run_predict(exact_config, out)
    k = exact_config["predict"]["n_samples"]
    assert counts["meshes"] == k * rows


def test_non_finite_prediction_raises_with_the_sample_id(trained, tmp_path, monkeypatch) -> None:
    real_measure = predict_module.measure_batch

    def poisoned_measure(*args, **kwargs):
        measured = real_measure(*args, **kwargs)
        measured[3, 1] = float("nan")  # mesh 3 of the first call: body 0 of the batch, sample 3
        return measured

    monkeypatch.setattr(predict_module, "measure_batch", poisoned_measure)
    config = make_config({"evaluate.views": [1], "evaluate.noise_deg": [0]})
    out = tmp_path / "poisoned"
    shutil.copytree(trained.out, out, ignore=shutil.ignore_patterns("predict"))
    split = DataSplit(config["data"]["n_train"], config["data"]["n_cal"], config["data"]["n_test"])
    first_id = ShapeDataset.for_cell(
        config, data_directory(out), split.range_of("cal"), views=4, noise_deg=0.0
    ).body_ids[0]
    with pytest.raises(
        PredictionRefusedError, match=rf"body {first_id}, sample 3, measurement chest"
    ):
        run_predict(config, out)
    assert PredictionRefusedError.exit_code == 4
    assert not (predict_directory(out) / DONE_MARKER_NAME).exists()


def test_stage_needs_the_train_stage(trained, tmp_path) -> None:
    out = tmp_path / "untrained"
    shutil.copytree(trained.out, out, ignore=shutil.ignore_patterns("predict", "train"))
    with pytest.raises(FileNotFoundError, match="train stage is not done"):
        run_predict(trained.config, out)
