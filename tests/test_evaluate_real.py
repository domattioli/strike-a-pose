"""Tests for real/evaluate_real.py: tables, skip counts, matching cell, merging, and refusals.

Every test builds a small BodyM-like or SSP-3D-like dataset inside tmp_path, an untrained model, and
the calibration files of a run. No licensed asset is read and nothing is downloaded.
"""

import csv
import json
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pytest
import torch

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.body.standin import StandInBody
from strike_a_pose.calibrate import QUANTILE_COLUMNS
from strike_a_pose.checkpoint import write_done_marker
from strike_a_pose.config import config_hash, load_config
from strike_a_pose.measure import MEASUREMENT_NAMES
from strike_a_pose.model.vae import ShapeVAE
from strike_a_pose.real.evaluate_real import (
    SUBJECT_COLUMNS,
    RealEvalError,
    run_real_eval,
)
from strike_a_pose.real.ssp3d import ground_truth_measurements, load_labels
from strike_a_pose.report.tables import REAL_RESULT_COLUMNS
from strike_a_pose.runrecord import start_run_record, write_run_record

CODE_VERSION = "test-version-7"
N_CAL = {"v1_n0": 31, "v2_n0": 29, "v4_n0": 27}
Q_HAT = 1.5
FLOOR_CM = 0.1
STEP_CM = 1.0
SIDE = 32

OVERRIDES: dict[str, object] = {
    "data.n_train": 64,
    "data.n_cal": 32,
    "data.n_test": 32,
    "data.shard_size": 32,
    "data.min_unflagged": 16,
    "calibrate.min_cal": 16,
    "train.epochs": 1,
    "predict.n_samples": 4,
    "camera.image_size": SIDE,
    "camera.focal_px": SIDE,
    "evaluate.views": [1, 2, 4],
    "evaluate.noise_deg": [0],
    "measure.step_cm": STEP_CM,
}


class Run(NamedTuple):
    """A configuration and the output folder that holds its model, quantiles, and run record."""

    config: dict[str, Any]
    out: Path


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def stand_in() -> StandInBody:
    return StandInBody(radial_segments=8, cap_rings=2)


# Mask pictures ---------------------------------------------------------------------------------


def person_mask() -> np.ndarray:
    """A 64 by 48 picture with one person-sized block (19 percent of the picture)."""
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:50, 14:34] = 255
    return mask


def two_person_mask() -> np.ndarray:
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:50, 4:14] = 255
    mask[10:50, 30:40] = 255
    return mask


def tiny_mask() -> np.ndarray:
    """A block of 16 pixels, which is below 2 percent of the picture."""
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:14, 10:14] = 255
    return mask


def write_png(path: Path, picture: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), picture)


# Run folder ------------------------------------------------------------------------------------


def make_config(tmp_path: Path, **paths: Path) -> dict[str, Any]:
    merged = dict(OVERRIDES)
    for name, path in paths.items():
        merged[f"real.{name}.path"] = str(path)
    tiny = Path(__file__).resolve().parent.parent / "configs" / "tiny.yaml"
    return load_config(tiny, [f"{key}={json.dumps(value)}" for key, value in merged.items()])


def make_run(
    tmp_path: Path, *, q_hat: float = Q_HAT, v1_q_hat: float | None = None, **paths: Path
) -> Run:
    """Write an untrained model, quantiles for three cells, the stage markers, and a run record."""
    config = make_config(tmp_path, **paths)
    out = tmp_path / "out"
    digest = config_hash(config)
    torch.manual_seed(0)
    (out / "train").mkdir(parents=True)
    torch.save({"model": ShapeVAE.from_config(config).state_dict()}, out / "train/model_final.pt")
    (out / "calibrate").mkdir()
    with (out / "calibrate/quantiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(QUANTILE_COLUMNS)
        for cell, views in (("v1_n0", 1), ("v2_n0", 2), ("v4_n0", 4)):
            level = v1_q_hat if (cell == "v1_n0" and v1_q_hat is not None) else q_hat
            for name in MEASUREMENT_NAMES:
                writer.writerow([cell, views, 0.0, name, N_CAL[cell], 0.1, level, FLOOR_CM])
    for stage in ("train", "calibrate"):
        write_done_marker(
            out / stage,
            stage=stage,
            config_hash=digest,
            seed=int(config["seed"]),
            code_version=CODE_VERSION,
            hardware_class="cpu",
            inputs={},
        )
    record = start_run_record(
        config, hardware_class="cpu", device_name="test cpu", code_version=CODE_VERSION
    )
    write_run_record(record, out / "run_record.json")
    return Run(config, out)


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


# BodyM fixture ---------------------------------------------------------------------------------

TAPE = ("171.5", "96.0", "81.2", "98.4", "55.1")


def write_split(
    root: Path, folder: str, subjects: dict[str, tuple[np.ndarray | None, np.ndarray | None]]
) -> None:
    """Write a split whose subject i has photo p<i>; a mask of None writes no file."""
    split = root / folder
    split.mkdir(parents=True)
    headers = ["subject_id", "Height (cm)", "Chest Girth (cm)", "Waist Girth (cm)"]
    headers += ["Hip Girth (cm)", "Thigh Girth (cm)"]
    with (split / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        for subject in subjects:
            values = ("", *TAPE[1:]) if subject.endswith("blank") else TAPE
            writer.writerow([subject, *values])
    (split / "hwg_metadata.csv").write_text("subject_id,Gender\n", encoding="utf-8")
    with (split / "subject_to_photo_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject_id", "photo_id"])
        for subject in subjects:
            if not subject.endswith("unmapped"):
                writer.writerow([subject, f"p_{subject}"])
    for subject, (front, side) in subjects.items():
        if front is not None:
            write_png(split / "frontal" / f"p_{subject}.png", front)
        if side is not None:
            write_png(split / "side" / f"p_{subject}.png", side)


@pytest.fixture
def bodym_root(tmp_path: Path) -> Path:
    """Test A: one good subject and five skipped (one for each reason). Test B: two good ones."""
    root = tmp_path / "bodym"
    person = person_mask()
    write_split(
        root,
        "testA",
        {
            "a_good": (person, person),
            "a_multi": (two_person_mask(), person),
            "a_empty": (person, np.zeros_like(person)),
            "a_tiny": (tiny_mask(), person),
            "a_unmapped": (person, person),
            "a_blank": (person, person),
            "a_nopair": (person, None),
        },
    )
    write_split(root, "testB", {"b_one": (person, person), "b_two": (person, person)})
    return root


# SSP-3D fixture --------------------------------------------------------------------------------

SSP_NAMES = ["s0.png", "s1.png", "s2.png", "s3.png", "s4.png"]


@pytest.fixture
def ssp3d_root(tmp_path: Path) -> Path:
    """Five subjects: two good silhouettes, one with two people, one tiny, one with no file."""
    root = tmp_path / "ssp3d"
    root.mkdir()
    rng = np.random.default_rng(3)
    np.savez(
        root / "labels.npz",
        fnames=np.array(SSP_NAMES),
        shapes=rng.normal(0.0, 0.5, size=(5, 10)),
        genders=np.array(["m", "f", "m", "f", "m"]),
        bbox_whs=rng.uniform(5.0, 40.0, size=(5, 4)),
    )
    silhouettes = [person_mask(), person_mask(), two_person_mask(), tiny_mask(), None]
    for name, mask in zip(SSP_NAMES, silhouettes, strict=True):
        if mask is not None:
            write_png(root / "silhouettes" / name, mask)
        write_png(root / "images" / name, np.full((64, 48, 3), 90, dtype=np.uint8))
    return root


def evaluate(run: Run, stand_in: StandInBody, dataset: str, **keywords: Any):
    return run_real_eval(
        run.config,
        run.out,
        dataset,
        body=stand_in,
        body_factory=lambda gender: stand_in,
        **keywords,
    )


# Tests -----------------------------------------------------------------------------------------


def test_bodym_tables_have_the_contract_columns_and_counts(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    result = evaluate(run, stand_in, "bodym")

    header, rows = read_table(run.out / "real/bodym/results.csv")
    assert header == list(REAL_RESULT_COLUMNS)
    assert len(rows) == 2 * len(MEASUREMENT_NAMES)
    assert [row["measurement"] for row in rows[:5]] == list(MEASUREMENT_NAMES)
    test_a = [row for row in rows if row["split"] == "testA"]
    test_b = [row for row in rows if row["split"] == "testB"]
    assert {row["n_subjects"] for row in test_a} == {"1"}
    # Skipped in test A: no mask, multi person, unusable, and the loader's three reasons.
    assert {row["n_skipped"] for row in test_a} == {"6"}
    assert {row["n_subjects"] for row in test_b} == {"2"}
    assert {row["n_skipped"] for row in test_b} == {"0"}
    assert result.evaluated == {"testA": 1, "testB": 2}
    assert result.skipped == {"testA": 6, "testB": 0}
    assert result.skip_reasons["testA"] == {
        "no_photo_map": 1,
        "no_mask_pair": 1,
        "missing_measurement": 1,
        "skipped_no_mask": 1,
        "skipped_multi_person": 1,
        "skipped_unusable": 1,
    }


def test_rows_carry_the_matching_cell_and_the_run_record(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root, v1_q_hat=9.0)
    evaluate(run, stand_in, "bodym")
    _, rows = read_table(run.out / "real/bodym/results.csv")
    record = json.loads((run.out / "run_record.json").read_text(encoding="utf-8"))
    for row in rows:
        assert row["dataset"] == "bodym"
        assert row["mask_source"] == "provided"
        assert row["cell_id"] == "v2_n0"
        assert row["n_cal"] == str(N_CAL["v2_n0"])
        assert float(row["nominal_level"]) == pytest.approx(0.9)
        assert float(row["q_hat"]) == Q_HAT
        assert row["seed"] == str(record["seed"])
        assert row["config_hash"] == record["config_hash"] == config_hash(run.config)
        assert row["code_version"] == CODE_VERSION
        assert row["hardware_class"] == "cpu"


def test_subjects_table_has_statuses_and_consistent_numbers(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    evaluate(run, stand_in, "bodym")
    header, subjects = read_table(run.out / "real/bodym/subjects.csv")
    assert header == list(SUBJECT_COLUMNS)
    statuses = {row["subject_id"]: row["mask_status"] for row in subjects}
    assert statuses == {
        "a_good": "ok",
        "a_multi": "skipped_multi_person",
        "a_empty": "skipped_no_mask",
        "a_tiny": "skipped_unusable",
        "b_one": "ok",
        "b_two": "ok",
    }
    for row in subjects:
        values = [row[column] for column in SUBJECT_COLUMNS[5:]]
        assert all(values) if row["mask_status"] == "ok" else not any(values)

    _, results = read_table(run.out / "real/bodym/results.csv")
    evaluated = [row for row in subjects if row["split"] == "testB"]
    for result in (row for row in results if row["split"] == "testB"):
        name = result["measurement"]
        truth = np.array([float(row[f"m_true_{name}"]) for row in evaluated])
        median = np.array([float(row[f"m_median_{name}"]) for row in evaluated])
        lower = np.array([float(row[f"lower_{name}"]) for row in evaluated])
        upper = np.array([float(row[f"upper_{name}"]) for row in evaluated])
        covered = [row[f"covered_{name}"] == "true" for row in evaluated]
        assert truth.tolist() == [float(TAPE[MEASUREMENT_NAMES.index(name)])] * 2
        assert float(result["mean_signed_error_cm"]) == pytest.approx(np.mean(median - truth))
        assert float(result["mae_cm"]) == pytest.approx(np.mean(np.abs(median - truth)))
        assert float(result["median_width_cm"]) == pytest.approx(np.median(upper - lower))
        assert float(result["coverage"]) == pytest.approx(np.mean(covered))
        assert covered == list((lower <= truth) & (truth <= upper))
        assert (lower >= 0.0).all() and (upper > lower).all()
        assert int(result["clipped_count"]) >= 0


def test_a_wider_quantile_widens_the_intervals(tmp_path, bodym_root, stand_in):
    narrow = make_run(tmp_path / "narrow", q_hat=1.0, bodym=bodym_root)
    wide = make_run(tmp_path / "wide", q_hat=2.0, bodym=bodym_root)
    evaluate(narrow, stand_in, "bodym")
    evaluate(wide, stand_in, "bodym")
    _, small = read_table(narrow.out / "real/bodym/results.csv")
    _, large = read_table(wide.out / "real/bodym/results.csv")
    for a, b in zip(small, large, strict=True):
        assert float(b["median_width_cm"]) > float(a["median_width_cm"])
        # The point predictions do not depend on the quantile.
        assert a["mean_signed_error_cm"] == b["mean_signed_error_cm"]


def test_the_result_is_reproducible(tmp_path, bodym_root, stand_in):
    first = make_run(tmp_path / "one", bodym=bodym_root)
    second = make_run(tmp_path / "two", bodym=bodym_root)
    evaluate(first, stand_in, "bodym")
    evaluate(second, stand_in, "bodym")
    for name in ("results.csv", "subjects.csv"):
        text_one = (first.out / "real/bodym" / name).read_text(encoding="utf-8")
        text_two = (second.out / "real/bodym" / name).read_text(encoding="utf-8")
        # The run records differ in time only, so the tables, which hold no time, are identical.
        assert text_one == text_two


def test_ssp3d_uses_one_view_the_v1_cell_and_gendered_ground_truth(tmp_path, ssp3d_root, stand_in):
    run = make_run(tmp_path, ssp3d=ssp3d_root, v1_q_hat=2.5)
    result = evaluate(run, stand_in, "ssp3d")
    _, rows = read_table(run.out / "real/ssp3d/results.csv")
    assert len(rows) == len(MEASUREMENT_NAMES)
    for row in rows:
        assert (row["split"], row["mask_source"], row["cell_id"]) == ("all", "provided", "v1_n0")
        assert row["n_cal"] == str(N_CAL["v1_n0"])
        assert float(row["q_hat"]) == 2.5
        assert (row["n_subjects"], row["n_skipped"]) == ("2", "3")
    assert result.skip_reasons["all"] == {
        "skipped_multi_person": 1,
        "skipped_unusable": 1,
        "skipped_no_mask": 1,
    }
    _, subjects = read_table(run.out / "real/ssp3d/subjects.csv")
    assert [row["subject_id"] for row in subjects if row["mask_status"] == "ok"] == ["s0", "s1"]
    truth = ground_truth_measurements(
        load_labels(ssp3d_root), lambda gender: stand_in, STEP_CM, indices=[0, 1]
    )
    for position, row in enumerate(row for row in subjects if row["mask_status"] == "ok"):
        for column, name in enumerate(MEASUREMENT_NAMES):
            assert float(row[f"m_true_{name}"]) == pytest.approx(truth[position, column])


class StubSam2:
    """A mask backend that records the box prompts and returns the person mask for every subject."""

    name = "sam2"

    def __init__(self) -> None:
        self.boxes: dict[str, tuple[float, ...] | None] = {}
        self.image_shapes: dict[str, tuple[int, ...]] = {}

    def segment(self, subject_id, image, box):
        self.boxes[subject_id] = box
        self.image_shapes[subject_id] = image.shape
        return person_mask().astype(bool)


def test_sam2_masks_get_the_photograph_and_the_box_and_rows_are_merged(
    tmp_path, ssp3d_root, stand_in
):
    run = make_run(tmp_path, ssp3d=ssp3d_root)
    evaluate(run, stand_in, "ssp3d")
    backend = StubSam2()
    result = evaluate(run, stand_in, "ssp3d", mask_source="sam2", mask_backend=backend)

    labels = load_labels(ssp3d_root)
    assert set(backend.boxes) == {Path(name).stem for name in SSP_NAMES}
    assert backend.boxes["s2"] == tuple(float(v) for v in labels.boxes[2][:4])
    assert set(backend.image_shapes.values()) == {(64, 48, 3)}
    assert result.mask_source == "sam2"
    assert result.evaluated == {"all": 5} and result.skipped == {"all": 0}

    _, rows = read_table(run.out / "real/ssp3d/results.csv")
    by_source = {row["mask_source"] for row in rows}
    assert by_source == {"provided", "sam2"}
    assert len(rows) == 2 * len(MEASUREMENT_NAMES)
    _, subjects = read_table(run.out / "real/ssp3d/subjects.csv")
    assert len(subjects) == 5 + 5  # five rows per mask source (the "provided" run kept all five)

    # A second run with the provided masks replaces its own rows and keeps the sam2 rows.
    evaluate(run, stand_in, "ssp3d")
    _, again = read_table(run.out / "real/ssp3d/results.csv")
    assert len(again) == len(rows)
    assert {row["mask_source"] for row in again} == {"provided", "sam2"}


def test_rows_of_another_configuration_are_not_kept(tmp_path, ssp3d_root, stand_in):
    run = make_run(tmp_path, ssp3d=ssp3d_root)
    evaluate(run, stand_in, "ssp3d", mask_source="sam2", mask_backend=StubSam2())
    results = run.out / "real/ssp3d/results.csv"
    results.write_text(
        results.read_text(encoding="utf-8").replace(config_hash(run.config), "0" * 64),
        encoding="utf-8",
    )
    evaluate(run, stand_in, "ssp3d")
    _, rows = read_table(results)
    assert {row["mask_source"] for row in rows} == {"provided"}
    _, subjects = read_table(run.out / "real/ssp3d/subjects.csv")
    assert {row["mask_source"] for row in subjects} == {"provided"}


def test_a_missing_dataset_names_the_configuration_key(tmp_path, stand_in):
    run = make_run(tmp_path)
    with pytest.raises(MissingAssetError, match=r"real\.bodym\.path"):
        evaluate(run, stand_in, "bodym")
    absent = make_run(tmp_path / "other", ssp3d=tmp_path / "nowhere")
    with pytest.raises(MissingAssetError, match=r"nowhere"):
        evaluate(absent, stand_in, "ssp3d")


def test_a_missing_input_stage_is_refused(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    (run.out / "calibrate/DONE.json").unlink()
    with pytest.raises(RealEvalError, match="calibrate stage is not done"):
        evaluate(run, stand_in, "bodym")
    assert not (run.out / "real").exists()


def test_a_run_record_of_another_configuration_is_refused(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    changed = dict(run.config)
    changed["predict"] = {**run.config["predict"], "n_samples": 5}
    with pytest.raises(RealEvalError, match="names configuration"):
        run_real_eval(changed, run.out, "bodym", body=stand_in)


def test_a_cell_without_quantiles_is_refused(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    path = run.out / "calibrate/quantiles.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line for line in lines if not line.startswith("v2_n0")) + "\n")
    with pytest.raises(RealEvalError, match="v2_n0"):
        evaluate(run, stand_in, "bodym")


def test_a_dataset_without_an_evaluable_subject_is_refused(tmp_path, stand_in):
    root = tmp_path / "bodym"
    write_split(root, "testA", {"a_empty": (np.zeros((64, 48), np.uint8),) * 2})
    write_split(root, "testB", {"b_tiny": (tiny_mask(), tiny_mask())})
    run = make_run(tmp_path, bodym=root)
    with pytest.raises(RealEvalError, match="no bodym subject could be evaluated"):
        evaluate(run, stand_in, "bodym")
    assert RealEvalError.exit_code == 4


def test_a_split_without_an_evaluable_subject_has_no_rows(tmp_path, stand_in):
    root = tmp_path / "bodym"
    write_split(root, "testA", {"a_empty": (np.zeros((64, 48), np.uint8),) * 2})
    write_split(root, "testB", {"b_one": (person_mask(), person_mask())})
    run = make_run(tmp_path, bodym=root)
    result = evaluate(run, stand_in, "bodym")
    _, rows = read_table(run.out / "real/bodym/results.csv")
    assert {row["split"] for row in rows} == {"testB"}
    assert result.evaluated == {"testA": 0, "testB": 1}
    assert result.skipped == {"testA": 1, "testB": 0}


def test_unknown_dataset_and_mask_source_are_refused(tmp_path, bodym_root, stand_in):
    run = make_run(tmp_path, bodym=bodym_root)
    with pytest.raises(ValueError, match="dataset must be one of"):
        evaluate(run, stand_in, "coco")
    with pytest.raises(ValueError, match="mask_source"):
        evaluate(run, stand_in, "ssp3d", mask_source="magic")
    with pytest.raises(ValueError, match="applies to ssp3d"):
        evaluate(run, stand_in, "bodym", mask_source="sam2")
