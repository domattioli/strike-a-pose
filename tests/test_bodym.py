"""Smoke tests for real/bodym.py: split folders, header-named columns, mask joins, and skips.

Every test builds a small BodyM-like directory inside tmp_path. No licensed asset is read.
"""

import csv
import logging
from pathlib import Path

import pytest

from strike_a_pose.assets import MissingAssetError
from strike_a_pose.real.bodym import (
    BODYM_PATH_KEY,
    MEASUREMENT_FILE,
    PHOTO_MAP_FILE,
    discover_split,
    load_bodym,
    load_split,
)

# Tape values in cm: height, chest, waist, hip, thigh.
TAPE_1001 = (171.5, 96.0, 81.2, 98.4, 55.1)
TAPE_1003 = (165.0, 90.0, 75.0, 92.0, 50.0)
TAPE_1004 = (180.0, 100.0, 88.0, 101.0, 58.0)
HEIGHT_2001 = 168.0
TAPE_2001 = (78.0, 95.0, 52.0, 92.5)  # chest, waist, hip, thigh

# Test A: the photo map lists p2 before p1 for subject 1001, and p2 has only a front mask, so the
# join must pick p1. Subject 1002 has no map row, 1003 has only a front mask, and 1004 has a blank
# waist.
MEASUREMENT_ROWS_A = [
    {"subject_id": "1001", "Height (cm)": "171.5", "Chest Girth (cm)": "96.0",
     "Waist Girth (cm)": "81.2", "Hip Girth (cm)": "98.4", "Thigh Girth (cm)": "55.1",
     "Weight (kg)": "70.2"},
    {"subject_id": "1002", "Height (cm)": "170.0", "Chest Girth (cm)": "94.0",
     "Waist Girth (cm)": "80.0", "Hip Girth (cm)": "97.0", "Thigh Girth (cm)": "54.0",
     "Weight (kg)": "68.0"},
    {"subject_id": "1003", "Height (cm)": "165.0", "Chest Girth (cm)": "90.0",
     "Waist Girth (cm)": "75.0", "Hip Girth (cm)": "92.0", "Thigh Girth (cm)": "50.0",
     "Weight (kg)": "60.0"},
    {"subject_id": "1004", "Height (cm)": "180.0", "Chest Girth (cm)": "100.0",
     "Waist Girth (cm)": "", "Hip Girth (cm)": "101.0", "Thigh Girth (cm)": "58.0",
     "Weight (kg)": "90.0"},
]
# Column order is deliberately not the standard order.
MEASUREMENT_HEADERS_A = [
    "Weight (kg)", "Thigh Girth (cm)", "subject_id", "Waist Girth (cm)",
    "Chest Girth (cm)", "Hip Girth (cm)", "Height (cm)",
]
PHOTO_ROWS_A = [("1001", "p2"), ("1001", "p1"), ("1003", "p3"), ("1004", "p4")]
FRONT_PHOTOS_A = ["p1", "p2", "p3", "p4"]
SIDE_PHOTOS_A = ["p1", "p4"]


def _write_rows(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV file with the given header order and LF line endings."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header names and rows of a CSV file written by the tests."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_masks(folder: Path, photos: list[str]) -> None:
    """Create one small mask file per photo id, named <photo id>.png."""
    folder.mkdir(parents=True)
    for photo in photos:
        (folder / f"{photo}.png").write_bytes(b"\x89PNG-test-mask")


def build_test_a(root: Path) -> Path:
    """Build the split folder Test-A with the mask folders named like the public layout."""
    folder = root / "Test-A"
    folder.mkdir(parents=True)
    _write_rows(folder / MEASUREMENT_FILE, MEASUREMENT_HEADERS_A, MEASUREMENT_ROWS_A)
    _write_rows(
        folder / "hwg_metadata.csv",
        ["subject_id", "Gender", "Weight (kg)", "Height (cm)"],
        [
            {"subject_id": "1001", "Gender": "f", "Weight (kg)": "70.2", "Height (cm)": "171.5"},
            {"subject_id": "1002", "Gender": "m", "Weight (kg)": "68.0", "Height (cm)": "170.0"},
            {"subject_id": "1003", "Gender": "f", "Weight (kg)": "60.0", "Height (cm)": "165.0"},
            {"subject_id": "1004", "Gender": "m", "Weight (kg)": "90.0", "Height (cm)": "180.0"},
        ],
    )
    _write_rows(
        folder / PHOTO_MAP_FILE,
        ["Subject ID", "Photo ID"],
        [{"Subject ID": s, "Photo ID": p} for s, p in PHOTO_ROWS_A],
    )
    _write_masks(folder / "Frontal Masks", FRONT_PHOTOS_A)
    _write_masks(folder / "Side Masks", SIDE_PHOTOS_A)
    return folder


def build_test_b(root: Path) -> Path:
    """Build the split folder test_b, whose height exists only in hwg_metadata.csv."""
    folder = root / "test_b"
    folder.mkdir(parents=True)
    _write_rows(
        folder / MEASUREMENT_FILE,
        [
            "subject_id",
            "Chest Girth (cm)",
            "Waist Girth (cm)",
            "Hip Girth (cm)",
            "Thigh Girth (cm)",
        ],
        [{
            "subject_id": "2001",
            "Chest Girth (cm)": "78.0",
            "Waist Girth (cm)": "95.0",
            "Hip Girth (cm)": "52.0",
            "Thigh Girth (cm)": "92.5",
        }],
    )
    _write_rows(
        folder / "hwg_metadata.csv",
        ["subject_id", "Height (cm)", "Weight (kg)", "Gender"],
        [{"subject_id": "2001", "Height (cm)": "168.0", "Weight (kg)": "66.0", "Gender": "f"}],
    )
    _write_rows(folder / PHOTO_MAP_FILE, ["subject_id", "photo_id"], [
        {"subject_id": "2001", "photo_id": "q1"},
    ])
    _write_masks(folder / "frontal_masks", ["q1"])
    _write_masks(folder / "side_masks", ["q1"])
    return folder


@pytest.fixture
def bodym_root(tmp_path: Path) -> Path:
    """A BodyM-like root with split folders Test-A and test_b."""
    root = tmp_path / "bodym"
    root.mkdir()
    build_test_a(root)
    build_test_b(root)
    return root


def test_columns_are_found_by_header_name_whatever_their_order(bodym_root):
    loaded = load_split(bodym_root, "testA")
    subject = next(s for s in loaded.subjects if s.subject_id == "1001")
    assert subject.measurements_cm == TAPE_1001
    assert loaded.columns["height"] == f"{MEASUREMENT_FILE}:Height (cm)"
    assert loaded.columns["waist"] == f"{MEASUREMENT_FILE}:Waist Girth (cm)"
    assert loaded.columns["subject"] == f"{MEASUREMENT_FILE}:subject_id"


def test_each_subject_joins_a_front_and_side_mask_through_a_shared_photo_id(bodym_root):
    loaded = load_split(bodym_root, "testA")
    subject = next(s for s in loaded.subjects if s.subject_id == "1001")
    assert subject.front_mask == bodym_root / "Test-A" / "Frontal Masks" / "p1.png"
    assert subject.side_mask == bodym_root / "Test-A" / "Side Masks" / "p1.png"
    assert subject.split == "testA"


def test_subjects_without_a_usable_pair_or_value_are_skipped_and_counted(bodym_root):
    loaded = load_split(bodym_root, "testA")
    assert [s.subject_id for s in loaded.subjects] == ["1001"]
    assert loaded.skipped == {"no_photo_map": 1, "no_mask_pair": 1, "missing_measurement": 1}


def test_height_falls_back_to_metadata_and_the_folder_name_is_normalized(bodym_root):
    loaded = load_split(bodym_root, "testB")
    assert loaded.columns["height"] == "hwg_metadata.csv:Height (cm)"
    (subject,) = loaded.subjects
    assert subject.measurements_cm == (HEIGHT_2001, *TAPE_2001)
    assert subject.front_mask.parent.name == "frontal_masks"


def test_load_bodym_returns_every_requested_split(bodym_root):
    loaded = load_bodym(bodym_root)
    assert set(loaded) == {"testA", "testB"}
    assert len(loaded["testA"].subjects) == 1
    assert len(loaded["testB"].subjects) == 1


def test_discover_split_finds_each_folder_and_file(bodym_root):
    layout = discover_split(bodym_root, "testA")
    assert layout.folder == bodym_root / "Test-A"
    assert layout.front_folder.name == "Frontal Masks"
    assert layout.side_folder.name == "Side Masks"
    assert layout.photo_map_csv.name == PHOTO_MAP_FILE


def test_a_missing_root_names_the_configuration_key(tmp_path):
    with pytest.raises(MissingAssetError) as caught:
        load_bodym(tmp_path / "absent")
    assert caught.value.key == BODYM_PATH_KEY


def test_a_missing_split_folder_is_refused_with_the_configuration_key(tmp_path):
    root = tmp_path / "bodym"
    build_test_b(root)
    with pytest.raises(MissingAssetError) as caught:
        load_split(root, "testA")
    assert caught.value.key == BODYM_PATH_KEY


def test_two_columns_for_one_measurement_are_refused(bodym_root):
    path = bodym_root / "Test-A" / MEASUREMENT_FILE
    headers, rows = _read_rows(path)
    for row in rows:
        row["Chest Girth Std (cm)"] = "1.0"
    _write_rows(path, [*headers, "Chest Girth Std (cm)"], rows)
    with pytest.raises(ValueError, match="more than one column for 'chest'"):
        load_split(bodym_root, "testA")


def test_a_subject_listed_twice_is_refused(bodym_root):
    path = bodym_root / "Test-A" / MEASUREMENT_FILE
    headers, rows = _read_rows(path)
    _write_rows(path, headers, [*rows, rows[0]])
    with pytest.raises(ValueError, match="'1001' twice"):
        load_split(bodym_root, "testA")


def test_the_columns_and_the_skip_counts_are_logged(bodym_root, caplog):
    with caplog.at_level(logging.INFO, logger="strike_a_pose.real.bodym"):
        load_split(bodym_root, "testA")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "BodyM testA" in text
    assert "Height (cm)" in text
    assert "missing_measurement" in text
