"""Pose-level frozen split (spec 002 FR-006, FR-007; SC-002, SC-003)."""

import json

import pytest

from strike_a_pose.data.split import (
    SPLITS,
    SplitError,
    check_split_for,
    load_split,
    make_split,
    parse_sample_id,
    write_split,
)
from strike_a_pose.datagen.io import dumps

CAMS = ["Front_0", "FrontRight_45", "Right_90", "BackRight_135", "Back_180", "Left_270"]
SHA = "ab" * 32


def synth_manifest(n_poses: int, cams=CAMS, name: str = "synth") -> dict:
    """Manifest-shaped dict (only the fields the split reads); no files."""
    return {
        "dataset_name": name,
        "samples": [{"id": f"{p:05d}_{c}"} for p in range(n_poses) for c in cams],
    }


def test_parse_sample_id():
    assert parse_sample_id("00012_Front_0") == (12, "Front_0")
    assert parse_sample_id("123456_Back_Left_225") == (123456, "Back_Left_225")
    for bad in ["Front_0", "00012", "00012_", "x1_Front_0"]:
        with pytest.raises(ValueError):
            parse_sample_id(bad)


def test_deterministic_bytes_and_seed_sensitive():
    m = synth_manifest(100)
    a, b = make_split(m, SHA, seed=0), make_split(m, SHA, seed=0)
    assert dumps(a) == dumps(b)
    assert make_split(m, SHA, seed=1)["poses"] != a["poses"]


def test_disjoint_cover_and_counts():
    m = synth_manifest(100)
    s = make_split(m, SHA, seed=7)
    sets = [set(s["poses"][k]) for k in SPLITS]
    assert sum(len(x) for x in sets) == len(set().union(*sets)) == 100
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 80, "val": 10, "test": 10}
    assert all(s["counts"][k]["samples"] == 6 * s["counts"][k]["poses"] for k in SPLITS)
    assert all(s["poses"][k] == sorted(s["poses"][k]) for k in SPLITS)
    assert s["unit"] == "pose_index" and s["manifest_sha256"] == SHA and s["seed"] == 7
    assert check_split_for(s, SHA, range(100)) == []


def test_min_one_pose_per_nonzero_fraction_and_zero_fraction_empty():
    s = make_split(synth_manifest(5), SHA, seed=0)  # round(0.5)=0 -> raised to 1
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 3, "val": 1, "test": 1}
    s = make_split(synth_manifest(2), SHA, seed=0, fractions=(0.5, 0.0, 0.5))
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 1, "val": 0, "test": 1}


def test_too_few_poses_for_train():
    with pytest.raises(SplitError, match="train >= 1"):
        make_split(synth_manifest(2), SHA, seed=0)  # 0.8/0.1/0.1 -> val 1, test 1, train 0


@pytest.mark.parametrize(
    "fr", [(0.8, 0.1), (0.5, 0.5, 0.5), (1.1, -0.1, 0.0), (0.0, 0.5, 0.5), (float("nan"), 0, 1)]
)
def test_bad_fractions(fr):
    with pytest.raises(SplitError):
        make_split(synth_manifest(10), SHA, seed=0, fractions=fr)


def test_write_refuses_existing_and_load_round_trip(tmp_path):
    s = make_split(synth_manifest(20), SHA, seed=3)
    p = write_split(s, tmp_path / "splits" / "x.json")
    with pytest.raises(FileExistsError):
        write_split(s, p)
    write_split(s, p, overwrite=True)
    assert load_split(p) == json.loads(dumps(s))


def _write_raw(tmp_path, obj):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(obj))
    return p


def test_load_rejects_bad_files(tmp_path):
    good = json.loads(dumps(make_split(synth_manifest(20), SHA, seed=3)))
    with pytest.raises(SplitError):
        load_split(tmp_path / "missing.json")
    with pytest.raises(SplitError, match="keys"):
        load_split(_write_raw(tmp_path, {**good, "extra": 1}))
    with pytest.raises(SplitError, match="split_version"):
        load_split(_write_raw(tmp_path, {**good, "split_version": 2}))
    overlap = {**good, "poses": {**good["poses"], "val": good["poses"]["train"][:1]}}
    with pytest.raises(SplitError, match="repeated"):
        load_split(_write_raw(tmp_path, overlap))


def test_check_split_for_reports_hash_and_coverage():
    s = make_split(synth_manifest(20), SHA, seed=3)
    errs = check_split_for(s, "cd" * 32, range(21))
    assert len(errs) == 2
    assert "manifest" in errs[0] and "cover" in errs[1]
