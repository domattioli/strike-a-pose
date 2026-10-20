"""Pose-level frozen split (spec 002 FR-006, FR-007; SC-002, SC-003) and split file v2
with the manifest content hash (spec 007 FR-001, FR-002, FR-004, FR-005, FR-010 a/c/d/g)."""

import json
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.data.split import (
    SPLIT_KEYS,
    SPLITS,
    SplitCommitWarning,
    SplitError,
    check_split_code,
    check_split_for,
    load_split,
    make_split,
    manifest_content_sha256,
    parse_sample_id,
    write_split,
)
from strike_a_pose.datagen.io import dumps

ROOT = Path(__file__).parents[1]
SPLIT_V1 = ROOT / "tests" / "fixtures" / "smpl_fixture_split_v1.json"  # pre-007 fixture split
CAMS = ["Front_0", "FrontRight_45", "Right_90", "BackRight_135", "Back_180", "Left_270"]
CODE = {"commit": "ab" * 20, "dirty": False}
RAW = "cd" * 32  # stand-in raw manifest hash (synthetic manifests have no file)
OTHER_CODES = {
    "commit": {"commit": "ef" * 20, "dirty": False},  # regenerated at another commit
    "dirty": {"commit": "ab" * 20, "dirty": True},  # same commit, uncommitted changes
    "unknown": {"commit": "unknown", "dirty": None},  # no git at generation time
}


def synth_manifest(n_poses: int, cams=CAMS, name: str = "synth") -> dict:
    """Manifest-shaped dict (the fields the split reads, plus code / seed); no files."""
    return {
        "dataset_name": name,
        "code": dict(CODE),
        "seed": 0,
        "samples": [
            {"id": f"{p:05d}_{c}", "labels_sha256": f"{p:064x}"}
            for p in range(n_poses)
            for c in cams
        ],
    }


def test_parse_sample_id():
    assert parse_sample_id("00012_Front_0") == (12, "Front_0")
    assert parse_sample_id("123456_Back_Left_225") == (123456, "Back_Left_225")
    for bad in ["Front_0", "00012", "00012_", "x1_Front_0"]:
        with pytest.raises(ValueError):
            parse_sample_id(bad)


def test_deterministic_bytes_and_seed_sensitive():
    m = synth_manifest(100)
    a, b = make_split(m, seed=0), make_split(m, seed=0)
    assert dumps(a) == dumps(b)
    assert make_split(m, seed=1)["poses"] != a["poses"]


def test_disjoint_cover_and_counts():
    m = synth_manifest(100)
    s = make_split(m, seed=7)
    sets = [set(s["poses"][k]) for k in SPLITS]
    assert sum(len(x) for x in sets) == len(set().union(*sets)) == 100
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 80, "val": 10, "test": 10}
    assert all(s["counts"][k]["samples"] == 6 * s["counts"][k]["poses"] for k in SPLITS)
    assert all(s["poses"][k] == sorted(s["poses"][k]) for k in SPLITS)
    assert s["unit"] == "pose_index" and s["seed"] == 7 and s["split_version"] == 2
    assert check_split_for(s, RAW, manifest_content_sha256(m), range(100)) == []


def test_v2_records_content_hash_and_code():
    """007 FR-002: content hash + verbatim code; assignment is pose-permutation-v1."""
    m = synth_manifest(20)
    s = make_split(m, seed=3)
    assert set(s) == SPLIT_KEYS[2] and "manifest_sha256" not in s
    assert s["manifest_content_sha256"] == manifest_content_sha256(m)
    assert s["manifest_code"] == CODE and s["manifest_code"] is not m["code"]
    perm = np.random.default_rng(3).permutation(np.arange(20)).tolist()  # 2 test, 2 val
    assert s["algorithm"] == "pose-permutation-v1" and s["poses"] == {
        "test": sorted(perm[:2]), "val": sorted(perm[2:4]), "train": sorted(perm[4:]),
    }  # fmt: skip


@pytest.mark.parametrize("code", list(OTHER_CODES.values()), ids=list(OTHER_CODES))
def test_content_hash_ignores_code(code):
    """007 FR-001: the generating code is not content."""
    m = synth_manifest(5)
    h = manifest_content_sha256(m)
    assert len(h) == 64
    assert manifest_content_sha256({**m, "code": code}) == h
    assert manifest_content_sha256({k: v for k, v in m.items() if k != "code"}) == h


def test_content_hash_sees_everything_else():
    m = synth_manifest(5)
    h = manifest_content_sha256(m)
    assert manifest_content_sha256(dict(reversed(m.items()))) == h  # key order is not content
    assert manifest_content_sha256({**m, "seed": 1}) != h
    assert manifest_content_sha256({**m, "dataset_name": "other"}) != h
    samples = [dict(s) for s in m["samples"]]
    samples[0]["labels_sha256"] = "f" * 64
    assert manifest_content_sha256({**m, "samples": samples}) != h


def test_min_one_pose_per_nonzero_fraction_and_zero_fraction_empty():
    s = make_split(synth_manifest(5), seed=0)  # round(0.5)=0 -> raised to 1
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 3, "val": 1, "test": 1}
    s = make_split(synth_manifest(2), seed=0, fractions=(0.5, 0.0, 0.5))
    assert {k: s["counts"][k]["poses"] for k in SPLITS} == {"train": 1, "val": 0, "test": 1}


def test_too_few_poses_for_train():
    with pytest.raises(SplitError, match="train >= 1"):
        make_split(synth_manifest(2), seed=0)  # 0.8/0.1/0.1 -> val 1, test 1, train 0


@pytest.mark.parametrize(
    "fr", [(0.8, 0.1), (0.5, 0.5, 0.5), (1.1, -0.1, 0.0), (0.0, 0.5, 0.5), (float("nan"), 0, 1)]
)
def test_bad_fractions(fr):
    with pytest.raises(SplitError):
        make_split(synth_manifest(10), seed=0, fractions=fr)


def test_write_refuses_existing_and_load_round_trip(tmp_path):
    s = make_split(synth_manifest(20), seed=3)
    p = write_split(s, tmp_path / "splits" / "x.json")
    with pytest.raises(FileExistsError):
        write_split(s, p)
    write_split(s, p, overwrite=True)
    assert load_split(p) == json.loads(dumps(s))


def _write_raw(tmp_path, obj):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(obj))
    return p


def test_load_accepts_v1_and_v2(tmp_path):
    """007 FR-005: split_version 1 files stay readable (incl. the committed smpl_proc_v1)."""
    v1 = load_split(SPLIT_V1)
    assert v1 == json.loads(SPLIT_V1.read_text()) and set(v1) == SPLIT_KEYS[1]
    assert v1["split_version"] == 1 and v1["poses"] == {"test": [0], "train": [1], "val": []}
    proc = load_split(ROOT / "splits" / "smpl_proc_v1.json")  # v1 until 007 follow-up F1
    assert proc["split_version"] == 1 and proc["counts"]["train"]["poses"] == 1600
    v2 = load_split(write_split(make_split(synth_manifest(20), seed=3), tmp_path / "v2.json"))
    assert v2["split_version"] == 2 and set(v2) == SPLIT_KEYS[2]


def test_load_rejects_bad_files(tmp_path):
    good = json.loads(dumps(make_split(synth_manifest(20), seed=3)))
    v1 = json.loads(SPLIT_V1.read_text())
    with pytest.raises(SplitError):
        load_split(tmp_path / "missing.json")
    for not_split in ({}, [1, 2], "split"):
        with pytest.raises(SplitError, match="not a split file"):
            load_split(_write_raw(tmp_path, not_split))
    for mixed in (
        {**good, "extra": 1},
        {**good, "manifest_sha256": RAW},  # v2 still carrying the v1 raw hash
        {**v1, "split_version": 2},  # v1 keys labelled version 2
        {**good, "split_version": 1},  # v2 keys labelled version 1
    ):
        with pytest.raises(SplitError, match="keys"):
            load_split(_write_raw(tmp_path, mixed))
    for version in (0, 3, "2", True, None):
        with pytest.raises(SplitError, match="split_version"):
            load_split(_write_raw(tmp_path, {**good, "split_version": version}))
    with pytest.raises(SplitError, match="unit"):
        load_split(_write_raw(tmp_path, {**good, "unit": "sample"}))
    with pytest.raises(SplitError, match="manifest_code"):
        load_split(_write_raw(tmp_path, {**good, "manifest_code": "ab" * 20}))
    overlap = {**good, "poses": {**good["poses"], "val": good["poses"]["train"][:1]}}
    with pytest.raises(SplitError, match="repeated"):
        load_split(_write_raw(tmp_path, overlap))


def test_check_split_for_reports_hash_and_coverage():
    m = synth_manifest(20)
    s = make_split(m, seed=3)
    errs = check_split_for(s, RAW, "ef" * 32, range(21))
    assert len(errs) == 2
    assert "split was made for manifest content" in errs[0] and "cover" in errs[1]
    assert check_split_for(s, "00" * 32, manifest_content_sha256(m), range(20)) == []  # no raw


def test_check_split_for_v1_is_strict_raw():
    """007 FR-005: v1 compares raw manifest bytes only, and says how to migrate."""
    v1 = load_split(SPLIT_V1)
    assert check_split_for(v1, v1["manifest_sha256"], "ef" * 32, [0, 1]) == []  # no content
    (err,) = check_split_for(v1, "ef" * 32, "ef" * 32, [0, 1])
    assert "split was made for manifest 06787bb039f7" in err
    assert "split_version 1 pins the raw manifest bytes" in err and "--overwrite" in err


def test_check_split_code():
    """007 FR-004 (c, d): exact comparison of the generating code; v1 has none."""
    assert issubclass(SplitCommitWarning, UserWarning)
    s = make_split(synth_manifest(5), seed=0)
    assert check_split_code(s, dict(CODE)) is None
    for code in OTHER_CODES.values():
        note = check_split_code(s, code)
        assert note is not None and "manifest content is identical" in note
        assert json.dumps(code, sort_keys=True) in note and json.dumps(CODE, sort_keys=True) in note
        assert "--overwrite" in note and "manifest_code" in note
    assert check_split_code(load_split(SPLIT_V1), OTHER_CODES["commit"]) is None
