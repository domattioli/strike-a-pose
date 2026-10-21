"""Slice 010 dataset realism options (spec: specs/010-dataset-realism/spec.md)."""

import json
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.datagen.config import ConfigError, config_from_dict, load_config
from strike_a_pose.datagen.generate import generate_dataset
from strike_a_pose.datagen.io import LABEL_KEYS, dumps, load_manifest, load_sample, validate_dataset
from strike_a_pose.datagen.procedural import (
    RANGES_DEG,
    RANGES_TABLES,
    RANGES_VERSION,
    euler_xyz_to_matrix,
    ranges_table,
    sample_procedural_poses,
)

from .test_procedural import _aa_to_matrix

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
BODY_FREE_LABELS = ("sample_id", "pose_index", "camera_name", "pose_source", "pose", "betas",
                    "camera")  # fmt: skip


def _fixture() -> dict:
    return load_manifest(FIXTURE)


def _json(x):
    return json.loads(dumps(x))


# --- FR-009: today's datasets stay byte-identical (pins against the committed fixture) -------


def test_fixture_config_serializes_to_fixture_manifest():
    assert load_config(ROOT / "configs" / "fixture.yaml").to_dict() == _fixture()["config"]


def test_default_config_serializes_as_before():
    want = {**_fixture()["config"], "dataset_name": "smpl_proc_v1", "num_poses": 2000}
    assert load_config(ROOT / "configs" / "default.yaml").to_dict() == want


def test_default_table_reproduces_fixture_poses():
    for p in sample_procedural_poses(2, 3):
        lab = json.loads((FIXTURE / "samples" / f"{p.index:05d}_Front_0.json").read_text())
        assert lab["pose_source"]["euler_deg"] == [list(e) for e in p.euler_deg]
        assert lab["pose"] == _json(list(p.pose))


def test_fixture_config_regenerates_body_free_content(synthetic_smpl_npz, tmp_path):
    """Everything the body model does not touch equals the committed fixture."""
    m = _fixture()
    cfg = config_from_dict({**m["config"], "smpl_model_path": str(synthetic_smpl_npz)})
    out = generate_dataset(cfg, seed=3, output_root=tmp_path)
    g = load_manifest(out)
    assert g["config"] == {**m["config"], "smpl_model_path": str(synthetic_smpl_npz)}
    assert g["inputs"]["pose_source"] == m["inputs"]["pose_source"]
    assert g["cameras"] == m["cameras"] and g["counts"] == m["counts"]
    assert [s["id"] for s in g["samples"]] == [s["id"] for s in m["samples"]]
    for s in m["samples"]:
        want = json.loads((FIXTURE / s["labels"]).read_text())
        got = json.loads((out / s["labels"]).read_text())
        assert set(got) == set(want) == LABEL_KEYS
        assert set(got["camera"]) == set(want["camera"])
        for k in BODY_FREE_LABELS:
            assert got[k] == want[k], (s["id"], k)


# --- FR-003: arms-down table -------------------------------------------------------------

V2 = "predecessor-v2-armsdown"
CHANGED = [(13, 2), (14, 2), (16, 2), (17, 2)]  # (joint, axis) narrowed by the v2 table


def _offsets() -> np.ndarray:
    off = np.zeros((24, 3))
    off[16, 2], off[17, 2] = -65.0, 65.0
    return off


def test_armsdown_table_definition():
    v1, v2 = RANGES_TABLES[RANGES_VERSION], RANGES_TABLES[V2]
    assert v1.ranges is RANGES_DEG and v1.offsets is None
    diff = [(j, a) for j in range(24) for a in range(3) if v1.ranges[j][a] != v2.ranges[j][a]]
    assert diff == CHANGED
    assert v2.ranges[13][2] == v2.ranges[14][2] == (-10, 10)
    assert v2.ranges[16][2] == v2.ranges[17][2] == (-20, 20)
    assert v2.offsets == {16: (0, 0, -65), 17: (0, 0, 65)}
    zero = [(j, a) for j in range(24) for a in range(3) if v1.ranges[j][a] == (0, 0)]
    assert all(v2.ranges[j][a] == (0, 0) for j, a in zero)  # hands etc. stay zero


def test_armsdown_samples_offsets_and_same_draws():
    off = _offsets()
    v2 = sample_procedural_poses(300, 7, V2)
    v1 = sample_procedural_poses(300, 7)
    keep = np.ones((24, 3), bool)
    for j, a in CHANGED:
        keep[j, a] = False
    for p, q in zip(v2, v1, strict=True):
        deg, base = np.array(p.euler_deg), np.array(q.euler_deg)
        for j in range(24):
            for a in range(3):
                lo, hi = RANGES_TABLES[V2].ranges[j][a]
                assert deg[j, a] == off[j, a] or lo + off[j, a] <= deg[j, a] < hi + off[j, a]
            r = euler_xyz_to_matrix(np.radians(deg[j]))
            assert np.allclose(_aa_to_matrix(np.array(p.pose).reshape(24, 3)[j]), r, atol=1e-6)
        assert (deg[keep] == base[keep]).all()  # every other draw identical (research R1)
        sparse = base[~keep] == 0
        assert ((deg - off)[~keep][sparse] == 0).all()  # same sparsification pattern
    assert np.mean([np.array(p.euler_deg)[16, 2] == -65.0 for p in v2]) > 0.3


def test_ranges_table_rows():
    v1_rows, v2_rows = ranges_table(), ranges_table(V2)
    assert all("offset_deg" not in r for r in v1_rows)
    assert [r["offset_deg"] for r in v2_rows if r["index"] in (16, 17)] == [[0, 0, -65], [0, 0, 65]]
    assert all(r["offset_deg"] == [0, 0, 0] for r in v2_rows if r["index"] not in (16, 17))
    assert v2_rows[16]["z_deg"] == [-20, 20] and v2_rows[13]["z_deg"] == [-10, 10]


def test_ranges_version_config():
    base = {"dataset_name": "x"}
    assert config_from_dict(base).ranges_version == RANGES_VERSION
    explicit = config_from_dict({**base, "ranges_version": RANGES_VERSION})
    assert "ranges_version" not in explicit.to_dict()
    assert explicit.to_dict() == config_from_dict(base).to_dict()
    assert config_from_dict({**base, "ranges_version": V2}).to_dict()["ranges_version"] == V2
    for bad in (
        {**base, "ranges_version": "nope"},
        {**base, "ranges_version": 2},
        {**base, "pose_source": "amass", "amass_subsets": ["A"], "ranges_version": V2},
    ):
        with pytest.raises(ConfigError):
            config_from_dict(bad)


def _synth_cfg(smpl: Path, tmp: Path, **extra) -> dict:
    return {
        "dataset_name": "synth", "output_root": str(tmp / "out"), "smpl_model_path": str(smpl),
        "num_poses": 3, "image": {"height": 64, "width": 48},
        "camera_ring": {"count": 8, "distance_m": 3.5, "height_m": 0.0, "focal_px": 80.0},
        **extra,
    }  # fmt: skip


def test_armsdown_dataset(synthetic_smpl_npz, tmp_path):
    out = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path,
                                                       ranges_version=V2)), seed=0)  # fmt: skip
    assert validate_dataset(out) == []
    m = load_manifest(out)
    assert m["config"]["ranges_version"] == V2
    src = m["inputs"]["pose_source"]
    assert src["ranges_version"] == V2 and src["ranges_deg"] == ranges_table(V2)
    for s in m["samples"]:
        lab = load_sample(out, s["id"])["labels"]
        assert lab["pose_source"]["ranges_version"] == V2
        assert -85 <= lab["pose_source"]["euler_deg"][16][2] < -45


def test_explicit_defaults_generate_identical_dataset(synthetic_smpl_npz, tmp_path):
    a = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path)), 0,
                         output_root=tmp_path / "a")  # fmt: skip
    explicit = _synth_cfg(synthetic_smpl_npz, tmp_path, ranges_version=RANGES_VERSION)
    b = generate_dataset(config_from_dict(explicit), 0, output_root=tmp_path / "b")
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()
    assert (a / "checksums.json").read_bytes() == (b / "checksums.json").read_bytes()
