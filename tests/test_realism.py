"""Slice 010 dataset realism options (spec: specs/010-dataset-realism/spec.md)."""

import json
from pathlib import Path

from strike_a_pose.datagen.config import config_from_dict, load_config
from strike_a_pose.datagen.generate import generate_dataset
from strike_a_pose.datagen.io import LABEL_KEYS, dumps, load_manifest
from strike_a_pose.datagen.procedural import sample_procedural_poses

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
