"""Slice 014 limb-torso intersection count (spec: specs/014-realism-ablations/spec.md)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from strike_a_pose.datagen.intersection import (
    MARGIN_M,
    TorsoVolume,
    count_arm_torso_intersections,
    kabsch,
    survey,
)
from strike_a_pose.datagen.procedural import ARMSDOWN_VERSION, sample_procedural_poses
from strike_a_pose.datagen.smpl import SMPLModel, rodrigues

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "intersection_check", ROOT / "scripts" / "intersection_check.py"
)
intersection_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(intersection_check)

ZERO_BETAS = np.zeros(10)


def _pose(**joint_z_deg: float) -> np.ndarray:
    """Axis-angle pose with z rotations (degrees) at named joint indices, e.g. j16=-135."""
    pose = np.zeros((24, 3))
    for name, deg in joint_z_deg.items():
        pose[int(name[1:]), 2] = np.radians(deg)
    return pose.reshape(-1)


def _widen_torso(path: Path, scale=(5.0, 2.2, 3.0)) -> Path:
    """Scale the synthetic torso cubes (joints 0, 3, 6, 9) about their centres: 0.30 m wide,
    0.13 m tall, 0.18 m deep each, closer to a human torso than the 6-cm cubes."""
    z = dict(np.load(path))
    v, w = z["v_template"].copy(), z["weights"]
    for j in (0, 3, 6, 9):
        sel = w[:, j] == 1.0
        c = v[sel].mean(axis=0)
        v[sel] = c + (v[sel] - c) * np.array(scale)
    z["v_template"] = v
    out = path.with_name(f"wide_{'_'.join(map(str, scale))}_{path.name}")
    np.savez(out, **z)
    return out


@pytest.fixture
def body(synthetic_smpl_npz) -> SMPLModel:
    """The synthetic body with a widened torso (the 6-cm torso cubes are hard to hit)."""
    return SMPLModel.load(_widen_torso(synthetic_smpl_npz))


def test_rest_pose_of_the_plain_synthetic_body_has_no_intersection(synthetic_smpl_npz):
    assert _count(SMPLModel.load(synthetic_smpl_npz), np.zeros(72)) == 0


def _count(body: SMPLModel, pose: np.ndarray, margin: float = MARGIN_M) -> int:
    rest, _ = body.forward(np.zeros(72), ZERO_BETAS)
    verts, _ = body.forward(pose, ZERO_BETAS)
    return count_arm_torso_intersections(rest, verts, body.weights, margin)


def test_kabsch_recovers_a_rigid_motion():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    r = rodrigues(np.array([0.3, -1.1, 0.7]))
    dst = src @ r.T + np.array([0.5, -2.0, 1.0])
    got, c_src, c_dst = kabsch(src, dst)
    assert np.allclose(got, r) and np.isclose(np.linalg.det(got), 1.0)
    assert np.allclose((src - c_src) @ got.T + c_dst, dst)


def test_rest_pose_has_no_intersection(body):
    assert _count(body, np.zeros(72)) == 0


def test_arm_driven_into_the_torso_is_counted(body):
    # left shoulder lowered 135 deg (negative z): the left elbow cube lands in the chest
    left = _count(body, _pose(j16=-135))
    right = _count(body, _pose(j17=135))  # the mirror image for the right arm
    assert left > 0 and right > 0
    assert _count(body, _pose(j16=-65, j17=65)) == 0  # arms down beside the torso: no contact
    assert _count(body, _pose(j16=-90, j17=90)) == 0  # straight down, 3 cm outside the chest
    both = _count(body, _pose(j16=-135, j17=135))
    assert both == left + right


def test_rigid_motion_of_the_whole_body_does_not_change_the_count(body):
    pose = _pose(j16=-135)
    want = _count(body, pose)
    turned = pose.copy()
    turned[:3] = [0.4, 2.0, -0.3]  # root orientation
    assert _count(body, turned) == want
    rest, _ = body.forward(np.zeros(72), ZERO_BETAS)
    verts, _ = body.forward(pose, ZERO_BETAS)
    moved = verts @ rodrigues(np.array([0.0, 0.0, 1.0])).T + np.array([3.0, -1.0, 2.0])
    assert count_arm_torso_intersections(rest, moved, body.weights) == want


def test_margin_is_monotone(body):
    pose = _pose(j16=-135, j17=135)
    counts = [_count(body, pose, m) for m in (0.0, 0.005, 0.01, 0.02)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_degenerate_torso_is_refused(body):
    rest, _ = body.forward(np.zeros(72), ZERO_BETAS)
    with pytest.raises(ValueError, match="half-extent"):
        TorsoVolume(rest, body.weights, margin_m=0.07)  # widened cube half-height 0.066
    w = body.weights.copy()
    w[w[:, 6] == 1.0] = np.eye(24)[12]  # spine2's vertices now belong to the neck
    with pytest.raises(ValueError, match="segment 6"):
        TorsoVolume(rest, w)


def test_survey_counts_and_fraction(body):
    poses = [np.zeros(72), _pose(j16=-135), _pose(j16=-65), _pose(j17=135)]
    s = survey(body, poses, [ZERO_BETAS] * 4)
    assert s.counts[0] == s.counts[2] == 0 and s.counts[1] > 0 and s.counts[3] > 0
    assert s.intersecting == 2 and s.fraction == 0.5 and set(s.rest_counts) == {0}
    assert s.arm_vertices == 8 * 8 and s.torso_vertices == {0: 8, 3: 8, 6: 8, 9: 8}
    with pytest.raises(ValueError):
        survey(body, poses, [ZERO_BETAS])


def test_armsdown_table_intersects_more_than_v1_on_a_wide_torso(synthetic_smpl_npz):
    wide = SMPLModel.load(_widen_torso(synthetic_smpl_npz))
    rates = {}
    for version in ("predecessor-v1-nohands", ARMSDOWN_VERSION):
        poses = [np.array(p.pose) for p in sample_procedural_poses(400, 0, version)]
        rates[version] = survey(wide, poses, [ZERO_BETAS] * len(poses)).fraction
    assert rates[ARMSDOWN_VERSION] > rates["predecessor-v1-nohands"]


# --- the script (spec 014 FR-006) --------------------------------------------------------


def _config(tmp_path: Path, **extra) -> Path:
    cfg = {"dataset_name": "synth", "pose_source": "procedural", "num_poses": 300,
           "smpl_model_path": str(tmp_path / "missing.npz"), **extra}  # fmt: skip
    p = tmp_path / f"gen_{len(list(tmp_path.glob('gen_*')))}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(capsys, *argv) -> tuple[int, str]:
    code = intersection_check.main([str(a) for a in argv])
    return code, capsys.readouterr().out


def test_script_reports_and_gates(synthetic_smpl_npz, tmp_path, capsys):
    wide = _widen_torso(synthetic_smpl_npz)
    v2a = _config(tmp_path, ranges_version=ARMSDOWN_VERSION)
    before = sorted(tmp_path.rglob("*"))
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide, "--json")
    r = json.loads(out)
    assert r["threshold"] == 0.05 and r["fraction"] == r["intersecting_poses"] / 300
    assert (code, r["result"]) == ((0, "PASS") if r["fraction"] <= 0.05 else (1, "FAIL"))
    assert r["ranges_version"] == ARMSDOWN_VERSION and r["poses"] == 300
    assert r["rest_poses_intersecting"] == 0 and r["intersecting_poses"] > 0
    assert len(r["worst_poses"]) == 5 and r["count_max"] >= r["count_median"] > 0
    assert r["torso_vertices"] == {"0": 8, "3": 8, "6": 8, "9": 8} and r["arm_vertices"] == 64
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide,
                     "--threshold", 0.0)  # fmt: skip
    assert code == 1 and "FAIL" in out and "/ 300 = " in out and "worst poses:" in out
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide,
                     "--num-poses", 20, "--threshold", 1.0)  # fmt: skip
    assert code == 0 and "/ 20 = " in out and "PASS" in out
    assert sorted(tmp_path.rglob("*")) == before  # writes nothing


def test_script_poses_are_the_dataset_poses(synthetic_smpl_npz, tmp_path, capsys, monkeypatch):
    seen = {}

    def spy(body, poses, betas, margin_m):
        seen["poses"], seen["betas"] = poses, betas
        return survey(body, poses, betas, margin_m)

    monkeypatch.setattr(intersection_check, "survey", spy)
    cfg = _config(tmp_path, num_poses=4, body_shape={"std": 1.0, "clip": 2.0})
    assert _run(capsys, "--config", cfg, "--seed", 7, "--smpl-model", synthetic_smpl_npz)[0] == 0
    want = sample_procedural_poses(4, 7)
    assert [list(p) for p in seen["poses"]] == [list(p.pose) for p in want]
    from strike_a_pose.datagen.shape import sample_betas

    assert all(np.array_equal(b, sample_betas(10, 1.0, 2.0, 7, i))
               for i, b in enumerate(seen["betas"]))  # fmt: skip


def test_script_refusals(synthetic_smpl_npz, tmp_path, capsys):
    amass = _config(tmp_path, pose_source="amass", amass_subsets=["A"])
    assert _run(capsys, "--config", amass, "--seed", 0, "--smpl-model", synthetic_smpl_npz)[0] == 2
    plain = _config(tmp_path)
    assert _run(capsys, "--config", plain, "--seed", 0)[0] == 3  # the config's model is missing
    huge = _widen_torso(synthetic_smpl_npz, scale=(9.0, 5.0, 3.0))  # swallows the shoulders
    code, out = _run(capsys, "--config", plain, "--seed", 0, "--smpl-model", huge, "--num-poses", 3)
    assert code == 2 and "INVALID" in out
    code, _ = _run(capsys, "--config", plain, "--seed", 0, "--smpl-model", synthetic_smpl_npz,
                   "--margin-m", 0.05)  # fmt: skip
    assert code == 2
