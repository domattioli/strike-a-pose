"""Limb-torso intersection count (specs 014 and 017 recalibration)."""

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
    min_inside_vertices,
    survey,
)
from strike_a_pose.datagen.procedural import (
    ARMSDOWN_VERSION,
    euler_xyz_to_matrix,
    matrix_to_axis_angle,
    sample_procedural_poses,
)
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
    rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)
    verts, _ = body.forward(pose, ZERO_BETAS)
    return count_arm_torso_intersections(rest, rest_j, verts, body.weights, margin)


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
    rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)
    verts, _ = body.forward(pose, ZERO_BETAS)
    moved = verts @ rodrigues(np.array([0.0, 0.0, 1.0])).T + np.array([3.0, -1.0, 2.0])
    assert count_arm_torso_intersections(rest, rest_j, moved, body.weights) == want


def test_margin_is_monotone(body):
    pose = _pose(j16=-135, j17=135)
    counts = [_count(body, pose, m) for m in (0.0, 0.005, 0.01, 0.02)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_degenerate_torso_is_refused(body):
    rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)
    with pytest.raises(ValueError, match="half-extent"):
        TorsoVolume(rest, rest_j, body.weights, margin_m=0.07)  # widened cube half-height 0.066
    w = body.weights.copy()
    w[w[:, 6] == 1.0] = np.eye(24)[12]  # spine2's vertices now belong to the neck
    with pytest.raises(ValueError, match="segment 6"):
        TorsoVolume(rest, rest_j, w)


def test_survey_counts_and_fraction(body):
    poses = [np.zeros(72), _pose(j16=-135), _pose(j16=-65), _pose(j17=135)]
    s = survey(body, poses, [ZERO_BETAS] * 4)
    assert s.counts[0] == s.counts[2] == 0 and s.counts[1] > 0 and s.counts[3] > 0
    assert s.intersecting == 2 and s.fraction == 0.5 and set(s.rest_counts) == {0}
    assert s.arm_vertices == 8 * 8 and s.torso_vertices == {0: 8, 3: 8, 6: 8, 9: 8}
    assert s.min_inside == 1
    with pytest.raises(ValueError):
        survey(body, poses, [ZERO_BETAS])


def test_min_inside_vertices():
    assert min_inside_vertices(64) == 1
    assert min_inside_vertices(2559) == 11
    assert min_inside_vertices(250) == 1
    assert min_inside_vertices(251) == 2
    assert min_inside_vertices(0) == 1


def test_spine3_is_capped_at_the_shoulders(body):
    rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)
    v = TorsoVolume(rest, rest_j, body.weights)
    assert v.spine3_half_x == pytest.approx(0.13)
    uncapped = TorsoVolume(rest, rest_j, body.weights, shoulder_clearance_m=0.0)
    assert uncapped.spine3_half_x == pytest.approx(0.14)
    with pytest.raises(ValueError, match="half-extent"):
        TorsoVolume(rest, rest_j, body.weights, shoulder_clearance_m=0.18)
    assert [s for s in v.segments if s.joint == 9][0].half[0] == v.spine3_half_x
    for a, b in zip(v.segments, uncapped.segments, strict=True):
        if a.joint != 9:
            assert np.array_equal(a.half, b.half)


def test_cap_only_removes_hits(body):
    rest, rest_j = body.forward(np.zeros(72), ZERO_BETAS)
    capped = TorsoVolume(rest, rest_j, body.weights)
    free = TorsoVolume(rest, rest_j, body.weights, shoulder_clearance_m=0.0)
    for p in sample_procedural_poses(400, 0):
        verts, _ = body.forward(np.array(p.pose), ZERO_BETAS)
        assert capped.count(verts) <= free.count(verts)


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
    assert r["check_version"] == "017" and r["shoulder_clearance_m"] == 0.05
    assert r["min_inside_vertices"] == 1 and r["spine3_half_x_m"] == 0.13
    assert len(r["worst_poses"]) == 5 and r["count_max"] >= r["count_median"] > 0
    assert r["torso_vertices"] == {"0": 8, "3": 8, "6": 8, "9": 8} and r["arm_vertices"] == 64
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide,
                     "--threshold", 0.0)  # fmt: skip
    assert code == 1 and "FAIL" in out and "/ 300 = " in out and "worst poses:" in out
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide,
                     "--num-poses", 20, "--threshold", 1.0)  # fmt: skip
    assert code == 0 and "/ 20 = " in out and "PASS" in out
    code, out = _run(capsys, "--config", v2a, "--seed", 0, "--smpl-model", wide,
                     "--shoulder-clearance-m", 0.0, "--json")  # fmt: skip
    assert json.loads(out)["spine3_half_x_m"] == 0.14
    assert sorted(tmp_path.rglob("*")) == before  # writes nothing


def test_script_poses_are_the_dataset_poses(synthetic_smpl_npz, tmp_path, capsys, monkeypatch):
    seen = {}

    def spy(body, poses, betas, margin_m, shoulder_clearance_m):
        seen["poses"], seen["betas"] = poses, betas
        return survey(body, poses, betas, margin_m, shoulder_clearance_m)

    monkeypatch.setattr(intersection_check, "survey", spy)
    cfg = _config(tmp_path, num_poses=4, body_shape={"std": 1.0, "clip": 2.0})
    assert _run(capsys, "--config", cfg, "--seed", 7, "--smpl-model", synthetic_smpl_npz)[0] == 0
    want = sample_procedural_poses(4, 7)
    assert [list(p) for p in seen["poses"]] == [list(p.pose) for p in want]
    from strike_a_pose.datagen.shape import sample_betas

    assert all(np.array_equal(b, sample_betas(10, 1.0, 2.0, 7, i))
               for i, b in enumerate(seen["betas"]))  # fmt: skip


def _arm_in_pelvis(path: Path) -> Path:
    z = dict(np.load(path))
    v, w = z["v_template"].copy(), z["weights"]
    v[w[:, 22] == 1.0] = v[w[:, 0] == 1.0].mean(axis=0)
    out = path.with_name(f"armin_{path.name}")
    np.savez(out, **z | {"v_template": v})
    return out


def test_script_refusals(synthetic_smpl_npz, tmp_path, capsys):
    amass = _config(tmp_path, pose_source="amass", amass_subsets=["A"])
    assert _run(capsys, "--config", amass, "--seed", 0, "--smpl-model", synthetic_smpl_npz)[0] == 2
    plain = _config(tmp_path)
    assert _run(capsys, "--config", plain, "--seed", 0)[0] == 3  # the config's model is missing
    huge = _arm_in_pelvis(synthetic_smpl_npz)  # the left hand starts inside the pelvis
    code, out = _run(capsys, "--config", plain, "--seed", 0, "--smpl-model", huge, "--num-poses", 3)
    assert code == 2 and "INVALID" in out
    code, _ = _run(capsys, "--config", plain, "--seed", 0, "--smpl-model", synthetic_smpl_npz,
                   "--margin-m", 0.05)  # fmt: skip
    assert code == 2


# --- real SMPL calibration (spec 017 FR-004, SC-001) ---------------------------------------


def _euler_pose(angles: dict[int, tuple[float, float, float]]) -> np.ndarray:
    pose = np.zeros(72)
    for j, deg in angles.items():
        m = euler_xyz_to_matrix(np.radians(np.array(deg, dtype=np.float64)))
        pose[3 * j : 3 * j + 3] = matrix_to_axis_angle(m)
    return pose


@pytest.mark.skipif(
    not (ROOT / "assets/smpl/SMPL_NEUTRAL.npz").exists(), reason="licensed SMPL asset absent"
)
def test_real_smpl_calibration():
    body = SMPLModel.load(ROOT / "assets/smpl/SMPL_NEUTRAL.npz")
    down = {16: (0, 0, -60), 17: (0, 0, 60)}
    safe = [
        {},
        down,
        {16: (0, -90, 0), 17: (0, 90, 0)},
        {**down, 18: (0, -100, 0), 19: (0, 100, 0)},
    ]
    for angles in safe:
        assert _count(body, _euler_pose(angles)) == 0, angles
    crossed = [
        ({16: (0, -90, 0), 18: (0, -150, 0)}, 500),
        ({16: (0, 0, -90), 18: (0, -30, 0)}, 100),
        ({16: (0, -80, 0), 17: (0, 80, 0), 18: (0, -150, 0), 19: (0, 150, 0)}, 1000),
    ]
    for angles, minimum in crossed:
        assert _count(body, _euler_pose(angles)) >= minimum, angles
    poses = [np.array(p.pose) for p in sample_procedural_poses(2000, 0)]
    s = survey(body, poses, [ZERO_BETAS] * len(poses))
    assert s.min_inside == 11 and s.arm_vertices == 2559
    assert abs(s.intersecting - 35) <= 2
    assert s.spine3_half_x == pytest.approx(0.1238, abs=5e-4)
