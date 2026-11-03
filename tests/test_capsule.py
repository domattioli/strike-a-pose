"""Spec 013: the interim capsule body (built in memory from the committed fixture)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from strike_a_pose.datagen import capsule as cap
from strike_a_pose.datagen.assets import MissingAssetError
from strike_a_pose.datagen.cameras import Camera, camera_ring, project
from strike_a_pose.datagen.cli import main
from strike_a_pose.datagen.config import CameraRingConfig, config_from_dict
from strike_a_pose.datagen.generate import generate_dataset
from strike_a_pose.datagen.io import load_manifest, validate_dataset
from strike_a_pose.datagen.procedural import (
    ARMSDOWN_VERSION,
    RANGES_VERSION,
    sample_procedural_poses,
)
from strike_a_pose.datagen.render import render
from strike_a_pose.datagen.smpl import SMPLModel, rodrigues

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"


@pytest.fixture(scope="module")
def body():
    rest, rec = cap.rest_skeleton(FIXTURE)
    arrays = cap.build_capsule_body(rest)
    return rest, rec, arrays, SMPLModel(arrays)


@pytest.fixture(scope="module")
def body_file(tmp_path_factory, body):
    path = tmp_path_factory.mktemp("capsule") / "capsule_v1.npz"
    path.write_bytes(cap.capsule_body_bytes(body[2]))
    return path


def _segment_ranges():
    """(start, stop) vertex ranges of the 28 capsules, in SEGMENTS order."""
    out, n = [], 0
    rest = cap.rest_skeleton(FIXTURE)[0]
    for _, sa, sb, r1, r2 in cap.SEGMENTS:
        v, _ = cap.capsule_mesh(cap._point(rest, sa), cap._point(rest, sb), r1, r2)
        out.append((n, n + len(v)))
        n += len(v)
    return out


def _fixture_labels():
    return [(p, json.loads(p.read_text())) for p in sorted((FIXTURE / "samples").glob("*.json"))]


def _cam(c):
    return Camera(c["name"], np.array(c["K"]), np.array(c["R"]), np.array(c["t"]),
                  c["width"], c["height"], c["yaw_deg"])  # fmt: skip


# (a) skeleton -------------------------------------------------------------------------------


def test_rest_skeleton_from_fixture(body):
    rest, rec, _, _ = body
    assert rec.shape == (2, 24, 3) and np.abs(rec[0] - rec[1]).max() < 5e-6
    assert np.abs(rec - rest).max() < 5e-6 and np.array_equal(rest, np.round(rest, 6))
    assert np.allclose(rest[0], [-0.0018, -0.2233, 0.0282], atol=1e-3)  # pelvis
    assert np.allclose(rest[[7, 8], 1], -1.09, atol=0.01)  # ankles
    assert abs(rest[15, 1] - 0.353) < 1e-3  # head joint (not the crown)
    assert rest[16, 0] > 0.15 and rest[17, 0] < -0.15  # left shoulder at +x


def test_rest_skeleton_missing_or_inconsistent_fixture(tmp_path):
    with pytest.raises(MissingAssetError):
        cap.rest_skeleton(tmp_path)
    (tmp_path / "samples").mkdir()
    labs = sorted((FIXTURE / "samples").glob("00000_*.json"))[:2]
    for i, p in enumerate(labs):
        lab = json.loads(p.read_text())
        lab["joints_3d_world"][3][0] += 0.01 * i
        (tmp_path / "samples" / p.name).write_text(json.dumps(lab))
    with pytest.raises(ValueError, match="differ between views"):
        cap.rest_skeleton(tmp_path)


# (b) arrays, (c) zero pose ------------------------------------------------------------------


def test_arrays(body):
    rest, _, a, m = body
    v = len(a["v_template"])
    assert (v, len(a["f"])) == (5952, 11744) and len(cap.SEGMENTS) == 28
    assert a["shapedirs"].shape == (v, 3, 10) and a["posedirs"].shape == (v, 3, 207)
    assert a["posedirs"].dtype == np.float32 and not a["posedirs"].any()
    assert a["f"].dtype == np.int64 and a["f"].min() == 0 and a["f"].max() < v
    markers = np.arange(v - 24, v)
    assert not np.isin(markers, a["f"]).any()
    assert np.array_equal(a["J_regressor"][:, markers], np.eye(24))
    assert a["J_regressor"].sum() == 24
    w = a["weights"]
    assert np.array_equal(w.sum(1), np.ones(v)) and set(np.unique(w)) == {0.0, 1.0}
    assert np.array_equal(a["kintree_table"][1], np.arange(24))
    assert a["kintree_table"][0, 0] == 2**32 - 1
    assert list(m.parents) == [-1, *a["kintree_table"][0, 1:]]
    assert m.num_shape_components == 10
    verts, joints = m.forward(np.zeros(72), np.zeros(10))
    assert np.allclose(verts, a["v_template"]) and np.allclose(joints, rest)
    assert a["v_template"][:, 1].min() == pytest.approx(-1.20, abs=0.01)  # soles
    assert a["v_template"][:, 1].max() == pytest.approx(0.53, abs=0.01)  # crown


# (d) kinematics ------------------------------------------------------------------------------


def test_fixture_poses_reproduce_label_joints(body):
    m = body[3]
    for _, lab in _fixture_labels()[::8]:
        _, joints = m.forward(np.array(lab["pose"]), np.zeros(10))
        assert np.abs(joints - np.array(lab["joints_3d_world"])).max() < 1e-5


def test_capsules_move_rigidly_with_their_joint(body):
    rest, _, a, m = body
    pose = np.random.default_rng(0).normal(0, 0.5, 72)
    verts, joints = m.forward(pose, np.zeros(10))
    rots = rodrigues(pose.reshape(24, 3))
    g = [None] * 24
    for i, p in enumerate(m.parents):
        g[i] = rots[i] if p < 0 else g[p] @ rots[i]
    for (lo, hi), (skin, sa, sb, *_r) in zip(_segment_ranges(), cap.SEGMENTS, strict=True):
        moved = joints[skin] + (a["v_template"][lo:hi] - rest[skin]) @ g[skin].T
        assert np.allclose(verts[lo:hi], moved, atol=1e-9)
        for joint, offset in (sa, sb):
            if not any(offset):  # cap centred on a joint lands on the posed joint
                centre = joints[skin] + g[skin] @ (rest[joint] - rest[skin])
                assert np.allclose(centre, joints[joint], atol=1e-9)


# (e) mesh -------------------------------------------------------------------------------------


def test_capsules_are_closed_and_consistently_wound(body):
    a = body[2]
    f = a["f"]
    for lo, hi in _segment_ranges():
        mine = f[(f >= lo).all(1) & (f < hi).all(1)]
        pairs = (zip(t, np.roll(t, -1), strict=True) for t in mine)
        directed = Counter((int(x), int(y)) for pr in pairs for x, y in pr)
        assert all(n == 1 for n in directed.values())  # consistent winding
        assert all((y, x) in directed for x, y in directed)  # every edge in exactly two faces
        assert len(np.unique(mine)) - len(directed) // 2 + len(mine) == 2  # Euler: sphere
    tri = a["v_template"][f]
    edges = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
    assert edges.max() <= 0.08  # render-cost bound (long faces make the rasterizer slow)


# (f) shape ------------------------------------------------------------------------------------


def test_shape_components(body):
    rest, _, a, m = body
    sd = a["shapedirs"]
    assert np.allclose(sd[:, :, 0], 0.04 * a["v_template"], atol=1e-6)
    assert not sd[:, :, 3:].any()
    _, j0 = m.forward(np.zeros(72), np.zeros(10))
    for k in (1, 2):
        b = np.zeros(10)
        b[k] = 2.0
        v, j = m.forward(np.zeros(72), b)
        assert np.allclose(j, j0) and not np.allclose(v, a["v_template"])
    b = np.zeros(10)
    b[0] = 2.0
    _, j = m.forward(np.zeros(72), b)
    assert np.allclose(j, 1.08 * rest, atol=1e-5)
    ranges = _segment_ranges()
    thigh = next(r for r, s in zip(ranges, cap.SEGMENTS, strict=True) if s[0] == 1)
    head = next(r for r, s in zip(ranges, cap.SEGMENTS, strict=True) if s[0] == 15)
    spine = next(r for r, s in zip(ranges, cap.SEGMENTS, strict=True) if s[0] == 3)
    assert sd[slice(*thigh), :, 1].any() and not sd[slice(*head), :, 1].any()
    assert sd[slice(*spine), :, 2].any() and not sd[slice(*thigh), :, 2].any()


# (g) determinism and CLI ----------------------------------------------------------------------


def test_bytes_deterministic_and_pinned(body, body_file):
    data, digest = cap.build_capsule_bytes(FIXTURE)
    assert data == body_file.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest() == cap.CAPSULE_V1_SHA256
    m = SMPLModel.load(body_file)
    assert np.array_equal(m.faces, body[3].faces)


def test_cli_make_capsule_body(tmp_path, capsys, monkeypatch):
    out = tmp_path / "a" / "capsule_v1.npz"
    args = ["make-capsule-body", "--out", str(out), "--fixture", str(FIXTURE)]
    assert main(args) == 0 and cap.CAPSULE_V1_SHA256 in capsys.readouterr().out
    assert hashlib.sha256(out.read_bytes()).hexdigest() == cap.CAPSULE_V1_SHA256
    assert main(args) == 0 and "unchanged" in capsys.readouterr().out
    out.write_bytes(b"other")
    assert main(args) == 4 and out.read_bytes() == b"other"
    assert main([*args, "--overwrite"]) == 0
    assert hashlib.sha256(out.read_bytes()).hexdigest() == cap.CAPSULE_V1_SHA256
    assert main([*args[:-1], str(tmp_path / "nope")]) == 3
    monkeypatch.setattr(cap, "CAPSULE_V1_SHA256", "0" * 64)
    other = tmp_path / "b.npz"
    assert main(["make-capsule-body", "--out", str(other), "--fixture", str(FIXTURE)]) == 1
    assert not other.exists() and "differs from the pinned" in capsys.readouterr().err


# (h) render sanity at full size ----------------------------------------------------------------


def test_fixture_views_match_smpl_silhouettes(body):
    m = body[3]
    for p, lab in _fixture_labels():
        cam = _cam(lab["camera"])
        verts, _ = m.forward(np.array(lab["pose"]), np.zeros(10))
        mask = render(verts, m.faces, cam)[1] > 0
        ref = np.array(Image.open(p.with_name(p.stem + "_mask.png"))) > 0
        rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
        assert rows.min() >= 2 and rows.max() <= cam.height - 3, p.name
        assert cols.min() >= 2 and cols.max() <= cam.width - 3, p.name
        kp = np.floor(np.array(lab["keypoints_2d"])).astype(int)
        assert mask[kp[:, 1], kp[:, 0]].all(), p.name  # every keypoint inside the silhouette
        assert (mask & ref).sum() / (mask | ref).sum() >= 0.70, p.name
        assert 0.85 <= mask.sum() / ref.sum() <= 1.15, p.name


def test_procedural_poses_render_in_frame(body):
    m = body[3]
    cams = camera_ring(CameraRingConfig(), 256, 192)
    for version in (RANGES_VERSION, ARMSDOWN_VERSION):
        for rec in sample_procedural_poses(2, 0, version):
            verts, joints = m.forward(np.array(rec.pose), np.zeros(10))
            for cam in cams:
                assert render(verts, m.faces, cam)[1].any()
                _, uv = project(cam, joints)
                assert ((uv >= 0) & (uv < [cam.width, cam.height])).all()


# (i) end to end through the unchanged generator ------------------------------------------------


@pytest.mark.parametrize("shape", [None, {"std": 1.0, "clip": 2.0}])
def test_generate_with_capsule_body(body_file, tmp_path, shape):
    cfg = config_from_dict({
        "dataset_name": "capsule_small", "smpl_model_path": str(body_file), "num_poses": 2,
        "image": {"height": 64, "width": 48}, "body_shape": shape,
        "camera_ring": {"count": 8, "distance_m": 3.5, "height_m": 0.0, "focal_px": 80.0},
    })  # fmt: skip
    out = generate_dataset(cfg, 0, output_root=tmp_path)
    assert validate_dataset(out) == []  # includes the SC-003 projection invariant
    man = load_manifest(out)
    assert man["inputs"]["smpl_model"] == {
        "filename": "capsule_v1.npz", "sha256": cap.CAPSULE_V1_SHA256,
    }  # fmt: skip
    for s in man["samples"]:
        assert np.array(Image.open(out / s["mask"])).any()


# (j) capsule twin configs (spec 013 FR-008) ------------------------------------------------

CAPSULE_V1_TWINS = {"generate": "default", "baselines": "baselines", "ae": "ae",
                    "ae_seed1": "ae_seed1", "ae_mask": "ae_mask", "ae_vae": "ae_vae",
                    "ae_vae_beta0.1": "ae_vae_beta0.1", "ae_vae_beta0.001": "ae_vae_beta0.001",
                    "ae_reg": "ae_reg", "probe": "probe"}  # fmt: skip


def _twins():
    for name, src in CAPSULE_V1_TWINS.items():
        yield ROOT / f"configs/{src}.yaml", ROOT / f"configs/capsule_proc_v1/{name}.yaml"
    for d in "abcd":
        for name in ("generate", "baselines", "ae", "probe"):
            yield (ROOT / f"configs/smpl_proc_v2{d}/{name}.yaml",
                   ROOT / f"configs/capsule_proc_v2{d}/{name}.yaml")  # fmt: skip


def _loader(path: Path):
    from strike_a_pose.datagen.config import load_config as gen
    from strike_a_pose.eval import config as baseline_config
    from strike_a_pose.eval import probe_config
    from strike_a_pose.train import config as train_config

    stem = path.stem
    if stem in ("generate", "default"):
        return gen
    if stem == "baselines":
        return baseline_config.load_config
    if stem == "probe":
        return probe_config.load_config
    return train_config.load_config


def test_capsule_configs_are_twins_of_the_smpl_configs():
    twins = list(_twins())
    on_disk = sorted((ROOT / "configs").glob("capsule_proc_*/*.yaml"))
    assert sorted(dst for _, dst in twins) == on_disk and len(on_disk) == 26
    for src, dst in twins:
        load = _loader(dst)
        smpl, capsule = load(src).to_dict(), load(dst).to_dict()
        assert set(smpl) == set(capsule), dst
        for key in smpl:
            want = smpl[key]
            if key == "smpl_model_path":
                want = "assets/synthetic/capsule_v1.npz"
            elif isinstance(want, str) and "smpl_proc_" in want:
                want = want.replace("smpl_proc_", "capsule_proc_")
            assert capsule[key] == want, (dst, key)
        text = dst.read_text()
        assert text.startswith(f"# Capsule twin of {src.relative_to(ROOT)} (spec 013)")
        body = text.split("\n", 1)[1]  # the header line names the SMPL source
        assert "smpl_proc_" not in body and "SMPL_NEUTRAL" not in body


def test_capsule_generate_configs_generate_on_capsule_body(body_file, tmp_path):
    from strike_a_pose.datagen.config import load_config

    for d in ("v1", "v2a", "v2b", "v2c", "v2d"):
        cfg = load_config(ROOT / f"configs/capsule_proc_{d}/generate.yaml").to_dict()
        assert cfg["dataset_name"] == f"capsule_proc_{d}"
        cfg |= {"smpl_model_path": str(body_file), "num_poses": 2,
                "image": {"height": 64, "width": 48},
                "camera_ring": {**cfg["camera_ring"], "focal_px": 80.0}}  # fmt: skip
        out = generate_dataset(config_from_dict(cfg), 0, output_root=tmp_path)
        assert validate_dataset(out) == [], d
        assert load_manifest(out)["config"] == cfg
