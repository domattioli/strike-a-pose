"""Slice 010 dataset realism options (spec: specs/010-dataset-realism/spec.md)."""

import json
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.datagen.cameras import camera_ring, in_frame, jittered_camera, project
from strike_a_pose.datagen.config import (
    BodyShapeConfig,
    CameraJitterConfig,
    CameraRingConfig,
    ConfigError,
    config_from_dict,
    load_config,
)
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
from strike_a_pose.datagen.shape import sample_betas

from .conftest import TPOSE
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


# --- FR-004: random body shape -----------------------------------------------------------


def test_body_shape_config():
    base = {"dataset_name": "x"}
    assert config_from_dict(base).body_shape is None
    off = config_from_dict({**base, "body_shape": None})
    assert off.to_dict() == config_from_dict(base).to_dict()
    on = config_from_dict({**base, "body_shape": {}})
    assert on.to_dict()["body_shape"] == {"std": 1.0, "clip": 2.0}
    ints = config_from_dict({**base, "body_shape": {"std": 1, "clip": 3}}).to_dict()["body_shape"]
    assert ints == {"std": 1.0, "clip": 3.0} and all(isinstance(v, float) for v in ints.values())
    for bad in ({"std": 0}, {"clip": -1.0}, {"std": True}, {"std": "1"}, {"bogus": 1},
                {"std": float("nan")}, [1.0]):  # fmt: skip
        with pytest.raises(ConfigError):
            config_from_dict({**base, "body_shape": bad})
    with pytest.raises(ConfigError):
        config_from_dict({**base, "num_betas": 0, "body_shape": {}})


def test_sample_betas_stream():
    a = sample_betas(10, 1.0, 2.0, seed=0, pose_index=4)
    assert a.shape == (10,) and np.abs(a).max() <= 2.0
    assert np.array_equal(a, sample_betas(10, 1.0, 2.0, seed=0, pose_index=4))
    assert not np.array_equal(a, sample_betas(10, 1.0, 2.0, seed=0, pose_index=5))
    assert not np.array_equal(a, sample_betas(10, 1.0, 2.0, seed=1, pose_index=4))
    assert np.array_equal(sample_betas(4, 1.0, 2.0, 0, 4), a[:4])  # sequential per pose
    many = np.concatenate([sample_betas(10, 1.0, 0.5, 0, i) for i in range(50)])
    assert np.abs(many).max() == 0.5 and 0.5 < np.mean(np.abs(many) == 0.5) < 0.75  # P = 0.62


def _labels(out: Path) -> dict:
    return {s["id"]: load_sample(out, s["id"])["labels"] for s in load_manifest(out)["samples"]}


def test_body_shape_dataset_zero_shapedirs(synthetic_smpl_npz, tmp_path):
    """The zero-shapedirs model ignores betas: only the label `betas` and the config differ."""
    cfg = _synth_cfg(synthetic_smpl_npz, tmp_path)
    plain = generate_dataset(config_from_dict(cfg), 0, output_root=tmp_path / "a")
    shaped = generate_dataset(config_from_dict({**cfg, "body_shape": {"std": 1.0, "clip": 2.0}}),
                              0, output_root=tmp_path / "b")  # fmt: skip
    assert validate_dataset(shaped) == []
    assert load_manifest(shaped)["config"]["body_shape"] == {"std": 1.0, "clip": 2.0}
    assert (plain / "checksums.json").read_bytes() == (shaped / "checksums.json").read_bytes()
    la, lb = _labels(plain), _labels(shaped)
    by_pose: dict[int, list] = {}
    for sid, lab in lb.items():
        assert {k: v for k, v in lab.items() if k != "betas"} == {
            k: v for k, v in la[sid].items() if k != "betas"
        }
        assert len(lab["betas"]) == 10 and max(abs(b) for b in lab["betas"]) <= 2.0
        by_pose.setdefault(lab["pose_index"], []).append(lab["betas"])
    for views in by_pose.values():
        assert all(v == views[0] for v in views)  # shared by the pose's 8 views
    assert by_pose[0] != by_pose[1]


def test_body_shape_changes_bone_lengths(shaped_smpl_npz, tmp_path):
    out = generate_dataset(config_from_dict(_synth_cfg(shaped_smpl_npz, tmp_path,
                                                       body_shape={})), seed=0)  # fmt: skip
    for lab in _labels(out).values():
        j = np.array(lab["joints_3d_world"])
        scale = 1.0 + 0.1 * lab["betas"][0]  # shape component 0 scales the synthetic body
        for child, parent in ((18, 16), (4, 1), (12, 9)):  # upper arm, thigh, neck
            rest = np.linalg.norm(TPOSE[child] - TPOSE[parent])
            assert np.isclose(np.linalg.norm(j[child] - j[parent]), rest * scale, atol=2e-6)


def test_num_betas_beyond_model_refused_before_writing(synthetic_smpl_npz, tmp_path):
    cfg = config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path, num_betas=11, body_shape={}))
    with pytest.raises(ConfigError, match="shape components"):
        generate_dataset(cfg, seed=0)
    assert not (tmp_path / "out").exists()
    plain = config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path, num_betas=11))
    assert validate_dataset(generate_dataset(plain, seed=0)) == []  # zeros: no check, as today


# --- FR-005: camera ring jitter -----------------------------------------------------------

JITTER = {"height_m": [-0.3, 0.3], "distance_m": [0.0, 0.5], "pitch_deg": [0.0, 5.0],
          "margin_px": 2, "max_attempts": 16}  # fmt: skip


def test_camera_jitter_config():
    base = {"dataset_name": "x"}
    assert config_from_dict(base).camera_jitter is None
    off = config_from_dict({**base, "camera_jitter": None})
    assert off.to_dict() == config_from_dict(base).to_dict()
    on = config_from_dict({**base, "camera_jitter": {}}).to_dict()["camera_jitter"]
    assert on == {"height_m": [-0.3, 0.3], "distance_m": [0.0, 0.5], "pitch_deg": [0.0, 5.0],
                  "margin_px": 2, "max_attempts": 16}  # fmt: skip
    ints = config_from_dict({**base, "camera_jitter": {"pitch_deg": [0, 5]}}).to_dict()
    assert ints["camera_jitter"]["pitch_deg"] == [0.0, 5.0]
    assert isinstance(ints["camera_jitter"]["pitch_deg"][1], float)
    for bad in (
        {"height_m": [-1.5, 0.0]}, {"distance_m": [-0.1, 0.5]}, {"distance_m": [0.0, 3.5]},
        {"pitch_deg": [0.0, 31.0]}, {"pitch_deg": [5.0, 0.0]}, {"pitch_deg": [1.0]},
        {"pitch_deg": "0 5"}, {"height_m": [0, float("inf")]}, {"margin_px": -1},
        {"margin_px": 1.5}, {"max_attempts": 0}, {"max_attempts": True}, {"bogus": 1},
    ):  # fmt: skip
        with pytest.raises(ConfigError):
            config_from_dict({**base, "camera_jitter": bad})


def test_jittered_camera_geometry():
    ring_cfg = CameraRingConfig()
    for ring_cam in camera_ring(ring_cfg, 256, 192):
        zero = jittered_camera(ring_cam, ring_cfg, 0.0, 0.0, 0.0)
        assert np.array_equal(zero.R, ring_cam.R) and np.array_equal(zero.t, ring_cam.t)
        cam = jittered_camera(ring_cam, ring_cfg, 0.3, 0.5, 5.0)
        assert (cam.name, cam.yaw_deg, cam.width, cam.height) == (
            ring_cam.name, ring_cam.yaw_deg, ring_cam.width, ring_cam.height,
        )  # fmt: skip
        assert np.array_equal(cam.K, ring_cam.K)
        assert np.allclose(cam.R @ cam.R.T, np.eye(3)) and np.isclose(np.linalg.det(cam.R), 1)
        assert abs(cam.R[0, 1]) < 1e-12  # no roll: the image x axis stays horizontal
        th = np.deg2rad(ring_cam.yaw_deg)
        assert np.allclose(-cam.R.T @ cam.t, [-4.0 * np.sin(th), 0.3, 4.0 * np.cos(th)])
        _, uv = project(cam, np.zeros((1, 3)))  # the ring's look-at point
        assert np.allclose(uv[0], [96.0, 128.0 - 320.0 * np.tan(np.deg2rad(5.0))])
    raised = CameraRingConfig(height_m=0.2)
    for ring_cam in camera_ring(raised, 256, 192):
        zero = jittered_camera(ring_cam, raised, 0.0, 0.0, 0.0)
        assert np.array_equal(zero.R, ring_cam.R) and np.array_equal(zero.t, ring_cam.t)


def test_in_frame():
    cam = camera_ring(CameraRingConfig(), 256, 192)[0]  # Front_0: v = 128 - 320 y / 3.5

    def at_v(v):
        return np.array([[0.0, -(v - 128.0) * 3.5 / 320.0, 0.0]])

    assert in_frame(cam, np.zeros((1, 3)), 2)
    assert in_frame(cam, at_v(253.5), 2) and not in_frame(cam, at_v(254.5), 2)  # v < H - 2
    assert in_frame(cam, at_v(2.0), 2) and not in_frame(cam, at_v(1.5), 2)  # v >= 2
    assert in_frame(cam, at_v(255.5), 0) and not in_frame(cam, at_v(256.5), 0)
    assert not in_frame(cam, np.array([[1.2, 0.0, 0.0]]), 2)  # u = 206 > W
    assert not in_frame(cam, np.array([[0.0, 0.0, 4.0]]), 2)  # behind the camera


def _ring_cameras(out: Path) -> dict:
    return {c["name"]: c for c in load_manifest(out)["cameras"]}


def test_jitter_dataset(synthetic_smpl_npz, tmp_path):
    cfg = _synth_cfg(synthetic_smpl_npz, tmp_path)
    plain = generate_dataset(config_from_dict(cfg), 0, output_root=tmp_path / "a")
    out = generate_dataset(config_from_dict({**cfg, "camera_jitter": JITTER}), 0,
                           output_root=tmp_path / "b")  # fmt: skip
    again = generate_dataset(config_from_dict({**cfg, "camera_jitter": JITTER}), 0,
                             output_root=tmp_path / "c")  # fmt: skip
    assert validate_dataset(out) == []
    for f in ("manifest.json", "checksums.json"):
        assert (out / f).read_bytes() == (again / f).read_bytes()
    m = load_manifest(out)
    assert m["cameras"] == load_manifest(plain)["cameras"]
    assert m["config"]["camera_jitter"] == JITTER
    ring, moved = _ring_cameras(out), 0
    for sid, lab in _labels(out).items():
        assert set(lab) == LABEL_KEYS
        cam, j = lab["camera"], lab["camera"]["jitter"]
        assert set(cam) == set(ring[cam["name"]]) | {"jitter"}
        assert set(j) == {"height_m", "distance_m", "pitch_deg", "attempts", "fallback"}
        for k in ("name", "K", "width", "height", "yaw_deg"):
            assert cam[k] == ring[cam["name"]][k]
        assert -0.3 <= j["height_m"] <= 0.3 and 0.0 <= j["distance_m"] <= 0.5
        assert 0.0 <= j["pitch_deg"] <= 5.0 and 1 <= j["attempts"] <= 16
        if not j["fallback"]:
            moved += cam["R"] != ring[cam["name"]]["R"]
            mask = load_sample(out, sid)["mask"] > 0
            assert not mask[:2].any() and not mask[-2:].any()  # margin_px = 2
            assert not mask[:, :2].any() and not mask[:, -2:].any()
    assert moved > 20


def test_jitter_fallback_and_zero_ranges(synthetic_smpl_npz, tmp_path):
    cfg = _synth_cfg(synthetic_smpl_npz, tmp_path)
    plain = generate_dataset(config_from_dict(cfg), 0, output_root=tmp_path / "a")
    impossible = {"height_m": [0, 0], "distance_m": [0, 0], "pitch_deg": [29, 30],
                  "max_attempts": 3}  # fmt: skip
    zero = {"height_m": [0, 0], "distance_m": [0, 0], "pitch_deg": [0, 0], "margin_px": 0}
    for name, jit in (("f", impossible), ("z", zero)):
        out = generate_dataset(config_from_dict({**cfg, "camera_jitter": jit}), 0,
                               output_root=tmp_path / name)  # fmt: skip
        assert (out / "checksums.json").read_bytes() == (plain / "checksums.json").read_bytes()
        base = _labels(plain)
        for sid, lab in _labels(out).items():
            j = lab["camera"].pop("jitter")
            assert lab == base[sid]  # ring camera exactly; nothing else changes
            if name == "f":
                assert j == {"height_m": 0.0, "distance_m": 0.0, "pitch_deg": 0.0,
                             "attempts": 3, "fallback": True}  # fmt: skip
            else:
                assert j["attempts"] == 1 and j["fallback"] is False


def test_jitter_retries(synthetic_smpl_npz, tmp_path):
    jit = {"pitch_deg": [0.0, 25.0], "margin_px": 2, "max_attempts": 8}
    out = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path,
                                                       camera_jitter=jit)), seed=0)  # fmt: skip
    attempts = [lab["camera"]["jitter"]["attempts"] for lab in _labels(out).values()]
    assert max(attempts) > 1 and validate_dataset(out) == []


def test_jittered_dataset_loads_with_ring_indices(synthetic_smpl_npz, tmp_path):
    from strike_a_pose.data import PoseImageDataset

    out = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path,
                                                       camera_jitter=JITTER, body_shape={})),
                           seed=0)  # fmt: skip
    ds = PoseImageDataset(out)
    ring = list(_ring_cameras(out))
    for i in range(len(ds)):
        item = ds[i]
        lab = load_sample(out, item["sample_id"])["labels"]
        assert item["camera_index"] == ring.index(item["camera_name"])
        assert float(item["camera"]["yaw_deg"]) == 45.0 * item["camera_index"]
        assert np.allclose(item["camera"]["R"].numpy(), lab["camera"]["R"], atol=1e-6)


def test_all_options_prefix_property(synthetic_smpl_npz, tmp_path):
    opts = {"ranges_version": V2, "body_shape": {}, "camera_jitter": JITTER}
    small = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path,
                                                         num_poses=2, **opts)),
                             0, output_root=tmp_path / "s")  # fmt: skip
    large = generate_dataset(config_from_dict(_synth_cfg(synthetic_smpl_npz, tmp_path,
                                                         num_poses=3, **opts)),
                             0, output_root=tmp_path / "l")  # fmt: skip
    assert validate_dataset(large) == []
    for s in load_manifest(small)["samples"]:
        for k in ("image", "mask", "labels"):
            assert (small / s[k]).read_bytes() == (large / s[k]).read_bytes(), s[k]


# --- FR-010: the dataset series configs (surrogate ruling Q1: chain) ---------------------

SERIES = {"smpl_proc_v2": "ranges_version", "smpl_proc_v3": "body_shape",
          "smpl_proc_v4": "camera_jitter", "smpl_proc_v5": "num_poses"}  # fmt: skip


def test_series_generation_configs_form_a_chain():
    prev = load_config(ROOT / "configs" / "default.yaml").to_dict()
    for ds, key in SERIES.items():
        cur = load_config(ROOT / "configs" / ds / "generate.yaml").to_dict()
        assert cur["dataset_name"] == ds
        assert {k for k in prev.keys() | cur.keys() if prev.get(k) != cur.get(k)} == {
            "dataset_name", key,
        }  # fmt: skip
        prev = cur
    v5 = load_config(ROOT / "configs" / "smpl_proc_v5" / "generate.yaml")
    assert v5.num_poses == 4000 and v5.ranges_version == V2
    assert v5.body_shape == BodyShapeConfig() and v5.camera_jitter == CameraJitterConfig()


def test_series_run_configs_match_v1():
    from strike_a_pose.eval import config as baseline_config
    from strike_a_pose.eval import probe_config
    from strike_a_pose.train import config as train_config

    loaders = {"baselines": baseline_config.load_config, "ae": train_config.load_config,
               "probe": probe_config.load_config}  # fmt: skip
    for ds in SERIES:
        for kind, load in loaders.items():
            v1 = load(ROOT / "configs" / f"{kind}.yaml").to_dict()
            cur = load(ROOT / "configs" / ds / f"{kind}.yaml").to_dict()
            want = {"run_name", "dataset_dir", "split_file"} | ({"checkpoint"} if kind == "probe"
                                                                else set())  # fmt: skip
            if ds == "smpl_proc_v5":
                want |= {"baselines": {"image_pca_ks"}, "ae": {"context_pca_k"}}.get(kind, set())
            assert {k for k in v1 if v1[k] != cur[k]} == want and set(cur) == set(v1)
            assert cur["run_name"] == f"{kind}-{ds}"
            assert cur["dataset_dir"] == f"data/generated/{ds}"
            assert cur["split_file"] == f"splits/{ds}.json"
    v5 = ROOT / "configs" / "smpl_proc_v5"
    assert baseline_config.load_config(v5 / "baselines.yaml").image_pca_ks == ()
    assert train_config.load_config(v5 / "ae.yaml").context_pca_k is None


def test_series_configs_generate_on_synthetic_model(synthetic_smpl_npz, tmp_path):
    for ds in SERIES:
        d = load_config(ROOT / "configs" / ds / "generate.yaml").to_dict()
        d |= {"smpl_model_path": str(synthetic_smpl_npz), "num_poses": 2,
              "image": {"height": 64, "width": 48},
              "camera_ring": {**d["camera_ring"], "focal_px": 80.0}}  # fmt: skip
        out = generate_dataset(config_from_dict(d), 0, output_root=tmp_path)
        assert validate_dataset(out) == [], ds
        assert load_manifest(out)["config"] == d
