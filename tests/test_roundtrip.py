"""Probe round-trip plausibility of VAE samples (spec 016)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from strike_a_pose.datagen.cameras import camera_ring
from strike_a_pose.datagen.config import CameraRingConfig
from strike_a_pose.datagen.render import render
from strike_a_pose.datagen.smpl import SMPLModel
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.roundtrip import (
    HIST_HEADER,
    HIST_ROW_H,
    HIST_W,
    calibrate_threshold,
    camera_from_dict,
    check_renderable,
    iou_histogram_figure,
    rerender_masks,
    silhouette_iou,
    summarize,
)
from strike_a_pose.eval.roundtrip_config import config_from_dict, load_config

ROOT = Path(__file__).parents[1]
FIXTURE_CFG = "configs/roundtrip_fixture.yaml"
MANIFEST = ROOT / "tests/fixtures/smpl_fixture/manifest.json"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_config():
    cfg = load_config(FIXTURE_CFG)
    assert cfg.calibration_percentile == 5.0 and cfg.control is True
    assert load_config(FIXTURE_CFG).seed == 0
    assert config_from_dict({"sample_run": "a", "probe_run": "b"}).figure_samples == 8
    base = {"sample_run": "a", "probe_run": "b"}
    bads = [
        {"bogus": 1}, {"calibration_percentile": 0}, {"calibration_percentile": 50.5},
        {"calibration_percentile": True}, {"calibration_percentile": "5"},
        {"figure_samples": 0}, {"figure_samples": 33}, {"seed": -1}, {"control": 1},
        {"run_name": "bad name"},
    ]  # fmt: skip
    for bad in bads:
        with pytest.raises(ConfigError):
            config_from_dict(base | bad)
    for missing in ("sample_run", "probe_run"):
        with pytest.raises(ConfigError):
            config_from_dict({k: v for k, v in base.items() if k != missing})
    assert config_from_dict(base | {"calibration_percentile": 5}).calibration_percentile == 5.0
    cfg = load_config(FIXTURE_CFG)
    new = cfg.with_overrides(sample_run="x", probe_run=None)
    assert new.sample_run == "x" and new.probe_run == cfg.probe_run
    assert new.with_overrides(sample_run=cfg.sample_run) == cfg


def test_silhouette_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[0, :4] = True
    b[0, 2:4] = True
    b[1, 0:4] = True
    assert a.sum() == 4 and b.sum() == 6 and (a & b).sum() == 2
    assert silhouette_iou(a, b) == 2 / 8
    assert silhouette_iou(a, a) == 1.0
    c = np.zeros((4, 4), bool)
    c[3] = True
    assert silhouette_iou(a, c) == 0.0
    assert silhouette_iou(c * False, c * False) == 0.0
    assert silhouette_iou(a, np.zeros((4, 4), bool)) == 0.0
    with pytest.raises(ValueError):
        silhouette_iou(a, np.zeros((4, 5), bool))


def test_calibrate_threshold():
    assert calibrate_threshold(np.array([0.9, 0.1, 0.5, 0.7]), 5.0) == 0.1
    assert calibrate_threshold(np.arange(100) / 100, 5.0) == 0.04
    x = np.random.default_rng(0).random(1000)
    for p in (1, 5, 10):
        tau = calibrate_threshold(x, p)
        assert (x >= tau).mean() >= 1 - p / 100
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([]), 5.0)


def test_camera_from_dict_matches_ring():
    m = _manifest()
    ring = camera_ring(CameraRingConfig(**m["config"]["camera_ring"]), 256, 192)
    assert len(ring) == len(m["cameras"])
    for k, want in enumerate(ring):
        got = camera_from_dict(m["cameras"][k])
        assert (got.name, got.width, got.height, got.yaw_deg) == (
            want.name, want.width, want.height, want.yaw_deg,
        )  # fmt: skip
        for f in ("K", "R", "t"):
            assert getattr(got, f).dtype == np.float64
            # the manifest stores cameras rounded to 6 significant digits, so not bit-equal
            assert np.allclose(getattr(got, f), getattr(want, f), rtol=0, atol=1e-5)


def test_check_renderable():
    m = _manifest()
    check_renderable(m)
    for key in ("camera_jitter", "body_shape"):
        bad = copy.deepcopy(m)
        bad["config"][key] = {"x": 1}
        with pytest.raises(ConfigError, match=key):
            check_renderable(bad)
        ok = copy.deepcopy(m)
        ok["config"][key] = None
        check_renderable(ok)


def test_rerender_identity(synthetic_smpl_npz):
    body = SMPLModel.load(synthetic_smpl_npz)
    rng = np.random.default_rng(0)
    p1 = 0.3 * rng.standard_normal(72)
    p1[[0, 22, 23]] = 0.0
    poses = np.stack([np.zeros(72), p1[:72].reshape(72)])
    poses[1].reshape(24, 3)[[0, 22, 23]] = 0.0
    cams = [camera_from_dict(c) for c in _manifest()["cameras"]]
    cam_idx = np.array([0, 3])
    out = rerender_masks(body, poses, cams, cam_idx, num_betas=10)
    assert out.dtype == bool and out.shape == (2, 256, 192)
    for i in range(2):
        verts = body.forward(poses[i], np.zeros(10))[0]
        want = render(verts, body.faces, cams[cam_idx[i]])[1] > 0
        assert np.array_equal(out[i], want)
        assert silhouette_iou(out[i], out[i]) == 1.0
    assert rerender_masks(body, poses[:0], cams, cam_idx[:0], 10).shape == (0, 256, 192)


def test_summarize():
    ious = np.array([0.0, 0.2, 0.6, 0.8])
    valid = np.array([False, True, True, False])
    yaw = np.array([0, 0, 3, 7])
    ctrl = np.array([0.1, 0.1, 0.7, 0.1])
    r = summarize(ious, valid, yaw, 1, 0.5, 8, ctrl)
    assert r["n"] == 4 and r["iou_mean"] == pytest.approx(0.4)
    assert r["iou_quantiles"]["p50"] == np.percentile(ious, 50)
    assert set(r["iou_quantiles"]) == {"p5", "p25", "p50", "p75", "p95"}
    assert r["plausible_fraction"] == 0.5 and r["plausible_and_valid_fraction"] == 0.25
    assert r["empty_silhouette"] == 1 and r["yaw_counts"] == [2, 0, 0, 1, 0, 0, 0, 1]
    assert r["control_iou_median"] == pytest.approx(0.1)
    assert r["control_plausible_fraction"] == 0.25
    r = summarize(ious, valid, yaw, 1, 0.5, 8)
    assert r["control_iou_median"] is None and r["control_plausible_fraction"] is None
    e = summarize(np.array([]), np.array([], bool), np.array([], int), 0, 0.5, 8)
    assert e["n"] == 0 and e["iou_mean"] is None and e["plausible_fraction"] is None
    assert e["plausible_and_valid_fraction"] is None
    assert all(v is None for v in e["iou_quantiles"].values())
    assert e["yaw_counts"] == [0] * 8


def test_iou_histogram_figure():
    d = {"a": np.array([0.1, 0.9]), "b": np.array([])}
    im = iou_histogram_figure(d, 0.5)
    assert im.mode == "L" and im.size == (HIST_W, HIST_HEADER + 2 * HIST_ROW_H)
    assert im.tobytes() == iou_histogram_figure(d, 0.5).tobytes()
