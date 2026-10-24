"""Probe round-trip plausibility of VAE samples (spec 016)."""

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.datagen.assets import sha256_file
from strike_a_pose.datagen.cameras import camera_ring
from strike_a_pose.datagen.config import CameraRingConfig
from strike_a_pose.datagen.render import render
from strike_a_pose.datagen.smpl import SMPLModel
from strike_a_pose.eval import probe_config, sample_config
from strike_a_pose.eval.cli import EXIT_CONFIG, main
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.probe import _encoders, load_or_extract, run_probe, score
from strike_a_pose.eval.roundtrip import (
    HIST_HEADER,
    HIST_ROW_H,
    HIST_W,
    SET_ORDER,
    calibrate_threshold,
    camera_from_dict,
    check_renderable,
    iou_histogram_figure,
    refit_probe,
    rerender_masks,
    run_round_trip,
    silhouette_iou,
    summarize,
)
from strike_a_pose.eval.roundtrip_config import config_from_dict, load_config
from strike_a_pose.eval.sample import run_sample
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
from strike_a_pose.runlog import round_sig
from tests.conftest import make_synthetic_smpl

ROOT = Path(__file__).parents[1]
FIXTURE_CFG = "configs/roundtrip_fixture.yaml"
CH = [16, 32, 64, 128, 256, 256]
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


# --------------------------------------------------------------------------- B2: run


def _ckpt(path: Path, variational: bool, seed: int, latent_dim: int = 64) -> Path:
    torch.manual_seed(seed)
    m = ConvAutoencoder((256, 192), latent_dim, CH, variational=variational)
    torch.save({"model_hparams": m.hparams(), "model": m.state_dict()}, path)
    return path


@pytest.fixture(scope="module")
def rt_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rt")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        vae = _ckpt(root / "vae.pt", True, 0)
        ae = _ckpt(root / "ae.pt", False, 1)
        scfg = sample_config.load_config("configs/sample_fixture.yaml").with_overrides(
            checkpoint=str(vae), reference_checkpoint=str(ae), run_root=str(root / "s")
        )
        sample_dir = run_sample(scfg)
        pcfg = probe_config.load_config("configs/probe_fixture.yaml").with_overrides(
            checkpoint=str(vae), run_root=str(root / "p"), cache_dir=str(root / "cache")
        )
        probe_dir = run_probe(pcfg)
        smpl = make_synthetic_smpl(root / "SMPL_SYNTH.npz")
    finally:
        os.chdir(cwd)
    return {"root": root, "vae": vae, "sample": sample_dir, "probe": probe_dir, "smpl": smpl,
            "pcfg": pcfg}  # fmt: skip


def _cfg(inp, run_root, **kw):
    return load_config(FIXTURE_CFG).with_overrides(
        sample_run=str(inp["sample"]), probe_run=str(inp["probe"]),
        smpl_model_path=str(inp["smpl"]), run_root=str(run_root), **kw,
    )  # fmt: skip


def test_fixture_round_trip_run(rt_inputs, tmp_path):
    t0 = time.monotonic()
    d = run_round_trip(_cfg(rt_inputs, tmp_path / "a"))
    assert time.monotonic() - t0 < 60
    assert {p.name for p in d.iterdir()} == {
        "config.yaml", "run_info.json", "samples.json", "round_trip_per_image.json",
        "round_trip.png", "round_trip_iou.png",
    }  # fmt: skip
    r = json.loads((d / "samples.json").read_text())
    src = json.loads((rt_inputs["sample"] / "samples.json").read_text())
    assert set(r["sets"]) == set(SET_ORDER)  # write_json_exact sorts keys
    for name, want in (("vae_prior", 8), ("ae_prior", 8), ("vae_recon_val", 4), ("real_test", 4)):
        assert r["sets"][name]["round_trip"]["n"] == want
        assert r["sets"][name]["n"] == want
    assert r["sets"]["real_test"]["round_trip"]["plausible_fraction"] >= 0.95
    for name, s in src["sets"].items():
        assert {k: v for k, v in r["sets"][name].items() if k != "round_trip"} == s
    per = json.loads((d / "round_trip_per_image.json").read_text())
    crit = r["round_trip_criteria"]
    assert crit["calibration"]["set"] == "real_test" and crit["calibration"]["split"] == "test"
    assert crit["calibration"]["tau"] == round_sig(
        calibrate_threshold(np.array(per["sets"]["real_test"]["iou"]), 5.0)
    )
    assert crit["smpl_model"]["matches_manifest"] is False
    assert crit["probe"]["refit"]["split"] == "test"
    assert crit["probe"]["refit"]["matches"] is True
    assert r["round_trip_version"] == 1 and "source_sample_run" in r
    info = json.loads((d / "run_info.json").read_text())
    assert info["peak_rss_mb"] > 0 and "wall_s" in info


def test_round_trip_deterministic(rt_inputs, tmp_path):
    a = run_round_trip(_cfg(rt_inputs, tmp_path / "a"))
    b = run_round_trip(_cfg(rt_inputs, tmp_path / "b"))
    for f in ("samples.json", "round_trip_per_image.json"):
        text = (a / f).read_text()
        if f == "samples.json":  # the run dir name is not in it, the source path is the same
            pass
        assert text == (b / f).read_text()


def test_round_trip_refusals(rt_inputs, tmp_path):
    def copy(src, name):
        dst = tmp_path / name
        shutil.copytree(src, dst)
        return dst

    def refuses(inp, match=None):
        root = tmp_path / "out"
        with pytest.raises(ConfigError, match=match):
            run_round_trip(_cfg(inp, root))
        assert not root.exists() or not any(root.iterdir())

    inp = dict(rt_inputs)
    other = copy(rt_inputs["probe"], "p_other")  # (a)
    m = json.loads((other / "metrics.json").read_text())
    m["encoder"]["checkpoint_sha256"] = "0" * 64
    (other / "metrics.json").write_text(json.dumps(m))
    refuses(inp | {"probe": other})
    tam = copy(rt_inputs["sample"], "s_tamper")  # (b)
    s = json.loads((tam / "samples.json").read_text())
    s["sets"]["vae_prior"]["foreground_px_mean"] += 1.0
    (tam / "samples.json").write_text(json.dumps(s))
    refuses(inp | {"sample": tam}, "not reproduced")
    refuses(inp | {"smpl": tmp_path / "missing.npz"})  # (c)
    nop = copy(rt_inputs["probe"], "p_notarget")  # (d)
    c = yaml.safe_load((nop / "config.yaml").read_text())
    c["targets"] = ["joints_world", "yaw"]
    (nop / "config.yaml").write_text(yaml.safe_dump(c))
    refuses(inp | {"probe": nop})
    nos = copy(rt_inputs["sample"], "s_nosamples")  # (e)
    (nos / "samples.json").unlink()
    refuses(inp | {"sample": nos})


def test_round_trip_cli(rt_inputs, tmp_path, capsys):
    argv = [
        "round-trip", "--config", FIXTURE_CFG, "--sample-run", str(rt_inputs["sample"]),
        "--probe-run", str(rt_inputs["probe"]), "--smpl-model", str(rt_inputs["smpl"]),
        "--run-root", str(tmp_path),
    ]  # fmt: skip
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(ln.startswith("wrote ") for ln in lines)
    sets = [ln for ln in lines if ln.startswith("  ") and "plausible" in ln and "valid" in ln]
    assert [ln.split(":")[0].strip() for ln in sets] == list(SET_ORDER)
    argv[argv.index("--smpl-model") + 1] = str(tmp_path / "nope.npz")
    assert main(argv) == EXIT_CONFIG


def test_refit_matches_probe(rt_inputs):
    pcfg = rt_inputs["pcfg"]
    metrics = json.loads((rt_inputs["probe"] / "metrics.json").read_text())
    model, _ = load_checkpoint(rt_inputs["vae"])
    kw = {"split_file": pcfg.split_file}
    ds = {
        s: PoseImageDataset(pcfg.dataset_dir, split=s, **kw) for s in ("train", *pcfg.eval_splits)
    }
    hashes = {
        "checkpoint_sha256": sha256_file(rt_inputs["vae"]),
        "manifest_sha256": ds["train"].manifest_sha256,
        "split_sha256": sha256_file(Path(pcfg.split_file)),
    }
    os.chdir(ROOT)
    data, _, _ = load_or_extract(pcfg, ds, _encoders(model, pcfg), hashes)
    pose_fn, yaw_fn, zs, _ = refit_probe(pcfg, metrics, data)
    a = data["test"]
    z = zs(a["latents"]["trained"])
    rec = metrics["splits"]["test"]["targets"]
    got = score("smpl_pose", pose_fn(z), a, 8)["angle_deg"]
    assert abs(got - rec["smpl_pose"]["mlp"]["angle_deg"]) < 0.05
    got = score("yaw", yaw_fn(z), a, 8)["accuracy"]
    assert abs(got - rec["yaw"]["mlp"]["accuracy"]) < 0.005
