"""Latent probe: ridge / angle / yaw units + fixture CLI end-to-end (spec 006)."""

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from strike_a_pose.data import make_split, manifest_content_sha256, write_split
from strike_a_pose.data.split import manifest_sha256
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.cli import main
from strike_a_pose.eval.probe import (
    Standardizer,
    _select,
    axis_angle_to_matrix,
    fit_mlp,
    ridge_fit,
    rotation_angle_deg,
    yaw_scores,
)
from strike_a_pose.train.cli import main as train_main

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "probe_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


# ------------------------------------------------------------------ units


def test_ridge_matches_normal_equations_and_recovers_linear_map():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(200, 5, generator=g, dtype=torch.float64)
    w_true = torch.randn(5, 3, generator=g, dtype=torch.float64)
    y = z @ w_true + torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    w, b = ridge_fit(z, y, 1e-12)
    assert torch.allclose(w, w_true, atol=1e-8) and torch.allclose(b, y.mean(0) - z.mean(0) @ w)
    lam = 0.3
    zc, yc = z - z.mean(0), y - y.mean(0)
    w_ref = torch.linalg.lstsq(
        torch.cat([zc, math.sqrt(lam * 200) * torch.eye(5, dtype=torch.float64)]),
        torch.cat([yc, torch.zeros(5, 3, dtype=torch.float64)]),
    ).solution
    assert torch.allclose(ridge_fit(z, y, lam)[0], w_ref, atol=1e-10)


def test_ridge_underdetermined_is_finite():
    z = torch.randn(8, 64, dtype=torch.float64)
    w, b = ridge_fit(z, torch.randn(8, 2, dtype=torch.float64), 0.1)
    assert torch.isfinite(w).all() and torch.isfinite(b).all()


def test_standardizer_constant_column():
    x = torch.tensor([[1.0, 5.0], [3.0, 5.0]])
    s = Standardizer.fit(x)
    assert torch.allclose(s(x), torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(s.inverse(s(x)), x.double())


def test_rotation_angle_hand_cases():
    zero = torch.zeros(1, 3)
    assert torch.allclose(axis_angle_to_matrix(zero)[0], torch.eye(3, dtype=torch.float64))
    rz90 = torch.tensor([[0.0, 0.0, math.pi / 2]])
    r = axis_angle_to_matrix(rz90)[0]
    assert torch.allclose(r @ torch.tensor([1.0, 0, 0], dtype=torch.float64),
                          torch.tensor([0.0, 1, 0], dtype=torch.float64), atol=1e-7)  # fmt: skip
    assert rotation_angle_deg(zero, zero).item() == pytest.approx(0.0, abs=1e-6)
    assert rotation_angle_deg(rz90, zero).item() == pytest.approx(90.0)
    assert rotation_angle_deg(rz90, -rz90).item() == pytest.approx(180.0, abs=1e-4)
    # same rotation, two axis-angle spellings (θ and θ − 2π about the flipped axis)
    a = torch.tensor([[0.0, 0.0, 3.0]])
    b = torch.tensor([[0.0, 0.0, 3.0 - 2 * math.pi]])
    assert rotation_angle_deg(a, b).item() == pytest.approx(0.0, abs=1e-4)


def test_yaw_scores_hand_case():
    true = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    pred = torch.tensor([0, 1, 6, 3, 0, 4, 6, 0])  # ok, ok, 180, ok, 180, 45, ok, 45
    r = yaw_scores(pred, true, 8)
    assert r["accuracy"] == 0.5
    assert r["errors_by_offset_deg"] == {"45": 2, "90": 0, "135": 0, "180": 2}
    conf = torch.tensor(r["confusion"])
    assert conf.sum() == 8 and conf.trace() == 4 and conf[2, 6] == 1 and conf[7, 0] == 1


def test_mlp_fits_separable_classes():
    g = torch.Generator().manual_seed(1)
    y = torch.arange(4).repeat(25)
    z = torch.nn.functional.one_hot(y, 4).double() * 3 + 0.1 * torch.randn(100, 4, generator=g)
    m = fit_mlp(z, y, True, (16,), 60, 20, 1e-2, 0.0, seed=0, n_classes=4)
    with torch.no_grad():
        assert (m(z.float()).argmax(1) == y).all()


def test_select_val_and_empty_fallback():
    val = {"n": 8, "z": torch.zeros(8, 1), "camera_index": torch.arange(8)}
    cands = {"0.1": lambda z: torch.zeros(8, dtype=torch.long), "1.0": lambda z: torch.arange(8)}
    k, sel = _select("yaw", cands, val, 8)
    assert k == "1.0" and sel["selected_by"] == "val accuracy" and sel["val"]["1.0"] == 1.0
    k, sel = _select("yaw", cands, {"n": 0}, 8)
    assert k == "0.1" and sel == {"value": 0.1, "selected_by": "default (empty val)", "val": {}}


# ------------------------------------------------------------------ end-to-end


@pytest.fixture(scope="module")
def ae_checkpoint(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ae")
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        assert train_main(["--config", "configs/ae_fixture.yaml", "--run-root", str(root)]) == 0
    finally:
        os.chdir(cwd)
    return next(root.iterdir()) / "checkpoint.pt"


def _probe(tmp_path: Path, ckpt: Path, *extra: str, cfg: Path = CFG) -> tuple[int, list[Path]]:
    root = tmp_path / "runs"
    code = main(["probe", "--config", str(cfg), "--run-root", str(root),
                 "--checkpoint", str(ckpt), *extra])  # fmt: skip
    return code, sorted(root.iterdir()) if root.exists() else []


def test_fixture_probe_run(tmp_path, capsys, ae_checkpoint):
    t0 = time.monotonic()
    code, dirs = _probe(tmp_path, ae_checkpoint, "--seed", "2")
    assert time.monotonic() - t0 < 60  # SC-001
    assert code == 0 and len(dirs) == 1
    d = dirs[0]
    assert d.name.startswith("probe-fixture-")
    assert {p.name for p in d.iterdir()} == {
        "config.yaml", "run_info.json", "metrics.json",
        "skeleton_test.png", "yaw_confusion_test.png",
    }  # fmt: skip
    out = capsys.readouterr().out
    assert "test (8)" in out and "val: no samples" in out

    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert cfg["seed"] == 2 and cfg["checkpoint"] == str(ae_checkpoint)

    info = json.loads((d / "run_info.json").read_text())
    ck = info["checkpoint"]
    assert ck["epoch"] == 10 and len(ck["sha256"]) == 64 and set(ck["train_run_git"]) == {
        "commit", "dirty",
    }  # fmt: skip
    assert info["cache"] == {"path": None, "hit": False}
    assert info["figures"] == ["skeleton_test.png", "yaw_confusion_test.png"]
    assert info["dataset"]["manifest_sha256"] == manifest_sha256(FIXTURE)
    assert info["dataset"]["manifest_content_sha256"] == manifest_content_sha256(
        load_manifest(FIXTURE)
    )  # spec 007 FR-007

    m = json.loads((d / "metrics.json").read_text())
    assert m["probe_version"] == 1 and m["encoder"]["epoch"] == 10
    assert m["classes"][0] == "Front_0" and len(m["classes"]) == 8
    assert m["fit"] == {"split": "train", "samples": 8, "poses": 1}
    assert m["splits"]["val"] == {"samples": 0, "poses": 0}
    probes = {"ridge", "mlp", "random_ridge", "random_mlp"}
    assert set(m["selection"]) == probes
    for p in probes:
        for t, sel in m["selection"][p].items():
            assert sel["selected_by"] == "default (empty val)"
            assert sel["value"] == (0.1 if "ridge" in p else 0.0), (p, t)
    tg = m["splits"]["test"]["targets"]
    assert set(tg) == {"joints_world", "joints_cam", "yaw", "smpl_pose"}
    assert set(tg["joints_world"]) == probes | {"mean_pose", "pca_k2"}
    assert set(tg["yaw"]) == probes | {"majority"}
    assert set(tg["smpl_pose"]) == probes | {"mean_pose"}
    for r in tg["joints_cam"].values():
        assert r["mpjpe_mm"] > 0 and r["kp2d_px"] > 0
    y = tg["yaw"]["mlp"]
    assert 0 <= y["accuracy"] <= 1 and sum(map(sum, y["confusion"])) == 8
    assert tg["joints_world"]["pca_k2"]["k_effective"] == 0  # 1 train pose: rank 0

    rows, cols = 4, 3
    assert Image.open(d / "skeleton_test.png").size == (
        cols * 192 + (cols + 1) * 4, 16 + rows * 256 + (rows + 1) * 4
    )  # fmt: skip


def test_repeatable_and_cache(tmp_path, ae_checkpoint):
    base = load_yaml(CFG) | {"cache_dir": str(tmp_path / "cache")}
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(base))
    a = _probe(tmp_path / "a", ae_checkpoint, cfg=cfg)[1][0]
    b = _probe(tmp_path / "b", ae_checkpoint, cfg=cfg)[1][0]
    ia, ib = (json.loads((x / "run_info.json").read_text()) for x in (a, b))
    assert ia["cache"]["hit"] is False and ib["cache"]["hit"] is True
    assert Path(ia["cache"]["path"]).is_file()
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    c = _probe(tmp_path / "c", ae_checkpoint)[1][0]  # no cache
    assert (a / "metrics.json").read_bytes() == (c / "metrics.json").read_bytes()


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def test_val_selection_with_val_split(tmp_path, ae_checkpoint):
    sp = write_split(
        make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)),
        tmp_path / "split.json",
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(load_yaml(CFG) | {
        "split_file": str(sp), "eval_splits": ["val"], "figure_split": "val",
        "random_encoder_control": False,
    }))  # fmt: skip
    code, (d,) = _probe(tmp_path, ae_checkpoint, cfg=cfg)
    assert code == 0
    m = json.loads((d / "metrics.json").read_text())
    assert set(m["selection"]) == {"ridge", "mlp"}
    sel = m["selection"]["ridge"]["joints_cam"]
    assert sel["selected_by"] == "val mpjpe_mm" and set(sel["val"]) == {"0.1", "1.0"}
    assert m["selection"]["mlp"]["yaw"]["selected_by"] == "val accuracy"
    assert "test" not in m["splits"] and m["splits"]["val"]["samples"] == 8


def test_probe_on_the_scored_best_checkpoint(tmp_path):
    """Spec 008 FR-008 / FR-009 (j): probe the train run's scored file (best.pt), as named
    by its metrics.json selection; small model on a fixture split with a val pose."""
    sp = write_split(
        make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)),
        tmp_path / "split.json",
    )
    ae = tmp_path / "ae.yaml"
    ae.write_text(yaml.safe_dump(load_yaml(ROOT / "configs" / "ae_fixture.yaml") | {
        "split_file": str(sp), "eval_splits": ["val"], "figure_split": "val", "epochs": 4,
        "channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8, "run_root": str(tmp_path / "ae"),
    }))  # fmt: skip
    assert train_main(["--config", str(ae)]) == 0
    (run,) = (tmp_path / "ae").iterdir()
    sel = json.loads((run / "metrics.json").read_text())["selection"]
    assert sel["weights"] == "best_val" and sel["checkpoint"] == "best.pt"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(load_yaml(CFG) | {
        "split_file": str(sp), "eval_splits": ["val"], "figure_split": "val",
        "random_encoder_control": False,
    }))  # fmt: skip
    code, (d,) = _probe(tmp_path, run / sel["checkpoint"], cfg=cfg)
    assert code == 0
    ck = json.loads((d / "run_info.json").read_text())["checkpoint"]
    assert ck["epoch"] == sel["epoch"] and ck["path"].endswith("best.pt")
    assert json.loads((d / "metrics.json").read_text())["encoder"]["latent_dim"] == 8


def test_errors(tmp_path, capsys, ae_checkpoint):
    code, dirs = _probe(tmp_path, tmp_path / "missing.pt")
    assert code == 2 and not dirs and "checkpoint" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(load_yaml(CFG) | {"targets": ["nope"]}))
    assert _probe(tmp_path, ae_checkpoint, cfg=bad) == (2, [])
    other = write_split(
        make_split({**load_manifest(FIXTURE), "seed": 99}, 0, (0.5, 0.0, 0.5)),  # other content
        tmp_path / "o.json",
    )
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(yaml.safe_dump(load_yaml(CFG) | {"split_file": str(other)}))
    assert _probe(tmp_path, ae_checkpoint, cfg=wrong) == (1, [])
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    assert _probe(tmp_path, junk)[0] == 2
