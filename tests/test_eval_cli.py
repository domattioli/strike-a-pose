"""`strike-a-pose-eval baseline` end-to-end on the committed fixture (spec 003 US2, SC-002)."""

import json
import time
from pathlib import Path

import pytest
import yaml
from PIL import Image

from strike_a_pose.data import make_split, write_split
from strike_a_pose.data.split import manifest_sha256
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.cli import main
from strike_a_pose.eval.report import FIG_HEADER, FIG_PAD

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "baselines_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


def _run(tmp_path: Path, *extra: str) -> tuple[int, list[Path]]:
    root = tmp_path / "runs"
    code = main(["baseline", "--config", str(CFG), "--run-root", str(root), *extra])
    return code, sorted(root.iterdir()) if root.exists() else []


def test_fixture_run_directory(tmp_path, capsys):
    t0 = time.monotonic()
    code, dirs = _run(tmp_path, "--seed", "3")
    assert time.monotonic() - t0 < 30  # SC-002
    assert code == 0 and len(dirs) == 1
    d = dirs[0]
    assert d.name.startswith("baselines-fixture-")
    assert {p.name for p in d.iterdir()} == {
        "config.yaml", "run_info.json", "metrics.json", "recon_test.png",
    }  # fmt: skip
    assert "test (8)" in capsys.readouterr().out

    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert cfg["seed"] == 3 and cfg["run_root"] == str(tmp_path / "runs")
    assert cfg["image_pca_ks"] == [2, 4] and cfg["dataset_dir"] == "tests/fixtures/smpl_fixture"

    info = json.loads((d / "run_info.json").read_text())
    assert info["seed"] == 3 and set(info["git"]) == {"commit", "dirty"}
    assert {"python", "torch", "numpy", "pillow", "platform"} <= set(info["env"])
    assert info["dataset"]["manifest_sha256"] == manifest_sha256(FIXTURE)
    assert len(info["split_file"]["sha256"]) == 64
    assert info["finished_utc"] and info["wall_s"] is not None and info["figure"]

    m = json.loads((d / "metrics.json").read_text())
    assert m["metrics_version"] == 1 and m["dataset"]["name"] == "smpl_fixture"
    assert m["fit"]["samples"] == 8 and m["fit"]["poses"] == 1
    assert m["fit"]["pose_pca"]["numerical_rank"] == 0  # one train pose: mean only
    assert m["fit"]["image_pca"]["n_components"] == 4
    assert m["splits"]["val"] == {"samples": 0, "poses": 0}
    t = m["splits"]["test"]
    assert t["samples"] == 8 and t["poses"] == 1
    assert set(t["image"]) == {"mean_image", "pca_k2", "pca_k4"}
    assert set(t["pose"]) == {"mean_pose", "pca_k2"}
    for r in t["image"].values():
        assert 0 < r["image_mse"] < 1 and r["psnr_db"] > 0 and 0 <= r["silhouette_iou"] <= 1
    assert t["image"]["pca_k4"]["k_effective"] == 4
    pk = t["pose"]["pca_k2"]
    assert pk["k_effective"] == 0 and pk["mpjpe_mm"] == t["pose"]["mean_pose"]["mpjpe_mm"]
    assert t["pose"]["mean_pose"]["mpjpe_mm"] > 0 and t["pose"]["mean_pose"]["kp2d_samples"] == 8

    fig = Image.open(d / "recon_test.png")
    rows, cols = 4, 4  # figure_samples; true + mean + 2 PCA
    assert fig.size == (
        cols * 192 + (cols + 1) * FIG_PAD,
        FIG_HEADER + rows * 256 + (rows + 1) * FIG_PAD,
    )


def test_metrics_bytes_repeat(tmp_path):
    a = _run(tmp_path / "a")[1][0] / "metrics.json"
    b = _run(tmp_path / "b")[1][0] / "metrics.json"
    assert a.read_bytes() == b.read_bytes()


def test_split_mismatch_exit_1_no_run_dir(tmp_path, capsys):
    other = make_split(load_manifest(FIXTURE), "0" * 64, 0, (0.5, 0.0, 0.5))
    sp = write_split(other, tmp_path / "other.json")
    cfg = yaml.safe_load(CFG.read_text()) | {"split_file": str(sp), "run_root": str(tmp_path / "r")}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert main(["baseline", "--config", str(p)]) == 1
    assert "manifest" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_config_error_exit_2(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset_dir: x\n")
    assert main(["baseline", "--config", str(p)]) == 2
    assert main(["baseline", "--config", str(tmp_path / "missing.yaml")]) == 2
