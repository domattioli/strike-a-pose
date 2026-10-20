"""AE training: overfit smoke + fixture CLI end-to-end (spec 004 US1/US2, SC-001)."""

import csv
import json
import time
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from strike_a_pose.data import PoseImageDataset, make_split, write_split
from strike_a_pose.data.split import manifest_sha256
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.report import FIG_HEADER, FIG_PAD
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import load_config
from strike_a_pose.train.trainer import (
    HISTORY_FIELDS,
    load_split,
    overfit_steps,
    run_training,
    to_float,
    train_epoch,
)

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.fixture(scope="module")
def train_images() -> torch.Tensor:
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    return load_split(ds, batch_size=4)["image"]


def test_load_split_matches_loader(train_images):
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    assert train_images.shape == (8, 1, 256, 192) and train_images.dtype == torch.uint8
    for i in (0, 7):
        assert torch.equal(to_float(train_images[i]), ds[i]["image"])  # bit-identical


def test_overfit_fixed_batch(train_images):
    """Constitution IV smoke: loss on one fixed batch falls within a few steps."""
    torch.manual_seed(0)
    x = to_float(train_images[:4])
    losses = overfit_steps(ConvAutoencoder(), x, steps=15, lr=3e-3)
    assert losses[-1] < 0.5 * losses[0]
    assert min(losses[-5:]) < min(losses[:5])


def test_train_epoch_partial_last_batch(train_images):
    torch.manual_seed(0)
    m = ConvAutoencoder()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss, lr = train_epoch(m, opt, train_images, 3, torch.Generator().manual_seed(0))  # 3+3+2
    assert 0 < loss < 1 and lr == 1e-3
    assert opt.state[next(iter(m.parameters()))]["step"] == 3


def _run(tmp_path: Path, *extra: str, cfg: Path = CFG) -> tuple[int, list[Path]]:
    root = tmp_path / "runs"
    code = main(["--config", str(cfg), "--run-root", str(root), *extra])
    return code, sorted(root.iterdir()) if root.exists() else []


def _history(d: Path) -> list[dict]:
    with (d / "history.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_fixture_train_run(tmp_path, capsys):
    t0 = time.monotonic()
    code, dirs = _run(tmp_path, "--seed", "3")
    assert time.monotonic() - t0 < 60  # SC-001
    assert code == 0 and len(dirs) == 1
    d = dirs[0]
    assert d.name.startswith("ae-fixture-")
    assert {p.name for p in d.iterdir()} == {
        "config.yaml", "run_info.json", "history.csv", "checkpoint.pt", "metrics.json",
        "recon_test.png",
    }  # fmt: skip
    out = capsys.readouterr().out
    assert "epoch 10/10" in out and "test (8)" in out

    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert cfg["seed"] == 3 and cfg["run_root"] == str(tmp_path / "runs")
    assert cfg["channels"] == [16, 32, 64, 128, 256, 256] and cfg["epochs"] == 10

    info = json.loads((d / "run_info.json").read_text())
    assert info["seed"] == 3 and set(info["git"]) == {"commit", "dirty"}
    assert {"python", "torch", "numpy", "pillow", "platform", "torch_threads"} <= set(info["env"])
    assert info["dataset"]["manifest_sha256"] == manifest_sha256(FIXTURE)
    assert info["model_parameters"] == 3_126_897
    assert info["wall_s"] is not None and info["train_wall_s"] is not None
    assert info["figure"] == "recon_test.png"

    hist = _history(d)
    assert tuple(hist[0]) == HISTORY_FIELDS and [r["epoch"] for r in hist] == [
        str(e) for e in range(1, 11)
    ]
    assert all(r["val_loss"] == "" for r in hist)  # fixture val is empty
    assert float(hist[-1]["train_loss"]) < float(hist[0]["train_loss"])  # loss decreases
    assert float(hist[-1]["lr"]) < float(hist[0]["lr"]) <= 1e-3  # cosine

    m = json.loads((d / "metrics.json").read_text())
    assert m["train_version"] == 1 and m["metrics_version"] == 1
    assert m["model"]["latent_dim"] == 64 and m["model"]["parameters"] == 3_126_897
    t = m["training"]
    assert t["epochs"] == 10 and t["steps"] == 20 and t["train_samples"] == 8
    assert t["final_train_loss"] < t["first_train_loss"] and t["final_val_loss"] is None
    assert m["context"] == {"fit_split": "train", "pca_k": 4, "pca_k_effective": 4}
    assert m["splits"]["val"] == {"samples": 0, "poses": 0}
    ts = m["splits"]["test"]
    assert ts["samples"] == 8 and set(ts["image"]) == {"ae", "mean_image", "pca_k4"}
    for r in ts["image"].values():
        assert 0 < r["image_mse"] < 1 and r["psnr_db"] > 0 and 0 <= r["silhouette_iou"] <= 1

    fig = Image.open(d / "recon_test.png")
    rows, cols = 4, 4  # figure_samples; true, ae, mean, pca
    assert fig.size == (
        cols * 192 + (cols + 1) * FIG_PAD,
        FIG_HEADER + rows * 256 + (rows + 1) * FIG_PAD,
    )

    model, ckpt = load_checkpoint(d / "checkpoint.pt")
    assert ckpt["epoch"] == 10 and ckpt["config"]["seed"] == 3 and model.latent_dim == 64


def test_repeatable(tmp_path):
    a = _run(tmp_path / "a")[1][0]
    b = _run(tmp_path / "b")[1][0]
    strip = [{k: r[k] for k in ("train_loss", "val_loss", "lr")} for r in _history(a)]
    assert strip == [{k: r[k] for k in ("train_loss", "val_loss", "lr")} for r in _history(b)]
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()


def test_split_mismatch_exit_1_no_run_dir(tmp_path, capsys):
    other = make_split(load_manifest(FIXTURE), "0" * 64, 0, (0.5, 0.0, 0.5))
    sp = write_split(other, tmp_path / "other.json")
    cfg = yaml.safe_load(CFG.read_text()) | {"split_file": str(sp), "run_root": str(tmp_path / "r")}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(p)]) == 1
    assert "manifest" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_config_errors_exit_2(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("dataset_dir: x\n")
    assert main(["--config", str(p)]) == 2
    bad = yaml.safe_load(CFG.read_text()) | {"channels": [8] * 7, "run_root": str(tmp_path / "r")}
    p.write_text(yaml.safe_dump(bad))
    assert main(["--config", str(p)]) == 2  # 192 not divisible by 2^7
    assert "divisible" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_resume_equals_uninterrupted(tmp_path, capsys):
    """Restart resilience (clarification 2026-10-20b): interrupted after epoch 4 and
    resumed (with a stray history row past the checkpoint) → same history + metrics."""
    full = _run(tmp_path / "full")[1][0]
    cfg = load_config(CFG).with_overrides(run_root=str(tmp_path / "cut"))
    d = run_training(cfg, stop_after_epoch=4)
    assert not (d / "metrics.json").exists()
    assert len(_history(d)) == 4 and torch.load(d / "checkpoint.pt")["epoch"] == 4
    with (d / "history.csv").open("a") as fh:
        fh.write("5,0.5,,0.001,1.0\n")  # killed after the history write, before the checkpoint
    assert main(["--resume", str(d)]) == 0
    assert "resuming" in capsys.readouterr().out
    keys = ("epoch", "train_loss", "val_loss", "lr")
    assert [{k: r[k] for k in keys} for r in _history(d)] == [
        {k: r[k] for k in keys} for r in _history(full)
    ]
    assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes()
    a, b = torch.load(d / "checkpoint.pt"), torch.load(full / "checkpoint.pt")
    assert a["epoch"] == b["epoch"] == 10
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in b["model"])
    info = json.loads((d / "run_info.json").read_text())
    assert [x["start_epoch"] for x in info["sessions"]] == [1, 5]
    assert info["finished_utc"] is not None and info["wall_s"] > 0
    assert main(["--resume", str(d)]) == 2  # already finished
    assert "already finished" in capsys.readouterr().err


def test_resume_errors(tmp_path, capsys):
    assert main(["--resume", str(tmp_path)]) == 2  # no checkpoint
    assert "no checkpoint" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["--resume", str(tmp_path), "--seed", "1"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["--config", str(CFG), "--resume", str(tmp_path)])


def _val_cfg(tmp_path: Path, root: str):
    """Fixture split with a val pose (train 1 / val 1 / test 0 poses)."""
    sp = tmp_path / "val_split.json"
    if not sp.exists():
        write_split(
            make_split(load_manifest(FIXTURE), manifest_sha256(FIXTURE), 0, (0.5, 0.5, 0.0)), sp
        )
    return load_config(CFG).with_overrides(
        run_root=str(tmp_path / root), split_file=str(sp), epochs=6,
        eval_splits=["val"], figure_split="val",
    )  # fmt: skip


def test_best_checkpoint_and_resume(tmp_path):
    """Surrogate ruling 2026-10-20 #3: additive best.pt (min val loss), resume-safe."""
    full = run_training(_val_cfg(tmp_path, "full"))
    vals = [float(r["val_loss"]) for r in _history(full)]
    best_epoch = vals.index(min(vals)) + 1
    model, ck = load_checkpoint(full / "best.pt")
    assert ck["epoch"] == best_epoch and model.latent_dim == 64
    assert format(ck["val_loss"], ".8g") == _history(full)[best_epoch - 1]["val_loss"]
    m = json.loads((full / "metrics.json").read_text())
    assert m["training"]["best_val_epoch"] == best_epoch
    info = json.loads((full / "run_info.json").read_text())
    assert info["best_val_epoch"] == best_epoch and info["best_checkpoint"] == "best.pt"
    assert torch.load(full / "checkpoint.pt")["epoch"] == 6  # final weights still evaluated

    cut = run_training(_val_cfg(tmp_path, "cut"), stop_after_epoch=3)
    assert main(["--resume", str(cut)]) == 0
    a, b = torch.load(cut / "best.pt"), torch.load(full / "best.pt")
    assert a["epoch"] == b["epoch"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in b["model"])
    assert (cut / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes()


def test_no_best_checkpoint_without_val(tmp_path):
    d = _run(tmp_path)[1][0]  # fixture val is empty
    assert not (d / "best.pt").exists()
    m = json.loads((d / "metrics.json").read_text())
    assert m["training"]["best_val_epoch"] is None and m["training"]["best_val_loss"] is None
