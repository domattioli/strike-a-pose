"""Gradient-loss training (spec 020 FR-001, FR-003)."""

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from strike_a_pose.data import PoseImageDataset, make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.models import ConvAutoencoder
from strike_a_pose.train import trainer
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import LEGACY_VALUES, TrainConfig, config_from_dict, load_config
from strike_a_pose.train.objective import Objective
from strike_a_pose.train.trainer import (
    CL,
    HISTORY_FIELDS,
    _complete_row,
    load_split,
    make_optimizer,
    run_training,
    train_epoch,
)

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def train_images() -> torch.Tensor:
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    return load_split(ds, batch_size=4)["image"]


def _cfg(tmp_path: Path, root: str, **kw) -> TrainConfig:
    sp = tmp_path / "val_split.json"
    if not sp.exists():
        write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), sp)
    return load_config(CFG).with_overrides(
        run_root=str(tmp_path / root), split_file=str(sp), epochs=3,
        eval_splits=["val"], figure_split="val", **SMALL, **kw,
    )  # fmt: skip


@pytest.fixture
def varied(monkeypatch):
    real = trainer.load_split

    def noisy(*args, **kw):
        a = real(*args, **kw)
        if "joints_world" in a:
            g = torch.Generator().manual_seed(7)
            a["joints_world"] += 0.05 * torch.randn(a["joints_world"].shape, generator=g)
        return a

    monkeypatch.setattr(trainer, "load_split", noisy)


def _rows(d: Path) -> list[dict]:
    with (d / "history.csv").open() as fh:
        return [{k: v for k, v in r.items() if k != "epoch_s"} for r in csv.DictReader(fh)]


def _metrics(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


def test_gradient_weight_default_and_legacy():
    cfg = TrainConfig(dataset_dir="d", split_file="s")
    assert cfg.gradient_weight == 0.0
    assert LEGACY_VALUES["gradient_weight"] == 0.0
    assert Objective().gradient_weight == 0.0
    c = load_config(CFG).with_overrides(gradient_weight=0.5, pose_weight=0.03)
    o = Objective.from_config(c)
    assert o.gradient_weight == 0.5 and o.pose_weight == 0.03


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"gradient_weight": -0.1}, ">= 0"),
        ({"gradient_weight": "x"}, "finite"),
        ({"gradient_weight": float("nan")}, "finite"),
        ({"gradient_weight": 0.5, "mask_weight": 4.0}, "mask_weight"),
        ({"gradient_weight": 0.5, "variational": True, "beta": 0.1}, "variational"),
    ],
)
def test_gradient_weight_invalid(kw, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict({"dataset_dir": "d", "split_file": "s", **kw})


def test_default_run_is_unchanged_bitwise(tmp_path):
    a = run_training(_cfg(tmp_path, "a"))
    b = run_training(_cfg(tmp_path, "b", gradient_weight=0.0))
    assert _rows(a) == _rows(b)
    ba, bb = torch.load(a / "best.pt"), torch.load(b / "best.pt")
    assert all(torch.equal(ba["model"][k], bb["model"][k]) for k in bb["model"])
    assert HISTORY_FIELDS[-1] == "train_grad"
    assert all(r["train_grad"] == "" for r in _rows(a))


def test_train_epoch_without_grad_matches(train_images):
    runs = []
    for obj in (Objective(), Objective(gradient_weight=0.0)):
        torch.manual_seed(0)
        m = ConvAutoencoder().to(memory_format=CL)
        gen = torch.Generator().manual_seed(0)
        opt = make_optimizer(m.parameters(), TrainConfig("d", "s.json"))
        runs.append((train_epoch(m, opt, train_images, 3, gen, objective=obj), m))
    (sa, a), (sb, b) = runs
    assert sa["train_loss"] == sb["train_loss"]
    assert sa["train_grad"] is None and sb["train_grad"] is None
    assert all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


def test_grad_run_records_term(tmp_path):
    d = run_training(_cfg(tmp_path, "g", gradient_weight=0.5))
    for r in _rows(d):
        g = float(r["train_grad"])
        assert math.isfinite(g) and g > 0
        assert float(r["train_loss"]) == pytest.approx(float(r["train_recon"]) + 0.5 * g, rel=1e-4)
    m = _metrics(d)
    assert m["training"]["objective"]["gradient_weight"] == 0.5
    assert 0.0 <= m["splits"]["val"]["image"]["ae"]["boundary_f1"] <= 1.0


def test_grad_and_pose_run(tmp_path, varied):
    d = run_training(_cfg(tmp_path, "gp", gradient_weight=0.5, pose_weight=0.03))
    for r in _rows(d):
        want = (
            float(r["train_recon"]) + 0.5 * float(r["train_grad"]) + 0.03 * float(r["train_pose"])
        )
        assert float(r["train_loss"]) == pytest.approx(want, rel=1e-4)


def test_grad_run_repeats(tmp_path):
    a = run_training(_cfg(tmp_path, "a", gradient_weight=0.5))
    b = run_training(_cfg(tmp_path, "b", gradient_weight=0.5))
    assert _rows(a) == _rows(b)
    assert _metrics(a)["splits"] == _metrics(b)["splits"]


def test_grad_resume_equals_uninterrupted(tmp_path):
    full = run_training(_cfg(tmp_path, "full", gradient_weight=0.5))
    d = run_training(_cfg(tmp_path, "cut", gradient_weight=0.5), stop_after_epoch=1)
    assert main(["--resume", str(d)]) == 0
    assert _rows(d) == _rows(full)
    ba, bb = torch.load(d / "best.pt"), torch.load(full / "best.pt")
    assert all(torch.equal(ba["model"][k], bb["model"][k]) for k in bb["model"])


def test_pre_020_history_rows_complete():
    row = {"epoch": 1, "train_loss": 1.0, "train_recon": 1.0, "train_kl": None, "train_pose": None}
    assert _complete_row(row)["train_grad"] is None
