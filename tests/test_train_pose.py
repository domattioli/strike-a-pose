"""Pose-supervised autoencoder training (spec 018 FR-001, FR-003-FR-005)."""

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from strike_a_pose.data import PoseImageDataset, make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.metrics import mpjpe
from strike_a_pose.models import ConvAutoencoder, PoseHead, PoseTargetStats, load_checkpoint
from strike_a_pose.models.pose_head import root_relative
from strike_a_pose.train import trainer
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import LEGACY_VALUES, TrainConfig, config_from_dict, load_config
from strike_a_pose.train.objective import Objective
from strike_a_pose.train.trainer import (
    CL,
    HISTORY_FIELDS,
    load_split,
    make_optimizer,
    pose_head_scores,
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
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.fixture(scope="module")
def train_images() -> torch.Tensor:
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    return load_split(ds, batch_size=4)["image"]


def _cfg(tmp_path: Path, root: str, **kw) -> TrainConfig:
    """Fixture images with a val pose (train 1 / val 1 / test 0: the committed fixture split
    has an empty val, hence no best.pt); only val is scored."""
    sp = tmp_path / "val_split.json"
    if not sp.exists():
        write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), sp)
    return load_config(CFG).with_overrides(
        run_root=str(tmp_path / root), split_file=str(sp), epochs=3,
        eval_splits=["val"], figure_split="val", **SMALL, **kw,
    )  # fmt: skip


@pytest.fixture
def varied(monkeypatch):
    """The fixture's train split is one pose, so no joint varies (no pose target); add a fixed
    per-image offset (seeded per call, so repeats and resumes see the same labels)."""
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


def test_pose_weight_default_and_legacy():
    cfg = TrainConfig(dataset_dir="d", split_file="s")
    assert cfg.pose_weight == 0.0
    assert LEGACY_VALUES["pose_weight"] == 0.0
    assert Objective().pose_weight == 0.0
    c = load_config(CFG).with_overrides(pose_weight=0.03)
    assert Objective.from_config(c).pose_weight == 0.03


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"pose_weight": -0.1}, ">= 0"),
        ({"pose_weight": 0.03, "variational": True, "beta": 0.1}, "variational"),
    ],
)
def test_pose_weight_invalid(kw, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict({"dataset_dir": "d", "split_file": "s", **kw})


def test_load_split_joints():
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    a = load_split(ds, 4, joints=True)
    n = a["n"]
    assert a["joints_world"].shape == (n, 24, 3) and a["joints_world"].dtype == torch.float32
    for i in (0, n - 1):
        assert torch.equal(a["joints_world"][i], ds[i]["joints_3d_world"].float())
    assert "joints_world" not in load_split(ds, 4)


def test_default_run_is_unchanged_bitwise(tmp_path):
    a = run_training(_cfg(tmp_path, "a"))
    b = run_training(_cfg(tmp_path, "b", pose_weight=0.0))
    assert _rows(a) == _rows(b)
    ba, bb = torch.load(a / "best.pt"), torch.load(b / "best.pt")
    assert "pose_head" not in ba
    assert all(torch.equal(ba["model"][k], bb["model"][k]) for k in bb["model"])
    with (a / "history.csv").open() as fh:
        rd = csv.reader(fh)
        assert tuple(next(rd)) == HISTORY_FIELDS
    assert "train_pose" in HISTORY_FIELDS
    assert all(r["train_pose"] == "" for r in _rows(a))
    assert "pose_head" not in _metrics(a)


def test_train_epoch_without_pose_matches_pre_018(train_images):
    runs = []
    for kw in ({}, {"pose": None}):
        torch.manual_seed(0)
        m = ConvAutoencoder().to(memory_format=CL)
        gen = torch.Generator().manual_seed(0)
        opt = make_optimizer(m.parameters(), TrainConfig("d", "s.json"))
        st = train_epoch(m, opt, train_images, 3, gen, objective=Objective(), **kw)
        runs.append((st, m))
    (sa, a), (sb, b) = runs
    assert sa["train_loss"] == sb["train_loss"]
    assert sa["train_pose"] is None
    assert all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


def _pose_run(tmp_path, root, **kw):
    return run_training(_cfg(tmp_path, root, pose_weight=0.03, **kw))


def test_pose_run_records_head(tmp_path, varied):
    d = _pose_run(tmp_path, "p")
    for r in _rows(d):
        assert math.isfinite(float(r["train_pose"]))
        want = float(r["train_recon"]) + 0.03 * float(r["train_pose"])
        assert float(r["train_loss"]) == pytest.approx(want, rel=1e-6)
    m = _metrics(d)
    assert m["pose_head"]["parameters"] == PoseHead(8).parameter_count()
    assert m["pose_head"]["pose_weight"] == 0.03
    assert m["pose_head"]["target"] == "joints_world"
    assert 0 < m["splits"]["val"]["pose_head"]["mpjpe_mm"] < 2000
    assert "pose_head_mpjpe_mm" in m["units"]
    assert m["training"]["objective"]["pose_weight"] == 0.03
    for f in ("best.pt", "checkpoint.pt"):
        ck = torch.load(d / f)
        assert {"pose_head", "pose_head_hparams", "pose_stats"} <= set(ck)


def test_pose_head_has_the_specified_size_at_default_latent(tmp_path, varied):
    cfg = load_config(CFG).with_overrides(run_root=str(tmp_path), epochs=1, pose_weight=0.03)
    d = run_training(cfg)
    assert _metrics(d)["pose_head"]["parameters"] == 100936


def test_pose_run_repeats(tmp_path, varied):
    a, b = _pose_run(tmp_path, "a"), _pose_run(tmp_path, "b")
    assert _rows(a) == _rows(b)
    assert _metrics(a)["splits"]["val"]["pose_head"] == _metrics(b)["splits"]["val"]["pose_head"]


def test_pose_resume_equals_uninterrupted(tmp_path, varied):
    full = _pose_run(tmp_path, "full")
    d = run_training(_cfg(tmp_path, "cut", pose_weight=0.03), stop_after_epoch=1)
    assert main(["--resume", str(d)]) == 0
    assert _rows(d) == _rows(full)
    ha, hb = torch.load(d / "best.pt")["pose_head"], torch.load(full / "best.pt")["pose_head"]
    assert all(torch.equal(ha[k], hb[k]) for k in hb)
    assert _metrics(d)["splits"]["val"]["pose_head"] == _metrics(full)["splits"]["val"]["pose_head"]


def test_pose_head_changes_the_encoder(tmp_path, varied):
    a = run_training(_cfg(tmp_path, "a"))
    b = _pose_run(tmp_path, "b")
    wa, wb = torch.load(a / "best.pt")["model"], torch.load(b / "best.pt")["model"]
    enc = [k for k in wa if k.startswith("encoder")]
    assert enc and any(not torch.equal(wa[k], wb[k]) for k in enc)


def test_probe_loads_pose_checkpoint(tmp_path, varied):
    d = _pose_run(tmp_path, "p")
    model, _ = load_checkpoint(d / "best.pt")
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    x = ds[0]["image"].reshape(1, 1, *ds.image_size)
    with torch.no_grad():
        assert model.encode(x).shape == (1, 8)


def test_head_mpjpe_is_the_probe_metric():
    torch.manual_seed(0)
    model = ConvAutoencoder((64, 64), 8, (4, 4, 4, 4, 4, 4)).to(memory_format=CL)
    head = PoseHead(8)
    for p in head.net[-1].parameters():
        torch.nn.init.zeros_(p)
    j = torch.randn(5, 24, 3, generator=torch.Generator().manual_seed(1))
    stats = PoseTargetStats.fit(root_relative(j).double())
    imgs = torch.randint(0, 256, (5, 1, 64, 64), dtype=torch.uint8)
    got = pose_head_scores(model, head, stats, imgs, j, 2)
    mean = stats.mean.double().reshape(1, 24, 3).expand(5, 24, 3)
    want = float(mpjpe(mean, root_relative(j).double().reshape(5, 24, 3)).mean() * 1000)
    assert got["mpjpe_mm"] == pytest.approx(want, rel=1e-6)
    assert got["target"] == "joints_world"


def test_pre_018_run_dir_resumes(tmp_path):
    full = run_training(_cfg(tmp_path, "full"))
    d = run_training(_cfg(tmp_path, "cut"), stop_after_epoch=1)
    c = yaml.safe_load((d / "config.yaml").read_text())
    del c["pose_weight"]
    (d / "config.yaml").write_text(yaml.safe_dump(c))
    ck = torch.load(d / "checkpoint.pt")
    del ck["config"]["pose_weight"]
    ck["history"] = [{k: v for k, v in r.items() if k != "train_pose"} for r in ck["history"]]
    torch.save(ck, d / "checkpoint.pt")
    assert main(["--resume", str(d)]) == 0
    assert _rows(d) == _rows(full)


def test_pose_without_varying_joints_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="varies"):
        _pose_run(tmp_path, "p")
