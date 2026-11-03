"""Run records and memory (spec 012 FR-010, FR-011): batched mean image, CPU model and the
dataset checksums hash in run_info.json (003 / 004 / 006)."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.eval.baselines import image_mean
from strike_a_pose.eval.cli import main as eval_main
from strike_a_pose.runlog import checksums_sha256, cpu_model, environment_summary
from strike_a_pose.train.config import load_config
from strike_a_pose.train.trainer import load_split, run_training, to_float

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"
CHECKSUMS = hashlib.sha256((FIXTURE / "checksums.json").read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def _one_shot(x: torch.Tensor) -> torch.Tensor:
    """The pre-012 computation (a float64 copy of the whole matrix)."""
    return (x.sum(dim=0, dtype=torch.float64) / x.shape[0]).to(torch.float32)


@pytest.mark.parametrize("batch", [1, 3, 7, 13, 64])
def test_image_mean_equals_the_one_shot_float64_mean_bitwise(batch):
    torch.manual_seed(0)
    u8 = torch.randint(0, 256, (13, 50), dtype=torch.uint8)
    ref = _one_shot(to_float(u8))
    assert torch.equal(image_mean(u8, batch), ref)  # uint8 rows, scaled per batch
    assert torch.equal(image_mean(to_float(u8), batch), ref)  # float32 rows
    assert image_mean(u8, batch).dtype == torch.float32


def test_image_mean_on_fixture_images():
    for split in ("train", "test"):
        ds = PoseImageDataset(FIXTURE, split=split, split_file=SPLIT)
        u8 = load_split(ds, 4)["image"].reshape(len(ds), -1)
        assert torch.equal(image_mean(u8, 3), _one_shot(to_float(u8)))


def test_cpu_model_and_environment_summary():
    env = environment_summary()
    assert env["cpu_model"] == cpu_model()
    info = Path("/proc/cpuinfo")
    names = [ln.split(":", 1)[1].strip() for ln in info.read_text().splitlines()
             if ln.startswith("model name")] if info.is_file() else []  # fmt: skip
    if names:
        assert env["cpu_model"] == names[0]
    else:
        assert env["cpu_model"] is None or isinstance(env["cpu_model"], str)


def test_checksums_sha256(tmp_path):
    assert checksums_sha256(FIXTURE) == CHECKSUMS
    assert checksums_sha256(tmp_path) is None  # no checksums.json


def test_run_info_records_cpu_model_and_checksums_hash(tmp_path):
    over = {"run_root": str(tmp_path / "train"), "epochs": 1, "channels": [4] * 6, "latent_dim": 8}
    train = run_training(load_config(ROOT / "configs" / "ae_fixture.yaml").with_overrides(**over))
    root = str(tmp_path / "eval")
    assert (
        eval_main(["baseline", "--config", "configs/baselines_fixture.yaml", "--run-root", root])
        == 0
    )
    probe = ["probe", "--config", "configs/probe_fixture.yaml", "--run-root", root]
    assert eval_main([*probe, "--checkpoint", str(train / "checkpoint.pt")]) == 0
    dirs = [train, *sorted((tmp_path / "eval").iterdir())]
    assert len(dirs) == 3
    for d in dirs:
        info = json.loads((d / "run_info.json").read_text())
        assert info["dataset"]["checksums_sha256"] == CHECKSUMS, d
        assert info["env"]["cpu_model"] == cpu_model(), d
