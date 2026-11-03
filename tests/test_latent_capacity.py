"""128-d latent end to end on the fixture (spec 019 FR-002)."""

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from strike_a_pose.data import make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval import probe_config, sample_config
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.probe import run_probe
from strike_a_pose.eval.roundtrip import run_round_trip
from strike_a_pose.eval.roundtrip_config import load_config as load_rt_config
from strike_a_pose.eval.sample import run_sample
from strike_a_pose.models import ConvAutoencoder, PoseHead, load_checkpoint
from strike_a_pose.train import trainer
from strike_a_pose.train.config import load_config
from strike_a_pose.train.trainer import run_training
from tests.conftest import make_synthetic_smpl

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
V3 = "configs/smpl_proc_v3/"
CH = [16, 32, 64, 128, 256, 256]
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 128}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.mark.parametrize(
    ("name", "base", "weight", "run_name"),
    [
        ("ae_l128.yaml", "ae.yaml", 0.0, "ae-l128-smpl_proc_v3"),
        ("ae_l128_pose_l0.03.yaml", "ae_pose_l0.03.yaml", 0.03, "ae-l128-pose-l0.03-smpl_proc_v3"),
        ("ae_l128_pose_l0.01.yaml", "ae_pose_l0.03.yaml", 0.01, "ae-l128-pose-l0.01-smpl_proc_v3"),
    ],
)
def test_shipped_l128_configs(name, base, weight, run_name):
    cfg = load_config(V3 + name)
    assert cfg.latent_dim == 128 and cfg.pose_weight == weight and cfg.run_name == run_name
    want = dataclasses.replace(
        load_config(V3 + base), run_name=run_name, latent_dim=128, pose_weight=weight
    )
    assert cfg == want


def test_model_sizes_128():
    m = ConvAutoencoder((256, 192), 128, CH)
    assert m.parameter_count() == 3520177 and m.bottleneck == (256, 4, 3)
    assert m(torch.zeros(2, 1, 256, 192)).shape == (2, 1, 256, 192)
    assert PoseHead(128).parameter_count() == 117320
    assert ConvAutoencoder((256, 192), 64, CH).parameter_count() == 3126897


def _train_cfg(tmp_path: Path, **kw):
    """Fixture images with a one-pose val (the committed split has an empty val)."""
    sp = tmp_path / "val_split.json"
    if not sp.exists():
        write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), sp)
    return load_config("configs/ae_fixture.yaml").with_overrides(
        run_root=str(tmp_path / "r"), split_file=str(sp), epochs=3,
        eval_splits=["val"], figure_split="val", **SMALL, **kw,
    )  # fmt: skip


@pytest.fixture
def varied(monkeypatch):
    """One train pose means no varying joint; add a seeded per-image offset (as 018 tests)."""
    real = trainer.load_split

    def noisy(*args, **kw):
        a = real(*args, **kw)
        if "joints_world" in a:
            g = torch.Generator().manual_seed(7)
            a["joints_world"] += 0.05 * torch.randn(a["joints_world"].shape, generator=g)
        return a

    monkeypatch.setattr(trainer, "load_split", noisy)


@pytest.mark.parametrize("pw", [0.0, 0.03])
def test_fixture_train_and_probe_128(tmp_path, varied, pw):
    d = run_training(_train_cfg(tmp_path, pose_weight=pw))
    metrics = json.loads((d / "metrics.json").read_text())
    ckpt_path = d / metrics["selection"]["checkpoint"]
    model, ckpt = load_checkpoint(ckpt_path)
    assert model.latent_dim == 128 and ckpt["model_hparams"]["latent_dim"] == 128
    if pw > 0:
        assert ckpt["pose_head_hparams"]["latent_dim"] == 128
        assert "pose_head" in ckpt and "pose_stats" in ckpt
        assert "pose_head" in metrics["splits"]["val"]
    else:
        assert "pose_head" not in ckpt
    pcfg = probe_config.load_config("configs/probe_fixture.yaml").with_overrides(
        checkpoint=str(ckpt_path), run_root=str(tmp_path / "p"), cache_dir=str(tmp_path / "c")
    )
    pm = json.loads((run_probe(pcfg) / "metrics.json").read_text())
    assert pm["encoder"]["latent_dim"] == 128
    jw = pm["splits"]["test"]["targets"]["joints_world"]
    for k in ("mlp", "random_mlp"):
        assert math.isfinite(jw[k]["mpjpe_mm"])


def _ckpt(path: Path, variational: bool, seed: int, latent_dim: int) -> Path:
    torch.manual_seed(seed)
    m = ConvAutoencoder((256, 192), latent_dim, CH, variational=variational)
    torch.save({"model_hparams": m.hparams(), "model": m.state_dict()}, path)
    return path


def test_sample_and_round_trip_128(tmp_path):
    vae = _ckpt(tmp_path / "vae.pt", True, 0, 128)
    ae = _ckpt(tmp_path / "ae.pt", False, 1, 128)
    scfg = sample_config.load_config("configs/sample_fixture.yaml").with_overrides(
        checkpoint=str(vae), reference_checkpoint=str(ae), run_root=str(tmp_path / "s")
    )
    sample_dir = run_sample(scfg)
    pcfg = probe_config.load_config("configs/probe_fixture.yaml").with_overrides(
        checkpoint=str(vae), run_root=str(tmp_path / "p"), cache_dir=str(tmp_path / "cache")
    )
    probe_dir = run_probe(pcfg)
    smpl = make_synthetic_smpl(tmp_path / "SMPL_SYNTH.npz")
    rcfg = load_rt_config("configs/roundtrip_fixture.yaml").with_overrides(
        sample_run=str(sample_dir), probe_run=str(probe_dir),
        smpl_model_path=str(smpl), run_root=str(tmp_path / "rt"),
    )  # fmt: skip
    d = run_round_trip(rcfg)
    assert (d / "samples.json").is_file() and (d / "round_trip_per_image.json").is_file()


def test_sample_refuses_mixed_latent(tmp_path):
    vae = _ckpt(tmp_path / "vae.pt", True, 0, 128)
    ref = _ckpt(tmp_path / "ae64.pt", False, 1, 64)
    scfg = sample_config.load_config("configs/sample_fixture.yaml").with_overrides(
        checkpoint=str(vae), reference_checkpoint=str(ref), run_root=str(tmp_path / "s")
    )
    with pytest.raises(ConfigError, match="latent_dim"):
        run_sample(scfg)
