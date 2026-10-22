"""VAE prior sampling evaluation (spec 015 FR-008)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from strike_a_pose.eval.cli import main
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.sample import classify, largest_component_fraction, nn_mse
from strike_a_pose.eval.sample_config import config_from_dict, load_config
from strike_a_pose.models import ConvAutoencoder

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "sample_fixture.yaml"
CH = [16, 32, 64, 128, 256, 256]


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def _ckpt(path: Path, variational: bool, seed: int, latent_dim: int = 64) -> Path:
    torch.manual_seed(seed)
    m = ConvAutoencoder((256, 192), latent_dim, CH, variational=variational)
    torch.save({"model_hparams": m.hparams(), "model": m.state_dict()}, path)
    return path


def test_largest_component_fraction():
    z = np.zeros((5, 5), dtype=bool)
    assert largest_component_fraction(z) == (0.0, 0)
    a = z.copy()
    a[1:3, 1:3] = True
    assert largest_component_fraction(a) == (1.0, 1)
    a[4, 4] = True
    assert largest_component_fraction(a) == (pytest.approx(0.8), 2)
    d = z.copy()
    d[0, 0] = d[1, 1] = True
    assert largest_component_fraction(d) == (1.0, 1)  # 8-connectivity


def test_sample_validity():
    m = np.zeros((10, 10), dtype=bool)
    assert classify(m, 2, 20) == "empty"
    m[1:4, 1:4] = True
    assert classify(m, 2, 20) == "valid"
    assert classify(m, 10, 20) == "area"
    m[6:9, 6:9] = True
    assert classify(m, 2, 20) == "components"


def test_nearest_neighbour_mse():
    g = torch.Generator().manual_seed(0)
    train = torch.randint(0, 256, (5, 12), generator=g, dtype=torch.uint8)
    s = train[[2, 4]].float() / 255.0 + 0.001 * torch.randn(2, 12, generator=g)
    for chunk in (2, 10):
        idx, mse = nn_mse(s, train, chunk=chunk)
        assert idx.tolist() == [2, 4]
        want = ((s.double() - train[[2, 4]].double() / 255.0) ** 2).mean(1)
        assert mse.dtype == torch.float64
        assert mse.tolist() == pytest.approx(want.tolist(), rel=1e-12)


def test_config():
    cfg = load_config(CFG)
    assert cfg.with_overrides(seed=3).seed == 3
    assert load_config("configs/sample_vae.yaml").figure_samples == 32
    base = cfg.to_dict()
    for bad in ({"bogus": 1}, {"n_samples": 0}, {"iou_threshold": 1.5}, {"figure_samples": 65}):
        with pytest.raises(ConfigError):
            config_from_dict(base | bad)


def _sample(tmp_path, vae, ref, *extra, root_name="runs"):
    root = tmp_path / root_name
    argv = ["sample", "--config", str(CFG), "--run-root", str(root), "--checkpoint", str(vae)]
    if ref is not None:
        argv += ["--reference-checkpoint", str(ref)]
    code = main([*argv, *extra])
    return code, sorted(root.iterdir()) if root.exists() else []


def test_fixture_sample_run(tmp_path):
    vae = _ckpt(tmp_path / "vae.pt", True, 0)
    ae = _ckpt(tmp_path / "ae.pt", False, 1)
    code, dirs = _sample(tmp_path, vae, ae)
    assert code == 0 and len(dirs) == 1
    d = dirs[0]
    assert {p.name for p in d.iterdir()} == {
        "config.yaml", "run_info.json", "samples.json", "samples.png",
    }  # fmt: skip
    r = json.loads((d / "samples.json").read_text())
    assert {"metrics_version", "checkpoint", "reference_checkpoint", "dataset", "criteria",
            "sets"} <= set(r)  # fmt: skip
    assert set(r["sets"]) == {"vae_prior", "ae_prior", "vae_recon_val"}
    cfg = load_config(CFG)
    assert r["sets"]["vae_prior"]["n"] == r["sets"]["ae_prior"]["n"] == cfg.n_samples
    assert 0 < r["sets"]["vae_recon_val"]["n"] <= cfg.n_recon
    for s in r["sets"].values():
        assert 0.0 <= s["valid_fraction"] <= 1.0
        fails = s["fail_empty"] + s["fail_area"] + s["fail_components"]
        assert fails + round(s["valid_fraction"] * s["n"]) == s["n"]
    info = json.loads((d / "run_info.json").read_text())
    assert isinstance(info["peak_rss_mb"], float) and info["peak_rss_mb"] > 0
    code, dirs = _sample(tmp_path, vae, ae, root_name="again")
    assert code == 0 and len(dirs) == 1
    r2 = json.loads((dirs[0] / "samples.json").read_text())
    assert r2["sets"] == r["sets"] and r2["criteria"] == r["criteria"]


def test_refusals(tmp_path):
    vae = _ckpt(tmp_path / "vae.pt", True, 0)
    ae = _ckpt(tmp_path / "ae.pt", False, 1)
    small = _ckpt(tmp_path / "ae32.pt", False, 1, latent_dim=32)
    for ck, ref in ((ae, None), (vae, vae), (vae, small)):
        code, dirs = _sample(tmp_path, ck, ref)
        assert code == 2 and dirs == []
