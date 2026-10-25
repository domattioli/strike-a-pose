"""Training variants (spec 011): config keys, optimizer, regularization, records, resume."""

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.eval.cli import main as eval_main
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import LEGACY_VALUES, TrainConfig, config_from_dict, load_config
from strike_a_pose.train.objective import Objective, gaussian_kl, recon_loss, shift_batch
from strike_a_pose.train.trainer import (
    CL,
    HISTORY_FIELDS,
    load_split,
    make_optimizer,
    resume_training,
    run_training,
    to_float,
    train_epoch,
)

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}  # fast; the rules do not need size
NEW_KEYS = {"mask_weight": 0.0, "variational": False, "beta": 0.0, "weight_decay": 0.0,
            "shift_px": 0, "dropout": 0.0}  # fmt: skip
REG = {"weight_decay": 0.05, "shift_px": 4, "dropout": 0.1}  # configs/ae_reg.yaml
MASK = {"mask_weight": 4.0}  # configs/ae_mask.yaml
VAE = {"variational": True, "beta": 0.1}  # fixture VAE runs
VAE_CONFIG = {"variational": True, "beta": 0.01}  # configs/ae_vae.yaml (spec 012: first beta)
ALL_KNOBS = {**REG, **MASK, **VAE}  # every knob on (resume test)
HIST_KEYS = ("train_loss", "val_loss", "lr", "train_recon", "train_kl")


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.fixture(scope="module")
def train_images() -> torch.Tensor:
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    return load_split(ds, batch_size=4)["image"]


def _cfg(tmp_path: Path, root: str, **kw) -> TrainConfig:
    return load_config(CFG).with_overrides(run_root=str(tmp_path / root), epochs=3, **SMALL, **kw)


def _hist(d: Path) -> list[dict]:
    with (d / "history.csv").open() as fh:
        return [{k: r[k] for k in HIST_KEYS} for r in csv.DictReader(fh)]


def _metrics(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


# ------------------------------------------------------------------ config (FR-001, FR-002)


def test_defaults_are_the_plain_ae_and_the_legacy_values():
    cfg = TrainConfig("d", "s.json")
    assert {k: getattr(cfg, k) for k in NEW_KEYS} == NEW_KEYS
    assert {k: LEGACY_VALUES[k] for k in NEW_KEYS} == NEW_KEYS
    assert config_from_dict({"dataset_dir": "d", "split_file": "s.json"}) == cfg


@pytest.mark.parametrize(
    "kw",
    [
        {"mask_weight": -1.0},
        {"mask_weight": float("inf")},
        {"mask_weight": None},
        {"variational": 1},
        {"variational": "true"},
        {"variational": True},  # beta 0
        {"beta": 0.1},  # without variational
        {"beta": -1.0, "variational": True},
        {"beta": float("nan"), "variational": True},
        {"weight_decay": -0.1},
        {"weight_decay": float("nan")},
        {"weight_decay": "0.1"},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"dropout": True},
        {"shift_px": -1},
        {"shift_px": 33},
        {"shift_px": 2.0},
        {"shift_px": True},
    ],
)
def test_invalid_variant_keys(kw):
    with pytest.raises(ConfigError, match=next(iter(kw))):
        config_from_dict({"dataset_dir": "d", "split_file": "s.json", **kw})


def test_variant_keys_are_normalized():
    cfg = config_from_dict(
        {
            "dataset_dir": "d",
            "split_file": "s.json",
            "weight_decay": 1,
            "shift_px": 32,
            "dropout": 0,
        }  # fmt: skip
    )
    assert (cfg.weight_decay, cfg.shift_px, cfg.dropout) == (1.0, 32, 0.0)
    assert isinstance(cfg.weight_decay, float) and isinstance(cfg.dropout, float)
    assert config_from_dict(yaml.safe_load(yaml.safe_dump(cfg.to_dict()))) == cfg


# ------------------------------------------------------------------ defaults (FR-011)


def _pre_011_epoch(model, opt, images: torch.Tensor, batch_size: int, gen) -> float:
    """Inline copy of the pre-011 ``train_epoch`` loop: the regression reference."""
    model.train()
    n = images.shape[0]
    perm = torch.randperm(n, generator=gen)
    total = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        x = to_float(images[idx]).contiguous(memory_format=CL)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(idx)
    return total / n


def test_defaults_equal_the_pre_011_loop(train_images):
    """US1-1: the default objective with AdamW(weight_decay=0) is bitwise today's loop with
    Adam: losses, weights and buffers, optimizer state, global RNG state."""
    runs = []
    for new in (False, True):
        torch.manual_seed(0)
        m = ConvAutoencoder().to(memory_format=CL)
        gen = torch.Generator().manual_seed(0)
        if new:
            opt = make_optimizer(m.parameters(), TrainConfig("d", "s.json"))
            losses = [train_epoch(m, opt, train_images, 3, gen)["train_loss"] for _ in range(2)]
        else:
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            losses = [_pre_011_epoch(m, opt, train_images, 3, gen) for _ in range(2)]
        runs.append((losses, m, opt, torch.get_rng_state()))
    (la, a, oa, ra), (lb, b, ob, rb) = runs
    assert isinstance(ob, torch.optim.AdamW) and ob.param_groups[0]["weight_decay"] == 0.0
    assert la == lb and torch.equal(ra, rb)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert all(torch.equal(oa.state[p][k], ob.state[q][k]) for k in oa.state[p])


def test_weight_decay_reaches_adamw():
    m = ConvAutoencoder((32, 16), 4, (2, 2))
    opt = make_optimizer(m.parameters(), TrainConfig("d", "s.json", weight_decay=0.05))
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["weight_decay"] == 0.05
    assert opt.param_groups[0]["decoupled_weight_decay"] is True


# ------------------------------------------------------------------ shift + dropout


def test_shift_batch_moves_each_image_with_zero_fill():
    u8 = torch.arange(1, 2 * 5 * 4 + 1).reshape(2, 1, 5, 4).to(torch.uint8)
    torch.manual_seed(3)
    off = torch.randint(-2, 3, (2, 2))  # the draw shift_batch makes
    torch.manual_seed(3)
    out = shift_batch(u8, 2)
    assert out.dtype == torch.uint8 and out.shape == u8.shape
    for b, (dy, dx) in enumerate(off.tolist()):
        for i in range(5):
            for j in range(4):
                si, sj = i - dy, j - dx
                want = int(u8[b, 0, si, sj]) if 0 <= si < 5 and 0 <= sj < 4 else 0
                assert int(out[b, 0, i, j]) == want


def test_shift_zero_is_the_identity_without_draws():
    u8 = torch.randint(0, 255, (3, 1, 8, 6), dtype=torch.uint8)
    state = torch.get_rng_state()
    assert shift_batch(u8, 0) is u8 and torch.equal(state, torch.get_rng_state())


def test_dropout_and_shift_change_training_only(train_images):
    def epoch(obj: Objective) -> tuple[float, torch.Tensor]:
        torch.manual_seed(0)
        m = ConvAutoencoder((256, 192), 8, (4, 4, 4, 4, 4, 4))
        opt = make_optimizer(m.parameters(), TrainConfig("d", "s.json"))
        st = train_epoch(m, opt, train_images, 4, torch.Generator().manual_seed(0), objective=obj)
        assert st["train_recon"] == st["train_loss"] and st["train_kl"] is None
        return st["train_loss"], torch.get_rng_state()

    base, rng0 = epoch(Objective())
    for obj in (Objective(dropout=0.5), Objective(shift_px=4)):
        loss, rng = epoch(obj)
        assert loss != base and not torch.equal(rng, rng0)


# ------------------------------------------------------------------ fixture runs


def test_regularized_run_repeats_and_records(tmp_path):
    a = run_training(_cfg(tmp_path, "a", **REG))
    b = run_training(_cfg(tmp_path, "b", **REG))
    with (a / "history.csv").open() as fh:
        assert tuple(next(csv.reader(fh))) == HISTORY_FIELDS
    ha = _hist(a)
    assert ha == _hist(b) and len(ha) == 3
    assert all(r["train_recon"] == r["train_loss"] and r["train_kl"] == "" for r in ha)
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    objective = _metrics(a)["training"]["objective"]
    assert objective == {"mask_weight": 0.0, "beta": 0.0, "pose_weight": 0.0, **REG}  # 018
    assert _metrics(a)["model"]["variational"] is False
    assert {k: yaml.safe_load((a / "config.yaml").read_text())[k] for k in REG} == REG


def test_resume_with_every_knob_equals_uninterrupted(tmp_path):
    full = run_training(_cfg(tmp_path, "full", **ALL_KNOBS))
    d = run_training(_cfg(tmp_path, "cut", **ALL_KNOBS), stop_after_epoch=1)
    assert not (d / "metrics.json").exists()
    resume_training(d)
    assert _hist(d) == _hist(full)
    assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes()
    a, b = torch.load(d / "checkpoint.pt"), torch.load(full / "checkpoint.pt")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in b["model"])


def test_run_dir_from_before_011_resumes(tmp_path):
    """US1-2: no 011 key in config.yaml / checkpoint.pt, history rows without the new columns,
    an Adam optimizer state: the run resumes with the legacy values and equals a full run."""
    full = run_training(_cfg(tmp_path, "full"))
    d = run_training(_cfg(tmp_path, "cut"), stop_after_epoch=1)
    c = yaml.safe_load((d / "config.yaml").read_text())
    (d / "config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in c.items() if k not in NEW_KEYS})
    )
    ck = torch.load(d / "checkpoint.pt")
    ck["config"] = {k: v for k, v in ck["config"].items() if k not in NEW_KEYS}
    old = ("train_recon", "train_kl")
    ck["history"] = [{k: v for k, v in r.items() if k not in old} for r in ck["history"]]
    del ck["model_hparams"]["variational"]
    for g in ck["optimizer"]["param_groups"]:
        g["decoupled_weight_decay"] = False  # as torch.optim.Adam saved it before 011
    torch.save(ck, d / "checkpoint.pt")
    assert main(["--resume", str(d)]) == 0
    assert _hist(d) == _hist(full)
    assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes()


# ------------------------------------------------------------------ mask-weighted MSE (FR-003)


def test_recon_loss_at_zero_is_mse_loss_bitwise():
    torch.manual_seed(0)
    x = (torch.rand(3, 1, 16, 12) > 0.7) * torch.rand(3, 1, 16, 12)
    y = torch.rand(3, 1, 16, 12, requires_grad=True)
    a = recon_loss(y, x, 0.0)
    (ga,) = torch.autograd.grad(a, y)
    b = F.mse_loss(y, x)
    (gb,) = torch.autograd.grad(b, y)
    assert torch.equal(a, b) and torch.equal(ga, gb)


def test_recon_loss_weights_body_pixels():
    x = torch.tensor([[[[0.0, 0.5], [0.0, 0.0]]]], dtype=torch.float64)  # one body pixel
    y = torch.tensor([[[[0.1, 0.3], [0.0, 0.2]]]], dtype=torch.float64)
    # w = [[1, 5], [1, 1]]: (0.01 + 5 * 0.04 + 0 + 0.04) / 8
    assert float(recon_loss(y, x, 4.0)) == pytest.approx(0.25 / 8, rel=1e-12)
    body, background = x.clone(), x.clone()
    body[0, 0, 0, 1] += 0.1  # the same error on the body pixel ...
    background[0, 0, 1, 1] += 0.1  # ... and on a background pixel
    ratio = float(recon_loss(body, x, 4.0) / recon_loss(background, x, 4.0))
    assert ratio == pytest.approx(5.0, rel=1e-9)
    empty = torch.zeros_like(x)  # no body pixel: w = 1, the plain MSE
    assert float(recon_loss(y, empty, 4.0)) == pytest.approx(float(F.mse_loss(y, empty)))


def test_fixture_silhouette_is_the_nonzero_pixels():
    """R3: the renderer draws the body at >= 38 / 255 on black, the mask at the same pixels,
    so the mask weight needs no train masks."""
    n = 0
    for split in ("train", "test"):
        ds = PoseImageDataset(FIXTURE, split=split, split_file=SPLIT)
        a = load_split(ds, batch_size=4, masks=True)
        assert torch.equal(a["image"].reshape(a["n"], -1) > 0, a["mask"])
        n += a["n"]
    assert n == 16


def test_mask_weighted_run_repeats_and_records(tmp_path):
    plain = run_training(_cfg(tmp_path, "plain"))
    a = run_training(_cfg(tmp_path, "a", **MASK))
    b = run_training(_cfg(tmp_path, "b", **MASK))
    assert _hist(a) == _hist(b) and _hist(a) != _hist(plain)
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    m = _metrics(a)
    assert m["training"]["objective"]["mask_weight"] == 4.0
    assert (
        m["splits"]["test"]["image"]["ae"].keys()
        == _metrics(plain)["splits"]["test"]["image"]["ae"].keys()
    )


# ------------------------------------------------------------------ VAE (FR-007, FR-008)


def test_vae_model_shapes_parameters_and_init():
    torch.manual_seed(0)
    plain = ConvAutoencoder()
    torch.manual_seed(0)
    vae = ConvAutoencoder(variational=True)
    assert (plain.parameter_count(), vae.parameter_count()) == (3_126_897, 3_323_569)
    sp, sv = plain.state_dict(), vae.state_dict()
    assert set(sv) == set(sp) | {"logvar.weight", "logvar.bias"}
    assert all(torch.equal(sp[k], sv[k]) for k in sp)  # logvar is built last
    assert plain.hparams()["variational"] is False and vae.hparams()["variational"] is True
    x = torch.rand(2, 1, 256, 192)
    mu, logvar = vae.encode_dist(x)
    assert mu.shape == logvar.shape == (2, 64) and torch.equal(mu, vae.encode(x))
    assert torch.equal(vae(x), vae(x)) and torch.equal(vae(x), vae.decode(vae.encode(x)))
    pmu, plv = plain.encode_dist(x)
    assert plv is None and torch.equal(pmu, plain.encode(x))


def test_checkpoints_with_and_without_variational_load(tmp_path):
    vae = ConvAutoencoder((32, 16), 4, (2, 2), variational=True)
    hp = {k: v for k, v in vae.hparams().items() if k != "variational"}  # a pre-011 file
    torch.save({"model": ConvAutoencoder(**hp).state_dict(), "model_hparams": hp}, tmp_path / "a")
    old, _ = load_checkpoint(tmp_path / "a")
    assert old.variational is False and old.logvar is None
    torch.save({"model": vae.state_dict(), "model_hparams": vae.hparams()}, tmp_path / "b")
    new, _ = load_checkpoint(tmp_path / "b")
    assert new.variational and torch.equal(new.logvar.weight, vae.logvar.weight)


def test_gaussian_kl():
    torch.manual_seed(0)
    assert float(gaussian_kl(torch.randn(5, 8), torch.randn(5, 8))) > 0
    assert float(gaussian_kl(torch.zeros(3, 4), torch.zeros(3, 4))) == 0.0
    # sum over dims, mean over the batch: rows 0.5 * (1 + 1 - 1 - 0) * 2 = 1 and 0
    two = gaussian_kl(torch.tensor([[1.0, 1.0], [0.0, 0.0]]), torch.zeros(2, 2))
    assert float(two) == pytest.approx(0.5)
    lv = torch.tensor([[math.log(4.0)]])  # var 4: 0.5 * (0 + 4 - 1 - ln 4)
    assert float(gaussian_kl(torch.zeros(1, 1), lv)) == pytest.approx(0.5 * (3 - math.log(4)))


def test_vae_run_repeats_records_and_probes(tmp_path):
    a = run_training(_cfg(tmp_path, "a", **VAE))
    b = run_training(_cfg(tmp_path, "b", **VAE))
    ha = _hist(a)
    assert ha == _hist(b)
    for r in ha:
        kl = float(r["train_kl"])
        assert math.isfinite(kl) and kl > 0 and float(r["train_loss"]) > float(r["train_recon"])
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    m = _metrics(a)
    assert m["model"]["variational"] is True and m["training"]["objective"]["beta"] == 0.1
    model, _ = load_checkpoint(a / "checkpoint.pt")
    assert model.variational and model.logvar is not None
    probe = ["probe", "--config", "configs/probe_fixture.yaml", "--checkpoint"]
    assert eval_main([*probe, str(a / "checkpoint.pt"), "--run-root", str(tmp_path / "p")]) == 0
    (run,) = (tmp_path / "p").iterdir()
    encoder = json.loads((run / "metrics.json").read_text())["encoder"]
    assert encoder["variational"] is True and encoder["latent_dim"] == 8


# ------------------------------------------------------------------ experiment configs (FR-013)


@pytest.mark.parametrize("name, knobs", [("mask", MASK), ("vae", VAE_CONFIG), ("reg", REG)])
def test_experiment_configs_differ_from_the_control_only_in_their_knobs(name, knobs):
    raw = yaml.safe_load((ROOT / "configs" / "ae.yaml").read_text())
    assert {k: raw[k] for k in NEW_KEYS} == NEW_KEYS  # the control lists the defaults
    base = load_config(ROOT / "configs" / "ae.yaml").to_dict()
    cfg = load_config(ROOT / "configs" / f"ae_{name}.yaml").to_dict()
    assert {k: v for k, v in cfg.items() if v != base[k]} == {
        "run_name": f"ae-{name}-smpl_proc_v1",
        **knobs,
    }
