"""VAE + pose head / gradient loss (spec 021 FR-001..FR-003)."""

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from strike_a_pose.data import PoseImageDataset, make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.cli import main as eval_main
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.metrics import gradient_loss
from strike_a_pose.models import ConvAutoencoder, PoseHead, load_checkpoint
from strike_a_pose.train import trainer
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import TrainConfig, config_from_dict, load_config
from strike_a_pose.train.objective import (
    Objective,
    gaussian_kl,
    recon_loss,
    reparameterize,
    shift_batch,
)
from strike_a_pose.train.trainer import (
    CL,
    PoseBatch,
    load_split,
    make_optimizer,
    pose_loss,
    run_training,
    to_float,
    train_epoch,
)

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}
VAE = {"variational": True, "beta": 0.1}


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


# frozen pre-021 loop (spec 021 FR-003); never edit
def _train_epoch_pre_021(
    model: ConvAutoencoder,
    opt: torch.optim.Optimizer,
    images: torch.Tensor,
    batch_size: int,
    gen: torch.Generator,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
    objective: Objective | None = None,
    pose: PoseBatch | None = None,
) -> dict:
    """One shuffled pass (last batch may be smaller) under ``objective`` (spec 011; None:
    the plain AE, today's computation). Returns the epoch means of the loss
    (``train_loss``), of its reconstruction term (``train_recon``) and of the KL term
    (``train_kl``, None unless variational), of the pose term (``train_pose``, None unless
    ``pose`` is given: spec 018, loss = recon + ``pose_weight``·pose), of the gradient term
    (``train_grad``, None unless ``gradient_weight`` > 0: spec 020, loss = recon +
    ``gradient_weight``·grad [+ pose]) and the last lr used
    (``lr``)."""
    obj = objective or Objective()
    model.train()
    if pose is not None:
        pose.head.train()
    n = images.shape[0]
    perm = torch.randperm(n, generator=gen)
    total = recon_total = 0.0
    kl_total: float | None = None
    pose_total: float | None = None if pose is None else 0.0
    grad_total: float | None = None if not obj.gradient_weight else 0.0
    lr = float("nan")
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        x = to_float(shift_batch(images[idx], obj.shift_px)).contiguous(memory_format=CL)
        opt.zero_grad(set_to_none=True)
        mu, logvar = model.encode_dist(x)  # (encoder(x), None) for the plain AE
        z = mu if logvar is None else reparameterize(mu, logvar)
        if obj.dropout:  # bottleneck dropout, training only (p = 0: no call, no draw)
            z = F.dropout(z, obj.dropout, training=True)
        y = model.decode(z)
        recon = recon_loss(y, x, obj.mask_weight)
        if logvar is None:
            loss = recon
            if obj.gradient_weight:  # spec 020: recon + gamma * gradient term
                g = gradient_loss(y, x)
                loss = recon + obj.gradient_weight * g
                grad_total += float(g.detach()) * len(idx)
            if pose is not None:  # spec 018: the head reads mu, before dropout
                pl = pose_loss(pose.head(mu), pose.targets[idx], pose.active)
                loss = loss + obj.pose_weight * pl
                pose_total += float(pl.detach()) * len(idx)
        else:  # KL in per-pixel units: (SSE + beta·KL) / (H·W) per image (spec 011)
            kl = gaussian_kl(mu, logvar)
            loss = recon + obj.beta * kl / (x.shape[2] * x.shape[3])
            kl_total = (kl_total or 0.0) + float(kl.detach()) * len(idx)
        loss.backward()
        lr = float(opt.param_groups[0]["lr"])
        opt.step()
        if sched is not None:
            sched.step()
        total += float(loss.detach()) * len(idx)
        recon_total += float(recon.detach()) * len(idx)
    kl_mean = None if kl_total is None else kl_total / n
    pose_mean = None if pose_total is None else pose_total / n
    grad_mean = None if grad_total is None else grad_total / n
    return {
        "train_loss": total / n,
        "train_recon": recon_total / n,
        "train_kl": kl_mean,
        "train_pose": pose_mean,
        "train_grad": grad_mean,
        "lr": lr,
    }


def _setup(variational, with_pose, image_size):
    torch.manual_seed(0)
    m = ConvAutoencoder(image_size, 8, [4] * 6, variational).to(memory_format=CL)
    pose = None
    params = list(m.parameters())
    if with_pose:
        head = PoseHead(8)
        n_targets = 72
        pose = (head, n_targets)
        params += list(head.parameters())
    opt = make_optimizer(params, TrainConfig("d", "s.json"))
    return m, pose, opt


@pytest.mark.parametrize(
    ("variational", "objective_kw", "with_pose"),
    [
        (False, {}, False),
        (True, {"beta": 0.1}, False),
        (False, {"pose_weight": 0.03}, True),
        (False, {"gradient_weight": 0.5}, False),
        (False, {"gradient_weight": 0.5, "pose_weight": 0.03}, True),
        (False, {"mask_weight": 4.0}, False),
    ],
)
def test_train_epoch_matches_pre_021(train_images, variational, objective_kw, with_pose):
    n = train_images.shape[0]
    image_size = tuple(train_images.shape[-2:])
    targets = torch.randn(n, 72, generator=torch.Generator().manual_seed(3))
    active = torch.ones(72, dtype=torch.bool)
    runs = []
    for fn in (train_epoch, _train_epoch_pre_021):
        m, hp, opt = _setup(variational, with_pose, image_size)
        pose = None if hp is None else PoseBatch(hp[0], targets, active)
        gen = torch.Generator().manual_seed(1)
        stats = []
        for _ in range(2):
            torch.manual_seed(5)
            stats.append(
                fn(m, opt, train_images, 3, gen, objective=Objective(**objective_kw), pose=pose)
            )
        runs.append((stats, m, hp))
    (sa, a, ha), (sb, b, hb) = runs
    assert sa == sb
    assert all(
        torch.equal(p, q)
        for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )
    if with_pose:
        assert all(
            torch.equal(p, q)
            for p, q in zip(ha[0].state_dict().values(), hb[0].state_dict().values(), strict=True)
        )


def test_variational_accepts_pose_and_gradient():
    base = {"dataset_dir": "d", "split_file": "s", "variational": True, "beta": 0.1}
    assert config_from_dict({**base, "pose_weight": 0.03}).pose_weight == 0.03
    assert config_from_dict({**base, "gradient_weight": 0.5}).gradient_weight == 0.5
    with pytest.raises(ConfigError, match="mask_weight"):
        config_from_dict({**base, "gradient_weight": 0.5, "mask_weight": 4.0})


def test_vae_pose_head_reads_mu(train_images, monkeypatch):
    seen_in: list[torch.Tensor] = []
    mus: list[torch.Tensor] = []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = PoseHead(8)

        def forward(self, x):
            seen_in.append(x.detach().clone())
            return self.inner(x)

    def fake_reparameterize(mu, logvar):
        mus.append(mu.detach().clone())
        return mu + 100.0

    monkeypatch.setattr(trainer, "reparameterize", fake_reparameterize)
    n = train_images.shape[0]
    image_size = tuple(train_images.shape[-2:])
    torch.manual_seed(0)
    m = ConvAutoencoder(image_size, 8, [4] * 6, True).to(memory_format=CL)
    spy = Spy()
    pose = PoseBatch(spy, torch.randn(n, 72), torch.ones(72, dtype=torch.bool))
    opt = make_optimizer([*m.parameters(), *spy.parameters()], TrainConfig("d", "s.json"))
    st = train_epoch(
        m, opt, train_images, n, torch.Generator().manual_seed(0),
        objective=Objective(beta=0.1, pose_weight=0.03), pose=pose,
    )  # fmt: skip
    assert st["train_pose"] is not None and st["train_kl"] is not None
    assert len(seen_in) == len(mus) == 1
    assert torch.equal(seen_in[0], mus[0])


def _image_hw() -> int:
    ds = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    h, w = ds.image_size
    return h * w


def _vae_pose_run(tmp_path, root="a", **kw):
    return run_training(_cfg(tmp_path, root, **VAE, pose_weight=0.03, **kw))


def test_vae_pose_run_records(tmp_path, varied):
    d = _vae_pose_run(tmp_path)
    hw = _image_hw()
    for r in _rows(d):
        kl, pose = float(r["train_kl"]), float(r["train_pose"])
        assert math.isfinite(kl) and kl > 0 and math.isfinite(pose) and pose > 0
        assert r["train_grad"] == ""
        want = float(r["train_recon"]) + 0.1 * kl / hw + 0.03 * pose
        assert float(r["train_loss"]) == pytest.approx(want, rel=1e-6)
    m = _metrics(d)
    assert math.isfinite(m["splits"]["val"]["pose_head"]["mpjpe_mm"])
    assert m["training"]["objective"]["beta"] == 0.1
    assert m["training"]["objective"]["pose_weight"] == 0.03
    assert "pose_head" in torch.load(d / "best.pt")
    model, _ = load_checkpoint(d / "best.pt")
    assert model.variational is True


def test_vae_pose_checkpoint_probes_and_samples(tmp_path, varied):
    d = _vae_pose_run(tmp_path)
    best = d / "best.pt"
    ae = run_training(_cfg(tmp_path, "ae"))
    code = eval_main(
        ["probe", "--config", "configs/probe_fixture.yaml", "--checkpoint", str(best),
         "--run-root", str(tmp_path / "p")]
    )  # fmt: skip
    assert code == 0
    code = eval_main(
        ["sample", "--config", "configs/sample_fixture.yaml", "--checkpoint", str(best),
         "--reference-checkpoint", str(ae / "best.pt"), "--run-root", str(tmp_path / "s")]
    )  # fmt: skip
    assert code == 0
    (run,) = (tmp_path / "s").iterdir()
    r = json.loads((run / "samples.json").read_text())
    assert 0.0 <= r["sets"]["vae_prior"]["valid_fraction"] <= 1.0


def test_vae_pose_run_repeats(tmp_path, varied):
    a, b = _vae_pose_run(tmp_path, "a"), _vae_pose_run(tmp_path, "b")
    assert _rows(a) == _rows(b)
    assert _metrics(a)["splits"]["val"]["pose_head"] == _metrics(b)["splits"]["val"]["pose_head"]


def test_vae_pose_resume_equals_uninterrupted(tmp_path, varied):
    full = _vae_pose_run(tmp_path, "full")
    d = run_training(_cfg(tmp_path, "cut", **VAE, pose_weight=0.03), stop_after_epoch=1)
    assert main(["--resume", str(d)]) == 0
    assert _rows(d) == _rows(full)
    ha, hb = torch.load(d / "best.pt")["pose_head"], torch.load(full / "best.pt")["pose_head"]
    assert all(torch.equal(ha[k], hb[k]) for k in hb)
    assert _metrics(d)["splits"]["val"]["pose_head"] == _metrics(full)["splits"]["val"]["pose_head"]


def test_vae_grad_run_composes(tmp_path):
    d = run_training(_cfg(tmp_path, "g", **VAE, gradient_weight=0.5))
    hw = _image_hw()
    for r in _rows(d):
        want = float(r["train_recon"]) + 0.5 * float(r["train_grad"])
        want += 0.1 * float(r["train_kl"]) / hw
        assert float(r["train_loss"]) == pytest.approx(want, rel=1e-6)
