"""Latent probes on a frozen 004 encoder (spec 006; contract: contracts/probe-run.md).

Every image of train / val / test is encoded once (cached by checkpoint, manifest and
split hashes). Closed-form ridge and small MLP probes are fit on train latents for
root-relative 3D joints (world and camera frame), camera yaw (8-way) and SMPL pose;
their one hyper-parameter is chosen on val; val and test are then scored with the
selected probes next to mean / majority / pose-PCA baselines fit on train.
"""

import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from strike_a_pose.data import DatasetError, PoseImageDataset, parse_sample_id
from strike_a_pose.datagen.assets import sha256_file
from strike_a_pose.datagen.skeleton import SMPL24_PARENTS
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
from strike_a_pose.runlog import (
    checksums_sha256,
    environment_summary,
    git_info,
    make_run_dir,
    seed_everything,
    utc_now,
    write_json_exact,
    write_yaml,
)

from .config import ConfigError
from .metrics import ROOT_JOINT, keypoint_error_2d, mpjpe, project_points
from .pca import fit_pca
from .probe_config import ProbeConfig
from .report import FIG_BG, FIG_HEADER, FIG_PAD, figure_indices, tile_origin

PROBE_VERSION = 1
METRICS_VERSION = 1
CL = torch.channels_last
STD_FLOOR = 1e-8
POSED_JOINTS = slice(1, 22)  # SMPL joints 1-21 (root and hands are zero in the data)
UNITS = {
    "mpjpe_mm": "mm, root-relative (pelvis) 3D joint error; joints_world in the world frame "
    "(as 003), joints_cam in the camera frame",
    "kp2d_px": "px, mean over visible joints; predicted root-relative joints + true root "
    "projected with the true camera",
    "accuracy": "fraction of images whose ring camera (8 fixed yaws) is predicted exactly",
    "angle_deg": "deg, geodesic angle between predicted and true joint rotations "
    "(axis-angle via Rodrigues), mean over joints 1-21",
}
# target -> (selection metric, higher is better)
SELECT = {
    "joints_world": ("mpjpe_mm", False),
    "joints_cam": ("mpjpe_mm", False),
    "yaw": ("accuracy", True),
    "smpl_pose": ("angle_deg", False),
}
TRUE_RGB, PRED_RGB = (57, 135, 229), (235, 104, 52)  # blue = true, orange = predicted


# --------------------------------------------------------------------------- latents


def extract(ds: PoseImageDataset, encoders: dict[str, nn.Module], batch_size: int) -> dict:
    """One pass over ``ds``: latents of every encoder + the labels the probes need."""
    n = len(ds)
    a: dict = {"n": n, "latents": {}}
    for name, m in encoders.items():
        a["latents"][name] = torch.empty((n, m.latent_dim), dtype=torch.float32)
    shapes = {"joints_world": (24, 3), "joints_cam": (24, 3), "kp2d": (24, 2), "K": (3, 3),
              "R": (3, 3), "t": (3,), "pose": (72,)}  # fmt: skip
    for k, shp in shapes.items():
        a[k] = torch.empty((n, *shp), dtype=torch.float64)
    a["visible"] = torch.empty((n, 24), dtype=torch.bool)
    a["camera_index"] = torch.empty(n, dtype=torch.int64)
    a["pose_index"] = torch.empty(n, dtype=torch.int64)
    i = 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            m = b["image"].shape[0]
            sl = slice(i, i + m)
            x = b["image"].contiguous(memory_format=CL)
            for name, enc in encoders.items():
                a["latents"][name][sl] = enc.encode(x)
            a["joints_world"][sl] = b["joints_3d_world"].double()
            a["joints_cam"][sl] = b["joints_3d_cam"].double()
            a["kp2d"][sl] = b["keypoints_2d"].double()
            a["pose"][sl] = b["pose"].double()
            for k in ("K", "R", "t"):
                a[k][sl] = b["camera"][k].double()
            a["visible"][sl] = b["visibility"]
            a["camera_index"][sl] = b["camera_index"]
            a["pose_index"][sl] = b["pose_index"]
            i += m
    return a


def _cache_meta(hashes: dict, cfg: ProbeConfig, encoders: list[str], splits: list[str]) -> dict:
    return {
        "probe_version": PROBE_VERSION,
        **hashes,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "encoders": sorted(encoders),
        "splits": sorted(splits),
    }


def cache_path(cache_dir: str | Path, hashes: dict) -> Path:
    keys = ("checkpoint_sha256", "manifest_sha256", "split_sha256")
    return Path(cache_dir) / ("-".join(hashes[k][:12] for k in keys) + ".pt")


def load_or_extract(
    cfg: ProbeConfig, datasets: dict, encoders: dict, hashes: dict
) -> tuple[dict, Path | None, bool]:
    """Latents + labels per split, from the cache when every key matches."""
    meta = _cache_meta(hashes, cfg, list(encoders), list(datasets))
    path = cache_path(cfg.cache_dir, hashes) if cfg.cache_dir else None
    if path is not None and path.is_file():
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if obj.get("meta") == meta:
                return obj["splits"], path, True
        except Exception:  # corrupt / foreign file → recompute
            pass
    for m in encoders.values():
        m.eval().to(memory_format=CL)
    splits = {s: extract(ds, encoders, cfg.batch_size) for s, ds in datasets.items()}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"meta": meta, "splits": splits}, tmp)
        os.replace(tmp, path)
    return splits, path, False


# --------------------------------------------------------------------------- probes


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardizer":
        x = x.double()
        return cls(x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(STD_FLOOR))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.double() - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x.double() * self.std + self.mean


def ridge_fit(z: torch.Tensor, y: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge with intercept: ``W = (ZcᵀZc + λ·n·I)⁻¹ ZcᵀYc``, ``b = ȳ − z̄ W`` (float64)."""
    z, y = z.double(), y.double()
    n, d = z.shape
    zm, ym = z.mean(dim=0), y.mean(dim=0)
    zc, yc = z - zm, y - ym
    a = zc.T @ zc + lam * n * torch.eye(d, dtype=torch.float64)
    w = torch.linalg.solve(a, zc.T @ yc)
    return w, ym - zm @ w


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...], d_out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU()]
            d_in = h
        self.net = nn.Sequential(*layers, nn.Linear(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(
    z: torch.Tensor,
    y: torch.Tensor,
    classify: bool,
    hidden: tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    n_classes: int = 0,
) -> MLPProbe:
    """Seeded Adam (cosine lr per step); MSE on ``y`` or cross-entropy on class ids."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x = z.to(torch.float32)
    t = y if classify else y.to(torch.float32)
    model = MLPProbe(x.shape[1], hidden, n_classes if classify else t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = x.shape[0]
    total = epochs * math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, total) / total))
    )
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            out = model(x[idx])
            loss = F.cross_entropy(out, t[idx]) if classify else F.mse_loss(out, t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


# --------------------------------------------------------------------------- metrics


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues: ``[..., 3]`` axis-angle (radians) → ``[..., 3, 3]`` rotation."""
    aa = aa.double()
    theta = torch.linalg.vector_norm(aa, dim=-1, keepdim=True)
    k = aa / theta.clamp_min(1e-12)
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    km = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1)
    km = km.reshape(*aa.shape[:-1], 3, 3)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=torch.float64).expand_as(km)
    return eye + s * km + (1 - c) * (km @ km)


def rotation_angle_deg(pred_aa: torch.Tensor, true_aa: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (degrees) between rotations given as axis-angle ``[..., 3]``."""
    rp, rt = axis_angle_to_matrix(pred_aa), axis_angle_to_matrix(true_aa)
    tr = (rp.transpose(-1, -2) @ rt).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0)))


def yaw_scores(pred: torch.Tensor, true: torch.Tensor, n_classes: int) -> dict:
    """Accuracy, confusion (rows true, cols predicted) and errors by ring offset."""
    pred, true = pred.long(), true.long()
    conf = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    conf.index_put_((true, pred), torch.ones_like(true), accumulate=True)
    d = (pred - true) % n_classes
    steps = torch.minimum(d, n_classes - d)
    step_deg = 360.0 / n_classes
    by_offset = {f"{s * step_deg:g}": int((steps == s).sum()) for s in range(1, n_classes // 2 + 1)}
    return {
        "accuracy": float((pred == true).double().mean()) if len(true) else float("nan"),
        "confusion": conf.tolist(),
        "errors_by_offset_deg": by_offset,
    }


def _rel(j: torch.Tensor) -> torch.Tensor:
    return (j - j[:, ROOT_JOINT : ROOT_JOINT + 1]).reshape(j.shape[0], -1)


def target_values(a: dict, target: str) -> torch.Tensor:
    if target == "joints_world":
        return _rel(a["joints_world"])
    if target == "joints_cam":
        return _rel(a["joints_cam"])
    if target == "yaw":
        return a["camera_index"]
    return a["pose"]


def project_pred(a: dict, target: str, pred: torch.Tensor) -> torch.Tensor:
    """Predicted root-relative joints + true root → pixels with the true camera."""
    n = pred.shape[0]
    p = pred.double().reshape(n, -1, 3)
    if target == "joints_world":
        root = a["joints_world"][:, ROOT_JOINT : ROOT_JOINT + 1]
        return project_points(p + root, a["K"], a["R"], a["t"])
    root = a["joints_cam"][:, ROOT_JOINT : ROOT_JOINT + 1]
    eye = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
    return project_points(p + root, a["K"], eye, torch.zeros(n, 3, dtype=torch.float64))


def score(target: str, pred: torch.Tensor, a: dict, n_classes: int) -> dict:
    """Score predictions (regression: ``[n, D]``; yaw: class ids ``[n]``) on split ``a``."""
    n = a["n"]
    if target == "yaw":
        return yaw_scores(pred, a["camera_index"], n_classes)
    if target == "smpl_pose":
        p, t = pred.double().reshape(n, 24, 3), a["pose"].reshape(n, 24, 3)
        return {
            "angle_deg": float(rotation_angle_deg(p[:, POSED_JOINTS], t[:, POSED_JOINTS]).mean())
        }
    t = target_values(a, target).reshape(n, -1, 3)
    e3 = mpjpe(pred.double().reshape(n, -1, 3), t) * 1000.0
    e2 = keypoint_error_2d(project_pred(a, target, pred), a["kp2d"], a["visible"])
    ok = ~torch.isnan(e2)
    return {
        "mpjpe_mm": float(e3.mean()),
        "kp2d_px": float(e2[ok].mean()) if ok.any() else float("nan"),
    }


# --------------------------------------------------------------------------- selection

Predictor = Callable[[torch.Tensor], torch.Tensor]  # latents [n, d] → predictions


def _select(
    target: str, candidates: dict[str, Predictor], val: dict | None, n_classes: int
) -> tuple[str, dict]:
    """Pick the candidate with the best val metric (ties → first; empty val → first)."""
    metric, higher = SELECT[target]
    first = next(iter(candidates))
    if val is None or val["n"] == 0:
        return first, {"value": float(first), "selected_by": "default (empty val)", "val": {}}
    vals = {k: score(target, fn(val["z"]), val, n_classes)[metric] for k, fn in candidates.items()}
    best = first
    for k, v in vals.items():
        if (v > vals[best]) if higher else (v < vals[best]):
            best = k
    return best, {"value": float(best), "selected_by": f"val {metric}", "val": vals}


def fit_probes(
    cfg: ProbeConfig, target: str, train: dict, val: dict | None, n_classes: int
) -> dict[str, tuple[Predictor, dict]]:
    """Ridge + MLP for one target on standardized latents ``train["z"]``."""
    y = target_values(train, target)
    classify = target == "yaw"
    out = {}

    ridge = {}
    y_r = F.one_hot(y, n_classes).double() if classify else y
    for lam in cfg.ridge_lambdas:
        w, b = ridge_fit(train["z"], y_r, lam)
        if classify:
            ridge[str(lam)] = lambda z, w=w, b=b: (z.double() @ w + b).argmax(dim=1)
        else:
            ridge[str(lam)] = lambda z, w=w, b=b: z.double() @ w + b
    k, sel = _select(target, ridge, val, n_classes)
    out["ridge"] = (ridge[k], sel | {"param": "lambda"})

    ys = None if classify else Standardizer.fit(y)
    mlp = {}
    for wd in cfg.mlp_weight_decays:
        m = fit_mlp(
            train["z"], y if classify else ys(y), classify, cfg.mlp_hidden, cfg.mlp_epochs,
            cfg.mlp_batch_size, cfg.mlp_lr, wd, cfg.seed, n_classes,
        )  # fmt: skip
        if classify:
            mlp[str(wd)] = lambda z, m=m: _no_grad(m, z).argmax(dim=1)
        else:
            mlp[str(wd)] = lambda z, m=m: ys.inverse(_no_grad(m, z))
    k, sel = _select(target, mlp, val, n_classes)
    out["mlp"] = (mlp[k], sel | {"param": "weight_decay"})
    return out


def _no_grad(m: nn.Module, z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return m(z.to(torch.float32))


def baselines(cfg: ProbeConfig, target: str, train: dict, n_classes: int) -> dict:
    """Mean / majority (+ pose PCA of the true joints) predictors on labels, not latents.

    Each maps a split dict to predictions: PCA reconstructs the split's *true* joints
    (oracle reference), the others ignore their input.
    """
    y = target_values(train, target)
    if target == "yaw":
        major = int(torch.bincount(y, minlength=n_classes).argmax())
        return {"majority": (lambda a: torch.full((a["n"],), major), None)}
    mean = y.mean(dim=0)
    out = {"mean_pose": (lambda a: mean.expand(a["n"], -1), None)}
    if target in ("joints_world", "joints_cam"):
        pca = fit_pca(y, max(cfg.pose_pca_ks))
        for k in cfg.pose_pca_ks:
            out[f"pca_k{k}"] = (
                lambda a, k=k: pca.reconstruct(target_values(a, target), k),
                {"k": k, "k_effective": pca.k_effective(k)},
            )
    return out


# --------------------------------------------------------------------------- figures


def _bones(draw: ImageDraw.ImageDraw, kp: torch.Tensor, ox: int, oy: int, rgb, width: int):
    for j, p in enumerate(SMPL24_PARENTS):
        if p < 0:
            continue
        (x0, y0), (x1, y1) = kp[p].tolist(), kp[j].tolist()
        draw.line((ox + x0, oy + y0, ox + x1, oy + y1), fill=rgb, width=width)
    r = width
    for x, y in kp.tolist():
        draw.ellipse((ox + x - r, oy + y - r, ox + x + r, oy + y + r), fill=rgb)


def skeleton_figure(
    images: torch.Tensor, true_kp: torch.Tensor, preds: list[tuple[str, torch.Tensor]]
) -> Image.Image:
    """Rows = samples; columns: true skeleton, then one column per predicted skeleton
    (true in thin blue underneath, prediction in orange), drawn on the input image."""
    n, _, h, w = images.shape
    nc = 1 + len(preds)
    size = (nc * w + (nc + 1) * FIG_PAD, FIG_HEADER + n * h + (n + 1) * FIG_PAD)
    fig = Image.new("RGB", size, (FIG_BG,) * 3)
    draw = ImageDraw.Draw(fig)
    labels = ["true (blue)"] + [f"{name} (orange)" for name, _ in preds]
    for c, label in enumerate(labels):
        draw.text((tile_origin(0, c, (h, w))[0] + 2, 2), label, fill=(255, 255, 255))
    for r in range(n):
        grey = Image.fromarray((images[r, 0].clamp(0, 1) * 255).round().byte().numpy())
        for c in range(nc):
            ox, oy = tile_origin(r, c, (h, w))
            fig.paste(grey.convert("RGB"), (ox, oy))
            if c == 0:
                _bones(draw, true_kp[r], ox, oy, TRUE_RGB, 2)
            else:
                _bones(draw, true_kp[r], ox, oy, TRUE_RGB, 1)
                _bones(draw, preds[c - 1][1][r], ox, oy, PRED_RGB, 2)
    return fig


CELL, MARGIN, TITLE = 40, 56, 34
SURFACE, INK, LOW, HIGH = (252, 252, 251), (11, 11, 11), (240, 239, 236), (16, 66, 129)


def confusion_figure(panels: list[tuple[str, list[list[int]]]], labels: list[str]) -> Image.Image:
    """Side-by-side count grids (rows true, columns predicted), shaded by row share."""
    k = len(labels)
    pw = MARGIN + k * CELL + FIG_PAD * 4
    ph = TITLE + MARGIN // 2 + k * CELL + 24
    fig = Image.new("RGB", (pw * len(panels), ph), SURFACE)
    draw = ImageDraw.Draw(fig)
    for p, (title, conf) in enumerate(panels):
        x0, y0 = p * pw + MARGIN, TITLE + MARGIN // 2
        draw.text((p * pw + 6, 4), title, fill=INK)
        draw.text((p * pw + 6, 18), "rows: true yaw; columns: predicted", fill=INK)
        for i, lab in enumerate(labels):
            draw.text((x0 + i * CELL + 4, y0 - 14), lab, fill=INK)
            draw.text((p * pw + 6, y0 + i * CELL + CELL // 2 - 6), lab, fill=INK)
        for i, row in enumerate(conf):
            tot = max(sum(row), 1)
            for j, v in enumerate(row):
                f = v / tot
                rgb = tuple(round(lo + f * (hi - lo)) for lo, hi in zip(LOW, HIGH, strict=True))
                box = (
                    x0 + j * CELL,
                    y0 + i * CELL,
                    x0 + (j + 1) * CELL - 2,
                    y0 + (i + 1) * CELL - 2,
                )
                draw.rectangle(box, fill=rgb)
                if v:
                    tc = SURFACE if f > 0.45 else INK
                    draw.text((box[0] + 4, box[1] + CELL // 2 - 7), str(v), fill=tc)
    return fig


# --------------------------------------------------------------------------- run


def _encoders(model: ConvAutoencoder, cfg: ProbeConfig) -> dict[str, nn.Module]:
    enc = {"trained": model}
    if cfg.random_encoder_control:
        torch.manual_seed(cfg.seed)
        hp = model.hparams()
        enc["random"] = ConvAutoencoder(hp["image_size"], hp["latent_dim"], hp["channels"]).eval()
    return enc


def _checkpoint_info(path: Path, ckpt: dict, sha: str) -> dict:
    info = {"path": str(path), "sha256": sha, "epoch": ckpt.get("epoch"), "train_run_git": None}
    ri = path.parent / "run_info.json"
    if ri.is_file():
        try:
            info["train_run_git"] = json.loads(ri.read_text()).get("git")
        except ValueError:
            pass
    return info


def run_probe(
    cfg: ProbeConfig, argv: list[str] | None = None, log: Callable[[str], None] | None = None
) -> Path:
    """Check inputs, extract latents, fit + select + score probes, write the run dir.

    Raises ``DatasetError`` (bad dataset / split) or ``ConfigError`` (missing or
    incompatible checkpoint) before creating anything; ``FileExistsError`` if the run
    directory exists.
    """
    say = log or (lambda _msg: None)
    t0 = time.monotonic()
    started = utc_now()
    seed_everything(cfg.seed)
    kw = {"split_file": cfg.split_file}
    datasets = {
        s: PoseImageDataset(cfg.dataset_dir, split=s, **kw) for s in ("train", *cfg.eval_splits)
    }
    train_ds = datasets["train"]
    if len(train_ds) == 0:
        raise DatasetError("train split has no samples")
    ck_path = Path(cfg.checkpoint)
    if not ck_path.is_file():
        raise ConfigError(f"checkpoint: no such file {ck_path}")
    try:
        model, ckpt = load_checkpoint(ck_path)
    except Exception as e:  # not a 004 checkpoint
        raise ConfigError(f"checkpoint: cannot load {ck_path}: {e}") from e
    if tuple(model.image_size) != tuple(train_ds.image_size):
        raise ConfigError(
            f"checkpoint image size {model.image_size} != dataset {train_ds.image_size}"
        )
    hashes = {
        "checkpoint_sha256": sha256_file(ck_path),
        "manifest_sha256": train_ds.manifest_sha256,
        "split_sha256": sha256_file(Path(cfg.split_file)),
    }
    classes = list(train_ds.cameras)
    n_classes = len(classes)

    run_dir = make_run_dir(cfg.run_root, cfg.run_name, started)
    say(f"run dir {run_dir}")
    write_yaml(run_dir / "config.yaml", cfg.to_dict())
    info = {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "seed": cfg.seed,
        "git": git_info(),
        "env": environment_summary(),
        "dataset": {
            "dir": cfg.dataset_dir,
            "manifest_sha256": hashes["manifest_sha256"],  # cache key (raw, spec 007)
            "manifest_content_sha256": train_ds.manifest_content_sha256,
            "checksums_sha256": checksums_sha256(cfg.dataset_dir),  # pixels (spec 012)
        },
        "split_file": {"path": cfg.split_file, "sha256": hashes["split_sha256"]},
        "checkpoint": _checkpoint_info(ck_path, ckpt, hashes["checkpoint_sha256"]),
        "started_utc": started.isoformat(),
        "finished_utc": None,
        "wall_s": None,
    }
    write_json_exact(run_dir / "run_info.json", info)

    te = time.monotonic()
    encoders = _encoders(model, cfg)
    data, cpath, hit = load_or_extract(cfg, datasets, encoders, hashes)
    encode_s = time.monotonic() - te
    say(f"latents {'from cache' if hit else 'encoded'} in {encode_s:.0f} s ({cpath})")

    selection: dict = {}
    results: dict = {s: {} for s in cfg.eval_splits}
    fig_preds: dict = {}
    for enc in encoders:
        prefix = "" if enc == "trained" else f"{enc}_"
        zs = Standardizer.fit(data["train"]["latents"][enc])
        for a in data.values():
            a["z"] = zs(a["latents"][enc])
            if enc == "trained":
                a["z_trained"] = a["z"]
        val = data["val"]
        for target in cfg.targets:
            tp = time.monotonic()
            for name, (fn, sel) in fit_probes(cfg, target, data["train"], val, n_classes).items():
                selection.setdefault(prefix + name, {})[target] = sel
                for s in cfg.eval_splits:
                    a = data[s]
                    if a["n"]:
                        res = score(target, fn(a["z"]), a, n_classes)
                        results[s].setdefault(target, {})[prefix + name] = res
                        if s == cfg.figure_split and not prefix:
                            fig_preds[(target, name)] = fn
            say(f"{enc} {target}: probes fit + scored ({time.monotonic() - tp:.0f} s)")
    for target in cfg.targets:
        for name, (fn, extra) in baselines(cfg, target, data["train"], n_classes).items():
            for s in cfg.eval_splits:
                a = data[s]
                if a["n"]:
                    r = score(target, fn(a), a, n_classes)
                    results[s].setdefault(target, {})[name] = r | (extra or {})

    splits = {}
    for s in cfg.eval_splits:
        a = data[s]
        out: dict = {
            "samples": a["n"],
            "poses": int(a["pose_index"].unique().numel()) if a["n"] else 0,
        }
        if a["n"]:
            out["targets"] = results[s]
        splits[s] = out

    figures = _write_figures(cfg, run_dir, datasets, data, fig_preds, results, classes)
    metrics = {
        "probe_version": PROBE_VERSION,
        "metrics_version": METRICS_VERSION,
        "dataset": {
            "name": train_ds.manifest["dataset_name"],
            "manifest_sha256": hashes["manifest_sha256"],
            "split_file_sha256": hashes["split_sha256"],
        },
        "encoder": {
            "checkpoint_sha256": hashes["checkpoint_sha256"],
            "epoch": ckpt.get("epoch"),
            "latent_dim": model.latent_dim,
            "variational": model.variational,  # spec 011: a VAE is probed through mu
            "random_encoder_control": cfg.random_encoder_control,
        },
        "fit": {
            "split": "train",
            "samples": len(train_ds),
            "poses": len({parse_sample_id(i)[0] for i in train_ds.sample_ids}),
        },
        "units": UNITS,
        "classes": classes,
        "selection": selection,
        "splits": splits,
    }
    write_json_exact(run_dir / "metrics.json", metrics)
    info |= {
        "cache": {"path": str(cpath) if cpath else None, "hit": hit},
        "encode_s": round(encode_s, 1),
        "figures": figures,
        "finished_utc": utc_now().isoformat(),
        "wall_s": round(time.monotonic() - t0, 1),
    }
    write_json_exact(run_dir / "run_info.json", info)
    return run_dir


def _write_figures(cfg, run_dir, datasets, data, fig_preds, results, classes) -> list[str]:
    s = cfg.figure_split
    a = data[s]
    if not a["n"]:
        return []
    out = []
    jt = next((t for t in ("joints_world", "joints_cam") if t in cfg.targets), None)
    if jt is not None:
        idx = figure_indices(a["n"], cfg.figure_samples)
        ds = datasets[s]
        imgs = torch.stack([ds[i]["image"] for i in idx])
        sub = {k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in a.items()}
        sub |= {"n": len(idx), "z": a["z_trained"][idx]}
        preds = [
            (name, project_pred(sub, jt, fig_preds[(jt, name)](sub["z"])))
            for name in ("ridge", "mlp")
        ]
        fn = f"skeleton_{s}.png"
        skeleton_figure(imgs, sub["kp2d"], preds).save(run_dir / fn, format="PNG")
        out.append(fn)
    if "yaw" in cfg.targets:
        labels = [f"{float(c.rsplit('_', 1)[-1]):g}" for c in classes]
        panels = [
            (f"{name} yaw probe, {s} (acc {results[s]['yaw'][name]['accuracy']:.3f})",
             results[s]["yaw"][name]["confusion"])
            for name in ("ridge", "mlp")
        ]  # fmt: skip
        fn = f"yaw_confusion_{s}.png"
        confusion_figure(panels, labels).save(run_dir / fn, format="PNG")
        out.append(fn)
    return out
