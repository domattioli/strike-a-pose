"""Mean / PCA baselines for images and root-relative 3D joints (spec 003).

Fit on the train split only; score the configured eval splits; write a run directory
(contract: specs/003-baselines-and-metrics/contracts/eval-run.md).
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from strike_a_pose.data import PoseImageDataset, parse_sample_id
from strike_a_pose.datagen.assets import sha256_file
from strike_a_pose.runlog import (
    checksums_sha256,
    environment_summary,
    git_info,
    make_run_dir,
    peak_rss_mb,
    seed_everything,
    utc_now,
    write_json_exact,
    write_yaml,
)

from .config import BaselineConfig
from .metrics import ROOT_JOINT, keypoint_error_2d, mpjpe, project_points
from .pca import PCA, fit_pca
from .report import figure_indices, grid_figure, region_definitions, score_images

METRICS_VERSION = 1
UNITS = {
    "image_mse": "per-pixel mean squared error, images in [0, 1] (PCA output clamped to [0, 1])",
    "psnr_db": "dB, 10*log10(1/max(MSE, 1e-10)) per image, averaged over images",
    "silhouette_iou": "IoU of (prediction > iou_threshold) vs true body mask, per image mean",
    "mpjpe_mm": "mm, root-relative (pelvis) 3D joint error, world frame (= camera frame)",
    "kp2d_px": "px, mean over visible joints; prediction + true root projected with true camera",
    "regions": (
        "per-pixel mean squared error (images in [0, 1]) over every pixel of the region, body "
        "and background, per image; mean over the images whose region is non-empty (samples); "
        "a region (top-level regions) is the union of the squares of side 2*half_size_px+1 px "
        "centred on the pixels of its visible joints, clipped to the image"
    ),
}


@dataclass(frozen=True)
class Baselines:
    mean_image: torch.Tensor  # [H*W] float32
    image_pca: PCA | None  # None when image_pca_ks is empty (spec 010 FR-014)
    mean_pose: torch.Tensor  # [J*3] float64, root-relative
    pose_pca: PCA


TRAIN_IMAGE_DTYPE_NO_PCA = torch.uint8  # spec 015 FR-004: 2.5 GB, not 10 GB, at 51,200 images


def load_arrays(
    ds: PoseImageDataset,
    batch_size: int,
    masks: bool = True,
    image_dtype: torch.dtype = torch.float32,
) -> dict:
    """Read a whole dataset (split) into preallocated tensors (images flattened): float32 in
    [0, 1], or with ``image_dtype=torch.uint8`` the lossless ``round(x * 255)`` (pixels are
    ``k/255``; spec 015 FR-004, research R1), which ``image_mean`` accepts as is."""
    n = len(ds)
    h, w = ds.image_size
    a: dict = {"n": n, "image_size": (h, w)}
    a["image"] = torch.empty((n, h * w), dtype=image_dtype)
    if masks:
        a["mask"] = torch.empty((n, h * w), dtype=torch.bool)
    a["pose_index"] = torch.empty(n, dtype=torch.int64)
    keys3 = {"joints": (24, 3), "kp2d": (24, 2), "K": (3, 3), "R": (3, 3), "t": (3,)}
    for k, shp in keys3.items():
        a[k] = torch.empty((n, *shp), dtype=torch.float64)
    a["visible"] = torch.empty((n, 24), dtype=torch.bool)
    i = 0
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        m = b["image"].shape[0]
        sl = slice(i, i + m)
        x = b["image"].reshape(m, -1)
        a["image"][sl] = torch.round(x * 255.0).to(torch.uint8) if image_dtype == torch.uint8 else x
        if masks:
            a["mask"][sl] = b["mask"].reshape(m, -1)
        a["pose_index"][sl] = b["pose_index"]
        a["joints"][sl] = b["joints_3d_world"].double()
        a["kp2d"][sl] = b["keypoints_2d"].double()
        a["visible"][sl] = b["visibility"]
        for k in ("K", "R", "t"):
            a[k][sl] = b["camera"][k].double()
        i += m
    return a


def _rel_flat(joints: torch.Tensor) -> torch.Tensor:
    rel = joints - joints[:, ROOT_JOINT : ROOT_JOINT + 1]
    return rel.reshape(rel.shape[0], -1)


MEAN_BATCH = 1024  # rows per float64 partial sum (spec 012)


def image_mean(rows: torch.Tensor, batch_size: int = MEAN_BATCH) -> torch.Tensor:
    """Per-pixel mean of image rows ``[n, P]`` (uint8, scaled ``/255`` per batch like the 004
    trainer's ``to_float``, or float32) as float32, accumulated in float64 batch by batch: no
    float64 copy of the whole matrix (spec 012: 10 GB for 25,600 images). Bit-identical to
    ``rows.sum(0, dtype=float64) / n``: the values are multiples of 2^-31 in [0, 1], so every
    float64 partial sum of fewer than 2^22 rows is exact (012 research R4)."""
    n = rows.shape[0]
    acc = torch.zeros(rows.shape[1], dtype=torch.float64)
    for i in range(0, n, batch_size):
        b = rows[i : i + batch_size]
        if b.dtype == torch.uint8:
            b = b.to(torch.float32) / 255.0
        acc += b.to(torch.float64).sum(dim=0)
    return (acc / n).to(torch.float32)


def fit_baselines(train: dict, cfg: BaselineConfig) -> Baselines:
    x = train["image"]
    mean_image = image_mean(x)
    image_pca = fit_pca(x, max(cfg.image_pca_ks)) if cfg.image_pca_ks else None
    rel = _rel_flat(train["joints"])
    return Baselines(mean_image, image_pca, rel.mean(dim=0), fit_pca(rel, max(cfg.pose_pca_ks)))


def _image_predictors(b: Baselines, cfg: BaselineConfig):
    yield "mean_image", None, lambda x: b.mean_image.expand(x.shape[0], -1)
    for k in cfg.image_pca_ks:
        yield f"pca_k{k}", k, lambda x, k=k: b.image_pca.reconstruct(x, k).clamp(0.0, 1.0)


def _pose_predictors(b: Baselines, cfg: BaselineConfig):
    yield "mean_pose", None, lambda r: b.mean_pose.expand(r.shape[0], -1)
    for k in cfg.pose_pca_ks:
        yield f"pca_k{k}", k, lambda r, k=k: b.pose_pca.reconstruct(r, k)


def score_split(b: Baselines, a: dict, cfg: BaselineConfig) -> dict:
    n = a["n"]
    out: dict = {"samples": n, "poses": int(a["pose_index"].unique().numel()) if n else 0}
    if n == 0:
        return out
    img: dict = {}
    labels = (a["kp2d"], a["visible"], a["image_size"])  # region metrics (spec 009)
    for name, k, pred in _image_predictors(b, cfg):
        r = score_images(
            pred, a["image"], a["mask"], cfg.batch_size, cfg.iou_threshold, labels=labels
        )
        if k is not None:
            r |= {"k": k, "k_effective": b.image_pca.k_effective(k)}
        img[name] = r
    pose: dict = {}
    joints = a["joints"]
    rel = _rel_flat(joints)
    root = joints[:, ROOT_JOINT : ROOT_JOINT + 1]
    for name, k, pred in _pose_predictors(b, cfg):
        p_rel = pred(rel).reshape(n, -1, 3)
        e3 = mpjpe(p_rel, rel.reshape(n, -1, 3)) * 1000.0
        kp = project_points(p_rel + root, a["K"], a["R"], a["t"])
        e2 = keypoint_error_2d(kp, a["kp2d"], a["visible"])
        ok = ~torch.isnan(e2)
        r = {
            "mpjpe_mm": float(e3.mean()),
            "kp2d_px": float(e2[ok].mean()) if ok.any() else float("nan"),
            "kp2d_samples": int(ok.sum()),
        }
        if k is not None:
            r |= {"k": k, "k_effective": b.pose_pca.k_effective(k)}
        pose[name] = r
    out["image"], out["pose"] = img, pose
    return out


def recon_figure(b: Baselines, a: dict, cfg: BaselineConfig) -> Image.Image | None:
    """Grid: header labels, then per sample: true | mean | PCA k for each k."""
    idx = figure_indices(a["n"], cfg.figure_samples)
    if not idx:
        return None
    x = a["image"][idx]
    cols = [("true", x)] + [
        (name if k is None else f"pca k={k}", pred(x))
        for name, k, pred in _image_predictors(b, cfg)
    ]
    return grid_figure(cols, a["image_size"])


def run_baselines(cfg: BaselineConfig, argv: list[str] | None = None) -> Path:
    """Check inputs, then fit + score + write the run directory; return its path.

    Raises ``DatasetError`` (bad dataset / split) before creating anything, and
    ``FileExistsError`` if the run directory already exists.
    """
    t0 = time.monotonic()
    started = utc_now()
    seed_everything(cfg.seed)
    kw = {"split_file": cfg.split_file}
    train_ds = PoseImageDataset(cfg.dataset_dir, split="train", **kw)
    eval_ds = {s: PoseImageDataset(cfg.dataset_dir, split=s, **kw) for s in cfg.eval_splits}
    split_sha = sha256_file(Path(cfg.split_file))

    run_dir = make_run_dir(cfg.run_root, cfg.run_name, started)
    write_yaml(run_dir / "config.yaml", cfg.to_dict())
    info = {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "seed": cfg.seed,
        "git": git_info(),
        "env": environment_summary(),
        "dataset": {
            "dir": cfg.dataset_dir,
            "manifest_sha256": train_ds.manifest_sha256,
            "manifest_content_sha256": train_ds.manifest_content_sha256,
            "checksums_sha256": checksums_sha256(cfg.dataset_dir),  # pixels (spec 012)
        },
        "split_file": {"path": cfg.split_file, "sha256": split_sha},
        "started_utc": started.isoformat(),
        "finished_utc": None,
        "wall_s": None,
    }
    write_json_exact(run_dir / "run_info.json", info)

    dtype = torch.float32 if cfg.image_pca_ks else TRAIN_IMAGE_DTYPE_NO_PCA  # spec 015 FR-004
    train = load_arrays(train_ds, cfg.batch_size, masks=False, image_dtype=dtype)
    b = fit_baselines(train, cfg)
    del train
    fit = {
        "split": "train",
        "samples": len(train_ds),
        "poses": len({parse_sample_id(s)[0] for s in train_ds.sample_ids}),
        "image_pca": None
        if b.image_pca is None
        else {
            "numerical_rank": b.image_pca.numerical_rank,
            "n_components": b.image_pca.n_components,
            "explained_variance_ratio": {
                str(k): b.image_pca.explained_variance_ratio(k) for k in cfg.image_pca_ks
            },
        },
        "pose_pca": {
            "numerical_rank": b.pose_pca.numerical_rank,
            "n_components": b.pose_pca.n_components,
            "explained_variance_ratio": {
                str(k): b.pose_pca.explained_variance_ratio(k) for k in cfg.pose_pca_ks
            },
        },
    }
    splits, figure = {}, None
    for s, ds in eval_ds.items():
        a = load_arrays(ds, cfg.batch_size, masks=True)
        splits[s] = score_split(b, a, cfg)
        if s == cfg.figure_split:
            fig = recon_figure(b, a, cfg)
            if fig is not None:
                figure = f"recon_{s}.png"
                fig.save(run_dir / figure, format="PNG")
        del a
    metrics = {
        "metrics_version": METRICS_VERSION,
        "dataset": {
            "name": train_ds.manifest["dataset_name"],
            "manifest_sha256": train_ds.manifest_sha256,
            "split_file_sha256": split_sha,
        },
        "fit": fit,
        "regions": region_definitions(),
        "units": UNITS,
        "splits": splits,
    }
    write_json_exact(run_dir / "metrics.json", metrics)
    info |= {
        "figure": figure,
        "finished_utc": utc_now().isoformat(),
        "wall_s": round(time.monotonic() - t0, 1),
        "peak_rss_mb": peak_rss_mb(),  # spec 015 FR-005
    }
    write_json_exact(run_dir / "run_info.json", info)
    return run_dir
