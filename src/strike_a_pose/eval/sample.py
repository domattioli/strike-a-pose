"""Sampling evaluation of a VAE prior (spec 015 FR-008; research R2-R5).

Decode ``z ~ N(0, I)`` with the VAE decoder (and, as a lower reference bar, with a plain AE's
decoder on the same z), and μ-reconstruct val images (upper bar). Each set is scored the same
way: the **valid fraction** (silhouette ``x > iou_threshold`` non-empty, its area inside the
[1st, 99th] percentile of the train masks' areas, and its largest 8-connected component
holding ≥ 95 % of it) and the **nearest-neighbour MSE** to the train images.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from strike_a_pose.data import DatasetError, PoseImageDataset
from strike_a_pose.datagen.assets import sha256_file
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
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

from .config import ConfigError
from .report import grid_figure
from .sample_config import SampleConfig

LARGEST_FRACTION = 0.95
AREA_PERCENTILES = (1.0, 99.0)
NN_CHUNK = 2048
METRICS_VERSION = 1
FIGURE_COLS = 8  # samples per figure row; each followed by its nearest train image


def largest_component_fraction(mask: np.ndarray) -> tuple[float, int]:
    """(largest 8-connected component / foreground, number of components) of a bool [H, W]
    mask; (0.0, 0) when empty. Iterative flood fill (no SciPy; research R3)."""
    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return 0.0, 0
    h, w = m.shape
    seen = np.zeros_like(m)
    best, count = 0, 0
    for y0, x0 in zip(*np.nonzero(m), strict=True):
        if seen[y0, x0]:
            continue
        count += 1
        size, stack = 0, [(int(y0), int(x0))]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        best = max(best, size)
    return best / total, count


def classify(mask: np.ndarray, lo: float, hi: float) -> str:
    """ "valid", "empty", "area" (outside [lo, hi] px) or "components" (research R2)."""
    area = int(np.asarray(mask).sum())
    if area == 0:
        return "empty"
    if not lo <= area <= hi:
        return "area"
    frac, _ = largest_component_fraction(mask)
    return "valid" if frac >= LARGEST_FRACTION else "components"


def nn_mse(
    samples: torch.Tensor, train_u8: torch.Tensor, chunk: int = NN_CHUNK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest train row of each sample (float32 distances per chunk), and the exact float64
    MSE to it (research R4). ``samples`` float [n, P] in [0, 1]; ``train_u8`` uint8 [N, P]."""
    s = samples.to(torch.float32)
    s2 = (s * s).sum(1, keepdim=True)
    best = torch.full((s.shape[0],), float("inf"))
    idx = torch.zeros(s.shape[0], dtype=torch.int64)
    for i in range(0, train_u8.shape[0], chunk):
        t = train_u8[i : i + chunk].to(torch.float32) / 255.0
        d = s2 + (t * t).sum(1)[None, :] - 2.0 * s @ t.T
        v, j = d.min(1)
        better = v < best
        best = torch.where(better, v, best)
        idx = torch.where(better, j + i, idx)
    nn = train_u8[idx].to(torch.float64) / 255.0
    mse = ((samples.to(torch.float64) - nn) ** 2).mean(1)
    return idx, mse


def score_set(
    images: torch.Tensor, train_u8: torch.Tensor, lo: float, hi: float, threshold: float,
    image_size: tuple[int, int],
) -> tuple[dict, torch.Tensor]:  # fmt: skip
    """The data-model "S" record of a set of images [n, P], and its NN indices."""
    n = images.shape[0]
    h, w = image_size
    fails = {"empty": 0, "area": 0, "components": 0}
    fg = (images > threshold).reshape(n, h, w).numpy()
    for k in range(n):
        c = classify(fg[k], lo, hi)
        if c != "valid":
            fails[c] += 1
    idx, mse = nn_mse(images, train_u8)
    valid = n - sum(fails.values())
    rec = {
        "n": n,
        "valid_fraction": valid / n if n else 0.0,
        "fail_empty": fails["empty"],
        "fail_area": fails["area"],
        "fail_components": fails["components"],
        "nn_mse_mean": float(mse.mean()) if n else None,
        "nn_mse_median": float(mse.median()) if n else None,
        "foreground_px_mean": float(fg.reshape(n, -1).sum(1).mean()) if n else None,
    }
    return rec, idx


def decode_batches(model: ConvAutoencoder, z: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Decoded images [n, H*W] in [0, 1]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            x = model.decode(z[i : i + batch_size])
            out.append(x.reshape(x.shape[0], -1).clamp(0.0, 1.0))
    return torch.cat(out) if out else torch.empty(0)


def _load(path: str, what: str) -> tuple[ConvAutoencoder, Path]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"{what}: no such file {p}")
    try:
        model, _ = load_checkpoint(p)
    except Exception as e:  # not a 004 checkpoint
        raise ConfigError(f"{what}: cannot load {p}: {e}") from e
    return model, p


def _train_arrays(ds: PoseImageDataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Train images uint8 [N, P] and true mask areas [N] in one pass."""
    n = len(ds)
    h, w = ds.image_size
    img = torch.empty((n, h * w), dtype=torch.uint8)
    area = torch.empty(n, dtype=torch.int64)
    i = 0
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        m = b["image"].shape[0]
        img[i : i + m] = torch.round(b["image"].reshape(m, -1) * 255.0).to(torch.uint8)
        area[i : i + m] = b["mask"].reshape(m, -1).sum(1)
        i += m
    return img, area


def _val_images(ds: PoseImageDataset, n: int, batch_size: int) -> torch.Tensor:
    out, got = [], 0
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        if got >= n:
            break
        x = b["image"][: n - got]
        out.append(x)
        got += x.shape[0]
    return torch.cat(out) if out else torch.empty(0)


def _figure(s: torch.Tensor, nn: torch.Tensor, k: int, image_size: tuple[int, int]):
    k = min(k, s.shape[0])
    cols = min(FIGURE_COLS, k)
    rows = k // cols
    columns = []
    for c in range(cols):
        sel = torch.arange(rows) * cols + c
        columns += [("vae prior", s[sel]), ("nearest train", nn[sel])]
    return grid_figure(columns, image_size)


def run_sample(
    cfg: SampleConfig, argv: list[str] | None = None, log: Callable[[str], None] | None = None
) -> Path:
    """Check inputs, decode and score the sets, write the run dir. Raises ``ConfigError`` /
    ``DatasetError`` before creating anything."""
    say = log or (lambda _msg: None)
    t0 = time.monotonic()
    started = utc_now()
    train_ds = PoseImageDataset(cfg.dataset_dir, split="train", split_file=cfg.split_file)
    val_ds = PoseImageDataset(cfg.dataset_dir, split=cfg.recon_split, split_file=cfg.split_file)
    if len(train_ds) == 0:
        raise DatasetError("train split has no samples")
    vae, vae_path = _load(cfg.checkpoint, "checkpoint")
    if not vae.variational:
        raise ConfigError("checkpoint: not a VAE (variational: false)")
    image_size = tuple(train_ds.image_size)
    if tuple(vae.image_size) != image_size:
        raise ConfigError(f"checkpoint image size {vae.image_size} != dataset {image_size}")
    ref = ref_path = None
    if cfg.reference_checkpoint is not None:
        ref, ref_path = _load(cfg.reference_checkpoint, "reference_checkpoint")
        if ref.variational:
            raise ConfigError("reference_checkpoint: must be a plain AE, got a VAE")
        if ref.latent_dim != vae.latent_dim or tuple(ref.image_size) != image_size:
            raise ConfigError("reference_checkpoint: latent_dim / image size differ from the VAE")
    seed_everything(cfg.seed)

    train_u8, areas = _train_arrays(train_ds, cfg.batch_size)
    lo, hi = (float(v) for v in np.percentile(areas.numpy(), AREA_PERCENTILES))
    say(f"train {train_u8.shape[0]} images; area bounds [{lo:.0f}, {hi:.0f}] px")
    g = torch.Generator().manual_seed(cfg.seed)
    z = torch.randn(cfg.n_samples, vae.latent_dim, generator=g)
    sets, nn_idx = {}, {}
    images = {"vae_prior": decode_batches(vae, z, cfg.batch_size)}
    if ref is not None:
        images["ae_prior"] = decode_batches(ref, z, cfg.batch_size)
    xv = _val_images(val_ds, min(cfg.n_recon, len(val_ds)), cfg.batch_size)
    with torch.no_grad():  # μ of the VAE (as the probe)
        bs = cfg.batch_size
        mu = [vae.encode(xv[i : i + bs]) for i in range(0, xv.shape[0], bs)]
    images["vae_recon_val"] = (
        decode_batches(vae, torch.cat(mu), bs) if mu else torch.empty(0, train_u8.shape[1])
    )
    for name, x in images.items():
        sets[name], nn_idx[name] = score_set(x, train_u8, lo, hi, cfg.iou_threshold, image_size)
        say(f"{name}: valid {sets[name]['valid_fraction']:.3f} nn_mse {sets[name]['nn_mse_mean']}")

    run_dir = make_run_dir(cfg.run_root, cfg.run_name, started)
    write_yaml(run_dir / "config.yaml", cfg.to_dict())
    split_sha = sha256_file(Path(cfg.split_file))
    ck = {"path": str(vae_path), "sha256": sha256_file(vae_path)}
    rck = None if ref_path is None else {"path": str(ref_path), "sha256": sha256_file(ref_path)}
    dataset = {
        "name": train_ds.manifest["dataset_name"],
        "manifest_sha256": train_ds.manifest_sha256,
        "split_file_sha256": split_sha,
    }
    report = {
        "metrics_version": METRICS_VERSION,
        "checkpoint": ck,
        "reference_checkpoint": rck,
        "dataset": dataset,
        "criteria": {
            "iou_threshold": cfg.iou_threshold,
            "largest_fraction": LARGEST_FRACTION,
            "area_percentiles": list(AREA_PERCENTILES),
            "area_bounds_px": [lo, hi],
            "train_images": int(train_u8.shape[0]),
        },
        "sets": sets,
    }
    write_json_exact(run_dir / "samples.json", report)
    s = images["vae_prior"]
    fig = _figure(s, train_u8[nn_idx["vae_prior"]].float() / 255.0, cfg.figure_samples, image_size)
    fig.save(run_dir / "samples.png", format="PNG")
    info = {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "seed": cfg.seed,
        "git": git_info(),
        "env": environment_summary(),
        "dataset": dataset
        | {
            "dir": cfg.dataset_dir,
            "checksums_sha256": checksums_sha256(cfg.dataset_dir),
        },  # fmt: skip
        "checkpoint": ck,
        "reference_checkpoint": rck,
        "started_utc": started.isoformat(),
        "finished_utc": utc_now().isoformat(),
        "wall_s": round(time.monotonic() - t0, 1),
        "peak_rss_mb": peak_rss_mb(),  # spec 015 FR-005
    }
    write_json_exact(run_dir / "run_info.json", info)
    return run_dir


__all__ = ["classify", "largest_component_fraction", "nn_mse", "run_sample", "score_set"]
