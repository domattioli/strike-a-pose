"""Autoencoder training + evaluation run (spec 004; contract: contracts/train-run.md).

Train split is held in memory as uint8 and scaled ``/255`` per batch (bit-identical
to the 002 loader's float values). The selected weights (spec 008: by default those of
the epoch with the lowest val loss, ``best.pt``) are scored with the 003 image metrics
next to the mean-image and PCA baselines refit on train in the same run.
"""

import csv
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from strike_a_pose.data import DatasetError, PoseImageDataset, parse_sample_id
from strike_a_pose.datagen.assets import sha256_file
from strike_a_pose.eval.baselines import UNITS, image_mean
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.pca import fit_pca
from strike_a_pose.eval.report import (
    figure_indices,
    grid_figure,
    region_definitions,
    score_images,
)
from strike_a_pose.models import ConvAutoencoder
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

from .config import TrainConfig, load_run_config, with_legacy_values
from .guards import (
    ABORTED,
    RunAborted,
    abort_record,
    acquire_lock,
    divergence_reason,
    early_stop_state,
    read_aborted,
    release_lock,
)
from .objective import Objective, gaussian_kl, recon_loss, reparameterize, shift_batch

TRAIN_VERSION = 2  # 2 (spec 008): metrics.json `ae` = the weights `selection` names
METRICS_VERSION = 1  # image metrics as in 003 metrics.json
IMAGE_UNITS = {k: UNITS[k] for k in ("image_mse", "psnr_db", "silhouette_iou", "regions")}
# train_recon / train_kl: spec 011 (appended; runs before it have the first five columns)
HISTORY_FIELDS = ("epoch", "train_loss", "val_loss", "lr", "epoch_s", "train_recon", "train_kl")
CHECKPOINT, BEST = "checkpoint.pt", "best.pt"  # resume state; best-val weights
CL = torch.channels_last


def load_split(ds: PoseImageDataset, batch_size: int, masks: bool = False) -> dict:
    """Read a dataset split into memory: uint8 images ``[n, 1, H, W]``; with ``masks`` (the
    scored splits) also bool masks ``[n, H*W]`` and the labels of the region metrics (spec
    009): 2D keypoints ``kp2d`` float32 ``[n, 24, 2]`` (px) and ``visible`` bool ``[n, 24]``."""
    n = len(ds)
    h, w = ds.image_size
    a: dict = {"n": n, "image_size": (h, w)}
    a["image"] = torch.empty((n, 1, h, w), dtype=torch.uint8)
    a["pose_index"] = torch.empty(n, dtype=torch.int64)
    if masks:
        a["mask"] = torch.empty((n, h * w), dtype=torch.bool)
        a["kp2d"] = torch.empty((n, 24, 2), dtype=torch.float32)
        a["visible"] = torch.empty((n, 24), dtype=torch.bool)
    i = 0
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        m = b["image"].shape[0]
        a["image"][i : i + m] = torch.round(b["image"] * 255.0).to(torch.uint8)
        a["pose_index"][i : i + m] = b["pose_index"]
        if masks:
            a["mask"][i : i + m] = b["mask"].reshape(m, -1)
            a["kp2d"][i : i + m] = b["keypoints_2d"]
            a["visible"][i : i + m] = b["visibility"]
        i += m
    return a


def to_float(u8: torch.Tensor) -> torch.Tensor:
    """uint8 → float32 in [0, 1] (same values as the 002 ``image_to_tensor``)."""
    return u8.to(torch.float32) / 255.0


def train_epoch(
    model: ConvAutoencoder,
    opt: torch.optim.Optimizer,
    images: torch.Tensor,
    batch_size: int,
    gen: torch.Generator,
    sched: torch.optim.lr_scheduler.LRScheduler | None = None,
    objective: Objective | None = None,
) -> dict:
    """One shuffled pass (last batch may be smaller) under ``objective`` (spec 011; None:
    the plain AE, today's computation). Returns the epoch means of the loss
    (``train_loss``), of its reconstruction term (``train_recon``) and of the KL term
    (``train_kl``, None unless variational) and the last lr used (``lr``)."""
    obj = objective or Objective()
    model.train()
    n = images.shape[0]
    perm = torch.randperm(n, generator=gen)
    total = recon_total = 0.0
    kl_total: float | None = None
    lr = float("nan")
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        x = to_float(shift_batch(images[idx], obj.shift_px)).contiguous(memory_format=CL)
        opt.zero_grad(set_to_none=True)
        mu, logvar = model.encode_dist(x)  # (encoder(x), None) for the plain AE
        z = mu if logvar is None else reparameterize(mu, logvar)
        if obj.dropout:  # bottleneck dropout, training only (p = 0: no call, no draw)
            z = F.dropout(z, obj.dropout, training=True)
        recon = recon_loss(model.decode(z), x, obj.mask_weight)
        if logvar is None:
            loss = recon
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
    return {"train_loss": total / n, "train_recon": recon_total / n, "train_kl": kl_mean, "lr": lr}


def make_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the configured decoupled weight decay over one param group (spec 011);
    at ``weight_decay`` 0 it steps exactly like the pre-011 Adam, whose saved states load."""
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


@torch.no_grad()
def predict_rows(model: ConvAutoencoder, rows: torch.Tensor) -> torch.Tensor:
    """Eval-mode reconstruction of flattened image rows ``[n, H*W]`` → ``[n, H*W]``."""
    model.eval()
    h, w = model.image_size
    x = rows.reshape(-1, 1, h, w).contiguous(memory_format=CL)
    return model(x).reshape(rows.shape[0], -1).contiguous()


@torch.no_grad()
def eval_loss(model: ConvAutoencoder, images: torch.Tensor, batch_size: int) -> float | None:
    """Mean per-image MSE (eval mode) over uint8 images; None when empty."""
    n = images.shape[0]
    if n == 0:
        return None
    model.eval()
    total = 0.0
    for i in range(0, n, batch_size):
        x = to_float(images[i : i + batch_size]).contiguous(memory_format=CL)
        total += float(((model(x) - x) ** 2).mean(dim=(1, 2, 3)).double().sum())
    return total / n


def overfit_steps(model: ConvAutoencoder, x: torch.Tensor, steps: int, lr: float) -> list[float]:
    """Adam steps on one fixed batch ``x [B, 1, H, W]``; returns the loss before each step."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), x)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def make_scheduler(opt: torch.optim.Optimizer, schedule: str, total_steps: int):
    if schedule == "constant":
        return None
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, total_steps) / total_steps))
    )


def _fmt(v: float | None, spec: str = ".8g") -> str:
    return "" if v is None else format(v, spec)


def _save_checkpoint(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state(gen: torch.Generator) -> dict:
    """Every RNG the run touches (shuffle generator + global Python / NumPy / torch)."""
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    return {
        "shuffle": gen.get_state(),
        "torch": torch.get_rng_state(),
        "numpy": [name, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)],
        "python": random.getstate(),
    }


def _set_rng_state(gen: torch.Generator, st: dict) -> None:
    gen.set_state(st["shuffle"])
    torch.set_rng_state(st["torch"])
    name, keys, pos, has_gauss, cached = st["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    v, internal, gauss = st["python"]
    random.setstate((v, tuple(internal), gauss))


def _write_history(path: Path, rows: list[dict]) -> None:
    """(Re)write ``history.csv`` from the checkpointed rows (drops rows past the checkpoint)."""
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(HISTORY_FIELDS)
        for r in rows:
            wr.writerow(_history_row(r))


def _is_new_best(v: float | None, best: dict | None) -> bool:
    """A finite val loss below the best so far (NaN never counts; spec 008)."""
    return v is not None and math.isfinite(v) and (best is None or v < best["val_loss"])


def best_val(rows: list[dict]) -> dict | None:
    """Earliest epoch with the lowest finite val loss: ``{"epoch", "val_loss"}``, or None."""
    best = None
    for r in rows:
        if _is_new_best(r["val_loss"], best):
            best = {"epoch": r["epoch"], "val_loss": r["val_loss"]}
    return best


def weight_selection(rows: list[dict], rule: str) -> dict:
    """The weights a finished history scores (spec 008): the ``metrics.json`` block.

    ``best_val``: the earliest epoch with the lowest finite val loss (``best.pt``), or the
    final epoch when val is empty or never finite; ``final``: the final epoch.
    """
    last = rows[-1]
    sel = {
        "select_weights": rule,
        "weights": "final",
        "selected_by": "config",
        "epoch": last["epoch"],
        "val_loss": last["val_loss"],
        "checkpoint": CHECKPOINT,
    }
    if rule == "final":
        return sel
    best = best_val(rows)
    if best is None:
        empty = all(r["val_loss"] is None for r in rows)
        why = "final (empty val)" if empty else "final (no finite val loss)"
        return sel | {"selected_by": why, "val_loss": None}
    return sel | best | {"weights": "best_val", "selected_by": "val loss", "checkpoint": BEST}


def _best_state(epoch: int, val_loss: float, weights: dict, hparams: dict, config: dict) -> dict:
    """``best.pt`` contents (structure unchanged since surrogate ruling 2026-10-20 #3)."""
    return {
        "train_version": TRAIN_VERSION,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_hparams": hparams,
        "model": weights,
        "config": config,
    }


def _read_best(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # truncated or foreign file: treated as missing
        return None


def _holds(b: dict | None, best: dict, cfg: TrainConfig) -> bool:
    """Whether ``best.pt`` contents ``b`` are this run's weights of epoch ``best``."""
    return (
        b is not None
        and b.get("epoch") == best["epoch"]
        and b.get("val_loss") == best["val_loss"]
        and with_legacy_values(b.get("config")) == cfg.to_dict()
    )


def reconcile_best(run_dir: Path, cfg: TrainConfig, ckpt: dict) -> None:
    """Resume (spec 008): make ``best.pt`` hold the checkpointed history's best epoch.

    ``checkpoint.pt`` is written before ``best.pt``, so a kill between the two writes of a
    new best epoch leaves the older ``best.pt`` while the checkpoint holds the new best's
    weights: ``best.pt`` is rewritten from it. The weights of an older best epoch cannot
    be recovered: a ``best_val`` run refuses (``ConfigError``); a ``final`` run, where
    ``best.pt`` is informational, continues.
    """
    best = best_val(ckpt["history"])
    if best is None:
        return
    path = run_dir / BEST
    found = _read_best(path)
    if _holds(found, best, cfg):
        return
    if best["epoch"] == ckpt["epoch"]:
        hp, config = ckpt["model_hparams"], cfg.to_dict()
        _save_checkpoint(
            path, _best_state(best["epoch"], best["val_loss"], ckpt["model"], hp, config)
        )
    elif cfg.select_weights == "best_val":
        have = "missing" if found is None else f"holds epoch {found.get('epoch')}"
        raise ConfigError(
            f"cannot resume {run_dir}: {BEST} ({have}) does not hold the best val epoch "
            f"{best['epoch']} and {CHECKPOINT} is at epoch {ckpt['epoch']}: the weights to "
            "score are lost (start a new run)"
        )


def load_selected_weights(
    model: ConvAutoencoder, run_dir: Path, selection: dict, cfg: TrainConfig
) -> None:
    """Put the weights ``selection`` names into ``model`` (spec 008; before evaluation).

    ``best_val``: read back from ``best.pt``, which must hold this run's selected epoch
    and val loss (else ``ConfigError``); ``final``: ``model`` already holds them.
    """
    if selection["weights"] != "best_val":
        return
    b = _read_best(run_dir / BEST)
    if not _holds(b, selection, cfg):
        raise ConfigError(
            f"{run_dir / BEST} does not hold the selected epoch {selection['epoch']} "
            f"(val loss {selection['val_loss']}) of this run"
        )
    model.load_state_dict(b["model"])


def _selection_line(sel: dict, last: dict) -> str:
    v = "-" if sel["val_loss"] is None else f"{sel['val_loss']:.5f}"
    line = (
        f"scoring {sel['weights']} weights: epoch {sel['epoch']}, val {v} "
        f"({sel['selected_by']}; {sel['checkpoint']})"
    )
    if sel["epoch"] != last["epoch"] and last["val_loss"] is not None:
        line += f"; final epoch {last['epoch']} val {last['val_loss']:.5f}"
    return line


def _history_row(r: dict) -> list:
    return [r["epoch"], _fmt(r["train_loss"]), _fmt(r["val_loss"]), _fmt(r["lr"], ".6g"),
            f"{r['epoch_s']:.1f}", _fmt(r["train_recon"]), _fmt(r["train_kl"])]  # fmt: skip


def _complete_row(r: dict) -> dict:
    """A checkpointed history row with the spec 011 columns (rows from before it lack them:
    the plain AE's reconstruction term is its loss, and it has no KL)."""
    return r if "train_recon" in r else r | {"train_recon": r["train_loss"], "train_kl": None}


def _setup(cfg: TrainConfig) -> dict:
    """Seed, open the splits and build the model; no files are written."""
    seed_everything(cfg.seed)
    kw = {"split_file": cfg.split_file}
    s: dict = {
        "train_ds": PoseImageDataset(cfg.dataset_dir, split="train", **kw),
        "val_ds": PoseImageDataset(cfg.dataset_dir, split="val", **kw),
        "eval_ds": {e: PoseImageDataset(cfg.dataset_dir, split=e, **kw) for e in cfg.eval_splits},
    }
    if len(s["train_ds"]) == 0:
        raise DatasetError("train split has no samples")
    try:
        model = ConvAutoencoder(
            s["train_ds"].image_size, cfg.latent_dim, cfg.channels, cfg.variational
        )
    except ValueError as e:
        raise ConfigError(f"model: {e}") from e
    s["model"] = model.to(memory_format=CL)
    s["split_sha"] = sha256_file(Path(cfg.split_file))
    return s


def _session(argv: list[str] | None, start_epoch: int) -> dict:
    return {
        "started_utc": utc_now().isoformat(),
        "start_epoch": start_epoch,
        "argv": list(sys.argv if argv is None else argv),
        "git": git_info(),
    }


def run_training(
    cfg: TrainConfig,
    argv: list[str] | None = None,
    log: Callable[[str], None] | None = None,
    stop_after_epoch: int | None = None,
) -> Path:
    """Check inputs, train, evaluate, write the run directory; return its path.

    Raises ``DatasetError`` (bad dataset / split) or ``ConfigError`` (model does not
    fit the image size) before creating anything; ``FileExistsError`` if the run
    directory exists. ``stop_after_epoch`` ends the process-level session after that
    epoch's checkpoint (simulated interruption, for tests); continue with
    :func:`resume_training`.
    """
    say = log or (lambda _msg: None)
    t0 = time.monotonic()
    started = utc_now()
    s = _setup(cfg)
    run_dir = make_run_dir(cfg.run_root, cfg.run_name, started)
    acquire_lock(run_dir, list(sys.argv if argv is None else argv))  # spec 012: one writer
    try:
        _start(cfg, run_dir, s, argv, started, t0, say, stop_after_epoch)
    finally:
        release_lock(run_dir)
    return run_dir


def _start(
    cfg: TrainConfig,
    run_dir: Path,
    s: dict,
    argv: list[str] | None,
    started,
    t0: float,
    say: Callable[[str], None],
    stop_after_epoch: int | None,
) -> None:
    """``run_training`` once the run dir exists and is locked."""
    say(f"run dir {run_dir} (resume: strike-a-pose-train --resume {run_dir})")
    write_yaml(run_dir / "config.yaml", cfg.to_dict())
    info = {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "seed": cfg.seed,
        "git": git_info(),
        "env": environment_summary(),
        "dataset": {
            "dir": cfg.dataset_dir,
            "manifest_sha256": s["train_ds"].manifest_sha256,  # resume check (raw, spec 007)
            "manifest_content_sha256": s["train_ds"].manifest_content_sha256,
            "checksums_sha256": checksums_sha256(cfg.dataset_dir),  # pixels (spec 012)
        },
        "split_file": {"path": cfg.split_file, "sha256": s["split_sha"]},
        "model_parameters": s["model"].parameter_count(),
        "history": "history.csv",
        "checkpoint": CHECKPOINT,
        "started_utc": started.isoformat(),
        "sessions": [_session(argv, 1) | {"started_utc": started.isoformat()}],
        "finished_utc": None,
        "wall_s": None,
        "train_wall_s": None,
        "figure": None,
    }
    write_json_exact(run_dir / "run_info.json", info)
    _train_and_evaluate(cfg, run_dir, info, s, None, t0, say, stop_after_epoch)


def resume_training(
    run_dir: str | Path,
    argv: list[str] | None = None,
    log: Callable[[str], None] | None = None,
    stop_after_epoch: int | None = None,
) -> Path:
    """Continue an interrupted run from its last end-of-epoch ``checkpoint.pt``.

    Restores model, optimizer, lr schedule, every RNG state and the loss history, so
    the finished run equals an uninterrupted one (same machine / thread count). If
    training had finished but evaluation had not, only evaluation runs. Raises
    ``ConfigError`` when the run cannot be resumed (no checkpoint, already finished,
    config changed, another live process holds the run dir), ``RunAborted`` for a run the
    divergence guard ended (spec 012) and ``DatasetError`` when the dataset or split file
    changed.
    """
    say = log or (lambda _msg: None)
    t0 = time.monotonic()
    run_dir = Path(run_dir)
    if (run_dir / "metrics.json").exists():
        raise ConfigError(f"cannot resume {run_dir}: run already finished (metrics.json)")
    if (run_dir / ABORTED).exists():
        rec = read_aborted(run_dir / ABORTED) or {}
        raise RunAborted(
            f"cannot resume {run_dir}: the run was aborted at epoch {rec.get('epoch', '?')} "
            f"({rec.get('reason', 'unreadable aborted.json')}; {ABORTED}): start a new run"
        )
    ck_path = run_dir / CHECKPOINT
    if not ck_path.is_file():
        raise ConfigError(f"cannot resume {run_dir}: no {CHECKPOINT} (start a new run)")
    acquire_lock(run_dir, list(sys.argv if argv is None else argv))  # spec 012: one writer
    try:
        _resume(run_dir, ck_path, argv, t0, say, stop_after_epoch)
    finally:
        release_lock(run_dir)
    return run_dir


def _resume(
    run_dir: Path,
    ck_path: Path,
    argv: list[str] | None,
    t0: float,
    say: Callable[[str], None],
    stop_after_epoch: int | None,
) -> None:
    """``resume_training`` once the run dir is locked."""
    cfg = load_run_config(run_dir / "config.yaml")  # keys newer than the run: legacy values
    ckpt = torch.load(ck_path, map_location="cpu", weights_only=True)
    if with_legacy_values(ckpt.get("config")) != cfg.to_dict() or "rng" not in ckpt:
        raise ConfigError(f"cannot resume {run_dir}: checkpoint does not match config.yaml")
    info = json.loads((run_dir / "run_info.json").read_text())
    s = _setup(cfg)
    if s["train_ds"].manifest_sha256 != info["dataset"]["manifest_sha256"]:
        raise DatasetError("dataset manifest changed since the run started")
    if s["split_sha"] != info["split_file"]["sha256"]:
        raise DatasetError("split file changed since the run started")
    reconcile_best(run_dir, cfg, ckpt)  # before this session is recorded (spec 008)
    info.setdefault("sessions", []).append(_session(argv, ckpt["epoch"] + 1))
    write_json_exact(run_dir / "run_info.json", info)
    say(f"resuming {run_dir} after epoch {ckpt['epoch']}/{cfg.epochs}")
    _train_and_evaluate(cfg, run_dir, info, s, ckpt, t0, say, stop_after_epoch)


def _train_and_evaluate(
    cfg: TrainConfig,
    run_dir: Path,
    info: dict,
    s: dict,
    ckpt: dict | None,
    t0: float,
    say: Callable[[str], None],
    stop_after_epoch: int | None,
) -> None:
    model, train_ds, val_ds = s["model"], s["train_ds"], s["val_ds"]
    train = load_split(train_ds, cfg.batch_size)
    val = load_split(val_ds, cfg.batch_size, masks="val" in cfg.eval_splits)
    say(f"loaded train {train['n']} / val {val['n']} images ({time.monotonic() - t0:.0f} s)")

    opt = make_optimizer(model.parameters(), cfg)
    objective = Objective.from_config(cfg)
    steps_per_epoch = math.ceil(train["n"] / cfg.batch_size)
    sched = make_scheduler(opt, cfg.lr_schedule, cfg.epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(cfg.seed)
    rows: list[dict] = []
    elapsed0 = train0 = 0.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None:
            sched.load_state_dict(ckpt["scheduler"])
        _set_rng_state(gen, ckpt["rng"])
        rows = [_complete_row(r) for r in ckpt["history"]]
        elapsed0, train0 = ckpt["elapsed_s"], ckpt["train_wall_s"]
    hist_path = run_dir / "history.csv"
    _write_history(hist_path, rows)
    best = best_val(rows)  # recomputed from the checkpointed history on resume
    stop = None  # spec 012: the early-stop record once the guard fires
    if ckpt is not None:  # resume: the guards decide from the checkpointed history
        _abort_if_diverged(cfg, run_dir, rows, say)
        stop = _early_stop(rows, cfg)
    if cfg.early_stop is not None and val["n"] == 0:
        say("early stop inactive: val is empty")
    t_train = time.monotonic()
    with hist_path.open("a", newline="") as fh:
        wr = csv.writer(fh)
        for epoch in range(len(rows) + 1, cfg.epochs + 1):
            if stop is not None:
                break
            te = time.monotonic()
            st = train_epoch(model, opt, train["image"], cfg.batch_size, gen, sched, objective)
            tl, lr = st["train_loss"], st["lr"]
            vl = eval_loss(model, val["image"], cfg.batch_size)
            dt = time.monotonic() - te
            row = {"epoch": epoch, "train_loss": tl, "val_loss": vl, "lr": lr, "epoch_s": dt,
                   "train_recon": st["train_recon"], "train_kl": st["train_kl"]}  # fmt: skip
            rows.append(row)
            wr.writerow(_history_row(row))
            fh.flush()
            new_best = _is_new_best(vl, best)
            if new_best:
                best = {"epoch": epoch, "val_loss": vl}
            now = time.monotonic()
            _save_checkpoint(
                run_dir / CHECKPOINT,
                {
                    "train_version": TRAIN_VERSION,
                    "epoch": epoch,
                    "model_hparams": model.hparams(),
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched is not None else None,
                    "rng": _rng_state(gen),
                    "history": rows,
                    "elapsed_s": elapsed0 + now - t0,
                    "train_wall_s": train0 + now - t_train,
                    "config": cfg.to_dict(),
                },
            )
            if new_best:  # after checkpoint.pt (spec 008): a kill in between is repaired
                state = model.state_dict()
                _save_checkpoint(
                    run_dir / BEST, _best_state(epoch, vl, state, model.hparams(), cfg.to_dict())
                )
            vs = "-" if vl is None else f"{vl:.5f}"
            kl = "" if st["train_kl"] is None else f" kl {st['train_kl']:.1f}"
            say(f"epoch {epoch}/{cfg.epochs} train {tl:.5f}{kl} val {vs} lr {lr:.2e} ({dt:.0f} s)")
            if stop_after_epoch is not None and stop_after_epoch <= epoch < cfg.epochs:
                say(f"stopped after epoch {epoch} (resume: --resume {run_dir})")
                return
            _abort_if_diverged(cfg, run_dir, rows, say)  # spec 012, before the early stop
            stop = _early_stop(rows, cfg)
            if stop is not None:
                say(f"early stop after epoch {epoch}/{cfg.epochs}: {stop['reason']}")
    train_wall = train0 + time.monotonic() - t_train
    selection = weight_selection(rows, cfg.select_weights)
    load_selected_weights(model, run_dir, selection, cfg)  # the scored weights (spec 008)
    say(_selection_line(selection, rows[-1]))
    model.eval()

    # Context baselines, fit on train only (003 code path: exact PCA, clamp to [0, 1]); the mean
    # image from the uint8 rows in float64 batches, the float32 matrix only for the PCA (spec 012).
    mean_image = image_mean(train["image"].reshape(train["n"], -1))
    pca = None
    if cfg.context_pca_k:
        x_train = to_float(train["image"]).reshape(train["n"], -1)
        pca = fit_pca(x_train, cfg.context_pca_k)
        del x_train
    del train
    preds = {
        "ae": lambda x: predict_rows(model, x),
        "mean_image": lambda x: mean_image.expand(x.shape[0], -1),
    }
    pca_name = f"pca_k{cfg.context_pca_k}"
    if pca is not None:
        preds[pca_name] = lambda x: pca.reconstruct(x, cfg.context_pca_k).clamp(0.0, 1.0)

    splits, figure = {}, None
    for name_s, ds in s["eval_ds"].items():
        a = val if name_s == "val" else load_split(ds, cfg.batch_size, masks=True)
        n = a["n"]
        out: dict = {"samples": n, "poses": int(a["pose_index"].unique().numel()) if n else 0}
        if n:
            x = to_float(a["image"]).reshape(n, -1)
            labels = (a["kp2d"], a["visible"], a["image_size"])  # region metrics (spec 009)
            img = {}
            for name, fn in preds.items():
                r = score_images(fn, x, a["mask"], cfg.batch_size, cfg.iou_threshold, labels=labels)
                if name == pca_name:
                    r |= {"k": cfg.context_pca_k, "k_effective": pca.k_effective(cfg.context_pca_k)}
                img[name] = r
            out["image"] = img
            if name_s == cfg.figure_split:
                idx = figure_indices(n, cfg.figure_samples)
                xi = x[idx]
                cols = [("true", xi)] + [
                    (f"pca k={cfg.context_pca_k}" if nm == pca_name else nm, fn(xi))
                    for nm, fn in preds.items()
                ]
                figure = f"recon_{name_s}.png"
                grid_figure(cols, a["image_size"]).save(run_dir / figure, format="PNG")
        splits[name_s] = out
        say(f"{name_s} ({n}): " + _headline(out))

    vals = [r["val_loss"] for r in rows]
    metrics = {
        "train_version": TRAIN_VERSION,
        "metrics_version": METRICS_VERSION,
        "dataset": {
            "name": train_ds.manifest["dataset_name"],
            "manifest_sha256": train_ds.manifest_sha256,
            "split_file_sha256": s["split_sha"],
        },
        "model": {
            "name": "ConvAutoencoder",
            **model.hparams(),
            "parameters": info["model_parameters"],
        },
        "training": {
            "epochs": cfg.epochs,
            "steps": cfg.epochs * steps_per_epoch,
            "train_samples": len(train_ds),
            "train_poses": len({parse_sample_id(i)[0] for i in train_ds.sample_ids}),
            "val_samples": len(val_ds),
            "first_train_loss": rows[0]["train_loss"],
            "final_train_loss": rows[-1]["train_loss"],
            "first_val_loss": vals[0],
            "final_val_loss": vals[-1],
            "best_val_epoch": best["epoch"] if best else None,
            "best_val_loss": best["val_loss"] if best else None,
            "objective": {"weight_decay": cfg.weight_decay, **asdict(objective)},  # spec 011
            **({"stopped_early": stop} if cfg.early_stop is not None else {}),  # spec 012
        },
        "context": {
            "fit_split": "train",
            "pca_k": cfg.context_pca_k,
            "pca_k_effective": pca.k_effective(cfg.context_pca_k) if pca is not None else None,
        },
        "selection": selection,
        "regions": region_definitions(),
        "units": IMAGE_UNITS,
        "splits": splits,
    }
    write_json_exact(run_dir / "metrics.json", metrics)
    info |= {
        "figure": figure,
        "best_val_epoch": best["epoch"] if best else None,
        "best_checkpoint": BEST if best else None,
        "finished_utc": utc_now().isoformat(),
        "wall_s": round(elapsed0 + time.monotonic() - t0, 1),
        "train_wall_s": round(train_wall, 1),
    }
    write_json_exact(run_dir / "run_info.json", info)


def _early_stop(rows: list[dict], cfg: TrainConfig) -> dict | None:
    """The early-stop record once the guard fires on ``rows`` (spec 012), else None."""
    st = early_stop_state(rows, cfg.early_stop)
    return None if st is None else st["fired"]


def _abort_if_diverged(
    cfg: TrainConfig, run_dir: Path, rows: list[dict], say: Callable[[str], None]
) -> None:
    """Divergence guard (spec 012): write ``aborted.json`` and raise ``RunAborted`` when the
    last epoch of ``rows`` diverged and ``abort_on_divergence`` is set."""
    if not cfg.abort_on_divergence or not rows:
        return
    reason = divergence_reason(rows, cfg.divergence_ratio)
    if reason is None:
        return
    path = run_dir / ABORTED
    tmp = path.with_name(f"{ABORTED}.tmp")
    write_json_exact(tmp, abort_record(rows, reason, cfg.divergence_ratio))
    os.replace(tmp, path)
    say(f"aborted after epoch {rows[-1]['epoch']}: {reason}")
    raise RunAborted(f"{reason} ({path})")


def _headline(out: dict) -> str:
    if not out["samples"]:
        return "no samples"
    return "; ".join(
        f"{k} mse {r['image_mse']:.5f} iou {r['silhouette_iou']:.3f}"
        for k, r in out["image"].items()
    )
