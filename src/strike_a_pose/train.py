"""Train the multi-view shape model with AdamW, checkpoints, resume, a time budget, and a history.

The stage reads ``<out>/data`` and writes ``<out>/train`` (contracts/artifacts.md): checkpoints, a
``history.csv``, ``model_final.pt``, and ``DONE.json``. Public sources:

* Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019,
  https://arxiv.org/abs/1711.05101), for the AdamW optimizer, and Loshchilov and Hutter, "SGDR:
  Stochastic Gradient Descent with Warm Restarts" (ICLR 2017, https://arxiv.org/abs/1608.03983),
  for the cosine learning-rate decay (here without restarts).
* A linear warm-up of the learning rate (Goyal et al., "Accurate, Large Minibatch SGD", 2017,
  https://arxiv.org/abs/1706.02677). ``WARMUP_FRACTION`` = 0.05 is a module constant, not a
  configuration key: the learning rate rises linearly from 1% of ``train.lr`` to ``train.lr`` over
  the first 5% of all optimizer steps (at least 1 step), then follows the cosine decay over the
  remaining steps down to 1% of ``train.lr``.
* The model and its loss follow Wu and Goodman 2018 and Kingma and Welling 2013 (research R7); this
  module only calls ``ShapeVAE.loss``.
* The PyTorch reproducibility notes (https://pytorch.org/docs/stable/notes/randomness.html) for the
  seeded generators and the deterministic flags that ``seeding.seed_torch`` sets (research R9).
* Checkpoint and resume within a stage follow research R10.

Splits (research R9, data-model.md DataSplit). The training sampler draws only from
``DataSplit.sampler_range``, the training range without its last ``n_monitor`` bodies. Those last
bodies are the loss-monitoring slice: their ``split`` stays ``train`` in the manifest, the loss on
them is logged as ``monitor_loss`` at the end of each epoch, and they take no gradient step. No
selection reads that loss, the calibration split, or the test split: the stage trains for the
configured number of epochs and saves the last model.

Determinism and resume. The order of the bodies in epoch ``e`` comes from ``rng_for(seed, stage,
e, marker)``, so it needs no stored state. The dataset draws are stateless as well (``set_epoch``).
The one stream that the checkpoint must carry is the global torch generator, which supplies the
latent samples of the loss. The data loader gets its own generator, so that creating an iterator
after a resume does not consume the global stream. A checkpoint holds the model, the optimizer, the
scheduler, the random states of torch (CPU and CUDA), NumPy, and Python, the epoch, the step, the
number of batches done in the epoch, and the history so far. On a CPU a run that stops at a
checkpoint and resumes reaches the same weights and losses as an uninterrupted run.

Checkpoints. ``checkpoints/step_NNNNNN.pt`` is written every ``train.checkpoint_every`` steps, and
``checkpoints/epoch_NNN.pt`` at the end of each epoch (after the monitor loss). The time budget is
tested at each checkpoint: when the next interval would pass the budget, the stage stops there, and
a later call with ``resume=True`` continues from the newest checkpoint (FR-029).
"""

import csv
import functools
import io
import logging
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from strike_a_pose.checkpoint import (
    StageStatus,
    TimeBudget,
    atomic_path,
    atomic_write_text,
    check_stage,
    clear_done_marker,
    input_hashes,
    write_done_marker,
)
from strike_a_pose.config import config_hash
from strike_a_pose.data.dataset import ShapeDataset, collate_samples
from strike_a_pose.data.generate import data_directory
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.device import select_device
from strike_a_pose.model.vae import LossTerms, ShapeVAE, ViewBatch
from strike_a_pose.runrecord import STAGES, RunRecord, current_code_version
from strike_a_pose.seeding import rng_for, seed_torch

__all__ = [
    "HISTORY_COLUMNS",
    "STAGE_NAME",
    "WARMUP_FRACTION",
    "TrainingResult",
    "epoch_batches",
    "learning_rate_factor",
    "train_directory",
    "train_model",
]

logger = logging.getLogger(__name__)

STAGE_NAME = "train"
TRAIN_DIRECTORY_NAME = "train"
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
HISTORY_FILE_NAME = "history.csv"
FINAL_MODEL_FILE_NAME = "model_final.pt"

# The columns of history.csv. Each row is one optimizer step. monitor_loss is filled on the last
# step of each epoch and empty on the other steps.
HISTORY_COLUMNS = (
    "step",
    "epoch",
    "loss",
    "nll_joint",
    "nll_single",
    "kl_joint",
    "kl_single",
    "kl_weight",
    "monitor_loss",
    "learning_rate",
    "wall_seconds",
)

# Random-stream identifiers inside the train stage (research R9). The dataset reads paths of four
# elements, so the three-element sampler and monitor paths below never meet its streams.
_STAGE_ID = STAGES.index(STAGE_NAME)
_ORDER_MARKER = 1
_MONITOR_MARKER = 2

# The schedule starts and ends at this share of the peak learning rate ``train.lr``.
_FINAL_LEARNING_RATE_SHARE = 0.01

# The warm-up covers this fraction of all optimizer steps (at least one step).
WARMUP_FRACTION = 0.05

_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainingResult:
    """What a call of ``train_model`` did.

    ``completed`` is False when the time budget stopped the stage at a checkpoint; the sap command
    then exits with code 7 and a later call with ``resume=True`` continues. ``skipped`` is True when
    a matching DONE.json made the call do nothing. ``final_loss`` is the loss of the last step taken
    and ``final_monitor_loss`` the monitor loss of the last finished epoch (None before the first
    epoch ends, or when the monitor slice holds no unflagged body). ``resumed_from_step`` is the
    step of the checkpoint that the call continued from, or None for a fresh start.
    """

    train_dir: Path
    completed: bool
    skipped: bool
    steps_done: int
    total_steps: int
    epochs_done: int
    final_loss: float | None
    final_monitor_loss: float | None
    resumed_from_step: int | None


def train_directory(out_dir: str | Path) -> Path:
    """Return the folder of this stage, ``<out>/train``."""
    return Path(out_dir) / TRAIN_DIRECTORY_NAME


def epoch_batches(seed: int, epoch: int, n_samples: int, batch_size: int) -> list[list[int]]:
    """Return the batches of one epoch: dataset positions in a seeded random order.

    The order depends only on the seed, the epoch, and the sizes, so a resumed run rebuilds the
    same batches. Every position from 0 to ``n_samples - 1`` appears once; the last batch may be
    shorter.
    """
    order = rng_for(seed, _STAGE_ID, epoch, _ORDER_MARKER).permutation(n_samples).tolist()
    return [order[start : start + batch_size] for start in range(0, n_samples, batch_size)]


def learning_rate_factor(step: int, total_steps: int) -> float:
    """Return the learning rate at a 0-based optimizer step as a multiple of ``train.lr``.

    The factor rises linearly from 0.01 at step 0 to 1 at step ``W = max(1, int(WARMUP_FRACTION *
    total_steps))``, then follows a cosine from 1 down to 0.01 at ``total_steps``. It depends only
    on the step, so a resumed run (the scheduler state holds the step) gets the same rates.
    """
    warmup = max(1, int(WARMUP_FRACTION * total_steps))
    floor = _FINAL_LEARNING_RATE_SHARE
    if step < warmup:
        return floor + (1.0 - floor) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    resume: bool = False,
    time_budget: TimeBudget | None = None,
    run_record: RunRecord | None = None,
) -> TrainingResult:
    """Train the model of a resolved configuration on ``<out_dir>/data`` into ``<out_dir>/train``.

    ``resume=False`` starts from scratch and removes the files of an earlier run. With
    ``resume=True`` a matching DONE.json skips the stage, and otherwise the newest checkpoint of the
    same configuration is continued (a checkpoint of another configuration is ignored). The stage
    needs the DONE.json of the generate stage, whose configuration hash it records as its input.
    ``run_record`` supplies the code version and the hardware class for DONE.json.

    Raises FileNotFoundError when the data stage is not done, ConfigError (exit code 2) for a bad
    configuration, and ValueError when the training range holds no unflagged body.
    """
    digest = config_hash(config)  # validates the configuration first
    seed = int(config["seed"])
    train_config = config["train"]
    epochs = int(train_config["epochs"])
    batch_size = int(train_config["batch_size"])
    checkpoint_every = int(train_config["checkpoint_every"])
    if min(epochs, batch_size, checkpoint_every) < 1:
        raise ValueError("train.epochs, train.batch_size, and train.checkpoint_every must be >= 1")

    train_dir = train_directory(out_dir)
    data_dir = data_directory(out_dir)
    inputs = input_hashes({"generate": data_dir})
    if "generate" not in inputs:
        raise FileNotFoundError(
            f"the data stage is not done: {data_dir / 'DONE.json'} is missing or unreadable"
        )
    choice = select_device(config["device"])
    device = choice.device
    train_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        check = check_stage(train_dir, stage=STAGE_NAME, config_hash=digest, inputs=inputs)
        if check.status is StageStatus.DONE:
            logger.info("training is done and current; skipping (%s)", check.reason)
            return _skipped_result(train_dir, epochs)
    clear_done_marker(train_dir)

    seed_torch(seed)
    split = _data_split(config)
    train_set = ShapeDataset.for_training(config, data_dir, split.sampler_range)
    monitor_set = ShapeDataset.for_training(config, data_dir, split.monitor_range)
    if len(train_set) == 0:
        raise ValueError(
            f"the training range {split.sampler_range} holds no unflagged body, so there is "
            "nothing to train on"
        )
    steps_per_epoch = math.ceil(len(train_set) / batch_size)
    total_steps = epochs * steps_per_epoch

    model = ShapeVAE.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["lr"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(learning_rate_factor, total_steps=total_steps)
    )

    state = _TrainingState()
    resumed_from: int | None = None
    if resume:
        loaded = _load_newest_checkpoint(train_dir, digest, steps_per_epoch, total_steps)
        if loaded is not None:
            resumed_from = _restore(loaded, model, optimizer, scheduler, state, device)
            logger.info(
                "resuming from step %d (epoch %d, batch %d)",
                state.step,
                state.epoch,
                state.batches_done,
            )
    if resumed_from is None:
        _remove_earlier_outputs(train_dir)

    loader_generator = torch.Generator()
    began = time.monotonic()
    wall_offset = state.wall_seconds

    def wall_seconds() -> float:
        return wall_offset + (time.monotonic() - began)

    interval_began = time.monotonic()
    last_monitor: float | None = state.monitor_losses[-1] if state.monitor_losses else None
    stopped_early = False
    while state.epoch < epochs and not stopped_early:
        epoch = state.epoch
        train_set.set_epoch(epoch)
        batches = epoch_batches(seed, epoch, len(train_set), batch_size)[state.batches_done :]
        loader = DataLoader(
            train_set,
            batch_sampler=batches,
            collate_fn=collate_samples,
            num_workers=0,
            generator=loader_generator,
        )
        model.train()
        for batch in loader:
            terms = model.loss(
                _to_device(batch.views, device),
                batch.betas.to(device),
                step=state.step,
                total_steps=total_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            optimizer.step()
            state.history.append(_history_row(state, epoch, terms, optimizer, wall_seconds()))
            scheduler.step()
            state.step += 1
            state.batches_done += 1
            epoch_finished = state.batches_done == steps_per_epoch
            if epoch_finished:
                last_monitor = _monitor_loss(model, monitor_set, config, epoch, device)
                state.history[-1]["monitor_loss"] = last_monitor
                state.monitor_losses.append(last_monitor)
                state.epoch += 1
                state.batches_done = 0
            on_interval = state.step % checkpoint_every == 0
            if not (epoch_finished or on_interval):
                continue
            state.wall_seconds = wall_seconds()
            if epoch_finished:
                _save_checkpoint(
                    train_dir,
                    f"epoch_{epoch:03d}.pt",
                    state,
                    model,
                    optimizer,
                    scheduler,
                    digest,
                    seed,
                )
            if on_interval and not epoch_finished:
                _save_checkpoint(
                    train_dir,
                    f"step_{state.step:06d}.pt",
                    state,
                    model,
                    optimizer,
                    scheduler,
                    digest,
                    seed,
                )
            _write_history(train_dir, state.history)
            interval_seconds = time.monotonic() - interval_began
            interval_began = time.monotonic()
            if state.step < total_steps and time_budget is not None:
                if time_budget.should_stop(interval_seconds):
                    stopped_early = True
                    break
            if epoch_finished:
                break  # the next epoch builds its own loader

    _write_history(train_dir, state.history)
    final_loss = state.history[-1]["loss"] if state.history else None
    if stopped_early:
        logger.warning(
            "the time budget stopped training at step %d of %d; resume to finish",
            state.step,
            total_steps,
        )
        return TrainingResult(
            train_dir,
            False,
            False,
            state.step,
            total_steps,
            state.epoch,
            final_loss,
            last_monitor,
            resumed_from,
        )

    with atomic_path(train_dir / FINAL_MODEL_FILE_NAME) as temporary:
        torch.save(
            {
                "model": {name: value.cpu() for name, value in model.state_dict().items()},
                "config_hash": digest,
                "seed": seed,
                "step": state.step,
                "epochs": state.epoch,
            },
            temporary,
        )
    record = run_record
    write_done_marker(
        train_dir,
        stage=STAGE_NAME,
        config_hash=digest,
        seed=seed,
        code_version=record.code_version if record is not None else current_code_version(),
        hardware_class=record.hardware_class if record is not None else choice.hardware_class,
        inputs=inputs,
    )
    return TrainingResult(
        train_dir,
        True,
        False,
        state.step,
        total_steps,
        state.epoch,
        final_loss,
        last_monitor,
        resumed_from,
    )


class _TrainingState:
    """The mutable position of a run: step, epoch, batches done in the epoch, and the logs."""

    def __init__(self) -> None:
        self.step = 0
        self.epoch = 0  # the epoch being trained, or the next one when batches_done is 0
        self.batches_done = 0
        self.wall_seconds = 0.0
        self.history: list[dict[str, Any]] = []
        self.monitor_losses: list[float | None] = []


def _data_split(config: Mapping[str, Any]) -> DataSplit:
    data = config["data"]
    return DataSplit(int(data["n_train"]), int(data["n_cal"]), int(data["n_test"]))


def _to_device(views: ViewBatch, device: torch.device) -> ViewBatch:
    return ViewBatch(*(tensor.to(device) for tensor in views))


def _history_row(
    state: _TrainingState,
    epoch: int,
    terms: LossTerms,
    optimizer: torch.optim.Optimizer,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "step": state.step + 1,
        "epoch": epoch,
        "loss": float(terms.total.detach()),
        "nll_joint": float(terms.nll_joint.detach()),
        "nll_single": float(terms.nll_single.detach()),
        "kl_joint": float(terms.kl_joint.detach()),
        "kl_single": float(terms.kl_single.detach()),
        "kl_weight": float(terms.kl_weight),
        "monitor_loss": None,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "wall_seconds": float(wall_seconds),
    }


@torch.no_grad()
def _monitor_loss(
    model: ShapeVAE,
    monitor_set: ShapeDataset,
    config: Mapping[str, Any],
    epoch: int,
    device: torch.device,
) -> float | None:
    """Return the mean loss over the monitor slice, with no gradient step, or None when empty.

    The latent samples come from a generator of its own, seeded by the run seed and the epoch, so
    the monitor leaves the global torch stream untouched. The KL weight is the final one (no step
    and no total given to the loss).
    """
    if len(monitor_set) == 0:
        return None
    seed = int(config["seed"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(rng_for(seed, _STAGE_ID, epoch, _MONITOR_MARKER).integers(2**62)))
    monitor_set.set_epoch(epoch)
    batch_size = int(config["train"]["batch_size"])
    loader = DataLoader(
        monitor_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        generator=torch.Generator(),
    )
    was_training = model.training
    model.eval()
    weighted_sum = 0.0
    count = 0
    for batch in loader:
        terms = model.loss(
            _to_device(batch.views, device), batch.betas.to(device), generator=generator
        )
        size = int(batch.betas.shape[0])
        weighted_sum += float(terms.total.detach()) * size
        count += size
    model.train(was_training)
    return weighted_sum / count


def _write_history(train_dir: Path, history: Sequence[Mapping[str, Any]]) -> None:
    """Write history.csv atomically, floats with repr precision and an empty cell for None."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HISTORY_COLUMNS)
    for row in history:
        writer.writerow(["" if row[name] is None else repr(row[name]) for name in HISTORY_COLUMNS])
    atomic_write_text(train_dir / HISTORY_FILE_NAME, buffer.getvalue())


def _remove_earlier_outputs(train_dir: Path) -> None:
    """Remove the checkpoints, history, and final model of an earlier run."""
    checkpoint_dir = train_dir / CHECKPOINT_DIRECTORY_NAME
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("*.pt"):
            path.unlink()
    for name in (HISTORY_FILE_NAME, FINAL_MODEL_FILE_NAME):
        (train_dir / name).unlink(missing_ok=True)


def _skipped_result(train_dir: Path, epochs: int) -> TrainingResult:
    """Describe a finished stage from its history.csv, for a call that skips the work."""
    rows = list(csv.DictReader((train_dir / HISTORY_FILE_NAME).open(encoding="utf-8")))
    monitors = [float(row["monitor_loss"]) for row in rows if row["monitor_loss"] != ""]
    steps = int(rows[-1]["step"]) if rows else 0
    return TrainingResult(
        train_dir,
        True,
        True,
        steps,
        steps,
        epochs,
        float(rows[-1]["loss"]) if rows else None,
        monitors[-1] if monitors else None,
        None,
    )


def _random_states() -> dict[str, Any]:
    """Return the random states of torch (CPU and CUDA), NumPy, and Python as plain values."""
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (name, keys.tolist(), int(position), int(has_gauss), float(cached)),
        "python": random.getstate(),
    }


def _set_random_states(states: Mapping[str, Any]) -> None:
    torch.set_rng_state(states["torch_cpu"])
    if states["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    name, keys, position, has_gauss, cached = states["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    version, internal, gauss = states["python"]
    random.setstate((version, tuple(internal), gauss))


def _save_checkpoint(
    train_dir: Path,
    file_name: str,
    state: _TrainingState,
    model: ShapeVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    digest: str,
    seed: int,
) -> None:
    payload = {
        "version": _CHECKPOINT_VERSION,
        "config_hash": digest,
        "seed": seed,
        "step": state.step,
        "epoch": state.epoch,
        "batches_done": state.batches_done,
        "wall_seconds": state.wall_seconds,
        "monitor_losses": list(state.monitor_losses),
        "history": list(state.history),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "random_states": _random_states(),
    }
    with atomic_path(train_dir / CHECKPOINT_DIRECTORY_NAME / file_name) as temporary:
        torch.save(payload, temporary)


def _checkpoint_step(path: Path, steps_per_epoch: int) -> int | None:
    """Return the step a checkpoint file name stands for, or None for an unknown name."""
    prefix, _, number = path.stem.partition("_")
    if not number.isdigit():
        return None
    if prefix == "step":
        return int(number)
    if prefix == "epoch":
        return (int(number) + 1) * steps_per_epoch
    return None


def _load_newest_checkpoint(
    train_dir: Path, digest: str, steps_per_epoch: int, total_steps: int
) -> dict[str, Any] | None:
    """Return the payload of the newest readable checkpoint of this configuration, or None."""
    candidates = []
    for path in (train_dir / CHECKPOINT_DIRECTORY_NAME).glob("*.pt"):
        step = _checkpoint_step(path, steps_per_epoch)
        if step is not None and step <= total_steps:
            candidates.append((step, path))
    for _, path in sorted(candidates, reverse=True):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError) as error:
            logger.warning("cannot read checkpoint %s (%s); trying an older one", path, error)
            continue
        if payload.get("config_hash") != digest or payload.get("version") != _CHECKPOINT_VERSION:
            logger.warning("checkpoint %s belongs to another configuration; ignored", path)
            continue
        return payload
    return None


def _restore(
    payload: Mapping[str, Any],
    model: ShapeVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: _TrainingState,
    device: torch.device,
) -> int:
    """Load a checkpoint payload into the model, optimizer, scheduler, state, and random streams."""
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for value in optimizer.state.values():
        for key, item in value.items():
            if isinstance(item, torch.Tensor):
                value[key] = item.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    state.step = int(payload["step"])
    state.epoch = int(payload["epoch"])
    state.batches_done = int(payload["batches_done"])
    state.wall_seconds = float(payload["wall_seconds"])
    state.history = [dict(row) for row in payload["history"]]
    state.monitor_losses = list(payload["monitor_losses"])
    _set_random_states(payload["random_states"])
    return state.step
