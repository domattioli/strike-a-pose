"""Run guards (spec 012; contract: specs/004-autoencoder-mvp/contracts/train-run.md).

Pure functions of a run's per-epoch history rows (``{"epoch", "train_loss", "val_loss", …}``:
full precision in ``checkpoint.pt``, 8 significant digits in ``history.csv``), so a resumed run
decides exactly as an uninterrupted one; and the run-dir lock that keeps one process per run
dir. No torch import (``scripts/run_status.py`` uses these as well).
"""

import json
import math
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strike_a_pose.eval.config import ConfigError

GRACE_EPOCHS = 3  # the train-loss ratio test starts after these epochs
NONFINITE_VAL_EPOCHS = 3  # consecutive non-finite val losses that abort a run
ABORTED, LOCK = "aborted.json", "lock"
TRAIN_MARKER = "strike-a-pose-train"  # in the command line of a live lock owner
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


class RunAborted(Exception):
    """The divergence guard ended the run, or a resume met an aborted run (CLI exit 5)."""


def _finite(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def early_stop_state(rows: Sequence[dict], es: Any) -> dict | None:
    """Early-stop state of a history under ``es`` (a ``train.config.EarlyStop``; None: off).

    An epoch with a finite val loss improves iff there is no anchor yet or its val loss is
    below the anchor by more than ``min_delta``; the anchor (epoch, loss) then moves to it.
    After an epoch with a finite val loss the run stops iff ``epoch >= min_epochs`` and
    ``epoch - anchor_epoch >= patience`` (NaN epochs count as epochs without improvement but
    never decide; an empty val never stops). The walk ends at the first such epoch. Returns
    ``{"anchor_epoch", "anchor_loss", "since", "earliest_stop", "fired"}``, ``fired`` None or
    ``{"epoch", "reason"}``.
    """
    if es is None:
        return None
    anchor_epoch = anchor = fired = None
    last = 0
    for r in rows:
        e, v = r["epoch"], r["val_loss"]
        last = e
        if not _finite(v):
            continue
        if anchor is None or v < anchor - es.min_delta:
            anchor_epoch, anchor = e, v
        elif e >= es.min_epochs and e - anchor_epoch >= es.patience:
            fired = {
                "epoch": e,
                "reason": (
                    f"no val loss below {anchor:.6g} (epoch {anchor_epoch}) by more than "
                    f"{es.min_delta:g} for {e - anchor_epoch} epochs "
                    f"(patience {es.patience}, min_epochs {es.min_epochs})"
                ),
            }
            break
    known = anchor_epoch is not None
    return {
        "anchor_epoch": anchor_epoch,
        "anchor_loss": anchor,
        "since": last - anchor_epoch if known else None,
        "earliest_stop": max(anchor_epoch + es.patience, es.min_epochs) if known else None,
        "fired": fired,
    }


def divergence_reason(rows: Sequence[dict], ratio: float) -> str | None:
    """Why the last epoch of ``rows`` diverged, or None. In this order: (a) its train loss is
    not finite; (b) after the first ``GRACE_EPOCHS`` epochs, its train loss exceeds ``ratio``
    × epoch 1's; (c) the last ``NONFINITE_VAL_EPOCHS`` val losses are all non-finite (an empty
    val, None, never counts)."""
    if not rows:
        return None
    e, t = rows[-1]["epoch"], rows[-1]["train_loss"]
    if not _finite(t):
        return f"train loss {t} at epoch {e} is not finite"
    t1 = rows[0]["train_loss"]
    if e > GRACE_EPOCHS and _finite(t1) and t > ratio * t1:
        return (
            f"train loss {t:.6g} at epoch {e} exceeds {ratio:g} x the epoch-1 train loss {t1:.6g}"
        )
    tail = rows[-NONFINITE_VAL_EPOCHS:]
    if len(tail) == NONFINITE_VAL_EPOCHS and all(
        r["val_loss"] is not None and not math.isfinite(r["val_loss"]) for r in tail
    ):
        return f"val loss not finite for {NONFINITE_VAL_EPOCHS} epochs ({tail[0]['epoch']}-{e})"
    return None


def _value(v: float | None) -> float | str | None:
    """A loss for JSON: non-finite values as their strings (the run JSON writer rejects inf)."""
    return v if v is None or math.isfinite(v) else str(v)


def abort_record(rows: Sequence[dict], reason: str, ratio: float) -> dict:
    """``aborted.json`` contents (spec 012 FR-005)."""
    return {
        "epoch": rows[-1]["epoch"],
        "reason": reason,
        "divergence_ratio": ratio,
        "first_train_loss": _value(rows[0]["train_loss"]),
        "last_losses": [
            {
                "epoch": r["epoch"],
                "train_loss": _value(r["train_loss"]),
                "val_loss": _value(r["val_loss"]),
            }
            for r in rows[-NONFINITE_VAL_EPOCHS:]
        ],
    }


def boot_id() -> str | None:
    """This machine boot's id (Linux), or None."""
    try:
        return BOOT_ID.read_text().strip() or None
    except OSError:
        return None


def _read_object(path: str | Path) -> dict | None:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def read_lock(path: str | Path) -> dict | None:
    """A lock file's contents, or None (missing or unreadable)."""
    return _read_object(path)


def read_aborted(path: str | Path) -> dict | None:
    """An ``aborted.json``'s contents, or None (missing or unreadable)."""
    return _read_object(path)


def lock_alive(info: Any) -> bool:
    """Whether the process a lock names is a live ``strike-a-pose-train`` of this boot (with
    ``/proc``; elsewhere: whether the pid exists). Dead, reused or other-boot pids are stale."""
    pid = info.get("pid") if isinstance(info, dict) else None
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    boot = boot_id()
    if boot and info.get("boot_id") and info["boot_id"] != boot:
        return False
    if Path("/proc/self").is_dir():
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        return TRAIN_MARKER.encode() in argv
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_lock(run_dir: str | Path, argv: Sequence[str]) -> dict:
    """Make this process the run dir's only writer (spec 012 FR-007).

    Raises ``ConfigError`` when ``run_dir/lock`` names a live owner; otherwise (no lock, or a
    stale one) writes ``{"pid", "boot_id", "argv", "started_utc"}`` through a temp file and
    re-reads it, so of two simultaneous starts the later writer wins and the other refuses.
    """
    path = Path(run_dir) / LOCK
    held = read_lock(path)
    if held is not None and lock_alive(held):
        raise ConfigError(
            f"{run_dir} is in use by process {held['pid']} ({path}): one process per run dir"
        )
    mine = {
        "pid": os.getpid(),
        "boot_id": boot_id(),
        "argv": list(argv),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    tmp = path.with_name(f"{LOCK}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(mine, sort_keys=True) + "\n")
    os.replace(tmp, path)
    if read_lock(path) != mine:
        raise ConfigError(f"{run_dir}: another process took the lock ({path})")
    return mine


def release_lock(run_dir: str | Path) -> None:
    """Remove ``run_dir/lock`` if this process holds it."""
    path = Path(run_dir) / LOCK
    held = read_lock(path)
    if held is not None and held.get("pid") == os.getpid():
        path.unlink(missing_ok=True)
