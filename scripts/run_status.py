"""Read-only status of a strike-a-pose-train run directory (spec 012; campaign check-ins).

    uv run --locked python scripts/run_status.py RUN_DIR [--json] [--now ISO8601]
        [--min-stale-min 10]

Prints epochs done / planned, the last and median epoch time, minutes since the last activity,
the best and the current val loss, the guards' state, train_kl (VAE), the run-dir lock, the
scored numbers of a finished run and one state: running, stalled, looks dead, not running,
finished, stopped early, aborted. Looks dead = no metrics.json / aborted.json, no live lock
owner, and nothing written for max(2 x median epoch time, --min-stale-min) minutes; stalled =
a live owner but nothing written for 3x that. History values come from history.csv (8
significant digits); the trainer's own records (metrics.json, aborted.json) are authoritative.
Writes nothing. Exit 0, or 2 when RUN_DIR is not a train run directory.
"""

import argparse
import csv
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from strike_a_pose.eval.config import ConfigError
from strike_a_pose.train.config import load_run_config
from strike_a_pose.train.guards import (
    ABORTED,
    LOCK,
    divergence_reason,
    early_stop_state,
    lock_alive,
    read_aborted,
    read_lock,
)

ACTIVITY = ("history.csv", "checkpoint.pt", "best.pt", "run_info.json", LOCK)  # + recon_*.png
STALL_FACTOR = 3
REGIONS = ("hands", "feet", "head")


def _float(v: str | None) -> float | None:
    return None if v in (None, "") else float(v)


def read_history(path: Path) -> list[dict]:
    """``history.csv`` rows as numbers (``train_kl`` None for runs before spec 011)."""
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return [
            {
                "epoch": int(r["epoch"]),
                "train_loss": _float(r["train_loss"]),
                "val_loss": _float(r["val_loss"]),
                "epoch_s": _float(r["epoch_s"]),
                "train_kl": _float(r.get("train_kl")),
            }
            for r in csv.DictReader(fh)
        ]


def _finite(v: float | None) -> bool:
    return v is not None and v == v and v not in (float("inf"), float("-inf"))


def _scored(metrics: dict) -> dict:
    """Val numbers of a finished run's scored weights (None when val is empty)."""
    ae = metrics.get("splits", {}).get("val", {}).get("image", {}).get("ae")
    if not ae:
        return {"val_image_mse": None, "val_region_mse_mean": None}
    regions = [ae.get("regions", {}).get(r, {}).get("image_mse") for r in REGIONS]
    mean = sum(regions) / len(regions) if all(v is not None for v in regions) else None
    return {"val_image_mse": ae.get("image_mse"), "val_region_mse_mean": mean}


def status(run_dir: str | Path, now: datetime, min_stale_min: float = 10.0) -> dict:
    """The run dir's status (JSON-ready dict); ``ValueError`` if it is not a train run dir."""
    d = Path(run_dir)
    if not (d / "config.yaml").is_file() or not (d / "run_info.json").is_file():
        raise ValueError(f"{d}: not a train run directory (no config.yaml / run_info.json)")
    try:
        cfg = load_run_config(d / "config.yaml")
    except ConfigError as e:
        raise ValueError(f"{d}: not a train run directory ({e})") from e
    rows = read_history(d / "history.csv")
    times = [r["epoch_s"] for r in rows if r["epoch_s"] is not None]
    median_s = statistics.median(times) if times else None
    files = [d / n for n in ACTIVITY] + sorted(d.glob("recon_*.png"))
    mtimes = [p.stat().st_mtime for p in files if p.is_file()]
    minutes = (now.timestamp() - max(mtimes)) / 60 if mtimes else None
    stale_after = max(2 * median_s / 60, min_stale_min) if median_s else min_stale_min
    lock = read_lock(d / LOCK)
    alive = lock is not None and lock_alive(lock)
    aborted = read_aborted(d / ABORTED) if (d / ABORTED).exists() else None
    metrics = (
        json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").is_file() else None
    )
    finite = [r for r in rows if _finite(r["val_loss"])]
    best = min(finite, key=lambda r: r["val_loss"]) if finite else None  # earliest on ties
    last_val = rows[-1]["val_loss"] if rows else None
    stale = minutes is not None and minutes > stale_after
    if aborted is not None:
        state = "aborted"
    elif metrics is not None:
        state = "stopped early" if metrics["training"].get("stopped_early") else "finished"
    elif alive:
        stalled = minutes is not None and minutes > STALL_FACTOR * stale_after
        state = "stalled" if stalled else "running"
    else:
        state = "looks dead" if stale else "not running"
    div = divergence_reason(rows, cfg.divergence_ratio) if cfg.abort_on_divergence else None
    kl = {r["epoch"]: r["train_kl"] for r in rows if r["train_kl"] is not None}
    out = {
        "run_dir": str(d),
        "run_name": cfg.run_name,
        "state": state,
        "epochs_done": len(rows),
        "epochs_planned": cfg.epochs,
        "last_epoch_s": times[-1] if times else None,
        "median_epoch_s": median_s,
        "minutes_since_activity": None if minutes is None else round(minutes, 1),
        "stale_after_min": round(stale_after, 1),
        "best_val_epoch": best["epoch"] if best else None,
        "best_val_loss": best["val_loss"] if best else None,
        "last_val_loss": last_val,
        "val_vs_best": last_val / best["val_loss"] - 1 if best and _finite(last_val) else None,
        "early_stop": early_stop_state(rows, cfg.early_stop),
        "divergence": "off"
        if not cfg.abort_on_divergence
        else (f"would abort: {div}" if div else "ok"),
        "train_kl_epoch10": kl.get(10),
        "train_kl_last": kl.get(rows[-1]["epoch"]) if rows else None,
        "lock": {"pid": lock.get("pid") if lock else None, "alive": alive},
        "selection": metrics.get("selection") if metrics else None,
        "aborted": aborted,
        **(_scored(metrics) if metrics else {"val_image_mse": None, "val_region_mse_mean": None}),
    }
    return out


def _g(v: float | None, spec: str = ".6g") -> str:
    return "-" if v is None else format(v, spec)


def render(st: dict) -> str:
    """Human-readable lines of :func:`status`."""
    lines = [
        f"run       {st['run_dir']} ({st['run_name']})",
        f"state     {st['state']}" + ("" if st["epochs_done"] else " (no epoch yet)"),
        f"epochs    {st['epochs_done']} / {st['epochs_planned']} "
        f"(last {_g(st['last_epoch_s'], '.1f')} s, median {_g(st['median_epoch_s'], '.1f')} s)",
        f"activity  {_g(st['minutes_since_activity'], '.1f')} min ago "
        f"(looks dead after {st['stale_after_min']:g} min without a live lock owner)",
        f"lock      pid {st['lock']['pid'] or '-'}, "
        + ("alive" if st["lock"]["alive"] else "no live owner"),
    ]
    vs = "" if st["val_vs_best"] is None else f" ({st['val_vs_best']:+.1%} vs best)"
    lines.append(
        f"val       best {_g(st['best_val_loss'])} at epoch {st['best_val_epoch'] or '-'}; "
        f"last {_g(st['last_val_loss'])}{vs}"
    )
    es = st["early_stop"]
    if es is None:
        lines.append("early     off")
    elif es["fired"]:
        lines.append(f"early     fired after epoch {es['fired']['epoch']}: {es['fired']['reason']}")
    else:
        lines.append(
            f"early     {es['since'] if es['since'] is not None else '-'} epochs since the anchor "
            f"epoch {es['anchor_epoch'] or '-'}; stops at epoch {es['earliest_stop'] or '-'} "
            "at the earliest"
        )
    lines.append(f"diverge   {st['divergence']}")
    if st["train_kl_last"] is not None:
        lines.append(
            f"train_kl  epoch 10: {_g(st['train_kl_epoch10'])}; last: {_g(st['train_kl_last'])}"
        )
    if st["aborted"]:
        lines.append(f"aborted   epoch {st['aborted'].get('epoch')}: {st['aborted'].get('reason')}")
    if st["selection"]:
        sel = st["selection"]
        lines.append(
            f"scored    {sel['weights']} epoch {sel['epoch']} ({sel['checkpoint']}); val "
            f"image_mse {_g(st['val_image_mse'])}, region mean {_g(st['val_region_mse_mean'])}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_status.py", description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="a strike-a-pose-train run directory")
    p.add_argument("--json", action="store_true", help="print the status as JSON")
    p.add_argument("--now", default=None, help="evaluate at this ISO 8601 time (default: now)")
    p.add_argument("--min-stale-min", type=float, default=10.0, help="dead-verdict floor (min)")
    a = p.parse_args(argv)
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(UTC)
    try:
        st = status(a.run_dir, now, a.min_stale_min)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(st, indent=2, sort_keys=True) if a.json else render(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
