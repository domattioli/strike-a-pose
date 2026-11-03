"""Pose-track report of spec 018 variants (spec 018 FR-006, `strike_a_pose.eval.pose_track`).

    uv run --locked python scripts/pose_track.py --seed RUN PROBE --seed RUN PROBE \
        [--seed RUN PROBE ...] --variant NAME RUN PROBE [--variant ...] [--split {val,test,both}]

Reads whole-image MSE (`splits.<split>.image.ae.image_mse` of the train run's metrics.json),
the pose head's MPJPE (`splits.<split>.pose_head.mpjpe_mm`, optional) and the probe's
world-frame MLP MPJPE (`splits.<split>.targets.joints_world.mlp.mpjpe_mm`). The seed runs
(>= 2) give the mean and sd per split. `--split` picks the split (default `val`). Prints
Markdown tables, the seed line and the operator ruling O2 second-seed verdict. Writes
nothing. Exit 0, or 2 on a missing file / bad input.
"""

import argparse
import json
import sys
from pathlib import Path

from strike_a_pose.eval.pose_track import (
    POSE_TARGET_MM,
    REAL_GAIN_SD,
    PoseEntry,
    rank,
    second_seed,
    seed_stats,
)


def _entry(name: str, run: str, probe: str, split: str) -> PoseEntry:
    m = json.loads((Path(run) / "metrics.json").read_text())["splits"][split]
    t = json.loads((Path(probe) / "metrics.json").read_text())["splits"][split]["targets"]
    head = (m.get("pose_head") or {}).get("mpjpe_mm")
    return PoseEntry(
        name,
        float(t["joints_world"]["mlp"]["mpjpe_mm"]),
        None if head is None else float(head),
        float(m["image"]["ae"]["image_mse"]),
    )


def _yn(v: bool | None) -> str:
    return "–" if v is None else ("yes" if v else "no")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", action="append", nargs=2, default=[], metavar=("RUN", "PROBE"))
    ap.add_argument("--variant", action="append", nargs=3, required=True,
                    metavar=("NAME", "RUN", "PROBE"))  # fmt: skip
    ap.add_argument("--split", choices=("val", "test", "both"), default="val")
    a = ap.parse_args(argv)
    if len(a.seed) < 2:
        print("error: need >= 2 --seed RUN PROBE", file=sys.stderr)
        return 2
    splits = ("val", "test") if a.split == "both" else (a.split,)
    try:
        for k, sp in enumerate(splits):
            seeds = [_entry(f"seed{i}", r, p, sp) for i, (r, p) in enumerate(a.seed)]
            mean, sd = seed_stats([s.probe_mm for s in seeds])
            ranks = rank([_entry(n, r, p, sp) for n, r, p in a.variant], seeds)
            if k:
                print()
            print(f"seeds: n={len(seeds)} mean {mean:.1f} mm sd {sd:.1f} mm ({sp})")
            print()
            print(f"| Rank | Variant | MSE vs seeds | Probe MPJPE {sp} (mm) | Δ vs seeds (mm) | "
                  f"Δ / sd | Real (< −{REAL_GAIN_SD:g} sd) | Head MPJPE {sp} (mm) | "
                  f"Probe ≤ {POSE_TARGET_MM:g} | Head ≤ {POSE_TARGET_MM:g} |")  # fmt: skip
            print("|---|---|---|---|---|---|---|---|---|---|")
            for i, r in enumerate(ranks, 1):
                head = "–" if r.head_mm is None else f"{r.head_mm:.1f}"
                print(f"| {i} | {r.name} | {r.d_mse_rel * 100:+.1f} % | {r.probe_mm:.1f} | "
                      f"{r.d_mm:+.1f} | {r.z:+.1f} | {_yn(r.real)} | {head} | "
                      f"{_yn(r.probe_ok)} | {_yn(r.head_ok)} |")  # fmt: skip
            need, why = second_seed(ranks, sd)
            print()
            print(
                f"second seed for {ranks[0].name} (operator ruling O2, {sp}): "
                f"{'yes' if need else 'no'} ({why})"
            )
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
