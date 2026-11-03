"""Promotion verdicts of campaign variants (spec 015 FR-006, `strike_a_pose.eval.promotion`).

    uv run --locked python scripts/promotion.py --r0 RUN --r0-probe PROBE --r0b RUN \
        --r0b-probe PROBE --variant NAME RUN PROBE [--variant ...] [--split {val,test,both}] \
        [--seed RUN PROBE --seed RUN PROBE [--seed ...]]

Reads whole-image MSE (`splits.<split>.image.ae.image_mse` of the train run's metrics.json),
the probe's world-frame MLP MPJPE (`splits.<split>.targets.joints_world.mlp.mpjpe_mm`) and,
for a VAE run (`variational: true` in its config.yaml), the last non-empty `train_kl` of its
history.csv. `--split` picks the split (default `val`); `both` prints the val table, the test
table and a table of whether the two verdicts agree, with noise gaps computed per split.
With >= 2 `--seed RUN PROBE` (plain-AE seed runs, as in pose_track.py) each table gains a
"Pose track" column (`PROMOTED (pose)` if the variant's probe MPJPE is below seed mean - 2 sd,
else `—`) and a "Pixel cost" column (whole-image MSE change vs the seed mean); with `--split
both` the agreement table gets a "Pose track val / test" column that promotes only if both
splits do. Without `--seed` the output is unchanged. 015 verdicts are not affected.
Prints Markdown tables. Writes nothing. Exit 0, or 2 on a missing file.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from strike_a_pose.eval.promotion import Measures, pose_track_verdict, verdict

POSE_LABEL = "PROMOTED (pose)"


def _measures(run: str, probe: str, split: str = "val") -> tuple[Measures, bool]:
    r, p = Path(run), Path(probe)
    mse = json.loads((r / "metrics.json").read_text())["splits"][split]["image"]["ae"]["image_mse"]
    t = json.loads((p / "metrics.json").read_text())["splits"][split]["targets"]
    mp = t["joints_world"]["mlp"]["mpjpe_mm"]
    variational = bool((yaml.safe_load((r / "config.yaml").read_text()) or {}).get("variational"))
    kl = None
    if variational:
        with (r / "history.csv").open(newline="") as f:
            vals = [row.get("train_kl") for row in csv.DictReader(f)]
        vals = [float(v) for v in vals if v not in (None, "")]
        kl = vals[-1] if vals else None
    return Measures(float(mse), float(mp), kl), variational


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for k in ("--r0", "--r0-probe", "--r0b", "--r0b-probe"):
        ap.add_argument(k, required=True)
    ap.add_argument("--variant", action="append", nargs=3, required=True,
                    metavar=("NAME", "RUN", "PROBE"))  # fmt: skip
    ap.add_argument("--seed", action="append", nargs=2, default=[], metavar=("RUN", "PROBE"))
    ap.add_argument("--split", choices=("val", "test", "both"), default="val")
    a = ap.parse_args(argv)
    splits = ("val", "test") if a.split == "both" else (a.split,)
    verdicts: dict[str, list] = {}
    pose: dict[str, list[tuple[str, str]]] = {}  # split -> [(label, pixel cost)] per variant
    try:
        for sp in splits:
            r0, _ = _measures(a.r0, a.r0_probe, sp)
            r0b, _ = _measures(a.r0b, a.r0b_probe, sp)
            rows = [(n, *_measures(r, p, sp)) for n, r, p in a.variant]
            if len(a.seed) >= 2:
                seeds = [_measures(r, p, sp)[0] for r, p in a.seed]
                mp = [x.mpjpe_mm for x in seeds]
                mse_mean = sum(x.mse for x in seeds) / len(seeds)
                pose[sp] = [
                    (
                        POSE_LABEL if pose_track_verdict(m.mpjpe_mm, mp) else "—",
                        f"{(m.mse / mse_mean - 1) * 100:+.1f} %",
                    )
                    for _, m, _ in rows
                ]
            verdicts[sp] = []
            for name, m, var in rows:
                try:
                    verdicts[sp].append((name, m, verdict(m, r0, r0b, variational=var)))
                except ValueError as e:
                    print(f"error: {name}: {e}", file=sys.stderr)
                    return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for k, sp in enumerate(splits):
        if k:
            print()
        extra = " Pose track | Pixel cost |" if pose else ""
        print(f"| Variant | {sp.capitalize()} MSE vs R0 | Probe MPJPE {sp} (mm) | Δ mm | "
              f"Verdict (015) | Reason |{extra}")  # fmt: skip
        print("|---|---|---|---|---|---|" + ("---|---|" if pose else ""))
        for i, (name, m, v) in enumerate(verdicts[sp]):
            tail = f" {pose[sp][i][0]} | {pose[sp][i][1]} |" if pose else ""
            print(f"| {name} | {v.d_mse_rel * 100:+.1f} % | {m.mpjpe_mm:.1f} | "
                  f"{v.d_mpjpe_mm:+.1f} | {v.label} | {v.reason} |{tail}")  # fmt: skip
    if a.split == "both":
        print()
        print("| Variant | Verdict val | Verdict test | Agree |"
              + (" Pose track val / test |" if pose else ""))  # fmt: skip
        print("|---|---|---|---|" + ("---|" if pose else ""))
        for i, ((name, _, v1), (_, _, v2)) in enumerate(
            zip(verdicts["val"], verdicts["test"], strict=True)
        ):
            agree = "yes" if v1.label == v2.label else "**no**"
            tail = ""
            if pose:
                pv, pt = pose["val"][i][0], pose["test"][i][0]
                both = pv == pt == POSE_LABEL
                tail = f" {POSE_LABEL if both else '—'} ({pv} / {pt}) |"
            print(f"| {name} | {v1.label} | {v2.label} | {agree} |{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
