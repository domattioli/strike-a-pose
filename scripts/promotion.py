"""Promotion verdicts of campaign variants (spec 015 FR-006, `strike_a_pose.eval.promotion`).

    uv run --locked python scripts/promotion.py --r0 RUN --r0-probe PROBE --r0b RUN \
        --r0b-probe PROBE --variant NAME RUN PROBE [--variant ...]

Reads whole-image val MSE (`splits.val.image.ae.image_mse` of the train run's metrics.json),
the probe's val world-frame MLP MPJPE (`splits.val.targets.joints_world.mlp.mpjpe_mm`) and,
for a VAE run (`variational: true` in its config.yaml), the last non-empty `train_kl` of its
history.csv. Prints a Markdown table. Writes nothing. Exit 0, or 2 on a missing file.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from strike_a_pose.eval.promotion import Measures, verdict


def _measures(run: str, probe: str) -> tuple[Measures, bool]:
    r, p = Path(run), Path(probe)
    mse = json.loads((r / "metrics.json").read_text())["splits"]["val"]["image"]["ae"]["image_mse"]
    t = json.loads((p / "metrics.json").read_text())["splits"]["val"]["targets"]
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
    a = ap.parse_args(argv)
    try:
        r0, _ = _measures(a.r0, a.r0_probe)
        r0b, _ = _measures(a.r0b, a.r0b_probe)
        rows = [(n, *_measures(r, p)) for n, r, p in a.variant]
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("| Variant | Val MSE vs R0 | Probe MPJPE val (mm) | Δ mm | Verdict (015) | Reason |")
    print("|---|---|---|---|---|---|")
    for name, m, var in rows:
        try:
            v = verdict(m, r0, r0b, variational=var)
        except ValueError as e:
            print(f"error: {name}: {e}", file=sys.stderr)
            return 2
        print(f"| {name} | {v.d_mse_rel * 100:+.1f} % | {m.mpjpe_mm:.1f} | "
              f"{v.d_mpjpe_mm:+.1f} | {v.label} | {v.reason} |")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
