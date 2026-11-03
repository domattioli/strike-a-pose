"""Goal verdict report of the generative pose VAE (spec 021 FR-004).

    uv run --locked python scripts/generative.py report
        --v1 TRAIN SAMPLE RT --v2 TRAIN SAMPLE RT --l2 RUN [--l2 RUN ...] --r0 RUN [--r0 RUN ...]
        [--extra NAME TRAIN SAMPLE RT ...] [--split {val,test,both}] [--json PATH]

Prints one Markdown table per split, the V2-vs-V1 prior comparison and a last line
`VERDICT <GOAL-MET|PARTIAL|INCOMPLETE> failing=<axes or -> next=<text or ->` (verdict of V1
over the requested splits). Exit 0 on any verdict, 2 on a usage or missing-path error.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from strike_a_pose.eval.generative import (
    GENERATIVE_VERSION,
    axis_checks,
    compare_prior,
    format_markdown,
    goal_line_pixel,
    goal_verdict,
    load_gen_entry,
    mean_mse,
)


def _num(x: float) -> float | None:
    return None if math.isnan(x) else x


def _report(a: argparse.Namespace) -> int:
    splits = ("val", "test") if a.split == "both" else (a.split,)
    try:
        v1 = load_gen_entry("V1", *a.v1, splits=splits)
        v2 = load_gen_entry("V2", *a.v2, splits=splits)
        extras = [load_gen_entry(n, *p, splits=splits) for n, *p in a.extra or []]
        l2, r0 = mean_mse(a.l2, splits), mean_mse(a.r0, splits)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e!r}", file=sys.stderr)
        return 2
    entries = [v1, v2, *extras]
    out = [f"## {s}\n\n{format_markdown(entries, l2, r0, s)}" for s in splits]
    label, diff, sd = compare_prior(v1, v2)
    cmp_line = f"V2 vs V1 prior: {label} (diff {diff:+.3f}, sd {sd:.3f})"
    checks = axis_checks(v1, l2, splits)
    verdict, failing, nxt = goal_verdict(checks)
    print("\n\n".join(out) + "\n\n" + cmp_line)
    print(f"VERDICT {verdict} failing={','.join(failing) or '-'} next={nxt or '-'}")
    if a.json:
        doc = {
            "generative_version": GENERATIVE_VERSION,
            "verdict": verdict,
            "failing": failing,
            "next": nxt,
            "checks": checks,
            "goal_line_pixel": goal_line_pixel(v1, r0, splits),
            "compare": {"label": label, "diff": _num(diff), "sd": _num(sd)},
            "entries": {e.name: asdict(e) for e in entries},
        }
        Path(a.json).write_text(json.dumps(doc, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    for k in ("--v1", "--v2"):
        rp.add_argument(k, nargs=3, required=True, metavar=("TRAIN", "SAMPLE", "RT"))
    rp.add_argument("--l2", action="append", required=True, metavar="RUN")
    rp.add_argument("--r0", action="append", required=True, metavar="RUN")
    rp.add_argument("--extra", nargs=4, action="append", metavar=("NAME", "TRAIN", "SAMPLE", "RT"))
    rp.add_argument("--split", choices=("val", "test", "both"), default="val")
    rp.add_argument("--json", metavar="PATH")
    return _report(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
