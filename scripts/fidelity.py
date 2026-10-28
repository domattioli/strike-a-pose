"""Rescore a run's checkpoint and report sharpness fidelity (spec 020 FR-005).

    uv run --locked python scripts/fidelity.py rescore RUN_DIR [--split val test]
        [--checkpoint P] [--force]
    uv run --locked python scripts/fidelity.py report --seed RUN --seed RUN [--seed RUN ...]
        --pose-ref RUN [--pose-ref RUN ...] --variant NAME=RUN [--variant ...]
        [--split {val,test,both}]

`rescore` writes RUN_DIR/fidelity.json (skipped with exit 0 if it exists, unless --force).
`report` prints one Markdown table per split and a seeds line; exit 2 on a usage or data error.
"""

import argparse
import statistics
import sys
from pathlib import Path

from strike_a_pose.eval.fidelity import (
    FIDELITY_FILE,
    fidelity_rows,
    format_markdown,
    load_entry,
    rescore,
    write_fidelity,
)


def _rescore(a: argparse.Namespace) -> int:
    run = Path(a.run_dir)
    if (run / FIDELITY_FILE).exists() and not a.force:
        print(f"{run / FIDELITY_FILE} exists: skipped (use --force)")
        return 0
    try:
        res = rescore(run, tuple(a.split), a.batch_size, a.checkpoint)
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {write_fidelity(run, res)}")
    return 0


def _report(a: argparse.Namespace) -> int:
    if len(a.seed) < 2:
        print("error: report needs >= 2 --seed runs", file=sys.stderr)
        return 2
    splits = ("val", "test") if a.split == "both" else (a.split,)
    var = []
    for item in a.variant or []:
        name, sep, run = item.partition("=")
        if not sep or not name or not run:
            print(f"error: --variant expects NAME=RUN, got {item!r}", file=sys.stderr)
            return 2
        var.append((name, run))
    out = []
    try:
        for sp in splits:
            seeds = [load_entry(r, sp) for r in a.seed]
            refs = [load_entry(r, sp) for r in a.pose_ref or []]
            vs = [load_entry(r, sp, name) for name, r in var]
            rows = fidelity_rows(seeds, refs, vs, sp)
            bf = [e.bf1 for e in seeds]
            pr = [e.probe_mm for e in seeds if e.probe_mm is not None]
            line = (
                f"seeds: n={len(bf)} bF1 mean {statistics.fmean(bf):.4f} "
                f"sd {statistics.stdev(bf):.4f}"
            )
            if len(pr) == len(seeds):
                line += f"; probe mean {statistics.fmean(pr):.1f} sd {statistics.stdev(pr):.2f}"
            else:
                line += "; probe n/a"
            out.append(format_markdown(rows, sp) + "\n\n" + line)
    except FileNotFoundError as e:
        print(f"error: missing {e.filename}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n\n".join(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("rescore")
    rs.add_argument("run_dir")
    rs.add_argument("--split", nargs="+", default=["val", "test"])
    rs.add_argument("--checkpoint")
    rs.add_argument("--batch-size", type=int, default=64)
    rs.add_argument("--force", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--seed", action="append", default=[], metavar="RUN")
    rp.add_argument("--pose-ref", action="append", default=[], metavar="RUN")
    rp.add_argument("--variant", action="append", metavar="NAME=RUN")
    rp.add_argument("--split", choices=("val", "test", "both"), default="val")
    a = ap.parse_args(argv)
    return _rescore(a) if a.cmd == "rescore" else _report(a)


if __name__ == "__main__":
    raise SystemExit(main())
