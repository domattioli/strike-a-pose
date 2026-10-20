"""`strike-a-pose-data` CLI (contract: dataset-layout.md)."""

import argparse
import sys

from .assets import MissingAssetError
from .config import ConfigError, load_config
from .generate import OutputExistsError, generate_dataset
from .io import load_manifest, validate_dataset

EXIT_CONFIG, EXIT_ASSETS, EXIT_EXISTS = 2, 3, 4


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="strike-a-pose-data", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a dataset from a config and seed")
    g.add_argument("--config", required=True)
    g.add_argument("--seed", required=True, type=int)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--output-root", default=None)
    v = sub.add_parser("validate", help="validate a dataset directory against the contract")
    v.add_argument("dataset_dir")
    sp = sub.add_parser("split", help="write a frozen pose-level train/val/test split file")
    sp.add_argument("dataset_dir")
    sp.add_argument("--seed", required=True, type=int)
    sp.add_argument("--out", required=True, help="split file to write (commit under splits/)")
    sp.add_argument(
        "--fractions", nargs=3, type=float, default=None, metavar=("TRAIN", "VAL", "TEST")
    )
    sp.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    if a.cmd == "split":
        return _split(a)

    if a.cmd == "validate":
        errs = validate_dataset(a.dataset_dir)
        for e in errs:
            print(e, file=sys.stderr)
        print("OK" if not errs else f"{len(errs)} problem(s)")
        return 0 if not errs else 1
    try:
        cfg = load_config(a.config)
        out = generate_dataset(cfg, a.seed, output_root=a.output_root, overwrite=a.overwrite)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except MissingAssetError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ASSETS
    except OutputExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_EXISTS
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    print(f"wrote {out}")
    return 0


def _split(a: argparse.Namespace) -> int:
    from strike_a_pose.data.split import (
        DEFAULT_FRACTIONS,
        SPLITS,
        SplitError,
        make_split,
        write_split,
    )

    errs = validate_dataset(a.dataset_dir)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        print(f"invalid dataset: {len(errs)} problem(s)", file=sys.stderr)
        return 1
    try:
        split = make_split(load_manifest(a.dataset_dir), a.seed, a.fractions or DEFAULT_FRACTIONS)
        out = write_split(split, a.out, overwrite=a.overwrite)
    except SplitError as e:
        print(f"split error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_EXISTS
    c = split["counts"]
    counts = ", ".join(f"{k} {c[k]['poses']}/{c[k]['samples']}" for k in SPLITS)
    print(f"wrote {out} (poses/samples: {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
