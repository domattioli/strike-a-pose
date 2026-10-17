"""`strike-a-pose-data` CLI (contract: dataset-layout.md)."""

import argparse
import sys

from .assets import MissingAssetError
from .config import ConfigError, load_config
from .generate import OutputExistsError, generate_dataset
from .io import validate_dataset

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
    a = p.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
