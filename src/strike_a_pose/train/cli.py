"""`strike-a-pose-train` CLI (contract: specs/004-autoencoder-mvp/contracts/train-run.md)."""

import argparse
import sys

from strike_a_pose.data import DatasetError
from strike_a_pose.eval.config import ConfigError

from .config import load_config

EXIT_DATA, EXIT_CONFIG, EXIT_EXISTS = 1, 2, 4


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="strike-a-pose-train", description=__doc__)
    p.add_argument("--config", required=True, help="train config YAML")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--run-root", default=None, help="override config run_root")
    a = p.parse_args(argv)

    from .trainer import run_training  # heavy imports only for real runs

    try:
        cfg = load_config(a.config).with_overrides(seed=a.seed, run_root=a.run_root)
        run_dir = run_training(
            cfg,
            argv=["strike-a-pose-train", *(sys.argv[1:] if argv is None else argv)],
            log=lambda m: print(m, flush=True),
        )
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DATA
    except FileExistsError as e:
        print(f"error: run directory exists: {e}", file=sys.stderr)
        return EXIT_EXISTS
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
