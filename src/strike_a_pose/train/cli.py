"""`strike-a-pose-train`: new run (``--config``) or continue one (``--resume RUN_DIR``).

Contract: specs/004-autoencoder-mvp/contracts/train-run.md.
"""

import argparse
import sys

from strike_a_pose.data import DatasetError
from strike_a_pose.eval.config import ConfigError

from .config import load_config

EXIT_DATA, EXIT_CONFIG, EXIT_EXISTS = 1, 2, 4


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="strike-a-pose-train", description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="train config YAML (new run)")
    g.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    p.add_argument("--run-root", default=None, help="override config run_root")
    a = p.parse_args(argv)
    if a.resume and (a.seed is not None or a.run_root is not None):
        p.error("--seed / --run-root cannot be combined with --resume")

    from .trainer import resume_training, run_training  # heavy imports only for real runs

    kw = {
        "argv": ["strike-a-pose-train", *(sys.argv[1:] if argv is None else argv)],
        "log": lambda m: print(m, flush=True),
    }
    try:
        if a.resume:
            run_dir = resume_training(a.resume, **kw)
        else:
            cfg = load_config(a.config).with_overrides(seed=a.seed, run_root=a.run_root)
            run_dir = run_training(cfg, **kw)
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
