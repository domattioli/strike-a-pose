"""`strike-a-pose-eval` CLI (contract: specs/003-baselines-and-metrics/contracts/eval-run.md)."""

import argparse
import json
import sys

from strike_a_pose.data import DatasetError

from .config import ConfigError, load_config

EXIT_DATA, EXIT_CONFIG, EXIT_EXISTS = 1, 2, 4
HEADLINE_IMAGE = ("mean_image", "pca_k64")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="strike-a-pose-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="fit mean/PCA baselines on train; score eval splits")
    b.add_argument("--config", required=True, help="baseline config YAML")
    b.add_argument("--seed", type=int, default=None, help="override config seed")
    b.add_argument("--run-root", default=None, help="override config run_root")
    a = p.parse_args(argv)

    from .baselines import run_baselines  # heavy imports only for real runs

    try:
        cfg = load_config(a.config).with_overrides(seed=a.seed, run_root=a.run_root)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        run_dir = run_baselines(cfg, argv=["strike-a-pose-eval", *(argv or sys.argv[1:])])
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DATA
    except FileExistsError as e:
        print(f"error: run directory exists: {e}", file=sys.stderr)
        return EXIT_EXISTS
    print(f"wrote {run_dir}")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for split, m in metrics["splits"].items():
        if not m["samples"]:
            print(f"  {split}: no samples")
            continue
        parts = [
            f"{n} mse {m['image'][n]['image_mse']:.5f} iou {m['image'][n]['silhouette_iou']:.3f}"
            for n in HEADLINE_IMAGE
            if n in m["image"]
        ]
        mp = m["pose"]["mean_pose"]
        parts.append(f"mean_pose mpjpe {mp['mpjpe_mm']:.1f} mm, 2d {mp['kp2d_px']:.1f} px")
        print(f"  {split} ({m['samples']}): " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
