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
    q = sub.add_parser("probe", help="probe a frozen 004 encoder's latents (spec 006)")
    q.add_argument("--config", required=True, help="probe config YAML")
    q.add_argument("--seed", type=int, default=None, help="override config seed")
    q.add_argument("--run-root", default=None, help="override config run_root")
    q.add_argument("--checkpoint", default=None, help="override config checkpoint")
    m = sub.add_parser(
        "sample", help="decode N(0, I) samples of a VAE checkpoint and score them (spec 015)"
    )
    m.add_argument("--config", required=True, help="sample config YAML")
    m.add_argument("--seed", type=int, default=None, help="override config seed")
    m.add_argument("--run-root", default=None, help="override config run_root")
    m.add_argument("--checkpoint", default=None, help="override config checkpoint (the VAE)")
    m.add_argument("--reference-checkpoint", default=None, help="override the plain-AE reference")
    a = p.parse_args(argv)
    full_argv = ["strike-a-pose-eval", *(sys.argv[1:] if argv is None else argv)]
    if a.cmd == "probe":
        return _probe(a, full_argv)
    if a.cmd == "sample":
        return _sample(a, full_argv)

    from .baselines import run_baselines  # heavy imports only for real runs

    try:
        cfg = load_config(a.config).with_overrides(seed=a.seed, run_root=a.run_root)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        run_dir = run_baselines(cfg, argv=full_argv)
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


def _probe(a: argparse.Namespace, argv: list[str]) -> int:
    from .probe import run_probe  # heavy imports only for real runs
    from .probe_config import load_config as load_probe_config

    try:
        cfg = load_probe_config(a.config).with_overrides(
            seed=a.seed, run_root=a.run_root, checkpoint=a.checkpoint
        )
        run_dir = run_probe(cfg, argv=argv, log=lambda m: print(m, flush=True))
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
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for split, m in metrics["splits"].items():
        if not m["samples"]:
            print(f"  {split}: no samples")
            continue
        parts = []
        for t, r in m["targets"].items():
            key = {"yaw": "accuracy", "smpl_pose": "angle_deg"}.get(t, "mpjpe_mm")
            names = [n for n in ("ridge", "mlp", "mean_pose", "majority") if n in r]
            parts.append(f"{t} {key} " + " ".join(f"{n} {r[n][key]:.4g}" for n in names))
        print(f"  {split} ({m['samples']}): " + "; ".join(parts))
    return 0


def _sample(a: argparse.Namespace, argv: list[str]) -> int:
    from .sample import run_sample  # heavy imports only for real runs
    from .sample_config import load_config as load_sample_config

    try:
        cfg = load_sample_config(a.config).with_overrides(
            seed=a.seed,
            run_root=a.run_root,
            checkpoint=a.checkpoint,
            reference_checkpoint=a.reference_checkpoint,
        )
        run_dir = run_sample(cfg, argv=argv, log=lambda m: print(m, flush=True))
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
    for name, s in json.loads((run_dir / "samples.json").read_text())["sets"].items():
        print(f"  {name}: valid {s['valid_fraction']:.3f} nn_mse {s['nn_mse_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
