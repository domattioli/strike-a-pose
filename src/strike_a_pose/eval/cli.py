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
    r = sub.add_parser(
        "round-trip", help="probe round-trip plausibility of a sample run's sets (spec 016)"
    )
    r.add_argument("--config", required=True, help="round-trip config YAML")
    r.add_argument("--sample-run", default=None, help="override config sample_run")
    r.add_argument("--probe-run", default=None, help="override config probe_run")
    r.add_argument("--smpl-model", default=None, help="override config smpl_model_path")
    r.add_argument("--run-root", default=None, help="override config run_root")
    a = p.parse_args(argv)
    full_argv = ["strike-a-pose-eval", *(sys.argv[1:] if argv is None else argv)]
    if a.cmd == "probe":
        return _probe(a, full_argv)
    if a.cmd == "sample":
        return _sample(a, full_argv)
    if a.cmd == "round-trip":
        return _round_trip(a, full_argv)

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


def _round_trip(a: argparse.Namespace, argv: list[str]) -> int:
    from .roundtrip import SET_ORDER, run_round_trip  # heavy imports only for real runs
    from .roundtrip_config import load_config as load_rt_config

    try:
        cfg = load_rt_config(a.config).with_overrides(
            sample_run=a.sample_run,
            probe_run=a.probe_run,
            smpl_model_path=a.smpl_model,
            run_root=a.run_root,
        )
        run_dir = run_round_trip(cfg, argv=argv, log=lambda m: print(m, flush=True))
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
    report = json.loads((run_dir / "samples.json").read_text())

    def f3(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    for name in (n for n in SET_ORDER if n in report["sets"]):
        s = report["sets"][name]
        rt = s["round_trip"]
        print(
            f"  {name}: plausible {f3(rt['plausible_fraction'])} valid {f3(s['valid_fraction'])}"
            f" iou_p50 {f3(rt['iou_quantiles']['p50'])}"
        )
    cal = report["round_trip_criteria"]["calibration"]
    print(f"  tau {cal['tau']:.4f} (p{cal['percentile']:g} of {cal['set']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
