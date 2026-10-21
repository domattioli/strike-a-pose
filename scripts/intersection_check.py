"""Limb–torso intersection check of a procedural dataset's poses (spec 014; campaign v2a gate).

    uv run --locked python scripts/intersection_check.py --config CFG --seed 0
        [--num-poses N] [--smpl-model PATH] [--threshold 0.05] [--margin-m 0.01] [--json]

Samples the config's poses exactly as `strike-a-pose-data generate` does (same table, seed and
pose stream; per-pose betas from the dataset's stream when body_shape is set), poses the body
model and counts, per pose, the arm vertices (dominant joints 16-23) inside the torso (dominant
joints 0, 3, 6, 9; per-segment inscribed elliptic cylinders, strike_a_pose.datagen.intersection).
FAIL when the fraction of poses with at least one such vertex exceeds the threshold. Writes
nothing. Exit 0 pass, 1 fail, 2 config error / AMASS source / invalid check (the rest pose
intersects, a degenerate torso segment), 3 missing body model.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from strike_a_pose.datagen.assets import MissingAssetError, sha256_file
from strike_a_pose.datagen.config import ConfigError, load_config
from strike_a_pose.datagen.intersection import (
    ARM_JOINTS,
    MARGIN_M,
    MAX_INTERSECTING_FRACTION,
    TORSO_JOINTS,
    survey,
)
from strike_a_pose.datagen.procedural import sample_procedural_poses
from strike_a_pose.datagen.shape import sample_betas
from strike_a_pose.datagen.smpl import SMPLModel

WORST = 5


def check(a: argparse.Namespace) -> tuple[int, dict]:
    cfg = load_config(a.config)
    if cfg.pose_source != "procedural":
        raise ConfigError("only pose_source: procedural has a ranges table to check")
    n = cfg.num_poses if a.num_poses is None else a.num_poses
    if n < 1:
        raise ConfigError("--num-poses must be >= 1")
    if not 0.0 <= a.threshold <= 1.0 or a.margin_m < 0:
        raise ConfigError("--threshold must be in [0, 1] and --margin-m >= 0")
    model_path = Path(a.smpl_model or cfg.smpl_model_path)
    body = SMPLModel.load(model_path)
    poses = sample_procedural_poses(n, a.seed, cfg.ranges_version)
    shape = cfg.body_shape
    betas = [
        np.zeros(cfg.num_betas)
        if shape is None
        else sample_betas(cfg.num_betas, shape.std, shape.clip, a.seed, p.index)
        for p in poses
    ]
    try:
        s = survey(body, [np.array(p.pose) for p in poses], betas, a.margin_m)
    except ValueError as e:
        raise ConfigError(f"invalid check: {e}") from e
    rest_bad = sum(c > 0 for c in s.rest_counts)
    hits = sorted(((c, i) for i, c in enumerate(s.counts) if c > 0), reverse=True)
    report = {
        "config": str(a.config), "dataset_name": cfg.dataset_name,
        "ranges_version": cfg.ranges_version, "seed": a.seed, "poses": n,
        "body_shape": shape is not None,
        "body_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "torso_joints": list(TORSO_JOINTS), "arm_joints": list(ARM_JOINTS),
        "torso_vertices": {str(k): v for k, v in s.torso_vertices.items()},
        "arm_vertices": s.arm_vertices, "margin_m": a.margin_m,
        "rest_poses_intersecting": rest_bad,
        "intersecting_poses": s.intersecting, "fraction": s.fraction, "threshold": a.threshold,
        "count_median": statistics.median(c for c, _ in hits) if hits else 0,
        "count_max": hits[0][0] if hits else 0,
        "worst_poses": [{"pose_index": i, "arm_vertices_inside": c} for c, i in hits[:WORST]],
    }  # fmt: skip
    if rest_bad:
        report["result"] = "INVALID"
        return 2, report
    report["result"] = "PASS" if s.fraction <= a.threshold else "FAIL"
    return (0 if report["result"] == "PASS" else 1), report


def _print(r: dict) -> None:
    print(
        f"intersection check: {r['config']} ({r['dataset_name']}, table {r['ranges_version']},"
        f" seed {r['seed']}, {r['poses']} poses{', body_shape' if r['body_shape'] else ''})"
    )
    print(f"body model: {r['body_model']['path']} (sha256 {r['body_model']['sha256'][:12]}…)")
    tv = ", ".join(f"{k}: {v}" for k, v in r["torso_vertices"].items())
    print(
        f"torso joints {r['torso_joints']} (vertices {tv}); arm joints 16-23 "
        f"({r['arm_vertices']} vertices); margin {r['margin_m']:.3f} m"
    )
    print(
        f"rest pose: {r['rest_poses_intersecting']} intersecting"
        + (" -> check INVALID for this body" if r["rest_poses_intersecting"] else " (valid)")
    )
    print(
        f"poses with an arm inside the torso: {r['intersecting_poses']} / {r['poses']} = "
        f"{100 * r['fraction']:.2f} % (threshold {100 * r['threshold']:.2f} %): {r['result']}"
    )
    if r["worst_poses"]:
        worst = ", ".join(f"{w['pose_index']} ({w['arm_vertices_inside']})"
                          for w in r["worst_poses"])  # fmt: skip
        print(
            f"arm vertices inside per intersecting pose: median {r['count_median']}, "
            f"max {r['count_max']}; worst poses: {worst}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="intersection_check.py", description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="a generation config (procedural source)")
    p.add_argument("--seed", required=True, type=int, help="the dataset's generation seed")
    p.add_argument("--num-poses", type=int, default=None, help="first N poses (default: all)")
    p.add_argument("--smpl-model", default=None, help="body model (default: the config's)")
    p.add_argument("--threshold", type=float, default=MAX_INTERSECTING_FRACTION)
    p.add_argument("--margin-m", type=float, default=MARGIN_M)
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    a = p.parse_args(argv)
    try:
        code, report = check(a)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    except MissingAssetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        _print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
