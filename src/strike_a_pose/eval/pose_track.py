"""Pose-track ranking of spec 018 variants (spec 018 FR-006).

Variants are ranked by the frozen-latent probe's world-frame MLP MPJPE against the mean and
sd (ddof 1) of >= 2 seed runs of the plain AE. A variant gains for real when it is more than
``REAL_GAIN_SD`` seed sd below the seed mean. The jointly trained pose head's MPJPE is
carried next to the probe, never instead of it. The 015 promotion rule is not touched.
"""

import math
import statistics
from dataclasses import dataclass

REAL_GAIN_SD = 2.0
POSE_TARGET_MM = 80.0
SECOND_SEED_SD = 3.0  # operator ruling O2 (spec 018): within 3 sd of the next variant / of 80 mm


@dataclass(frozen=True)
class PoseEntry:
    name: str
    probe_mm: float
    head_mm: float | None
    mse: float


@dataclass(frozen=True)
class PoseRank:
    name: str
    probe_mm: float
    head_mm: float | None
    d_mse_rel: float
    d_mm: float
    z: float
    real: bool
    probe_ok: bool
    head_ok: bool | None


def seed_stats(values: list[float]) -> tuple[float, float]:
    """Mean and sample sd (ddof 1); ``ValueError`` for < 2 values or a zero sd."""
    if len(values) < 2:
        raise ValueError("seed stats: need >= 2 values")
    sd = statistics.stdev(values)
    if not sd > 0 or not math.isfinite(sd):
        raise ValueError("seed stats: sd is 0 (identical seeds)")
    return statistics.fmean(values), sd


def rank(entries: list[PoseEntry], seeds: list[PoseEntry]) -> list[PoseRank]:
    """Variants sorted by probe MPJPE ascending (ties by name) with the seed comparison."""
    mean, sd = seed_stats([s.probe_mm for s in seeds])
    mse_mean = statistics.fmean([s.mse for s in seeds])
    out = []
    for e in sorted(entries, key=lambda e: (e.probe_mm, e.name)):
        d = e.probe_mm - mean
        out.append(
            PoseRank(
                e.name,
                e.probe_mm,
                e.head_mm,
                e.mse / mse_mean - 1.0,
                d,
                d / sd,
                d < -REAL_GAIN_SD * sd,
                e.probe_mm <= POSE_TARGET_MM,
                None if e.head_mm is None else e.head_mm <= POSE_TARGET_MM,
            )
        )
    return out


def second_seed(ranks: list[PoseRank], sd: float) -> tuple[bool, str]:
    """Operator ruling O2 (spec 018): the best variant (lowest probe MPJPE) gets a second seed
    only if its probe gain is within ``SECOND_SEED_SD`` sd of the next variant, or its probe or
    head MPJPE is within ``SECOND_SEED_SD`` sd of the 80 mm line. Returns (needed, reason)."""
    if not ranks:
        return False, "no variants"
    best, why = ranks[0], []
    if len(ranks) > 1 and ranks[1].probe_mm - best.probe_mm <= SECOND_SEED_SD * sd:
        why.append(f"within {SECOND_SEED_SD:g} sd of {ranks[1].name}")
    near = SECOND_SEED_SD * sd
    if abs(best.probe_mm - POSE_TARGET_MM) <= near:
        why.append(f"probe within {SECOND_SEED_SD:g} sd of {POSE_TARGET_MM:g} mm")
    if best.head_mm is not None and abs(best.head_mm - POSE_TARGET_MM) <= near:
        why.append(f"head within {SECOND_SEED_SD:g} sd of {POSE_TARGET_MM:g} mm")
    return bool(why), "; ".join(why) if why else "clear of the next variant and of 80 mm"
