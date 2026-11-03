"""Goal verdict for the generative pose VAE (spec 021 FR-004).

Fixed thresholds (spec 021 Q8) over three artifact kinds: a train run's ``metrics.json``, a
sample run's ``samples.json`` (prior valid fraction) and a round-trip run's ``samples.json``
(plausible fraction vs the shuffled-pose control). Pure functions; no I/O but the loaders.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

GENERATIVE_VERSION = 1
HEAD_MAX_MM = 80.0
MSE_COST_REL = 0.05
PRIOR_VALID_MIN = 0.56
MARGIN_MIN_PTS = 13.0
COMPARE_SD = 2.0
EPS = 1e-12
AXES = ("pose", "pixel", "valid", "margin")
NEXT_STEP = {
    "pose": "beta 0.03",
    "pixel": "beta 0.03",
    "valid": "head on z",
    "margin": "head on z",
}
# Operator ruling 2026-11-03: when pose / pixel and generation both fail, "head on z" is
# recommended first, before "beta 0.03" (spec 021 Clarifications).
STEP_ORDER = ("head on z", "beta 0.03")


@dataclass(frozen=True)
class GenEntry:
    name: str
    head_mm: dict[str, float | None]  # split -> splits[s].pose_head.mpjpe_mm
    mse: dict[str, float]  # split -> splits[s].image.ae.image_mse
    prior_valid: float | None  # sample run sets.vae_prior.valid_fraction
    prior_n: int | None  # sample run sets.vae_prior.n
    rt_plausible: float | None  # round-trip sets.vae_prior.round_trip.plausible_fraction
    rt_control: float | None  # ... .round_trip.control_plausible_fraction

    @property
    def margin_pts(self) -> float | None:
        """Round-trip margin over the shuffled-pose control, in percentage points."""
        if self.rt_plausible is None or self.rt_control is None:
            return None
        return 100.0 * (self.rt_plausible - self.rt_control)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text())


def load_gen_entry(
    name: str,
    train_run: str | Path,
    sample_run: str | Path | None = None,
    roundtrip_run: str | Path | None = None,
    splits: tuple[str, ...] = ("val", "test"),
) -> GenEntry:
    """Read the three artifacts of one model (spec 021 FR-004); a missing file raises
    ``FileNotFoundError`` naming the path, a None run leaves its fields None."""
    m = _read(Path(train_run) / "metrics.json")["splits"]
    head = {s: (m[s].get("pose_head") or {}).get("mpjpe_mm") for s in splits}
    mse = {s: m[s]["image"]["ae"]["image_mse"] for s in splits}
    valid = n = plaus = ctrl = None
    if sample_run is not None:
        p = _read(Path(sample_run) / "samples.json")["sets"]["vae_prior"]
        valid, n = p["valid_fraction"], p["n"]
    if roundtrip_run is not None:
        rt = _read(Path(roundtrip_run) / "samples.json")["sets"]["vae_prior"]["round_trip"]
        plaus, ctrl = rt["plausible_fraction"], rt["control_plausible_fraction"]
    return GenEntry(name, head, mse, valid, n, plaus, ctrl)


def mean_mse(runs: list[str | Path], splits: tuple[str, ...]) -> dict[str, float]:
    """Arithmetic mean of the image MSE over ``runs``, per split."""
    if not runs:
        raise ValueError("need >= 1 run")
    ms = [_read(Path(r) / "metrics.json")["splits"] for r in runs]
    return {s: sum(m[s]["image"]["ae"]["image_mse"] for m in ms) / len(ms) for s in splits}


def axis_checks(
    e: GenEntry, l2_mse: dict[str, float], splits: tuple[str, ...]
) -> dict[str, bool | None]:
    """Pass / fail (None: not measured) of the four axes of the goal line."""
    heads = [e.head_mm.get(s) for s in splits]
    pose = None if any(h is None for h in heads) else all(h <= HEAD_MAX_MM + EPS for h in heads)
    pixel = all(e.mse[s] <= l2_mse[s] * (1 + MSE_COST_REL) + EPS for s in splits)
    valid = None if e.prior_valid is None else e.prior_valid >= PRIOR_VALID_MIN - EPS
    margin = None if e.margin_pts is None else e.margin_pts >= MARGIN_MIN_PTS - EPS
    return {"pose": pose, "pixel": pixel, "valid": valid, "margin": margin}


def goal_verdict(checks: dict[str, bool | None]) -> tuple[str, list[str], str]:
    """``(verdict, failing axes, next step)``: INCOMPLETE if an axis is None (then the list
    names those axes), GOAL-MET if all pass, else PARTIAL with the failing axes in
    :data:`AXES` order and the unique next steps, ``head on z`` first (operator ruling)."""
    unknown = [a for a in AXES if checks[a] is None]
    if unknown:
        return "INCOMPLETE", unknown, ""
    failing = [a for a in AXES if not checks[a]]
    if not failing:
        return "GOAL-MET", [], ""
    steps = {NEXT_STEP[a] for a in failing}
    return "PARTIAL", failing, "; ".join(s for s in STEP_ORDER if s in steps)


def compare_prior(v1: GenEntry, v2: GenEntry) -> tuple[str, float, float]:
    """Two-proportion test on the prior valid fraction (spec 021 Q9): ``(label, V1 - V2, sd)``."""
    nan = float("nan")
    if None in (v1.prior_valid, v2.prior_valid, v1.prior_n, v2.prior_n):
        return "INCOMPLETE", nan, nan
    p1, p2 = v1.prior_valid, v2.prior_valid
    diff = p1 - p2
    sd = math.sqrt(p1 * (1 - p1) / v1.prior_n + p2 * (1 - p2) / v2.prior_n)
    if diff > COMPARE_SD * sd:
        return "HELPS", diff, sd
    if diff < -COMPARE_SD * sd:
        return "HURTS", diff, sd
    return "NO-DIFFERENCE", diff, sd


def goal_line_pixel(e: GenEntry, r0_mse: dict[str, float], splits: tuple[str, ...]) -> bool:
    """Informational: MSE within +5 % of the 64-d plain-AE mean on every split (never gated)."""
    return all(e.mse[s] <= r0_mse[s] * (1 + MSE_COST_REL) + EPS for s in splits)


def _f(x: float | None, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def format_markdown(
    entries: list[GenEntry], l2_mse: dict[str, float], r0_mse: dict[str, float], split: str
) -> str:
    """One Markdown table for ``split``."""
    rows = [
        "| model | head mm | MSE | MSE / L2 | pixel (≤ L2 +5 %) | vs R0 64-d +5 % "
        "| prior valid | RT margin pts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        mse = e.mse[split]
        px = "PASS" if mse <= l2_mse[split] * (1 + MSE_COST_REL) + EPS else "FAIL"
        r0 = "PASS" if goal_line_pixel(e, r0_mse, (split,)) else "FAIL"
        rows.append(
            f"| {e.name} | {_f(e.head_mm.get(split), '.1f')} | {mse:.6g} "
            f"| {mse / l2_mse[split]:.4f} | {px} | {r0} "
            f"| {_f(e.prior_valid, '.3f')} | {_f(e.margin_pts, '.1f')} |"
        )
    return "\n".join(rows)
