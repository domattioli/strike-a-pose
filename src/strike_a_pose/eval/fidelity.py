"""Rescore and fidelity report (spec 020 FR-005).

``rescore`` scores an existing checkpoint (whole-image MSE, IoU, boundary F1, region MSE) into
``<run>/fidelity.json`` without retraining; ``fidelity_rows`` compares variants with their
references under the pre-registered verdict of spec 020 Q8.
"""

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from strike_a_pose.data import PoseImageDataset
from strike_a_pose.models import load_checkpoint
from strike_a_pose.runlog import dumps_exact

from .metrics import BOUNDARY_TOL_PX
from .report import score_images

FIDELITY_VERSION = 1
REAL_GAIN_SD = 2.0  # a boundary-F1 gain counts above this many seed sd (spec 020 Q8)
MSE_COST_REL = 0.05  # whole-image MSE may rise by at most this fraction
PROBE_COST_SD = 2.0  # probe MPJPE may rise by at most this many probe sd
FIDELITY_FILE = "fidelity.json"
EPS = 1e-12  # float tolerance on the bounds only


def rescore(
    run_dir: str | Path,
    splits: tuple[str, ...] = ("val", "test"),
    batch_size: int = 64,
    checkpoint: str | Path | None = None,
) -> dict:
    """Score a run's selected weights on ``splits`` (eval mode, deterministic); the dict is
    the ``fidelity.json`` of data-model.md. Dataset, split file and threshold come from the
    run's ``config.yaml``; the weights default to ``metrics.json`` ``selection.checkpoint``
    (else ``best.pt``)."""
    from strike_a_pose.train.config import load_config
    from strike_a_pose.train.trainer import CL, load_split, predict_rows, to_float

    run_dir = Path(run_dir)
    raw = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    cfg = load_config(run_dir / "config.yaml")
    if checkpoint is None:
        mpath = run_dir / "metrics.json"
        name = "best.pt"
        if mpath.exists():
            import json

            name = json.loads(mpath.read_text()).get("selection", {}).get("checkpoint", name)
        checkpoint = run_dir / name
    model, _ = load_checkpoint(checkpoint)
    model = model.to(memory_format=CL)
    out_splits: dict = {}
    for s in splits:
        ds = PoseImageDataset(cfg.dataset_dir, split=s, split_file=cfg.split_file)
        a = load_split(ds, batch_size, masks=True)
        n = a["n"]
        if n == 0:
            out_splits[s] = {"samples": 0}
            continue
        x = to_float(a["image"]).reshape(n, -1)
        with torch.no_grad():
            r = score_images(
                lambda b: predict_rows(model, b),
                x,
                a["mask"],
                batch_size,
                cfg.iou_threshold,
                labels=(a["kp2d"], a["visible"], a["image_size"]),
                image_size=a["image_size"],
            )
        out_splits[s] = {
            "samples": n,
            "image_mse": r["image_mse"],
            "silhouette_iou": r["silhouette_iou"],
            "boundary_f1": r["boundary_f1"],
            "regions": r["regions"],
        }
    return {
        "fidelity_version": FIDELITY_VERSION,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "dataset_dir": str(cfg.dataset_dir),
        "split_file": str(cfg.split_file),
        "iou_threshold": cfg.iou_threshold,
        "boundary_tol_px": BOUNDARY_TOL_PX,
        "pose_weight": float(raw.get("pose_weight", 0.0)),
        "gradient_weight": float(raw.get("gradient_weight", 0.0)),
        "splits": out_splits,
    }


def write_fidelity(run_dir: str | Path, result: dict) -> Path:
    """Write ``<run_dir>/fidelity.json``."""
    path = Path(run_dir) / FIDELITY_FILE
    path.write_text(dumps_exact(result))
    return path


@dataclass(frozen=True)
class FidelityEntry:
    name: str
    kind: str  # "plain" | "pose"
    mse: float
    iou: float
    bf1: float
    regions: dict  # name -> {"image_mse": f, "samples": n}
    probe_mm: float | None


def load_entry(run_dir: str | Path, split: str, name: str | None = None) -> FidelityEntry:
    """One run's numbers for ``split`` from ``fidelity.json``, ``config.yaml`` and, when
    present, ``probed.txt`` (its first token is the probe run directory)."""
    import json

    run_dir = Path(run_dir)
    fid = json.loads((run_dir / FIDELITY_FILE).read_text())
    s = fid["splits"][split]
    probe_mm = None
    pt = run_dir / "probed.txt"
    if pt.exists() and pt.read_text().split():
        pm = Path(pt.read_text().split()[0]) / "metrics.json"
        t = json.loads(pm.read_text())["splits"][split]["targets"]
        probe_mm = float(t["joints_world"]["mlp"]["mpjpe_mm"])
    kind = "pose" if fid.get("pose_weight", 0.0) > 0 else "plain"
    return FidelityEntry(
        name or run_dir.name,
        kind,
        s["image_mse"],
        s["silhouette_iou"],
        s["boundary_f1"],
        s["regions"],
        probe_mm,
    )


@dataclass(frozen=True)
class FidelityRow:
    name: str
    kind: str
    ref: str  # "seeds" | "pose-ref" for variants and reference rows alike
    mse: float
    mse_rel: float | None
    iou: float
    bf1: float
    d_bf1: float | None
    d_bf1_sd: float | None
    hands: float
    feet: float
    head: float
    probe_mm: float | None
    d_probe_mm: float | None
    d_probe_sd: float | None
    verdict: str


def _stats(values: list[float], what: str) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError(f"{what}: need >= 2 values")
    return statistics.fmean(values), statistics.stdev(values)


def _row(e: FidelityEntry, ref: str, **kw) -> FidelityRow:
    reg = e.regions
    base = {
        "mse_rel": None, "d_bf1": None, "d_bf1_sd": None, "d_probe_mm": None,
        "d_probe_sd": None, "verdict": "—",
    }  # fmt: skip
    return FidelityRow(
        e.name, e.kind, ref, e.mse, iou=e.iou, bf1=e.bf1,
        hands=reg["hands"]["image_mse"], feet=reg["feet"]["image_mse"],
        head=reg["head"]["image_mse"], probe_mm=e.probe_mm, **(base | kw),
    )  # fmt: skip


def fidelity_rows(
    seeds: list[FidelityEntry],
    pose_refs: list[FidelityEntry],
    variants: list[FidelityEntry],
    split: str,
) -> list[FidelityRow]:
    """Reference rows (verdict "—") then one row per variant against its reference mean
    (plain: the ``seeds``; pose: the ``pose_refs``), with the sd of boundary F1 and of probe
    MPJPE from the ``seeds`` (ddof 1). Verdict (spec 020 Q8): NO-PROBE if the variant or the
    needed probe numbers are missing; SHARPER iff d_bf1 > REAL_GAIN_SD sd, MSE <= +MSE_COST_REL
    and probe <= +PROBE_COST_SD probe sd; SHARPER-COSTLY if the gain is real but MSE or probe
    fails; else NO-GAIN. ``split`` only labels the call (entries are per split)."""
    del split
    _, sd_bf1 = _stats([e.bf1 for e in seeds], "seeds")
    probes = [e.probe_mm for e in seeds]
    sd_probe = _stats(probes, "seeds probe")[1] if None not in probes else None
    refs = {"plain": ("seeds", seeds), "pose": ("pose-ref", pose_refs)}
    rows = [_row(e, "seeds") for e in seeds] + [_row(e, "pose-ref") for e in pose_refs]
    for v in variants:
        ref_name, group = refs[v.kind]
        if not group:
            raise ValueError(f"{v.name}: no {ref_name} reference runs")
        m_bf1 = statistics.fmean(e.bf1 for e in group)
        m_mse = statistics.fmean(e.mse for e in group)
        d_bf1 = v.bf1 - m_bf1
        d_sd = d_bf1 / sd_bf1 if sd_bf1 > 0 else math.copysign(math.inf, d_bf1)
        mse_rel = v.mse / m_mse - 1.0
        have = v.probe_mm is not None and None not in [e.probe_mm for e in group]
        have = have and sd_probe is not None
        d_mm = d_psd = None
        if have:
            d_mm = v.probe_mm - statistics.fmean(e.probe_mm for e in group)
            d_psd = d_mm / sd_probe if sd_probe > 0 else math.copysign(math.inf, d_mm)
        if not have:
            verdict = "NO-PROBE"
        elif d_sd <= REAL_GAIN_SD:
            verdict = "NO-GAIN"
        elif mse_rel <= MSE_COST_REL + EPS and d_psd <= PROBE_COST_SD + EPS:
            verdict = "SHARPER"
        else:
            verdict = "SHARPER-COSTLY"
        rows.append(
            _row(
                v,
                ref_name,
                mse_rel=mse_rel,
                d_bf1=d_bf1,
                d_bf1_sd=d_sd,
                d_probe_mm=d_mm,
                d_probe_sd=d_psd,
                verdict=verdict,
            )  # fmt: skip
        )
    return rows


def _f(x: float | None, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def format_markdown(rows: list[FidelityRow], split: str) -> str:
    """One Markdown table of the report rows for ``split``."""
    head = (
        f"| Run ({split}) | Kind | Ref | MSE | MSE vs ref | IoU | bF1 | Δ bF1 | Δ bF1 / sd | "
        "hands | feet | head | Probe (mm) | Δ probe (mm) | Δ probe / sd | Verdict |"
    )
    lines = [head, "|" + "---|" * 16]
    for r in rows:
        mr = "—" if r.mse_rel is None else f"{r.mse_rel * 100:+.1f} %"
        lines.append(
            f"| {r.name} | {r.kind} | {r.ref} | {r.mse:.5f} | {mr} | {r.iou:.4f} | {r.bf1:.4f} | "
            f"{_f(r.d_bf1, '+.4f')} | {_f(r.d_bf1_sd, '+.2f')} | {r.hands:.5f} | {r.feet:.5f} | "
            f"{r.head:.5f} | {_f(r.probe_mm, '.1f')} | {_f(r.d_probe_mm, '+.1f')} | "
            f"{_f(r.d_probe_sd, '+.2f')} | {r.verdict} |"
        )
    return "\n".join(lines)
