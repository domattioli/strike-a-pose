"""Probe round-trip plausibility of VAE samples (spec 016).

Encode an image with the VAE (μ), predict the SMPL pose and ring camera with the probe, re-render
the predicted mean-shape body with the repo's NumPy rasterizer and score the silhouette IoU against
the image. A sample is *plausible* when its IoU reaches τ, the 5th percentile of the same score on
real test images (spec 016 FR-005, FR-006).
"""

import numpy as np
from PIL import Image, ImageDraw

from strike_a_pose.datagen.cameras import Camera
from strike_a_pose.datagen.render import render
from strike_a_pose.datagen.smpl import SMPLModel

from .config import ConfigError

ROUND_TRIP_VERSION = 1
CONTROL_STREAM = 16
QUANTILES = (5, 25, 50, 75, 95)
HIST_BINS = 20
HIST_W = 480
HIST_HEADER = 20
HIST_ROW_H = 60
SET_ORDER = ("vae_prior", "ae_prior", "vae_recon_val", "real_test")
_NOT_RENDERABLE = ("camera_jitter", "body_shape")
_HIST_X0 = 110  # left edge of the bars (the set name sits left of it)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """|a ∧ b| / |a ∨ b| of two bool masks; 0.0 when the union is empty (spec 016 FR-005)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def calibrate_threshold(ious: np.ndarray, percentile: float) -> float:
    """τ = ``percentile`` of the real-image IoUs, ``method="lower"`` (an order statistic, so
    ``mean(ious >= τ) >= 1 - percentile / 100``; spec 016 FR-006)."""
    ious = np.asarray(ious, dtype=np.float64)
    if ious.size == 0:
        raise ValueError("cannot calibrate on an empty IoU array")
    return float(np.percentile(ious, percentile, method="lower"))


def camera_from_dict(d: dict) -> Camera:
    """A manifest ``cameras[k]`` entry as a ``Camera`` (spec 016 FR-005)."""
    return Camera(
        name=d["name"],
        K=np.asarray(d["K"], dtype=np.float64),
        R=np.asarray(d["R"], dtype=np.float64),
        t=np.asarray(d["t"], dtype=np.float64),
        width=int(d["width"]),
        height=int(d["height"]),
        yaw_deg=float(d["yaw_deg"]),
    )


def check_renderable(manifest: dict) -> None:
    """Refuse datasets whose renders need a per-image camera or body shape (spec 016 FR-002)."""
    cfg = manifest.get("config", {})
    for key in _NOT_RENDERABLE:
        if cfg.get(key) is not None:
            raise ConfigError(f"round trip needs mean-shape ring-camera data; dataset has {key}")


def rerender_masks(
    body: SMPLModel, poses: np.ndarray, cams: list[Camera], cam_idx: np.ndarray, num_betas: int
) -> np.ndarray:
    """Silhouettes [n, H, W] (bool) of ``poses[i]`` (zero betas) seen by ``cams[cam_idx[i]]``
    (spec 016 FR-005)."""
    n = len(poses)
    out = np.zeros((n, cams[0].height, cams[0].width), dtype=bool)
    betas = np.zeros(num_betas)
    for i in range(n):
        verts, _ = body.forward(poses[i], betas)
        out[i] = render(verts, body.faces, cams[int(cam_idx[i])])[1] > 0
    return out


def summarize(
    ious: np.ndarray,
    valid: np.ndarray,
    yaw: np.ndarray,
    empty: int,
    tau: float,
    n_classes: int,
    control: np.ndarray | None = None,
) -> dict:
    """The data-model "RT" record of one set (spec 016 FR-006, FR-007, FR-008)."""
    ious = np.asarray(ious, dtype=np.float64)
    n = int(ious.size)
    valid = np.asarray(valid, dtype=bool)
    rec: dict = {"n": n}
    if n == 0:
        rec |= {
            "iou_mean": None,
            "iou_quantiles": {f"p{q}": None for q in QUANTILES},
            "plausible_fraction": None,
            "plausible_and_valid_fraction": None,
        }
    else:
        plausible = ious >= tau
        rec |= {
            "iou_mean": float(ious.mean()),
            "iou_quantiles": {f"p{q}": float(np.percentile(ious, q)) for q in QUANTILES},
            "plausible_fraction": float(plausible.mean()),
            "plausible_and_valid_fraction": float((plausible & valid).mean()),
        }
    rec["empty_silhouette"] = int(empty)
    rec["yaw_counts"] = [int(c) for c in np.bincount(np.asarray(yaw, int), minlength=n_classes)]
    if control is None or n == 0:
        rec |= {"control_iou_median": None, "control_plausible_fraction": None}
    else:
        c = np.asarray(control, dtype=np.float64)
        rec |= {
            "control_iou_median": float(np.median(c)),
            "control_plausible_fraction": float((c >= tau).mean()),
        }
    return rec


def iou_histogram_figure(dists: dict[str, np.ndarray], tau: float) -> Image.Image:
    """Grey IoU histogram per set (one row each, dict order) with a black line at τ."""
    fig = Image.new("L", (HIST_W, HIST_HEADER + HIST_ROW_H * len(dists)), 255)
    draw = ImageDraw.Draw(fig)
    draw.text((4, 4), f"round-trip IoU, tau {tau:.4f}", fill=0)
    span = HIST_W - 4 - _HIST_X0
    bin_w = span // HIST_BINS
    for r, (name, x) in enumerate(dists.items()):
        y0 = HIST_HEADER + r * HIST_ROW_H
        draw.text((4, y0 + 2), name, fill=0)
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            continue
        counts, _ = np.histogram(x, bins=HIST_BINS, range=(0, 1))
        base, top = y0 + HIST_ROW_H - 4, HIST_ROW_H - 10
        for b, c in enumerate(counts):
            h = int(round(top * c / counts.max()))
            if h:
                xl = _HIST_X0 + b * bin_w
                draw.rectangle([xl, base - h, xl + bin_w - 2, base], fill=96)
        xt = _HIST_X0 + int(round(float(tau) * bin_w * HIST_BINS))
        draw.line([xt, y0 + 2, xt, base], fill=0)
    return fig


__all__ = [
    "calibrate_threshold",
    "camera_from_dict",
    "check_renderable",
    "iou_histogram_figure",
    "rerender_masks",
    "silhouette_iou",
    "summarize",
]
