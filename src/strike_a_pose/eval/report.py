"""Shared reporting helpers for baseline (003) and autoencoder (004) runs.

Reconstruction grid figure (contract: specs/003-baselines-and-metrics/contracts/eval-run.md)
and batched image scoring with the 003 metric functions, per-region MSE included (spec 009).
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from strike_a_pose.data import tensor_to_image
from strike_a_pose.datagen.skeleton import SMPL24_JOINT_NAMES

from .metrics import (
    BOUNDARY_TOL_PX,
    IOU_THRESHOLD,
    REGIONS,
    boundary_f1,
    image_mse,
    masked_mse,
    psnr,
    region_mask,
    silhouette_iou,
)

FIG_PAD, FIG_HEADER, FIG_BG = 4, 16, 48
# True 2D keypoints [n, J, 2] (u, v px), visibility [n, J] and image size (H, W) of scored rows.
RegionLabels = tuple[torch.Tensor, torch.Tensor, tuple[int, int]]


def figure_indices(n: int, rows: int) -> list[int]:
    """Up to ``rows`` evenly spaced indices in ``[0, n)`` (first and last included)."""
    if n == 0:
        return []
    return sorted({int(i) for i in np.linspace(0, n - 1, min(rows, n)).round()})


def tile_origin(row: int, col: int, image_size: tuple[int, int]) -> tuple[int, int]:
    """Top-left ``(x, y)`` of the tile at ``(row, col)`` in :func:`grid_figure`."""
    h, w = image_size
    return FIG_PAD + col * (w + FIG_PAD), FIG_HEADER + FIG_PAD + row * (h + FIG_PAD)


def grid_figure(
    columns: Sequence[tuple[str, torch.Tensor]], image_size: tuple[int, int]
) -> Image.Image:
    """Grey grid: a header row of labels, then one row per sample, one column each.

    ``columns`` = ``[(label, images [n, H*W] or [n, 1, H, W] in [0, 1]), ...]``, all
    with the same ``n``; every tile is shown at native resolution (no resampling).
    """
    h, w = image_size
    nc = len(columns)
    nr = int(columns[0][1].shape[0]) if nc else 0
    fig = Image.new(
        "L", (nc * w + (nc + 1) * FIG_PAD, FIG_HEADER + nr * h + (nr + 1) * FIG_PAD), FIG_BG
    )
    draw = ImageDraw.Draw(fig)
    for c, (label, imgs) in enumerate(columns):
        if imgs.shape[0] != nr:
            raise ValueError(f"column {label!r} has {imgs.shape[0]} rows, expected {nr}")
        x0, _ = tile_origin(0, c, image_size)
        draw.text((x0 + 2, 2), label, fill=255)
        for r in range(nr):
            tile = Image.fromarray(tensor_to_image(imgs[r].reshape(h, w)))
            fig.paste(tile, tile_origin(r, c, image_size))
    return fig


def score_images(
    pred_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    threshold: float = IOU_THRESHOLD,
    *,
    labels: RegionLabels | None = None,
    image_size: tuple[int, int] | None = None,
) -> dict:
    """Mean per-image ``image_mse``, ``psnr_db``, ``silhouette_iou``, ``boundary_f1`` over all rows.

    ``images [n, H*W]`` float in [0, 1]; ``masks [n, H*W]`` bool; ``pred_fn`` maps a
    batch of image rows to predicted rows of the same shape and is called once per batch.
    With ``labels`` (the rows' keypoints, visibility and image size; spec 009) the result
    gains ``regions``: per region of ``REGIONS``, ``image_mse`` = mean over the rows whose
    region is non-empty of the per-row MSE inside it (NaN if none) and ``samples`` = their
    count, from the same predictions; masks are built per batch, never stored.
    ``boundary_f1`` (spec 020 FR-004) needs the image size: ``image_size``, else ``labels[2]``,
    else ``ValueError``.
    """
    n = images.shape[0]
    if image_size is None and labels is not None:
        image_size = labels[2]
    if image_size is None:
        raise ValueError("image_size required")
    if labels is not None:
        kp, vis, (h, w) = labels
        if kp.shape[0] != n or vis.shape[0] != n or h * w != images.shape[1]:
            raise ValueError(
                f"labels for {kp.shape[0]} rows of {h}x{w} do not fit images {tuple(images.shape)}"
            )
    mse, ps, iou, bf1 = [], [], [], []
    reg: dict[str, list[torch.Tensor]] = {name: [] for name in REGIONS}
    for i in range(0, n, batch_size):
        x = images[i : i + batch_size]
        p = pred_fn(x)
        mse.append(image_mse(p, x))
        ps.append(psnr(p, x))
        iou.append(silhouette_iou(p, masks[i : i + batch_size], threshold))
        bf1.append(
            boundary_f1(
                p, masks[i : i + batch_size], threshold, BOUNDARY_TOL_PX, image_size=image_size
            )
        )
        if labels is not None:
            sl = slice(i, i + x.shape[0])
            for name, (joints, half) in REGIONS.items():
                m = region_mask(kp[sl], vis[sl], (h, w), joints, half).reshape(x.shape[0], -1)
                reg[name].append(masked_mse(p, x, m))
    out: dict = {
        "image_mse": float(torch.cat(mse).double().mean()),
        "psnr_db": float(torch.cat(ps).double().mean()),
        "silhouette_iou": float(torch.cat(iou).double().mean()),
        "boundary_f1": float(torch.cat(bf1).double().mean()),
    }
    if labels is not None:
        out["regions"] = {name: _region_record(torch.cat(v)) for name, v in reg.items()}
    return out


def _region_record(values: torch.Tensor) -> dict:
    """Mean over the non-empty samples (non-NaN values) and their count."""
    ok = ~torch.isnan(values)
    k = int(ok.sum())
    return {"image_mse": float(values[ok].double().mean()) if k else float("nan"), "samples": k}


def region_definitions() -> dict:
    """The ``metrics.json`` top-level ``regions`` block (spec 009): per region, its SMPL-24
    joints by name and its half-size in px, as scored."""
    return {
        name: {"joints": [SMPL24_JOINT_NAMES[j] for j in joints], "half_size_px": half}
        for name, (joints, half) in REGIONS.items()
    }
