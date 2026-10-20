"""Shared reporting helpers for baseline (003) and autoencoder (004) runs.

Reconstruction grid figure (contract: specs/003-baselines-and-metrics/contracts/eval-run.md)
and batched image scoring with the 003 metric functions.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from strike_a_pose.data import tensor_to_image

from .metrics import IOU_THRESHOLD, image_mse, psnr, silhouette_iou

FIG_PAD, FIG_HEADER, FIG_BG = 4, 16, 48


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
) -> dict[str, float]:
    """Mean per-image ``image_mse``, ``psnr_db``, ``silhouette_iou`` over all rows.

    ``images [n, H*W]`` float in [0, 1]; ``masks [n, H*W]`` bool; ``pred_fn`` maps a
    batch of image rows to predicted rows of the same shape.
    """
    mse, ps, iou = [], [], []
    for i in range(0, images.shape[0], batch_size):
        x = images[i : i + batch_size]
        p = pred_fn(x)
        mse.append(image_mse(p, x))
        ps.append(psnr(p, x))
        iou.append(silhouette_iou(p, masks[i : i + batch_size], threshold))
    return {
        "image_mse": float(torch.cat(mse).double().mean()),
        "psnr_db": float(torch.cat(ps).double().mean()),
        "silhouette_iou": float(torch.cat(iou).double().mean()),
    }
