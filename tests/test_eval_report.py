"""Shared figure + scoring helpers (004 plan; research R0 regression guard)."""

import numpy as np
import pytest
import torch

from strike_a_pose.data import tensor_to_image
from strike_a_pose.eval import image_mse, psnr, silhouette_iou
from strike_a_pose.eval.report import (
    FIG_HEADER,
    FIG_PAD,
    figure_indices,
    grid_figure,
    score_images,
    tile_origin,
)


def _blob_images(n: int, h: int, w: int, shift: int) -> torch.Tensor:
    """Distinct off-centre rectangles per row/column, so misplacement is visible."""
    x = torch.zeros(n, h, w)
    for i in range(n):
        y0, x0 = 2 + i + shift, 1 + 2 * i + shift
        x[i, y0 : y0 + h // 3, x0 : x0 + w // 4] = 0.2 + 0.1 * i + 0.05 * shift
    return x.reshape(n, h * w)


@pytest.mark.parametrize("image_size", [(24, 16), (16, 24)])
def test_grid_tiles_match_columns(image_size):
    """R0: each tile sits at its documented offset and equals its column input."""
    h, w = image_size
    cols = [(f"c{c}", _blob_images(3, h, w, c)) for c in range(4)]
    fig = np.asarray(grid_figure(cols, image_size))
    assert fig.shape == (FIG_HEADER + 3 * h + 4 * FIG_PAD, 4 * w + 5 * FIG_PAD)
    for c, (_, imgs) in enumerate(cols):
        for r in range(3):
            x0, y0 = tile_origin(r, c, image_size)
            tile = fig[y0 : y0 + h, x0 : x0 + w]
            np.testing.assert_array_equal(tile, tensor_to_image(imgs[r].reshape(h, w)))
    # [n, 1, H, W] input gives the same figure as flattened rows
    cols4 = [(lab, imgs.reshape(-1, 1, h, w)) for lab, imgs in cols]
    np.testing.assert_array_equal(np.asarray(grid_figure(cols4, image_size)), fig)


def test_grid_rejects_ragged_columns():
    with pytest.raises(ValueError, match="rows"):
        grid_figure([("a", torch.zeros(2, 12)), ("b", torch.zeros(3, 12))], (4, 3))


def test_figure_indices():
    assert figure_indices(0, 6) == []
    assert figure_indices(3, 6) == [0, 1, 2]
    assert figure_indices(1600, 6) == [0, 320, 640, 959, 1279, 1599]


def test_score_images_matches_metrics():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(7, 20, generator=g)
    m = x > 0.5
    pred = lambda t: (t * 0.9).clamp(0, 1)  # noqa: E731
    r = score_images(pred, x, m, batch_size=3, threshold=0.4)
    p = pred(x)
    assert r["image_mse"] == pytest.approx(float(image_mse(p, x).double().mean()))
    assert r["psnr_db"] == pytest.approx(float(psnr(p, x).double().mean()))
    assert r["silhouette_iou"] == pytest.approx(float(silhouette_iou(p, m, 0.4).double().mean()))
