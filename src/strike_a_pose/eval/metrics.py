"""Reconstruction and pose metrics (contract: specs/003-baselines-and-metrics/contracts).

Every function takes NumPy arrays or torch tensors with a leading batch dim and
returns per-sample values ``[B]`` of the same array type (``region_mask``: a bool
``[B, H, W]`` mask). Math runs in torch; NumPy inputs keep their dtype.
"""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F

Array = TypeVar("Array", np.ndarray, torch.Tensor)

IOU_THRESHOLD = 0.075  # half the renderer's ambient floor (body >= 0.15, background 0)
PSNR_MSE_FLOOR = 1e-10  # caps PSNR at 100 dB for max_val = 1
ROOT_JOINT = 0  # pelvis (smpl24)
BOUNDARY_TOL_PX = 1  # boundary-F1 match tolerance in px (spec 020 FR-002)

# Image regions (spec 009 FR-001): name -> (SMPL-24 joint indices, half-size h in px); each
# region is the union of (2h + 1)-px squares on its joints. Sized for the 001 camera ring
# (3.5 m, focal 320 px, 256 x 192, ~91 px/m); generalization: h = f * s / z per joint.
REGIONS: dict[str, tuple[tuple[int, ...], int]] = {
    "hands": ((22, 23), 10),  # left_hand, right_hand (knuckles): 21 px ~ 23 cm
    "feet": ((7, 8, 10, 11), 8),  # left/right ankle, left/right foot: 17 px ~ 19 cm each
    "head": ((15,), 14),  # head (low in the head; no face joint): 29 px ~ 32 cm
}


def _t(x) -> torch.Tensor:
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.as_tensor(x)


def _out(y: torch.Tensor, like) -> Array:
    return y.numpy() if isinstance(like, np.ndarray) else y


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def image_mse(pred: Array, target: Array) -> Array:
    """Per-image mean squared error over all non-batch elements."""
    p, t = _t(pred), _t(target)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(t.shape)}")
    return _out(_flat((p - t) ** 2).mean(dim=1), pred)


def psnr(pred: Array, target: Array, max_val: float = 1.0) -> Array:
    """Per-image PSNR in dB: ``10 log10(max_val² / max(mse, 1e-10))``."""
    mse = _t(image_mse(pred, target))
    return _out(10.0 * torch.log10(max_val**2 / mse.clamp_min(PSNR_MSE_FLOOR)), pred)


def silhouette_iou(pred: Array, mask: Array, threshold: float = IOU_THRESHOLD) -> Array:
    """IoU of ``pred > threshold`` and the true bool ``mask``; 1.0 if both are empty."""
    p, m = _t(pred), _t(mask)
    if p.shape != m.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(m.shape)}")
    a, b = _flat(p > threshold), _flat(m.bool())
    inter = (a & b).sum(dim=1).to(torch.float64)
    union = (a | b).sum(dim=1).to(torch.float64)
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))
    dtype = p.dtype if p.is_floating_point() else torch.float64
    return _out(iou.to(dtype), pred)


def gradient_loss(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Squared image-gradient difference (spec 020 FR-002), ``y, x`` float ``[B, 1, H, W]``.

    Mean squared difference of the forward differences along H plus along W; differentiable
    in ``y``; 0 iff ``y - x`` is constant per image.
    """
    dh = (y[..., 1:, :] - y[..., :-1, :]) - (x[..., 1:, :] - x[..., :-1, :])
    dw = (y[..., :, 1:] - y[..., :, :-1]) - (x[..., :, 1:] - x[..., :, :-1])
    return dh.pow(2).mean() + dw.pow(2).mean()


def _contour(b: torch.Tensor) -> torch.Tensor:
    """Contour ``b - erode(b)`` of float ``[B, 1, H, W]`` 0/1 bodies (3x3 erosion); pixels
    outside the image count as body, so a body touching the border has no contour there."""
    eroded = -F.max_pool2d(-F.pad(b, (1, 1, 1, 1), value=1.0), 3, 1, 0)
    return b - eroded


def boundary_f1(
    pred: Array,
    mask: Array,
    threshold: float = IOU_THRESHOLD,
    tol_px: int = BOUNDARY_TOL_PX,
    *,
    image_size: tuple[int, int] | None = None,
) -> Array:
    """Per-image F1 of the ``tol_px``-tolerance match between the contours of
    ``pred > threshold`` and of the true bool ``mask`` (spec 020 FR-002).

    ``pred``/``mask`` are ``[B, H, W]`` or flat ``[B, H*W]`` (then ``image_size=(H, W)`` is
    required). Both contours empty → 1.0; exactly one empty → 0.0. ``[B]`` out, in the array
    type of ``pred`` (float64 for non-float input, as :func:`silhouette_iou`).
    """
    p, m = _t(pred), _t(mask)
    if p.shape != m.shape:
        raise ValueError(f"shape mismatch {tuple(p.shape)} vs {tuple(m.shape)}")
    if p.ndim == 2:
        if image_size is None:
            raise ValueError("image_size required for flat [B, H*W] input")
        p, m = p.reshape(p.shape[0], 1, *image_size), m.reshape(m.shape[0], 1, *image_size)
    elif p.ndim == 3:
        p, m = p[:, None], m[:, None]
    else:
        raise ValueError(f"expected [B, H*W] or [B, H, W], got {tuple(p.shape)}")
    cp = _contour((p > threshold).to(torch.float32))
    cm = _contour(m.bool().to(torch.float32))
    k = 2 * tol_px + 1
    near_m = F.max_pool2d(cm, k, 1, tol_px)
    near_p = F.max_pool2d(cp, k, 1, tol_px)
    n_p, n_m = cp.sum(dim=(1, 2, 3)).double(), cm.sum(dim=(1, 2, 3)).double()
    hit_p = (cp * near_m).sum(dim=(1, 2, 3)).double()
    hit_m = (cm * near_p).sum(dim=(1, 2, 3)).double()
    prec = hit_p / n_p.clamp_min(1)
    rec = hit_m / n_m.clamp_min(1)
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp_min(1e-300), 0.0 * prec)
    f1 = torch.where((n_p == 0) & (n_m == 0), torch.ones_like(f1), f1)
    f1 = torch.where((n_p == 0) != (n_m == 0), torch.zeros_like(f1), f1)
    dtype = p.dtype if p.is_floating_point() else torch.float64
    return _out(f1.to(dtype), pred)


def mpjpe(pred: Array, target: Array, root: int = ROOT_JOINT) -> Array:
    """Mean per-joint position error after subtracting each pose's root joint.

    ``[B, J, 3]`` → ``[B]`` in the input units (metres → multiply by 1000 for mm).
    """
    p, t = _t(pred), _t(target)
    if p.shape != t.shape or p.ndim != 3:
        raise ValueError(f"expected matching [B, J, 3], got {tuple(p.shape)}, {tuple(t.shape)}")
    p = p - p[:, root : root + 1]
    t = t - t[:, root : root + 1]
    return _out(torch.linalg.vector_norm(p - t, dim=-1).mean(dim=1), pred)


def keypoint_error_2d(pred: Array, target: Array, visible: Array | None = None) -> Array:
    """Mean Euclidean pixel error over (visible) joints: ``[B, J, 2]`` → ``[B]``.

    Samples without any visible joint get NaN.
    """
    p, t = _t(pred), _t(target)
    if p.shape != t.shape or p.ndim != 3:
        raise ValueError(f"expected matching [B, J, 2], got {tuple(p.shape)}, {tuple(t.shape)}")
    err = torch.linalg.vector_norm(p - t, dim=-1)
    if visible is None:
        return _out(err.mean(dim=1), pred)
    v = _t(visible).bool()
    n = v.sum(dim=1)
    s = torch.where(v, err, torch.zeros_like(err)).sum(dim=1)
    nan = torch.full_like(s, float("nan"))
    return _out(torch.where(n > 0, s / n.clamp_min(1), nan), pred)


def region_mask(
    keypoints: Array,
    visible: Array,
    image_size: tuple[int, int],
    joints: Sequence[int],
    half_size: int,
) -> Array:
    """Pixels of a region: bool ``[B, H, W]`` from keypoints ``[B, J, 2]`` and ``visible [B, J]``.

    The pixel holding keypoint (u, v) (pixels, u right, v down) is row ⌊v⌋, column ⌊u⌋
    (the renderer samples pixel centres). Pixel (r, c) belongs to the region iff some
    listed joint is visible with ``|r - ⌊v⌋| <= half_size`` and ``|c - ⌊u⌋| <= half_size``:
    the union of the visible joints' (2h + 1)² squares, clipped to the image; invisible
    joints add nothing. Returned as the keypoints' array type.
    """
    kp, vis = _t(keypoints), _t(visible).bool()
    if kp.ndim != 3 or kp.shape[-1] != 2 or tuple(vis.shape) != tuple(kp.shape[:2]):
        raise ValueError(
            f"expected keypoints [B, J, 2] and visible [B, J], got {tuple(kp.shape)}, "
            f"{tuple(vis.shape)}"
        )
    if half_size < 0:
        raise ValueError(f"half_size must be >= 0, got {half_size}")
    h, w = image_size
    rows = torch.arange(h, dtype=torch.float64)
    cols = torch.arange(w, dtype=torch.float64)
    mask = torch.zeros((kp.shape[0], h, w), dtype=torch.bool)
    for j in joints:  # floor in float64: exact for float32 and float64 keypoints
        c = torch.floor(kp[:, j, 0].to(torch.float64))
        r = torch.floor(kp[:, j, 1].to(torch.float64))
        in_r = ((rows[None] - r[:, None]).abs() <= half_size) & vis[:, j, None]
        in_c = (cols[None] - c[:, None]).abs() <= half_size
        mask |= in_r[:, :, None] & in_c[:, None, :]
    return _out(mask, keypoints)


def masked_mse(pred: Array, target: Array, mask: Array) -> Array:
    """Per-sample mean of ``(pred - target)²`` over the ``mask`` pixels; NaN where empty.

    ``pred``, ``target`` and bool ``mask`` of equal shape ``[B, ...]``; ``[B]`` of ``pred``'s
    array type and float dtype (as :func:`image_mse`).
    """
    p, t, m = _t(pred), _t(target), _t(mask)
    if p.shape != t.shape or p.shape != m.shape:
        raise ValueError(
            f"shape mismatch {tuple(p.shape)} vs {tuple(t.shape)} vs mask {tuple(m.shape)}"
        )
    if not p.is_floating_point():
        p = p.to(torch.float64)
    se = _flat((p - t) ** 2)
    m = _flat(m.bool())
    n = m.sum(dim=1)
    s = torch.where(m, se, torch.zeros_like(se)).sum(dim=1)
    nan = torch.full_like(s, float("nan"))
    return _out(torch.where(n > 0, s / n.clamp_min(1), nan), pred)


def project_points(points: Array, K: Array, R: Array, t: Array) -> Array:
    """World points ``[B, J, 3]`` → pixels ``[B, J, 2]`` with per-sample ``K, R, t``.

    ``x_c = R x + t``; ``u = K00 x_c/z_c + K02``, ``v = K11 y_c/z_c + K12`` (001 cameras).
    """
    x, k, r, tt = _t(points), _t(K), _t(R), _t(t)
    xc = x @ r.transpose(-1, -2) + tt[:, None, :]
    z = xc[..., 2]
    z = torch.where(z.abs() < 1e-12, torch.full_like(z, 1e-12), z)
    u = k[:, None, 0, 0] * xc[..., 0] / z + k[:, None, 0, 2]
    v = k[:, None, 1, 1] * xc[..., 1] / z + k[:, None, 1, 2]
    return _out(torch.stack([u, v], dim=-1), points)
