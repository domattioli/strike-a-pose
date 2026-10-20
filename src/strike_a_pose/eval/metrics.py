"""Reconstruction and pose metrics (contract: specs/003-baselines-and-metrics/contracts).

Every function takes NumPy arrays or torch tensors with a leading batch dim and
returns per-sample values ``[B]`` of the same array type. Math runs in torch; NumPy
inputs keep their dtype.
"""

from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", np.ndarray, torch.Tensor)

IOU_THRESHOLD = 0.075  # half the renderer's ambient floor (body >= 0.15, background 0)
PSNR_MSE_FLOOR = 1e-10  # caps PSNR at 100 dB for max_val = 1
ROOT_JOINT = 0  # pelvis (smpl24)


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
