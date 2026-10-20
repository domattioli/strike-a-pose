"""Image and keypoint normalization (contract: specs/002-data-pipeline/contracts).

Images: 1 channel, float32, ``uint8 / 255`` in ``[0, 1]``.
Keypoint helpers use only slicing and arithmetic, so they accept NumPy arrays and
torch tensors with any leading batch dims (``[..., J, D]``).
"""

from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", np.ndarray, torch.Tensor)


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Grey uint8 ``[H, W]`` → float32 ``[1, H, W]`` in ``[0, 1]``."""
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError(f"expected uint8 [H, W] image, got {img.dtype} {img.shape}")
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Inverse of :func:`image_to_tensor` (``[1, H, W]`` or ``[H, W]``); clamps to [0, 1]."""
    t = t.detach().cpu()
    if t.ndim == 3:
        t = t[0]
    return torch.round(t.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Mask uint8 ``[H, W]`` (255 = body) → bool ``[1, H, W]``."""
    return torch.from_numpy(np.asarray(mask) > 127).unsqueeze(0)


def root_relative(joints: Array, root: int = 0) -> Array:
    """Subtract the root joint (default pelvis, index 0): ``[..., J, 3]``."""
    return joints - joints[..., root : root + 1, :]


def add_root(joints_rel: Array, root_pos: Array) -> Array:
    """Inverse of :func:`root_relative`; ``root_pos`` is ``[..., 3]``."""
    return joints_rel + root_pos[..., None, :]


def normalize_keypoints_2d(kp: Array, width: float, height: float) -> Array:
    """Pixels ``(u, v)`` → ``(2u/W - 1, 2v/H - 1)``: image edges ±1, centre 0.

    x and y are scaled separately (anisotropic when W != H); measure pixel errors
    after :func:`denormalize_keypoints_2d`.
    """
    u = kp[..., 0:1] * (2.0 / width) - 1.0
    v = kp[..., 1:2] * (2.0 / height) - 1.0
    return _cat(kp, u, v)


def denormalize_keypoints_2d(kp_n: Array, width: float, height: float) -> Array:
    """Inverse of :func:`normalize_keypoints_2d`."""
    u = (kp_n[..., 0:1] + 1.0) * (width / 2.0)
    v = (kp_n[..., 1:2] + 1.0) * (height / 2.0)
    return _cat(kp_n, u, v)


def _cat(like: Array, u: Array, v: Array) -> Array:
    if isinstance(like, torch.Tensor):
        return torch.cat([u, v], dim=-1)
    return np.concatenate([u, v], axis=-1)
