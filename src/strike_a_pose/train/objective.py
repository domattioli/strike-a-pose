"""Training objective and train-batch regularization (spec 011; contract:
specs/004-autoencoder-mvp/contracts/train-run.md, "Training objective").

Every knob defaults to the plain autoencoder's computation and, while off, draws from no
RNG, so a default run is bitwise the pre-011 one. The draws of a knob that is on (shift
offsets, the VAE's noise, dropout masks) come from the global torch RNG, which the trainer
saves per epoch.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Objective:
    """Per-batch training knobs (spec 011); ``Objective()`` is the plain AE."""

    mask_weight: float = 0.0
    beta: float = 0.0
    shift_px: int = 0
    dropout: float = 0.0
    pose_weight: float = 0.0  # spec 018 FR-001
    gradient_weight: float = 0.0  # spec 020 FR-001

    @classmethod
    def from_config(cls, cfg: Any) -> "Objective":
        return cls(
            mask_weight=cfg.mask_weight,
            beta=cfg.beta,
            shift_px=cfg.shift_px,
            dropout=cfg.dropout,
            pose_weight=cfg.pose_weight,
            gradient_weight=cfg.gradient_weight,
        )


def recon_loss(y: torch.Tensor, x: torch.Tensor, mask_weight: float) -> torch.Tensor:
    """Reconstruction term: ``F.mse_loss(y, x)`` at ``mask_weight`` 0 (today's call), else the
    mask-weighted MSE ``sum(w·(y − x)²) / sum(w)`` over the batch with ``w = 1 + α·[x > 0]``
    (the silhouette is the non-zero pixels of the rendered image, spec 011 research R3)."""
    if mask_weight == 0:
        return F.mse_loss(y, x)
    w = 1.0 + mask_weight * (x > 0).to(y.dtype)
    return (w * (y - x) ** 2).sum() / w.sum()


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL(N(mu, exp(logvar)) || N(0, I))`` in nats per image (sum over the latent
    dimensions, mean over the batch)."""
    return (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).sum(dim=1).mean()


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``z = mu + exp(logvar / 2)·eps``, ``eps ~ N(0, I)`` from the global torch RNG."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def shift_batch(u8: torch.Tensor, k: int) -> torch.Tensor:
    """Each image of ``u8 [B, C, H, W]`` moved by its own integer offset ``(dy, dx)`` in
    ``[-k, k]²`` (one ``torch.randint`` draw per batch), zero filled:
    ``out[b, :, i, j] = u8[b, :, i - dy, j - dx]``. Values are copied, never interpolated;
    ``k == 0`` returns ``u8`` itself without drawing."""
    if k == 0:
        return u8
    b, _, h, w = u8.shape
    off = torch.randint(-k, k + 1, (b, 2))
    padded = F.pad(u8, (k, k, k, k))
    out = torch.empty_like(u8)
    for i, (dy, dx) in enumerate(off.tolist()):
        out[i] = padded[i, :, k - dy : k - dy + h, k - dx : k - dx + w]
    return out
