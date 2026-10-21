"""Training objective and train-batch regularization (spec 011; contract:
specs/004-autoencoder-mvp/contracts/train-run.md, "Training objective").

Every knob defaults to the plain autoencoder's computation and, while off, draws from no
RNG, so a default run is bitwise the pre-011 one. The draws of a knob that is on (shift
offsets, dropout masks) come from the global torch RNG, which the trainer saves per epoch.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Objective:
    """Per-batch training knobs (spec 011); ``Objective()`` is the plain AE."""

    shift_px: int = 0
    dropout: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "Objective":
        return cls(shift_px=cfg.shift_px, dropout=cfg.dropout)


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
