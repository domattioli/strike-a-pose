"""Pose head for the pose-supervised autoencoder (spec 018 FR-002).

An MLP on the latent predicting root-relative world-frame joints, the per-coordinate
standardization of its targets (train-split statistics) and the masked MSE loss.
"""

from dataclasses import dataclass

import torch
from torch import nn

POSE_TARGET = "joints_world"
POSE_JOINTS = 24
POSE_DIM = 72
POSE_HEAD_HIDDEN = (256, 256)
POSE_STD_FLOOR_M = 1e-6


def root_relative(j: torch.Tensor) -> torch.Tensor:
    """``[n, 24, 3]`` world joints minus the root joint, flattened to ``[n, 72]`` (spec 018
    FR-002; same as the probe's ``_rel``)."""
    return (j - j[:, 0:1]).reshape(j.shape[0], -1)


class PoseHead(nn.Module):
    """MLP ``latent_dim -> 256 -> 256 -> 72`` with ReLU (spec 018 FR-002)."""

    def __init__(
        self,
        latent_dim: int,
        hidden: tuple[int, ...] | list[int] = POSE_HEAD_HIDDEN,
        d_out: int = POSE_DIM,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = tuple(hidden)
        self.d_out = d_out
        layers: list[nn.Module] = []
        d = latent_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def hparams(self) -> dict:
        return {"latent_dim": self.latent_dim, "hidden": list(self.hidden), "d_out": self.d_out}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True)
class PoseTargetStats:
    """Train-split mean / population sd per coordinate and the active mask (spec 018 Q2)."""

    mean: torch.Tensor
    scale: torch.Tensor
    active: torch.Tensor

    @classmethod
    def fit(cls, y: torch.Tensor) -> "PoseTargetStats":
        if y.shape[0] == 0:
            raise ValueError("pose stats: empty")
        y = y.double()
        mean = y.mean(0)
        sd = y.std(0, unbiased=False)
        active = sd > POSE_STD_FLOOR_M
        scale = torch.where(active, sd, torch.ones_like(sd))
        return cls(mean.float(), scale.float(), active)

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.scale

    def unstandardize(self, s: torch.Tensor) -> torch.Tensor:
        out = s * self.scale + self.mean
        return torch.where(self.active, out, self.mean.expand_as(out))

    def to_dict(self) -> dict:
        return {"mean": self.mean, "scale": self.scale, "active": self.active}

    @classmethod
    def from_dict(cls, d: dict) -> "PoseTargetStats":
        return cls(d["mean"], d["scale"], d["active"])


def pose_loss(pred: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """MSE over the active coordinates of standardized ``pred`` / ``target`` (spec 018 Q2)."""
    return ((pred - target)[:, active] ** 2).mean()
