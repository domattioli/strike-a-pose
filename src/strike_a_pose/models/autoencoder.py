"""Deterministic convolutional autoencoder for 1-channel images (spec 004 FR-001).

Encoder: stride-2 3x3 conv blocks (BatchNorm + LeakyReLU 0.2) → Linear → latent.
Decoder: Linear → mirrored stride-2 4x4 transposed-conv blocks → sigmoid in [0, 1].
Optional VAE head (spec 011): a log-variance Linear beside the encoder's last Linear; the
mean stays ``encode``, so evaluation and probes never sample.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

DEFAULT_CHANNELS = (16, 32, 64, 128, 256, 256)
LEAK = 0.2


def _down(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(LEAK)]


def _up(cin: int, cout: int) -> list[nn.Module]:
    return [
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(LEAK),
    ]


class ConvAutoencoder(nn.Module):
    """``encode [B, 1, H, W] → [B, latent_dim]``; ``decode`` back to ``[B, 1, H, W]``."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 192),
        latent_dim: int = 64,
        channels: Sequence[int] = DEFAULT_CHANNELS,
        variational: bool = False,
    ) -> None:
        super().__init__()
        h, w = (int(v) for v in image_size)
        ch = [int(c) for c in channels]
        if not ch or min(ch) < 1 or latent_dim < 1:
            raise ValueError("channels must be a non-empty list of positive ints; latent_dim >= 1")
        f = 2 ** len(ch)
        if h % f or w % f:
            raise ValueError(
                f"image size {h}x{w} not divisible by 2^{len(ch)} = {f} (one per channel block)"
            )
        self.image_size, self.latent_dim, self.channels = (h, w), int(latent_dim), tuple(ch)
        self.bottleneck = (ch[-1], h // f, w // f)
        flat = ch[-1] * (h // f) * (w // f)
        down = [m for cin, cout in zip([1, *ch[:-1]], ch, strict=True) for m in _down(cin, cout)]
        self.encoder = nn.Sequential(*down, nn.Flatten(), nn.Linear(flat, latent_dim))
        rev = ch[::-1]
        up = [m for cin, cout in zip(rev[:-1], rev[1:], strict=True) for m in _up(cin, cout)]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat),
            nn.Unflatten(1, self.bottleneck),
            *up,
            nn.ConvTranspose2d(rev[-1], 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        # spec 011: the VAE's log-variance head on the flattened bottleneck, built last so
        # every other parameter initializes exactly as for the plain AE at the same seed.
        self.variational = bool(variational)
        self.logvar = nn.Linear(flat, latent_dim) if self.variational else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(mu, logvar)`` of the latent (spec 011); ``(encode(x), None)`` for the plain AE."""
        if self.logvar is None:
            return self.encoder(x), None
        h = self.encoder[:-1](x)
        return self.encoder[-1](h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def hparams(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "latent_dim": self.latent_dim,
            "channels": list(self.channels),
            "variational": self.variational,
        }


def load_checkpoint(path: str | Path) -> tuple[ConvAutoencoder, dict]:
    """Rebuild the model (eval mode) from a 004 ``checkpoint.pt``; returns (model, ckpt)."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = ConvAutoencoder(**ckpt["model_hparams"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
