"""Models (slice 004): plain convolutional autoencoder.

Contract: ``specs/004-autoencoder-mvp/contracts/train-run.md``.
"""

from .autoencoder import DEFAULT_CHANNELS, ConvAutoencoder, load_checkpoint
from .pose_head import PoseHead, PoseTargetStats

__all__ = ["DEFAULT_CHANNELS", "ConvAutoencoder", "PoseHead", "PoseTargetStats", "load_checkpoint"]
