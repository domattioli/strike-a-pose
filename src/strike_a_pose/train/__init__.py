"""Training (slice 004): config, AE training loop + evaluation, CLI.

Contract: ``specs/004-autoencoder-mvp/contracts/train-run.md``.
"""

from .config import TrainConfig, config_from_dict, load_config

__all__ = ["TrainConfig", "config_from_dict", "load_config"]
