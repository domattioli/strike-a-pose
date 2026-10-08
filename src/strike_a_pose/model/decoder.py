"""Heteroscedastic shape decoder: an MLP from a latent sample to coefficient mean and log-scale.

The decoder is the generative half of the shape model (FR-008, research R7). For a latent sample
``z`` of the fused posterior it returns, for each of the ``n_betas`` shape coefficients, the mean
and the log of the standard deviation (the log-scale) of a Gaussian. The spread of that Gaussian is
the model's own estimate of the shape variation that the latent sample does not explain, so a
measurement interval built from decoded samples carries the uncertainty of the latent and the
uncertainty of the head. Public sources:

* Kingma and Welling, "Auto-Encoding Variational Bayes" (2013, https://arxiv.org/abs/1312.6114),
  for the decoder as the likelihood network of a variational autoencoder (research R7).
* Nix and Weigend, "Estimating the mean and variance of the target probability distribution"
  (IEEE International Conference on Neural Networks, 1994), and Kendall and Gal, "What
  Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017,
  https://arxiv.org/abs/1703.04977), for a head that predicts a mean and a log-scale and is
  trained by the negative log-likelihood of the target.

Log-scale bound. The raw network output ``r`` for the log-scale passes through
``LOG_SCALE_LIMIT * tanh(r / LOG_SCALE_LIMIT)``. Near 0 this is ``r`` itself, so a new network
predicts a scale of about 1, which is the scale of a standard-normal shape coefficient
(contracts/config.md, ``body.beta_clip``). Far from 0 the log-scale stays inside plus or minus
``LOG_SCALE_LIMIT``, so the scale lies between 0.0067 and 148 and ``exp(-log_scale)`` cannot
overflow the negative log-likelihood. Unlike a hard clamp, the bound keeps a non-zero gradient at
every value.
"""

import math
from numbers import Integral
from typing import NamedTuple

import torch
from torch import Tensor, nn

__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_HIDDEN_LAYERS",
    "LOG_SCALE_LIMIT",
    "DecodedShape",
    "ShapeDecoder",
    "gaussian_nll",
]

# Width and depth of the hidden layers. contracts/config.md has no key for them, so they are fixed
# here: the head maps 8 or 16 latent values to 20 outputs, and two layers of this width fit that.
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 2

# The log-scale is bounded to plus or minus this value (see the module docstring).
LOG_SCALE_LIMIT = 5.0

_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


class DecodedShape(NamedTuple):
    """The Gaussian over the shape coefficients that the decoder returns for a latent sample.

    ``mean`` is the mean of each coefficient. ``log_scale`` is the natural logarithm of its
    standard deviation, so the standard deviation is ``exp(log_scale)``. Both have the shape of
    the latent input with the last axis replaced by ``n_betas``.
    """

    mean: Tensor
    log_scale: Tensor


class ShapeDecoder(nn.Module):
    """An MLP that maps a latent sample to the mean and log-scale of the shape coefficients.

    The network has ``hidden_layers`` hidden layers of width ``hidden_dim`` with SiLU activations,
    and one linear output layer of ``2 * n_betas`` values: the first ``n_betas`` are the means and
    the rest are the raw values of the log-scales (see the module docstring for the bound).
    """

    def __init__(
        self,
        latent_dim: int,
        n_betas: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("n_betas", n_betas),
            ("hidden_dim", hidden_dim),
            ("hidden_layers", hidden_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, not {value!r}")
        latent_dim, n_betas = int(latent_dim), int(n_betas)
        hidden_dim, hidden_layers = int(hidden_dim), int(hidden_layers)
        self.latent_dim = latent_dim
        self.n_betas = n_betas
        layers: list[nn.Module] = []
        width = latent_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.SiLU())
            width = hidden_dim
        layers.append(nn.Linear(width, 2 * n_betas))
        self.network = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> DecodedShape:
        """Return the mean and log-scale of the shape coefficients for latent samples ``z``.

        ``z`` has shape (..., latent_dim) with any number of leading axes, such as (batch,) for one
        sample per body or (batch, samples) for several. The result has shape (..., n_betas) for
        both fields. Raises ValueError when the last axis of ``z`` is not ``latent_dim``.
        """
        if z.ndim < 1 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must end in an axis of length {self.latent_dim} (model.latent_dim), "
                f"not shape {tuple(z.shape)}"
            )
        mean, raw_log_scale = self.network(z).split(self.n_betas, dim=-1)
        log_scale = LOG_SCALE_LIMIT * torch.tanh(raw_log_scale / LOG_SCALE_LIMIT)
        return DecodedShape(mean, log_scale)

    def sample(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Draw one shape-coefficient vector per latent sample from the decoded Gaussian.

        This is the "sampled from the head" step of research R7: ``mean + exp(log_scale) * eps``
        with ``eps`` standard normal. The result has shape (..., n_betas). The noise comes from
        ``generator`` when one is given (it must live on the device of ``z``) and from the global
        torch generator otherwise. The draw tracks gradients through ``mean`` and ``log_scale``.
        """
        decoded = self(z)
        noise = torch.randn(
            decoded.mean.shape,
            generator=generator,
            device=decoded.mean.device,
            dtype=decoded.mean.dtype,
        )
        return decoded.mean + torch.exp(decoded.log_scale) * noise


def gaussian_nll(target: Tensor, mean: Tensor, log_scale: Tensor) -> Tensor:
    """Return the negative log-likelihood, in nats, of target under independent Gaussians.

    Each coordinate adds ``0.5 ((target - mean) / scale)^2 + log_scale + 0.5 log(2 pi)`` with
    ``scale = exp(log_scale)``. The coordinates are summed over the last axis, so the result has
    the broadcast shape of the three arguments without that axis. The arguments broadcast against
    each other, so one target of shape (batch, 1, n_betas) can score a mean of shape
    (batch, views, n_betas). The caller keeps ``log_scale`` bounded, as ``ShapeDecoder`` does.
    """
    standardized = (target - mean) * torch.exp(-log_scale)
    return (0.5 * standardized * standardized + log_scale + _HALF_LOG_TWO_PI).sum(dim=-1)
