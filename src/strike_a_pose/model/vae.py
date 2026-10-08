"""Shape VAE: view encoder, product-of-experts fusion, and shape decoder, with loss and sampling.

This module assembles the multi-view shape model of FR-006 to FR-009 (research R7) from
``encoder.py``, ``fusion.py``, and ``decoder.py``, and holds the training loss and the sampler of
shape coefficients. One model serves every view count from 1 to 4 (FR-009). Public sources:

* Wu and Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning" (NeurIPS
  2018, https://arxiv.org/abs/1802.05335), for the product of experts with a prior expert over any
  subset of the inputs, and for the sub-sampled objective: the loss of the joint posterior plus the
  loss of each single-input posterior.
* Kingma and Welling, "Auto-Encoding Variational Bayes" (2013, https://arxiv.org/abs/1312.6114),
  for the reparameterized latent sample, the Gaussian likelihood, and the closed-form KL divergence
  to a standard normal prior.
* Sønderby, Raiko, Maaløe, Sønderby, and Winther, "Ladder Variational Autoencoders" (NeurIPS
  2016, https://arxiv.org/abs/1602.02282), for the linear warm-up of the KL weight from 0.

Layout. A batch of B bodies has V view slots (``camera.n_cameras``, 4 by default), and slot i holds
camera i of the rig. ``ViewBatch`` carries the silhouettes (B, V, S, S), the camera encodings
(B, V, ``CAMERA_ENCODING_DIM``), and the boolean view mask (B, V) that is true for a present view.
A body may have any subset of 1 to V views present. The content of an absent slot is never read:
only present views go through the encoder, so padding that holds NaN or any other value changes
nothing, and an absent slot receives no gradient.

Model. The encoder turns each present view into a Gaussian expert ``(mu_v, logvar_v)`` over the
latent vector. The fusion multiplies the experts of the present views with the standard-normal
prior expert, which gives the fused posterior of the body. The decoder maps a latent sample to the
mean and log-scale of the shape coefficients. The posterior of a single view is the product of the
prior expert and that view's expert (``fusion.single_view_posteriors``), so the fused posterior of
a body with one view equals that view's posterior bit for bit.

Objective (research R7). For one body, with a one-sample estimate of each expectation,

    loss = NLL(joint) + mean over present views v of NLL(v)
           + w * (KL(joint) + mean over present views v of KL(v))

where ``joint`` is the fused posterior of all present views and ``v`` is the posterior of view v
alone. ``NLL(q)`` is the Gaussian negative log-likelihood of the true shape coefficients, summed
over the coefficients, at a latent sample drawn from ``q`` and decoded by the decoder. ``KL(q)`` is
the divergence of ``q`` from the standard normal prior, summed over the latent dimensions. The
weight is ``w = kl_weight * kl_warmup_factor(step, total_steps)``. The batch loss is the mean over
bodies.

Design notes. Three points of research R7 are fixed here, because the text leaves them open.

* Every term carries its own KL divergence. Research R7 names one KL term. In the cited objective
  each term is a full evidence lower bound, so the KL is taken for the joint posterior and for each
  single-view posterior, and ``kl_weight`` is the weight of the KL against the NLL in every term.
* A body with one view is scored twice, once as the joint posterior and once as the single-view
  posterior, which are the same distribution. Its two latent samples are drawn independently. This
  is the cited objective applied to a body with one input. The single-view terms give the
  single-view posterior, which the 1-view cells of the experiment use, a training signal from every
  body whatever its view count.
* The warm-up factor is 0 at step 0, rises linearly, and reaches 1 after the first 20% of the
  steps (``KL_WARMUP_FRACTION``).

Random numbers. Every random draw takes an optional ``torch.Generator`` that must live on the
device of the model, and uses the global torch generator when none is given, so a caller can make a
call reproducible and independent of other draws. The generator is read in a fixed order that
depends on the shapes only: ``loss`` draws the noise of the joint posteriors (B, L), then of the
single-view posteriors (B, V, L), and ``sample_betas_from_posterior`` draws the latent noise
(B, K, L), then the noise of the decoder head (B, K, n_betas). Two calls with generators in the same
state and with posteriors of the same shape therefore share their noise.
"""

import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from strike_a_pose.camera import CAMERA_ENCODING_DIM
from strike_a_pose.model.decoder import ShapeDecoder, gaussian_nll
from strike_a_pose.model.encoder import ViewEncoder
from strike_a_pose.model.fusion import Posterior, product_of_experts, single_view_posteriors

__all__ = [
    "KL_WARMUP_FRACTION",
    "LossTerms",
    "ShapeVAE",
    "ViewBatch",
    "ViewExperts",
    "kl_to_standard_normal",
    "kl_warmup_factor",
]

# Research R7: the KL weight rises linearly over the first 20% of the optimizer steps.
KL_WARMUP_FRACTION = 0.2


class ViewBatch(NamedTuple):
    """The views of a batch of bodies, stored in a fixed number of view slots.

    ``silhouettes`` has shape (batch, slots, S, S) for ``camera.image_size`` S: masks of 0 and 1 in
    any numeric or boolean type. ``cameras`` has shape (batch, slots, ``CAMERA_ENCODING_DIM``): the
    encoding of ``R_given`` and ``t_true`` from ``camera.encode_camera``. ``view_mask`` has shape
    (batch, slots), dtype bool, and is true where the slot holds a view. The content of a slot
    whose mask is false is never read.
    """

    silhouettes: Tensor
    cameras: Tensor
    view_mask: Tensor


class ViewExperts(NamedTuple):
    """The Gaussian expert of every view slot: the encoder output, before any fusion.

    ``mu_v`` and ``logvar_v`` have shape (batch, slots, latent_dim). An absent slot holds exact
    zeros, which the fusion ignores through the view mask. These are experts, not posteriors: the
    posterior of a view is its expert times the prior expert.
    """

    mu_v: Tensor
    logvar_v: Tensor


class LossTerms(NamedTuple):
    """The loss of a batch and its parts, each a scalar tensor that is the mean over bodies.

    ``total`` is ``nll_joint + nll_single + kl_weight * (kl_joint + kl_single)`` and is the value to
    differentiate. ``nll_single`` and ``kl_single`` are averaged over the present views of each body
    before the mean over bodies. ``kl_weight`` is the effective weight that multiplied the KL terms
    (the configured weight times the warm-up factor), a plain float.
    """

    total: Tensor
    nll_joint: Tensor
    nll_single: Tensor
    kl_joint: Tensor
    kl_single: Tensor
    kl_weight: float


def kl_warmup_factor(
    step: int, total_steps: int, warmup_fraction: float = KL_WARMUP_FRACTION
) -> float:
    """Return the linear KL warm-up factor of an optimizer step, from 0 up to 1.

    The factor is ``step / (warmup_fraction * total_steps)``, limited to 1. It is 0 at step 0, 0.5
    halfway through the warm-up, and 1 from step ``warmup_fraction * total_steps`` on, which is
    after the first 20% of the steps by default (research R7). A ``warmup_fraction`` of 0 gives 1
    at every step. ``step`` counts the optimizer steps taken before this one, from 0, and may exceed
    ``total_steps``. Raises ValueError for a negative step, a ``total_steps`` below 1, or a
    fraction outside [0, 1].
    """
    if step < 0:
        raise ValueError(f"step must be at least 0, not {step}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, not {total_steps}")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError(f"warmup_fraction must be from 0 to 1, not {warmup_fraction}")
    warmup_steps = warmup_fraction * total_steps
    if warmup_steps <= 0.0:
        return 1.0
    return min(1.0, step / warmup_steps)


def kl_to_standard_normal(posterior: Posterior) -> Tensor:
    """Return KL(q || N(0, I)) in nats for diagonal Gaussians q, summed over the latent axis.

    The closed form is ``0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)`` (Kingma and Welling 2013,
    appendix B). ``posterior`` holds ``mu`` and ``logvar`` of shape (..., L), and the result has
    shape (...). It is 0 for the prior itself and positive for any other Gaussian.
    """
    mu, logvar = posterior
    return 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar).sum(dim=-1)


class ShapeVAE(nn.Module):
    """The multi-view shape model: encoder, product-of-experts fusion, and decoder in one module.

    ``encoder`` and ``decoder`` must share ``latent_dim``. ``kl_weight`` is the final weight of the
    KL terms of the loss (``train.kl_weight``). The sub-modules are ``encoder`` and ``decoder``, so
    the state dict holds the keys ``encoder.*`` and ``decoder.*`` and nothing else (the fusion has
    no parameters). Use ``from_config`` to build the model of a resolved configuration.
    """

    def __init__(
        self, encoder: ViewEncoder, decoder: ShapeDecoder, kl_weight: float = 1e-3
    ) -> None:
        super().__init__()
        if encoder.latent_dim != decoder.latent_dim:
            raise ValueError(
                f"the encoder latent size {encoder.latent_dim} and the decoder latent size "
                f"{decoder.latent_dim} must be equal (model.latent_dim)"
            )
        if not (math.isfinite(kl_weight) and kl_weight >= 0.0):
            raise ValueError(f"kl_weight must be a finite number of at least 0, not {kl_weight}")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.latent_dim
        self.n_betas = decoder.n_betas
        self.kl_weight = float(kl_weight)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ShapeVAE":
        """Build the model of a resolved configuration.

        Reads ``model.latent_dim``, ``model.channels``, ``model.camera_embed_dim``,
        ``camera.image_size``, ``body.n_betas``, and ``train.kl_weight``.
        """
        encoder = ViewEncoder.from_config(config)
        decoder = ShapeDecoder(
            latent_dim=config["model"]["latent_dim"], n_betas=config["body"]["n_betas"]
        )
        return cls(encoder, decoder, kl_weight=config["train"]["kl_weight"])

    def encode(self, views: ViewBatch) -> ViewExperts:
        """Return the expert of every present view slot, encoding each view on its own (FR-006).

        Only the present views go through the encoder, and an absent slot holds zeros in the
        result. Encoding does not depend on the other views of the body, so a caller can encode all
        slots once and fuse different subsets of them, which keeps the per-view experts identical
        across the view-count cells.
        """
        _check_views(views)
        silhouettes, cameras, view_mask = views
        reference = next(self.encoder.parameters())
        shape = (*view_mask.shape, self.latent_dim)
        mu_v = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
        logvar_v = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
        if bool(view_mask.any()):
            present_mu, present_logvar = self.encoder(silhouettes[view_mask], cameras[view_mask])
            mu_v[view_mask] = present_mu
            logvar_v[view_mask] = present_logvar
        return ViewExperts(mu_v, logvar_v)

    def fuse(self, experts: ViewExperts, view_mask: Tensor) -> Posterior:
        """Return the fused posterior of the views that ``view_mask`` selects (FR-007).

        ``experts`` come from ``encode`` and ``view_mask`` has shape (batch, slots), true for a view
        to include. It may select fewer views than were encoded, which is how one encoding serves
        several view counts. Every body needs at least one selected view. The result has shape
        (batch, latent_dim); see ``fusion.product_of_experts`` for the arithmetic and the
        guarantee that more views never raise the variance.
        """
        _require_a_view_per_body(view_mask)
        return product_of_experts(experts.mu_v, experts.logvar_v, view_mask)

    def forward(self, views: ViewBatch) -> Posterior:
        """Return the fused posterior of the present views of each body: encode, then fuse."""
        return self.fuse(self.encode(views), views.view_mask)

    def loss(
        self,
        views: ViewBatch,
        betas: Tensor,
        step: int | None = None,
        total_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> LossTerms:
        """Return the sub-sampled negative evidence lower bound of a batch (research R7).

        ``betas`` holds the true shape coefficients, shape (batch, n_betas). Every body needs at
        least one present view. ``step`` and ``total_steps`` set the KL warm-up: the KL terms are
        weighted by ``kl_weight * kl_warmup_factor(step, total_steps)``. With neither given the
        warm-up is over and the weight is ``kl_weight``, which is the weight to use for the
        monitoring loss. Giving only one of them raises ValueError. The module docstring gives the
        formula. Latent samples come from ``generator``, or from the global torch generator.
        """
        _check_views(views)
        _require_a_view_per_body(views.view_mask)
        weight = self._kl_weight_at(step, total_steps)
        experts = self.encode(views)
        view_mask = views.view_mask
        targets = self._check_betas(betas, view_mask.shape[0], experts.mu_v)

        joint = product_of_experts(experts.mu_v, experts.logvar_v, view_mask)
        singles = single_view_posteriors(experts.mu_v, experts.logvar_v)
        joint_decoded = self.decoder(_draw_latent(joint, generator))
        single_decoded = self.decoder(_draw_latent(singles, generator))

        nll_joint = gaussian_nll(targets, joint_decoded.mean, joint_decoded.log_scale)
        nll_singles = gaussian_nll(
            targets.unsqueeze(1), single_decoded.mean, single_decoded.log_scale
        )
        nll_single = _mean_over_present(nll_singles, view_mask)
        kl_joint = kl_to_standard_normal(joint)
        kl_single = _mean_over_present(kl_to_standard_normal(singles), view_mask)

        nll_joint_mean, nll_single_mean = nll_joint.mean(), nll_single.mean()
        kl_joint_mean, kl_single_mean = kl_joint.mean(), kl_single.mean()
        total = nll_joint_mean + nll_single_mean + weight * (kl_joint_mean + kl_single_mean)
        return LossTerms(
            total, nll_joint_mean, nll_single_mean, kl_joint_mean, kl_single_mean, weight
        )

    @torch.no_grad()
    def sample_betas(
        self, views: ViewBatch, K: int, generator: torch.Generator | None = None
    ) -> Tensor:
        """Return K shape-coefficient samples per body for the present views (FR-008, research R7).

        The views are encoded and fused, and the result is ``sample_betas_from_posterior`` of the
        fused posterior: shape (batch, K, n_betas), no gradients. ``K`` is the number of latent
        samples per body (``predict.n_samples``). To score several view counts of the same bodies,
        call ``encode`` once and ``fuse`` and ``sample_betas_from_posterior`` per count.
        """
        count = _check_sample_count(K)
        return self.sample_betas_from_posterior(self(views), count, generator)

    @torch.no_grad()
    def sample_betas_from_posterior(
        self, posterior: Posterior, K: int, generator: torch.Generator | None = None
    ) -> Tensor:
        """Return K shape-coefficient samples per body from a fused posterior, no gradients.

        ``posterior`` holds ``mu`` and ``logvar`` of shape (batch, latent_dim), such as the result
        of ``fuse``. Each of the K latent samples of a body is decoded to a Gaussian over the shape
        coefficients, and one coefficient vector is drawn from it ("sampled from the head",
        research R7). The result has shape (batch, K, n_betas). Raises ValueError when K is not an
        integer of at least 1 or the posterior does not have shape (batch, latent_dim).
        """
        count = _check_sample_count(K)
        mu, logvar = posterior
        if mu.ndim != 2 or mu.shape[-1] != self.latent_dim or logvar.shape != mu.shape:
            raise ValueError(
                f"the posterior must have shape (batch, {self.latent_dim}) "
                f"(model.latent_dim); got {tuple(mu.shape)} and {tuple(logvar.shape)}"
            )
        noise = torch.randn(
            (mu.shape[0], count, self.latent_dim),
            generator=generator,
            device=mu.device,
            dtype=mu.dtype,
        )
        latent = mu.unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * noise
        return self.decoder.sample(latent, generator)

    def _kl_weight_at(self, step: int | None, total_steps: int | None) -> float:
        """Return the effective KL weight: the full weight, or the warmed-up weight of a step."""
        if step is None and total_steps is None:
            return self.kl_weight
        if step is None or total_steps is None:
            raise ValueError("pass both step and total_steps for the KL warm-up, or neither")
        return self.kl_weight * kl_warmup_factor(step, total_steps)

    def _check_betas(self, betas: Tensor, batch: int, like: Tensor) -> Tensor:
        """Return betas as a tensor of the model's dtype, after checking its shape."""
        if tuple(betas.shape) != (batch, self.n_betas):
            raise ValueError(
                f"betas must have shape ({batch}, {self.n_betas}) (batch, body.n_betas); "
                f"got {tuple(betas.shape)}"
            )
        return betas.to(dtype=like.dtype)


def _check_views(views: ViewBatch) -> None:
    """Raise ValueError when the three tensors of a view batch do not describe the same slots."""
    silhouettes, cameras, view_mask = views
    if silhouettes.ndim != 4:
        raise ValueError(
            f"silhouettes must have shape (batch, slots, S, S); got {tuple(silhouettes.shape)}"
        )
    expected_cameras = (*silhouettes.shape[:2], CAMERA_ENCODING_DIM)
    if tuple(cameras.shape) != expected_cameras:
        raise ValueError(
            f"cameras must have shape {expected_cameras} (batch, slots, camera encoding); "
            f"got {tuple(cameras.shape)}"
        )
    if view_mask.dtype != torch.bool or tuple(view_mask.shape) != tuple(silhouettes.shape[:2]):
        raise ValueError(
            f"view_mask must be a bool tensor of shape {tuple(silhouettes.shape[:2])} "
            f"(batch, slots); got {view_mask.dtype} with shape {tuple(view_mask.shape)}"
        )


def _check_sample_count(K: object) -> int:
    """Return K as an int, or raise ValueError unless it is an integer of at least 1."""
    if isinstance(K, bool) or not isinstance(K, Integral) or K < 1:
        raise ValueError(f"K must be an integer of at least 1, not {K!r}")
    return int(K)


def _require_a_view_per_body(view_mask: Tensor) -> None:
    """Raise ValueError naming the first body whose view mask selects no view."""
    has_view = view_mask.any(dim=-1)
    if not bool(has_view.all()):
        first = int(torch.nonzero(~has_view.reshape(-1))[0])
        raise ValueError(
            f"every body needs at least one present view; the body at position {first} of the "
            "batch has none"
        )


def _draw_latent(posterior: Posterior, generator: torch.Generator | None) -> Tensor:
    """Return one reparameterized latent sample per Gaussian: ``mu + exp(logvar / 2) * eps``."""
    mu, logvar = posterior
    noise = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + torch.exp(0.5 * logvar) * noise


def _mean_over_present(values: Tensor, view_mask: Tensor) -> Tensor:
    """Return the mean of values (batch, slots) over each body's present slots, shape (batch,)."""
    present_values = torch.where(view_mask, values, torch.zeros_like(values))
    return present_values.sum(dim=1) / view_mask.sum(dim=1).to(values.dtype)
