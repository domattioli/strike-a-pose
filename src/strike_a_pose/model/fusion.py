"""Product-of-experts fusion of per-view Gaussian experts and a standard normal prior expert.

This module implements the fusion of research R7 (FR-007). Public sources: Hinton, "Training
Products of Experts by Minimizing Contrastive Divergence" (Neural Computation 14(8), 2002), for the
product of experts; and Wu and Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised
Learning" (NeurIPS 2018, https://arxiv.org/abs/1802.05335), for the prior expert and the fusion of
one Gaussian expert per input.

Each view v of a body gives one diagonal Gaussian expert over the latent vector, with mean ``mu_v``
and log-variance ``logvar_v``. The prior expert is the standard normal N(0, I). The product of
Gaussian densities is proportional to a Gaussian density whose precision (the inverse variance) is
the sum of the precisions, and whose mean is the precision-weighted average of the means:

    precision_v      = exp(-logvar_v)
    total_precision  = 1 + sum over present views of precision_v        (the prior has precision 1)
    mu               = (sum over present views of mu_v * precision_v) / total_precision
                                                                         (the prior has mean 0)
    logvar           = -log(total_precision)

Every precision is positive, so the fused variance 1 / total_precision never exceeds the variance of
the prior or the variance of any contributing view, and adding a view never increases it (FR-007,
SC-004). The posterior of one view, the "per-view posterior" of data-model.md, is the product of the
prior expert and the expert of that view. It is the fusion of that single view, so a fusion of one
view equals the per-view posterior exactly (spec edge case: placement noise 0 with 1 view).

Tensor layout. A sample with k views (1 to 4) is stored in a fixed number of view slots with a view
mask: ``mu_v`` and ``logvar_v`` have shape (..., V, L) for V slots and L latent dimensions, and the
boolean ``view_mask`` has shape (..., V). A slot whose mask is false is absent. The values in an
absent slot are never read into the result, not even when they are NaN or infinite, and an absent
slot passes no gradient. A sample with no present view gives the prior itself (mean 0, log-variance
0).

Floating-point order. The precisions are added one slot at a time in slot order, starting from the
prior, and an absent slot adds exactly 0. Adding a non-negative term never lowers a floating-point
sum, rounding is monotone, and the same additions run in the same order for every view mask. For
the same per-view experts, the fused precision with a subset of the views is therefore never above
the fused precision with a superset, in floating point and not only in exact arithmetic. The order
does not depend on the batch size or the thread count, because the additions are separate
elementwise operations and not one reduction kernel. ``fused_precision`` returns that value. A
floating-point division is monotone too, so ``1 / fused_precision`` is the fused variance with the
same exact ordering. ``logvar`` is ``-log(total_precision)``: the library ``log`` and ``exp`` are
monotone in practice but not by construction, so compare precisions when exactness matters.
"""

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "Posterior",
    "fused_precision",
    "product_of_experts",
    "single_view_posteriors",
]

# The prior expert N(0, I) has precision 1 and mean 0.
_PRIOR_PRECISION = 1.0


class Posterior(NamedTuple):
    """A diagonal Gaussian over the latent vector: its mean and its log-variance.

    Both tensors have shape (..., L). This is the PerViewPosterior and the FusedPosterior of
    data-model.md.
    """

    mu: Tensor
    logvar: Tensor


def product_of_experts(
    mu_v: Tensor, logvar_v: Tensor, view_mask: Tensor | None = None
) -> Posterior:
    """Return the product of the prior expert and the present view experts as a Posterior.

    ``mu_v`` and ``logvar_v`` have shape (..., V, L), the mean and log-variance of the expert of
    each of V view slots. ``view_mask`` has shape (..., V) and is true for a present slot (a boolean
    tensor, or numbers in which nonzero means present); None means that every slot is present. The
    result has shape (..., L). Absent slots contribute nothing, whatever they hold.
    """
    _check_experts(mu_v, logvar_v)
    present = _present_slots(mu_v, view_mask)
    precision_v = _slot_precisions(logvar_v, present)
    weighted_v = _slot_means(mu_v, present) * precision_v
    total_precision = _add_in_slot_order(_prior_precision(precision_v), precision_v)
    weighted_total = _add_in_slot_order(_zeros_without_slots(precision_v), weighted_v)
    return Posterior(weighted_total / total_precision, -torch.log(total_precision))


def fused_precision(logvar_v: Tensor, view_mask: Tensor | None = None) -> Tensor:
    """Return the precision of the fused posterior, the prior's precision 1 plus the present ones.

    ``logvar_v`` has shape (..., V, L) and ``view_mask`` has shape (..., V), as in
    ``product_of_experts``. The result has shape (..., L). The fused variance is its reciprocal.
    For the same ``logvar_v``, a view mask that selects a subset of the views gives a result that is
    never above the result of a view mask that selects a superset, exactly in floating point (see
    the module docstring), so this is the value to compare when checking SC-004 bit for bit.
    """
    _check_slot_tensor(logvar_v, "logvar_v")
    present = _present_slots(logvar_v, view_mask)
    precision_v = _slot_precisions(logvar_v, present)
    return _add_in_slot_order(_prior_precision(precision_v), precision_v)


def single_view_posteriors(mu_v: Tensor, logvar_v: Tensor) -> Posterior:
    """Return the per-view posterior of every view slot: the prior expert times that view's expert.

    ``mu_v`` and ``logvar_v`` have shape (..., V, L), and the result has the same shape. Slot v of
    the result is ``product_of_experts`` applied to the single view v, so it equals the fused
    posterior of a sample whose only present view is v, bit for bit. The sub-sampled objective of
    research R7 averages the loss over these posteriors for the present slots.
    """
    _check_experts(mu_v, logvar_v)
    return product_of_experts(mu_v.unsqueeze(-2), logvar_v.unsqueeze(-2))


def _check_slot_tensor(tensor: Tensor, name: str) -> None:
    """Raise when a tensor lacks a slot axis or is not a floating-point tensor."""
    if tensor.dim() < 2:
        raise ValueError(
            f"{name} must have shape (..., slots, latent_dim); got the shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor; got {tensor.dtype}")


def _check_experts(mu_v: Tensor, logvar_v: Tensor) -> None:
    """Raise when the expert tensors are not floating-point slot tensors of one shape."""
    _check_slot_tensor(mu_v, "mu_v")
    _check_slot_tensor(logvar_v, "logvar_v")
    if mu_v.shape != logvar_v.shape:
        raise ValueError(
            "mu_v and logvar_v must have the same shape; "
            f"got {tuple(mu_v.shape)} and {tuple(logvar_v.shape)}"
        )


def _present_slots(experts: Tensor, view_mask: Tensor | None) -> Tensor:
    """Return the view mask as a boolean tensor of shape (..., V, 1), all true for a None mask.

    ``experts`` is any tensor of shape (..., V, L) that fixes the expected shape and the device.
    """
    slot_shape = experts.shape[:-1]
    if view_mask is None:
        return torch.ones((*slot_shape, 1), dtype=torch.bool, device=experts.device)
    if view_mask.shape != slot_shape:
        raise ValueError(
            f"view_mask must have shape {tuple(slot_shape)}, the shape of the experts without "
            f"the latent axis; got {tuple(view_mask.shape)}"
        )
    present = view_mask if view_mask.dtype == torch.bool else view_mask != 0
    return present.unsqueeze(-1)


def _slot_precisions(logvar_v: Tensor, present: Tensor) -> Tensor:
    """Return exp(-logvar_v) for present slots and exactly 0 for absent slots.

    The log-variance of an absent slot is replaced by 0 before the exponential, so a NaN or an
    infinite value there can neither reach the result nor turn the gradient into NaN (an infinite
    value times the zero gradient of a masked slot would).
    """
    neutral = torch.zeros_like(logvar_v)
    safe_logvar = torch.where(present, logvar_v, neutral)
    return torch.where(present, torch.exp(-safe_logvar), neutral)


def _slot_means(mu_v: Tensor, present: Tensor) -> Tensor:
    """Return mu_v for present slots and exactly 0 for absent slots."""
    return torch.where(present, mu_v, torch.zeros_like(mu_v))


def _prior_precision(precision_v: Tensor) -> Tensor:
    """Return the precision of the prior expert, shaped (..., L) like one fused value."""
    return precision_v.new_full((*precision_v.shape[:-2], precision_v.shape[-1]), _PRIOR_PRECISION)


def _zeros_without_slots(precision_v: Tensor) -> Tensor:
    """Return zeros shaped (..., L), the starting point of the weighted sum of means."""
    return precision_v.new_zeros((*precision_v.shape[:-2], precision_v.shape[-1]))


def _add_in_slot_order(start: Tensor, terms: Tensor) -> Tensor:
    """Return start plus the slots of terms, added one slot at a time from slot 0 upward.

    Each addition is its own elementwise operation, so the order of the additions is the slot order
    on every device and for every batch size, and an absent slot (an exact 0) leaves the sum as it
    was. A single reduction kernel would leave the order to the implementation.
    """
    total = start
    for slot in range(terms.shape[-2]):
        total = total + terms[..., slot, :]
    return total
