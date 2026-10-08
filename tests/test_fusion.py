"""Smoke tests for fusion.py: prior-times-expert product, one-view equality, and variance laws."""

import itertools
import math

import pytest
import torch

from strike_a_pose.camera import CAMERA_ENCODING_DIM
from strike_a_pose.model.encoder import ViewEncoder
from strike_a_pose.model.fusion import (
    fused_precision,
    product_of_experts,
    single_view_posteriors,
)

# Four view slots, one per camera of the rig, and a short latent vector to keep the tests quick.
SLOTS = 4
LATENT = 6

# A small stand-in encoder gives real encoder experts. Its silhouettes are 32 px, not 64 px.
ENCODER_IMAGE_SIZE = 32
ENCODER_CHANNELS = [8, 16, 32]
ENCODER_CAMERA_EMBED = 8


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    """Run on one thread, so these tests stay fast while other jobs share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def random_experts(batch=5, seed=0, spread=8.0):
    """Return random expert means and log-variances of shape (batch, SLOTS, LATENT) in float32."""
    generator = torch.Generator().manual_seed(seed)
    mu_v = torch.randn(batch, SLOTS, LATENT, generator=generator)
    logvar_v = (torch.rand(batch, SLOTS, LATENT, generator=generator) * 2 - 1) * spread
    return mu_v, logvar_v


def encoder_experts(batch=2, seed=0):
    """Return the expert means and log-variances of a small ViewEncoder run on random views."""
    generator = torch.Generator().manual_seed(seed)
    silhouettes = (
        torch.rand(batch, SLOTS, ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE, generator=generator) > 0.5
    ).float()
    cameras = torch.randn(batch, SLOTS, CAMERA_ENCODING_DIM, generator=generator)
    torch.manual_seed(seed)
    encoder = ViewEncoder(
        image_size=ENCODER_IMAGE_SIZE,
        channels=ENCODER_CHANNELS,
        latent_dim=LATENT,
        camera_embed_dim=ENCODER_CAMERA_EMBED,
    )
    with torch.no_grad():
        return encoder(silhouettes, cameras)


def view_mask(present, batch):
    """Return a boolean mask of shape (batch, SLOTS) that is true for the slots in ``present``."""
    mask = torch.zeros(batch, SLOTS, dtype=torch.bool)
    mask[:, list(present)] = True
    return mask


def all_subsets():
    """Return every non-empty subset of the view slots, as a tuple of slot numbers in order."""
    return [
        subset
        for size in range(1, SLOTS + 1)
        for subset in itertools.combinations(range(SLOTS), size)
    ]


def gaussian_log_density(point, mu, logvar):
    """Return the log-density of diagonal Gaussians N(mu, exp(logvar)) at ``point``.

    The density is summed over the latent axis, which is the last axis of every argument.
    """
    standardized = (point - mu) * torch.exp(-0.5 * logvar)
    return -0.5 * (standardized.pow(2) + logvar + math.log(2.0 * math.pi)).sum(dim=-1)


def test_two_expert_product_matches_the_closed_form():
    # Two views and the prior expert N(0, I), in double precision.
    # Latent 0: view A has variance 1 (precision 1) and view B has variance 0.25 (precision 4).
    # With the prior's precision 1 the total is 6, the mean is (1 * 1 + 4 * -3) / 6 = -11 / 6,
    # and the variance is 1 / 6.
    # Latent 1: view A has variance 4 (precision 0.25) and view B has variance 0.5 (precision 2).
    # The total is 13 / 4, the mean is (0.25 * 0.5 + 2 * 2) / (13 / 4) = 33 / 26, and the
    # variance is 4 / 13.
    mu_v = torch.tensor([[[1.0, 0.5], [-3.0, 2.0]]], dtype=torch.float64)
    logvar_v = torch.log(torch.tensor([[[1.0, 4.0], [0.25, 0.5]]], dtype=torch.float64))
    fused = product_of_experts(mu_v, logvar_v)
    expected_mu = torch.tensor([[-11.0 / 6.0, 33.0 / 26.0]], dtype=torch.float64)
    expected_variance = torch.tensor([[1.0 / 6.0, 4.0 / 13.0]], dtype=torch.float64)
    torch.testing.assert_close(fused.mu, expected_mu, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fused.logvar.exp(), expected_variance, rtol=1e-12, atol=1e-12)

    # An absent slot between the two views, holding NaN and infinity, changes nothing.
    nan_mu = torch.full((1, 1, 2), float("nan"), dtype=torch.float64)
    inf_logvar = torch.full((1, 1, 2), float("inf"), dtype=torch.float64)
    padded_mu = torch.cat([mu_v[:, :1], nan_mu, mu_v[:, 1:]], dim=1)
    padded_logvar = torch.cat([logvar_v[:, :1], inf_logvar, logvar_v[:, 1:]], dim=1)
    padded = product_of_experts(padded_mu, padded_logvar, torch.tensor([[True, False, True]]))
    assert torch.equal(padded.mu, fused.mu)
    assert torch.equal(padded.logvar, fused.logvar)


@pytest.mark.parametrize("slot", range(SLOTS))
def test_one_view_posterior_is_the_prior_expert_times_the_encoder_expert(slot):
    mu_v, logvar_v = encoder_experts()
    mu_v, logvar_v = mu_v.double(), logvar_v.double()
    batch = mu_v.shape[0]
    alone = product_of_experts(mu_v, logvar_v, view_mask([slot], batch))

    # Closed form: the prior has precision 1, so the posterior has precision 1 + exp(-logvar_v).
    precision = torch.exp(-logvar_v[:, slot])
    torch.testing.assert_close(alone.logvar, -torch.log1p(precision), rtol=1e-12, atol=1e-12)
    expected_mu = mu_v[:, slot] * precision / (1.0 + precision)
    torch.testing.assert_close(alone.mu, expected_mu, rtol=1e-12, atol=1e-12)

    # Independent check: the product of the two densities is the posterior up to a constant
    # that does not depend on the point, so their log-density difference must not vary with it.
    generator = torch.Generator().manual_seed(4)
    points = torch.randn(6, 1, LATENT, generator=generator, dtype=torch.float64)
    zeros = torch.zeros(LATENT, dtype=torch.float64)
    difference = (
        gaussian_log_density(points, alone.mu, alone.logvar)
        - gaussian_log_density(points, zeros, zeros)
        - gaussian_log_density(points, mu_v[:, slot], logvar_v[:, slot])
    )
    torch.testing.assert_close(
        difference, difference[:1].expand_as(difference), rtol=0.0, atol=1e-9
    )


@pytest.mark.parametrize("slot", range(SLOTS))
def test_one_view_fused_posterior_is_the_per_view_posterior_bit_for_bit(slot):
    mu_v, logvar_v = encoder_experts()
    per_view = single_view_posteriors(mu_v, logvar_v)

    # The other slots hold NaN and infinities. None of them may reach the result.
    others = [other for other in range(SLOTS) if other != slot]
    dirty_mu = mu_v.clone()
    dirty_logvar = logvar_v.clone()
    dirty_mu[:, others] = float("nan")
    dirty_logvar[:, others[0]] = float("inf")
    dirty_logvar[:, others[1]] = -float("inf")
    dirty_logvar[:, others[2]] = float("nan")
    fused = product_of_experts(dirty_mu, dirty_logvar, view_mask([slot], mu_v.shape[0]))
    assert torch.equal(fused.mu, per_view.mu[:, slot])
    assert torch.equal(fused.logvar, per_view.logvar[:, slot])

    # A layout with the single view as its only slot gives the same bits.
    alone = product_of_experts(mu_v[:, slot : slot + 1], logvar_v[:, slot : slot + 1])
    assert torch.equal(alone.mu, per_view.mu[:, slot])
    assert torch.equal(alone.logvar, per_view.logvar[:, slot])


def test_fused_variance_is_at_most_that_of_each_contributing_view_and_of_the_prior():
    mu_v, logvar_v = random_experts(batch=64, spread=8.0)
    expert_variance = logvar_v.exp()
    per_view_variance = single_view_posteriors(mu_v, logvar_v).logvar.exp()
    for subset in all_subsets():
        variance = product_of_experts(mu_v, logvar_v, view_mask(subset, 64)).logvar.exp()
        assert (variance <= 1.0).all(), subset
        for slot in subset:
            assert (variance <= expert_variance[:, slot]).all(), (subset, slot)
            assert (variance <= per_view_variance[:, slot]).all(), (subset, slot)


def test_adding_a_view_never_increases_the_fused_variance():
    batch = 2000
    mu_v, logvar_v = random_experts(batch=batch, seed=3, spread=10.0)
    subsets = all_subsets()
    precision = {subset: fused_precision(logvar_v, view_mask(subset, batch)) for subset in subsets}
    variance = {
        subset: product_of_experts(mu_v, logvar_v, view_mask(subset, batch)).logvar.exp()
        for subset in subsets
    }
    nested_pairs = [
        (smaller, larger)
        for smaller, larger in itertools.permutations(subsets, 2)
        if set(smaller) < set(larger)
    ]
    assert len(nested_pairs) == 50
    for smaller, larger in nested_pairs:
        # Precisions add non-negative terms, so this order holds exactly in floating point.
        assert (precision[smaller] <= precision[larger]).all(), (smaller, larger)
        # The variances come from log and exp, which are monotone up to rounding.
        assert (variance[larger] <= variance[smaller] * (1.0 + 1e-6)).all(), (smaller, larger)

    # The same holds for the real encoder experts when the views are added in camera order.
    encoder_mu, encoder_logvar = encoder_experts()
    previous = None
    for count in range(1, SLOTS + 1):
        mask = torch.zeros(encoder_mu.shape[0], SLOTS, dtype=torch.bool)
        mask[:, :count] = True
        current = fused_precision(encoder_logvar, mask)
        if previous is not None:
            assert (current >= previous).all(), count
        previous = current


def test_absent_views_are_never_read_and_pass_no_gradient():
    mu_v, logvar_v = random_experts(batch=5)
    mask = view_mask([0, 2], 5)
    absent = ~mask
    clean_mu = torch.where(mask.unsqueeze(-1), mu_v, torch.zeros_like(mu_v))
    clean_logvar = torch.where(mask.unsqueeze(-1), logvar_v, torch.zeros_like(logvar_v))
    dirty_mu = torch.where(mask.unsqueeze(-1), mu_v, torch.full_like(mu_v, float("nan")))
    dirty_logvar = torch.where(
        mask.unsqueeze(-1), logvar_v, torch.full_like(logvar_v, float("-inf"))
    )
    reference = product_of_experts(clean_mu, clean_logvar, mask)
    dirty = product_of_experts(dirty_mu, dirty_logvar, mask)
    assert torch.equal(reference.mu, dirty.mu)
    assert torch.equal(reference.logvar, dirty.logvar)

    mu_with_grad = dirty_mu.clone().requires_grad_(True)
    logvar_with_grad = dirty_logvar.clone().requires_grad_(True)
    fused_with_grad = product_of_experts(mu_with_grad, logvar_with_grad, mask)
    (fused_with_grad.mu.pow(2).sum() + fused_with_grad.logvar.sum()).backward()
    assert torch.isfinite(mu_with_grad.grad).all()
    assert torch.isfinite(logvar_with_grad.grad).all()
    assert (mu_with_grad.grad[absent] == 0).all() and (logvar_with_grad.grad[absent] == 0).all()
    assert (logvar_with_grad.grad[mask] != 0).any()


def test_no_present_view_gives_the_prior():
    mu_v, logvar_v = random_experts(batch=5)
    no_view = torch.zeros(5, SLOTS, dtype=torch.bool)
    fused = product_of_experts(mu_v, logvar_v, no_view)
    assert torch.equal(fused.mu, torch.zeros_like(fused.mu))
    assert torch.equal(fused.logvar, torch.zeros_like(fused.logvar))
    assert torch.equal(fused_precision(logvar_v, no_view), torch.ones_like(fused.mu))
