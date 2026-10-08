"""Tests for decoder.py (research R7): output shapes, finite mean and log-scale, and its bound."""

from pathlib import Path

import pytest
import torch
from torch import nn

from strike_a_pose.config import load_config
from strike_a_pose.model.decoder import LOG_SCALE_LIMIT, DecodedShape, ShapeDecoder, gaussian_nll

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Run this module on one thread: its tensors are tiny, and other builders share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def last_linear(decoder: ShapeDecoder) -> nn.Linear:
    """The output layer of the decoder, which holds the means and the raw log-scales."""
    return [module for module in decoder.modules() if isinstance(module, nn.Linear)][-1]


@pytest.mark.parametrize("leading", [(), (5,), (5, 3), (2, 5, 3)])
def test_mean_and_log_scale_have_the_latent_leading_shape_with_n_betas_last(leading):
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    with torch.no_grad():
        out = decoder(torch.randn(*leading, 8))
    assert isinstance(out, DecodedShape)
    assert out.mean.shape == (*leading, 10)
    assert out.log_scale.shape == (*leading, 10)
    assert torch.isfinite(out.mean).all() and torch.isfinite(out.log_scale).all()


@pytest.mark.parametrize("name", ["tiny.yaml", "full.yaml"])
def test_head_sizes_follow_the_model_and_body_sections(name):
    config = load_config(CONFIG_DIR / name)
    latent_dim = config["model"]["latent_dim"]
    n_betas = config["body"]["n_betas"]
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=latent_dim, n_betas=n_betas)
    assert (decoder.latent_dim, decoder.n_betas) == (latent_dim, n_betas)
    with torch.no_grad():
        out = decoder(torch.randn(4, latent_dim))
    assert out.mean.shape == (4, n_betas)
    assert out.log_scale.shape == (4, n_betas)
    assert torch.isfinite(out.mean).all() and torch.isfinite(out.log_scale).all()


def test_zero_and_huge_latent_samples_give_finite_outputs_with_bounded_log_scale():
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    with torch.no_grad():
        for z in (torch.zeros(4, 8), 1e6 * torch.randn(4, 8)):
            out = decoder(z)
            assert torch.isfinite(out.mean).all() and torch.isfinite(out.log_scale).all()
            assert bool((out.log_scale.abs() <= LOG_SCALE_LIMIT).all())


def test_a_new_decoder_predicts_a_scale_near_one_and_saturates_at_the_limit():
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    z = torch.randn(64, 8)
    output_layer = last_linear(decoder)
    with torch.no_grad():
        # A log-scale near 0 is a scale near 1, the scale of a standard normal coefficient.
        assert float(decoder(z).log_scale.abs().max()) < 0.5
        output_layer.bias.fill_(1000.0)
        high = decoder(z).log_scale
        output_layer.bias.fill_(-1000.0)
        low = decoder(z).log_scale
    assert torch.allclose(high, torch.full_like(high, LOG_SCALE_LIMIT))
    assert torch.allclose(low, torch.full_like(low, -LOG_SCALE_LIMIT))


def test_negative_log_likelihood_of_the_decoded_shape_is_finite_at_the_bounds():
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    output_layer = last_linear(decoder)
    target = torch.zeros(3, 10)
    with torch.no_grad():
        for bias in (1000.0, -1000.0):
            output_layer.bias.fill_(bias)
            out = decoder(torch.zeros(3, 8))
            nll = gaussian_nll(target, out.mean, out.log_scale)
            assert nll.shape == (3,)
            assert torch.isfinite(nll).all()


def test_sample_has_the_decoded_shape_and_repeats_for_one_generator_seed():
    torch.manual_seed(0)
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    z = torch.randn(5, 3, 8)
    with torch.no_grad():
        first = decoder.sample(z, torch.Generator().manual_seed(4))
        second = decoder.sample(z, torch.Generator().manual_seed(4))
    assert first.shape == (5, 3, 10)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_sizes_must_be_integers_of_at_least_one(bad):
    with pytest.raises(ValueError):
        ShapeDecoder(latent_dim=bad, n_betas=10)
    with pytest.raises(ValueError):
        ShapeDecoder(latent_dim=8, n_betas=bad)


def test_latent_samples_must_end_in_latent_dim():
    decoder = ShapeDecoder(latent_dim=8, n_betas=10)
    with pytest.raises(ValueError, match="z must end in an axis of length 8"):
        decoder(torch.zeros(3, 7))
    with pytest.raises(ValueError, match="z must end in an axis of length 8"):
        decoder(torch.tensor(1.0))
