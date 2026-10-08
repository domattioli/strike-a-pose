"""Smoke tests for vae.py at 64 px: finite loss, backward through every parameter, and shapes."""

import pytest
import torch

from strike_a_pose.camera import CAMERA_ENCODING_DIM
from strike_a_pose.config import load_config
from strike_a_pose.model.vae import LossTerms, ShapeVAE, ViewBatch

# Four view slots, one per camera of the rig, and 64 px images, which build_model sets explicitly.
SLOTS = 4
IMAGE_SIZE = 64
LOSS_PARTS = ("total", "nll_joint", "nll_single", "kl_joint", "kl_single")


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    """Run on one thread, so these tests stay fast while other jobs share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


def build_model(config_path):
    """Return the ShapeVAE of the tiny configuration, with its images set to 64 px."""
    overrides = [f"camera.image_size={IMAGE_SIZE}", f"camera.focal_px={IMAGE_SIZE}"]
    config = load_config(config_path, overrides)
    assert config["camera"]["image_size"] == IMAGE_SIZE
    torch.manual_seed(0)
    model = ShapeVAE.from_config(config)
    assert model.encoder.image_size == IMAGE_SIZE
    return model


def make_views(counts, seed=0):
    """Return a batch with one body per entry of ``counts``.

    Body i has its first counts[i] slots present and the other slots absent.
    """
    generator = torch.Generator().manual_seed(seed)
    batch = len(counts)
    silhouettes = (
        torch.rand(batch, SLOTS, IMAGE_SIZE, IMAGE_SIZE, generator=generator) > 0.5
    ).float()
    cameras = torch.randn(batch, SLOTS, CAMERA_ENCODING_DIM, generator=generator)
    view_mask = torch.arange(SLOTS).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    return ViewBatch(silhouettes, cameras, view_mask)


def make_betas(batch, n_betas, seed=1):
    """Return shape coefficients of shape (batch, n_betas), clipped to plus or minus 3."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_betas, generator=generator).clamp(-3.0, 3.0)


def check_loss_is_finite_and_backward_reaches_every_parameter(model, views):
    """Check each loss part is a finite scalar and each parameter gets a finite gradient."""
    betas = make_betas(views.view_mask.shape[0], model.n_betas)
    terms = model.loss(views, betas, generator=torch.Generator().manual_seed(2))
    assert isinstance(terms, LossTerms)
    for name in LOSS_PARTS:
        value = getattr(terms, name)
        assert value.ndim == 0 and bool(torch.isfinite(value)), name
    terms.total.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


@pytest.mark.parametrize("views", [1, 2, 3, 4])
def test_loss_is_finite_and_backward_works_for_each_view_count_at_64_px(tiny_config_path, views):
    model = build_model(tiny_config_path)
    check_loss_is_finite_and_backward_reaches_every_parameter(model, make_views([views, views]))


def test_loss_is_finite_and_backward_works_for_bodies_with_mixed_view_counts(tiny_config_path):
    model = build_model(tiny_config_path)
    check_loss_is_finite_and_backward_reaches_every_parameter(model, make_views([1, 2, 3, 4]))


def test_nan_in_absent_slots_leaves_the_loss_and_gradients_finite(tiny_config_path):
    model = build_model(tiny_config_path)
    clean = make_views([1, 2, 3, 4])
    absent = ~clean.view_mask
    silhouettes = clean.silhouettes.clone()
    cameras = clean.cameras.clone()
    silhouettes[absent] = float("nan")
    cameras[absent] = float("nan")
    padded = ViewBatch(silhouettes, cameras, clean.view_mask)
    check_loss_is_finite_and_backward_reaches_every_parameter(model, padded)


@pytest.mark.parametrize("views", [1, 2, 3, 4])
def test_shapes_for_each_view_count_at_64_px(tiny_config_path, views):
    model = build_model(tiny_config_path)
    batch = make_views([views, views, views])
    latent_dim = model.latent_dim

    experts = model.encode(batch)
    assert experts.mu_v.shape == (3, SLOTS, latent_dim)
    assert experts.logvar_v.shape == (3, SLOTS, latent_dim)
    absent = ~batch.view_mask
    assert bool((experts.mu_v[absent] == 0).all()) and bool((experts.logvar_v[absent] == 0).all())

    posterior = model(batch)
    assert posterior.mu.shape == (3, latent_dim)
    assert posterior.logvar.shape == (3, latent_dim)

    decoded = model.decoder(posterior.mu)
    assert decoded.mean.shape == (3, model.n_betas)
    assert decoded.log_scale.shape == (3, model.n_betas)

    samples = model.sample_betas(batch, 5, torch.Generator().manual_seed(3))
    assert samples.shape == (3, 5, model.n_betas)
    assert bool(torch.isfinite(samples).all())
