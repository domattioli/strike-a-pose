"""Tests for encoder.py (research R7): block plan from model.channels, shapes, camera input."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from strike_a_pose.camera import CAMERA_ENCODING_DIM, encode_camera, noise_rotation
from strike_a_pose.config import load_config
from strike_a_pose.model.encoder import LOGVAR_BOUND, ViewEncoder

FULL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "full.yaml"

# The smallest change of mu_v that counts as a camera effect. In the camera test below, a 2 degree
# turn moves every mu_v by at least 3e-4, and float32 rounding at the typical size of mu_v (0.08)
# is about 1e-8, so 1e-5 separates the two with a wide margin.
MIN_MU_CHANGE = 1e-5


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Run this module on one thread: its tensors are tiny, and other builders share the machine."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture
def small_model_config(tiny_config_path: Path, small_config: dict[str, object]) -> dict[str, Any]:
    """The tiny configuration with the test overrides of contracts/config.md (32 pixel images)."""
    overrides = [f"{key}={json.dumps(value)}" for key, value in small_config.items()]
    return load_config(tiny_config_path, overrides=overrides)


def random_views(
    batch: int, views: int, size: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random binary silhouettes (batch, views, size, size) and camera codes (batch, views, 9)."""
    generator = torch.Generator().manual_seed(seed)
    silhouettes = (torch.rand(batch, views, size, size, generator=generator) > 0.5).float()
    cameras = torch.randn(batch, views, CAMERA_ENCODING_DIM, generator=generator)
    return silhouettes, cameras


def camera_code(rotation: np.ndarray, translation: np.ndarray) -> torch.Tensor:
    """The float32 camera encoding of one placement, as camera.encode_camera builds it."""
    return torch.as_tensor(encode_camera(rotation, translation), dtype=torch.float32)


def assert_block_plan(encoder: ViewEncoder, channels: list[int]) -> None:
    """One stride-2 convolution per entry of channels, with the entries as output channels."""
    convolutions = [module for module in encoder.modules() if isinstance(module, nn.Conv2d)]
    assert len(encoder.blocks) == len(channels)
    assert [conv.out_channels for conv in convolutions] == channels
    assert [conv.in_channels for conv in convolutions] == [1, *channels[:-1]]
    for conv in convolutions:
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (2, 2)
        assert conv.padding == (1, 1)


def test_from_config_reads_the_model_section(small_model_config):
    encoder = ViewEncoder.from_config(small_model_config)
    assert encoder.image_size == 32
    assert encoder.channels == (8, 16, 32, 32, 32)
    assert (encoder.latent_dim, encoder.camera_embed_dim) == (8, 16)
    # Five stride-2 blocks take 32 pixels down to 1 (16, 8, 4, 2, 1).
    assert encoder.feature_map_size == 1
    assert_block_plan(encoder, [8, 16, 32, 32, 32])


def test_tiny_configuration_at_64_pixels_has_the_planned_shapes(tiny_config_path):
    config = load_config(tiny_config_path)
    encoder = ViewEncoder.from_config(config)
    channels = config["model"]["channels"]
    assert_block_plan(encoder, channels)
    # 64 pixels go down to 2 after five blocks. The head reads the whole flattened map (no
    # pooling) joined with the camera embedding.
    assert encoder.feature_map_size == 2
    camera_width = config["model"]["camera_embed_dim"]
    assert encoder.trunk[0].in_features == channels[-1] * 2 * 2 + camera_width

    silhouettes, cameras = random_views(batch=2, views=4, size=64)
    with torch.no_grad():
        mu_v, logvar_v = encoder(silhouettes, cameras)
    assert mu_v.shape == (2, 4, 8)
    assert logvar_v.shape == (2, 4, 8)
    assert torch.isfinite(mu_v).all() and torch.isfinite(logvar_v).all()


@pytest.mark.parametrize(
    ("image_size", "channels"),
    [
        (32, [8, 16, 32, 32, 32]),
        (64, [8, 16, 32, 32, 32]),
        # 100 pixels go down to 50, 25 and 13: the side rounds up at each block.
        (100, [4, 8, 16]),
        (128, [32, 64, 128, 256, 256]),
    ],
)
def test_each_block_outputs_its_channels_at_half_the_side_rounded_up(image_size, channels):
    torch.manual_seed(0)
    encoder = ViewEncoder(
        image_size=image_size, channels=channels, latent_dim=4, camera_embed_dim=5
    )
    features = torch.rand(2, 1, image_size, image_size)
    side = image_size
    with torch.no_grad():
        for block, out_channels in zip(encoder.blocks, channels, strict=True):
            features = block(features)
            side = math.ceil(side / 2)
            assert features.shape == (2, out_channels, side, side)
    assert encoder.feature_map_size == side


def test_full_configuration_builds_the_encoder_of_the_kaggle_run():
    config = load_config(FULL_CONFIG_PATH)
    encoder = ViewEncoder.from_config(config)
    assert encoder.image_size == 128
    assert encoder.channels == (32, 64, 128, 256, 256)
    assert encoder.feature_map_size == 4
    assert encoder.latent_dim == 16
    assert encoder.camera_embed_dim == 64
    assert encoder.trunk[0].in_features == 256 * 4 * 4 + 64

    torch.manual_seed(0)
    silhouettes, cameras = random_views(batch=1, views=1, size=128)
    with torch.no_grad():
        mu_v, logvar_v = encoder(silhouettes, cameras)
    assert mu_v.shape == (1, 1, 16)
    assert logvar_v.shape == (1, 1, 16)
    assert torch.isfinite(mu_v).all() and torch.isfinite(logvar_v).all()


@pytest.mark.parametrize("views", [1, 2, 3, 4])
def test_outputs_have_shape_batch_views_latent_for_one_to_four_views(small_model_config, views):
    torch.manual_seed(0)
    encoder = ViewEncoder.from_config(small_model_config)
    silhouettes, cameras = random_views(batch=2, views=views, size=32)
    with torch.no_grad():
        mu_v, logvar_v = encoder(silhouettes, cameras)
    assert mu_v.shape == (2, views, 8)
    assert logvar_v.shape == (2, views, 8)
    assert torch.isfinite(mu_v).all() and torch.isfinite(logvar_v).all()


@pytest.mark.parametrize("leading", [(), (3,), (2, 3)])
def test_leading_dimensions_pass_through_to_the_outputs(small_model_config, leading):
    torch.manual_seed(0)
    encoder = ViewEncoder.from_config(small_model_config)
    silhouettes = torch.rand(*leading, 32, 32)
    cameras = torch.randn(*leading, CAMERA_ENCODING_DIM)
    with torch.no_grad():
        mu_v, logvar_v = encoder(silhouettes, cameras)
    assert mu_v.shape == (*leading, 8)
    assert logvar_v.shape == (*leading, 8)


def test_a_camera_change_moves_mu_v_for_every_view(small_model_config):
    torch.manual_seed(1)
    encoder = ViewEncoder.from_config(small_model_config)
    silhouettes, _ = random_views(batch=2, views=4, size=32, seed=1)
    translation = np.array([0.0, 0.9, 3.0])
    shape = (2, 4, CAMERA_ENCODING_DIM)
    placed = camera_code(np.eye(3), translation).expand(shape)
    # A 2 degree turn about the vertical axis, one of the noise levels of evaluate.noise_deg.
    turned = camera_code(noise_rotation([0.0, 1.0, 0.0], 2.0), translation).expand(shape)
    moved = camera_code(np.eye(3), translation + np.array([0.0, 0.0, 0.5])).expand(shape)
    with torch.no_grad():
        mu_placed, _ = encoder(silhouettes, placed)
        mu_again, _ = encoder(silhouettes, placed)
        mu_turned, _ = encoder(silhouettes, turned)
        mu_moved, _ = encoder(silhouettes, moved)
    # The same camera gives the same mu_v, so every change below comes from the camera alone.
    assert torch.equal(mu_placed, mu_again)
    for other in (mu_turned, mu_moved):
        change = (mu_placed - other).abs().amax(dim=-1)
        assert bool((change > MIN_MU_CHANGE).all()), change


def test_logvar_stays_within_the_bound_for_extreme_weights(small_model_config):
    torch.manual_seed(2)
    encoder = ViewEncoder.from_config(small_model_config)
    silhouettes, cameras = random_views(batch=2, views=4, size=32, seed=2)
    with torch.no_grad():
        encoder.logvar_head.bias.fill_(1000.0)
        _, high = encoder(silhouettes, cameras)
        encoder.logvar_head.bias.fill_(-1000.0)
        _, low = encoder(silhouettes, cameras)
    assert torch.isfinite(high).all() and torch.isfinite(low).all()
    assert bool((high <= LOGVAR_BOUND).all()) and bool((high > 0.99 * LOGVAR_BOUND).all())
    assert bool((low >= -LOGVAR_BOUND).all()) and bool((low < -0.99 * LOGVAR_BOUND).all())


@pytest.mark.parametrize("channels", [[], [8, 0], [8.0, 16], "8,16"])
def test_channels_must_be_a_non_empty_list_of_positive_integers(channels):
    with pytest.raises(ValueError, match="channels"):
        ViewEncoder(image_size=32, channels=channels, latent_dim=8, camera_embed_dim=16)


def test_mismatched_inputs_raise_value_error(small_model_config):
    encoder = ViewEncoder.from_config(small_model_config)
    silhouettes, cameras = random_views(batch=2, views=4, size=32)
    with pytest.raises(ValueError, match="silhouette must have shape"):
        encoder(torch.zeros(2, 4, 31, 32), cameras)
    with pytest.raises(ValueError, match="camera must have shape"):
        encoder(silhouettes, cameras[..., :8])
    with pytest.raises(ValueError, match="camera must have shape"):
        encoder(silhouettes, cameras[:, :3])
