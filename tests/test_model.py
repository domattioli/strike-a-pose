"""ConvAutoencoder shapes, range, errors, checkpoint round trip (spec 004 FR-001)."""

import pytest
import torch

from strike_a_pose.models import DEFAULT_CHANNELS, ConvAutoencoder, load_checkpoint


def test_default_shapes_and_range():
    torch.manual_seed(0)
    m = ConvAutoencoder().eval()
    x = torch.rand(2, 1, 256, 192)
    with torch.no_grad():
        z = m.encode(x)
        y = m.decode(z)
    assert z.shape == (2, 64)
    assert y.shape == x.shape and float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert m.bottleneck == (256, 4, 3)
    assert m.channels == DEFAULT_CHANNELS
    assert m.parameter_count() == 3_126_897  # research R2 (≈ 3.1 M)


def test_custom_latent_and_channels():
    m = ConvAutoencoder((32, 16), latent_dim=8, channels=(4, 8)).eval()
    with torch.no_grad():
        assert m.encode(torch.zeros(3, 1, 32, 16)).shape == (3, 8)
        assert m(torch.zeros(3, 1, 32, 16)).shape == (3, 1, 32, 16)
    assert m.hparams() == {
        "image_size": [32, 16], "latent_dim": 8, "channels": [4, 8], "variational": False,
    }  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [
        {"image_size": (250, 192)},
        {"image_size": (256, 192), "channels": (8,) * 7},  # 192 / 2^7 not integral
        {"channels": ()},
        {"channels": (8, 0)},
        {"latent_dim": 0},
    ],
)
def test_invalid_arguments(kw):
    with pytest.raises(ValueError):
        ConvAutoencoder(**kw)


def test_deterministic_init():
    torch.manual_seed(1)
    a = ConvAutoencoder((16, 16), 4, (4, 4))
    torch.manual_seed(1)
    b = ConvAutoencoder((16, 16), 4, (4, 4))
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p, q)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    m = ConvAutoencoder((16, 16), 4, (4, 8)).eval()
    p = tmp_path / "checkpoint.pt"
    torch.save({"model": m.state_dict(), "model_hparams": m.hparams(), "epoch": 3}, p)
    m2, ckpt = load_checkpoint(p)
    assert ckpt["epoch"] == 3 and not m2.training
    x = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
