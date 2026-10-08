"""Per-view silhouette encoder: a strided CNN with camera conditioning (research R7, FR-006).

The encoder reads one silhouette and the camera placement given to the model, and returns the mean
``mu_v`` and the log-variance ``logvar_v`` of one diagonal Gaussian over the latent vector. That
pair is the expert of the view in the product of experts of ``fusion.py``. It is not yet the
per-view posterior: the posterior of one view is the product of this expert and the standard normal
prior expert, which ``fusion.single_view_posteriors`` forms. Views are encoded on their own
(FR-006), so nothing here knows how many other views a sample has.

Public sources (research R7): Wu and Goodman, "Multimodal Generative Models for Scalable
Weakly-Supervised Learning" (NeurIPS 2018, https://arxiv.org/abs/1802.05335), for one inference
network per input whose Gaussian output is an expert; Kingma and Welling, "Auto-Encoding
Variational Bayes" (2013, https://arxiv.org/abs/1312.6114), for the mean and log-variance heads;
Zhou, Barnes, Lu, Yang, and Li, "On the Continuity of Rotation Representations in Neural Networks"
(CVPR 2019, https://arxiv.org/abs/1812.07035), for the 6D rotation in the camera input, which
``camera.encode_camera`` builds; and Wu and He, "Group Normalization" (ECCV 2018,
https://arxiv.org/abs/1803.08494), for the normalization layers.

Inputs. ``silhouette`` has shape (..., S, S) for ``image_size`` S: a mask of 0 and 1 in any numeric
or boolean type. ``camera`` has shape (..., ``CAMERA_ENCODING_DIM``): the encoding of ``R_given``
and ``t_true`` from ``camera.encode_camera``. The leading dimensions are free but must agree, so
(count, S, S) and (batch, views, S, S) both work, and the outputs have shape (..., latent_dim).
A slot that holds no view (the padding of a sample with fewer views) may be encoded like any
other image or left out of the call; the fusion ignores it through the view mask either way.

Layers.

* Convolution. ``channels`` has one entry per strided block. A block is a 3 by 3 convolution with
  stride 2 and padding 1, a group normalization with one group, and a SiLU activation. Each block
  halves the height and the width, rounding up, so five blocks take 128 pixels to 4. The
  one-group normalization works on each image over all of its channels and positions. A result
  therefore never depends on the other images of the batch, which matters because the padding
  slots of a batch share it with real views, and it stays defined when the last feature maps shrink
  to one pixel. Batch normalization breaks both properties.
* The last feature map is flattened, not pooled, so the encoder keeps where the silhouette lies in
  the image. The apparent size of a body carries its scale.
* Camera embedding. The camera encoding goes through two linear layers with a SiLU between them
  and becomes ``camera_embed_dim`` values.
* Heads. The flattened features and the camera embedding are joined, pass one hidden layer with a
  SiLU, and feed two linear heads, one for ``mu_v`` and one for ``logvar_v``. The hidden layer is as
  wide as the last convolution, or twice the latent size when that is larger. The weights use the
  PyTorch default initialization, so a seed set before construction fixes them. No layer starts at
  zero, so the camera embedding changes ``mu_v`` from the first step.

Bound on the log-variance. ``logvar_v`` is ``LOGVAR_BOUND * tanh(raw / LOGVAR_BOUND)``, so it
never leaves the interval from -10 to 10 (single precision reaches an end point only when the raw
output passes about 90) and follows the raw output closely near 0. The precision exp(-logvar_v)
then stays between about 4.5e-5 and 22,026 whatever the weights do, so the precisions that the
fusion adds are finite in single precision during training.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn

from strike_a_pose.camera import CAMERA_ENCODING_DIM

__all__ = ["LOGVAR_BOUND", "ViewEncoder"]

# The log-variance of an expert is squashed into the interval from -LOGVAR_BOUND to LOGVAR_BOUND.
LOGVAR_BOUND = 10.0

# Every convolution block halves the height and width (kernel 3, stride 2, padding 1).
_KERNEL_SIZE = 3
_STRIDE = 2
_PADDING = 1


class ViewEncoder(nn.Module):
    """Encode one silhouette and its camera into the Gaussian expert ``(mu_v, logvar_v)``.

    ``image_size`` is the side of the square silhouette (``camera.image_size``), ``channels`` the
    output channels of the strided blocks (``model.channels``), ``latent_dim`` the length of the
    latent vector (``model.latent_dim``), and ``camera_embed_dim`` the length of the camera
    embedding (``model.camera_embed_dim``). Attributes ``image_size``, ``channels``,
    ``latent_dim``, ``camera_embed_dim``, and ``feature_map_size`` (the side of the last feature
    map) record the shape of the network.
    """

    def __init__(
        self,
        image_size: int,
        channels: Sequence[int],
        latent_dim: int,
        camera_embed_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = _positive_int(image_size, "image_size")
        if isinstance(channels, (str, bytes)) or len(channels) == 0:
            raise ValueError(
                f"channels must be a non-empty list of positive integers; got {channels!r}"
            )
        self.channels = tuple(_positive_int(count, "channels entry") for count in channels)
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.camera_embed_dim = _positive_int(camera_embed_dim, "camera_embed_dim")

        blocks = []
        in_channels = 1
        for out_channels in self.channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=_KERNEL_SIZE,
                        stride=_STRIDE,
                        padding=_PADDING,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.SiLU(),
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.feature_map_size = _feature_map_size(self.image_size, len(self.channels))
        flattened = self.channels[-1] * self.feature_map_size**2
        self.camera_embedding = nn.Sequential(
            nn.Linear(CAMERA_ENCODING_DIM, self.camera_embed_dim),
            nn.SiLU(),
            nn.Linear(self.camera_embed_dim, self.camera_embed_dim),
        )
        hidden = max(self.channels[-1], 2 * self.latent_dim)
        self.trunk = nn.Sequential(
            nn.Linear(flattened + self.camera_embed_dim, hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, self.latent_dim)
        self.logvar_head = nn.Linear(hidden, self.latent_dim)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ViewEncoder":
        """Build the encoder of a resolved configuration (``model.*`` and ``camera.image_size``)."""
        model = config["model"]
        return cls(
            image_size=config["camera"]["image_size"],
            channels=model["channels"],
            latent_dim=model["latent_dim"],
            camera_embed_dim=model["camera_embed_dim"],
        )

    def forward(self, silhouette: Tensor, camera: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mu_v, logvar_v)``, each of shape (..., latent_dim), for the given views."""
        batch_shape = self._batch_shape(silhouette, camera)
        count = math.prod(batch_shape)
        dtype = self.mu_head.weight.dtype
        images = silhouette.to(dtype).reshape(count, 1, self.image_size, self.image_size)
        codes = camera.to(dtype).reshape(count, CAMERA_ENCODING_DIM)

        features = self.blocks(images).flatten(start_dim=1)
        embedding = self.camera_embedding(codes)
        hidden = self.trunk(torch.cat([features, embedding], dim=1))

        mu_v = self.mu_head(hidden)
        logvar_v = LOGVAR_BOUND * torch.tanh(self.logvar_head(hidden) / LOGVAR_BOUND)
        return (
            mu_v.reshape(*batch_shape, self.latent_dim),
            logvar_v.reshape(*batch_shape, self.latent_dim),
        )

    def _batch_shape(self, silhouette: Tensor, camera: Tensor) -> tuple[int, ...]:
        """Return the leading dimensions shared by the inputs, or raise when they do not fit."""
        side = self.image_size
        if silhouette.dim() < 2 or tuple(silhouette.shape[-2:]) != (side, side):
            raise ValueError(
                f"silhouette must have shape (..., {side}, {side}) for image_size {side}; "
                f"got {tuple(silhouette.shape)}"
            )
        batch_shape = tuple(silhouette.shape[:-2])
        if tuple(camera.shape) != (*batch_shape, CAMERA_ENCODING_DIM):
            raise ValueError(
                f"camera must have shape {(*batch_shape, CAMERA_ENCODING_DIM)}, the leading "
                f"dimensions of the silhouette plus {CAMERA_ENCODING_DIM} encoding values; "
                f"got {tuple(camera.shape)}"
            )
        return batch_shape


def _positive_int(value: object, name: str) -> int:
    """Return value as an int when it is an integer of at least 1; raise ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1; got {value!r}")
    return int(value)


def _feature_map_size(image_size: int, blocks: int) -> int:
    """Return the side of the feature map after the given number of strided blocks."""
    size = image_size
    for _ in range(blocks):
        size = (size + 2 * _PADDING - _KERNEL_SIZE) // _STRIDE + 1
    return size
