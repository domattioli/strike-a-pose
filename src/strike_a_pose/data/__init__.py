"""Data pipeline (slice 002): dataset loader, normalization, frozen split.

Contract: ``specs/002-data-pipeline/contracts/sample-contract.md``.
"""

from .dataset import DatasetError, PoseImageDataset
from .split import (
    DEFAULT_FRACTIONS,
    SPLITS,
    SplitCommitWarning,
    SplitError,
    load_split,
    make_split,
    manifest_content_sha256,
    manifest_sha256,
    parse_sample_id,
    write_split,
)
from .transforms import (
    add_root,
    denormalize_keypoints_2d,
    image_to_tensor,
    mask_to_tensor,
    normalize_keypoints_2d,
    root_relative,
    tensor_to_image,
)

__all__ = [
    "DEFAULT_FRACTIONS",
    "SPLITS",
    "DatasetError",
    "PoseImageDataset",
    "SplitCommitWarning",
    "SplitError",
    "add_root",
    "denormalize_keypoints_2d",
    "image_to_tensor",
    "load_split",
    "make_split",
    "manifest_content_sha256",
    "manifest_sha256",
    "mask_to_tensor",
    "normalize_keypoints_2d",
    "parse_sample_id",
    "root_relative",
    "tensor_to_image",
    "write_split",
]
