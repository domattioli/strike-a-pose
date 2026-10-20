"""Evaluation (slice 003): metrics, PCA, mean/PCA baselines, eval CLI.

Contract: ``specs/003-baselines-and-metrics/contracts/eval-run.md``.
"""

from .config import BaselineConfig, ConfigError, config_from_dict, load_config
from .metrics import (
    IOU_THRESHOLD,
    PSNR_MSE_FLOOR,
    ROOT_JOINT,
    image_mse,
    keypoint_error_2d,
    mpjpe,
    project_points,
    psnr,
    silhouette_iou,
)
from .pca import PCA, fit_pca

__all__ = [
    "IOU_THRESHOLD",
    "PCA",
    "PSNR_MSE_FLOOR",
    "ROOT_JOINT",
    "BaselineConfig",
    "ConfigError",
    "config_from_dict",
    "fit_pca",
    "image_mse",
    "keypoint_error_2d",
    "load_config",
    "mpjpe",
    "project_points",
    "psnr",
    "silhouette_iou",
]
