"""Generation config (YAML) — see specs/001-dataset-generation/data-model.md."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

POSE_SOURCES = ("procedural", "amass")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ImageConfig:
    height: int = 256
    width: int = 192


@dataclass(frozen=True)
class CameraRingConfig:
    count: int = 8
    distance_m: float = 3.5
    height_m: float = 0.0
    focal_px: float = 320.0


@dataclass(frozen=True)
class GenerationConfig:
    dataset_name: str
    pose_source: str = "procedural"  # "procedural" (default) | "amass"
    amass_subsets: tuple[str, ...] = ()
    output_root: str = "data/generated"
    smpl_model_path: str = "assets/smpl/SMPL_NEUTRAL.npz"
    amass_root: str = "assets/amass"
    num_poses: int = 2000
    frame_stride: int = 30
    num_betas: int = 10
    image: ImageConfig = field(default_factory=ImageConfig)
    camera_ring: CameraRingConfig = field(default_factory=CameraRingConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["amass_subsets"] = list(self.amass_subsets)
        return d


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return data


def config_from_dict(data: dict[str, Any]) -> GenerationConfig:
    data = dict(_build(GenerationConfig, data, "config"))
    try:
        image = ImageConfig(**_build(ImageConfig, data.pop("image", {}), "image"))
        ring = CameraRingConfig(
            **_build(CameraRingConfig, data.pop("camera_ring", {}), "camera_ring")
        )
        source = data.get("pose_source", "procedural")
        if source not in POSE_SOURCES:
            raise ConfigError(f"pose_source must be one of {list(POSE_SOURCES)}")
        subsets = data.pop("amass_subsets", [] if source == "procedural" else None)
        if not isinstance(subsets, list) or not all(isinstance(s, str) for s in subsets):
            raise ConfigError("amass_subsets: must be a list of strings")
        if source == "amass" and not subsets:
            raise ConfigError("amass_subsets: required non-empty list when pose_source is amass")
        cfg = GenerationConfig(amass_subsets=tuple(subsets), image=image, camera_ring=ring, **data)
    except TypeError as e:
        raise ConfigError(str(e)) from e
    _check(cfg)
    return cfg


def _check(cfg: GenerationConfig) -> None:
    if not isinstance(cfg.dataset_name, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", cfg.dataset_name
    ):
        raise ConfigError("dataset_name must match [A-Za-z0-9_-]+")
    ints = {
        "num_poses": cfg.num_poses, "frame_stride": cfg.frame_stride,
        "image.height": cfg.image.height, "image.width": cfg.image.width,
        "camera_ring.count": cfg.camera_ring.count,
    }  # fmt: skip
    for k, v in ints.items():
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ConfigError(f"{k} must be an integer >= 1")
    if not isinstance(cfg.num_betas, int) or cfg.num_betas < 0:
        raise ConfigError("num_betas must be an integer >= 0")
    for k, v in {
        "camera_ring.distance_m": cfg.camera_ring.distance_m,
        "camera_ring.focal_px": cfg.camera_ring.focal_px,
    }.items():
        if not isinstance(v, int | float) or v <= 0:
            raise ConfigError(f"{k} must be a number > 0")
    if not isinstance(cfg.camera_ring.height_m, int | float):
        raise ConfigError("camera_ring.height_m must be a number")


def load_config(path: Path | str) -> GenerationConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
