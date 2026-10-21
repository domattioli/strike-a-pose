"""Generation config (YAML) — see specs/001-dataset-generation/data-model.md."""

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .procedural import RANGES_TABLES, RANGES_VERSION

POSE_SOURCES = ("procedural", "amass")
# Generator fields added after dataset layout v1 -> the value that reproduces earlier
# datasets. to_dict() omits a field while it holds this value (spec 010 FR-002).
ADDED_FIELD_DEFAULTS: dict[str, Any] = {
    "ranges_version": RANGES_VERSION,
    "body_shape": None,
    "camera_jitter": None,
}


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
class BodyShapeConfig:
    """Per-pose random betas, N(0, std^2) clipped to [-clip, clip] (spec 010 FR-004)."""

    std: float = 1.0
    clip: float = 2.0


@dataclass(frozen=True)
class CameraJitterConfig:
    """Per-sample camera around its ring camera (spec 010 FR-005): height and distance deltas
    (m), then aimed at the ring's look-at point and tilted down by pitch_deg (> 0 = down);
    accepted only if every mesh vertex projects margin_px inside the image."""

    height_m: tuple[float, float] = (-0.3, 0.3)
    distance_m: tuple[float, float] = (0.0, 0.5)
    pitch_deg: tuple[float, float] = (0.0, 5.0)
    margin_px: int = 2
    max_attempts: int = 16


# camera_jitter range bounds (lo_min, hi_max): sane, upright cameras, never closer than the ring
_JITTER_BOUNDS = {"height_m": (-1.0, 1.0), "distance_m": (0.0, 3.0), "pitch_deg": (-30.0, 30.0)}


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
    # spec 010 (left out of to_dict() while at their defaults, see ADDED_FIELD_DEFAULTS)
    ranges_version: str = RANGES_VERSION  # procedural table version
    body_shape: BodyShapeConfig | None = None  # None: mean shape (betas = 0)
    camera_jitter: CameraJitterConfig | None = None  # None: the ring cameras

    def to_dict(self) -> dict[str, Any]:
        """Resolved config for the manifest; fields added after layout v1 are left out while
        they hold their default, so earlier configs serialize unchanged (spec 010 FR-002)."""
        d = asdict(self)
        d["amass_subsets"] = list(self.amass_subsets)
        for key, default in ADDED_FIELD_DEFAULTS.items():
            if getattr(self, key) == default:
                del d[key]
        if "camera_jitter" in d:
            for key in _JITTER_BOUNDS:
                d["camera_jitter"][key] = list(d["camera_jitter"][key])
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
        if data.get("body_shape") is not None:
            data["body_shape"] = _body_shape(data["body_shape"])
        if data.get("camera_jitter") is not None:
            data["camera_jitter"] = _camera_jitter(data["camera_jitter"])
        cfg = GenerationConfig(amass_subsets=tuple(subsets), image=image, camera_ring=ring, **data)
    except TypeError as e:
        raise ConfigError(str(e)) from e
    _check(cfg)
    return cfg


def _number(v: Any, where: str, positive: bool = False) -> float:
    if not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v):
        raise ConfigError(f"{where} must be a finite number")
    if positive and v <= 0:
        raise ConfigError(f"{where} must be > 0")
    return float(v)


def _body_shape(data: Any) -> BodyShapeConfig:
    d = _build(BodyShapeConfig, data, "body_shape")
    vals = {k: _number(v, f"body_shape.{k}", positive=True) for k, v in d.items()}
    return BodyShapeConfig(**vals)


def _camera_jitter(data: Any) -> CameraJitterConfig:
    d = dict(_build(CameraJitterConfig, data, "camera_jitter"))
    for key, (lo_min, hi_max) in _JITTER_BOUNDS.items():
        if key in d:
            v = d[key]
            if not isinstance(v, list | tuple) or len(v) != 2:
                raise ConfigError(f"camera_jitter.{key} must be a [lo, hi] pair")
            lo, hi = (_number(x, f"camera_jitter.{key}") for x in v)
            if not lo_min <= lo <= hi <= hi_max:
                raise ConfigError(f"camera_jitter.{key} needs {lo_min} <= lo <= hi <= {hi_max}")
            d[key] = (lo, hi)
    for key, least in (("margin_px", 0), ("max_attempts", 1)):
        if key in d:
            v = d[key]
            if not isinstance(v, int) or isinstance(v, bool) or v < least:
                raise ConfigError(f"camera_jitter.{key} must be an integer >= {least}")
    return CameraJitterConfig(**d)


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
    if not isinstance(cfg.ranges_version, str) or cfg.ranges_version not in RANGES_TABLES:
        raise ConfigError(f"ranges_version must be one of {list(RANGES_TABLES)}")
    if cfg.pose_source != "procedural" and cfg.ranges_version != RANGES_VERSION:
        raise ConfigError("ranges_version applies to pose_source: procedural only")
    if cfg.body_shape is not None and cfg.num_betas < 1:
        raise ConfigError("body_shape needs num_betas >= 1")


def load_config(path: Path | str) -> GenerationConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return config_from_dict(data)
