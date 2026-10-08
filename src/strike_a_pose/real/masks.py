"""Person masks for real photographs: the MaskBackend protocol, provided and SAM 2 masks, status.

A real photograph without a provided silhouette gets a person mask from a promptable segmentation
model (FR-019). ``ProvidedMasks`` returns the silhouettes that a dataset supplies (research R12, the
default ``real.ssp3d.mask_source`` of config.md). ``Sam2Masks`` recomputes a mask from the
photograph with the SAM 2 image predictor and a box prompt taken from the dataset bounding box
(research R12).

Every subject gets one status, decided in this order (data-model.md, RealImageSubject):

* no mask gives ``skipped_no_mask``;
* two connected components that each hold at least 10 percent of the mask area give
  ``skipped_multi_person`` (checked before usability, so the status does not depend on the order of
  the rules: a two-person mask can also fail the usability rule);
* a mask that fails the usability rule gives ``skipped_unusable``: it covers less than
  ``real.mask_min_area_fraction`` of the image, or its largest connected component holds less than
  ``real.mask_min_component_fraction`` of the mask area;
* otherwise the status is ``ok``.

The SAM 2 package is an optional extra (``[real]`` in pyproject.toml). It is imported only when a
``Sam2Masks`` object is built, so the rest of the package works without it. The checkpoint is read
from the asset root and is never downloaded (constitution Principle II). Public sources: Ravi et
al., "SAM 2: Segment Anything in Images and Videos" (2024, https://github.com/facebookresearch/sam2,
Apache-2.0), used through its public ``build_sam2`` and ``SAM2ImagePredictor`` interface
(research R12).
"""

from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from strike_a_pose.assets import require_asset

__all__ = [
    "DEFAULT_MIN_AREA_FRACTION",
    "DEFAULT_MIN_COMPONENT_FRACTION",
    "MULTI_PERSON_COMPONENT_FRACTION",
    "SAM2_ASSET_KEY",
    "SAM2_CONFIG",
    "BoundingBox",
    "MaskBackend",
    "MaskStatus",
    "ProvidedMasks",
    "Sam2Masks",
    "SubjectMask",
    "mask_status",
    "segment_subject",
]

# The configuration key that names the SAM 2 checkpoint (config.md, real section).
SAM2_ASSET_KEY = "real.sam2.checkpoint"

# The model configuration of the SAM 2.1 hiera base plus checkpoint, resolved by the sam2 package.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Defaults of the usability rule (config.md: real.mask_min_area_fraction and
# real.mask_min_component_fraction). Callers pass the configured values.
DEFAULT_MIN_AREA_FRACTION = 0.02
DEFAULT_MIN_COMPONENT_FRACTION = 0.90

# A connected component holds a person when it has at least this share of the mask area (R12).
MULTI_PERSON_COMPONENT_FRACTION = 0.10

# A bounding box in pixels of the photograph: x0, y0, x1, y1.
BoundingBox = tuple[float, float, float, float]


class MaskStatus(str, Enum):
    """The per-subject mask status of RealImageSubject (data-model.md)."""

    OK = "ok"
    SKIPPED_NO_MASK = "skipped_no_mask"
    SKIPPED_UNUSABLE = "skipped_unusable"
    SKIPPED_MULTI_PERSON = "skipped_multi_person"


class SubjectMask(NamedTuple):
    """The mask of one subject and its status. The mask is None when no mask was produced."""

    mask: NDArray[np.bool_] | None
    status: MaskStatus


@runtime_checkable
class MaskBackend(Protocol):
    """A source of person masks. ``name`` is the value of the mask_source column."""

    name: str

    def segment(
        self,
        subject_id: str,
        image: NDArray[np.uint8],
        box: BoundingBox | None,
    ) -> NDArray[np.bool_] | None:
        """Return the mask of one subject as a 2-D boolean array, or None when no mask is produced.

        ``image`` is the photograph as an RGB uint8 array of shape (H, W, 3). ``box`` is the dataset
        bounding box of the person, or None when the dataset has none.
        """
        ...


class ProvidedMasks:
    """Silhouettes that the dataset supplies, keyed by subject identifier (mask_source provided)."""

    name = "provided"

    def __init__(self, masks: dict[str, NDArray[np.generic]]) -> None:
        self._masks = {key: np.asarray(mask).astype(bool) for key, mask in masks.items()}

    def segment(
        self,
        subject_id: str,
        image: NDArray[np.uint8],
        box: BoundingBox | None,
    ) -> NDArray[np.bool_] | None:
        """Return the supplied silhouette of the subject; the image and the box are not used."""
        return self._masks.get(subject_id)


class Sam2Masks:
    """Person masks from SAM 2 with a box prompt (mask_source ``sam2``).

    Building the object imports the sam2 package and reads the checkpoint. A missing package raises
    ``ImportError`` that names the ``[real]`` extra. A missing checkpoint raises
    ``MissingAssetError``, which names the file and the key ``real.sam2.checkpoint``.
    """

    name = "sam2"

    def __init__(
        self,
        checkpoint: str,
        asset_root: str | Path | None,
        *,
        device: str = "cpu",
    ) -> None:
        build_sam2, predictor_class = _import_sam2()
        checkpoint_path = require_asset(
            checkpoint,
            SAM2_ASSET_KEY,
            None if asset_root is None else Path(asset_root),
        )
        model = build_sam2(SAM2_CONFIG, str(checkpoint_path), device=device)
        self._predictor = predictor_class(model)

    def segment(
        self,
        subject_id: str,
        image: NDArray[np.uint8],
        box: BoundingBox | None,
    ) -> NDArray[np.bool_] | None:
        """Return the SAM 2 mask inside the box prompt; None when there is no box to prompt with."""
        if box is None:
            return None
        self._predictor.set_image(image)
        masks, _scores, _logits = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=False,
        )
        return np.asarray(masks[0]).astype(bool)


def _import_sam2() -> tuple[object, object]:
    """Import the sam2 entry points on demand. Raise ImportError when the package is absent."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise ImportError(
            "SAM 2 masks need the optional package sam2; install the extra [real] "
            "(pip install 'strike_a_pose[real]')"
        ) from error
    return build_sam2, SAM2ImagePredictor


def mask_status(
    mask: NDArray[np.generic] | None,
    *,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION,
) -> MaskStatus:
    """Return the status of one mask.

    The rules live in one place, ``strike_a_pose.real.common.mask_status``; this wrapper only
    supplies the thresholds as keyword arguments and returns the enum member.
    """
    from strike_a_pose.real import common

    if mask is None:
        return MaskStatus.SKIPPED_NO_MASK
    binary = np.asarray(mask).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"a mask must be two-dimensional, got shape {binary.shape}")
    status = common.mask_status(
        binary,
        {
            "mask_min_area_fraction": min_area_fraction,
            "mask_min_component_fraction": min_component_fraction,
        },
    )
    return MaskStatus(status)


def segment_subject(
    backend: MaskBackend,
    subject_id: str,
    image: NDArray[np.uint8],
    box: BoundingBox | None,
    *,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION,
) -> SubjectMask:
    """Return the mask of one subject from the backend, with its status (FR-019)."""
    mask = backend.segment(subject_id, image, box)
    status = mask_status(
        mask,
        min_area_fraction=min_area_fraction,
        min_component_fraction=min_component_fraction,
    )
    return SubjectMask(None if mask is None else np.asarray(mask).astype(bool), status)


def _component_areas(binary: NDArray[np.bool_]) -> list[int]:
    """Return the areas of the connected components of a non-empty mask, largest first."""
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    # Label 0 is the background, so the first row of the statistics is skipped.
    areas = [int(area) for area in stats[1:, cv2.CC_STAT_AREA]]
    return sorted(areas, reverse=True)
