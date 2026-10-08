"""Preprocessing of real photographs: padding, resize, threshold, mask status, nominal cameras.

Real silhouettes (BodyM, SSP-3D) enter the model through the same path as the synthetic renders.
This module holds that path for specs/001-kill-test-mvp: pad the mask to a square, resize it to
``camera.image_size`` with area interpolation, threshold it at 0.5, then decide its status with the
usability and multi-person rules of research R12 and data-model.md (the status order is fixed:
no mask, then multi-person, then unusable, then ok). It also builds the nominal camera of each
real view from ``real.nominal_camera`` (research R12 and contracts/config.md).

Public sources:

* The area interpolation of OpenCV ``cv2.resize`` with ``INTER_AREA`` (OpenCV documentation of
  geometric image transformations, https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).
* Connected components with OpenCV ``cv2.connectedComponentsWithStats`` and 8-connectivity
  (the same OpenCV documentation, image segmentation section).
* The pinhole camera and look-at conventions of camera.py (Hartley and Zisserman, 2004), reused
  through ``camera.intrinsics``, ``camera.orbit_position``, and ``camera.look_at_extrinsics``.
* The multi-person component fraction of 10 percent is the rule of research R12 for this project;
  no published method is implemented here beyond the OpenCV calls above.
"""

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.camera import intrinsics, look_at_extrinsics, orbit_position

__all__ = [
    "MASK_STATUSES",
    "MULTI_PERSON_COMPONENT_FRACTION",
    "NominalCamera",
    "mask_status",
    "nominal_camera",
    "pad_to_square",
    "prepare_mask",
]

# A mask with two connected components that each hold at least this share of the mask area is a
# two-person mask (research R12).
MULTI_PERSON_COMPONENT_FRACTION = 0.10

# The mask statuses of data-model.md, in the order in which they are decided.
MASK_STATUSES = ("skipped_no_mask", "skipped_multi_person", "skipped_unusable", "ok")

# The threshold applied to the area-resized mask, which holds values between 0 and 1.
_THRESHOLD = 0.5

# The neighbourhood of the connected-component labelling.
_CONNECTIVITY = 8

NominalCamera = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def pad_to_square(image: ArrayLike) -> NDArray[Any]:
    """Return the image padded with zeros to a square, centered on the shorter side.

    The side of the square is the longer side of the input. When the padding is odd, the extra row
    or column goes to the bottom or the right. The input is a two-dimensional array with at least
    one pixel, and the output has the same dtype. Raises ValueError for any other shape.
    """
    grid = np.asarray(image)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"an image must be a non-empty 2D array, not shape {grid.shape}")
    height, width = grid.shape
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    square = np.zeros((side, side), dtype=grid.dtype)
    square[top : top + height, left : left + width] = grid
    return square


def prepare_mask(mask: ArrayLike, image_size: int) -> NDArray[np.uint8]:
    """Return the real mask as a binary square of ``image_size`` pixels, with values 0 and 1.

    The mask is padded to a square, resized with area interpolation (``cv2.INTER_AREA``), and
    thresholded at 0.5: a pixel is set when at least half of its area was foreground. A mask with
    values up to 255 (such as an 8-bit photograph mask) is scaled to 0 to 1 first, so a boolean,
    0 and 1, or 0 and 255 mask all work. Raises ValueError for a mask that is not a non-empty 2D
    array, has a value outside 0 to 255, or ``image_size`` below 1.
    """
    if image_size < 1:
        raise ValueError(f"image_size must be at least 1, not {image_size}")
    grid = np.asarray(mask)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"a mask must be a non-empty 2D array, not shape {grid.shape}")
    values = grid.astype(np.float32)
    if values.size and (values.min() < 0.0 or values.max() > 255.0):
        raise ValueError("a mask must hold values from 0 to 255")
    if values.max() > 1.0:
        values = values / 255.0
    square = pad_to_square(values).astype(np.float32)
    resized = cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return (resized >= _THRESHOLD).astype(np.uint8)


def mask_status(mask: ArrayLike | None, real: Mapping[str, Any]) -> str:
    """Return the status of a prepared mask: one of MASK_STATUSES.

    ``mask`` is a two-dimensional binary array (see ``prepare_mask``), or None when the dataset
    gives no mask for the photograph. ``real`` is the ``real`` section of a resolved configuration.
    The rules run in this order, so the status does not depend on which rule is checked first:

    1. No mask, or a mask with no set pixels, gives ``skipped_no_mask``.
    2. Two connected components that each hold at least ``MULTI_PERSON_COMPONENT_FRACTION`` (10
       percent) of the mask area give ``skipped_multi_person``.
    3. A mask whose area is below ``real["mask_min_area_fraction"]`` of the image, or whose largest
       component holds less than ``real["mask_min_component_fraction"]`` of the mask area, gives
       ``skipped_unusable``.
    4. Any other mask gives ``ok``.

    Raises ValueError when ``mask`` is not a two-dimensional array of 0 and 1.
    """
    if mask is None:
        return "skipped_no_mask"
    grid = np.asarray(mask)
    if grid.ndim != 2:
        raise ValueError(f"a mask must be a 2D array, not shape {grid.shape}")
    if not np.all((grid == 0) | (grid == 1)):
        raise ValueError("a mask must hold only the values 0 and 1")
    binary = grid.astype(np.uint8)
    mask_area = int(np.count_nonzero(binary))
    if mask_area == 0:
        return "skipped_no_mask"

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=_CONNECTIVITY
    )
    component_areas = stats[1:count, cv2.CC_STAT_AREA].astype(np.float64)

    significant = component_areas >= MULTI_PERSON_COMPONENT_FRACTION * mask_area
    if int(np.count_nonzero(significant)) >= 2:
        return "skipped_multi_person"

    area_fraction = mask_area / binary.size
    largest_fraction = float(component_areas.max()) / mask_area
    usable = (
        area_fraction >= float(real["mask_min_area_fraction"])
        and largest_fraction >= float(real["mask_min_component_fraction"])
    )
    return "ok" if usable else "skipped_unusable"


def nominal_camera(
    real: Mapping[str, Any], camera: Mapping[str, Any], azimuth_deg: float
) -> NominalCamera:
    """Return ``(K, R, t)`` of the nominal camera of a real view at one azimuth.

    The camera stands ``real["nominal_camera"]["distance_m"]`` metres away from the vertical line
    through the origin, at the height ``height_m`` above the floor, and it looks at the point
    ``lookat_height_m`` above the floor on that line (the center of an average upright body, as in
    the synthetic rigs). Azimuth 0 puts the camera on the +z side and 90 degrees on the +x side,
    as in camera.py. ``K`` comes from ``camera["image_size"]`` and ``camera["focal_px"]``. ``R`` and
    ``t`` are the world-to-camera pose. The nominal camera has no placement noise.

    Raises ValueError when the configured distance is not positive or the camera sits on its
    target.
    """
    nominal = real["nominal_camera"]
    distance = float(nominal["distance_m"])
    height = float(nominal["height_m"])
    look_height = float(nominal["lookat_height_m"])
    if not distance > 0.0:
        raise ValueError(f"nominal_camera.distance_m must be positive, not {distance}")
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    target = np.array([0.0, look_height, 0.0], dtype=np.float64)
    position = orbit_position(float(azimuth_deg), distance, height, center)
    rotation, translation = look_at_extrinsics(position, target)
    K = intrinsics(int(camera["image_size"]), float(camera["focal_px"]))
    return K, rotation, translation
