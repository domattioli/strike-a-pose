"""Camera geometry of a rig: intrinsics, look-at pose, placement noise, 6D encoding, projection.

Every view of a body comes from a pinhole camera with fixed intrinsics and a known placement. This
module holds that geometry for specs/001-kill-test-mvp: the CameraPlacement entity of
data-model.md, the camera section of contracts/config.md, and the camera encoding of research R7.
Public sources:

* The pinhole camera model and the OpenCV camera conventions: Hartley and Zisserman, "Multiple
  View Geometry in Computer Vision" (2nd edition, 2004), and the OpenCV documentation of camera
  calibration and 3D reconstruction (https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
* Rodrigues' rotation formula, for the placement-noise rotation about an axis
  (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).
* Zhou, Barnes, Lu, Yang, and Li, "On the Continuity of Rotation Representations in Neural
  Networks" (CVPR 2019, https://arxiv.org/abs/1812.07035), for the 6D rotation representation
  (research R7).

Conventions:

* The world frame is in metres with the vertical axis y (up) and the floor at y = 0. A body faces
  +z and its left side is +x, as in body/base.py.
* The azimuth of a camera is the angle from the +z axis toward the +x axis, in degrees, around the
  vertical line through the point the rig is centered on. Azimuth 0 puts the camera in front of a
  body, and azimuth 90 degrees puts it on the body's left. The distance is horizontal, and the
  height is the y coordinate of the camera, so it is measured from the floor.
* The camera frame has x to the right, y down, and z forward. The extrinsics go from world to
  camera: a world point X maps to ``R X + t`` with R a proper rotation.
* Pixel centers lie at integer coordinates, as in OpenCV and the cv2 drawing calls, so the
  principal point of a square image of S pixels is at ((S - 1) / 2, (S - 1) / 2). A projected
  point can go to a cv2 drawing call as it is.
* Array arguments may carry leading batch dimensions, which broadcast against each other.
  ``project_points`` is the exception: it handles one camera at a time.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.config import ConfigError

__all__ = [
    "CAMERA_ENCODING_DIM",
    "TRANSLATION_SCALE_M",
    "Rig",
    "azimuths_are_separated",
    "encode_camera",
    "given_rotation",
    "intrinsics",
    "look_at_extrinsics",
    "noise_rotation",
    "orbit_position",
    "project_points",
    "random_unit_vectors",
    "rotation_angle_deg",
    "rotation_to_6d",
    "sample_rig",
]

# Research R7: the encoder receives the translation divided by 5 m, which turns the camera
# distances of 2.5 m to 4 m into values near 0.5 to 0.8.
TRANSLATION_SCALE_M = 5.0

# The camera encoding is the 6D rotation (6 values) followed by the scaled translation (3 values).
CAMERA_ENCODING_DIM = 9

# The world's up direction, +y. The look-at pose keeps the horizon of the image level against it.
_WORLD_UP = np.array([0.0, 1.0, 0.0])

# A vector shorter than this (in metres, or as a unit-less axis) has no usable direction.
_MIN_LENGTH = 1e-9

# A viewing direction whose sine against the vertical is below this is "straight up or down", and
# the roll of the image is then undefined.
_MIN_HORIZONTAL_SINE = 1e-6

# Whole-rig redraws before sampling gives up. With the defaults (4 cameras, 20 degrees) one draw in
# two is accepted, so this limit is reached only when the separation is almost too wide to fit.
_MAX_RIG_ATTEMPTS = 100_000


@dataclass(frozen=True, eq=False)
class Rig:
    """The true cameras of one body, stacked over the cameras of the rig.

    The arrays have the layout of one body in a shard (contracts/artifacts.md), and camera ``i`` of
    the rig is view ``i`` of the body. ``K`` is shared by all cameras. ``R_true`` and ``t_true`` are
    the world-to-camera pose (``R_true @ X + t_true``). ``azimuth_deg``, ``distance_m``, and
    ``height_m`` are the cylindrical coordinates of each camera around the rig's center.
    ``lookat`` is the point each camera is aimed at. ``noise_axis`` holds one unit vector per
    camera, drawn once and stored: every cell rotates the camera about it by its own noise angle.
    """

    K: NDArray[np.float64]  # (3, 3)
    R_true: NDArray[np.float64]  # (n_cameras, 3, 3)
    t_true: NDArray[np.float64]  # (n_cameras, 3)
    azimuth_deg: NDArray[np.float64]  # (n_cameras,)
    distance_m: NDArray[np.float64]  # (n_cameras,)
    height_m: NDArray[np.float64]  # (n_cameras,)
    lookat: NDArray[np.float64]  # (n_cameras, 3)
    noise_axis: NDArray[np.float64]  # (n_cameras, 3)


def intrinsics(image_size: int, focal_px: float) -> NDArray[np.float64]:
    """Return the 3 by 3 intrinsic matrix K of the square pinhole camera of a configuration.

    The focal length is ``focal_px`` pixels on both axes, there is no skew, and the principal point
    is the image center, ``(image_size - 1) / 2`` on both axes (pixel centers at integer
    coordinates). The field of view is ``2 * atan(image_size / (2 * focal_px))`` on both axes, which
    stays 53 degrees when the focal length moves with the image size.
    """
    if image_size < 1:
        raise ValueError(f"image_size must be at least 1, not {image_size}")
    if not (math.isfinite(focal_px) and focal_px > 0.0):
        raise ValueError(f"focal_px must be a positive number, not {focal_px}")
    center = (float(image_size) - 1.0) / 2.0
    focal = float(focal_px)
    return np.array([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float64)


def orbit_position(
    azimuth_deg: ArrayLike, distance_m: ArrayLike, height_m: ArrayLike, center: ArrayLike
) -> NDArray[np.float64]:
    """Return the world position of a camera on a circle around the vertical line through center.

    Azimuth 0 puts the camera on the +z side of ``center`` and azimuth 90 degrees on its +x side.
    ``distance_m`` is the horizontal distance from the vertical line through ``center``, and
    ``height_m`` is the y coordinate of the camera, not a height above ``center``. The last axis of
    the result holds x, y, z.
    """
    azimuth = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    distance = np.asarray(distance_m, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    middle = _as_points(center, "center")
    x = middle[..., 0] + distance * np.sin(azimuth)
    z = middle[..., 2] + distance * np.cos(azimuth)
    return np.stack(np.broadcast_arrays(x, height, z), axis=-1)


def look_at_extrinsics(
    position: ArrayLike, target: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the world-to-camera pose ``(R, t)`` of a camera at position that looks at target.

    The optical axis points at ``target``, and the image is upright against the world's +y axis:
    the camera's x axis (image right) is ``forward x up`` and its y axis (image down) is
    ``forward x right``. Then ``R`` has the camera axes as rows, and ``t = -R @ position``, so a
    world point ``X`` is at ``R @ X + t`` in the camera frame and ``position`` maps to the origin.
    Raises ValueError when the camera sits on its target or looks straight up or down.
    """
    eye = _as_points(position, "position")
    aim = _as_points(target, "target")
    forward = aim - eye
    distance = np.linalg.norm(forward, axis=-1, keepdims=True)
    if not np.all(distance > _MIN_LENGTH):
        raise ValueError("a camera cannot look at a target at its own position")
    forward = forward / distance
    right = np.cross(forward, _WORLD_UP)
    right_length = np.linalg.norm(right, axis=-1, keepdims=True)
    if not np.all(right_length > _MIN_HORIZONTAL_SINE):
        raise ValueError("a camera looking straight up or down has no defined image roll")
    right = right / right_length
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=-2)
    translation = -np.einsum("...ij,...j->...i", rotation, eye)
    return rotation, translation


def azimuths_are_separated(azimuth_deg: ArrayLike, min_separation_deg: float) -> bool:
    """Return True when every two azimuths are at least min_separation_deg apart on the circle.

    The distance between two azimuths is the shorter way around the circle, so 350 and 5 degrees
    are 15 degrees apart. The boundary is inclusive. Fewer than two azimuths have no pair.
    """
    azimuth = np.asarray(azimuth_deg, dtype=np.float64).ravel()
    if azimuth.size < 2:
        return True
    # The closest pair on a circle is a pair that is next to each other in sorted order, including
    # the pair that wraps from the largest azimuth back to the smallest.
    ordered = np.sort(np.mod(azimuth, 360.0))
    gaps = np.diff(ordered, append=ordered[0] + 360.0)
    return bool(np.all(gaps >= min_separation_deg))


def sample_rig(rng: np.random.Generator, camera: Mapping[str, Any], center: ArrayLike) -> Rig:
    """Draw the cameras of one body around center, the center of its posed mesh's bounding box.

    ``camera`` is the ``camera`` section of a resolved configuration (``config["camera"]``). The
    function reads ``n_cameras``, ``image_size``, ``focal_px``, ``distance_m``, ``height_m``,
    ``min_separation_deg``, and ``lookat_jitter_m``. Heights are measured from the floor y = 0, so
    the caller must stand the posed mesh on the floor before it takes the bounding-box center.

    The azimuths are uniform around the circle. The whole rig is redrawn until every pair is at
    least ``min_separation_deg`` apart, which keeps the draw uniform over the allowed rigs and
    keeps every subset of cameras separated, such as the first k cameras of a k-view cell. The
    distance and the height are uniform in their ranges, independently per camera. Camera i stands
    at its distance, height, and azimuth around the vertical line through ``center`` and looks at
    ``center`` plus a jitter that is uniform within the configured bound per axis, independently
    per camera. The placement-noise axis of each camera is a uniform random unit vector: the
    caller stores it, and each cell rotates about it by its own angle.

    The generator is read in this fixed order, so a rig depends only on the generator state:
    azimuth draws (``n_cameras`` values per attempt), distances, heights, look-at jitter
    (``n_cameras`` by 3), then noise-axis normal draws (``n_cameras`` by 3).

    Raises ConfigError (exit code 2) when ``n_cameras`` cameras cannot be kept
    ``min_separation_deg`` apart, which config.py cannot see because the rule spans two keys, and
    ValueError when ``center`` is not one point of three finite numbers.
    """
    n_cameras = int(camera["n_cameras"])
    min_separation = float(camera["min_separation_deg"])
    distance_low, distance_high = camera["distance_m"]
    height_low, height_high = camera["height_m"]
    jitter = float(camera["lookat_jitter_m"])
    middle = _as_points(center, "center")
    if middle.shape != (3,):
        raise ValueError(f"center must be one point of shape (3,), not shape {middle.shape}")

    azimuth = _draw_separated_azimuths(rng, n_cameras, min_separation)
    distance = rng.uniform(distance_low, distance_high, size=n_cameras)
    height = rng.uniform(height_low, height_high, size=n_cameras)
    lookat = middle + rng.uniform(-jitter, jitter, size=(n_cameras, 3))
    noise_axis = random_unit_vectors(rng, n_cameras)

    position = orbit_position(azimuth, distance, height, middle)
    rotation, translation = look_at_extrinsics(position, lookat)
    return Rig(
        K=intrinsics(camera["image_size"], camera["focal_px"]),
        R_true=rotation,
        t_true=translation,
        azimuth_deg=azimuth,
        distance_m=distance,
        height_m=height,
        lookat=lookat,
        noise_axis=noise_axis,
    )


def random_unit_vectors(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    """Return count unit vectors of shape (count, 3), uniform on the sphere.

    Each vector is a standard normal draw divided by its length, which is uniform on the sphere
    because the normal density depends on the length only. A zero-length draw has probability zero
    in practice, and ``noise_rotation`` refuses a zero axis.
    """
    vectors = rng.standard_normal((count, 3))
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def noise_rotation(axis: ArrayLike, angle_deg: ArrayLike) -> NDArray[np.float64]:
    """Return the rotation matrix for a turn of angle_deg degrees about axis (right-hand rule).

    This is Rodrigues' formula, ``I + sin(a) [n]x + (1 - cos(a)) [n]x^2`` for the unit axis ``n``,
    with ``1 - cos(a)`` written as ``2 sin(a / 2)^2`` so that small angles keep full precision.
    The axis is normalized here, so it need not be a unit vector. An angle of 0 gives the identity
    exactly. ``axis`` has shape (..., 3), ``angle_deg`` broadcasts against its leading dimensions,
    and the result has shape (..., 3, 3). Raises ValueError for a zero-length axis.
    """
    direction = _as_points(axis, "axis")
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    if not np.all(length > _MIN_LENGTH):
        raise ValueError("a rotation axis must have a length above zero")
    direction = direction / length
    angle = np.radians(np.asarray(angle_deg, dtype=np.float64))
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    zero = np.zeros_like(x)
    cross = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    sine = np.sin(angle)[..., None, None]
    versine = (2.0 * np.sin(0.5 * angle) ** 2)[..., None, None]
    return np.eye(3) + sine * cross + versine * (cross @ cross)


def given_rotation(
    true_rotation: ArrayLike, noise_axis: ArrayLike, noise_deg: ArrayLike
) -> NDArray[np.float64]:
    """Return the rotation given to the model, ``R_noise(noise_deg, noise_axis) @ R_true``.

    The noise turns the camera by ``noise_deg`` degrees about ``noise_axis``, an axis fixed in the
    camera frame (data-model.md, CameraPlacement; FR-005). The translation stays as it was, and
    the silhouette stays rendered from the true placement. With 0 degrees the result is
    ``true_rotation`` exactly.
    """
    return noise_rotation(noise_axis, noise_deg) @ _as_matrices(true_rotation, "true_rotation")


def rotation_angle_deg(rotation: ArrayLike) -> NDArray[np.float64]:
    """Return the angle in degrees, from 0 to 180, of each rotation matrix in rotation.

    The angle comes from ``atan2`` of the sine and the cosine, so it stays accurate for tiny
    angles and for angles near 180 degrees, where ``arccos`` of the cosine alone loses digits. The
    sine is ``||R - R^T||_F / (2 sqrt(2))`` and the cosine is ``(trace(R) - 1) / 2``.
    """
    matrices = _as_matrices(rotation, "rotation")
    cosine = 0.5 * (np.trace(matrices, axis1=-2, axis2=-1) - 1.0)
    skew = matrices - np.swapaxes(matrices, -1, -2)
    sine = np.sqrt(np.sum(skew * skew, axis=(-2, -1)) / 8.0)
    return np.degrees(np.arctan2(sine, cosine))


def rotation_to_6d(rotation: ArrayLike) -> NDArray[np.float64]:
    """Return the 6D representation of each rotation matrix: its first two columns, joined.

    This is the mapping of Zhou et al. 2019 (research R7) that drops the last column of the
    matrix. The result has shape (..., 6): the first column, then the second column. The third
    column is the cross product of the two, so no information is lost.
    """
    matrices = _as_matrices(rotation, "rotation")
    return np.concatenate([matrices[..., :, 0], matrices[..., :, 1]], axis=-1)


def encode_camera(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Return the camera encoding for the model: the 6D rotation, then ``t / 5 m`` (R7).

    This is the camera placement given to the encoder in research R7. ``rotation`` has shape
    (..., 3, 3) and ``translation`` has shape (..., 3). The result has shape
    (..., ``CAMERA_ENCODING_DIM``) in float64. The caller passes ``R_given`` and ``t_true``.
    """
    code = rotation_to_6d(rotation)
    shift = _as_points(translation, "translation") / TRANSLATION_SCALE_M
    batch = np.broadcast_shapes(code.shape[:-1], shift.shape[:-1])
    code = np.broadcast_to(code, (*batch, 6))
    shift = np.broadcast_to(shift, (*batch, 3))
    return np.concatenate([code, shift], axis=-1)


def project_points(
    points: ArrayLike, K: ArrayLike, R: ArrayLike, t: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project world points through one pinhole camera and return ``(pixels, depth)``.

    ``points`` has shape (..., 3), ``K`` is 3 by 3, ``R`` is 3 by 3, and ``t`` has shape (3,).
    The camera point is ``R @ X + t``, the depth is its z coordinate, and the pixel is
    ``K @ camera_point`` divided by the depth. Pixels have shape (..., 2) as (column, row) with
    pixel centers at integer coordinates, and depth has shape (...). A point at or behind the
    camera plane (depth not above 0) has no image: its pixel is NaN, and the caller drops it.
    """
    world = _as_points(points, "points")
    intrinsic = _as_matrices(K, "K")
    rotation = _as_matrices(R, "R")
    translation = _as_points(t, "t")
    camera_points = world @ rotation.T + translation
    depth = camera_points[..., 2]
    in_front = depth > 0.0
    safe_depth = np.where(in_front, depth, 1.0)
    pixels = (camera_points @ intrinsic.T)[..., :2] / safe_depth[..., None]
    return np.where(in_front[..., None], pixels, np.nan), depth


def _draw_separated_azimuths(
    rng: np.random.Generator, n_cameras: int, min_separation_deg: float
) -> NDArray[np.float64]:
    """Draw n_cameras azimuths in [0, 360) and redraw the whole set until every pair is apart."""
    if n_cameras >= 2 and n_cameras * min_separation_deg >= 360.0:
        raise ConfigError(
            f"camera.min_separation_deg is {min_separation_deg:g}, which leaves no room for "
            f"{n_cameras} cameras around a circle; camera.n_cameras times "
            "camera.min_separation_deg must stay below 360",
            "camera.min_separation_deg",
        )
    for _ in range(_MAX_RIG_ATTEMPTS):
        azimuth = rng.uniform(0.0, 360.0, size=n_cameras)
        if azimuths_are_separated(azimuth, min_separation_deg):
            return azimuth
    raise ConfigError(
        f"no placement of {n_cameras} cameras at least {min_separation_deg:g} degrees apart was "
        f"found in {_MAX_RIG_ATTEMPTS} draws; lower camera.min_separation_deg",
        "camera.min_separation_deg",
    )


def _as_points(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return values as a float64 array whose last axis holds 3 finite coordinates."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must end in an axis of length 3, not shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must hold finite numbers")
    return array


def _as_matrices(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return values as a float64 array whose last two axes form a finite 3 by 3 matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must end in a 3 by 3 matrix, not shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must hold finite numbers")
    return array
