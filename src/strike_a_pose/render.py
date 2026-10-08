"""Silhouette renderer: the union of a posed mesh's projected triangles as a binary mask.

The silhouette of an opaque mesh is the union of the projections of all its triangles, so no depth
test is needed (research R1). The renderer projects every vertex once through a pinhole camera,
drops the faces that are behind or too near the camera, rounds the corners to sixteenths of a pixel,
and fills each triangle into one ``uint8`` canvas. Filling is integer arithmetic, so the same mesh
and camera give the same bits on every machine (constitution Principle V), and it needs no display,
OpenGL, EGL, or OSMesa.

Public source: the OpenCV drawing functions ``fillConvexPoly`` and ``fillPoly`` and their ``shift``
parameter for fixed-point sub-pixel coordinates
(https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html), as chosen in research R1. The
pinhole projection is ``camera.project_points``.

Conventions:

* Projected points are (column, row) pairs, and pixel centers lie at integer coordinates, as in
  camera.py and in the OpenCV drawing calls. A mask is indexed ``mask[row, column]``.
* One ``cv2.fillPoly`` call over all triangles fills with the even-odd rule, so triangles that
  overlap in the image cancel each other (research R1, amendment). Each triangle therefore goes
  through its own ``cv2.fillConvexPoly`` call onto the same canvas, which keeps every overlap
  filled: a true union.
* OpenCV draws a triangle together with its outline and fills every scanline from its rounded start
  to its rounded end, both included. A drawn shape is therefore about one pixel wider and taller
  than its exact outline (a square of side s pixels fills about (s + 1) squared pixels), and
  triangles that share an edge leave no gap along it.
* A face with a corner nearer than ``NEAR_PLANE_M`` is dropped, not clipped, because
  ``camera.project_points`` applies no near-plane clamp and a corner at a tiny positive depth
  would give a huge pixel value. For a body 2.5 m or more from the camera no face is cut.
"""

import math
import operator
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.camera import project_points

__all__ = [
    "NEAR_PLANE_M",
    "SUBPIXEL_SHIFT",
    "Silhouette",
    "fill_union",
    "project_faces",
    "render_silhouette",
    "silhouette_from_mask",
]

# Number of fractional bits of the corner coordinates passed to OpenCV (research R1): a corner is
# stored in units of 1/16 pixel.
SUBPIXEL_SHIFT = 4

# A face with a corner nearer to the camera plane than this (metres) is dropped. The value is far
# below the 2.5 m minimum camera distance of the rigs and far above numerical noise at depth 0.
NEAR_PLANE_M = 0.05

# Corner units per pixel.
_UNITS_PER_PIXEL = float(1 << SUBPIXEL_SHIFT)

# Corners are clamped to this many pixels from the origin, so the int32 conversion cannot overflow
# and a pathological mesh cannot make OpenCV walk through millions of scanlines. A mesh in front of
# the near plane never comes close: a corner 10 m off-axis at the near plane sits near 26,000 px.
_MAX_ABS_PIXEL = 1 << 20


@dataclass(frozen=True, eq=False)
class Silhouette:
    """The silhouette of one body seen by one camera (data-model.md, Silhouette).

    ``mask`` is a ``uint8`` array of shape (image_size, image_size) with values 0 and 1, indexed by
    (row, column). ``area_fraction`` is the share of pixels that are set, and it is 0.0 for an empty
    mask: the caller flags ``empty_mask`` when it is not above 0. ``touches_border`` is True when
    any pixel of the first or last row or column is set, which means the body is partly outside
    the frame: the caller flags ``out_of_frame``.
    """

    mask: NDArray[np.uint8]
    area_fraction: float
    touches_border: bool


def silhouette_from_mask(mask: ArrayLike) -> Silhouette:
    """Return the Silhouette of a binary mask: the mask with its area fraction and border flag.

    ``mask`` is a two-dimensional array of 0 and 1 (or False and True) with at least one pixel. A
    ``uint8`` array is used as it is, without a copy. The same two definitions serve synthetic and
    real masks. Raises ValueError for any other shape or a value other than 0 and 1.
    """
    grid = np.asarray(mask)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"a mask must be a non-empty 2D array, not shape {grid.shape}")
    if not np.all((grid == 0) | (grid == 1)):
        raise ValueError("a mask must hold only the values 0 and 1")
    binary = np.asarray(grid, dtype=np.uint8)
    touches = bool(binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any())
    return Silhouette(
        mask=binary,
        area_fraction=float(np.count_nonzero(binary)) / binary.size,
        touches_border=touches,
    )


def project_faces(
    vertices: ArrayLike,
    faces: ArrayLike,
    K: ArrayLike,
    R: ArrayLike,
    t: ArrayLike,
    near_m: float = NEAR_PLANE_M,
) -> NDArray[np.int32]:
    """Return the fixed-point image corners of the faces that lie in front of the camera.

    ``vertices`` has shape (V, 3) in metres and ``faces`` has shape (F, 3) with vertex indices.
    ``K``, ``R``, and ``t`` are one pinhole camera as in ``camera.project_points``, with ``R`` and
    ``t`` the world-to-camera pose. Every vertex is projected once, and a face is kept only when
    all three of its vertices have a depth of at least ``near_m`` metres; a face with a nearer
    corner, or a corner behind the camera, is dropped whole.

    The result has shape (n_kept, 3, 2) and dtype int32, in the order of ``faces``: for each kept
    face, its three corners as (column, row) in units of 1/16 pixel (``SUBPIXEL_SHIFT``), rounded
    to the nearest unit (ties to even) and clamped to within 2**20 pixels of the origin. Raises
    ValueError for a malformed mesh, a non-finite vertex, or a ``near_m`` that is not above 0.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), not shape {points.shape}")
    triangles = _as_faces(faces, points.shape[0])
    if not (math.isfinite(near_m) and near_m > 0.0):
        raise ValueError(f"near_m must be a positive number of metres, not {near_m}")

    pixels, depth = project_points(points, K, R, t)
    face_in_front = (depth >= near_m)[triangles].all(axis=1)
    corners = pixels[triangles[face_in_front]]
    fixed = np.rint(corners * _UNITS_PER_PIXEL)
    limit = _MAX_ABS_PIXEL * _UNITS_PER_PIXEL
    np.clip(fixed, -limit, limit, out=fixed)
    return fixed.astype(np.int32)


def fill_union(triangles: NDArray[np.int32], image_size: int) -> NDArray[np.uint8]:
    """Fill fixed-point triangles into one square mask and return it, with overlaps filled.

    ``triangles`` has shape (n, 3, 2) and dtype int32, as ``project_faces`` returns it: three
    (column, row) corners per triangle in units of 1/16 pixel. The mask is ``uint8`` of shape
    (image_size, image_size), 1 inside any triangle (outline included) and 0 elsewhere. Parts of
    triangles outside the image are clipped. Raises ValueError for another shape or dtype, which
    would silently misread pixel coordinates as fixed-point units, or an ``image_size`` below 1.
    """
    size = _as_image_size(image_size)
    corners = np.asarray(triangles)
    if corners.dtype != np.int32 or corners.ndim != 3 or corners.shape[1:] != (3, 2):
        raise ValueError(
            "triangles must be an int32 array of shape (n, 3, 2), as project_faces returns it; "
            f"got dtype {corners.dtype} and shape {corners.shape}"
        )
    corners = np.ascontiguousarray(corners)
    canvas = np.zeros((size, size), dtype=np.uint8)
    # The loop is the cost of a render, about 1 microsecond per triangle: keep it free of lookups
    # and keyword arguments.
    fill = cv2.fillConvexPoly
    line_type = cv2.LINE_8
    for triangle in corners:
        fill(canvas, triangle, 1, line_type, SUBPIXEL_SHIFT)
    return canvas


def render_silhouette(
    vertices: ArrayLike,
    faces: ArrayLike,
    K: ArrayLike,
    R: ArrayLike,
    t: ArrayLike,
    image_size: int,
    near_m: float = NEAR_PLANE_M,
) -> Silhouette:
    """Render the silhouette of a mesh seen by one pinhole camera (research R1).

    The arguments are those of ``project_faces`` plus the side of the square mask in pixels
    (``camera.image_size``). Pass the true camera placement, ``R_true`` and ``t_true``, never the
    rotation with placement noise: silhouettes stay rendered from the true placement (FR-005).
    Faces with a corner nearer than ``near_m`` are dropped, so a mesh entirely behind the camera
    gives an empty mask. The cost is about 10 ms for the 8,448 faces of the stand-in body.
    """
    size = _as_image_size(image_size)
    triangles = project_faces(vertices, faces, K, R, t, near_m)
    return silhouette_from_mask(fill_union(triangles, size))


def _as_image_size(image_size: int) -> int:
    """Return image_size as an int of at least 1; raise ValueError or TypeError otherwise."""
    size = operator.index(image_size)
    if size < 1:
        raise ValueError(f"image_size must be at least 1, not {size}")
    return size


def _as_faces(faces: ArrayLike, vertex_count: int) -> NDArray[np.int64]:
    """Return faces as an int64 array of shape (F, 3) whose indices all name an existing vertex."""
    triangles = np.asarray(faces)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), not shape {triangles.shape}")
    if triangles.size and not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError(f"faces must hold integer vertex indices, not dtype {triangles.dtype}")
    triangles = triangles.astype(np.int64, copy=False)
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertex_count):
        raise ValueError(
            f"faces must hold vertex indices from 0 to {vertex_count - 1}, but they range from "
            f"{triangles.min()} to {triangles.max()}"
        )
    return triangles
