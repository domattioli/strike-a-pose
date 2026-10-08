"""Smoke tests for render.py: union fill, near-plane drops, area fraction, and the border flag."""

import itertools
import time
import warnings

import cv2
import numpy as np
import pytest

from strike_a_pose.body.base import canonical_mesh
from strike_a_pose.body.standin import NUM_BETAS, StandInBody
from strike_a_pose.camera import (
    intrinsics,
    look_at_extrinsics,
    noise_rotation,
    orbit_position,
    project_points,
)
from strike_a_pose.render import (
    NEAR_PLANE_M,
    SUBPIXEL_SHIFT,
    Silhouette,
    fill_union,
    project_faces,
    render_silhouette,
    silhouette_from_mask,
)

# The unit cube centered at the origin. Corner i has the coordinates of the binary digits of i, with
# x the highest digit: 0 means -0.5 and 1 means +0.5. Each side is a quadrilateral with its corners
# in order around it, and each quadrilateral is split into two triangles.
CUBE_VERTICES = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))
_CUBE_SIDES = (
    (0, 1, 3, 2),  # x = -0.5
    (4, 5, 7, 6),  # x = +0.5
    (0, 1, 5, 4),  # y = -0.5
    (2, 3, 7, 6),  # y = +0.5
    (0, 2, 6, 4),  # z = -0.5
    (1, 3, 7, 5),  # z = +0.5
)
CUBE_FACES = np.array(
    [triangle for a, b, c, d in _CUBE_SIDES for triangle in ((a, b, c), (a, c, d))]
)

# The area tests use a large canvas. OpenCV fills a shape together with its outline, so a drawn
# shape is about one pixel wider and taller than its exact outline, and that adds roughly 2 / s to
# the area of a square of side s pixels. The side here is 204.8 px, so the excess stays near 1%. A
# cube with a side of 110 px on a 128 px canvas measured 2% to 3% too large, which is why the area
# tests do not use the 64 px and 128 px of the experiment configurations.
CUBE_IMAGE_SIZE = 256
CUBE_DISTANCE_M = 1.75  # the front face is then 1.25 m from the camera: side = 256 / 1.25 px

# Offsets (x, y) in metres of the cube from the camera axis. They vary where the edges fall between
# pixels, and they stay under 0.12 m so the front face keeps inside the 256 px frame.
CUBE_OFFSETS = [(0.0, 0.0), (0.037, -0.021), (0.1, 0.1), (-0.08, 0.06)]

# A camera at the world origin that looks along +z: camera coordinates equal world coordinates, a
# point has depth z, and its pixel is K (x, y) / z.
IDENTITY_ROTATION = np.eye(3)
ZERO_TRANSLATION = np.zeros(3)

# The unit cube 3 m in front of that camera.
CUBE_IN_FRONT = CUBE_VERTICES + np.array([0.0, 0.0, 3.0])

STANDIN_BETAS = np.zeros(NUM_BETAS)


# Helpers


def _on_axis_camera(distance_m):
    """Return (R, t) of a camera on the +z axis at distance_m that looks at the origin."""
    return look_at_extrinsics([0.0, 0.0, distance_m], [0.0, 0.0, 0.0])


def _render_cube(image_size, distance_m, offset=(0.0, 0.0), scale=1.0):
    """Render the unit cube (scaled by scale and moved by offset in x and y) from the +z axis."""
    vertices = CUBE_VERTICES * scale + np.array([offset[0], offset[1], 0.0])
    K = intrinsics(image_size, float(image_size))
    R, t = _on_axis_camera(distance_m)
    return render_silhouette(vertices, CUBE_FACES, K, R, t, image_size)


def _render_in_camera_frame(vertices, faces, size=64, **keywords):
    """Render a mesh in camera coordinates (R = I, t = 0); keywords reach render_silhouette."""
    K = intrinsics(size, float(size))
    return render_silhouette(
        vertices, faces, K, IDENTITY_ROTATION, ZERO_TRANSLATION, size, **keywords
    )


def _project_in_camera_frame(vertices, faces, size=64, **keywords):
    """Return project_faces for a mesh given in camera coordinates (R = I, t = 0)."""
    K = intrinsics(size, float(size))
    return project_faces(vertices, faces, K, IDENTITY_ROTATION, ZERO_TRANSLATION, **keywords)


def _filled_pixel(coordinate):
    """Return the pixel that OpenCV's fill reaches for an edge at coordinate (in pixels).

    The coordinate is rounded to the 1/16 pixel grid (shift 4), then to the nearest pixel with
    halves going up. A fill covers every pixel from the rounded start to the rounded end, both
    included.
    """
    sixteenths = np.rint(coordinate * 16.0)
    return int(np.floor(sixteenths / 16.0 + 0.5))


def _rectangle_mask(size, left, top, right, bottom):
    """Return the mask of the pixels between the rounded left and right and top and bottom edges."""
    rows = slice(_filled_pixel(top), _filled_pixel(bottom) + 1)
    columns = slice(_filled_pixel(left), _filled_pixel(right) + 1)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[rows, columns] = 1
    return mask


def _render_pixel_mesh(pixels, faces, size, depth=3.0):
    """Render a mesh that is given by its target (column, row) pixels.

    The vertices are put at the given depth in front of a camera with R = I and t = 0, where
    x = (column - c) * depth / f and y = (row - c) * depth / f project back to those pixels.
    """
    K = intrinsics(size, float(size))
    pixels = np.asarray(pixels, dtype=np.float64)
    x = (pixels[:, 0] - K[0, 2]) * depth / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * depth / K[1, 1]
    vertices = np.stack([x, y, np.full(x.shape, float(depth))], axis=-1)
    return render_silhouette(vertices, faces, K, IDENTITY_ROTATION, ZERO_TRANSLATION, size)


def _convex_hull(points):
    """Return the convex hull of 2D points in an order whose shoelace sum is positive."""
    points = np.asarray(points, dtype=np.float64)
    indices = cv2.convexHull(points.astype(np.float32), returnPoints=False).ravel()
    hull = points[indices]
    return hull if _signed_area(hull) > 0.0 else hull[::-1]


def _signed_area(polygon):
    """Return the shoelace area of a polygon: positive for the order that _convex_hull returns."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _distance_inside(hull, size):
    """Return, per pixel center, the distance in pixels to the hull boundary: negative outside.

    This is the smallest signed distance to the lines through the edges of a convex polygon, which
    does not touch OpenCV. The array has shape (size, size), indexed by (row, column).
    """
    columns, rows = np.meshgrid(np.arange(size), np.arange(size))
    distances = []
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (rows - start[1]) - edge[1] * (columns - start[0])
        distances.append(cross / np.hypot(edge[0], edge[1]))
    return np.min(distances, axis=0)


def _jittered_grid(rng, cells, low, high):
    """Return pixels and faces of the square low..high split into cells by cells quadrilaterals.

    Each quadrilateral becomes two triangles that share an edge with the next, so the mesh has seams
    everywhere. The corners on the boundary stay on the square, and the interior corners move by up
    to 2 px, which keeps every cell a proper quadrilateral.
    """
    axis = np.linspace(low, high, cells + 1)
    columns, rows = np.meshgrid(axis, axis)
    grid = np.stack([columns, rows], axis=-1)
    grid[1:-1, 1:-1] += rng.uniform(-2.0, 2.0, size=grid[1:-1, 1:-1].shape)
    index = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    faces = []
    for row, column in itertools.product(range(cells), repeat=2):
        a, b = index[row, column], index[row, column + 1]
        c, d = index[row + 1, column + 1], index[row + 1, column]
        faces += [(a, b, c), (a, c, d)] if (row + column) % 2 else [(a, b, d), (b, c, d)]
    return grid.reshape(-1, 2), np.array(faces)


def _storage_variants(vertices, faces, K, R, t):
    """Yield (name, arguments): the same numbers in other array layouts and types."""
    yield "fortran order", (np.asfortranarray(vertices), faces, K, R, t)
    yield "nested lists", (vertices.tolist(), faces.tolist(), K.tolist(), R.tolist(), t.tolist())
    yield "int32 faces", (vertices, faces.astype(np.int32), K, R, t)
    yield "fortran camera", (vertices, faces, np.asfortranarray(K), np.asfortranarray(R), t)
    yield "copies", (vertices.copy(), faces.copy(), K.copy(), R.copy(), t.copy())


@pytest.fixture(scope="module")
def standin_mesh():
    """The canonical-pose vertices of the stand-in body, its faces, and its bounding-box center."""
    body = StandInBody()
    vertices, _ = canonical_mesh(body, STANDIN_BETAS)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return vertices, np.asarray(body.faces), center


def _standin_silhouette(standin_mesh, size, azimuth_deg, distance_m=3.0, height_m=1.2):
    """Render the stand-in body from a camera on a circle around it, aimed at its center.

    Returns the silhouette and the camera (K, R, t).
    """
    vertices, faces, center = standin_mesh
    K = intrinsics(size, float(size))
    position = orbit_position(azimuth_deg, distance_m, height_m, center)
    R, t = look_at_extrinsics(position, center)
    return render_silhouette(vertices, faces, K, R, t, size), (K, R, t)


# Constants


def test_the_fixed_point_shift_is_4_and_the_near_distance_is_5_cm():
    assert SUBPIXEL_SHIFT == 4  # research R1: coordinates in sixteenths of a pixel
    assert NEAR_PLANE_M == 0.05


# The unit cube


@pytest.mark.parametrize("offset", CUBE_OFFSETS)
def test_a_unit_cube_renders_the_expected_square_area_within_two_percent(offset):
    # Seen along its axis, the cube's silhouette is its front face: a square of side f / (d - 0.5)
    # pixels, with f the focal length (equal to the image size here). Its 12 triangles overlap in
    # the image (the back face lies inside the front face), so an even-odd fill would cancel them.
    silhouette = _render_cube(CUBE_IMAGE_SIZE, CUBE_DISTANCE_M, offset)
    side_px = CUBE_IMAGE_SIZE / (CUBE_DISTANCE_M - 0.5)
    expected_area = side_px**2
    area = int(silhouette.mask.sum())
    assert abs(area - expected_area) / expected_area < 0.02
    assert not silhouette.touches_border


@pytest.mark.parametrize("offset", CUBE_OFFSETS)
def test_the_cube_mask_is_exactly_the_rounded_front_face_with_no_hole_and_no_overhang(offset):
    # The pinhole formulas by hand: u = c + f x / z and v = c - f y / z for a camera on the +z axis
    # (image y points down, world y up), with the front face at depth d - 0.5.
    silhouette = _render_cube(CUBE_IMAGE_SIZE, CUBE_DISTANCE_M, offset)
    center = (CUBE_IMAGE_SIZE - 1) / 2
    scale = CUBE_IMAGE_SIZE / (CUBE_DISTANCE_M - 0.5)
    expected = _rectangle_mask(
        CUBE_IMAGE_SIZE,
        left=center + scale * (offset[0] - 0.5),
        top=center - scale * (offset[1] + 0.5),
        right=center + scale * (offset[0] + 0.5),
        bottom=center - scale * (offset[1] - 0.5),
    )
    assert np.array_equal(silhouette.mask, expected)


@pytest.mark.parametrize(
    ("axis", "angle_deg"),
    [([1.0, 1.0, 0.0], 35.0), ([1.0, 2.0, 3.0], 50.0), ([0.0, 1.0, 0.0], 30.0)],
)
def test_a_tilted_cube_fills_its_exact_hexagon(axis, angle_deg):
    # A cube seen from a generic direction projects to a hexagon, the convex hull of its eight
    # corners. Front and back faces overlap in the image, so the union has to be filled.
    size = CUBE_IMAGE_SIZE
    K = intrinsics(size, float(size))
    R, t = _on_axis_camera(2.2)
    vertices = CUBE_VERTICES @ noise_rotation(axis, angle_deg).T
    silhouette = render_silhouette(vertices, CUBE_FACES, K, R, t, size)

    corners, _ = project_points(vertices, K, R, t)
    hull = _convex_hull(corners)
    area = int(silhouette.mask.sum())
    assert abs(area - _signed_area(hull)) / _signed_area(hull) < 0.02

    inside = _distance_inside(hull, size)
    mask = silhouette.mask.astype(bool)
    assert mask[inside >= 0.25].all()  # every pixel center that is clearly inside is set
    assert not mask[inside < -1.0].any()  # nothing is set more than a pixel outside


def test_the_area_fraction_is_the_share_of_set_pixels():
    silhouette = _render_cube(CUBE_IMAGE_SIZE, CUBE_DISTANCE_M, (0.037, -0.021))
    count = int(silhouette.mask.sum())
    assert silhouette.area_fraction == count / CUBE_IMAGE_SIZE**2
    assert 0.0 < silhouette.area_fraction < 1.0


# The union fill


def test_overlapping_triangles_are_filled_as_a_union_and_do_not_cancel():
    # Two squares overlap in pixels 15 to 30. A single cv2.fillPoly call over all four triangles
    # would use the even-odd rule and cancel that block (research R1, amendment).
    pixels = [[5, 5], [30, 5], [30, 30], [5, 30], [15, 15], [45, 15], [45, 45], [15, 45]]
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
    silhouette = _render_pixel_mesh(pixels, faces, 64)
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[5:31, 5:31] = 1
    expected[15:46, 15:46] = 1
    assert silhouette.mask[20, 20] == 1
    assert np.array_equal(silhouette.mask, expected)


@pytest.mark.parametrize("seed", range(6))
def test_triangles_that_share_edges_leave_no_seam(seed):
    pixels, faces = _jittered_grid(np.random.default_rng(seed), cells=6, low=10.0, high=50.0)
    silhouette = _render_pixel_mesh(pixels, faces, 64)
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[10:51, 10:51] = 1
    assert np.array_equal(silhouette.mask, expected)


@pytest.mark.parametrize(
    ("left", "top", "right", "bottom"),
    [
        (10.0, 12.0, 20.0, 30.0),  # whole pixels: both ends are filled, 11 by 19 pixels
        (10.4, 12.3, 20.6, 30.7),
        (10.49, 12.49, 20.49, 30.49),  # just under a half pixel: the 1/16 grid rounds it up
        (10.2, 12.2, 10.3, 12.3),  # smaller than a pixel: one pixel
    ],
)
def test_a_rectangle_fills_the_pixels_between_its_rounded_corners(left, top, right, bottom):
    pixels = [[left, top], [right, top], [right, bottom], [left, bottom]]
    silhouette = _render_pixel_mesh(pixels, [(0, 1, 2), (0, 2, 3)], 64)
    assert np.array_equal(silhouette.mask, _rectangle_mask(64, left, top, right, bottom))


def test_the_fill_clips_at_the_image_edge():
    pixels = [[-20, 10], [20, 10], [20, 30], [-20, 30]]
    silhouette = _render_pixel_mesh(pixels, [(0, 1, 2), (0, 2, 3)], 64)
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[10:31, 0:21] = 1
    assert np.array_equal(silhouette.mask, expected)
    assert silhouette.touches_border


def test_the_mask_has_the_documented_type_shape_and_values():
    silhouette = _render_cube(64, 3.0, scale=0.5)
    assert isinstance(silhouette, Silhouette)
    assert silhouette.mask.dtype == np.uint8
    assert silhouette.mask.shape == (64, 64)
    assert set(np.unique(silhouette.mask)) == {0, 1}


# Faces behind the camera and near the camera


def test_a_cube_behind_the_camera_renders_an_empty_mask():
    # The camera sits at z = 3 and looks away from the cube, along +z.
    R, t = look_at_extrinsics([0.0, 0.0, 3.0], [0.0, 0.0, 6.0])
    K = intrinsics(64, 64.0)
    assert project_faces(CUBE_VERTICES, CUBE_FACES, K, R, t).shape == (0, 3, 2)
    silhouette = render_silhouette(CUBE_VERTICES, CUBE_FACES, K, R, t, 64)
    assert silhouette.mask.shape == (64, 64)
    assert int(silhouette.mask.sum()) == 0
    assert silhouette.area_fraction == 0.0
    assert silhouette.touches_border is False


def test_a_cube_at_negative_depth_renders_an_empty_mask():
    vertices = CUBE_VERTICES + np.array([0.0, 0.0, -5.0])
    silhouette = _render_in_camera_frame(vertices, CUBE_FACES)
    assert int(silhouette.mask.sum()) == 0
    assert silhouette.area_fraction == 0.0
    assert silhouette.touches_border is False


def test_a_cube_cut_by_the_camera_plane_keeps_only_the_faces_wholly_in_front():
    # The camera is at the cube's center. The side faces have corners at depth -0.5 and +0.5, and
    # only the two triangles of the z = +0.5 side have all corners in front.
    assert _project_in_camera_frame(CUBE_VERTICES, CUBE_FACES).shape == (2, 3, 2)
    silhouette = _render_in_camera_frame(CUBE_VERTICES, CUBE_FACES)
    assert silhouette.touches_border  # the near face is 0.5 m away and larger than the image


def test_a_face_with_a_corner_nearer_than_the_near_distance_is_dropped():
    # The fourth corner is at 4.5 cm. It would project far outside the image, and the face is
    # dropped rather than clipped.
    vertices = np.array(
        [[0.0, 0.0, 3.0], [0.5, 0.0, 3.0], [0.0, 0.5, 3.0], [0.2, 0.2, 0.9 * NEAR_PLANE_M]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    assert _project_in_camera_frame(vertices, faces).shape == (1, 3, 2)
    near_only = _render_in_camera_frame(vertices, faces[1:])
    assert int(near_only.mask.sum()) == 0


def test_a_corner_exactly_at_the_near_distance_is_kept_and_the_distance_can_be_changed():
    vertices = np.array([[0.0, 0.0, NEAR_PLANE_M], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]])
    faces = np.array([[0, 1, 2]])
    assert _project_in_camera_frame(vertices, faces).shape[0] == 1
    assert _project_in_camera_frame(vertices, faces, near_m=2.0 * NEAR_PLANE_M).shape[0] == 0
    assert _project_in_camera_frame(vertices, faces, near_m=0.5 * NEAR_PLANE_M).shape[0] == 1


def test_the_near_distance_can_be_changed_through_render_silhouette_too():
    vertices = np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0], [0.0, 0.3, 1.0]])
    faces = np.array([[0, 1, 2]])
    assert int(_render_in_camera_frame(vertices, faces).mask.sum()) > 0
    assert int(_render_in_camera_frame(vertices, faces, near_m=1.5).mask.sum()) == 0


def test_corners_far_from_the_image_are_clamped_without_overflow():
    # A corner at the near distance and 1e9 m off-axis lands 1e12 px away; its fixed-point value
    # would not fit an int32, and numpy's float to int32 cast would warn or give garbage.
    vertices = np.array([[0.0, 0.0, 3.0], [1e9, 0.0, NEAR_PLANE_M], [0.0, 1e9, NEAR_PLANE_M]])
    faces = np.array([[0, 1, 2]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corners = _project_in_camera_frame(vertices, faces)
        silhouette = _render_in_camera_frame(vertices, faces)
    assert corners.dtype == np.int32
    assert np.abs(corners.astype(np.int64)).max() < 2**30
    # The triangle spans the lower right quadrant of the image, so it runs off two edges.
    assert silhouette.touches_border
    assert silhouette.mask[63, 63] == 1
    assert silhouette.mask[0, 0] == 0


# Fixed-point corners


def test_corners_are_in_sixteenths_of_a_pixel_in_the_order_of_the_faces():
    K = intrinsics(64, 64.0)
    R, t = _on_axis_camera(3.0)
    corners = project_faces(CUBE_VERTICES, CUBE_FACES, K, R, t)
    pixels, _ = project_points(CUBE_VERTICES, K, R, t)
    assert corners.dtype == np.int32
    assert corners.shape == (12, 3, 2)
    assert np.array_equal(corners, np.rint(pixels[CUBE_FACES] * 16.0).astype(np.int32))


# Area fraction and the border flag


@pytest.mark.parametrize(
    ("offset", "touches_border", "empty"),
    [
        ((0.0, 0.0), False, False),
        ((0.9, 0.0), False, False),
        ((1.3, 0.0), True, False),  # crosses the right edge
        ((-1.3, 0.0), True, False),  # the left edge
        ((0.0, 1.3), True, False),  # the top edge (world y up)
        ((0.0, -1.3), True, False),  # the bottom edge
        ((5.0, 0.0), False, True),  # wholly outside the frame: nothing is set, so no border pixel
    ],
)
def test_the_border_flag_is_set_when_the_body_reaches_the_image_edge(offset, touches_border, empty):
    # A cube of side 0.5 m at 3 m, seen with a 53 degree field of view: the half-width of the
    # frame at the cube's plane is about 1.48 m.
    silhouette = _render_cube(64, 3.0, offset, scale=0.5)
    assert silhouette.touches_border is touches_border
    assert (silhouette.area_fraction == 0.0) is empty


@pytest.mark.parametrize(
    ("row", "column"), [(0, 3), (7, 3), (3, 0), (3, 7), (0, 0), (7, 7), (0, 7), (7, 0)]
)
def test_one_set_pixel_on_any_edge_of_the_mask_touches_the_border(row, column):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[row, column] = 1
    result = silhouette_from_mask(mask)
    assert result.touches_border is True
    assert result.area_fraction == 1 / 64


def test_a_mask_that_stays_inside_the_edge_does_not_touch_the_border():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 1:7] = 1
    result = silhouette_from_mask(mask)
    assert result.touches_border is False
    assert result.area_fraction == 36 / 64


def test_an_empty_mask_has_area_fraction_zero_and_touches_nothing():
    result = silhouette_from_mask(np.zeros((8, 8), dtype=np.uint8))
    assert result.area_fraction == 0.0
    assert result.touches_border is False


def test_a_boolean_mask_becomes_uint8_and_a_uint8_mask_is_used_as_it_is():
    boolean = np.zeros((4, 4), dtype=bool)
    boolean[1:3, 1:3] = True
    converted = silhouette_from_mask(boolean)
    assert converted.mask.dtype == np.uint8
    assert converted.area_fraction == 0.25
    unsigned = boolean.astype(np.uint8)
    assert silhouette_from_mask(unsigned).mask is unsigned


# Determinism


def test_rendering_the_same_mesh_twice_is_bit_identical(standin_mesh):
    first, _ = _standin_silhouette(standin_mesh, 64, 37.0)
    second, _ = _standin_silhouette(standin_mesh, 64, 37.0)
    assert first.mask.tobytes() == second.mask.tobytes()
    assert first.area_fraction == second.area_fraction
    assert first.touches_border == second.touches_border
    assert int(first.mask.sum()) > 0


def test_the_mask_does_not_depend_on_how_the_input_arrays_are_stored(standin_mesh):
    vertices, faces, center = standin_mesh
    K = intrinsics(64, 64.0)
    R, t = look_at_extrinsics(orbit_position(110.0, 3.0, 1.2, center), center)
    reference = render_silhouette(vertices, faces, K, R, t, 64)
    for name, arguments in _storage_variants(vertices, faces, K, R, t):
        repeat = render_silhouette(*arguments, 64)
        assert repeat.mask.tobytes() == reference.mask.tobytes(), name


def test_a_posed_body_renders_bit_identically_on_repeat():
    body = StandInBody()
    rng = np.random.default_rng(7)
    betas = np.clip(rng.standard_normal(NUM_BETAS), -2.0, 2.0)
    pose = 0.5 * rng.standard_normal(63)
    vertices = body.vertices(betas, np.zeros(3), pose)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    K = intrinsics(128, 128.0)
    R, t = look_at_extrinsics(orbit_position(200.0, 3.2, 1.0, center), center)
    first = render_silhouette(vertices, body.faces, K, R, t, 128)
    second = render_silhouette(vertices, body.faces, K, R, t, 128)
    assert first.mask.tobytes() == second.mask.tobytes()
    assert int(first.mask.sum()) > 0


# The stand-in body


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("size", [32, 64, 128])
def test_the_stand_in_body_renders_an_upright_silhouette_that_stays_in_frame(
    standin_mesh, size, azimuth
):
    silhouette, _ = _standin_silhouette(standin_mesh, size, azimuth)
    rows = np.flatnonzero(silhouette.mask.any(axis=1))
    columns = np.flatnonzero(silhouette.mask.any(axis=0))
    assert 0.02 < silhouette.area_fraction < 0.25
    assert not silhouette.touches_border
    assert rows.size > columns.size  # a standing body is taller than wide


def test_the_stand_in_body_is_wider_seen_from_the_front_than_from_the_side(standin_mesh):
    front, _ = _standin_silhouette(standin_mesh, 128, 0.0)
    side, _ = _standin_silhouette(standin_mesh, 128, 90.0)
    front_width = np.flatnonzero(front.mask.any(axis=0)).size
    side_width = np.flatnonzero(side.mask.any(axis=0)).size
    assert front_width > 2 * side_width
    assert front.area_fraction > side.area_fraction


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 135.0])
def test_every_projected_vertex_of_the_stand_in_body_lies_next_to_a_set_pixel(
    standin_mesh, azimuth
):
    # Each vertex is the corner of a triangle, and the fill draws the corner, so the pixel of every
    # vertex is set; the dilation by one pixel only absorbs the tie at a half pixel. Swapped axes or
    # a wrong scale would put vertices far from the mask.
    vertices = standin_mesh[0]
    silhouette, (K, R, t) = _standin_silhouette(standin_mesh, 128, azimuth)
    pixels, _ = project_points(vertices, K, R, t)
    nearby = cv2.dilate(silhouette.mask, np.ones((3, 3), dtype=np.uint8))
    columns = np.floor(pixels[:, 0] + 0.5).astype(int)
    rows = np.floor(pixels[:, 1] + 0.5).astype(int)
    assert nearby[rows, columns].all()


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 135.0])
def test_the_stand_in_mask_spans_the_extent_of_its_projected_vertices(standin_mesh, azimuth):
    # The topmost, bottommost, leftmost, and rightmost points of a union of triangles are corners,
    # so the mask reaches the extreme vertices to within the pixel of rounding. A mask that is
    # upside down or mirrored would put the head where the feet are.
    silhouette, (K, R, t) = _standin_silhouette(standin_mesh, 128, azimuth)
    pixels, _ = project_points(standin_mesh[0], K, R, t)
    rows = np.flatnonzero(silhouette.mask.any(axis=1))
    columns = np.flatnonzero(silhouette.mask.any(axis=0))
    assert abs(rows[0] - np.floor(pixels[:, 1].min() + 0.5)) <= 1
    assert abs(rows[-1] - np.floor(pixels[:, 1].max() + 0.5)) <= 1
    assert abs(columns[0] - np.floor(pixels[:, 0].min() + 0.5)) <= 1
    assert abs(columns[-1] - np.floor(pixels[:, 0].max() + 0.5)) <= 1


def test_a_render_takes_a_small_fraction_of_a_second(standin_mesh):
    # Research R1 measures about 10 ms per view. The limit here is 25 times that, taken over the
    # best of five renders, so a loaded machine does not fail it but a per-pixel Python loop would.
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        _standin_silhouette(standin_mesh, 128, 20.0)
        durations.append(time.perf_counter() - start)
    assert min(durations) < 0.25


# Refusals


@pytest.mark.parametrize(
    "vertices",
    [np.zeros(3), np.zeros((4, 2)), np.zeros((2, 4, 3)), np.array([[0.0, 0.0, np.nan]] * 4)],
)
def test_vertices_that_are_not_an_array_of_finite_points_are_refused(vertices):
    with pytest.raises(ValueError):
        _project_in_camera_frame(vertices, [[0, 1, 2]])


@pytest.mark.parametrize(
    "faces",
    [
        np.array([0, 1, 2]),  # one-dimensional
        np.array([[0, 1]]),  # two corners
        np.array([[0.0, 1.0, 2.0]]),  # float indices
        np.array([[0, 1, 4]]),  # an index equal to the vertex count
        np.array([[-1, 1, 2]]),  # a negative index would wrap around silently
    ],
)
def test_faces_that_do_not_name_vertices_of_the_mesh_are_refused(faces):
    vertices = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [1.0, 1.0, 3.0]])
    with pytest.raises(ValueError):
        _project_in_camera_frame(vertices, faces)


@pytest.mark.parametrize("near_m", [0.0, -0.1, float("nan"), float("inf")])
def test_a_near_distance_that_is_not_a_positive_number_is_refused(near_m):
    with pytest.raises(ValueError):
        _project_in_camera_frame(CUBE_VERTICES, CUBE_FACES, near_m=near_m)


@pytest.mark.parametrize("image_size", [0, -4])
def test_an_image_size_below_one_is_refused(image_size):
    # The camera is valid, so the refusal comes from the renderer and not from intrinsics().
    K = intrinsics(64, 64.0)
    arguments = (CUBE_IN_FRONT, CUBE_FACES, K, IDENTITY_ROTATION, ZERO_TRANSLATION)
    with pytest.raises(ValueError):
        render_silhouette(*arguments, image_size)
    with pytest.raises(ValueError):
        fill_union(np.zeros((0, 3, 2), dtype=np.int32), image_size)


def test_an_image_size_that_is_not_an_integer_is_refused():
    K = intrinsics(64, 64.0)
    arguments = (CUBE_IN_FRONT, CUBE_FACES, K, IDENTITY_ROTATION, ZERO_TRANSLATION)
    with pytest.raises(TypeError):
        render_silhouette(*arguments, 64.0)
    with pytest.raises(TypeError):
        fill_union(np.zeros((0, 3, 2), dtype=np.int32), 64.0)


@pytest.mark.parametrize(
    "triangles",
    [
        np.zeros((2, 3, 2), dtype=np.float64),  # pixels, not fixed-point units
        np.zeros((2, 3, 2), dtype=np.int64),
        np.zeros((2, 3), dtype=np.int32),
        np.zeros((2, 4, 2), dtype=np.int32),
    ],
)
def test_fill_union_refuses_corners_that_are_not_int32_triangles(triangles):
    with pytest.raises(ValueError):
        fill_union(triangles, 8)


@pytest.mark.parametrize(
    "mask",
    [np.zeros(4), np.zeros((2, 2, 2)), np.zeros((0, 4)), np.full((4, 4), 2), np.full((4, 4), 0.5)],
)
def test_silhouette_from_mask_refuses_anything_but_a_binary_image(mask):
    with pytest.raises(ValueError):
        silhouette_from_mask(mask)


def test_a_mesh_with_no_faces_renders_an_empty_mask():
    silhouette = _render_in_camera_frame(CUBE_VERTICES, np.zeros((0, 3), dtype=np.int64), size=32)
    assert silhouette.mask.shape == (32, 32)
    assert int(silhouette.mask.sum()) == 0
    assert silhouette.area_fraction == 0.0
