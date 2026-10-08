"""Tests for measure.py: the five definitions of research R6 on capsules and the mannequin.

Expected values come from geometry that is known exactly (a capsule of radius r has circumference
2 pi r, a prism has the perimeter of its cross-section, the mannequin has its height parameter) and
from a slow implementation of the same definitions that lives in this file. The slow version cuts
one triangle at a time and builds an exact convex hull with Andrew's monotone chain (A. M. Andrew,
"Another efficient algorithm for convex hulls in two dimensions", Information Processing Letters
9(5), 1979), so it shares no code with the batched Cauchy-formula version under test.
"""

import math

import numpy as np
import pytest
import torch

from strike_a_pose import measure as measure_module
from strike_a_pose.body.base import JOINT_INDEX, NUM_JOINTS, canonical_mesh
from strike_a_pose.body.standin import StandInBody, shape_parameters
from strike_a_pose.measure import (
    DEFAULT_MEMORY_BUDGET_MB,
    MEASUREMENT_NAMES,
    NUM_DIRECTIONS,
    NUM_MEASUREMENTS,
    measure_batch,
    measure_mesh,
    slice_nan_flags,
)

HEIGHT, CHEST, WAIST, HIP, THIGH = range(5)

# Research R6 requires the perimeter error to stay under 0.1 percent. The rectangle rule over 64
# directions is off by at most (pi / 64)^2 / 12 of the perimeter, so a tighter bound also holds.
RESEARCH_PERIMETER_TOLERANCE = 1e-3
CAUCHY_ERROR_BOUND = (math.pi / NUM_DIRECTIONS) ** 2 / 12.0
CAUCHY_TOLERANCE = 1.05 * CAUCHY_ERROR_BOUND

TORSO_PARTS = ("pelvis", "spine1", "spine2", "spine3")
HIP_PARTS = TORSO_PARTS + ("left_hip", "right_hip")


# Slow independent implementation of research R6.


def hull_perimeter(points: np.ndarray) -> float:
    """Return the perimeter of the convex hull of 2D points, by Andrew's monotone chain."""
    unique = sorted({(float(x), float(z)) for x, z in points})

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    def half_hull(sequence):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0.0:
                chain.pop()
            chain.append(point)
        return chain

    hull = half_hull(unique)[:-1] + half_hull(reversed(unique))[:-1]
    if len(hull) < 2:
        return 0.0
    closed = [*hull, hull[0]]
    return sum(math.dist(first, second) for first, second in zip(closed, closed[1:], strict=False))


def slice_perimeter(vertices, faces, in_part, height) -> float:
    """Return P(h, S) of research R6 by cutting one triangle at a time; NaN below 3 cut points.

    A vertex is above the plane when y > h. A triangle with vertices on both sides has one lone
    vertex on its own side; the plane cuts the two edges that leave the lone vertex.
    """
    triangles = faces[in_part[faces].any(axis=1)]
    corners = vertices[triangles]
    above = corners[:, :, 1] > height
    mixed = above.any(axis=1) & ~above.all(axis=1)
    corners, above = corners[mixed], above[mixed]
    lone = np.where(above.sum(axis=1) == 1, above.argmax(axis=1), (~above).argmax(axis=1))
    rows = np.arange(len(lone))
    points = []
    for offset in (1, 2):
        start = corners[rows, lone]
        stop = corners[rows, (lone + offset) % 3]
        fraction = (height - start[:, 1]) / (stop[:, 1] - start[:, 1])
        points.append(start[:, [0, 2]] + fraction[:, None] * (stop[:, [0, 2]] - start[:, [0, 2]]))
    points = np.concatenate(points)
    return math.nan if len(points) < 3 else hull_perimeter(points)


def slow_measurements(vertices, joints, faces, part_ids, step_cm) -> np.ndarray:
    """Return height, chest, waist, hip, and thigh in cm for one mesh, by the slow method."""
    step = step_cm / 100.0

    def part_mask(names):
        return np.isin(part_ids, [JOINT_INDEX[name] for name in names])

    def joint_height(name):
        return joints[JOINT_INDEX[name], 1]

    def search(names, low, high, pick):
        in_part = part_mask(names)
        count = int(math.floor((high - low + 1e-9) / step)) + 1
        perimeters = [
            slice_perimeter(vertices, faces, in_part, low + index * step) for index in range(count)
        ]
        # One degenerate slice makes the whole search NaN; max and min alone would skip a NaN.
        if count <= 0 or any(math.isnan(value) for value in perimeters):
            return math.nan
        return pick(perimeters)

    height = vertices[:, 1].max() - vertices[:, 1].min()
    hip_joint = joint_height("left_hip")
    thigh_level = hip_joint - 0.30 * (hip_joint - joint_height("left_knee"))
    values = [
        height,
        search(TORSO_PARTS, joint_height("spine2"), joint_height("spine3"), max),
        search(TORSO_PARTS, joint_height("pelvis"), joint_height("spine2"), min),
        search(HIP_PARTS, joint_height("pelvis") - 0.10 * height, joint_height("pelvis"), max),
        search(("left_hip",), thigh_level, thigh_level, max),
    ]
    return 100.0 * np.array(values)


# Meshes with known cross-sections.


def regular_polygon(sides: int, radius: float, turn: float = 0.0) -> np.ndarray:
    """Return the (x, z) corners of a regular polygon with the given circumradius."""
    angles = turn + 2.0 * math.pi * np.arange(sides) / sides
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def polygon_perimeter(polygon: np.ndarray) -> float:
    """Return the length of the closed outline of a polygon."""
    return float(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).sum())


def ring_strip_faces(sides: int, first_ring: int, ring_count: int) -> list[tuple[int, int, int]]:
    """Return the triangles that join consecutive rings of `sides` vertices each."""
    faces = []
    for ring in range(first_ring, first_ring + ring_count - 1):
        for side in range(sides):
            after = (side + 1) % sides
            a, b = ring * sides + side, ring * sides + after
            c, d = (ring + 1) * sides + after, (ring + 1) * sides + side
            faces.extend([(a, b, c), (a, c, d)])
    return faces


def prism(polygon: np.ndarray, ring_heights) -> tuple[np.ndarray, np.ndarray]:
    """Return an open prism with the polygon as cross-section: rings at the given heights."""
    sides = len(polygon)
    vertices = np.array([(x, y, z) for y in ring_heights for x, z in polygon])
    faces = np.array(ring_strip_faces(sides, 0, len(ring_heights)), dtype=np.int64)
    return vertices, faces


def capsule(radius: float, straight_length: float = 1.0, sides: int = 48):
    """Return a closed capsule along y: a regular prism from y = 0 to y = straight_length, capped.

    Both caps are spherical. Every cross-section of the straight part is a regular `sides`-gon
    with the given circumradius.
    """
    cap_rings, body_rings = 6, 3
    profile = []  # (ring radius, ring height) from the bottom up
    for index in range(1, cap_rings + 1):
        angle = 0.5 * math.pi * index / cap_rings
        height = -radius * math.cos(angle) if index < cap_rings else 0.0
        profile.append((radius * math.sin(angle), height))
    for index in range(1, body_rings + 1):
        profile.append((radius, straight_length * index / (body_rings + 1)))
    profile.append((radius, straight_length))
    for index in range(cap_rings - 1, 0, -1):
        angle = 0.5 * math.pi * index / cap_rings
        profile.append((radius * math.sin(angle), straight_length + radius * math.cos(angle)))

    angles = 2.0 * math.pi * np.arange(sides) / sides
    vertices = [(0.0, -radius, 0.0)]
    for ring_radius, height in profile:
        vertices.extend(
            (ring_radius * math.cos(angle), height, ring_radius * math.sin(angle))
            for angle in angles
        )
    vertices.append((0.0, straight_length + radius, 0.0))
    top_pole = len(vertices) - 1

    faces = [(0, 1 + (side + 1) % sides, 1 + side) for side in range(sides)]
    faces += [(a + 1, b + 1, c + 1) for a, b, c in ring_strip_faces(sides, 0, len(profile))]
    last_ring = 1 + (len(profile) - 1) * sides
    faces += [(top_pole, last_ring + side, last_ring + (side + 1) % sides) for side in range(sides)]
    return np.array(vertices), np.array(faces, dtype=np.int64)


def capsule_batch(radii, **options):
    """Return the vertices (B, V, 3) of capsules of the given radii, and their shared faces."""
    meshes = [capsule(float(radius), **options) for radius in radii]
    return np.stack([vertices for vertices, _ in meshes]), meshes[0][1]


def cone_mesh(radius_at, sides: int = 24, ring_heights=(0.0, 0.5, 1.0)):
    """Return an open prism whose regular cross-section has the circumradius radius_at(height).

    The side edges are straight between rings, so a slice at height h has circumradius
    radius_at(h) as long as radius_at is linear between the ring heights.
    """
    rings = [regular_polygon(sides, radius_at(y)) for y in ring_heights]
    vertices = np.array(
        [(x, y, z) for y, ring in zip(ring_heights, rings, strict=True) for x, z in ring]
    )
    return vertices, np.array(ring_strip_faces(sides, 0, len(ring_heights)), dtype=np.int64)


def joints_with(batch: int, **heights) -> np.ndarray:
    """Return (batch, 22, 3) joints at the origin, with the named joints at the given heights."""
    joints = np.zeros((batch, NUM_JOINTS, 3))
    for name, value in heights.items():
        joints[:, JOINT_INDEX[name], 1] = value
    return joints


def polygon_sides_perimeter(sides: int, radius: float) -> float:
    """Return the perimeter of a regular polygon with `sides` corners on a circle of the radius."""
    return 2.0 * sides * radius * math.sin(math.pi / sides)


# Fixtures.


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Run this module on one thread: its meshes are small, and a busy machine stalls a team."""
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(saved)


@pytest.fixture(scope="module")
def mannequin() -> StandInBody:
    return StandInBody()


@pytest.fixture(scope="module")
def meshes(mannequin):
    """Eight canonical mannequin meshes: the zero shape and seven random ones (read only)."""
    rng = np.random.default_rng(20261007)
    betas = np.vstack([np.zeros(10), np.clip(rng.standard_normal((7, 10)), -3.0, 3.0)])
    vertices, joints = canonical_mesh(mannequin, betas)
    for array in (betas, vertices, joints):
        array.setflags(write=False)
    return betas, vertices, joints


# Definitions on exact geometry.


def test_the_five_measurements_come_in_the_documented_order():
    assert MEASUREMENT_NAMES == ("height", "chest", "waist", "hip", "thigh")
    assert NUM_MEASUREMENTS == 5
    assert (HEIGHT, CHEST, WAIST, HIP, THIGH) == (0, 1, 2, 3, 4)
    assert NUM_DIRECTIONS == 64
    assert NUM_DIRECTIONS & (NUM_DIRECTIONS - 1) == 0  # the fixed-order sum halves repeatedly
    assert DEFAULT_MEMORY_BUDGET_MB > 0


def test_a_batch_of_capsules_of_radius_r_has_a_thigh_circumference_of_two_pi_r():
    radii = np.array([0.02, 0.045, 0.0646, 0.085, 0.105, 0.15])
    vertices, faces = capsule_batch(radii)
    part_ids = np.full(vertices.shape[1], JOINT_INDEX["left_hip"])
    # The slice lies 0.3 of the hip-to-knee distance below the hip joint, inside the straight part.
    joints = joints_with(len(radii), left_hip=1.0, left_knee=0.0)

    result = measure_batch(vertices, faces, part_ids, joints, 0.5).numpy()

    assert result[:, THIGH] == pytest.approx(200.0 * math.pi * radii, rel=0.01)
    expected = 100.0 * np.array([polygon_sides_perimeter(48, radius) for radius in radii])
    assert result[:, THIGH] == pytest.approx(expected, rel=CAUCHY_TOLERANCE)
    assert result[:, HEIGHT] == pytest.approx(100.0 * (1.0 + 2.0 * radii), rel=1e-12)


def test_a_batch_of_torso_capsules_has_chest_waist_and_hip_of_two_pi_r():
    radii = np.array([0.03, 0.05, 0.1, 0.14, 0.2])
    vertices, faces = capsule_batch(radii)
    part_ids = np.full(vertices.shape[1], JOINT_INDEX["spine2"])
    # All three search ranges lie inside the straight part. The pelvis joint sits at a vertex ring.
    joints = joints_with(
        len(radii), pelvis=0.5, spine2=0.65, spine3=0.85, left_hip=0.4, left_knee=0.1
    )

    result = measure_batch(vertices, faces, part_ids, joints, 0.5).numpy()

    expected = 100.0 * np.array([polygon_sides_perimeter(48, radius) for radius in radii])
    for column in (CHEST, WAIST, HIP):
        assert result[:, column] == pytest.approx(200.0 * math.pi * radii, rel=0.01)
        assert result[:, column] == pytest.approx(expected, rel=CAUCHY_TOLERANCE)
    assert np.isnan(result[:, THIGH]).all()  # no vertex belongs to the left hip part
    assert result[:, HEIGHT] == pytest.approx(100.0 * (1.0 + 2.0 * radii), rel=1e-12)


def test_the_height_equals_the_mannequin_height(mannequin, meshes):
    betas, vertices, joints = meshes
    # Shapes that change only the height coefficient, so a height error cannot hide.
    height_betas = np.zeros((5, 10))
    height_betas[:, 0] = [-3.0, -1.0, 0.0, 1.0, 3.0]
    height_vertices, height_joints = canonical_mesh(mannequin, height_betas)

    for shape, mesh, mesh_joints in (
        (betas, vertices, joints),
        (height_betas, height_vertices, height_joints),
    ):
        result = measure_batch(mesh, mannequin.faces, mannequin.part_ids, mesh_joints, 0.5)
        assert result[:, HEIGHT].numpy() == pytest.approx(
            100.0 * shape_parameters(shape).height_m, rel=1e-12
        )
    assert result[2, HEIGHT].item() == pytest.approx(175.0)  # the zero shape is 1.75 m tall


def test_the_mannequin_thigh_is_the_regular_polygon_perimeter(mannequin, meshes):
    betas, vertices, joints = meshes
    result = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 0.5).numpy()

    radius = shape_parameters(betas).thigh_radius_m
    expected = 100.0 * np.array([polygon_sides_perimeter(16, r) for r in radius])
    assert result[:, THIGH] == pytest.approx(expected, rel=CAUCHY_TOLERANCE)
    # A 16-sided capsule is 0.64 percent under a circle, inside the 1 percent of the task.
    assert result[:, THIGH] == pytest.approx(200.0 * math.pi * radius, rel=0.01)
    assert (result[:, THIGH] < 200.0 * math.pi * radius).all()


@pytest.mark.parametrize("step_cm", [0.5, 3.0, 1000.0])
def test_all_five_definitions_match_the_slow_implementation(mannequin, meshes, step_cm):
    _, vertices, joints = meshes
    count = 3 if step_cm == 0.5 else 8
    result = measure_batch(
        vertices[:count], mannequin.faces, mannequin.part_ids, joints[:count], step_cm
    ).numpy()

    slow = np.array(
        [
            slow_measurements(vertices[i], joints[i], mannequin.faces, mannequin.part_ids, step_cm)
            for i in range(count)
        ]
    )
    assert np.isfinite(slow).all()
    assert result[:, HEIGHT] == pytest.approx(slow[:, HEIGHT], rel=1e-12)
    assert result == pytest.approx(slow, rel=RESEARCH_PERIMETER_TOLERANCE)


def test_a_finer_step_never_loosens_the_extremes(mannequin, meshes):
    """Step 0.5 cm tries every slice that step 1 cm tries, so its maximum is never lower."""
    _, vertices, joints = meshes
    coarse = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 1.0).numpy()
    fine = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 0.5).numpy()

    assert (fine[:, CHEST] >= coarse[:, CHEST]).all()
    assert (fine[:, HIP] >= coarse[:, HIP]).all()
    assert (fine[:, WAIST] <= coarse[:, WAIST]).all()
    assert (fine[:, [HEIGHT, THIGH]] == coarse[:, [HEIGHT, THIGH]]).all()


def test_the_step_sets_the_slice_grid():
    """On cones the perimeter changes with height, so the ends of each search range show."""
    sides = 24
    widening, faces = cone_mesh(lambda y: 0.1 + 0.1 * y, sides)
    narrowing, _ = cone_mesh(lambda y: 0.2 - 0.1 * y, sides)
    vertices = np.stack([widening, narrowing])
    parts = np.full(vertices.shape[1], JOINT_INDEX["spine2"])
    joints = joints_with(2, pelvis=0.2, spine2=0.3075, spine3=0.315, left_hip=0.9, left_knee=0.5)

    fine = measure_batch(vertices, faces, parts, joints, 0.5).numpy()
    coarse = measure_batch(vertices, faces, parts, joints, 1.0).numpy()

    def girth(height, widens):
        radius = 0.1 + 0.1 * height if widens else 0.2 - 0.1 * height
        return 100.0 * polygon_sides_perimeter(sides, radius)

    # Chest range 0.3075 to 0.315 m: slices 0.3075 and 0.3125 at 0.5 cm, only 0.3075 at 1 cm.
    # The maximum of the widening cone is at the last slice.
    assert fine[0, CHEST] == pytest.approx(girth(0.3125, True), rel=CAUCHY_TOLERANCE)
    assert coarse[0, CHEST] == pytest.approx(girth(0.3075, True), rel=CAUCHY_TOLERANCE)
    # Waist range 0.2 to 0.3075 m: the last slice is 0.305 at 0.5 cm and 0.30 at 1 cm. The
    # minimum of the narrowing cone is at the last slice.
    assert fine[1, WAIST] == pytest.approx(girth(0.305, False), rel=CAUCHY_TOLERANCE)
    assert coarse[1, WAIST] == pytest.approx(girth(0.30, False), rel=CAUCHY_TOLERANCE)
    # Hip range: 0.10 of the height (1 m) below the pelvis joint, so 0.1 to 0.2 m, on both grids.
    # The maximum of the narrowing cone is at the first slice, of the widening cone at the last.
    for steps in (fine, coarse):
        assert steps[1, HIP] == pytest.approx(girth(0.1, False), rel=CAUCHY_TOLERANCE)
        assert steps[0, HIP] == pytest.approx(girth(0.2, True), rel=CAUCHY_TOLERANCE)
    # The other ends of the other ranges are on both grids, so those values agree.
    assert fine[0, WAIST] == pytest.approx(coarse[0, WAIST], rel=1e-12)
    assert fine[1, CHEST] == pytest.approx(coarse[1, CHEST], rel=1e-12)


def test_the_thigh_slice_lies_30_percent_of_the_way_from_the_left_hip_to_the_left_knee():
    sides = 24
    vertices, faces = cone_mesh(lambda y: 0.1 + 0.1 * y, sides)
    parts = np.full(len(vertices), JOINT_INDEX["left_hip"])
    # The right hip and knee sit elsewhere, so only the left joints can give the right answer.
    joints = joints_with(1, left_hip=0.9, left_knee=0.3, right_hip=0.55, right_knee=0.1)

    thigh = measure_batch(vertices[None], faces, parts, joints, 0.5)[0, THIGH].item()

    level = 0.9 - 0.3 * (0.9 - 0.3)  # 0.72 m
    assert thigh == pytest.approx(
        100.0 * polygon_sides_perimeter(sides, 0.1 + 0.1 * level), rel=CAUCHY_TOLERANCE
    )


def test_a_slice_with_fewer_than_three_cut_points_is_nan():
    parts = np.full(4, JOINT_INDEX["left_hip"])
    joints = joints_with(1, left_hip=1.0, left_knee=0.0)  # the slice is at 0.7 m
    one_triangle = np.array([[0.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 1.0, 0.1], [-0.1, 1.0, 0.0]])

    # One triangle is cut along a segment: two points.
    alone = measure_batch(one_triangle[None], np.array([[0, 1, 2]]), parts, joints, 0.5)
    assert math.isnan(alone[0, THIGH].item())
    assert alone[0, HEIGHT].item() == pytest.approx(100.0)

    # A second triangle that shares an edge adds one more point: three points have a perimeter.
    two_triangles = measure_batch(
        one_triangle[None], np.array([[0, 1, 2], [0, 2, 3]]), parts, joints, 0.5
    )
    corners = np.array([[0.07, 0.0], [0.0, 0.07], [-0.07, 0.0]])
    assert two_triangles[0, THIGH].item() == pytest.approx(
        100.0 * polygon_perimeter(corners), rel=CAUCHY_TOLERANCE
    )


# The plane test and the part sets.


def test_vertex_rings_on_the_slice_plane_never_add_or_drop_a_point():
    hip_height, knee_height = 1.0, 0.0
    # The slice height, written as in research R6.
    level = hip_height - 0.30 * (hip_height - knee_height)
    polygon = regular_polygon(12, 0.1)
    expected = 100.0 * polygon_perimeter(polygon)
    joints = joints_with(1, left_hip=hip_height, left_knee=knee_height)
    parts = np.full(12 * 3, JOINT_INDEX["left_hip"])

    def thigh(ring_heights):
        vertices, faces = prism(polygon, ring_heights)
        return measure_batch(vertices[None], faces, parts, joints, 0.5)[0, THIGH].item()

    # A ring on the plane counts as below it: the ring above cuts it exactly once, at the ring.
    assert thigh([level - 0.2, level, level + 0.2]) == pytest.approx(expected, rel=CAUCHY_TOLERANCE)
    assert thigh([level, level + 0.1, level + 0.2]) == pytest.approx(expected, rel=CAUCHY_TOLERANCE)
    # Nothing lies above the plane, so no edge is cut.
    assert math.isnan(thigh([level - 0.2, level - 0.1, level]))


def test_a_triangle_with_one_vertex_in_the_part_set_is_cut():
    polygon = regular_polygon(12, 0.1)
    vertices, faces = prism(polygon, [0.0, 1.0, 2.0, 3.0])
    # Rings 0 and 1 belong to the left hip, rings 2 and 3 to the left knee.
    parts = np.repeat([JOINT_INDEX["left_hip"], JOINT_INDEX["left_knee"]], 24)

    def thigh(hip_height):
        joints = joints_with(1, left_hip=hip_height, left_knee=0.0)
        return measure_batch(vertices[None], faces, parts, joints, 0.5)[0, THIGH].item()

    # The slice at 1.4 m cuts only the triangles between ring 1 and ring 2, which each touch ring 1.
    assert thigh(2.0) == pytest.approx(100.0 * polygon_perimeter(polygon), rel=CAUCHY_TOLERANCE)
    # The slice at 2.1 m cuts only triangles that touch no left hip vertex.
    assert math.isnan(thigh(3.0))


def test_the_hull_spans_concavities_like_a_tape_measure():
    notched = np.array(
        [
            (-0.1, -0.1),
            (0.1, -0.1),
            (0.1, -0.02),
            (0.0, -0.02),
            (0.0, 0.02),
            (0.1, 0.02),
            (0.1, 0.1),
            (-0.1, 0.1),
        ]
    )
    vertices, faces = prism(notched, [0.0, 0.5, 1.0])
    parts = np.full(len(vertices), JOINT_INDEX["left_hip"])
    joints = joints_with(1, left_hip=1.0, left_knee=0.0)

    thigh = measure_batch(vertices[None], faces, parts, joints, 0.5)[0, THIGH].item()

    assert polygon_perimeter(notched) == pytest.approx(1.0)  # the outline is 100 cm long ...
    assert thigh == pytest.approx(80.0, rel=CAUCHY_TOLERANCE)  # ... but the hull is the square


def test_only_the_parts_of_a_definition_change_its_value(mannequin, meshes):
    _, vertices, joints = meshes
    base = measure_batch(vertices[:1], mannequin.faces, mannequin.part_ids, joints[:1], 0.5)

    def measured(edit):
        edited = vertices[:1].copy()
        edit(edited[0])
        return measure_batch(edited, mannequin.faces, mannequin.part_ids, joints[:1], 0.5)

    def part_mask(*names):
        return np.isin(mannequin.part_ids, [JOINT_INDEX[name] for name in names])

    def shifted(*names):
        def edit(mesh):
            mesh[part_mask(*names), 0] += 1.0
            mesh[part_mask(*names), 2] += 0.3

        return measured(edit)

    def stretched(name):
        def edit(mesh):
            part = mesh[part_mask(name)]
            part[:, 0] = part[:, 0].mean() + 1.5 * (part[:, 0] - part[:, 0].mean())
            mesh[part_mask(name)] = part

        return measured(edit)

    # Collars, arms, neck, head, shins, and feet belong to no circumference (the collars would
    # carry the chest maximum up to the shoulders), and sideways moves leave the height alone.
    outside = (
        "left_collar right_collar left_shoulder right_shoulder left_elbow right_elbow "
        "left_wrist right_wrist neck head left_knee right_knee left_ankle right_ankle "
        "left_foot right_foot"
    ).split()
    assert torch.equal(shifted(*outside), base)
    for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
        assert torch.equal(shifted(name), base)

    # Lower the collars to the start of the chest range and move them aside: if they counted as
    # torso, the chest maximum would move to them.
    def lowered_collars(mesh):
        collars = part_mask("left_collar", "right_collar")
        drop = joints[0, JOINT_INDEX["left_collar"], 1] - joints[0, JOINT_INDEX["spine2"], 1]
        mesh[collars, 1] -= drop
        mesh[collars, 0] += 1.0

    assert torch.equal(measured(lowered_collars), base)

    # Stretching a part that a definition uses changes that definition and no other one.
    unchanged = [HEIGHT, CHEST, WAIST, THIGH]
    right_hip = stretched("right_hip")  # in the hip definition, but the thigh is the left one
    assert right_hip[0, HIP] != base[0, HIP]
    assert torch.equal(right_hip[0, unchanged], base[0, unchanged])
    left_hip = stretched("left_hip")
    assert left_hip[0, THIGH] != base[0, THIGH]
    assert left_hip[0, HIP] != base[0, HIP]
    assert torch.equal(left_hip[0, [HEIGHT, CHEST, WAIST]], base[0, [HEIGHT, CHEST, WAIST]])
    spine2 = stretched("spine2")
    assert spine2[0, CHEST] != base[0, CHEST]
    assert torch.equal(spine2[0, [HEIGHT, THIGH]], base[0, [HEIGHT, THIGH]])


def test_the_measurements_do_not_depend_on_where_a_body_stands(mannequin, meshes):
    _, vertices, joints = meshes
    base = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 0.5).numpy()
    # A different sideways offset for every mesh puts the origin outside most slices.
    offsets = np.array([0.0, 1.7, -0.8, 25.0, -3.0, 0.4, 9.0, -12.5])
    moved = vertices.copy()
    moved[:, :, 0] += offsets[:, None]
    moved[:, :, 2] -= 1.3 * offsets[:, None]

    result = measure_batch(moved, mannequin.faces, mannequin.part_ids, joints, 0.5).numpy()

    assert result[:, HEIGHT] == pytest.approx(base[:, HEIGHT], rel=1e-12)
    assert result == pytest.approx(base, rel=1e-9)


# Accuracy of the perimeter.


def test_the_perimeter_error_stays_under_a_tenth_of_a_percent():
    rng = np.random.default_rng(64)
    worst = 0.0
    shapes = []
    for sides in (3, 4, 5, 7, 16, 33, 64):  # regular polygons at random turns
        shapes.append([regular_polygon(sides, 0.1, turn) for turn in rng.uniform(0, 6.3, 8)])
    for width, depth in ((0.1, 0.1), (0.2, 0.05), (0.3, 0.015), (0.3, 0.005)):  # axis-aligned
        shapes.append(
            [np.array([(-width, -depth), (width, -depth), (width, depth), (-width, depth)])]
        )
    for aspect in (1.0, 2.0, 5.0, 10.0):  # turned ellipses
        ellipses = []
        for turn in rng.uniform(0, 6.3, 6):
            ring = regular_polygon(64, 1.0, 0.0)
            scaled = ring * np.array([0.1 * aspect, 0.1])
            rotation = np.array(
                [[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]
            )
            ellipses.append(scaled @ rotation.T)
        shapes.append(ellipses)

    for polygons in shapes:
        vertices = np.stack([prism(polygon, [0.0, 0.5, 1.0])[0] for polygon in polygons])
        faces = prism(polygons[0], [0.0, 0.5, 1.0])[1]
        parts = np.full(vertices.shape[1], JOINT_INDEX["left_hip"])
        joints = joints_with(len(polygons), left_hip=1.0, left_knee=0.0)
        measured = measure_batch(vertices, faces, parts, joints, 0.5)[:, THIGH].numpy() / 100.0
        exact = np.array([polygon_perimeter(polygon) for polygon in polygons])
        worst = max(worst, float(np.abs(measured / exact - 1.0).max()))

    assert worst < RESEARCH_PERIMETER_TOLERANCE
    assert worst <= CAUCHY_TOLERANCE  # the rectangle-rule bound (pi / 64)^2 / 12 of the perimeter


# NaN on degenerate slices.


def test_nan_marks_an_empty_slice_and_leaves_the_height(mannequin, meshes):
    _, vertices, joints = meshes
    # Every joint 10 m up puts every search range above the whole mesh.
    lifted = joints.copy()
    lifted[:, :, 1] += 10.0

    result = measure_batch(vertices, mannequin.faces, mannequin.part_ids, lifted, 0.5).numpy()

    assert np.isnan(result[:, CHEST:]).all()
    assert np.isfinite(result[:, HEIGHT]).all()
    assert slice_nan_flags(result).all()


def test_nan_marks_a_part_set_without_vertices(mannequin, meshes):
    _, vertices, joints = meshes
    no_left_hip = mannequin.part_ids.copy()
    no_left_hip[no_left_hip == JOINT_INDEX["left_hip"]] = JOINT_INDEX["right_hip"]

    result = measure_batch(vertices, mannequin.faces, no_left_hip, joints, 0.5).numpy()

    assert np.isnan(result[:, THIGH]).all()
    assert np.isfinite(result[:, [HEIGHT, CHEST, WAIST, HIP]]).all()


def test_nan_marks_a_search_range_that_runs_past_the_mesh():
    radii = np.array([0.05, 0.1])
    vertices, faces = capsule_batch(radii)
    parts = np.full(vertices.shape[1], JOINT_INDEX["spine2"])
    # The chest range reaches 2 m, past the capsule top, so its upper slices are empty.
    joints = joints_with(2, pelvis=0.5, spine2=0.65, spine3=2.0, left_hip=0.4, left_knee=0.1)

    result = measure_batch(vertices, faces, parts, joints, 0.5).numpy()

    assert np.isnan(result[:, CHEST]).all()
    assert np.isfinite(result[:, [HEIGHT, WAIST, HIP]]).all()


def test_nan_in_one_mesh_does_not_reach_the_others(mannequin, meshes):
    _, vertices, joints = meshes
    batch_joints = joints[:3].copy()
    batch_joints[1, :, 1] += 10.0

    result = measure_batch(vertices[:3], mannequin.faces, mannequin.part_ids, batch_joints, 0.5)

    assert slice_nan_flags(result).tolist() == [False, True, False]
    for index in (0, 2):
        alone = measure_mesh(
            vertices[index], mannequin.faces, mannequin.part_ids, batch_joints[index], 0.5
        )
        assert torch.equal(result[index], alone)


def test_a_nan_vertex_height_gives_a_nan_height_and_a_flag(mannequin, meshes):
    _, vertices, joints = meshes
    broken = vertices[:2].copy()
    head_vertex = int(np.flatnonzero(mannequin.part_ids == JOINT_INDEX["head"])[0])
    broken[1, head_vertex, 1] = np.nan

    result = measure_batch(broken, mannequin.faces, mannequin.part_ids, joints[:2], 0.5)

    assert slice_nan_flags(result).tolist() == [False, True]
    assert torch.isnan(result[1, HEIGHT])
    # The head belongs to no circumference. Only the hip range reads the height, so only it is lost.
    assert torch.isnan(result[1, HIP])
    assert torch.isfinite(result[1, [CHEST, WAIST, THIGH]]).all()


@pytest.mark.parametrize("bad_height", [math.nan, math.inf, -math.inf])
def test_a_joint_height_that_is_not_finite_gives_nan_for_the_ranges_that_use_it(
    mannequin, meshes, bad_height
):
    _, vertices, joints = meshes
    broken = joints[:2].copy()
    broken[1, JOINT_INDEX["pelvis"], 1] = bad_height

    result = measure_batch(vertices[:2], mannequin.faces, mannequin.part_ids, broken, 0.5)

    # The pelvis height starts the waist and ends the hip range; the chest and thigh avoid it.
    assert torch.isnan(result[1, [WAIST, HIP]]).all()
    assert torch.isfinite(result[1, [HEIGHT, CHEST, THIGH]]).all()
    assert torch.isfinite(result[0]).all()


def test_a_search_range_of_millions_of_slices_is_refused(mannequin, meshes):
    _, vertices, joints = meshes
    absurd = joints[:1].copy()
    absurd[0, JOINT_INDEX["spine3"], 1] = 1.0e5  # a chest range 100 km tall

    with pytest.raises(ValueError, match="more than 1000000 slices"):
        measure_batch(vertices[:1], mannequin.faces, mannequin.part_ids, absurd, 0.5)


def test_a_mesh_without_faces_has_a_height_and_no_circumferences(mannequin, meshes):
    _, vertices, joints = meshes
    no_faces = np.zeros((0, 3), dtype=np.int64)

    result = measure_batch(vertices[:2], no_faces, mannequin.part_ids, joints[:2], 0.5).numpy()

    assert np.isfinite(result[:, HEIGHT]).all()
    assert np.isnan(result[:, CHEST:]).all()


def test_slice_nan_flags_marks_rows_with_a_non_finite_entry():
    rows = torch.tensor(
        [
            [175.0, 90.0, 80.0, 91.0, 53.0],
            [175.0, math.nan, 80.0, 91.0, 53.0],
            [175.0, 90.0, 80.0, 91.0, math.inf],
        ]
    )
    assert slice_nan_flags(rows).tolist() == [False, True, True]
    assert slice_nan_flags(rows.numpy()).tolist() == [False, True, True]
    assert slice_nan_flags(rows[0]).item() is False


# Batching, chunking, devices, and arguments.


def test_the_batched_result_equals_the_single_mesh_result(mannequin, meshes):
    _, vertices, joints = meshes
    batched = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 0.5)

    assert batched.shape == (8, 5)
    assert batched.dtype == torch.float64
    for index in range(len(vertices)):
        single = measure_mesh(
            vertices[index], mannequin.faces, mannequin.part_ids, joints[index], 0.5
        )
        assert single.shape == (5,)
        assert torch.equal(single, batched[index])
    reversed_batch = measure_batch(
        vertices[::-1], mannequin.faces, mannequin.part_ids, joints[::-1], 0.5
    )
    assert torch.equal(reversed_batch.flip(0), batched)


@pytest.mark.parametrize("memory_budget_mb", [0.05, 0.001])
def test_chunking_for_memory_changes_no_result(mannequin, meshes, memory_budget_mb):
    _, vertices, joints = meshes
    full = measure_batch(vertices[:4], mannequin.faces, mannequin.part_ids, joints[:4], 0.5)

    chunked = measure_batch(
        vertices[:4],
        mannequin.faces,
        mannequin.part_ids,
        joints[:4],
        0.5,
        memory_budget_mb=memory_budget_mb,
    )

    assert torch.equal(chunked, full)


def test_a_small_memory_budget_splits_the_batch_into_chunks_of_meshes(
    monkeypatch, mannequin, meshes
):
    _, vertices, joints = meshes
    chunk_sizes = []
    original = measure_module._measure_chunk

    def recording(mesh, *arguments):
        chunk_sizes.append(mesh.shape[0])
        return original(mesh, *arguments)

    monkeypatch.setattr(measure_module, "_measure_chunk", recording)
    faces, parts = mannequin.faces, mannequin.part_ids

    measure_batch(vertices, faces, parts, joints, 0.5)
    assert chunk_sizes == [8]  # the default budget holds all eight meshes

    chunk_sizes.clear()
    measure_batch(vertices, faces, parts, joints, 0.5, memory_budget_mb=2.0)
    assert sum(chunk_sizes) == 8
    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) < 8


def test_rows_of_slices_are_processed_in_blocks_without_changing_results():
    """Every edge of this prism is cut by every slice, so the rows of cut points dominate memory."""
    sides = 24
    radii = np.array([0.04, 0.09, 0.15])
    polygons = [regular_polygon(sides, radius) for radius in radii]
    vertices = np.stack([prism(polygon, [0.0, 1.0])[0] for polygon in polygons])
    faces = prism(polygons[0], [0.0, 1.0])[1]
    parts = np.full(vertices.shape[1], JOINT_INDEX["spine2"])
    joints = joints_with(3, pelvis=0.5, spine2=0.6, spine3=0.7, left_hip=0.9, left_knee=0.3)

    whole = measure_batch(vertices, faces, parts, joints, 0.5)
    # With this budget one mesh and 10 slices make a block, and 4 rows make a block of points.
    blocks = measure_batch(vertices, faces, parts, joints, 0.5, memory_budget_mb=0.008)

    # No left hip vertex exists, so the thigh column is NaN in both results; NaN equals NaN here.
    torch.testing.assert_close(blocks, whole, rtol=0.0, atol=0.0, equal_nan=True)
    expected = 100.0 * np.array([polygon_sides_perimeter(sides, radius) for radius in radii])
    for column in (CHEST, WAIST, HIP):
        assert whole[:, column].numpy() == pytest.approx(expected, rel=CAUCHY_TOLERANCE)


def test_the_thread_count_changes_no_result(mannequin, meshes):
    _, vertices, joints = meshes
    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = measure_batch(vertices[:4], mannequin.faces, mannequin.part_ids, joints[:4], 0.5)
        torch.set_num_threads(3)
        three = measure_batch(vertices[:4], mannequin.faces, mannequin.part_ids, joints[:4], 0.5)
    finally:
        torch.set_num_threads(saved)
    assert torch.equal(one, three)


def test_arrays_tensors_and_precisions_give_the_same_measurements(mannequin, meshes):
    _, vertices, joints = meshes
    faces, parts = mannequin.faces, mannequin.part_ids
    reference = measure_batch(vertices[:2], faces, parts, joints[:2], 0.5)

    as_tensors = measure_batch(
        torch.from_numpy(vertices[:2].copy()),
        torch.from_numpy(faces),
        torch.from_numpy(parts),
        torch.from_numpy(joints[:2].copy()),
        0.5,
        device="cpu",
    )
    as_lists = measure_batch(vertices[:2].tolist(), faces.tolist(), parts.tolist(), joints[:2], 0.5)
    single_precision = measure_batch(
        vertices[:2].astype(np.float32), faces, parts, joints[:2].astype(np.float32), 0.5
    )
    narrow_indices = measure_batch(
        vertices[:2], faces.astype(np.int32), parts.astype(np.uint8), joints[:2], 0.5
    )

    assert torch.equal(as_tensors, reference)
    assert torch.equal(as_lists, reference)
    assert torch.equal(narrow_indices, reference)
    assert reference.device.type == "cpu" and reference.dtype == torch.float64
    # The mesh is the single-precision rounding of the float64 mesh, a change of a few micrometres.
    assert torch.allclose(single_precision, reference, rtol=1e-4)


def test_gradients_are_not_tracked(mannequin, meshes):
    _, vertices, joints = meshes
    tracked = torch.from_numpy(vertices[:1].copy()).requires_grad_(True)

    result = measure_batch(tracked, mannequin.faces, mannequin.part_ids, joints[:1], 0.5)

    assert not result.requires_grad


def test_an_empty_batch_gives_an_empty_result(mannequin):
    vertices = np.zeros((0, mannequin.vertex_count, 3))
    joints = np.zeros((0, NUM_JOINTS, 3))

    result = measure_batch(vertices, mannequin.faces, mannequin.part_ids, joints, 0.5)

    assert result.shape == (0, 5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"step_cm": 0.0}, "step_cm"),
        ({"step_cm": -0.5}, "step_cm"),
        ({"step_cm": math.nan}, "step_cm"),
        ({"step_cm": math.inf}, "step_cm"),
        ({"memory_budget_mb": 0.0}, "memory_budget_mb"),
        ({"vertices": np.zeros((2, 4268))}, "vertices must have shape"),
        ({"vertices": np.zeros((2, 4268, 2))}, "vertices must have shape"),
        ({"vertices": np.zeros((2, 0, 3))}, "vertices must have shape"),
        ({"joints": np.zeros((3, 22, 3))}, "joints must have shape"),
        ({"joints": np.zeros((2, 21, 3))}, "joints must have shape"),
        ({"faces": np.zeros((5, 4), dtype=np.int64)}, "faces must have shape"),
        ({"faces": np.full((5, 3), 4268, dtype=np.int64)}, "faces must index vertices"),
        ({"faces": np.full((5, 3), -1, dtype=np.int64)}, "faces must index vertices"),
        ({"faces": np.zeros((5, 3))}, "faces must hold integers"),
        ({"part_ids": np.zeros(10, dtype=np.int64)}, "part_ids must have one entry per vertex"),
        ({"part_ids": np.zeros(4268)}, "part_ids must hold integers"),
    ],
)
def test_invalid_arguments_are_refused_with_a_clear_message(mannequin, meshes, change, message):
    _, vertices, joints = meshes
    arguments = {
        "vertices": vertices[:2],
        "faces": mannequin.faces,
        "part_ids": mannequin.part_ids,
        "joints": joints[:2],
        "step_cm": 0.5,
    }
    arguments.update(change)

    with pytest.raises(ValueError, match=message):
        measure_batch(**arguments)
