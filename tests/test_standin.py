"""Smoke tests for body/standin.py: vertex count, height, pose subtrees, shape, and determinism."""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_POSE_SIZE,
    joint_subtree,
)
from strike_a_pose.body.standin import BETA_LIMIT, NUM_BETAS, StandInBody, shape_parameters

# Vertices and faces of the default mannequin (research R4): 2 + 2 * 6 * 16 vertices per capsule,
# 4 * 16 * 6 faces per capsule, and 22 capsules.
TOTAL_VERTICES = 4268
TOTAL_FACES = 8448
VERTICES_PER_PART = 194

ZERO_ROOT = np.zeros(ROOT_POSE_SIZE)
ZERO_POSE = np.zeros(BODY_POSE_SIZE)

# The proportions at coefficient zero, and the order in which the coefficients are listed.
NOMINAL_PROPORTIONS = {
    "height_m": 1.75,
    "torso_width_m": 0.34,
    "torso_depth_m": 0.235,
    "hip_width_m": 0.19,
    "thigh_radius_m": 0.085,
    "arm_radius_m": 0.045,
    "shoulder_width_m": 0.36,
    "head_radius_m": 0.085,
    "leg_ratio": 0.52,
    "waist_indent": 0.12,
}
PROPORTION_FIELDS = tuple(NOMINAL_PROPORTIONS)

# The program that a child process runs; its digest must equal the digest of this process.
_DIGEST_PROGRAM = """
import hashlib
import numpy as np
from strike_a_pose.body.standin import StandInBody

betas = np.linspace(-2.0, 2.0, 10)
root = np.array([0.4, -0.2, 0.9])
pose = np.sin(np.arange(63) * 0.37) * 0.8
body = StandInBody()
digest = hashlib.sha256()
digest.update(body.vertices(betas, root, pose).tobytes())
digest.update(body.joints(betas, root, pose).tobytes())
print(digest.hexdigest())
"""


@pytest.fixture(scope="module")
def body() -> StandInBody:
    return StandInBody()


def _random_inputs(seed: int, count: int):
    """Return shape coefficients, root poses, and body poses for ``count`` random bodies."""
    rng = np.random.default_rng(seed)
    betas = np.clip(rng.standard_normal((count, NUM_BETAS)), -3.0, 3.0)
    root = 0.8 * rng.standard_normal((count, ROOT_POSE_SIZE))
    pose = 0.7 * rng.standard_normal((count, BODY_POSE_SIZE))
    return betas, root, pose


def _rest(body: StandInBody, betas):
    """Return the vertices and joints of the zero pose for the shape coefficients ``betas``."""
    return body.vertices(betas, ZERO_ROOT, ZERO_POSE), body.joints(betas, ZERO_ROOT, ZERO_POSE)


def _rotation_matrix(axis_angle) -> np.ndarray:
    """Return the rotation matrix of an axis-angle vector, from Rodrigues' formula.

    The axis form R = cos(t) I + sin(t) K + (1 - cos(t)) n n^T uses the unit axis n and the
    cross-product matrix K of n. standin.py writes the same rotation another way, so the two can
    check each other.
    """
    angle = float(np.linalg.norm(axis_angle))
    if angle == 0.0:
        return np.eye(3)
    x, y, z = np.asarray(axis_angle, dtype=np.float64) / angle
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    axis = np.array([x, y, z])
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
    )


def _reference_pose(rest_vertices, rest_joints, part_ids, pose_root, pose_body):
    """Pose a rest mesh with plain loops: forward kinematics by joint, then rigid skinning.

    Returns the posed vertices and the posed joints.
    """
    rotations = [_rotation_matrix(pose_root)]
    for joint in range(1, NUM_JOINTS):
        rotations.append(_rotation_matrix(pose_body[3 * (joint - 1) : 3 * joint]))
    global_rotations = [None] * NUM_JOINTS
    posed_joints = np.zeros((NUM_JOINTS, 3))
    for joint in range(NUM_JOINTS):
        parent = PARENTS[joint]
        if parent < 0:
            global_rotations[joint] = rotations[joint]
            posed_joints[joint] = rest_joints[joint]
        else:
            global_rotations[joint] = global_rotations[parent] @ rotations[joint]
            offset = rest_joints[joint] - rest_joints[parent]
            posed_joints[joint] = posed_joints[parent] + global_rotations[parent] @ offset
    posed_vertices = np.zeros_like(rest_vertices)
    for index, owner in enumerate(part_ids):
        offset = rest_vertices[index] - rest_joints[owner]
        posed_vertices[index] = global_rotations[owner] @ offset + posed_joints[owner]
    return posed_vertices, posed_joints


def _digest_in_this_process() -> str:
    """The digest that ``_DIGEST_PROGRAM`` prints, computed in the test process."""
    betas = np.linspace(-2.0, 2.0, 10)
    root = np.array([0.4, -0.2, 0.9])
    pose = np.sin(np.arange(63) * 0.37) * 0.8
    body = StandInBody()
    digest = hashlib.sha256()
    digest.update(body.vertices(betas, root, pose).tobytes())
    digest.update(body.joints(betas, root, pose).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- joint table and sizes


def test_joint_names_and_part_sizes_match_the_base_table(body):
    assert body.joint_names == JOINT_NAMES
    assert body.vertex_count == TOTAL_VERTICES
    assert body.faces.shape == (TOTAL_FACES, 3)
    counts = np.bincount(body.part_ids, minlength=NUM_JOINTS)
    assert np.array_equal(counts, np.full(NUM_JOINTS, VERTICES_PER_PART))


def test_canonical_pose_is_all_zero(body):
    pose = body.canonical()
    assert pose.pose_root.shape == (ROOT_POSE_SIZE,)
    assert pose.pose_body.shape == (BODY_POSE_SIZE,)
    assert not pose.pose_root.any()
    assert not pose.pose_body.any()


def test_vertex_count_is_stable_for_every_shape_and_pose(body):
    betas, root, pose = _random_inputs(1, 6)
    betas[0, 0] = 4.0 * BETA_LIMIT  # far beyond the limit: clipped, still the same count
    pose[1] *= 20.0  # a pose far beyond the usual range
    assert body.vertices(betas[0], root[0], pose[0]).shape == (TOTAL_VERTICES, 3)
    assert body.vertices(betas, root, pose).shape == (6, TOTAL_VERTICES, 3)
    assert body.joints(betas, root, pose).shape == (6, NUM_JOINTS, 3)
    assert np.isfinite(body.vertices(betas, root, pose)).all()


@pytest.mark.parametrize(("radial_segments", "cap_rings"), [(8, 2), (16, 6), (5, 1)])
def test_vertex_and_face_counts_follow_the_resolution(radial_segments, cap_rings):
    model = StandInBody(radial_segments=radial_segments, cap_rings=cap_rings)
    per_capsule = 2 + 2 * cap_rings * radial_segments
    assert model.vertex_count == NUM_JOINTS * per_capsule
    assert model.faces.shape == (NUM_JOINTS * 4 * radial_segments * cap_rings, 3)
    vertices = model.vertices(np.zeros(NUM_BETAS), ZERO_ROOT, ZERO_POSE)
    assert vertices.shape == (model.vertex_count, 3)
    assert np.isfinite(vertices).all()


def test_faces_and_part_ids_are_copies(body):
    faces = body.faces
    faces[:] = 0
    parts = body.part_ids
    parts[:] = 0
    assert body.faces.max() == TOTAL_VERTICES - 1
    assert body.part_ids.max() == NUM_JOINTS - 1


def test_each_part_is_a_closed_surface_with_outward_faces(body):
    vertices, _ = _rest(body, np.full(NUM_BETAS, 1.5))
    faces = body.faces
    owners = body.part_ids
    assert np.unique(faces).size == TOTAL_VERTICES  # every vertex belongs to some face
    for joint in range(NUM_JOINTS):
        part_faces = faces[owners[faces[:, 0]] == joint]
        assert np.all(owners[part_faces] == joint)  # rigid skinning: no face crosses two parts
        directed = {
            (int(a), int(b))
            for face in part_faces
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        }
        assert len(directed) == 3 * len(part_faces)  # no edge is used twice in one direction
        assert all((b, a) in directed for a, b in directed)  # each edge has its reverse: closed
        triangles = vertices[part_faces]
        volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
        assert volume.sum() / 6.0 > 0.0  # normals point out of the capsule
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        assert areas.min() > 1e-9


# ---------------------------------------------------------------- shape and height


def test_coefficient_zero_raises_the_height_monotonically(body):
    values = np.linspace(-BETA_LIMIT, BETA_LIMIT, 21)
    betas = np.zeros((values.size, NUM_BETAS))
    betas[:, 0] = values
    vertices = body.vertices(betas, ZERO_ROOT, ZERO_POSE)
    heights = vertices[..., 1].max(axis=1) - vertices[..., 1].min(axis=1)
    assert np.all(np.diff(heights) > 0.0)
    # Height is 1.75 m times (1 + 0.04 * coefficient), as shape_parameters states.
    assert np.allclose(heights, 1.75 * (1.0 + 0.04 * values), rtol=0.0, atol=1e-12)


def test_soles_rest_on_the_floor_and_the_head_is_the_top(body):
    betas, _, _ = _random_inputs(3, 5)
    vertices, _ = _rest(body, betas)
    assert np.allclose(vertices[..., 1].min(axis=1), 0.0, rtol=0.0, atol=1e-12)
    assert np.allclose(
        vertices[..., 1].max(axis=1), shape_parameters(betas).height_m, rtol=0.0, atol=1e-12
    )
    highest = vertices[..., 1].argmax(axis=1)
    assert np.all(body.part_ids[highest] == JOINT_INDEX["head"])


def test_left_joints_sit_on_positive_x_and_mirror_the_right_joints(body):
    betas, _, _ = _random_inputs(4, 1)
    _, joints = _rest(body, betas[0])
    for name in JOINT_NAMES:
        if name.startswith("left_"):
            left = joints[JOINT_INDEX[name]]
            right = joints[JOINT_INDEX["right_" + name[len("left_") :]]]
            assert left[0] > 0.0
            assert np.allclose(left, right * np.array([-1.0, 1.0, 1.0]), rtol=0.0, atol=1e-12)


def test_zero_coefficients_give_the_nominal_proportions():
    shape = shape_parameters(np.zeros(NUM_BETAS))
    for field, nominal in NOMINAL_PROPORTIONS.items():
        assert float(getattr(shape, field)) == pytest.approx(nominal, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("coefficient", range(NUM_BETAS))
def test_each_coefficient_moves_only_its_own_proportion(coefficient):
    betas = np.zeros((5, NUM_BETAS))
    betas[:, coefficient] = np.linspace(-2.0, 2.0, 5)
    shape = shape_parameters(betas)
    field = PROPORTION_FIELDS[coefficient]
    assert np.all(np.diff(getattr(shape, field)) > 0.0)
    for other in PROPORTION_FIELDS:
        if other != field:
            assert np.ptp(getattr(shape, other)) == 0.0


def test_coefficients_beyond_the_limit_are_clipped(body):
    wild = np.array([50.0, -50.0, 7.0, -9.0, 5.0001, 0.0, 1.0, 2.0, -6.0, 4.9])
    clipped = np.clip(wild, -BETA_LIMIT, BETA_LIMIT)
    wild_vertices = body.vertices(wild, ZERO_ROOT, ZERO_POSE)
    assert np.isfinite(wild_vertices).all()
    assert np.array_equal(wild_vertices, body.vertices(clipped, ZERO_ROOT, ZERO_POSE))


# ---------------------------------------------------------------- pose


@pytest.mark.parametrize("joint", range(1, NUM_JOINTS))
def test_rotating_a_joint_moves_its_subtree_and_nothing_else(body, joint):
    betas = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.6, -0.3, 0.2])
    rest_vertices, rest_joints = _rest(body, betas)
    pose = np.zeros(BODY_POSE_SIZE)
    pose[3 * (joint - 1) : 3 * joint] = [0.5, -0.3, 0.4]
    vertices = body.vertices(betas, ZERO_ROOT, pose)
    joints = body.joints(betas, ZERO_ROOT, pose)
    subtree = joint_subtree(joint)
    owners = body.part_ids

    # The rotated joint's own part and every part below it move; every other part is unchanged.
    for part in range(NUM_JOINTS):
        in_part = owners == part
        if part in subtree:
            displacement = np.linalg.norm(vertices[in_part] - rest_vertices[in_part], axis=1)
            assert displacement.mean() > 1e-3
        else:
            assert np.array_equal(vertices[in_part], rest_vertices[in_part])

    # The rotated joint stays put, its descendants move, and every other joint is unchanged.
    for other in range(NUM_JOINTS):
        if other == joint or other not in subtree:
            assert np.array_equal(joints[other], rest_joints[other])
        else:
            assert not np.allclose(joints[other], rest_joints[other], rtol=0.0, atol=1e-9)

    # Rigid motion: distances inside each moved part are preserved, and so is every bone length.
    for part in subtree:
        indices = np.flatnonzero(owners == part)[:20]
        before = rest_vertices[indices]
        after = vertices[indices]
        distance_before = np.linalg.norm(before[:, None] - before[None], axis=-1)
        distance_after = np.linalg.norm(after[:, None] - after[None], axis=-1)
        assert np.allclose(distance_before, distance_after, rtol=0.0, atol=1e-12)
    for child in range(1, NUM_JOINTS):
        parent = PARENTS[child]
        before = np.linalg.norm(rest_joints[child] - rest_joints[parent])
        after = np.linalg.norm(joints[child] - joints[parent])
        assert abs(after - before) < 1e-12


def test_root_pose_turns_the_whole_body_about_the_pelvis(body):
    betas = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.6, -0.3, 0.2])
    root = np.array([0.2, 1.1, -0.4])
    rest_vertices, rest_joints = _rest(body, betas)
    rotation = _rotation_matrix(root)
    pelvis = rest_joints[0]
    vertices = body.vertices(betas, root, ZERO_POSE)
    joints = body.joints(betas, root, ZERO_POSE)
    assert np.array_equal(joints[0], pelvis)
    expected_vertices = (rest_vertices - pelvis) @ rotation.T + pelvis
    expected_joints = (rest_joints - pelvis) @ rotation.T + pelvis
    assert np.allclose(vertices, expected_vertices, rtol=0.0, atol=1e-12)
    assert np.allclose(joints, expected_joints, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_posed_body_matches_plain_loop_forward_kinematics(body, seed):
    betas, root, pose = _random_inputs(10 + seed, 1)
    rest_vertices, rest_joints = _rest(body, betas[0])
    expected_vertices, expected_joints = _reference_pose(
        rest_vertices, rest_joints, body.part_ids, root[0], pose[0]
    )
    joints = body.joints(betas[0], root[0], pose[0])
    vertices = body.vertices(betas[0], root[0], pose[0])
    assert np.allclose(joints, expected_joints, rtol=0.0, atol=1e-12)
    assert np.allclose(vertices, expected_vertices, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e-3, 0.7, 7.0, 40.0])
def test_angles_from_tiny_to_beyond_two_pi_match_the_reference(body, scale):
    rng = np.random.default_rng(5)
    betas = np.zeros(NUM_BETAS)
    pose = rng.standard_normal(BODY_POSE_SIZE) * scale
    rest_vertices, rest_joints = _rest(body, betas)
    expected, _ = _reference_pose(rest_vertices, rest_joints, body.part_ids, ZERO_ROOT, pose)
    vertices = body.vertices(betas, ZERO_ROOT, pose)
    assert np.allclose(vertices, expected, rtol=0.0, atol=1e-9 * max(1.0, scale))


# ---------------------------------------------------------------- determinism


def test_same_inputs_give_identical_bytes_and_inputs_are_not_changed(body):
    betas, root, pose = _random_inputs(20, 1)
    betas, root, pose = betas[0], root[0], pose[0]
    originals = (betas.copy(), root.copy(), pose.copy())
    first = body.vertices(betas, root, pose)
    again = body.vertices(betas, root, pose)
    fresh = StandInBody().vertices(betas, root, pose)
    assert first.tobytes() == again.tobytes() == fresh.tobytes()
    joints = body.joints(betas, root, pose)
    assert joints.tobytes() == StandInBody().joints(betas, root, pose).tobytes()
    for kept, original in zip((betas, root, pose), originals, strict=True):
        assert np.array_equal(kept, original)


def test_a_batch_gives_the_same_bytes_as_one_body_at_a_time(body):
    betas, root, pose = _random_inputs(21, 7)
    batch = body.vertices(betas, root, pose)
    batch_joints = body.joints(betas, root, pose)
    for index in range(7):
        single = body.vertices(betas[index], root[index], pose[index])
        assert np.array_equal(batch[index], single)
        single_joints = body.joints(betas[index], root[index], pose[index])
        assert np.array_equal(batch_joints[index], single_joints)
    # Reversing the batch reverses the results and changes no value.
    reversed_batch = body.vertices(betas[::-1], root[::-1], pose[::-1])
    assert np.array_equal(reversed_batch[::-1], batch)


@pytest.mark.parametrize("hash_seed", ["0", "4242"])
def test_a_separate_process_with_another_hash_seed_gives_the_same_bytes(hash_seed):
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    completed = subprocess.run(
        [sys.executable, "-c", _DIGEST_PROGRAM],
        capture_output=True,
        text=True,
        env=environment,
        timeout=50,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == _digest_in_this_process()


# ---------------------------------------------------------------- input and resolution checks


def test_wrong_sizes_and_bad_broadcasts_raise_value_error(body):
    with pytest.raises(ValueError, match="betas"):
        body.vertices(np.zeros(9), ZERO_ROOT, ZERO_POSE)
    with pytest.raises(ValueError, match="pose_root"):
        body.vertices(np.zeros(NUM_BETAS), np.zeros(4), ZERO_POSE)
    with pytest.raises(ValueError, match="pose_body"):
        body.joints(np.zeros(NUM_BETAS), ZERO_ROOT, np.zeros(62))
    with pytest.raises(ValueError, match="do not broadcast"):
        body.vertices(np.zeros((2, NUM_BETAS)), np.zeros((3, 3)), ZERO_POSE)
    with pytest.raises(ValueError):
        body.vertices(1.0, ZERO_ROOT, ZERO_POSE)


@pytest.mark.parametrize("kwargs", [{"radial_segments": 2}, {"cap_rings": 0}])
def test_a_resolution_below_the_minimum_is_refused(kwargs):
    with pytest.raises(ValueError):
        StandInBody(**kwargs)
