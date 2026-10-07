"""Smoke tests for camera.py: intrinsics, poses, rigs, placement noise, 6D encoding, projection."""

import warnings

import cv2
import numpy as np
import pytest

from strike_a_pose import camera as camera_module
from strike_a_pose.camera import (
    CAMERA_ENCODING_DIM,
    TRANSLATION_SCALE_M,
    azimuths_are_separated,
    encode_camera,
    given_rotation,
    intrinsics,
    look_at_extrinsics,
    noise_rotation,
    orbit_position,
    project_points,
    random_unit_vectors,
    rotation_angle_deg,
    rotation_to_6d,
    sample_rig,
)
from strike_a_pose.config import ConfigError, resolve_config
from strike_a_pose.seeding import rng_for

# The bounding-box center of an average upright body that stands on the floor (research R12).
CENTER = np.array([0.0, 0.9, 0.0])


def _camera_section(**changes):
    """The camera section of the default configuration, with some keys changed."""
    section = dict(resolve_config({"seed": 1})["camera"])
    section.update(changes)
    return section


def _smallest_gap_deg(azimuths):
    """The smallest distance between two azimuths on the circle, by a plain loop over all pairs."""
    gaps = []
    for first in range(len(azimuths)):
        for second in range(first + 1, len(azimuths)):
            difference = (azimuths[first] - azimuths[second] + 180.0) % 360.0 - 180.0
            gaps.append(abs(difference))
    return min(gaps)


def _rotation_from_6d(code):
    """Map 6D to a rotation matrix by Gram-Schmidt, as Zhou et al. 2019 define it (the inverse)."""
    first, second = code[..., :3], code[..., 3:]
    b1 = first / np.linalg.norm(first, axis=-1, keepdims=True)
    b2 = second - np.sum(b1 * second, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _opencv_pixels(points, K, R, t):
    """Project points with cv2.perspectiveTransform and the 4 by 4 matrix [[K R, K t], [R_z, t_z]].

    The last row gives the depth, so OpenCV divides by it. This stands in for cv2.projectPoints,
    which needs a rotation vector, and cv2.Rodrigues is only accurate to about 1e-5 near 180
    degrees, where look-at poses sit.
    """
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = K @ R
    matrix[:3, 3] = K @ t
    matrix[3, :3] = R[2]
    matrix[3, 3] = t[2]
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 3), matrix)
    return projected.reshape(-1, 3)[:, :2]


# Intrinsics


def test_intrinsics_put_the_focal_length_on_the_diagonal_and_the_principal_point_at_the_center():
    K = intrinsics(64, 80.0)
    assert K.dtype == np.float64
    assert np.array_equal(K, [[80.0, 0.0, 31.5], [0.0, 80.0, 31.5], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_the_field_of_view_stays_53_degrees_when_the_focal_length_follows_the_size(image_size):
    # contracts/config.md: the small test (32), tiny (64), and full (128) configurations.
    K = intrinsics(image_size, float(image_size))
    half_width = K[0, 2] + 0.5  # the image edge lies half a pixel beyond the last pixel center
    field_of_view = 2.0 * np.degrees(np.arctan(half_width / K[0, 0]))
    assert field_of_view == pytest.approx(53.13, abs=0.01)


@pytest.mark.parametrize(
    ("image_size", "focal_px"),
    [(0, 64.0), (-3, 64.0), (64, 0.0), (64, -1.0), (64, float("nan")), (64, float("inf"))],
)
def test_intrinsics_refuse_values_that_make_no_camera(image_size, focal_px):
    with pytest.raises(ValueError):
        intrinsics(image_size, focal_px)


# Camera position and look-at pose


@pytest.mark.parametrize(
    ("azimuth_deg", "expected"),
    [
        (0.0, [0.0, 1.2, 3.0]),
        (90.0, [3.0, 1.2, 0.0]),
        (180.0, [0.0, 1.2, -3.0]),
        (270.0, [-3.0, 1.2, 0.0]),
    ],
)
def test_azimuth_runs_from_plus_z_toward_plus_x(azimuth_deg, expected):
    position = orbit_position(azimuth_deg, 3.0, 1.2, CENTER)
    assert np.allclose(position, expected, atol=1e-12)


def test_the_orbit_follows_the_center_and_the_height_is_the_camera_y_coordinate():
    position = orbit_position(90.0, 2.0, 1.5, [0.5, 0.9, -0.25])
    assert np.allclose(position, [2.5, 1.5, -0.25], atol=1e-12)


def test_a_camera_in_front_of_the_body_has_the_hand_derived_pose():
    # At (0, 0.9, 3) the camera looks along -z with +y up: image right is world +x and image
    # down is world -y, so R is a half turn about x. The world origin lies 3 m ahead, so t_z = 3.
    position = orbit_position(0.0, 3.0, 0.9, CENTER)
    rotation, translation = look_at_extrinsics(position, CENTER)
    assert np.allclose(rotation, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], atol=1e-12)
    assert np.allclose(translation, [0.0, 0.9, 3.0], atol=1e-12)


def test_a_camera_on_the_left_of_the_body_has_the_hand_derived_pose():
    # At (3, 0.9, 0) the camera looks along -x: image right is world -z, image down is world -y.
    position = orbit_position(90.0, 3.0, 0.9, CENTER)
    rotation, translation = look_at_extrinsics(position, CENTER)
    assert np.allclose(rotation, [[0, 0, -1], [0, -1, 0], [-1, 0, 0]], atol=1e-12)
    assert np.allclose(translation, [0.0, 0.9, 3.0], atol=1e-12)


def test_look_at_poses_are_proper_rotations_that_center_the_target_and_stay_upright():
    rng = rng_for(5, 1)
    K = intrinsics(64, 64.0)
    position = rng.uniform(-4.0, 4.0, size=(60, 3)) + [0.0, 1.0, 0.0]
    target = rng.uniform(-0.5, 0.5, size=(60, 3))
    rotation, translation = look_at_extrinsics(position, target)
    assert np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    # The camera center -R^T t is the position the pose was built for.
    assert np.allclose(-np.einsum("nji,nj->ni", rotation, translation), position, atol=1e-12)
    for view in range(len(position)):
        pose = (K, rotation[view], translation[view])
        pixel, depth = project_points(target[view], *pose)
        assert np.allclose(pixel, K[:2, 2], atol=1e-9)  # the target is on the optical axis
        assert depth == pytest.approx(np.linalg.norm(target[view] - position[view]), abs=1e-12)
        # World up stays image up: a point above the target has the same column and a lower row.
        above, _ = project_points(target[view] + [0.0, 0.1, 0.0], *pose)
        assert above[0] == pytest.approx(pixel[0], abs=1e-9)
        assert above[1] < pixel[1]


def test_look_at_refuses_a_camera_on_its_target_or_straight_above_or_below_it():
    with pytest.raises(ValueError, match="own position"):
        look_at_extrinsics([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="straight up or down"):
        look_at_extrinsics([0.0, 3.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="straight up or down"):
        look_at_extrinsics([0.0, -3.0, 0.0], [0.0, 0.0, 0.0])


# Rigs


@pytest.mark.parametrize("n_cameras", [1, 2, 4])
def test_a_rig_holds_the_configured_number_of_cameras_in_the_layout_of_a_shard(n_cameras):
    rig = sample_rig(rng_for(1, 0), _camera_section(n_cameras=n_cameras), CENTER)
    assert np.array_equal(rig.K, intrinsics(64, 64.0))
    assert rig.R_true.shape == (n_cameras, 3, 3)
    assert rig.t_true.shape == (n_cameras, 3)
    assert rig.azimuth_deg.shape == rig.distance_m.shape == rig.height_m.shape == (n_cameras,)
    assert rig.lookat.shape == rig.noise_axis.shape == (n_cameras, 3)


def test_each_camera_stands_where_its_azimuth_distance_and_height_say():
    center = np.array([0.1, 0.95, -0.05])
    for body in range(20):
        rig = sample_rig(rng_for(1, 2, body), _camera_section(), center)
        position = -np.einsum("nji,nj->ni", rig.R_true, rig.t_true)
        horizontal = position[:, [0, 2]] - center[[0, 2]]
        assert np.allclose(np.hypot(horizontal[:, 0], horizontal[:, 1]), rig.distance_m, atol=1e-12)
        assert np.allclose(position[:, 1], rig.height_m, atol=1e-12)
        azimuth = np.degrees(np.arctan2(horizontal[:, 0], horizontal[:, 1])) % 360.0
        assert np.all(np.abs((azimuth - rig.azimuth_deg + 180.0) % 360.0 - 180.0) < 1e-9)


def test_every_camera_aims_at_the_center_within_the_jitter():
    jitter = 0.05
    offsets = []
    for body in range(20):
        rig = sample_rig(rng_for(1, 3, body), _camera_section(lookat_jitter_m=jitter), CENTER)
        offsets.append(rig.lookat - CENTER)
        assert np.all(np.abs(rig.lookat - CENTER) <= jitter)
        for view in range(4):
            pixel, depth = project_points(
                rig.lookat[view], rig.K, rig.R_true[view], rig.t_true[view]
            )
            assert np.allclose(pixel, rig.K[:2, 2], atol=1e-9)
            assert depth > 0.0
    # The jitter is drawn per camera, on both sides of the center, up to its bound.
    offsets = np.concatenate(offsets)
    assert np.all(offsets.min(axis=0) < -0.04)
    assert np.all(offsets.max(axis=0) > 0.04)


def test_without_jitter_every_camera_looks_exactly_at_the_center():
    rig = sample_rig(rng_for(1, 4), _camera_section(lookat_jitter_m=0.0), CENTER)
    assert np.array_equal(rig.lookat, np.tile(CENTER, (4, 1)))


def test_distance_and_height_stay_in_their_ranges_and_cover_them():
    distances, heights = [], []
    for body in range(300):
        rig = sample_rig(rng_for(1, 5, body), _camera_section(), CENTER)
        distances.extend(rig.distance_m)
        heights.extend(rig.height_m)
    assert 2.5 <= min(distances) < 2.6
    assert 3.9 < max(distances) <= 4.0
    assert 0.8 <= min(heights) < 0.9
    assert 1.7 < max(heights) <= 1.8


def test_azimuths_are_uniform_around_the_body():
    azimuths = np.concatenate(
        [
            sample_rig(rng_for(1, 6, body), _camera_section(), CENTER).azimuth_deg
            for body in range(1500)
        ]
    )
    assert np.all((azimuths >= 0.0) & (azimuths < 360.0))
    radians = np.radians(azimuths)
    assert abs(np.mean(np.cos(radians))) < 0.05
    assert abs(np.mean(np.sin(radians))) < 0.05
    quarters = np.histogram(azimuths, bins=4, range=(0.0, 360.0))[0] / azimuths.size
    assert np.all(np.abs(quarters - 0.25) < 0.03)


def test_a_rig_depends_only_on_the_generator_state():
    first = sample_rig(rng_for(1, 7), _camera_section(), CENTER)
    again = sample_rig(rng_for(1, 7), _camera_section(), CENTER)
    other = sample_rig(rng_for(1, 8), _camera_section(), CENTER)
    for name in ("K", "R_true", "t_true", "azimuth_deg", "distance_m", "height_m", "lookat"):
        assert np.array_equal(getattr(first, name), getattr(again, name))
    assert np.array_equal(first.noise_axis, again.noise_axis)
    assert not np.array_equal(first.azimuth_deg, other.azimuth_deg)
    assert not np.array_equal(first.noise_axis, other.noise_axis)


def test_the_rig_stores_one_unit_noise_axis_per_camera():
    rig = sample_rig(rng_for(1, 10), _camera_section(), CENTER)
    assert np.allclose(np.linalg.norm(rig.noise_axis, axis=1), 1.0, atol=1e-12)
    assert len({tuple(axis) for axis in rig.noise_axis}) == 4


def test_a_rig_refuses_a_center_that_is_not_one_point():
    with pytest.raises(ValueError, match="center"):
        sample_rig(rng_for(1, 9), _camera_section(), [0.0, 0.9])
    with pytest.raises(ValueError, match="center"):
        sample_rig(rng_for(1, 9), _camera_section(), np.zeros((2, 3)))


# Minimum separation


@pytest.mark.parametrize(
    ("azimuths", "minimum", "expected"),
    [
        ([0.0, 20.0], 20.0, True),  # the boundary is inclusive
        ([0.0, 19.9], 20.0, False),
        ([350.0, 5.0], 20.0, False),  # 15 degrees apart across 0 and 360
        ([350.0, 10.0], 20.0, True),
        ([-10.0, 10.0], 20.0, True),  # a negative azimuth is the same direction as 350
        ([10.0, 350.0, 180.0], 20.0, True),
        ([359.0, 100.0, 200.0, 1.0], 20.0, False),
        ([0.0, 90.0, 180.0, 270.0], 90.0, True),
        ([0.0, 90.0, 180.0, 270.0], 90.5, False),
        ([45.0], 400.0, True),  # one azimuth has no pair
        ([], 400.0, True),
    ],
)
def test_the_separation_rule_measures_the_shorter_way_around_the_circle(
    azimuths, minimum, expected
):
    assert azimuths_are_separated(azimuths, minimum) is expected


def test_separation_is_enforced():
    count = 300
    minimum = 20.0
    constrained = _camera_section(min_separation_deg=minimum)
    unconstrained = _camera_section(min_separation_deg=0.0)
    kept = [
        _smallest_gap_deg(sample_rig(rng_for(1, 11, body), constrained, CENTER).azimuth_deg)
        for body in range(count)
    ]
    free = [
        _smallest_gap_deg(sample_rig(rng_for(1, 11, body), unconstrained, CENTER).azimuth_deg)
        for body in range(count)
    ]
    assert min(kept) >= minimum
    # Four cameras drawn freely break a 20 degree rule in about half of the rigs, so the check
    # above would fail if the sampler ignored the rule.
    assert sum(gap < minimum for gap in free) > count // 4


def test_a_wide_separation_holds_for_every_pair_of_the_rig():
    section = _camera_section(n_cameras=3, min_separation_deg=100.0)
    for body in range(100):
        azimuth = sample_rig(rng_for(1, 12, body), section, CENTER).azimuth_deg
        assert _smallest_gap_deg(azimuth) >= 100.0


def test_one_camera_needs_no_separation():
    section = _camera_section(n_cameras=1, min_separation_deg=180.0)
    assert sample_rig(rng_for(1, 15), section, CENTER).azimuth_deg.shape == (1,)


@pytest.mark.parametrize(
    ("n_cameras", "separation"), [(4, 90.0), (4, 100.0), (2, 180.0), (3, 120.0)]
)
def test_cameras_that_cannot_fit_around_the_circle_are_a_configuration_error(n_cameras, separation):
    section = _camera_section(n_cameras=n_cameras, min_separation_deg=separation)
    # The rule is checked before any draw, so the message is about the rule and the error is fast.
    with pytest.raises(ConfigError, match="must stay below 360") as error:
        sample_rig(rng_for(1, 13), section, CENTER)
    assert error.value.key == "camera.min_separation_deg"


def test_sampling_gives_up_with_a_configuration_error_when_the_cameras_almost_cannot_fit(
    monkeypatch,
):
    # Four cameras 89.9 degrees apart have a chance of about 1e-9 per draw. A low limit keeps the
    # test fast; the real limit is 100,000 draws.
    monkeypatch.setattr(camera_module, "_MAX_RIG_ATTEMPTS", 50)
    section = _camera_section(min_separation_deg=89.9)
    with pytest.raises(ConfigError, match="lower camera.min_separation_deg") as error:
        sample_rig(rng_for(1, 14), section, CENTER)
    assert error.value.key == "camera.min_separation_deg"


# Placement noise


@pytest.mark.parametrize("noise_deg", [0.0, 0.5, 2.0, 5.0, 45.0, 90.0, 179.0])
def test_the_rotation_angle_equals_the_noise_level(noise_deg):
    rig = sample_rig(rng_for(1, 16), _camera_section(), CENTER)
    true_rotation = rig.R_true[0]
    axes = random_unit_vectors(rng_for(1, 17), 200)
    given = given_rotation(true_rotation, axes, noise_deg)
    # The turn from the true to the given rotation is R_given R_true^T, and it is the noise.
    turn = given @ true_rotation.T
    assert np.allclose(rotation_angle_deg(turn), noise_deg, atol=1e-9, rtol=0.0)
    # It is exactly the noise rotation in the camera frame, R_given = R_noise @ R_true. The other
    # order, R_true @ R_noise, has the same angle but turns about a different axis.
    assert np.allclose(turn, noise_rotation(axes, noise_deg), atol=1e-12)
    # OpenCV measures the same angle as the length of the rotation vector (its R to vector
    # conversion loses digits near 180 degrees, so only the smaller angles are compared).
    if noise_deg < 100.0:
        for index in range(0, 200, 10):
            vector, _ = cv2.Rodrigues(turn[index])
            assert np.degrees(np.linalg.norm(vector)) == pytest.approx(noise_deg, abs=1e-8)
    # The same turn is the OpenCV rotation for the vector axis * angle.
    for index in range(0, 200, 10):
        expected, _ = cv2.Rodrigues(axes[index] * np.radians(noise_deg))
        assert np.allclose(noise_rotation(axes[index], noise_deg), expected, atol=1e-12)


def test_the_rotation_angle_of_a_rig_camera_equals_the_noise_level_of_every_cell():
    rig = sample_rig(rng_for(1, 26), _camera_section(), CENTER)
    for noise_deg in (0.0, 2.0, 5.0):
        given = given_rotation(rig.R_true, rig.noise_axis, noise_deg)
        turn = given @ np.swapaxes(rig.R_true, -1, -2)
        assert np.allclose(rotation_angle_deg(turn), noise_deg, atol=1e-9, rtol=0.0)
        # The given rotation is still a proper rotation.
        assert np.allclose(given @ np.swapaxes(given, -1, -2), np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(given), 1.0, atol=1e-12)


def test_zero_degrees_is_the_identity():
    axes = random_unit_vectors(rng_for(1, 18), 50)
    assert np.array_equal(noise_rotation(axes, 0.0), np.broadcast_to(np.eye(3), (50, 3, 3)))
    rig = sample_rig(rng_for(1, 19), _camera_section(), CENTER)
    given = given_rotation(rig.R_true, rig.noise_axis, 0.0)
    assert np.array_equal(given, rig.R_true)
    # The encoding that the model receives is unchanged as well.
    assert np.array_equal(encode_camera(given, rig.t_true), encode_camera(rig.R_true, rig.t_true))


def test_a_positive_noise_level_changes_the_rotation_code_but_not_the_translation_code():
    rig = sample_rig(rng_for(1, 27), _camera_section(), CENTER)
    quiet = encode_camera(rig.R_true, rig.t_true)
    noisy = encode_camera(given_rotation(rig.R_true, rig.noise_axis, 2.0), rig.t_true)
    assert not np.allclose(noisy[:, :6], quiet[:, :6], atol=1e-4)
    assert np.array_equal(noisy[:, 6:], quiet[:, 6:])


def test_the_noise_rotation_turns_right_handed_about_its_axis():
    quarter_turn = noise_rotation([0.0, 0.0, 1.0], 90.0)
    assert np.allclose(quarter_turn @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(quarter_turn @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-12)
    axis = np.array([1.0, 2.0, -2.0]) / 3.0
    turn = noise_rotation(axis, 37.0)
    assert np.allclose(turn @ axis, axis, atol=1e-12)  # the axis stays where it is
    assert np.allclose(turn @ turn.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(turn) == pytest.approx(1.0, abs=1e-12)
    # An axis that is not a unit vector gives the same turn.
    assert np.allclose(noise_rotation([2.0, 4.0, -4.0], 37.0), turn, atol=1e-12)
    # A negative angle undoes a positive one.
    assert np.allclose(noise_rotation(axis, -37.0), turn.T, atol=1e-12)


def test_the_noise_rotation_broadcasts_over_bodies_views_and_levels():
    axes = random_unit_vectors(rng_for(1, 20), 12).reshape(3, 4, 3)  # 3 bodies by 4 views
    levels = np.array([0.0, 2.0, 5.0])
    one_level = noise_rotation(axes, 2.0)
    assert one_level.shape == (3, 4, 3, 3)
    per_body = noise_rotation(axes, levels[:, None])  # one level for each body
    assert per_body.shape == (3, 4, 3, 3)
    assert np.allclose(per_body[2, 1], noise_rotation(axes[2, 1], 5.0), atol=1e-15)
    assert np.allclose(one_level[1, 3], noise_rotation(axes[1, 3], 2.0), atol=1e-15)
    assert noise_rotation(axes[0, 0], 2.0).shape == (3, 3)


def test_the_noise_rotation_refuses_an_axis_with_no_direction():
    with pytest.raises(ValueError, match="length above zero"):
        noise_rotation([0.0, 0.0, 0.0], 5.0)
    with pytest.raises(ValueError, match="finite"):
        noise_rotation([np.nan, 0.0, 1.0], 5.0)
    with pytest.raises(ValueError, match="length 3"):
        noise_rotation([0.0, 1.0], 5.0)


def test_random_axes_are_unit_vectors_spread_evenly_over_the_sphere():
    axes = random_unit_vectors(rng_for(1, 21), 20000)
    assert axes.shape == (20000, 3)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
    assert np.all(np.abs(axes.mean(axis=0)) < 0.02)
    # On a uniform sphere every coordinate is uniform on [-1, 1] (Archimedes), so its mean square
    # is 1/3 and its mean fourth power is 1/5. The fourth power tells a uniform sphere from, for
    # example, normalized points of a cube, which also have a mean square of 1/3.
    assert np.allclose((axes**2).mean(axis=0), 1.0 / 3.0, atol=0.01)
    assert np.allclose((axes**4).mean(axis=0), 1.0 / 5.0, atol=0.01)


@pytest.mark.parametrize("angle_deg", [0.0, 1e-6, 0.01, 2.0, 90.0, 179.99, 180.0])
def test_the_rotation_angle_stays_accurate_for_tiny_and_near_half_turn_angles(angle_deg):
    rotation = noise_rotation([0.3, -0.5, 0.8], angle_deg)
    assert rotation_angle_deg(rotation) == pytest.approx(angle_deg, abs=1e-9)


# 6D encoding


def test_the_6d_encoding_is_the_first_two_columns_one_after_the_other():
    rotation = noise_rotation([0.0, 0.0, 1.0], 90.0)  # columns (0, 1, 0), (-1, 0, 0), (0, 0, 1)
    assert np.allclose(rotation_to_6d(rotation), [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], atol=1e-12)
    generic = np.arange(9.0).reshape(3, 3)  # the entries show which column goes where
    assert np.array_equal(rotation_to_6d(generic), [0.0, 3.0, 6.0, 1.0, 4.0, 7.0])


def test_the_6d_encoding_keeps_the_whole_rotation():
    rng = rng_for(1, 22)
    random_turns = noise_rotation(random_unit_vectors(rng, 500), rng.uniform(0.0, 180.0, 500))
    poses = np.concatenate(
        [sample_rig(rng_for(1, 28, body), _camera_section(), CENTER).R_true for body in range(50)]
    )
    for rotations in (random_turns, poses):
        code = rotation_to_6d(rotations)
        assert code.shape == (len(rotations), 6)
        assert np.allclose(_rotation_from_6d(code), rotations, atol=1e-12)


def test_the_camera_encoding_appends_the_translation_scaled_by_one_fifth():
    assert CAMERA_ENCODING_DIM == 9
    assert TRANSLATION_SCALE_M == 5.0
    for body in range(50):
        rig = sample_rig(rng_for(1, 23, body), _camera_section(), CENTER)
        code = encode_camera(rig.R_true, rig.t_true)
        assert code.shape == (4, CAMERA_ENCODING_DIM)
        assert np.array_equal(code[:, :6], rotation_to_6d(rig.R_true))
        assert np.array_equal(code[:, 6:], rig.t_true / 5.0)
        assert np.all(np.abs(code[:, 6:]) < 1.0)  # inputs of order one for the network
    single = encode_camera(rig.R_true[0], rig.t_true[0])
    assert single.shape == (CAMERA_ENCODING_DIM,)
    assert np.array_equal(single, code[0])
    batch = encode_camera(np.stack([rig.R_true] * 3), np.stack([rig.t_true] * 3))
    assert batch.shape == (3, 4, CAMERA_ENCODING_DIM)
    assert np.array_equal(batch[2], code)
    # One translation serves every rotation of the batch.
    shared = encode_camera(rig.R_true, rig.t_true[0])
    assert shared.shape == (4, CAMERA_ENCODING_DIM)
    assert np.array_equal(shared[:, 6:], np.tile(rig.t_true[0] / 5.0, (4, 1)))


def test_the_6d_encoding_refuses_something_that_is_not_a_matrix_of_3_by_3():
    with pytest.raises(ValueError, match="3 by 3"):
        rotation_to_6d(np.eye(2))
    with pytest.raises(ValueError, match="3 by 3"):
        rotation_to_6d(np.zeros(9))
    with pytest.raises(ValueError, match="length 3"):
        encode_camera(np.eye(3), np.zeros(2))


# Projection


def test_projection_of_a_known_point():
    K = intrinsics(128, 128.0)  # focal length 128 px, principal point (63.5, 63.5)
    rotation, translation = look_at_extrinsics(orbit_position(0.0, 3.0, 0.9, CENTER), CENTER)
    points = np.array(
        [
            [0.0, 0.9, 0.0],  # the target: on the optical axis, 3 m ahead
            [0.3, 0.9, 0.0],  # 0.3 m to the right of the axis (world +x for a camera at +z)
            [0.0, 1.5, 0.0],  # 0.6 m above the target
            [0.3, 0.9, 1.0],  # 1 m nearer to the camera: depth 2 m
        ]
    )
    pixels, depth = project_points(points, K, rotation, translation)
    assert np.allclose(depth, [3.0, 3.0, 3.0, 2.0], atol=1e-12)
    # column = 63.5 + 128 x / depth, row = 63.5 - 128 y / depth (image rows run downward).
    expected = [[63.5, 63.5], [76.3, 63.5], [63.5, 37.9], [82.7, 63.5]]
    assert np.allclose(pixels, expected, atol=1e-9)
    # Seen from the left of the body (azimuth 90), the body's back (-z) is on the image right.
    rotation, translation = look_at_extrinsics(orbit_position(90.0, 3.0, 0.9, CENTER), CENTER)
    pixels, depth = project_points([[0.0, 0.9, -0.3], [0.0, 0.9, 0.0]], K, rotation, translation)
    assert np.allclose(depth, [3.0, 3.0], atol=1e-12)
    assert np.allclose(pixels, [[76.3, 63.5], [63.5, 63.5]], atol=1e-9)


def test_projection_follows_the_pinhole_law_for_size_and_distance():
    K = intrinsics(128, 128.0)
    ends = np.array([[-0.5, 0.9, 0.0], [0.5, 0.9, 0.0]])  # a 1 m bar across the view
    for distance, span in ((2.0, 64.0), (4.0, 32.0), (8.0, 16.0)):
        rotation, translation = look_at_extrinsics(
            orbit_position(0.0, distance, 0.9, CENTER), CENTER
        )
        pixels, _ = project_points(ends, K, rotation, translation)
        assert pixels[1, 0] - pixels[0, 0] == pytest.approx(span, abs=1e-9)
        assert pixels[0, 1] == pytest.approx(pixels[1, 1], abs=1e-9)


def test_projection_agrees_with_opencv_for_sampled_rigs():
    rng = rng_for(1, 24)
    for body in range(20):
        rig = sample_rig(rng_for(1, 25, body), _camera_section(), CENTER)
        points = CENTER + rng.uniform(-1.0, 1.0, size=(30, 3))
        for view in range(4):
            pose = (rig.K, rig.R_true[view], rig.t_true[view])
            pixels, depth = project_points(points, *pose)
            assert np.all(depth > 0.0)
            assert np.allclose(pixels, _opencv_pixels(points, *pose), atol=1e-9)


def test_points_at_or_behind_the_camera_plane_have_no_pixels():
    K = intrinsics(64, 64.0)
    rotation, translation = look_at_extrinsics(orbit_position(0.0, 3.0, 0.9, CENTER), CENTER)
    points = np.array(
        [
            [0.0, 0.9, 3.0],  # on the camera: depth 0
            [0.5, 0.9, 3.0],  # on the camera plane, off the axis: depth 0
            [0.0, 0.9, 4.0],  # behind the camera: depth -1
            [0.0, 0.9, 2.0],  # in front: depth 1
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a point without an image must not raise a division warning
        pixels, depth = project_points(points, K, rotation, translation)
    assert np.allclose(depth, [0.0, 0.0, -1.0, 1.0], atol=1e-12)
    assert np.all(np.isnan(pixels[:3]))
    assert np.allclose(pixels[3], K[:2, 2], atol=1e-9)


def test_projection_keeps_the_leading_shape_of_the_points():
    K = intrinsics(64, 64.0)
    rotation, translation = look_at_extrinsics(orbit_position(0.0, 3.0, 0.9, CENTER), CENTER)
    grid = np.zeros((2, 5, 3)) + [0.0, 0.9, 0.0]
    pixels, depth = project_points(grid, K, rotation, translation)
    assert pixels.shape == (2, 5, 2)
    assert depth.shape == (2, 5)
    pixel, one_depth = project_points([0.0, 0.9, 0.0], K, rotation, translation)
    assert pixel.shape == (2,)
    assert np.shape(one_depth) == ()
    assert np.allclose(pixel, K[:2, 2], atol=1e-9)


def test_projection_refuses_points_or_matrices_of_the_wrong_shape():
    K = intrinsics(64, 64.0)
    rotation, translation = look_at_extrinsics(orbit_position(0.0, 3.0, 0.9, CENTER), CENTER)
    with pytest.raises(ValueError, match="length 3"):
        project_points(np.zeros((4, 2)), K, rotation, translation)
    with pytest.raises(ValueError, match="3 by 3"):
        project_points(np.zeros((4, 3)), K[:2], rotation, translation)
    with pytest.raises(ValueError, match="finite"):
        project_points([[np.inf, 0.0, 0.0]], K, rotation, translation)
