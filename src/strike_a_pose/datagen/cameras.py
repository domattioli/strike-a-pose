"""Fixed named camera ring, projection, visibility (research R5/R6); ring jitter (spec 010)."""

from dataclasses import dataclass, replace

import numpy as np

from .config import CameraJitterConfig, CameraRingConfig

JITTER_STREAM = 2  # SeedSequence tag of the per-sample camera stream (shape uses 1)

RING8_NAMES = (
    "Front_0", "FrontRight_45", "Right_90", "BackRight_135",
    "Back_180", "BackLeft_225", "Left_270", "FrontLeft_315",
)  # fmt: skip


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray  # (3,3)
    R: np.ndarray  # (3,3) world->camera, OpenCV convention
    t: np.ndarray  # (3,)
    width: int
    height: int
    yaw_deg: float

    def to_dict(self) -> dict:
        return {
            "name": self.name, "K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
            "width": self.width, "height": self.height, "yaw_deg": self.yaw_deg,
        }  # fmt: skip


def look_at(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return r, -r @ center


def camera_ring(ring: CameraRingConfig, height: int, width: int) -> list[Camera]:
    k = np.array(
        [[ring.focal_px, 0.0, width / 2.0], [0.0, ring.focal_px, height / 2.0], [0.0, 0.0, 1.0]]
    )
    cams = []
    for i in range(ring.count):
        yaw = 360.0 * i / ring.count
        th = np.deg2rad(yaw)
        d, h = ring.distance_m, ring.height_m
        center = np.array([-d * np.sin(th), h, d * np.cos(th)])
        r, t = look_at(center, np.array([0.0, h, 0.0]))
        name = RING8_NAMES[i] if ring.count == 8 else f"Cam{i}_{round(yaw)}"
        cams.append(Camera(name, k, r, t, width, height, float(yaw)))
    return cams


def jittered_camera(
    ring_cam: Camera, ring: CameraRingConfig, dh: float, dd: float, dp_deg: float
) -> Camera:
    """The ring camera moved by ``dh`` (height) and ``dd`` (distance) at its yaw, aimed at the
    ring's look-at point ``(0, h, 0)``, then tilted down by ``dp_deg`` (no roll; spec 010 FR-005).

    Name, yaw, intrinsics and image size are the ring camera's. Zero deltas give its ``R, t``
    bit for bit (the same ``look_at`` call as ``camera_ring``).
    """
    th = np.deg2rad(ring_cam.yaw_deg)
    d, h = ring.distance_m + dd, ring.height_m + dh
    center = np.array([-d * np.sin(th), h, d * np.cos(th)])
    aim = np.arctan2(dh, d)  # depression angle toward the look-at point
    target = np.array([0.0, h - d * np.tan(aim + np.deg2rad(dp_deg)), 0.0])
    r, t = look_at(center, target)
    return replace(ring_cam, R=r, t=t)


def in_frame(cam: Camera, verts_world: np.ndarray, margin: int) -> bool:
    """Every vertex in front of the camera and projected into [m, W - m) x [m, H - m)."""
    xc, uv = project(cam, verts_world)
    return bool(
        (xc[:, 2] > 1e-6).all()
        and (uv[:, 0] >= margin).all() and (uv[:, 0] < cam.width - margin).all()
        and (uv[:, 1] >= margin).all() and (uv[:, 1] < cam.height - margin).all()
    )  # fmt: skip


def choose_camera(
    ring_cam: Camera,
    ring_index: int,
    verts_world: np.ndarray,
    ring: CameraRingConfig,
    jitter: CameraJitterConfig,
    seed: int,
    pose_index: int,
) -> tuple[Camera, dict]:
    """First in-frame candidate of the sample's own stream, else the ring camera (fallback).

    Returns the used camera and its label record ``{height_m, distance_m, pitch_deg,
    attempts, fallback}`` (deltas of the used camera; zeros on fallback).
    """
    rng = np.random.default_rng([seed, JITTER_STREAM, pose_index, ring_index])
    for attempt in range(1, jitter.max_attempts + 1):
        dh = float(rng.uniform(*jitter.height_m))
        dd = float(rng.uniform(*jitter.distance_m))
        dp = float(rng.uniform(*jitter.pitch_deg))
        cam = jittered_camera(ring_cam, ring, dh, dd, dp)
        if in_frame(cam, verts_world, jitter.margin_px):
            rec = {"height_m": dh, "distance_m": dd, "pitch_deg": dp, "attempts": attempt}
            return cam, rec | {"fallback": False}
    rec = {"height_m": 0.0, "distance_m": 0.0, "pitch_deg": 0.0, "attempts": jitter.max_attempts}
    return ring_cam, rec | {"fallback": True}


def project(cam: Camera, x_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N,3) -> (camera-frame points (N,3), pixel coords (N,2))."""
    xc = np.asarray(x_world, dtype=np.float64) @ cam.R.T + cam.t
    z = np.where(np.abs(xc[:, 2]) < 1e-12, 1e-12, xc[:, 2])
    uv = np.stack(
        [cam.K[0, 0] * xc[:, 0] / z + cam.K[0, 2], cam.K[1, 1] * xc[:, 1] / z + cam.K[1, 2]], axis=1
    )
    return xc, uv


def visibility(xc: np.ndarray, uv: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        (xc[:, 2] > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )  # fmt: skip
