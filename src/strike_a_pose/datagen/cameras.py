"""Fixed named camera ring, projection, visibility (research R5/R6)."""

from dataclasses import dataclass

import numpy as np

from .config import CameraRingConfig

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
