"""Minimal NumPy SMPL model: loading + linear blend skinning (research R2)."""

import pickle
from pathlib import Path

import numpy as np

from .assets import MissingAssetError, require_file
from .skeleton import NUM_JOINTS

_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")
_PKL_HINT = (
    "The .pkl could not be read without chumpy; convert it once to .npz with the "
    "keys v_template, shapedirs, posedirs, J_regressor, weights, kintree_table, f "
    "(see docs/data-assets.md) and point smpl_model_path at the .npz."
)


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-12, 1.0, theta)
    k = aa / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    kmat = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1).reshape(
        *aa.shape[:-1], 3, 3
    )
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    eye = np.broadcast_to(np.eye(3), kmat.shape)
    return eye + s * kmat + (1.0 - c) * (kmat @ kmat)


def _dense(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


class SMPLModel:
    """SMPL body model. `forward(pose72, betas) -> (verts (V,3), joints (24,3))`."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.v_template = _dense(arrays["v_template"])
        self.shapedirs = _dense(arrays["shapedirs"])
        self.posedirs = _dense(arrays["posedirs"])
        self.J_regressor = _dense(arrays["J_regressor"])
        self.weights = _dense(arrays["weights"])
        self.faces = np.asarray(arrays["f"], dtype=np.int64)
        kt = np.asarray(arrays["kintree_table"]).astype(np.int64)
        parents = kt[0].copy()
        parents[0] = -1
        self.parents = parents
        if self.J_regressor.shape[0] != NUM_JOINTS or len(parents) != NUM_JOINTS:
            raise ValueError(f"expected a {NUM_JOINTS}-joint SMPL model")

    @property
    def num_shape_components(self) -> int:
        """Betas beyond this count are ignored by ``forward``."""
        return int(self.shapedirs.shape[-1])

    @classmethod
    def load(cls, path: Path | str) -> "SMPLModel":
        path = require_file(Path(path), "SMPL model file")
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in _KEYS}
            elif path.suffix == ".pkl":
                with open(path, "rb") as f:
                    raw = pickle.load(f, encoding="latin1")
                arrays = {k: raw[k] for k in _KEYS}
            else:
                raise MissingAssetError(path, "SMPL model (.npz or .pkl)")
        except ModuleNotFoundError as e:
            raise MissingAssetError(path, "readable SMPL model", _PKL_HINT) from e
        except KeyError as e:
            raise MissingAssetError(path, f"SMPL model key {e}") from e
        return cls(arrays)

    def forward(self, pose: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(pose, dtype=np.float64).reshape(NUM_JOINTS, 3)
        betas = np.asarray(betas, dtype=np.float64).ravel()
        nb = min(len(betas), self.shapedirs.shape[-1])
        v_shaped = self.v_template + self.shapedirs[..., :nb] @ betas[:nb]
        j_rest = self.J_regressor @ v_shaped
        rots = rodrigues(pose)
        pose_feature = (rots[1:] - np.eye(3)).ravel()
        v_posed = v_shaped + self.posedirs @ pose_feature

        g = np.zeros((NUM_JOINTS, 4, 4))
        for i in range(NUM_JOINTS):
            local = np.eye(4)
            local[:3, :3] = rots[i]
            p = self.parents[i]
            local[:3, 3] = j_rest[i] - (j_rest[p] if p >= 0 else 0.0)
            g[i] = local if p < 0 else g[p] @ local
        joints = g[:, :3, 3].copy()
        a = g.copy()
        a[:, :3, 3] -= np.einsum("jab,jb->ja", g[:, :3, :3], j_rest)
        t = np.einsum("vj,jab->vab", self.weights, a)
        verts = np.einsum("vab,vb->va", t[:, :3, :3], v_posed) + t[:, :3, 3]
        return verts, joints
