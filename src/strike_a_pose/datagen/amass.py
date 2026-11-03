"""AMASS pose enumeration and seeded sampling (research R3/R4)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import MissingAssetError, require_dir


@dataclass(frozen=True)
class PoseRecord:
    index: int
    source_file: str  # POSIX path relative to amass_root
    frame: int
    pose: tuple[float, ...]  # 72 axis-angle values, root zeroed
    source_root_orient: tuple[float, float, float]


def _pose_files(root: Path, subset: str) -> list[Path]:
    d = require_dir(root / subset, f"AMASS sub-collection '{subset}'")
    files = [p for p in d.rglob("*.npz") if p.name != "shape.npz"]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def enumerate_candidates(
    root: Path | str, subsets: tuple[str, ...] | list[str], stride: int
) -> list[tuple[str, int]]:
    root = require_dir(Path(root), "AMASS root directory")
    out: list[tuple[str, int]] = []
    for subset in subsets:
        for p in _pose_files(root, subset):
            with np.load(p, allow_pickle=False) as z:
                if "poses" not in z.files:
                    continue
                n = z["poses"].shape[0]
            rel = p.relative_to(root).as_posix()
            out.extend((rel, f) for f in range(0, n, stride))
    if not out:
        raise MissingAssetError(root, f"AMASS pose files in subsets {list(subsets)}")
    return out


def to_smpl72(poses_row: np.ndarray) -> np.ndarray:
    """AMASS SMPL+H row (>=66 values) -> SMPL 72 with root and hand joints zeroed."""
    out = np.zeros(72, dtype=np.float64)
    out[3:66] = np.asarray(poses_row[3:66], dtype=np.float64)
    return out


def sample_poses(
    root: Path | str, subsets: tuple[str, ...] | list[str], stride: int, n: int, seed: int
) -> list[PoseRecord]:
    root = Path(root)
    cands = enumerate_candidates(root, subsets, stride)
    if n > len(cands):
        raise ValueError(f"num_poses={n} exceeds {len(cands)} candidate frames (stride {stride})")
    picks = np.random.default_rng(seed).choice(len(cands), size=n, replace=False)
    cache: dict[str, np.ndarray] = {}
    records = []
    for i, ci in enumerate(picks.tolist()):
        rel, frame = cands[ci]
        if rel not in cache:
            with np.load(root / rel, allow_pickle=False) as z:
                cache[rel] = np.asarray(z["poses"], dtype=np.float64)
        row = cache[rel][frame]
        records.append(
            PoseRecord(
                index=i, source_file=rel, frame=frame,
                pose=tuple(to_smpl72(row).tolist()),
                source_root_orient=tuple(float(x) for x in row[:3]),
            )
        )  # fmt: skip
    return records
