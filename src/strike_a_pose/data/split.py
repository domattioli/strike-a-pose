"""Frozen pose-level train/val/test split (contract: specs/002-data-pipeline/contracts).

Torch-free (used by the ``strike-a-pose-data split`` CLI).
"""

import hashlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from strike_a_pose.datagen.io import read_json, write_json

SPLITS = ("train", "val", "test")
SPLIT_VERSION = 1
ALGORITHM = "pose-permutation-v1"
UNIT = "pose_index"
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_KEYS = frozenset({
    "split_version", "algorithm", "unit", "dataset_name", "manifest_sha256", "seed",
    "fractions", "poses", "counts",
})  # fmt: skip


class SplitError(ValueError):
    """Invalid split request or split file."""


def parse_sample_id(sample_id: str) -> tuple[int, str]:
    """``"00012_Front_0"`` → ``(12, "Front_0")`` (001: ``f"{pose:05d}_{camera}"``)."""
    head, sep, cam = sample_id.partition("_")
    if not sep or not cam or not head.isdigit():
        raise ValueError(f"sample id {sample_id!r} is not '<pose_index>_<camera_name>'")
    return int(head), cam


def manifest_sha256(dataset_dir: Path | str) -> str:
    """SHA-256 of the dataset's ``manifest.json`` bytes."""
    return hashlib.sha256((Path(dataset_dir) / "manifest.json").read_bytes()).hexdigest()


def _check_fractions(fractions: Sequence[float]) -> dict[str, float]:
    if len(fractions) != 3:
        raise SplitError(f"need 3 fractions (train val test), got {len(fractions)}")
    fr = dict(zip(SPLITS, (float(f) for f in fractions), strict=True))
    if any(not math.isfinite(f) or f < 0 for f in fr.values()):
        raise SplitError(f"fractions must be finite and >= 0, got {fr}")
    if abs(sum(fr.values()) - 1.0) > 1e-6:
        raise SplitError(f"fractions must sum to 1, got {sum(fr.values())}")
    if fr["train"] <= 0:
        raise SplitError("train fraction must be > 0")
    return fr


def make_split(
    manifest: dict,
    manifest_sha256: str,
    seed: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict:
    """Assign whole poses to train/val/test (algorithm ``pose-permutation-v1``)."""
    fr = _check_fractions(fractions)
    per_pose: dict[int, int] = {}
    for s in manifest["samples"]:
        p, _ = parse_sample_id(s["id"])
        per_pose[p] = per_pose.get(p, 0) + 1
    poses = sorted(per_pose)
    n = len(poses)
    sizes = {}
    for name in ("test", "val"):
        k = round(fr[name] * n)
        sizes[name] = max(k, 1) if fr[name] > 0 else 0
    n_train = n - sizes["test"] - sizes["val"]
    if n_train < 1:
        raise SplitError(
            f"{n} pose(s) cannot give train >= 1 with test={sizes['test']}, val={sizes['val']}"
            " (minimum 1 pose per non-zero fraction); lower the val/test fractions"
        )
    perm = np.random.default_rng(seed).permutation(np.array(poses, dtype=np.int64)).tolist()
    nt, nv = sizes["test"], sizes["val"]
    assigned = {"test": perm[:nt], "val": perm[nt : nt + nv], "train": perm[nt + nv :]}
    assigned = {k: sorted(int(p) for p in v) for k, v in assigned.items()}
    return {
        "split_version": SPLIT_VERSION,
        "algorithm": ALGORITHM,
        "unit": UNIT,
        "dataset_name": manifest["dataset_name"],
        "manifest_sha256": manifest_sha256,
        "seed": int(seed),
        "fractions": fr,
        "poses": assigned,
        "counts": {
            k: {"poses": len(v), "samples": sum(per_pose[p] for p in v)}
            for k, v in assigned.items()
        },
    }


def write_split(split: dict, path: Path | str, overwrite: bool = False) -> Path:
    """Write a split file; refuse to replace an existing one unless ``overwrite``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (frozen split); pass --overwrite to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, split)
    return path


def load_split(path: Path | str) -> dict:
    """Read a split file and check its structure (version, keys, disjoint pose sets)."""
    try:
        split = read_json(Path(path))
    except (OSError, ValueError) as e:
        raise SplitError(f"cannot read split file {path}: {e}") from e
    if not isinstance(split, dict) or set(split) != SPLIT_KEYS:
        raise SplitError(f"{path}: not a split file (keys mismatch)")
    if split["split_version"] != SPLIT_VERSION or split["unit"] != UNIT:
        raise SplitError(
            f"{path}: split_version {split['split_version']} / unit {split['unit']!r} "
            f"unsupported (expected {SPLIT_VERSION} / {UNIT!r})"
        )
    if set(split["poses"]) != set(SPLITS):
        raise SplitError(f"{path}: poses must have keys {SPLITS}")
    seen: set[int] = set()
    for name in SPLITS:
        cur = set(split["poses"][name])
        if cur & seen or len(cur) != len(split["poses"][name]):
            raise SplitError(f"{path}: pose indices repeated across or within splits")
        seen |= cur
    return split


def check_split_for(split: dict, manifest_sha256: str, pose_indices: Iterable[int]) -> list[str]:
    """Problems using ``split`` with a dataset (hash and pose coverage)."""
    errs = []
    if split["manifest_sha256"] != manifest_sha256:
        errs.append(
            f"split was made for manifest {split['manifest_sha256'][:12]}… "
            f"({split['dataset_name']}), dataset manifest is {manifest_sha256[:12]}…"
        )
    covered = {p for name in SPLITS for p in split["poses"][name]}
    if covered != set(pose_indices):
        errs.append("split pose indices do not cover exactly the dataset's poses")
    return errs
