"""Frozen pose-level train/val/test split (contract: specs/002-data-pipeline/contracts).

Split files are ``split_version`` 2 (spec 007): they pin the dataset's manifest *content*
(the manifest without its generating ``code``) and record that ``code`` for information.
``split_version`` 1 files (raw manifest bytes) stay readable, with their strict check.
Torch-free (used by the ``strike-a-pose-data split`` CLI).
"""

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from strike_a_pose.datagen.io import dumps, read_json, write_json

SPLITS = ("train", "val", "test")
SPLIT_VERSION = 2  # written
LEGACY_SPLIT_VERSION = 1  # read only: pins the raw manifest bytes
ALGORITHM = "pose-permutation-v1"
UNIT = "pose_index"
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
_COMMON_KEYS = frozenset(
    {"split_version", "algorithm", "unit", "dataset_name", "seed", "fractions", "poses", "counts"}
)
SPLIT_KEYS = {
    LEGACY_SPLIT_VERSION: _COMMON_KEYS | {"manifest_sha256"},
    SPLIT_VERSION: _COMMON_KEYS | {"manifest_content_sha256", "manifest_code"},
}


class SplitError(ValueError):
    """Invalid split request or split file."""


class SplitCommitWarning(UserWarning):
    """A split file matches the dataset's manifest content but records other generating code.

    The manifest ``code`` (commit, dirty flag) differs from the split's ``manifest_code``;
    the split is used. ``warnings.simplefilter("error", SplitCommitWarning)`` makes it fatal.
    """


def parse_sample_id(sample_id: str) -> tuple[int, str]:
    """``"00012_Front_0"`` → ``(12, "Front_0")`` (001: ``f"{pose:05d}_{camera}"``)."""
    head, sep, cam = sample_id.partition("_")
    if not sep or not cam or not head.isdigit():
        raise ValueError(f"sample id {sample_id!r} is not '<pose_index>_<camera_name>'")
    return int(head), cam


def manifest_sha256(dataset_dir: Path | str) -> str:
    """SHA-256 of the dataset's ``manifest.json`` bytes (raw; includes the generating code)."""
    return hashlib.sha256((Path(dataset_dir) / "manifest.json").read_bytes()).hexdigest()


def manifest_content_sha256(manifest: dict) -> str:
    """SHA-256 of a parsed manifest without its ``code`` key, as the 001 serializer writes it.

    Independent of the generating commit and of on-disk formatting; for a manifest written
    by 001 it is the SHA-256 of the ``manifest.json`` 001 would write without ``code``.
    """
    return hashlib.sha256(dumps({k: v for k, v in manifest.items() if k != "code"})).hexdigest()


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


def make_split(manifest: dict, seed: int, fractions: Sequence[float] = DEFAULT_FRACTIONS) -> dict:
    """Assign whole poses to train/val/test (algorithm ``pose-permutation-v1``).

    Returns a ``split_version`` 2 dict; the content hash and ``manifest_code`` come from
    ``manifest`` itself.
    """
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
        "manifest_content_sha256": manifest_content_sha256(manifest),
        "manifest_code": dict(manifest["code"]),
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
    """Read a split file (``split_version`` 1 or 2) and check its structure."""
    try:
        split = read_json(Path(path))
    except (OSError, ValueError) as e:
        raise SplitError(f"cannot read split file {path}: {e}") from e
    if not isinstance(split, dict) or "split_version" not in split:
        raise SplitError(f"{path}: not a split file (no split_version)")
    version = split["split_version"]
    if type(version) is not int or version not in SPLIT_KEYS:
        raise SplitError(
            f"{path}: split_version {version!r} unsupported (expected one of {sorted(SPLIT_KEYS)})"
        )
    if set(split) != SPLIT_KEYS[version]:
        raise SplitError(f"{path}: not a split file (keys mismatch for split_version {version})")
    if split["unit"] != UNIT:
        raise SplitError(f"{path}: unit {split['unit']!r} unsupported (expected {UNIT!r})")
    if version == SPLIT_VERSION and not isinstance(split["manifest_code"], dict):
        raise SplitError(f"{path}: manifest_code must be an object")
    if set(split["poses"]) != set(SPLITS):
        raise SplitError(f"{path}: poses must have keys {SPLITS}")
    seen: set[int] = set()
    for name in SPLITS:
        cur = set(split["poses"][name])
        if cur & seen or len(cur) != len(split["poses"][name]):
            raise SplitError(f"{path}: pose indices repeated across or within splits")
        seen |= cur
    return split


def check_split_for(
    split: dict, raw_sha256: str, content_sha256: str, pose_indices: Iterable[int]
) -> list[str]:
    """Problems using ``split`` with a dataset of these manifest hashes and poses.

    ``split_version`` 2 compares the manifest content hash; legacy 1 the raw manifest
    bytes hash. Pose coverage must be exact for both.
    """
    errs = []
    name = split["dataset_name"]
    if split["split_version"] == LEGACY_SPLIT_VERSION:
        if split["manifest_sha256"] != raw_sha256:
            errs.append(
                f"split was made for manifest {split['manifest_sha256'][:12]}… ({name}), "
                f"dataset manifest is {raw_sha256[:12]}… (split_version 1 pins the raw manifest "
                "bytes, including the generating code commit; for the same dataset regenerated "
                "by other code, rewrite the split with `strike-a-pose-data split DATASET_DIR "
                "--seed SEED --out SPLIT_FILE --overwrite`, which writes split_version 2)"
            )
    elif split["manifest_content_sha256"] != content_sha256:
        errs.append(
            f"split was made for manifest content {split['manifest_content_sha256'][:12]}… "
            f"({name}), dataset manifest content is {content_sha256[:12]}…"
        )
    covered = {p for s in SPLITS for p in split["poses"][s]}
    if covered != set(pose_indices):
        errs.append("split pose indices do not cover exactly the dataset's poses")
    return errs


def check_split_code(split: dict, manifest_code: object) -> str | None:
    """Warning text when a ``split_version`` 2 split records other generating code.

    ``None`` when ``manifest_code`` (the dataset manifest's ``code``) equals the split's
    ``manifest_code``, and for legacy v1 splits (their raw hash already covers the code).
    Call after :func:`check_split_for` passed (the content is then identical).
    """
    if split["split_version"] == LEGACY_SPLIT_VERSION or split["manifest_code"] == manifest_code:
        return None
    return (
        f"dataset generated by code {_compact(manifest_code)}, split made from a dataset "
        f"generated by code {_compact(split['manifest_code'])}; manifest content is identical, "
        "so the split is used (rewriting the split with `strike-a-pose-data split ... "
        "--overwrite` refreshes manifest_code)"
    )


def _compact(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
