"""PyTorch dataset over a slice-001 dataset directory (contract: specs/002-data-pipeline).

Items are dicts (see ``contracts/sample-contract.md``); labels stay in source units.
"""

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from strike_a_pose.datagen.io import (
    MANIFEST_KEYS,
    SCHEMA_VERSION,
    check_labels,
    read_json,
    validate_dataset,
)
from strike_a_pose.datagen.skeleton import NUM_JOINTS, SKELETON_NAME

from .split import (
    SPLITS,
    SplitCommitWarning,
    SplitError,
    check_split_code,
    check_split_for,
    load_split,
    manifest_content_sha256,
    manifest_sha256,
    parse_sample_id,
)
from .transforms import image_to_tensor, mask_to_tensor


class DatasetError(ValueError):
    """A dataset directory, split file or sample does not satisfy the contracts."""


@dataclass(frozen=True)
class _Entry:
    sample_id: str
    pose_index: int
    camera_name: str
    camera_index: int
    image: str
    mask: str
    labels: str


def _index_manifest(d: Path) -> tuple[dict, list[_Entry], list[str]]:
    """Cheap construction checks (data-model.md); returns manifest, entries, problems."""
    try:
        m = read_json(d / "manifest.json")
    except (OSError, ValueError) as e:
        return {}, [], [f"cannot read manifest: {e}"]
    if not isinstance(m, dict) or set(m) != MANIFEST_KEYS:
        return {}, [], ["manifest keys do not match layout contract v1"]
    if m["schema_version"] != SCHEMA_VERSION:
        return {}, [], [f"schema_version {m['schema_version']} != {SCHEMA_VERSION} (unsupported)"]
    errs = []
    if m["complete"] is not True:
        errs.append("manifest not marked complete (generation unfinished)")
    sk = m["skeleton"]
    if sk.get("name") != SKELETON_NAME or len(sk.get("joints", [])) != NUM_JOINTS:
        errs.append(f"skeleton is not {SKELETON_NAME} with {NUM_JOINTS} joints")
    c, samples = m["counts"], m["samples"]
    if not (c["samples"] == len(samples) == c["poses"] * c["cameras"]):
        errs.append(
            f"counts inconsistent: samples={c['samples']}, listed={len(samples)}, "
            f"poses x cameras={c['poses']}x{c['cameras']}"
        )
    cams = [cam["name"] for cam in m["cameras"]]
    cam_index = {n: i for i, n in enumerate(cams)}
    entries, seen, missing = [], set(), []
    for s in samples:
        sid = s["id"]
        try:
            pose, cam = parse_sample_id(sid)
        except ValueError as e:
            errs.append(str(e))
            continue
        if sid in seen:
            errs.append(f"duplicate sample id {sid}")
        seen.add(sid)
        if cam not in cam_index:
            errs.append(f"{sid}: camera {cam!r} not in manifest cameras")
            continue
        if not all((d / s[k]).is_file() for k in ("image", "mask", "labels")):
            missing.append(sid)
        entries.append(_Entry(sid, pose, cam, cam_index[cam], s["image"], s["mask"], s["labels"]))
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        errs.append(f"{len(missing)} sample(s) with missing files: {shown}")
    return m, entries, errs


class PoseImageDataset(Dataset):
    """Samples of a 001 dataset, optionally restricted to one split of a split file."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str | None = None,
        split_file: str | Path | None = None,
        transform: Callable[[dict], dict] | None = None,
        verify: bool = False,
    ) -> None:
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if split is not None and split_file is None:
            raise ValueError("split requires split_file (no default split path)")
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.split = split
        try:
            m, entries, errs = _index_manifest(self.dataset_dir)
            if not errs:
                img = m["config"]["image"]
                self.image_size = (int(img["height"]), int(img["width"]))
        except (KeyError, TypeError, AttributeError, ValueError) as err:
            msg = f"invalid dataset {self.dataset_dir}: malformed manifest ({err!r})"
            raise DatasetError(msg) from err
        if not errs and verify:
            errs = validate_dataset(self.dataset_dir)
        if errs:
            raise DatasetError(_fmt(f"invalid dataset {self.dataset_dir}", errs))
        self.manifest = m
        self.manifest_sha256 = manifest_sha256(self.dataset_dir)  # raw bytes (incl. code)
        self.manifest_content_sha256 = manifest_content_sha256(m)  # without code (spec 007)
        self.cameras = [c["name"] for c in m["cameras"]]
        if split_file is not None:
            try:
                sp = load_split(split_file)
            except SplitError as e:
                raise DatasetError(str(e)) from e
            poses = {e.pose_index for e in entries}
            errs = check_split_for(sp, self.manifest_sha256, self.manifest_content_sha256, poses)
            if errs:
                raise DatasetError(_fmt(f"split file {split_file} does not fit dataset", errs))
            note = check_split_code(sp, m["code"])
            if note is not None:  # stacklevel 1: one location → shown once per process
                msg = f"split file {split_file}, dataset {self.dataset_dir}: {note}"
                warnings.warn(msg, SplitCommitWarning, stacklevel=1)
            if split is not None:
                keep = set(sp["poses"][split])
                entries = [e for e in entries if e.pose_index in keep]
        self._entries = entries
        self.sample_ids = [e.sample_id for e in entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, i: int) -> dict:
        e = self._entries[i]
        try:
            item = self._read(e)
        except DatasetError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as err:
            raise DatasetError(f"{e.sample_id}: cannot read sample: {err}") from err
        return self.transform(item) if self.transform is not None else item

    def _read(self, e: _Entry) -> dict:
        d = self.dataset_dir
        lab = json.loads((d / e.labels).read_text())
        errs = check_labels(lab, e.sample_id)
        ids = (lab["sample_id"], lab["pose_index"], lab["camera_name"]) if not errs else None
        if ids is not None and ids != (e.sample_id, e.pose_index, e.camera_name):
            errs.append(f"{e.sample_id}: label ids {ids} do not match manifest entry")
        img = np.asarray(Image.open(d / e.image).convert("L"))
        mask = np.asarray(Image.open(d / e.mask).convert("L"))
        for name, a in (("image", img), ("mask", mask)):
            if a.shape != self.image_size:
                errs.append(f"{e.sample_id}: {name} size {a.shape} != {self.image_size}")
        if errs:
            raise DatasetError("; ".join(errs))
        cam = lab["camera"]
        return {
            "image": image_to_tensor(img),
            "mask": mask_to_tensor(mask),
            "keypoints_2d": _f32(lab["keypoints_2d"]),
            "visibility": torch.as_tensor(np.asarray(lab["visibility"], dtype=bool)),
            "joints_3d_cam": _f32(lab["joints_3d_cam"]),
            "joints_3d_world": _f32(lab["joints_3d_world"]),
            "pose": _f32(lab["pose"]),
            "betas": _f32(lab["betas"]),
            "camera": {
                "K": _f32(cam["K"]),
                "R": _f32(cam["R"]),
                "t": _f32(cam["t"]),
                "yaw_deg": _f32(cam["yaw_deg"]),
            },
            "sample_id": e.sample_id,
            "pose_index": e.pose_index,
            "camera_name": e.camera_name,
            "camera_index": e.camera_index,
        }


def _f32(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _fmt(head: str, errs: list[str]) -> str:
    return head + ":\n" + "\n".join(f"  - {x}" for x in errs)
