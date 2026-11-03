"""Dataset layout v1: label/manifest writers, readers, validation (contract)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .skeleton import NUM_JOINTS

SCHEMA_VERSION = 1
LABEL_KEYS = frozenset({
    "schema_version", "sample_id", "pose_index", "camera_name", "pose_source", "pose", "betas",
    "joints_3d_world", "joints_3d_cam", "keypoints_2d", "visibility", "camera",
})  # fmt: skip
MANIFEST_KEYS = frozenset({
    "schema_version", "dataset_name", "complete", "seed", "config", "code", "inputs",
    "skeleton", "cameras", "counts", "samples",
})  # fmt: skip


def _round(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return _round(x.tolist())
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v) for v in x]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, float | np.floating):
        r = round(float(x), 6)
        return 0.0 if r == 0 else r  # normalise -0.0
    if isinstance(x, np.integer):
        return int(x)
    return x


def dumps(obj: Any) -> bytes:
    return (json.dumps(_round(obj), sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, obj: Any) -> str:
    """Write JSON (atomic temp + rename); return its sha256."""
    data = dumps(obj)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_manifest(dataset_dir: Path | str) -> dict:
    return read_json(Path(dataset_dir) / "manifest.json")


def load_sample(dataset_dir: Path | str, sample_id: str) -> dict:
    """Return {'image': (H,W,3) uint8, 'mask': (H,W) uint8, 'labels': dict}."""
    d = Path(dataset_dir) / "samples"
    return {
        "image": np.asarray(Image.open(d / f"{sample_id}.png").convert("RGB")),
        "mask": np.asarray(Image.open(d / f"{sample_id}_mask.png").convert("L")),
        "labels": read_json(d / f"{sample_id}.json"),
    }


def check_labels(lab: dict, sid: str) -> list[str]:
    """Problems with one label dict: key set, array shapes, SC-003 invariant."""
    errs = []
    if set(lab) != LABEL_KEYS:
        return [f"{sid}: label keys {sorted(set(lab) ^ LABEL_KEYS)} mismatch"]
    shapes = {
        "pose": (72,),
        "joints_3d_world": (NUM_JOINTS, 3),
        "joints_3d_cam": (NUM_JOINTS, 3),
        "keypoints_2d": (NUM_JOINTS, 2),
        "visibility": (NUM_JOINTS,),
    }
    for k, shp in shapes.items():
        if np.asarray(lab[k]).shape != shp:
            errs.append(f"{sid}: {k} shape {np.asarray(lab[k]).shape} != {shp}")
    if errs:
        return errs
    cam = lab["camera"]
    k, r, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])
    xw = np.array(lab["joints_3d_world"])
    xc = xw @ r.T + t
    if np.abs(xc - np.array(lab["joints_3d_cam"])).max() > 1e-4:
        errs.append(f"{sid}: joints_3d_cam inconsistent with camera")
    front = xc[:, 2] > 1e-6
    uv = np.stack(
        [k[0, 0] * xc[:, 0] / xc[:, 2] + k[0, 2], k[1, 1] * xc[:, 1] / xc[:, 2] + k[1, 2]], 1
    )
    if front.any() and np.abs(uv[front] - np.array(lab["keypoints_2d"])[front]).max() > 1.0:
        errs.append(f"{sid}: keypoints_2d differ from projection by > 1 px (SC-003)")
    return errs


def validate_dataset(dataset_dir: Path | str) -> list[str]:
    d = Path(dataset_dir)
    try:
        m = load_manifest(d)
    except (OSError, json.JSONDecodeError) as e:
        return [f"cannot read manifest: {e}"]
    errs = []
    if set(m) != MANIFEST_KEYS:
        return [f"manifest keys mismatch: {sorted(set(m) ^ MANIFEST_KEYS)}"]
    if m["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {m['schema_version']} != {SCHEMA_VERSION}")
    if m["complete"] is not True:
        errs.append("manifest not marked complete")
    if m["counts"]["samples"] != len(m["samples"]):
        errs.append("counts.samples does not match sample list")
    for s in m["samples"]:
        sid = s["id"]
        paths = [d / s["image"], d / s["mask"], d / s["labels"]]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            errs.append(f"{sid}: missing {missing}")
            continue
        raw = (d / s["labels"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != s["labels_sha256"]:
            errs.append(f"{sid}: labels sha256 mismatch")
        errs.extend(check_labels(json.loads(raw), sid))
    return errs
