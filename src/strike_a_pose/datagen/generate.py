"""Dataset generation orchestration (plan: pipeline)."""

import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import PIL
from PIL import Image

from .amass import sample_poses
from .assets import require_dir, require_file, sha256_file
from .cameras import camera_ring, project, visibility
from .config import GenerationConfig
from .io import SCHEMA_VERSION, write_json
from .procedural import SPARSIFY_MIN_KEEP, ranges_table, sample_procedural_poses
from .render import render
from .skeleton import SKELETON_NAME, SMPL24_JOINT_NAMES, SMPL24_PARENTS
from .smpl import SMPLModel


class BodyModel(Protocol):
    faces: np.ndarray

    def forward(self, pose: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class OutputExistsError(Exception):
    pass


def git_commit() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=here, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )  # fmt: skip
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, format="PNG", optimize=False, compress_level=6)


def generate_dataset(
    cfg: GenerationConfig,
    seed: int,
    output_root: Path | str | None = None,
    overwrite: bool = False,
    body_model: BodyModel | None = None,
) -> Path:
    started = datetime.now(UTC).isoformat()
    # 1. assets first: fail fast, write nothing.
    smpl_path = require_file(Path(cfg.smpl_model_path), "SMPL model file")
    if cfg.pose_source == "amass":
        amass_root = require_dir(Path(cfg.amass_root), "AMASS root directory")
    if body_model is None:
        body_model = SMPLModel.load(smpl_path)
    if cfg.pose_source == "amass":
        poses = sample_poses(amass_root, cfg.amass_subsets, cfg.frame_stride, cfg.num_poses, seed)
        used = sorted({p.source_file for p in poses})
        pose_input = {
            "type": "amass",
            "files": [{"file": f, "sha256": sha256_file(amass_root / f)} for f in used],
        }

        def source_label(rec):
            return {"type": "amass", "file": rec.source_file, "frame": rec.frame,
                    "root_orient": list(rec.source_root_orient)}  # fmt: skip
    else:
        version = cfg.ranges_version
        poses = sample_procedural_poses(cfg.num_poses, seed, version)
        pose_input = {
            "type": "procedural", "ranges_version": version,
            "euler_order": "scipy xyz (extrinsic, R = Rz @ Ry @ Rx), SMPL joint-local axes",
            "sparsify_min_keep": SPARSIFY_MIN_KEEP,
            "ranges_deg": ranges_table(version),
        }  # fmt: skip

        def source_label(rec):
            return {"type": "procedural", "ranges_version": version,
                    "euler_deg": [list(e) for e in rec.euler_deg]}  # fmt: skip

    # 2. output dir
    out = Path(output_root if output_root is not None else cfg.output_root) / cfg.dataset_name
    if out.exists():
        if not overwrite:
            raise OutputExistsError(f"{out} already exists; pass --overwrite to regenerate it")
        shutil.rmtree(out)

    random.seed(seed)
    np.random.seed(seed % 2**32)
    cams = camera_ring(cfg.camera_ring, cfg.image.height, cfg.image.width)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": cfg.dataset_name,
        "complete": False,
        "seed": seed,
        "config": cfg.to_dict(),
        "code": git_commit(),
        "inputs": {
            "smpl_model": {"filename": smpl_path.name, "sha256": sha256_file(smpl_path)},
            "pose_source": pose_input,
        },
        "skeleton": {
            "name": SKELETON_NAME,
            "joints": list(SMPL24_JOINT_NAMES),
            "parents": list(SMPL24_PARENTS),
        },
        "cameras": [c.to_dict() for c in cams],
        "counts": {"poses": len(poses), "cameras": len(cams), "samples": len(poses) * len(cams)},
        "samples": [],
    }
    (out / "samples").mkdir(parents=True)
    write_json(out / "manifest.json", manifest)

    betas = np.zeros(cfg.num_betas)
    samples, checksums = [], {}
    for rec in poses:
        verts, joints = body_model.forward(np.array(rec.pose), betas)
        for cam in cams:
            sid = f"{rec.index:05d}_{cam.name}"
            xc, uv = project(cam, joints)
            rgb, mask = render(verts, body_model.faces, cam)
            img_rel, mask_rel, lab_rel = (
                f"samples/{sid}.png", f"samples/{sid}_mask.png", f"samples/{sid}.json",
            )  # fmt: skip
            _save_png(out / img_rel, rgb)
            _save_png(out / mask_rel, mask)
            labels = {
                "schema_version": SCHEMA_VERSION, "sample_id": sid, "pose_index": rec.index,
                "camera_name": cam.name,
                "pose_source": source_label(rec),
                "pose": list(rec.pose), "betas": betas,
                "joints_3d_world": joints, "joints_3d_cam": xc, "keypoints_2d": uv,
                "visibility": visibility(xc, uv, cam.height, cam.width),
                "camera": cam.to_dict(),
            }  # fmt: skip
            sha = write_json(out / lab_rel, labels)
            samples.append(
                {"id": sid, "image": img_rel, "mask": mask_rel, "labels": lab_rel,
                 "labels_sha256": sha}
            )  # fmt: skip
            checksums[img_rel] = sha256_file(out / img_rel)
            checksums[mask_rel] = sha256_file(out / mask_rel)

    write_json(out / "checksums.json", checksums)
    write_json(
        out / "run_info.json",
        {
            "started_utc": started, "finished_utc": datetime.now(UTC).isoformat(),
            "python": sys.version, "numpy": np.__version__, "pillow": PIL.__version__,
            "platform": platform.platform(), "argv": sys.argv, "cwd": os.getcwd(), "seed": seed,
        },
    )  # fmt: skip
    manifest["samples"] = samples
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)
    return out
