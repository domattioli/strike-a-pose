import json

import numpy as np
import pytest

from strike_a_pose.datagen.generate import OutputExistsError, generate_dataset
from strike_a_pose.datagen.io import LABEL_KEYS, load_manifest, load_sample, validate_dataset
from strike_a_pose.datagen.smpl import SMPLModel


def _dilate(mask, r):
    m = mask > 0
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def test_generate_us1(small_cfg):
    out = generate_dataset(small_cfg, seed=3)
    m = load_manifest(out)
    assert m["complete"] is True and m["counts"] == {"poses": 2, "cameras": 8, "samples": 16}
    assert len(m["samples"]) == 16 and m["samples"][0]["id"] == "00000_Front_0"
    assert validate_dataset(out) == []
    assert m["seed"] == 3 and m["config"]["num_poses"] == 2
    assert len(m["inputs"]["smpl_model"]["sha256"]) == 64 and m["inputs"]["pose_source"]["files"]
    assert m["code"]["commit"]
    for s in m["samples"]:
        smp = load_sample(out, s["id"])
        lab = smp["labels"]
        assert smp["image"].shape == (64, 48, 3) and smp["mask"].shape == (64, 48)
        assert set(lab) == LABEL_KEYS
        vis = np.array(lab["visibility"])
        assert vis.any()
        uv = np.array(lab["keypoints_2d"])[vis].astype(int)
        inside = _dilate(smp["mask"], 2)[uv[:, 1], uv[:, 0]]
        assert inside.all(), s["id"]  # US1-2: keypoints land on the silhouette
        assert smp["mask"].any() and (smp["image"][smp["mask"] == 0] == 0).all()


def test_output_exists_and_overwrite(small_cfg):
    out = generate_dataset(small_cfg, seed=0)
    with pytest.raises(OutputExistsError):
        generate_dataset(small_cfg, seed=0)
    assert generate_dataset(small_cfg, seed=0, overwrite=True) == out


def test_reproducible_us2(small_cfg, tmp_path):
    a = generate_dataset(small_cfg, seed=5, output_root=tmp_path / "a")
    b = generate_dataset(small_cfg, seed=5, output_root=tmp_path / "b")
    c = generate_dataset(small_cfg, seed=6, output_root=tmp_path / "c")
    for rel in ["manifest.json", "checksums.json"] + [
        s["labels"] for s in load_manifest(a)["samples"]
    ]:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    assert (a / "manifest.json").read_bytes() != (c / "manifest.json").read_bytes()
    info = json.loads((a / "run_info.json").read_text())
    assert {"python", "numpy", "pillow", "platform", "started_utc"} <= set(info)


def test_interrupted_run_is_incomplete(small_cfg):
    real = SMPLModel.load(small_cfg.smpl_model_path)

    class Boom:
        faces = real.faces
        calls = 0

        def forward(self, pose, betas):
            Boom.calls += 1
            if Boom.calls > 1:
                raise RuntimeError("interrupted")
            return real.forward(pose, betas)

    with pytest.raises(RuntimeError):
        generate_dataset(small_cfg, seed=0, body_model=Boom())
    out = __import__("pathlib").Path(small_cfg.output_root) / small_cfg.dataset_name
    assert load_manifest(out)["complete"] is False
    assert "manifest not marked complete" in validate_dataset(out)


def test_validate_detects_tampering(small_cfg):
    out = generate_dataset(small_cfg, seed=0)
    s = load_manifest(out)["samples"][0]
    lab = json.loads((out / s["labels"]).read_text())
    lab["keypoints_2d"][0][0] += 5
    (out / s["labels"]).write_text(json.dumps(lab))
    (out / load_manifest(out)["samples"][1]["image"]).unlink()
    errs = validate_dataset(out)
    assert any("sha256" in e for e in errs) and any("SC-003" in e for e in errs)
    assert any("missing" in e for e in errs)
