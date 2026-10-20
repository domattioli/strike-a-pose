"""PoseImageDataset on the committed fixture + corrupted copies (spec 002 FR-001..003, FR-008)."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from strike_a_pose.data import DatasetError, PoseImageDataset, make_split, write_split
from strike_a_pose.data.split import manifest_sha256
from strike_a_pose.datagen.io import load_manifest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SPLIT = ROOT / "splits" / "smpl_fixture.json"
H, W, J = 256, 192, 24

TENSOR_SPEC = {
    "image": (torch.float32, (1, H, W)),
    "mask": (torch.bool, (1, H, W)),
    "keypoints_2d": (torch.float32, (J, 2)),
    "visibility": (torch.bool, (J,)),
    "joints_3d_cam": (torch.float32, (J, 3)),
    "joints_3d_world": (torch.float32, (J, 3)),
    "pose": (torch.float32, (72,)),
    "betas": (torch.float32, (10,)),
}
CAMERA_SPEC = {"K": (3, 3), "R": (3, 3), "t": (3,), "yaw_deg": ()}


@pytest.fixture
def copy(tmp_path) -> Path:
    d = tmp_path / "smpl_fixture"
    shutil.copytree(FIXTURE, d)
    return d


def _edit_manifest(d: Path, fn) -> None:
    m = json.loads((d / "manifest.json").read_text())
    fn(m)
    (d / "manifest.json").write_text(json.dumps(m))


def test_fixture_items_follow_contract():
    t0 = time.perf_counter()
    ds = PoseImageDataset(FIXTURE)
    assert len(ds) == 16 and ds.image_size == (H, W) and ds.split is None
    m = load_manifest(FIXTURE)
    assert ds.sample_ids == [s["id"] for s in m["samples"]]
    for i in range(len(ds)):
        it = ds[i]
        assert set(it) == set(TENSOR_SPEC) | {
            "camera", "sample_id", "pose_index", "camera_name", "camera_index",
        }  # fmt: skip
        for k, (dt, shp) in TENSOR_SPEC.items():
            assert it[k].dtype == dt and tuple(it[k].shape) == shp, k
        for k, shp in CAMERA_SPEC.items():
            assert it["camera"][k].dtype == torch.float32 and tuple(it["camera"][k].shape) == shp
        sid = it["sample_id"]
        lab = json.loads((FIXTURE / "samples" / f"{sid}.json").read_text())
        png = np.asarray(Image.open(FIXTURE / "samples" / f"{sid}.png"))[..., 0]
        assert torch.equal(it["image"][0], torch.from_numpy(png.astype(np.float32) / 255))
        assert 0.0 <= float(it["image"].min()) and float(it["image"].max()) <= 1.0
        assert torch.equal(it["mask"][0], it["image"][0] > 0)
        assert np.allclose(it["keypoints_2d"].numpy(), lab["keypoints_2d"], atol=1e-4)
        assert np.allclose(it["joints_3d_cam"].numpy(), lab["joints_3d_cam"], atol=1e-6)
        assert np.allclose(it["camera"]["K"].numpy(), lab["camera"]["K"])
        assert it["visibility"].tolist() == lab["visibility"]
        assert (it["pose_index"], it["camera_name"]) == (lab["pose_index"], lab["camera_name"])
        assert ds.cameras[it["camera_index"]] == it["camera_name"]
    assert time.perf_counter() - t0 < 5.0  # SC-001


def test_default_collate_batches():
    batch = next(iter(DataLoader(PoseImageDataset(FIXTURE), batch_size=4)))
    assert batch["image"].shape == (4, 1, H, W) and batch["mask"].dtype == torch.bool
    assert batch["camera"]["K"].shape == (4, 3, 3)
    assert batch["pose_index"].dtype == torch.int64 and len(batch["sample_id"]) == 4


def test_transform_applied():
    ds = PoseImageDataset(FIXTURE, transform=lambda s: {**s, "image": s["image"] * 2})
    assert float(ds[0]["image"].max()) > 1.0


def test_split_views_together_and_disjoint():
    tr = PoseImageDataset(FIXTURE, split="train", split_file=SPLIT)
    te = PoseImageDataset(FIXTURE, split="test", split_file=SPLIT)
    va = PoseImageDataset(FIXTURE, split="val", split_file=SPLIT)
    assert (len(tr), len(te), len(va)) == (8, 8, 0)
    for ds in (tr, te):
        poses = {ds[i]["pose_index"] for i in range(len(ds))}
        cams = [ds[i]["camera_name"] for i in range(len(ds))]
        assert len(poses) == 1 and cams == ds.cameras  # all 8 views of one pose
    assert not set(tr.sample_ids) & set(te.sample_ids)
    split = json.loads(SPLIT.read_text())
    assert split["manifest_sha256"] == manifest_sha256(FIXTURE)


def test_verify_full_passes_on_fixture():
    assert len(PoseImageDataset(FIXTURE, verify=True)) == 16


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda m: m.update(complete=False), "not marked complete"),
        (lambda m: m.update(schema_version=2), "schema_version 2"),
        (lambda m: m.pop("seed"), "keys"),
        (lambda m: m["counts"].update(poses=3), "counts inconsistent"),
        (lambda m: m["samples"][0].update(id="00000_Nowhere_0"), "not in manifest cameras"),
        (lambda m: m["samples"][1].update(id=m["samples"][0]["id"]), "duplicate"),
        (lambda m: m["skeleton"].update(name="coco17"), "skeleton"),
        (lambda m: m.update(skeleton=None), "malformed manifest"),
        (lambda m: m["config"].pop("image"), "malformed manifest"),
    ],
)
def test_bad_manifest(copy, edit, match):
    _edit_manifest(copy, edit)
    with pytest.raises(DatasetError, match=match):
        PoseImageDataset(copy)


def test_missing_dir_and_files(tmp_path, copy):
    with pytest.raises(DatasetError, match="cannot read manifest"):
        PoseImageDataset(tmp_path / "nothing")
    (copy / "samples" / "00001_Back_180.png").unlink()
    with pytest.raises(DatasetError, match="missing files: 00001_Back_180"):
        PoseImageDataset(copy)


def test_checksum_mismatch_only_with_verify(copy):
    p = copy / "samples" / "00000_Front_0.json"
    lab = json.loads(p.read_text())
    lab["pose"][3] += 0.01
    p.write_text(json.dumps(lab))
    assert len(PoseImageDataset(copy)) == 16  # cheap checks pass
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        PoseImageDataset(copy, verify=True)


def test_bad_label_on_read(copy):
    p = copy / "samples" / "00000_Right_90.json"
    lab = json.loads(p.read_text())
    lab["keypoints_2d"] = lab["keypoints_2d"][:10]
    p.write_text(json.dumps(lab))
    ds = PoseImageDataset(copy)
    with pytest.raises(DatasetError, match="00000_Right_90.*keypoints_2d shape"):
        ds[ds.sample_ids.index("00000_Right_90")]


def test_label_ids_must_match_manifest(copy):
    p = copy / "samples" / "00000_Right_90.json"
    lab = json.loads(p.read_text())
    lab["pose_index"] = 1
    p.write_text(json.dumps(lab))
    ds = PoseImageDataset(copy)
    with pytest.raises(DatasetError, match="do not match manifest"):
        ds[ds.sample_ids.index("00000_Right_90")]


def test_wrong_image_size_on_read(copy):
    Image.new("RGB", (W, H - 1)).save(copy / "samples" / "00001_Front_0.png")
    ds = PoseImageDataset(copy)
    with pytest.raises(DatasetError, match=r"00001_Front_0: image size \(255, 192\)"):
        ds[ds.sample_ids.index("00001_Front_0")]


def test_split_argument_errors():
    with pytest.raises(ValueError, match="split_file"):
        PoseImageDataset(FIXTURE, split="train")
    with pytest.raises(ValueError, match="one of"):
        PoseImageDataset(FIXTURE, split="holdout", split_file=SPLIT)


def test_split_for_other_dataset_rejected(copy, tmp_path):
    _edit_manifest(copy, lambda m: m.update(seed=99))  # same samples, different manifest bytes
    with pytest.raises(DatasetError, match="split was made for manifest"):
        PoseImageDataset(copy, split="train", split_file=SPLIT)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(DatasetError, match="not a split file"):
        PoseImageDataset(FIXTURE, split="train", split_file=bad)


def test_split_written_for_copy_loads(copy, tmp_path):
    split = make_split(load_manifest(copy), manifest_sha256(copy), seed=5, fractions=(0.5, 0, 0.5))
    p = write_split(split, tmp_path / "s.json")
    tr = PoseImageDataset(copy, split="train", split_file=p)
    assert {tr[i]["pose_index"] for i in range(len(tr))} == set(split["poses"]["train"])
