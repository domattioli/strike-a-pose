"""Baseline config (spec 003 FR-006, contract eval-run.md)."""

from pathlib import Path

import pytest

from strike_a_pose.eval.config import BaselineConfig, ConfigError, config_from_dict, load_config

ROOT = Path(__file__).parents[1]
BASE = {"dataset_dir": "d", "split_file": "s.json"}


def test_defaults():
    c = config_from_dict(BASE)
    assert c == BaselineConfig("d", "s.json")
    assert c.eval_splits == ("val", "test") and c.figure_split == "val"
    assert c.image_pca_ks == (8, 16, 32, 64, 128) and c.pose_pca_ks == (4, 8, 16, 32)
    assert c.iou_threshold == 0.075 and c.seed == 0 and c.run_root == "runs"
    assert config_from_dict(c.to_dict()) == c


def test_committed_configs_load():
    full = load_config(ROOT / "configs" / "baselines.yaml")
    assert full.dataset_dir == "data/generated/smpl_proc_v1"
    assert full.split_file == "splits/smpl_proc_v1.json" and 64 in full.image_pca_ks
    fx = load_config(ROOT / "configs" / "baselines_fixture.yaml")
    assert fx.dataset_dir == "tests/fixtures/smpl_fixture" and fx.figure_split == "test"


def test_overrides_are_validated():
    c = config_from_dict(BASE).with_overrides(seed=5, run_root=None)
    assert c.seed == 5 and c.run_root == "runs"
    with pytest.raises(ConfigError):
        config_from_dict(BASE).with_overrides(seed="x")


@pytest.mark.parametrize(
    "bad",
    [
        {"bogus": 1},
        {"image_pca_ks": []},
        {"image_pca_ks": [0, 4]},
        {"pose_pca_ks": [2, 2]},
        {"pose_pca_ks": [1.5]},
        {"image_pca_ks": [True]},
        {"eval_splits": ["val", "holdout"]},
        {"eval_splits": ["test", "test"]},
        {"eval_splits": []},
        {"figure_split": "train"},
        {"iou_threshold": 0.0},
        {"iou_threshold": 1.0},
        {"iou_threshold": "x"},
        {"batch_size": 0},
        {"seed": 1.0},
        {"run_name": "a b"},
        {"dataset_dir": ""},
    ],
)
def test_errors(bad):
    with pytest.raises(ConfigError):
        config_from_dict({**BASE, **bad})


def test_missing_required_and_not_mapping(tmp_path):
    with pytest.raises(ConfigError, match="split_file"):
        config_from_dict({"dataset_dir": "d"})
    p = tmp_path / "c.yaml"
    p.write_text("- 1\n")
    with pytest.raises(ConfigError):
        load_config(p)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
