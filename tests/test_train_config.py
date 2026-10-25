"""Train config parsing and validation (specs 004 and 008; contracts/train-run.md)."""

from pathlib import Path

import pytest
import yaml

from strike_a_pose.eval.config import ConfigError
from strike_a_pose.train import TrainConfig, config_from_dict, load_config
from strike_a_pose.train.config import LEGACY_VALUES, load_run_config, with_legacy_values

ROOT = Path(__file__).parents[1]
BASE = {"dataset_dir": "d", "split_file": "s.json"}


def test_defaults():
    c = config_from_dict(BASE)
    assert c == TrainConfig("d", "s.json")
    assert c.latent_dim == 64 and c.channels == (16, 32, 64, 128, 256, 256)
    assert (c.epochs, c.batch_size, c.lr, c.lr_schedule) == (60, 64, 1e-3, "cosine")
    assert c.eval_splits == ("val", "test") and c.context_pca_k == 64
    assert c.select_weights == "best_val"  # spec 008: best-val weights are scored by default
    d = c.to_dict()
    assert d["channels"] == [16, 32, 64, 128, 256, 256] and d["eval_splits"] == ["val", "test"]
    assert config_from_dict(d) == c


@pytest.mark.parametrize(
    "name",
    [
        "ae.yaml",
        "ae_fixture.yaml",
        "smpl_proc_v3/ae_pose_l0.003.yaml",
        "smpl_proc_v3/ae_pose_l0.03.yaml",
        "smpl_proc_v3/ae_pose_l0.03_mask.yaml",
    ],
)
def test_shipped_configs_load(name):
    c = load_config(ROOT / "configs" / name)
    assert c.figure_split in c.eval_splits
    assert c.select_weights == "best_val"


def test_overrides_and_null_pca():
    c = config_from_dict(BASE | {"context_pca_k": None, "lr": 1})
    assert c.context_pca_k is None and c.lr == 1.0 and isinstance(c.lr, float)
    o = c.with_overrides(seed=5, run_root="x", lr=None)
    assert (o.seed, o.run_root, o.lr) == (5, "x", 1.0)


@pytest.mark.parametrize(
    "extra",
    [
        {"nope": 1},
        {"epochs": 0},
        {"epochs": 1.5},
        {"batch_size": True},
        {"latent_dim": -1},
        {"seed": "0"},
        {"lr": 0},
        {"lr": -1e-3},
        {"lr": "fast"},
        {"lr": float("nan")},
        {"channels": []},
        {"channels": [16, 0]},
        {"channels": 16},
        {"lr_schedule": "step"},
        {"eval_splits": ["train", "val"]},
        {"eval_splits": ["val", "val"]},
        {"eval_splits": []},
        {"figure_split": "test", "eval_splits": ["val"]},
        {"iou_threshold": 1.0},
        {"context_pca_k": 0},
        {"run_name": "bad name"},
        {"dataset_dir": ""},
        {"select_weights": "best"},
        {"select_weights": ""},
        {"select_weights": True},
        {"select_weights": None},
    ],
)
def test_invalid(extra):
    with pytest.raises(ConfigError):
        config_from_dict(BASE | extra)


def test_missing_required_and_unreadable(tmp_path):
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict({"dataset_dir": "d"})
    with pytest.raises(ConfigError):
        config_from_dict([1, 2])
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_run_dir_config_legacy_values(tmp_path):
    """Spec 008 FR-007: a run dir's config.yaml without select_weights (started before 008)
    reads as `final` on --resume; a new run from such a file gets the default."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(BASE))
    assert load_config(p).select_weights == "best_val"
    assert load_run_config(p).select_weights == "final"
    assert LEGACY_VALUES["select_weights"] == "final"
    p.write_text(yaml.safe_dump(BASE | {"select_weights": "best_val"}))
    assert load_run_config(p).select_weights == "best_val"  # a present key wins
    assert with_legacy_values(BASE) == BASE | LEGACY_VALUES and with_legacy_values(None) is None
    with pytest.raises(ConfigError, match="cannot read"):
        load_run_config(tmp_path / "missing.yaml")
