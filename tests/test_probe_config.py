"""Probe config parsing and validation (spec 006; contracts/probe-run.md)."""

from pathlib import Path

import pytest

from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.probe_config import TARGETS, ProbeConfig, config_from_dict, load_config

ROOT = Path(__file__).parents[1]
BASE = {"dataset_dir": "d", "split_file": "s.json", "checkpoint": "c.pt"}


def test_defaults():
    c = config_from_dict(BASE)
    assert c == ProbeConfig("d", "s.json", "c.pt")
    assert c.targets == TARGETS and c.eval_splits == ("val", "test")
    assert c.ridge_lambdas == (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert c.mlp_hidden == (256, 256) and c.mlp_weight_decays == (0.0, 1e-4, 1e-3)
    assert c.cache_dir == "runs/probe-cache" and c.random_encoder_control is True
    d = c.to_dict()
    assert d["mlp_hidden"] == [256, 256] and config_from_dict(d) == c


@pytest.mark.parametrize("name", ["probe.yaml", "probe_fixture.yaml"])
def test_shipped_configs_load(name):
    c = load_config(ROOT / "configs" / name)
    assert c.figure_split in c.eval_splits and "val" in c.eval_splits


def test_overrides_and_casts():
    c = config_from_dict(
        BASE | {"ridge_lambdas": [1, 2], "mlp_weight_decays": [0], "cache_dir": None}
    )
    assert c.ridge_lambdas == (1.0, 2.0) and all(isinstance(x, float) for x in c.ridge_lambdas)
    assert c.cache_dir is None
    o = c.with_overrides(checkpoint="other.pt", seed=3, run_root=None)
    assert (o.checkpoint, o.seed, o.run_root) == ("other.pt", 3, "runs")


@pytest.mark.parametrize(
    "extra",
    [
        {"bogus": 1},
        {"checkpoint": ""},
        {"seed": "0"},
        {"batch_size": 0},
        {"mlp_epochs": True},
        {"mlp_lr": 0},
        {"targets": []},
        {"targets": ["yaw", "yaw"]},
        {"targets": ["keypoints"]},
        {"ridge_lambdas": [0.0]},
        {"ridge_lambdas": [1, 1]},
        {"mlp_weight_decays": [-1e-4]},
        {"mlp_hidden": [0]},
        {"pose_pca_ks": [0]},
        {"eval_splits": ["test"], "figure_split": "test"},
        {"eval_splits": ["val", "train"]},
        {"figure_split": "train"},
        {"run_name": "a b"},
        {"random_encoder_control": 1},
        {"cache_dir": ""},
    ],
)
def test_errors(extra):
    with pytest.raises(ConfigError):
        config_from_dict(BASE | extra)


@pytest.mark.parametrize("key", ["dataset_dir", "split_file", "checkpoint"])
def test_missing_required(key):
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict({k: v for k, v in BASE.items() if k != key})
