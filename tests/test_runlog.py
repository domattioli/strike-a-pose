"""Run-directory helpers (spec 003 FR-007, FR-008)."""

import json
import random
import re
from datetime import UTC, datetime

import numpy as np
import pytest
import torch
import yaml

from strike_a_pose.runlog import (
    dumps_exact,
    environment_summary,
    git_info,
    make_run_dir,
    peak_rss_mb,
    round_sig,
    seed_everything,
    write_yaml,
)


def _draws():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_repeats_draws():
    seed_everything(7)
    a = _draws()
    seed_everything(7)
    assert _draws() == a
    seed_everything(8)
    assert _draws() != a


def test_environment_and_git():
    env = environment_summary()
    keys = {"python", "platform", "numpy", "torch", "pillow", "cpu_count", "torch_threads",
            "cpu_model"}  # fmt: skip
    assert set(env) == keys
    g = git_info()
    assert set(g) == {"commit", "dirty"}
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", g["commit"])


def test_make_run_dir(tmp_path):
    t = datetime(2026, 10, 20, 1, 2, 3, tzinfo=UTC)
    d = make_run_dir(tmp_path / "runs", "base-1", now=t)
    assert d.is_dir() and d.name == "base-1-20261020T010203Z"
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path / "runs", "base-1", now=t)
    with pytest.raises(ValueError):
        make_run_dir(tmp_path, "bad name/..", now=t)


def test_round_sig_and_dumps_exact():
    assert round_sig(0.000123456789) == 0.000123457
    assert round_sig(123456.789) == 123457.0
    assert round_sig(-1.23456789) == -1.23457
    assert round_sig(float("nan")) is None
    assert round_sig({"a": [np.float32(0.5), np.int64(3), True]}) == {"a": [0.5, 3, True]}
    s = dumps_exact({"b": 1.0, "a": {"z": float("nan"), "y": 2}})
    assert s.endswith("}\n") and s.index('"a"') < s.index('"b"')
    assert json.loads(s) == {"a": {"y": 2, "z": None}, "b": 1.0}
    with pytest.raises(ValueError):
        dumps_exact({"x": float("inf")})


def test_write_yaml_roundtrip(tmp_path):
    obj = {"b": [1, 2], "a": "x", "c": 0.075}
    write_yaml(tmp_path / "c.yaml", obj)
    assert yaml.safe_load((tmp_path / "c.yaml").read_text()) == obj


def test_peak_rss_mb():
    """Spec 015 FR-005: process peak RSS in MiB, rounded to 0.1."""
    v = peak_rss_mb()
    assert isinstance(v, float) and v > 10.0 and v == round(v, 1)
