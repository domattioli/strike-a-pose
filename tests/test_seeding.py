"""Smoke tests for seeding.py: repeatable streams, separate streams per path, and torch flags."""

import os

import numpy as np
import pytest
import torch

from strike_a_pose.seeding import rng_for, seed_torch


def _draw(generator: np.random.Generator) -> np.ndarray:
    """Return the first draws of a generator, as an array that can be compared exactly."""
    return generator.random(8)


def test_same_seed_and_path_repeat_the_stream():
    first = _draw(rng_for(20261007, 3, 17, 42))
    second = _draw(rng_for(20261007, 3, 17, 42))
    assert np.array_equal(first, second)


def test_each_path_in_a_grid_gets_its_own_stream():
    # Every (stage, shard, body) of this grid must start a different stream.
    starts = {
        tuple(_draw(rng_for(7, stage, shard, body)))
        for stage in range(3)
        for shard in range(4)
        for body in range(5)
    }
    assert len(starts) == 3 * 4 * 5


@pytest.mark.parametrize(
    "other",
    [
        (8, 3, 17, 42),  # another seed
        (7, 4, 17, 42),  # another stage
        (7, 3, 18, 42),  # another shard
        (7, 3, 17, 43),  # another body
        (7, 3, 17),  # a shorter path
        (7, 3, 17, 42, 0),  # a longer path that ends in zero
    ],
)
def test_another_seed_or_path_gives_another_stream(other):
    assert not np.array_equal(_draw(rng_for(7, 3, 17, 42)), _draw(rng_for(*other)))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((7, 1, 2), (7, 1, 2, 0)),  # NumPy pads a short entropy list with zeros
        ((7,), (7, 0)),  # the same case with an empty path
        ((7, 2**32), (7, 0, 1)),  # a value above 32 bits spans two 32-bit words
        ((2**32, 0), (0, 1, 0)),  # a large seed against a small seed with a path
    ],
)
def test_arguments_that_bare_entropy_lists_would_merge_stay_distinct(left, right):
    assert not np.array_equal(_draw(rng_for(*left)), _draw(rng_for(*right)))


@pytest.mark.parametrize(
    ("bad_value", "error"),
    [(-1, ValueError), (1.5, TypeError), ("7", TypeError), (True, TypeError)],
)
def test_rejects_values_that_are_not_non_negative_integers(bad_value, error):
    with pytest.raises(error):
        rng_for(bad_value, 1)
    with pytest.raises(error):
        rng_for(7, bad_value)


@pytest.fixture
def restore_torch_flags():
    """Restore the global torch flags that seed_torch changes, for the other tests."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def test_seed_torch_repeats_draws_and_sets_deterministic_flags(monkeypatch, restore_torch_flags):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_torch(5)
    first = torch.rand(6)
    seed_torch(5)
    assert torch.equal(torch.rand(6), first)
    seed_torch(6)
    assert not torch.equal(torch.rand(6), first)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_seed_torch_keeps_a_workspace_setting_already_present(monkeypatch, restore_torch_flags):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_torch(5)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
