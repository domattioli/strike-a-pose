"""Seeded random streams for each (seed, path) pair, and the deterministic torch flags.

Stages draw their random numbers from ``rng_for``, so each shard, body, and stage has its own
stream, and a rerun reproduces the same numbers on any machine. The design is research R9 in
specs/001-kill-test-mvp/research.md. The public sources are the NumPy documentation of
``SeedSequence`` and ``PCG64``, and the PyTorch reproducibility notes for
``torch.use_deterministic_algorithms``.
"""

import os
from numbers import Integral

import numpy as np
import torch

__all__ = ["rng_for", "seed_torch"]

# NumPy mixes entropy in 32-bit words, so each integer is split into words of this width.
_WORD_BITS = 32
_WORD_MASK = (1 << _WORD_BITS) - 1

# cuBLAS needs a fixed workspace size for deterministic matrix products on CUDA (research R9).
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def rng_for(seed: int, *path: int) -> np.random.Generator:
    """Return the PCG64 generator for one seed and one path, such as (stage, shard, body).

    The same arguments always give the same stream. Distinct arguments give distinct entropy lists,
    so the streams differ even where bare values would merge: a path and the same path followed by
    a zero, or a seed above 32 bits.
    """
    _check_non_negative_int(seed, "seed")
    for position, element in enumerate(path):
        _check_non_negative_int(element, f"path element {position}")
    # Bare values would collide. NumPy pads a short entropy list with zeros and splits a value
    # above 32 bits into two words, so [1, 2] and [1, 2, 0] would share a stream. Each value is
    # therefore written with its word count, and the path length comes before the path.
    entropy = [*_entropy_words(seed), *_entropy_words(len(path))]
    for element in path:
        entropy.extend(_entropy_words(element))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def seed_torch(seed: int) -> None:
    """Seed torch on every device and turn on the deterministic flags of research R9.

    Call this before the first CUDA matrix product. The helper sets CUBLAS_WORKSPACE_CONFIG only
    when the environment does not already set it. An operation that has no deterministic version
    gives a warning instead of an error.
    """
    _check_non_negative_int(seed, "seed")
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)


def _check_non_negative_int(value: object, name: str) -> None:
    """Raise TypeError unless value is a non-bool integer, and ValueError if it is negative."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, not {value}")


def _entropy_words(value: int) -> list[int]:
    """Return the word count of a non-negative integer, then its 32-bit words, lowest word first."""
    words = []
    rest = int(value)
    while True:
        words.append(rest & _WORD_MASK)
        rest >>= _WORD_BITS
        if rest == 0:
            break
    return [len(words), *words]
