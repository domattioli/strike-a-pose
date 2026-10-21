"""Random body shape (spec 010 FR-004): per-pose SMPL betas from their own random stream."""

import numpy as np

SHAPE_STREAM = 1  # SeedSequence tag; the pose stream is default_rng(seed) (procedural.py)


def sample_betas(num_betas: int, std: float, clip: float, seed: int, pose_index: int) -> np.ndarray:
    """``num_betas`` values N(0, std^2) clipped to [-clip, clip] for one pose.

    Seeded by (seed, SHAPE_STREAM, pose_index), so the draws do not depend on other poses or
    on any other option, and the pose stream is untouched.
    """
    rng = np.random.default_rng([seed, SHAPE_STREAM, pose_index])
    return np.clip(rng.normal(0.0, std, num_betas), -clip, clip)
