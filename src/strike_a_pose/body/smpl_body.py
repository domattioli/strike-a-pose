"""SMPL body model of one gender through the smplx package, for SSP-3D ground truth (research R5).

SMPL (Loper et al., 2015) is the body model that SMPL-X extends. SSP-3D labels are SMPL shape
coefficients, not SMPL-X ones, so the SSP-3D ground truth needs SMPL (research R5, FR-018). The
model files ``smpl/SMPL_MALE.pkl``, ``smpl/SMPL_FEMALE.pkl``, and ``smpl/SMPL_NEUTRAL.pkl`` are
read from the asset root. A missing file raises ``MissingAssetError``, which names the file and the
configuration key ``real.ssp3d.path``. Nothing here downloads a file.

``SmplBody`` has the interface and the conventions of ``SmplxBody`` in ``smplx_body.py``: the
``BodyModel`` protocol of ``base.py``, metres, float64 arrays, the canonical zero pose, and part ids
from the argmax of the skinning weights. SMPL has 24 joints. The first 22 are the body joints of
``base.py``; joints 22 and 23 are the left and right hand. The pose of SMPL has 23 rotations, and
the two hand rotations are not an argument of this package, so they stay at zero.

Public source. Loper et al., "SMPL: A Skinned Multi-Person Linear Model" (ACM Transactions on
Graphics 34(6), 2015), as implemented by the smplx package (https://pypi.org/project/smplx, version
0.1.28, research R5).
"""

from pathlib import Path

import numpy as np
import torch

from strike_a_pose.assets import ROOT_PLACEHOLDER, require_asset
from strike_a_pose.body.smplx_body import (
    DEFAULT_N_BETAS,
    FloatArray,
    SmplPackageBody,
    import_smplx,
)

__all__ = ["SMPL_ASSET_KEY", "SMPL_GENDERS", "SmplBody"]

# The genders of the SMPL model files, which are named SMPL_<GENDER>.pkl in upper case.
SMPL_GENDERS: tuple[str, ...] = ("male", "female", "neutral")

# The configuration key that needs the SMPL files (assets.py lists them under real.ssp3d.path).
SMPL_ASSET_KEY = "real.ssp3d.path"

# The hand joints of SMPL (joints 22 and 23). Their rotations, after the 63 body values, are zero.
SMPL_HAND_JOINTS = 2


class SmplBody(SmplPackageBody):
    """The SMPL body model of one gender, read from ``<asset root>/smpl/SMPL_<GENDER>.pkl``."""

    def __init__(
        self,
        asset_root: str | Path | None,
        gender: str = "neutral",
        *,
        n_betas: int = DEFAULT_N_BETAS,
    ) -> None:
        """Load the model file of the gender. Raise ``ValueError`` for any other gender."""
        if gender not in SMPL_GENDERS:
            raise ValueError(f"gender must be one of {', '.join(SMPL_GENDERS)}, got {gender!r}")
        root = None if asset_root is None else Path(asset_root)
        model_file = require_asset(
            f"{ROOT_PLACEHOLDER}/smpl/SMPL_{gender.upper()}.pkl", SMPL_ASSET_KEY, root
        )
        smplx = import_smplx()
        model = smplx.create(
            str(model_file),
            "smpl",
            gender=gender,
            num_betas=n_betas,
            dtype=torch.float64,
        )
        super().__init__(model, n_betas=n_betas)
        self.gender = gender

    def _evaluate(
        self, betas: FloatArray, pose_root: FloatArray, pose_body: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        count = betas.shape[0]
        hand_joints = np.zeros((count, 3 * SMPL_HAND_JOINTS))
        return self._run(
            betas=betas,
            global_orient=pose_root,
            body_pose=np.concatenate([pose_body, hand_joints], axis=1),
            transl=np.zeros((count, 3)),
        )
