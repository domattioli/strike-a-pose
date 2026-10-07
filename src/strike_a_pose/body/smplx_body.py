"""SMPL-X body model through the smplx package, read from the asset root (research R5).

The smplx package is an optional extra (``[body]``). It is imported when a model is first built,
not when this module is imported, so the module needs neither the package nor a licensed file to
load. The licensed file ``smplx/SMPLX_NEUTRAL.npz`` is looked up under the asset root at that time.
A missing file raises ``MissingAssetError``, which names the file and the configuration key
``body.model`` (FR-021). Nothing here downloads a file.

``SmplxBody`` implements the ``BodyModel`` protocol of ``base.py``, with these conventions:

* The 22 body joints are the first 22 joints of SMPL-X, in the order of ``JOINT_NAMES``. The
  parent table of the model must give these 22 joints the parents of ``base.PARENTS``; the
  constructor checks it.
* ``pose_root`` is the global orientation and ``pose_body`` the rotations of joints 1 to 21, both
  as axis-angle values in radians. The jaw, eye, and finger rotations are zero, and so are the
  expression coefficients, so the two pose arguments give the whole pose.
* The canonical pose is the zero pose of the model, with every rotation zero (FR-004).
* ``part_ids`` gives each vertex the body joint with the largest skinning weight (``lbs_weights``,
  research R6). A largest-weight joint outside the 22 body joints (the jaw, the eyes, the finger
  joints) passes the vertex to its nearest ancestor among the 22 body joints.
* Lengths are in metres, with the vertical axis y, as in ``base.py``. Arrays are NumPy float64 in
  and out. The model runs in float64 on the CPU.

Public source. Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and Body from a Single
Image" (CVPR 2019, https://smpl-x.is.tue.mpg.de/) defines SMPL-X. Loper et al., "SMPL: A Skinned
Multi-Person Linear Model" (ACM Transactions on Graphics 34(6), 2015) defines the skinning that the
smplx package implements (https://pypi.org/project/smplx, version 0.1.28, research R5).
"""

import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.assets import ROOT_PLACEHOLDER, require_asset
from strike_a_pose.body.base import (
    BODY_POSE_SIZE,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    ROOT_POSE_SIZE,
    BodyPose,
)

__all__ = [
    "DEFAULT_N_BETAS",
    "FloatArray",
    "HAND_POSE_SIZE",
    "NUM_EXPRESSION_COEFFICIENTS",
    "SMPLX_ASSET_KEY",
    "SMPLX_ASSET_PATH",
    "SmplPackageBody",
    "SmplxBody",
    "import_smplx",
]

# The asset path of the SMPL-X neutral model, as assets.py lists it, and the configuration key
# that needs it (contracts/config.md, body.model).
SMPLX_ASSET_PATH = f"{ROOT_PLACEHOLDER}/smplx/SMPLX_NEUTRAL.npz"
SMPLX_ASSET_KEY = "body.model"

# The type of the float64 arrays that the body models take and return, of any shape.
FloatArray = NDArray[np.float64]

# Shape coefficients used when none is given (contracts/config.md, body.n_betas).
DEFAULT_N_BETAS = 10

# Expression coefficients of SMPL-X, all zero here, and the hand pose: 15 joints of 3 values each.
NUM_EXPRESSION_COEFFICIENTS = 10
HAND_POSE_SIZE = 3 * 15


def import_smplx() -> ModuleType:
    """Import the smplx package. The package is optional, so the import waits for first use."""
    try:
        import smplx
    except ImportError as error:
        raise ImportError(
            "the smplx package is needed for the SMPL-X and SMPL body models; install the body "
            "extra, strike_a_pose[body] (research R5)"
        ) from error
    return smplx


class SmplPackageBody:
    """A body model that the smplx package has built from an SMPL-family file.

    This class holds what ``SmplxBody`` and ``SmplBody`` share: the checks of the model's parent
    table and skinning weights, the part ids, the faces, the canonical pose, and the conversions of
    inputs and outputs. A subclass builds the model and sets its forward pass in ``_evaluate``.
    """

    def __init__(self, model: Any, n_betas: int) -> None:
        """Take a model that the smplx package has built, and check its tables."""
        parents = np.asarray(_to_numpy(model.parents), dtype=np.int64)
        weights = _to_numpy(model.lbs_weights)
        faces = np.asarray(_to_numpy(model.faces), dtype=np.int64)
        if weights.ndim != 2 or weights.shape[1] != parents.shape[0]:
            raise ValueError(
                "lbs_weights must have shape (vertices, joints) with one column per joint; got "
                f"{weights.shape} for {parents.shape[0]} joints"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (triangles, 3), got {faces.shape}")
        self._model = model
        self.n_betas = n_betas
        self._faces = faces
        owners = _body_owners(parents)
        self._part_ids = owners[np.argmax(weights, axis=1)]

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The 22 body joint names in model order, equal to ``JOINT_NAMES``."""
        return JOINT_NAMES

    @property
    def faces(self) -> NDArray[np.int64]:
        """Triangle vertex indices, shape (F, 3). Each access returns a new array."""
        return self._faces.copy()

    @property
    def part_ids(self) -> NDArray[np.int64]:
        """For each vertex the index of the body joint that owns it, shape (V,). A new array."""
        return self._part_ids.copy()

    @property
    def vertex_count(self) -> int:
        """Number of vertices of every mesh this model returns."""
        return int(self._part_ids.shape[0])

    def canonical(self) -> BodyPose:
        """Return the canonical pose: the zero pose of the model, with every rotation zero."""
        return BodyPose(
            pose_root=np.zeros(ROOT_POSE_SIZE, dtype=np.float64),
            pose_body=np.zeros(BODY_POSE_SIZE, dtype=np.float64),
        )

    def vertices(self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike) -> FloatArray:
        """Return the posed mesh vertices in metres, shape (..., V, 3)."""
        batch_shape, flat_betas, flat_root, flat_body = _flatten_inputs(
            betas, pose_root, pose_body, self.n_betas
        )
        vertices, _ = self._evaluate(flat_betas, flat_root, flat_body)
        return vertices.reshape(batch_shape + (self.vertex_count, 3))

    def joints(self, betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike) -> FloatArray:
        """Return the posed positions of the 22 body joints in metres, shape (..., 22, 3)."""
        batch_shape, flat_betas, flat_root, flat_body = _flatten_inputs(
            betas, pose_root, pose_body, self.n_betas
        )
        _, joints = self._evaluate(flat_betas, flat_root, flat_body)
        return joints.reshape(batch_shape + (NUM_JOINTS, 3))

    def _evaluate(
        self, betas: FloatArray, pose_root: FloatArray, pose_body: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return (vertices, joints) for the flat arrays that ``_flatten_inputs`` returns."""
        raise NotImplementedError("a subclass gives the forward pass of its model")

    def _run(self, **arguments: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Call the smplx model with float64 tensors; return the vertices and the 22 body joints."""
        tensors = {
            name: torch.tensor(value, dtype=torch.float64) for name, value in arguments.items()
        }
        with torch.no_grad():
            output = self._model(**tensors)
        vertices = _to_numpy(output.vertices).astype(np.float64)
        joints = _to_numpy(output.joints)[:, :NUM_JOINTS].astype(np.float64)
        return vertices, joints


class SmplxBody(SmplPackageBody):
    """The SMPL-X neutral body model, read from ``<asset root>/smplx/SMPLX_NEUTRAL.npz``."""

    def __init__(self, asset_root: str | Path | None, *, n_betas: int = DEFAULT_N_BETAS) -> None:
        """Load the model file. Raise ``MissingAssetError`` when the root or the file is absent."""
        root = None if asset_root is None else Path(asset_root)
        model_file = require_asset(SMPLX_ASSET_PATH, SMPLX_ASSET_KEY, root)
        smplx = import_smplx()
        # Zero hand poses use the 45 axis-angle values per hand (use_pca=False). flat_hand_mean
        # makes the zero pose a flat open hand, so every rotation of the zero pose is zero.
        model = smplx.create(
            str(model_file),
            "smplx",
            gender="neutral",
            ext="npz",
            num_betas=n_betas,
            num_expression_coeffs=NUM_EXPRESSION_COEFFICIENTS,
            use_pca=False,
            flat_hand_mean=True,
            dtype=torch.float64,
        )
        super().__init__(model, n_betas=n_betas)

    def _evaluate(
        self, betas: FloatArray, pose_root: FloatArray, pose_body: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        count = betas.shape[0]
        return self._run(
            betas=betas,
            global_orient=pose_root,
            body_pose=pose_body,
            left_hand_pose=np.zeros((count, HAND_POSE_SIZE)),
            right_hand_pose=np.zeros((count, HAND_POSE_SIZE)),
            jaw_pose=np.zeros((count, 3)),
            leye_pose=np.zeros((count, 3)),
            reye_pose=np.zeros((count, 3)),
            expression=np.zeros((count, NUM_EXPRESSION_COEFFICIENTS)),
            transl=np.zeros((count, 3)),
        )


def _body_owners(parents: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return, for each joint of the model, the body joint that owns the vertices it carries.

    A body joint owns itself. Any other joint passes its vertices to the owner of its parent, which
    gives the nearest of its ancestors among the 22 body joints. A parent must have a lower index
    than its child, so one pass in index order sees every parent first.
    """
    count = parents.shape[0]
    if count < NUM_JOINTS or not np.array_equal(parents[:NUM_JOINTS], np.asarray(PARENTS)):
        raise ValueError(
            "the first 22 joints of the body model must have the parents of base.PARENTS; "
            f"got {parents[:NUM_JOINTS].tolist()}"
        )
    owners = np.arange(count, dtype=np.int64)
    for joint in range(NUM_JOINTS, count):
        parent = int(parents[joint])
        if not 0 <= parent < joint:
            raise ValueError(f"joint {joint} has parent {parent}; a parent must have a lower index")
        owners[joint] = owners[parent]
    return owners


def _flatten_inputs(
    betas: ArrayLike, pose_root: ArrayLike, pose_body: ArrayLike, n_betas: int
) -> tuple[tuple[int, ...], FloatArray, FloatArray, FloatArray]:
    """Check the three inputs, broadcast their batch dimensions, and flatten them.

    Returns the batch shape and the flat arrays of shape (count, n_betas), (count, 3), and
    (count, 63), where count is the product of the batch shape.
    """
    betas_array = _as_float_array(betas, "betas", n_betas)
    root_array = _as_float_array(pose_root, "pose_root", ROOT_POSE_SIZE)
    body_array = _as_float_array(pose_body, "pose_body", BODY_POSE_SIZE)
    try:
        batch_shape = np.broadcast_shapes(
            betas_array.shape[:-1], root_array.shape[:-1], body_array.shape[:-1]
        )
    except ValueError as error:
        raise ValueError(
            "the leading dimensions of betas, pose_root, and pose_body do not broadcast: "
            f"{betas_array.shape}, {root_array.shape}, {body_array.shape}"
        ) from error
    count = math.prod(batch_shape)

    def flatten(array: FloatArray, size: int) -> FloatArray:
        return np.broadcast_to(array, batch_shape + (size,)).reshape(count, size)

    return (
        batch_shape,
        flatten(betas_array, n_betas),
        flatten(root_array, ROOT_POSE_SIZE),
        flatten(body_array, BODY_POSE_SIZE),
    )


def _as_float_array(value: ArrayLike, name: str, size: int) -> FloatArray:
    """Convert to a float64 array whose last dimension has the given size."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have a last dimension of {size}, got shape {array.shape}")
    return array


def _to_numpy(value: Any) -> NDArray[Any]:
    """Return a NumPy array for a torch tensor (detached, on the CPU) or for an array-like value."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
