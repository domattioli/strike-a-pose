"""SSP-3D loader: labels, silhouettes, and SMPL ground-truth measurements (research R12, FR-018).

SSP-3D (Sengupta et al., "Synthetic Training for Accurate 3D Human Pose and Shape Estimation in the
Wild", BMVC 2020, https://github.com/akashsengupta1997/SSP-3D, MIT license) holds 311 photographs of
tightly clothed sports persons. Its ``labels.npz`` holds file names, SMPL pose and shape
coefficients, genders, 2D joints, camera translations, and bounding boxes. The README does not list
the key names `[assumed; research R12]`, so ``load_labels`` logs every key with its shape and finds
each field among the likely names. A missing field raises ``Ssp3dError``, which lists the keys that
the file does hold.

Ground truth (FR-018). The five measurements of a subject come from the SMPL mesh of the subject's
gender and shape coefficients, in the canonical pose, with the definitions of ``measure.py``
(research R5 and R6). The pose of the label is ignored, because measurements depend on shape only
(FR-004). ``SmplBody`` reads the licensed model files from the asset root; a test passes the
procedural stand-in body through ``body_factory`` instead.

Silhouettes. The dataset provides one silhouette per photograph, in ``silhouettes/<name>.png``
`[assumed; the layout is not in the README]`. ``load_silhouette`` returns it as a 0 and 1 mask, or
``None`` when the file is absent (status ``skipped_no_mask`` in research R12). Mask preprocessing
and the usability rule belong to ``real/common.py``.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from strike_a_pose.assets import require_asset
from strike_a_pose.body.base import BodyModel, canonical_mesh
from strike_a_pose.body.smpl_body import SMPL_ASSET_KEY, SmplBody
from strike_a_pose.measure import measure_batch

__all__ = [
    "LABELS_FILE",
    "SILHOUETTE_FOLDER",
    "Ssp3dError",
    "Ssp3dLabels",
    "ground_truth_measurements",
    "load_labels",
    "load_silhouette",
    "normalise_gender",
    "smpl_body_factory",
]

logger = logging.getLogger(__name__)

LABELS_FILE = "labels.npz"
SILHOUETTE_FOLDER = "silhouettes"

# Likely key names of each field of labels.npz, in the order that they are tried `[assumed]`.
_FILENAME_KEYS = ("fnames", "filenames", "file_names", "image_names", "imgnames")
_SHAPE_KEYS = ("shapes", "betas", "shape", "shape_params")
_GENDER_KEYS = ("genders", "gender")
_BOX_KEYS = ("bbox_whs", "bboxes", "bbox", "boxes")

_GENDER_NAMES = {"m": "male", "male": "male", "f": "female", "female": "female"}

# A silhouette pixel above this grey value belongs to the person.
_SILHOUETTE_THRESHOLD = 127


class Ssp3dError(ValueError):
    """The SSP-3D labels or files do not have the expected content."""


@dataclass(frozen=True, eq=False)
class Ssp3dLabels:
    """The fields of ``labels.npz`` that this package uses; N is the number of subjects.

    ``boxes`` is the bounding-box array of the dataset exactly as stored (the box prompt of SAM 2
    in research R12), or ``None`` when the file has none.
    """

    keys: tuple[str, ...]
    filenames: tuple[str, ...]
    betas: NDArray[np.float64]  # shape (N, n_betas)
    genders: tuple[str, ...]  # "male" or "female"
    boxes: NDArray[np.float64] | None

    def __len__(self) -> int:
        return len(self.filenames)


def normalise_gender(value: object) -> str:
    """Return "male" or "female" for a stored gender such as "m", b"f", or "Male"."""
    text = value.decode("ascii", errors="replace") if isinstance(value, bytes) else str(value)
    name = _GENDER_NAMES.get(text.strip().lower())
    if name is None:
        raise Ssp3dError(f"unknown gender {value!r}; expected m, f, male, or female")
    return name


def _find_key(keys: Iterable[str], candidates: Sequence[str], field: str) -> str:
    available = tuple(keys)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise Ssp3dError(
        f"{LABELS_FILE} has no {field} field; tried {', '.join(candidates)}. "
        f"Keys present: {', '.join(available) or 'none'}"
    )


def load_labels(dataset_dir: str | Path) -> Ssp3dLabels:
    """Read ``<dataset_dir>/labels.npz`` and log its keys with their shapes.

    Raises ``MissingAssetError`` (naming ``real.ssp3d.path``) when the file is absent, and
    ``Ssp3dError`` when a needed field is missing or the fields disagree in length.
    """
    path = require_asset(str(Path(dataset_dir) / LABELS_FILE), SMPL_ASSET_KEY, None)
    try:
        archive = np.load(path, allow_pickle=False)
    except ValueError as error:
        raise Ssp3dError(
            f"{path} needs pickled arrays, which this loader refuses: {error}"
        ) from error
    with archive:
        keys = tuple(archive.files)
        for key in keys:
            logger.info(
                "%s key %s: shape %s, dtype %s",
                LABELS_FILE,
                key,
                archive[key].shape,
                archive[key].dtype,
            )
        filenames = tuple(
            str(name) for name in archive[_find_key(keys, _FILENAME_KEYS, "file name")]
        )
        betas = np.asarray(archive[_find_key(keys, _SHAPE_KEYS, "shape")], dtype=np.float64)
        genders = tuple(
            normalise_gender(g) for g in archive[_find_key(keys, _GENDER_KEYS, "gender")]
        )
        box_keys = [key for key in _BOX_KEYS if key in keys]
        boxes = np.asarray(archive[box_keys[0]], dtype=np.float64) if box_keys else None
    count = len(filenames)
    if betas.ndim != 2 or betas.shape[0] != count or len(genders) != count:
        raise Ssp3dError(
            f"{LABELS_FILE} fields disagree: {count} file names, shapes {betas.shape}, "
            f"{len(genders)} genders"
        )
    if boxes is not None and boxes.shape[0] != count:
        raise Ssp3dError(f"{LABELS_FILE} has {boxes.shape[0]} boxes for {count} file names")
    return Ssp3dLabels(keys=keys, filenames=filenames, betas=betas, genders=genders, boxes=boxes)


def load_silhouette(dataset_dir: str | Path, filename: str) -> NDArray[np.uint8] | None:
    """Return the provided silhouette of a photograph as a 0 and 1 mask, or None when absent."""
    path = Path(dataset_dir) / SILHOUETTE_FOLDER / f"{Path(filename).stem}.png"
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("silhouette %s cannot be decoded; treated as absent", path)
        return None
    return (image > _SILHOUETTE_THRESHOLD).astype(np.uint8)


def smpl_body_factory(asset_root: str | Path | None) -> Callable[[str], BodyModel]:
    """Return a function that loads the SMPL model of a gender from the asset root (research R5)."""
    root = None if asset_root is None else Path(asset_root)
    return lambda gender: SmplBody(root, gender)


def ground_truth_measurements(
    labels: Ssp3dLabels,
    body_factory: Callable[[str], BodyModel],
    step_cm: float,
    *,
    indices: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Return height, chest, waist, hip, and thigh in cm for subjects, shape (len(indices), 5).

    Each subject's mesh is the canonical-pose body of its gender and shape coefficients (FR-018).
    ``body_factory(gender)`` returns the body model for ``"male"`` or ``"female"``; it is called
    once per gender present. ``indices`` selects subjects (default: all, in file order). A row is
    NaN where a measurement slice is degenerate (``measure.py``).
    """
    chosen = list(range(len(labels))) if indices is None else [int(i) for i in indices]
    result = np.full((len(chosen), 5), np.nan, dtype=np.float64)
    bodies: dict[str, BodyModel] = {}
    for gender in sorted({labels.genders[i] for i in chosen}):
        positions = [row for row, i in enumerate(chosen) if labels.genders[i] == gender]
        body = bodies.setdefault(gender, body_factory(gender))
        betas = labels.betas[[chosen[row] for row in positions]]
        vertices, joints = canonical_mesh(body, betas)
        measured = measure_batch(vertices, body.faces, body.part_ids, joints, step_cm)
        result[positions] = measured.cpu().numpy()
    return result
