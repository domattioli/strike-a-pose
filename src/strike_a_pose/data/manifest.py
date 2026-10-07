"""Manifest and shard I/O: manifest.csv rows, shard npz files, packed silhouettes, atomic writes.

The column names, shard keys, and file layout follow contracts/artifacts.md, and the rows follow
the BodySample and CameraPlacement entities of data-model.md. The pose widths come from
body/base.py. Every float is written with repr, the shortest text that reads back as the same
float, so two runs with the same seed write the same bytes (FR-024, research R9). CSV lines end
with a bare line feed on every platform. Each file reaches its final name through
checkpoint.atomic_path, so a partial file never carries a final name. Silhouettes are packed one
bit per pixel with numpy.packbits and unpacked with numpy.unpackbits, as documented in the NumPy
reference at https://numpy.org/doc/stable/reference/generated/numpy.packbits.html. This module
implements no published method, so it cites no algorithm source.
"""

import csv
import io
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strike_a_pose.body.base import BODY_POSE_SIZE, ROOT_POSE_SIZE
from strike_a_pose.checkpoint import atomic_path

__all__ = [
    "BETA_COUNT",
    "CAMERA_COUNT",
    "FLAG_BITS",
    "FLAG_NAMES",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "MEASUREMENT_NAMES",
    "POSE_SOURCES",
    "SHARD_DIRECTORY_NAME",
    "SHARD_KEYS",
    "SPLITS",
    "ManifestRow",
    "ShardArrays",
    "flags_from_text",
    "flags_to_text",
    "format_float",
    "read_manifest",
    "read_shard",
    "render_manifest",
    "shard_path",
    "write_manifest",
    "write_shard",
]

# The manifest file name inside the data directory, and the folder that holds the shard files.
MANIFEST_NAME = "manifest.csv"
SHARD_DIRECTORY_NAME = "shards"

# Every rig has four cameras, so the manifest and every shard carry four views per body.
CAMERA_COUNT = 4
# Shape coefficients per body. The manifest names them betas_0 to betas_9.
BETA_COUNT = 10
# The five measurements in column order, in centimeters (FR-004).
MEASUREMENT_NAMES: tuple[str, ...] = ("height", "chest", "waist", "hip", "thigh")
# The data splits by body id (data-model.md, DataSplit).
SPLITS: tuple[str, ...] = ("train", "cal", "test")
# The pose sources of the configuration (contracts/config.md, pose.source).
POSE_SOURCES: tuple[str, ...] = ("amass", "limits")
# The flag names. Flag FLAG_NAMES[i] is bit i of the flags bit field (data-model.md, Body flags).
FLAG_NAMES: tuple[str, ...] = ("empty_mask", "out_of_frame", "slice_nan")
FLAG_BITS: Mapping[str, int] = MappingProxyType(
    {name: 1 << index for index, name in enumerate(FLAG_NAMES)}
)


def _camera_columns(camera: int) -> tuple[str, ...]:
    """Return the six manifest columns of one camera, in contract order."""
    prefix = f"cam{camera}_"
    return (
        f"{prefix}azimuth_deg",
        f"{prefix}distance_m",
        f"{prefix}height_m",
        f"{prefix}noise_axis_x",
        f"{prefix}noise_axis_y",
        f"{prefix}noise_axis_z",
    )


# The manifest header of contracts/artifacts.md: 111 columns in this order.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "body_id",
    "shard",
    "split",
    "pose_source",
    "pose_rejections",
    "flags",
    *(f"betas_{index}" for index in range(BETA_COUNT)),
    *(f"pose_root_{index}" for index in range(ROOT_POSE_SIZE)),
    *(f"pose_body_{index}" for index in range(BODY_POSE_SIZE)),
    *(column for camera in range(CAMERA_COUNT) for column in _camera_columns(camera)),
    *(f"{name}_cm" for name in MEASUREMENT_NAMES),
)

# The arrays of one shard file, in the order they are written (contracts/artifacts.md).
SHARD_KEYS: tuple[str, ...] = (
    "body_id",
    "masks",
    "K",
    "R_true",
    "t_true",
    "noise_axis",
    "betas",
    "pose_root",
    "pose_body",
    "measurements",
    "flags",
)


def format_float(value: float) -> str:
    """Return the repr of a float: the shortest text that reads back as the same float.

    The value is converted to a built-in float first. Under NumPy 2, repr of a numpy float64
    prints as np.float64(0.1), which would make the file depend on the array type.
    """
    return repr(float(value))


def flags_to_text(flags: int) -> str:
    """Return the names of the flags set in the bit field, in bit order, joined by semicolons.

    A body with no flags gives the empty text, so the flags cell of a clean body is empty.
    """
    value = _as_count("flags", flags)
    if value >= 1 << len(FLAG_NAMES):
        raise ValueError(f"flags must be a bit field below {1 << len(FLAG_NAMES)}; got {value}")
    return ";".join(name for name in FLAG_NAMES if value & FLAG_BITS[name])


def flags_from_text(text: str) -> int:
    """Return the bit field of semicolon-separated flag names; the empty text means no flags."""
    if text == "":
        return 0
    value = 0
    for name in text.split(";"):
        if name not in FLAG_BITS:
            raise ValueError(f"unknown flag {name!r}; the flags are {', '.join(FLAG_NAMES)}")
        value |= FLAG_BITS[name]
    return value


def _as_count(name: str, value: object, minimum: int = 0) -> int:
    """Return value as a built-in int, or raise ValueError when it is not an integer."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {number}")
    return number


def _name_text(value: object) -> str:
    """Return the text of a name given as a string or as a member of a str-based Enum.

    The member's value is used. str() would print Split.TRAIN for a (str, Enum) member, which is
    not a name of the manifest.
    """
    if isinstance(value, Enum):
        value = value.value
    return str(value)


def _freeze(nested: Any) -> Any:
    """Turn nested lists into nested tuples and leave the values in place."""
    if isinstance(nested, list):
        return tuple(_freeze(item) for item in nested)
    return nested


def _as_floats(name: str, value: ArrayLike, shape: tuple[int, ...]) -> Any:
    """Return value as nested tuples of built-in floats of the given shape, or raise ValueError."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return _freeze(array.tolist())


@dataclass(frozen=True)
class ManifestRow:
    """One body's row of manifest.csv: identity, split, pose, camera rig, and measurements.

    The arrays may be given as lists or NumPy arrays. Construction checks every field and stores
    the arrays as tuples of built-in floats, so rows with the same values compare equal. The
    camera fields hold one value per camera (view), and noise_axis holds one unit vector per
    camera. flags is the bit field of FLAG_NAMES. measurements_cm follows MEASUREMENT_NAMES.
    """

    body_id: int
    shard: int
    split: str
    pose_source: str
    pose_rejections: int
    flags: int
    betas: tuple[float, ...]
    pose_root: tuple[float, ...]
    pose_body: tuple[float, ...]
    azimuth_deg: tuple[float, ...]
    distance_m: tuple[float, ...]
    height_m: tuple[float, ...]
    noise_axis: tuple[tuple[float, float, float], ...]
    measurements_cm: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check every field and store it in its normalized form."""
        split = _name_text(self.split)
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}; got {split!r}")
        pose_source = _name_text(self.pose_source)
        if pose_source not in POSE_SOURCES:
            raise ValueError(
                f"pose_source must be one of {', '.join(POSE_SOURCES)}; got {pose_source!r}"
            )
        normalized: dict[str, Any] = {
            "body_id": _as_count("body_id", self.body_id),
            "shard": _as_count("shard", self.shard),
            "split": split,
            "pose_source": pose_source,
            "pose_rejections": _as_count("pose_rejections", self.pose_rejections),
            "flags": _as_count("flags", self.flags),
            "betas": _as_floats("betas", self.betas, (BETA_COUNT,)),
            "pose_root": _as_floats("pose_root", self.pose_root, (ROOT_POSE_SIZE,)),
            "pose_body": _as_floats("pose_body", self.pose_body, (BODY_POSE_SIZE,)),
            "azimuth_deg": _as_floats("azimuth_deg", self.azimuth_deg, (CAMERA_COUNT,)),
            "distance_m": _as_floats("distance_m", self.distance_m, (CAMERA_COUNT,)),
            "height_m": _as_floats("height_m", self.height_m, (CAMERA_COUNT,)),
            "noise_axis": _as_floats("noise_axis", self.noise_axis, (CAMERA_COUNT, 3)),
            "measurements_cm": _as_floats(
                "measurements_cm", self.measurements_cm, (len(MEASUREMENT_NAMES),)
            ),
        }
        if normalized["flags"] >= 1 << len(FLAG_NAMES):
            raise ValueError(
                f"flags must be a bit field below {1 << len(FLAG_NAMES)}; got {normalized['flags']}"
            )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


def _row_cells(row: ManifestRow) -> list[str]:
    """Return the text of every column of one row, in the order of MANIFEST_COLUMNS."""
    cells = [
        str(row.body_id),
        str(row.shard),
        row.split,
        row.pose_source,
        str(row.pose_rejections),
        flags_to_text(row.flags),
    ]
    cells += [format_float(value) for value in row.betas]
    cells += [format_float(value) for value in row.pose_root]
    cells += [format_float(value) for value in row.pose_body]
    for camera in range(CAMERA_COUNT):
        cells += [
            format_float(row.azimuth_deg[camera]),
            format_float(row.distance_m[camera]),
            format_float(row.height_m[camera]),
        ]
        cells += [format_float(value) for value in row.noise_axis[camera]]
    cells += [format_float(value) for value in row.measurements_cm]
    return cells


def _row_from_fields(fields: Mapping[str, str]) -> ManifestRow:
    """Build one row from the text of its columns, keyed by column name."""
    return ManifestRow(
        body_id=int(fields["body_id"]),
        shard=int(fields["shard"]),
        split=fields["split"],
        pose_source=fields["pose_source"],
        pose_rejections=int(fields["pose_rejections"]),
        flags=flags_from_text(fields["flags"]),
        betas=[float(fields[f"betas_{index}"]) for index in range(BETA_COUNT)],
        pose_root=[float(fields[f"pose_root_{index}"]) for index in range(ROOT_POSE_SIZE)],
        pose_body=[float(fields[f"pose_body_{index}"]) for index in range(BODY_POSE_SIZE)],
        azimuth_deg=[float(fields[f"cam{camera}_azimuth_deg"]) for camera in range(CAMERA_COUNT)],
        distance_m=[float(fields[f"cam{camera}_distance_m"]) for camera in range(CAMERA_COUNT)],
        height_m=[float(fields[f"cam{camera}_height_m"]) for camera in range(CAMERA_COUNT)],
        noise_axis=[
            [float(fields[f"cam{camera}_noise_axis_{axis}"]) for axis in ("x", "y", "z")]
            for camera in range(CAMERA_COUNT)
        ],
        measurements_cm=[float(fields[f"{name}_cm"]) for name in MEASUREMENT_NAMES],
    )


def render_manifest(rows: Iterable[ManifestRow]) -> str:
    """Return the manifest as CSV text: the header row, then one row per body in the order given.

    The text depends only on the rows and their order, so the same rows always give the same text.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for row in rows:
        writer.writerow(_row_cells(row))
    return buffer.getvalue()


def write_manifest(destination: str | os.PathLike[str], rows: Iterable[ManifestRow]) -> Path:
    """Write the manifest CSV atomically and return its path.

    The bytes are the UTF-8 encoding of render_manifest, written with explicit line feeds. The
    text is encoded here rather than written in text mode, because text mode on Windows would turn
    each line feed into a carriage return and line feed pair.
    """
    text = render_manifest(rows)
    final = Path(destination)
    with atomic_path(final) as temporary:
        temporary.write_bytes(text.encode("utf-8"))
    return final


def read_manifest(source: str | os.PathLike[str]) -> list[ManifestRow]:
    """Read a manifest CSV back into rows, after checking its header against MANIFEST_COLUMNS."""
    path = Path(source)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if tuple(header) != MANIFEST_COLUMNS:
            raise ValueError(f"{path} does not carry the manifest header of contracts/artifacts.md")
        rows: list[ManifestRow] = []
        for line_number, cells in enumerate(reader, start=2):
            if len(cells) != len(MANIFEST_COLUMNS):
                raise ValueError(
                    f"{path} line {line_number} has {len(cells)} columns; "
                    f"the manifest has {len(MANIFEST_COLUMNS)}"
                )
            try:
                rows.append(_row_from_fields(dict(zip(MANIFEST_COLUMNS, cells, strict=True))))
            except ValueError as error:
                raise ValueError(f"{path} line {line_number}: {error}") from error
    return rows


def shard_path(data_directory: str | os.PathLike[str], index: int) -> Path:
    """Return data/shards/shard_NNNN.npz for shard number index, with four zero-padded digits."""
    number = _as_count("shard index", index)
    return Path(data_directory) / SHARD_DIRECTORY_NAME / f"shard_{number:04d}.npz"


def _packed_width(image_size: int) -> int:
    """Return the number of bytes that hold one square mask of image_size pixels per side."""
    return (image_size * image_size + 7) // 8


def _shard_layout(count: int, image_size: int) -> dict[str, tuple[tuple[int, ...], type]]:
    """Return the shape and dtype of every shard array for count bodies and the mask side."""
    return {
        "body_id": ((count,), np.int64),
        "masks": ((count, CAMERA_COUNT, _packed_width(image_size)), np.uint8),
        "K": ((3, 3), np.float64),
        "R_true": ((count, CAMERA_COUNT, 3, 3), np.float64),
        "t_true": ((count, CAMERA_COUNT, 3), np.float64),
        "noise_axis": ((count, CAMERA_COUNT, 3), np.float64),
        "betas": ((count, BETA_COUNT), np.float64),
        "pose_root": ((count, ROOT_POSE_SIZE), np.float64),
        "pose_body": ((count, BODY_POSE_SIZE), np.float64),
        "measurements": ((count, len(MEASUREMENT_NAMES)), np.float64),
        "flags": ((count,), np.int64),
    }


@dataclass(frozen=True, eq=False)
class ShardArrays:
    """The arrays of one shard file, with the masks unpacked to one value per pixel.

    The shapes follow contracts/artifacts.md. Every array except K starts with the body axis. The
    per-camera arrays put the camera axis second, after the body axis. The field names are the
    keys of the npz file.
    """

    body_id: NDArray[np.int64]  # (bodies,)
    masks: NDArray[np.uint8]  # (bodies, cameras, side, side), values 0 and 1
    K: NDArray[np.float64]  # (3, 3), shared by the shard
    R_true: NDArray[np.float64]  # (bodies, cameras, 3, 3)
    t_true: NDArray[np.float64]  # (bodies, cameras, 3)
    noise_axis: NDArray[np.float64]  # (bodies, cameras, 3)
    betas: NDArray[np.float64]  # (bodies, 10)
    pose_root: NDArray[np.float64]  # (bodies, 3)
    pose_body: NDArray[np.float64]  # (bodies, 63)
    measurements: NDArray[np.float64]  # (bodies, 5), centimeters
    flags: NDArray[np.int64]  # (bodies,), bit field of FLAG_NAMES


def write_shard(
    destination: str | os.PathLike[str],
    *,
    body_id: ArrayLike,
    masks: ArrayLike,
    K: ArrayLike,
    R_true: ArrayLike,
    t_true: ArrayLike,
    noise_axis: ArrayLike,
    betas: ArrayLike,
    pose_root: ArrayLike,
    pose_body: ArrayLike,
    measurements: ArrayLike,
    flags: ArrayLike,
) -> Path:
    """Write one shard npz atomically, packing the masks one bit per pixel, and return its path.

    masks holds one value per pixel with shape (bodies, cameras, side, side) and values 0 and 1.
    Every other argument has the shape of its ShardArrays field. Each array is converted to the
    dtype of the file (int64 or float64, uint8 for packed masks), and a shape or dtype that cannot
    be converted raises ValueError. The arrays are written in SHARD_KEYS order, so repeated writes
    of the same arrays give the same bytes.
    """
    mask_array = np.asarray(masks)
    if (
        mask_array.ndim != 4
        or mask_array.shape[1] != CAMERA_COUNT
        or mask_array.shape[2] != mask_array.shape[3]
    ):
        raise ValueError(
            f"masks must have shape (bodies, {CAMERA_COUNT}, side, side); got {mask_array.shape}"
        )
    if not np.isin(mask_array, (0, 1)).all():
        raise ValueError("masks must hold only the values 0 and 1")
    count = mask_array.shape[0]
    side = mask_array.shape[2]
    packed = np.packbits(mask_array.astype(bool).reshape(count, CAMERA_COUNT, side * side), axis=-1)
    given: dict[str, ArrayLike] = {
        "body_id": body_id,
        "masks": packed,
        "K": K,
        "R_true": R_true,
        "t_true": t_true,
        "noise_axis": noise_axis,
        "betas": betas,
        "pose_root": pose_root,
        "pose_body": pose_body,
        "measurements": measurements,
        "flags": flags,
    }
    layout = _shard_layout(count, side)
    arrays: dict[str, np.ndarray] = {}
    for key in SHARD_KEYS:
        shape, dtype = layout[key]
        array = np.asarray(given[key])
        if array.shape != shape:
            raise ValueError(f"{key} must have shape {shape}; got {array.shape}")
        try:
            arrays[key] = array.astype(dtype, casting="same_kind")
        except TypeError as error:
            raise ValueError(
                f"{key} cannot be stored as {np.dtype(dtype).name}; got {array.dtype.name}"
            ) from error
    final = Path(destination)
    with atomic_path(final) as temporary:
        np.savez(temporary, **arrays)
    return final


def read_shard(source: str | os.PathLike[str], image_size: int) -> ShardArrays:
    """Read one shard npz and unpack its masks to image_size by image_size pixels.

    image_size is camera.image_size of the configuration. The file must hold exactly the keys of
    SHARD_KEYS, with the shapes and dtypes that write_shard writes; anything else raises ValueError.
    """
    side = _as_count("image_size", image_size, minimum=1)
    with np.load(source, allow_pickle=False) as archive:
        stored = sorted(archive.files)
        if stored != sorted(SHARD_KEYS):
            raise ValueError(
                f"{source} must hold the arrays {', '.join(SHARD_KEYS)}; it holds "
                f"{', '.join(stored)}"
            )
        arrays = {key: archive[key] for key in SHARD_KEYS}
    body_id = arrays["body_id"]
    if body_id.ndim != 1:
        raise ValueError(f"{source}: body_id must be one-dimensional; got shape {body_id.shape}")
    count = body_id.shape[0]
    for key, (shape, dtype) in _shard_layout(count, side).items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(
                f"{source}: {key} must have shape {shape} and dtype {np.dtype(dtype).name}; "
                f"got shape {arrays[key].shape} and dtype {arrays[key].dtype.name}"
            )
    masks = np.unpackbits(arrays["masks"], axis=-1, count=side * side).reshape(
        count, CAMERA_COUNT, side, side
    )
    return ShardArrays(
        body_id=arrays["body_id"],
        masks=masks,
        K=arrays["K"],
        R_true=arrays["R_true"],
        t_true=arrays["t_true"],
        noise_axis=arrays["noise_axis"],
        betas=arrays["betas"],
        pose_root=arrays["pose_root"],
        pose_body=arrays["pose_body"],
        measurements=arrays["measurements"],
        flags=arrays["flags"],
    )
