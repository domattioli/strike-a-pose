"""BodyM loader: split folders, header-named CSV columns, subject-to-photo joins, tape values.

The loader reads the BodyM layout of research R12 from the folder that the operator placed under
real.bodym.path (FR-017). For each split it finds the frontal and side mask folders and the
measurements.csv, hwg_metadata.csv, and subject_to_photo_map.csv files. It finds CSV columns by
header name, so column order does not matter, and it logs every column and folder that it uses.
Each subject is joined to one front mask and one side mask that share a photo id. A subject is
skipped and counted when it has no photo map row, no shared front and side photo id, or no finite
value for one of the five tape measurements; the first failing reason is the one counted. Masks
are returned as paths and are not opened here, because mask usability is decided in real/common.py.
The public source is the BodyM dataset page https://registry.opendata.aws/bodym/ (research R12).
The folder and file names are assumptions of R12, and the loader logs what it finds.
"""

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from strike_a_pose.assets import MissingAssetError

__all__ = [
    "BODYM_PATH_KEY",
    "BODYM_SPLITS",
    "MEASUREMENT_FILE",
    "MEASUREMENT_NAMES",
    "METADATA_FILE",
    "PHOTO_MAP_FILE",
    "SKIP_REASONS",
    "BodyMSplitLayout",
    "BodyMSplitLoad",
    "BodyMSubject",
    "discover_split",
    "load_bodym",
    "load_split",
]

logger = logging.getLogger(__name__)

BODYM_PATH_KEY = "real.bodym.path"
BODYM_SPLITS = ("testA", "testB")
MEASUREMENT_NAMES = ("height", "chest", "waist", "hip", "thigh")
MEASUREMENT_FILE = "measurements.csv"
METADATA_FILE = "hwg_metadata.csv"
PHOTO_MAP_FILE = "subject_to_photo_map.csv"
# Reasons for skipping a subject, in the order that the loader checks them.
SKIP_REASONS = ("no_photo_map", "no_mask_pair", "missing_measurement")


@dataclass(frozen=True)
class BodyMSplitLayout:
    """The folders and files of one BodyM split, found on disk."""

    split: str
    folder: Path
    front_folder: Path
    side_folder: Path
    measurements_csv: Path
    metadata_csv: Path
    photo_map_csv: Path


@dataclass(frozen=True)
class BodyMSubject:
    """One BodyM subject with its front and side masks and five tape measurements in cm.

    The measurements follow MEASUREMENT_NAMES: height, chest, waist, hip, thigh.
    """

    split: str
    subject_id: str
    front_mask: Path
    side_mask: Path
    measurements_cm: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class BodyMSplitLoad:
    """The subjects of one split, the skip counts by reason, and the columns that were used.

    columns maps each measurement name to "<file name>:<header>", and the keys "subject" and
    "photo" name the subject and photo columns the same way.
    """

    split: str
    subjects: tuple[BodyMSubject, ...]
    skipped: Mapping[str, int]
    columns: Mapping[str, str] = field(default_factory=dict)


def _normal(text: str) -> str:
    """Return text in lower case with every character that is not a letter or digit removed."""
    return "".join(character for character in text.lower() if character.isalnum())


def _single_child_folder(folder: Path, keyword: str, split: str) -> Path:
    """Return the one child folder of a split whose name contains keyword."""
    matches = [
        child
        for child in sorted(folder.iterdir())
        if child.is_dir() and keyword in _normal(child.name)
    ]
    if not matches:
        raise MissingAssetError(folder / f"{keyword} mask folder of {split}", BODYM_PATH_KEY)
    if len(matches) > 1:
        names = [child.name for child in matches]
        raise ValueError(f"BodyM {split} has more than one {keyword} mask folder: {names}")
    return matches[0]


def discover_split(root: Path, split: str) -> BodyMSplitLayout:
    """Find the folders and files of one split under the BodyM root.

    The split folder is the child of root whose name equals the split name once case and
    punctuation are removed, so "Test-A", "test_a", and "testA" all name split testA. Raise
    MissingAssetError when the root or a split folder or file is absent, and ValueError when a
    name is ambiguous.
    """
    if not root.is_dir():
        raise MissingAssetError(root, BODYM_PATH_KEY)
    wanted = _normal(split)
    folders = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and _normal(child.name) == wanted
    ]
    if not folders:
        raise MissingAssetError(root / split, BODYM_PATH_KEY)
    if len(folders) > 1:
        raise ValueError(f"BodyM root {root} has more than one folder for {split}: {folders}")
    folder = folders[0]
    files = {child.name.lower(): child for child in folder.iterdir() if child.is_file()}
    paths: dict[str, Path] = {}
    for name in (MEASUREMENT_FILE, METADATA_FILE, PHOTO_MAP_FILE):
        if name.lower() not in files:
            raise MissingAssetError(folder / name, BODYM_PATH_KEY)
        paths[name] = files[name.lower()]
    return BodyMSplitLayout(
        split=split,
        folder=folder,
        front_folder=_single_child_folder(folder, "front", split),
        side_folder=_single_child_folder(folder, "side", split),
        measurements_csv=paths[MEASUREMENT_FILE],
        metadata_csv=paths[METADATA_FILE],
        photo_map_csv=paths[PHOTO_MAP_FILE],
    )


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header names and the rows of a CSV file. A byte order mark is ignored."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [
            {key: value or "" for key, value in row.items() if key is not None} for row in reader
        ]
    return headers, rows


def _find_column(headers: Sequence[str], keyword: str, path: Path) -> str | None:
    """Return the one header whose normalized name contains keyword, or None when there is none.

    Raise ValueError when more than one header contains keyword, because the loader must not
    guess which column is the measurement.
    """
    matches = [header for header in headers if keyword in _normal(header)]
    if len(matches) > 1:
        raise ValueError(f"{path.name} has more than one column for {keyword!r}: {matches}")
    return matches[0] if matches else None


def _require_column(headers: Sequence[str], keyword: str, path: Path) -> str:
    """Return the one header that contains keyword; raise ValueError when there is none."""
    column = _find_column(headers, keyword, path)
    if column is None:
        raise ValueError(f"{path.name} has no column for {keyword!r}; headers are {list(headers)}")
    return column


def _finite(raw: str | None) -> float | None:
    """Return the value as a finite float, or None when it is absent, blank, or not a number."""
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _index_masks(folder: Path) -> dict[str, Path]:
    """Return the files of a mask folder keyed by file stem, in sorted name order."""
    index: dict[str, Path] = {}
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.stem not in index:
            index[child.stem] = child
    return index


def _measurement_source(
    name: str,
    tables: Sequence[tuple[Path, list[str], list[dict[str, str]], str]],
) -> tuple[str, str, dict[str, str]]:
    """Return (file name, header, subject-to-raw-value) for one measurement.

    tables holds (path, headers, rows, subject column) in the order to search. The first table
    with a column for the measurement wins, so the measurements file is searched before the
    metadata file.
    """
    for path, headers, rows, subject_column in tables:
        column = _find_column(headers, name, path)
        if column is not None:
            values = {row[subject_column].strip(): row[column] for row in rows}
            return path.name, column, values
    names = [path.name for path, _, _, _ in tables]
    raise ValueError(f"no column for the {name} measurement in {names}")


def load_split(root: Path, split: str) -> BodyMSplitLoad:
    """Load the subjects of one split from the BodyM root (FR-017).

    The subjects keep the order of measurements.csv. Raise ValueError for a subject id that
    appears twice in measurements.csv, and for a file that lacks a required column.
    """
    layout = discover_split(root, split)
    headers_m, rows_m = _read_table(layout.measurements_csv)
    headers_d, rows_d = _read_table(layout.metadata_csv)
    headers_p, rows_p = _read_table(layout.photo_map_csv)

    subject_m = _require_column(headers_m, "subject", layout.measurements_csv)
    subject_p = _require_column(headers_p, "subject", layout.photo_map_csv)
    photo_p = _require_column(headers_p, "photo", layout.photo_map_csv)
    subject_d = _find_column(headers_d, "subject", layout.metadata_csv)

    sources: dict[str, tuple[str, str, dict[str, str]]] = {}
    for name in MEASUREMENT_NAMES:
        tables = [(layout.measurements_csv, headers_m, rows_m, subject_m)]
        if subject_d is not None:
            tables.append((layout.metadata_csv, headers_d, rows_d, subject_d))
        sources[name] = _measurement_source(name, tables)

    photos: dict[str, list[str]] = {}
    for row in rows_p:
        subject_id = row[subject_p].strip()
        photo_id = row[photo_p].strip()
        if not subject_id or not photo_id:
            continue
        ids = photos.setdefault(subject_id, [])
        if photo_id not in ids:
            ids.append(photo_id)

    front_masks = _index_masks(layout.front_folder)
    side_masks = _index_masks(layout.side_folder)
    skipped = {reason: 0 for reason in SKIP_REASONS}
    subjects: list[BodyMSubject] = []
    seen: set[str] = set()
    for row in rows_m:
        subject_id = row[subject_m].strip()
        if not subject_id:
            continue
        if subject_id in seen:
            raise ValueError(f"{layout.measurements_csv.name} lists subject {subject_id!r} twice")
        seen.add(subject_id)

        if subject_id not in photos:
            skipped["no_photo_map"] += 1
            continue
        photo_id = next(
            (photo for photo in photos[subject_id] if photo in front_masks and photo in side_masks),
            None,
        )
        if photo_id is None:
            skipped["no_mask_pair"] += 1
            continue
        values = [_finite(sources[name][2].get(subject_id)) for name in MEASUREMENT_NAMES]
        if any(value is None for value in values):
            skipped["missing_measurement"] += 1
            continue
        subjects.append(
            BodyMSubject(
                split=split,
                subject_id=subject_id,
                front_mask=front_masks[photo_id],
                side_mask=side_masks[photo_id],
                measurements_cm=tuple(values),  # type: ignore[arg-type]
            )
        )

    columns = {name: f"{file_name}:{header}" for name, (file_name, header, _) in sources.items()}
    columns["subject"] = f"{layout.measurements_csv.name}:{subject_m}"
    columns["photo"] = f"{layout.photo_map_csv.name}:{photo_p}"
    logger.info(
        "BodyM %s: folder %s, front masks %s, side masks %s, columns %s",
        split,
        layout.folder,
        layout.front_folder.name,
        layout.side_folder.name,
        columns,
    )
    logger.info("BodyM %s: %d subjects evaluated, skipped %s", split, len(subjects), skipped)
    return BodyMSplitLoad(
        split=split,
        subjects=tuple(subjects),
        skipped=skipped,
        columns=columns,
    )


def load_bodym(root: Path, splits: Sequence[str] = BODYM_SPLITS) -> dict[str, BodyMSplitLoad]:
    """Load each requested split from the BodyM root, keyed by split name.

    The root is the resolved real.bodym.path (see assets.require_asset). Nothing is downloaded.
    """
    return {split: load_split(root, split) for split in splits}
