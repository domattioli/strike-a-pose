"""Sharded generation of synthetic bodies: shapes, poses, camera rigs, silhouettes, measurements.

This is the generate stage of specs/001-kill-test-mvp (FR-001 to FR-005, FR-024, FR-029). It writes
everything under ``<out>/data`` (contracts/artifacts.md). Each body gets the following, in this
order (data-model.md, BodySample and CameraPlacement):

1. Shape coefficients: ``body.n_betas`` standard normal draws, clipped to plus or minus
   ``body.beta_clip``.
2. A pose from the source that ``pose.source`` names. A draw that the pose filter rejects
   (pose/filters.py, research R3) is redrawn, and the rejections are counted. A body that needs
   more than ``pose.max_rejections`` rejections stops the stage (exit code 4).
3. The posed mesh, stood on the floor (its lowest vertex at y = 0, because camera heights are
   measured from the floor, camera.py). The rig of four cameras is aimed at the center of the
   mesh's axis-aligned bounding box (``camera.sample_rig``).
4. One silhouette per camera, rendered from the true placement (render.py, research R1). A body
   carries the flag ``empty_mask`` when any view has no pixel set, and ``out_of_frame`` when any
   view touches the image border.
5. The five ground-truth measurements of the canonical-pose mesh of the shape coefficients, with
   the search step ``measure.step_cm`` (measure.py, research R6). A body carries the flag
   ``slice_nan`` when a measurement is not finite. The measurements are computed on the CPU
   whatever device the run uses, so the device never changes the data.

Random streams (research R9). Body ``b`` of shard ``s`` draws from ``rng_for(seed, stage, s, b,
part)``, where ``stage`` is the index of ``generate`` in ``runrecord.STAGES`` and ``part`` is 0 for
the shape coefficients, 1 for the pose, and 2 for the camera rig. This extends the key
``[seed, stage_id, shard, body]`` of research R9 by one element. A shape therefore never changes
with the pose source or the pose limits, and a rig never changes with the number of pose
rejections. The streams of different bodies are independent, so shards can be made in any order.

Shards and resume (FR-029). Shard ``s`` holds the bodies ``s * data.shard_size`` up to the next
multiple, in the files ``shards/shard_NNNN.npz`` and ``shards/shard_NNNN.csv``
(contracts/artifacts.md; data/manifest.py writes both). Each file is written atomically, the npz
first, so a partial file never carries a final name. A shard is done when both files exist, the npz
opens and holds exactly the body ids of the shard, and the CSV holds the same body ids. The stage
writes ``summary.json`` with the configuration hash before the first shard. A resumed run keeps the
done shards only when that hash equals the current one, so shards of another configuration are
never reused. Any other start removes the old shard files first, so shards of two configurations
never mix. A time budget (``checkpoint.TimeBudget``) stops the run between shards.

The end of the stage. After the last shard, ``manifest.csv`` is the shard CSV files joined in shard
order. ``summary.json`` is written next: the counts of contracts/artifacts.md, plus ``n_draws`` and
``rejection_rate``, the share of pose draws that the filter rejected, which FR-001 requires the
stage to report (the rate is also logged). Then the stage refuses, with exit code 4 and without
``DONE.json``, when fewer than ``data.min_unflagged`` unflagged calibration or test bodies remain
(SC-002). The message names both counts. Otherwise ``DONE.json`` is written last.

Use from the command line. ``generate_dataset`` is the whole stage. Its result has
``completed = False`` when the time budget ended the run early, which the command reports with exit
code 7 and the resume command. ``GenerationRefusedError.exit_code`` is the exit code 4 of
contracts/cli.md. ConfigError (exit code 2) and MissingAssetError (exit code 3) are raised before
anything is written, with one exception: a camera rig that cannot be placed
(``camera.min_separation_deg`` too wide for ``camera.n_cameras``) raises its ConfigError from the
first body.

Public sources. This module composes the modules named above and adds no method of its own. Its
parts cite theirs: the OpenCV polygon fill of the renderer (research R1), the convex-hull perimeter
rule of the measurements (research R6), and the NumPy ``SeedSequence`` streams (research R9).
"""

import csv
import json
import logging
import re
import shutil
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

from strike_a_pose.assets import asset_root
from strike_a_pose.body.base import BodyModel, canonical_mesh
from strike_a_pose.body.smplx_body import SmplxBody
from strike_a_pose.body.standin import StandInBody
from strike_a_pose.camera import Rig, intrinsics, sample_rig
from strike_a_pose.checkpoint import (
    DONE_MARKER_NAME,
    TimeBudget,
    atomic_path,
    atomic_write_text,
    clear_done_marker,
    write_done_marker,
)
from strike_a_pose.config import ConfigError, config_hash
from strike_a_pose.data.manifest import (
    BETA_COUNT,
    CAMERA_COUNT,
    FLAG_BITS,
    FLAG_NAMES,
    MANIFEST_NAME,
    SHARD_DIRECTORY_NAME,
    SPLITS,
    ManifestRow,
    read_manifest,
    render_manifest,
    shard_path,
    write_manifest,
    write_shard,
)
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.device import select_device
from strike_a_pose.measure import measure_batch, slice_nan_flags
from strike_a_pose.pose.filters import PoseFilter
from strike_a_pose.pose.source import PoseSource, create_pose_source
from strike_a_pose.render import render_silhouette
from strike_a_pose.runrecord import STAGES, RunRecord, start_run_record
from strike_a_pose.seeding import rng_for

__all__ = [
    "DATA_DIRECTORY_NAME",
    "STAGE_NAME",
    "STAGE_REFUSED_EXIT_CODE",
    "SUMMARY_NAME",
    "GenerationRefusedError",
    "GenerationResult",
    "create_body_model",
    "data_directory",
    "generate_dataset",
]

logger = logging.getLogger(__name__)

# The stage name in run_record.json and DONE.json, and the folder of this stage under --out.
STAGE_NAME = "generate"
DATA_DIRECTORY_NAME = "data"
# The file of the counts, which also records the configuration hash that a resumed run checks.
SUMMARY_NAME = "summary.json"
# The exit code of contracts/cli.md for a stage that refuses to go on.
STAGE_REFUSED_EXIT_CODE = 4

# The first element of every stream key after the seed (research R9).
_STAGE_ID = STAGES.index(STAGE_NAME)
# The last element of the stream key of a body: which part of the body the stream draws.
_BETAS_STREAM = 0
_POSE_STREAM = 1
_RIG_STREAM = 2

# Canonical meshes measured at one time. The result of a mesh does not depend on the chunk
# (measure.py), so this only bounds memory: about 16 MB of float64 vertices for SMPL-X.
_MEASURE_CHUNK = 64

# The names that the stage creates in the data folder. A temporary name is what
# checkpoint.atomic_path gives a file while it is being written.
_SHARD_FILE = re.compile(r"shard_[0-9]{4,}\.(?:npz|csv)")
_TEMPORARY_FILE = re.compile(r"\..+\.temporary\.[a-z]+")


class GenerationRefusedError(RuntimeError):
    """The stage refuses to go on, which the sap command reports with exit code 4.

    Two causes: a body needed more pose draws than ``pose.max_rejections`` allows, or too few
    unflagged calibration or test bodies remain (``data.min_unflagged``, SC-002). The message
    names the body, or both counts.
    """

    exit_code = STAGE_REFUSED_EXIT_CODE


@dataclass(frozen=True, eq=False)
class GenerationResult:
    """What a call of ``generate_dataset`` did, and the contents of ``summary.json``.

    ``completed`` is False when the time budget stopped the stage before the last shard. The sap
    command then exits with code 7, and a later call with ``resume=True`` continues. ``summary``
    holds the fields of summary.json: the counts, and the rejection rate of FR-001.
    """

    data_dir: Path
    summary: Mapping[str, Any]
    completed: bool
    shards_total: int
    shards_generated: int
    shards_reused: int

    @property
    def rejection_rate(self) -> float:
        """Rejected pose draws as a share of all pose draws (FR-001); 0 when nothing was drawn."""
        return float(self.summary["rejection_rate"])


@dataclass(frozen=True)
class _ShardPlan:
    """Which bodies one shard holds: body ids ``first`` up to, not including, ``stop``."""

    index: int
    first: int
    stop: int


@dataclass(frozen=True)
class _ShardStats:
    """The counts of one shard that summary.json needs, taken from its manifest rows."""

    shard: int
    n_bodies: int
    n_flagged: int  # bodies with at least one flag
    n_rejections: int
    flag_counts: Mapping[str, int]  # bodies that carry each flag; a body may carry several
    unflagged: Mapping[str, int]  # bodies without any flag, by split


@dataclass(frozen=True, eq=False)
class _Context:
    """Everything that is built once per run and shared by all bodies."""

    config: Mapping[str, Any]
    body: BodyModel
    faces: NDArray[np.int64]
    part_ids: NDArray[np.int64]
    source: PoseSource
    pose_filter: PoseFilter
    split: DataSplit
    camera_matrix: NDArray[np.float64]  # K, shared by every camera (3, 3)


@dataclass(frozen=True, eq=False)
class _GeneratedBody:
    """One body before its shard is assembled."""

    body_id: int
    betas: NDArray[np.float64]  # (10,)
    pose_root: NDArray[np.float64]  # (3,)
    pose_body: NDArray[np.float64]  # (63,)
    rejections: int
    rig: Rig
    masks: NDArray[np.uint8]  # (cameras, side, side)
    render_flags: int  # the empty_mask and out_of_frame bits


def data_directory(out_dir: str | Path) -> Path:
    """Return the folder of this stage, ``<out>/data``."""
    return Path(out_dir) / DATA_DIRECTORY_NAME


def create_body_model(config: Mapping[str, Any]) -> BodyModel:
    """Build the body model that ``body.model`` names: the stand-in mannequin or SMPL-X.

    SMPL-X is read from the asset root, so a missing file raises MissingAssetError (exit code 3),
    and a missing ``smplx`` package raises ImportError. The stand-in needs neither.
    """
    name = config["body"]["model"]
    if name == "standin":
        return StandInBody()
    if name == "smplx":
        return SmplxBody(asset_root(config), n_betas=config["body"]["n_betas"])
    raise ConfigError(
        f"configuration key 'body.model' must be one of standin, smplx; got {name!r}", "body.model"
    )


def generate_dataset(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    resume: bool = False,
    time_budget: TimeBudget | None = None,
    run_record: RunRecord | None = None,
) -> GenerationResult:
    """Generate the bodies of a resolved configuration into ``<out_dir>/data``.

    ``resume=False`` generates every shard and replaces the files of an earlier run. With
    ``resume=True``, each done shard of the same configuration is kept (see the module docstring).
    ``time_budget`` stops the run between shards; the result then has ``completed=False``.
    ``run_record`` supplies the code version and the hardware class for summary.json and
    DONE.json. Without it, a record is made from the configuration and its device request.

    Raises ConfigError (exit code 2) for a configuration that the shard format cannot hold,
    MissingAssetError (exit code 3) for a missing licensed asset, and GenerationRefusedError (exit
    code 4) when a body cannot get an accepted pose or too few unflagged bodies remain.
    """
    digest = config_hash(config)  # validates the configuration first
    split = _check_configuration(config)
    record = _provenance(config, digest, run_record)
    data_dir = data_directory(out_dir)
    body = create_body_model(config)
    context = _Context(
        config=config,
        body=body,
        faces=body.faces,
        part_ids=body.part_ids,
        source=create_pose_source(config),
        pose_filter=PoseFilter.from_config(config, body),
        split=split,
        camera_matrix=intrinsics(config["camera"]["image_size"], config["camera"]["focal_px"]),
    )
    plans = _shard_plans(split.n_bodies, config["data"]["shard_size"])
    trusted = _prepare_data_directory(data_dir, digest, resume)
    _write_summary(data_dir, _build_summary([], config, record))
    logger.info(
        "generating %d bodies in %d shards of up to %d bodies (configuration %s)",
        split.n_bodies,
        len(plans),
        config["data"]["shard_size"],
        digest[:12],
    )

    stats: list[_ShardStats] = []
    generated = reused = 0
    out_of_time = False
    last_seconds = 0.0
    # The length of a shard is measured on the clock of the budget, so that the estimate of the
    # next shard and the budget itself agree on what a second is.
    clock = time_budget.elapsed if time_budget is not None else time.perf_counter
    for plan in plans:
        rows = _done_shard_rows(data_dir, plan) if trusted else None
        if rows is not None:
            stats.append(_shard_stats(plan.index, rows))
            reused += 1
            continue
        if out_of_time or (time_budget is not None and time_budget.should_stop(last_seconds)):
            out_of_time = True  # the shards after this one are still counted when they are done
            continue
        began = clock()
        rows = _generate_shard(context, plan, data_dir)
        last_seconds = clock() - began
        stats.append(_shard_stats(plan.index, rows))
        generated += 1
        logger.info(
            "generated shard %d of %d (%d bodies, %.1f s)",
            plan.index + 1,
            len(plans),
            len(rows),
            last_seconds,
        )

    completed = len(stats) == len(plans)
    if completed:
        _write_joined_manifest(data_dir, plans)
    summary = _build_summary(stats, config, record)
    _write_summary(data_dir, summary)
    result = GenerationResult(
        data_dir=data_dir,
        summary=summary,
        completed=completed,
        shards_total=len(plans),
        shards_generated=generated,
        shards_reused=reused,
    )
    _log_rejections(summary)
    if not completed:
        logger.warning(
            "the time budget ended the run after %d of %d shards; resume to finish",
            len(stats),
            len(plans),
        )
        return result
    _check_unflagged(summary)
    write_done_marker(
        data_dir,
        stage=STAGE_NAME,
        config_hash=record.config_hash,
        seed=record.seed,
        code_version=record.code_version,
        hardware_class=record.hardware_class,
        inputs={},
    )
    return result


def _check_configuration(config: Mapping[str, Any]) -> DataSplit:
    """Refuse a configuration that the shard format cannot hold, and return the data split.

    The manifest and the shards store ten shape coefficients and four cameras per body
    (data/manifest.py), so the configuration must ask for exactly those.
    """
    betas = config["body"]["n_betas"]
    if betas != BETA_COUNT:
        raise ConfigError(
            f"configuration key 'body.n_betas' must be {BETA_COUNT} for generation, because the "
            f"manifest and the shards store {BETA_COUNT} shape coefficients; got {betas}",
            "body.n_betas",
        )
    cameras = config["camera"]["n_cameras"]
    if cameras != CAMERA_COUNT:
        raise ConfigError(
            f"configuration key 'camera.n_cameras' must be {CAMERA_COUNT} for generation, because "
            f"the manifest and the shards store {CAMERA_COUNT} views per body; got {cameras}",
            "camera.n_cameras",
        )
    data = config["data"]
    try:
        return DataSplit(data["n_train"], data["n_cal"], data["n_test"])
    except ValueError as error:
        raise ConfigError(f"configuration key 'data.n_train': {error}", "data.n_train") from error


def _provenance(config: Mapping[str, Any], digest: str, run_record: RunRecord | None) -> RunRecord:
    """Return the run record whose code version and hardware class the stage files carry.

    A record that the caller passes must belong to this configuration. Without one, the record
    is made from the configuration and the device that its ``device`` key requests (FR-028).
    """
    if run_record is not None:
        if run_record.config_hash != digest:
            raise ValueError(
                "the run record belongs to another configuration: its config_hash is "
                f"{run_record.config_hash[:12]}, the configuration's is {digest[:12]}"
            )
        return run_record
    choice = select_device(config["device"])
    return start_run_record(
        config, hardware_class=choice.hardware_class, device_name=choice.device_name
    )


def _shard_plans(n_bodies: int, shard_size: int) -> list[_ShardPlan]:
    """Return the shards that hold ``n_bodies`` bodies, ``shard_size`` bodies to a shard."""
    return [
        _ShardPlan(index, first, min(first + shard_size, n_bodies))
        for index, first in enumerate(range(0, n_bodies, shard_size))
    ]


def _recorded_config_hash(data_dir: Path) -> str | None:
    """Return the configuration hash in summary.json, or None when there is none to read."""
    try:
        document = json.loads((data_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = document.get("config_hash") if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def _prepare_data_directory(data_dir: Path, digest: str, resume: bool) -> bool:
    """Clear what the stage rewrites, remove untrusted shards, and say whether shards are reusable.

    The done marker and manifest.csv go first, so that an interrupted run cannot look done.
    Shards are reusable only when ``resume`` is set and summary.json names this configuration.
    Otherwise the old shard files are removed here, before the new summary names this
    configuration: a later resume then never meets a shard of another configuration. Temporary
    files of an interrupted write are always removed.
    """
    shard_dir = data_dir / SHARD_DIRECTORY_NAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    clear_done_marker(data_dir)
    (data_dir / MANIFEST_NAME).unlink(missing_ok=True)
    trusted = resume and _recorded_config_hash(data_dir) == digest
    has_shards = False
    for folder in (data_dir, shard_dir):
        for path in list(folder.iterdir()):
            if _TEMPORARY_FILE.fullmatch(path.name):
                path.unlink()
            elif folder == shard_dir and _SHARD_FILE.fullmatch(path.name):
                has_shards = True
                if not trusted:
                    path.unlink()
    if has_shards and not trusted:
        reason = (
            "were not made under this configuration" if resume else "are replaced (no --resume)"
        )
        logger.warning("the shards in %s %s; every shard is generated again", shard_dir, reason)
    return trusted


def _done_shard_rows(data_dir: Path, plan: _ShardPlan) -> list[ManifestRow] | None:
    """Return the manifest rows of a done shard, or None when the shard has to be generated.

    A shard is done when its npz and its CSV exist, the npz opens and lists exactly the body ids
    of the plan, and the CSV holds the same body ids. Only the body ids are read from the npz, so
    the check stays cheap for a shard of masks.
    """
    archive = shard_path(data_dir, plan.index)
    table = archive.with_suffix(".csv")
    if not (archive.is_file() and table.is_file()):
        return None
    try:
        rows = read_manifest(table)
        stored_ids = _stored_body_ids(archive)
    except (OSError, EOFError, ValueError, KeyError, csv.Error, zipfile.BadZipFile):
        return None
    expected = list(range(plan.first, plan.stop))
    if stored_ids != expected or [row.body_id for row in rows] != expected:
        return None
    return rows


def _stored_body_ids(archive: Path) -> list[int]:
    """Return the body ids listed in a shard npz, reading only that one array.

    Raises OSError, EOFError, ValueError, KeyError, or zipfile.BadZipFile for a file that is not
    a readable npz archive with a ``body_id`` array. The file is opened here, so that it is closed
    on every path: numpy.load leaves a file that it opened itself open when the archive is damaged.
    """
    with archive.open("rb") as handle:
        loaded = np.load(handle, allow_pickle=False)
        if not isinstance(loaded, NpzFile):  # a plain .npy file under an .npz name
            raise ValueError(f"{archive} is not an npz archive")
        with loaded:
            return loaded["body_id"].tolist()


def _shard_stats(index: int, rows: Sequence[ManifestRow]) -> _ShardStats:
    """Count the flags, the rejections, and the unflagged bodies of one shard."""
    flag_counts = dict.fromkeys(FLAG_NAMES, 0)
    unflagged = dict.fromkeys(SPLITS, 0)
    flagged = 0
    for row in rows:
        if row.flags == 0:
            unflagged[row.split] += 1
            continue
        flagged += 1
        for name in FLAG_NAMES:
            if row.flags & FLAG_BITS[name]:
                flag_counts[name] += 1
    return _ShardStats(
        shard=index,
        n_bodies=len(rows),
        n_flagged=flagged,
        n_rejections=sum(row.pose_rejections for row in rows),
        flag_counts=flag_counts,
        unflagged=unflagged,
    )


def _build_summary(
    stats: Sequence[_ShardStats], config: Mapping[str, Any], record: RunRecord
) -> dict[str, Any]:
    """Return the fields of summary.json for the shards in ``stats`` (contracts/artifacts.md).

    ``n_flagged`` counts the bodies that carry each flag, so a body with two flags counts twice;
    ``per_shard.n_flagged`` counts bodies. ``n_draws`` is every pose draw, accepted or rejected
    (each body has one accepted draw), and ``rejection_rate`` is ``n_rejections / n_draws``
    (FR-001).
    """
    n_bodies = sum(shard.n_bodies for shard in stats)
    n_rejections = sum(shard.n_rejections for shard in stats)
    n_draws = n_bodies + n_rejections
    return {
        "n_bodies": n_bodies,
        "n_flagged": {name: sum(shard.flag_counts[name] for shard in stats) for name in FLAG_NAMES},
        "n_unflagged": {name: sum(shard.unflagged[name] for shard in stats) for name in SPLITS},
        "min_unflagged": config["data"]["min_unflagged"],
        "n_rejections": n_rejections,
        "n_draws": n_draws,
        "rejection_rate": n_rejections / n_draws if n_draws else 0.0,
        "per_shard": [
            {
                "shard": shard.shard,
                "n_bodies": shard.n_bodies,
                "n_flagged": shard.n_flagged,
                "n_rejections": shard.n_rejections,
            }
            for shard in sorted(stats, key=lambda shard: shard.shard)
        ],
        "config_hash": record.config_hash,
        "seed": record.seed,
        "code_version": record.code_version,
        "hardware_class": record.hardware_class,
    }


def _write_summary(data_dir: Path, summary: Mapping[str, Any]) -> None:
    """Write summary.json atomically."""
    atomic_write_text(
        data_dir / SUMMARY_NAME, json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def _log_rejections(summary: Mapping[str, Any]) -> None:
    """Report the pose rejection rate that FR-001 requires."""
    logger.info(
        "pose rejections: %d of %d draws were rejected (rate %.1f%%) for %d bodies",
        summary["n_rejections"],
        summary["n_draws"],
        100.0 * summary["rejection_rate"],
        summary["n_bodies"],
    )


def _check_unflagged(summary: Mapping[str, Any]) -> None:
    """Refuse when too few unflagged calibration or test bodies remain (SC-002).

    The message names both counts, whichever one is short, and the stage writes no DONE.json.
    """
    minimum = summary["min_unflagged"]
    n_cal = summary["n_unflagged"]["cal"]
    n_test = summary["n_unflagged"]["test"]
    if n_cal < minimum or n_test < minimum:
        flagged = ", ".join(f"{name} {count}" for name, count in summary["n_flagged"].items())
        raise GenerationRefusedError(
            f"only {n_cal} unflagged calibration bodies and {n_test} unflagged test bodies remain, "
            f"but data.min_unflagged requires at least {minimum} of each; no {DONE_MARKER_NAME} "
            f"is written. Bodies carrying each flag: {flagged}. Generate more bodies "
            "(data.n_cal, data.n_test) or change the settings that flag them"
        )


def _write_joined_manifest(data_dir: Path, plans: Sequence[_ShardPlan]) -> None:
    """Write manifest.csv: the header once, then the rows of every shard CSV in shard order."""
    header = render_manifest(()).encode("utf-8")
    with atomic_path(data_dir / MANIFEST_NAME) as temporary, temporary.open("wb") as joined:
        joined.write(header)
        for plan in plans:
            table = shard_path(data_dir, plan.index).with_suffix(".csv")
            with table.open("rb") as shard_rows:
                if shard_rows.readline() != header:
                    raise ValueError(f"{table} does not carry the manifest header")
                shutil.copyfileobj(shard_rows, joined)


def _generate_shard(context: _Context, plan: _ShardPlan, data_dir: Path) -> list[ManifestRow]:
    """Generate, measure, and write one shard; return its manifest rows.

    The npz is written first and the CSV second, so a shard counts as done only when both exist.
    """
    bodies = [
        _generate_body(context, plan.index, body_id) for body_id in range(plan.first, plan.stop)
    ]
    betas = np.stack([body.betas for body in bodies])
    measurements = _measure_canonical(context, betas)
    flags = np.array([body.render_flags for body in bodies], dtype=np.int64)
    flags[slice_nan_flags(measurements).numpy()] |= FLAG_BITS["slice_nan"]

    archive = write_shard(
        shard_path(data_dir, plan.index),
        body_id=np.arange(plan.first, plan.stop, dtype=np.int64),
        masks=np.stack([body.masks for body in bodies]),
        K=context.camera_matrix,
        R_true=np.stack([body.rig.R_true for body in bodies]),
        t_true=np.stack([body.rig.t_true for body in bodies]),
        noise_axis=np.stack([body.rig.noise_axis for body in bodies]),
        betas=betas,
        pose_root=np.stack([body.pose_root for body in bodies]),
        pose_body=np.stack([body.pose_body for body in bodies]),
        measurements=measurements,
        flags=flags,
    )
    pose_source = context.config["pose"]["source"]
    rows = [
        ManifestRow(
            body_id=body.body_id,
            shard=plan.index,
            split=context.split.split_of(body.body_id).value,
            pose_source=pose_source,
            pose_rejections=body.rejections,
            flags=int(flags[position]),
            betas=body.betas,
            pose_root=body.pose_root,
            pose_body=body.pose_body,
            azimuth_deg=body.rig.azimuth_deg,
            distance_m=body.rig.distance_m,
            height_m=body.rig.height_m,
            noise_axis=body.rig.noise_axis,
            measurements_cm=measurements[position],
        )
        for position, body in enumerate(bodies)
    ]
    write_manifest(archive.with_suffix(".csv"), rows)
    return rows


def _generate_body(context: _Context, shard: int, body_id: int) -> _GeneratedBody:
    """Draw the shape, the pose, and the rig of one body, and render its four silhouettes."""
    config = context.config
    seed = config["seed"]
    betas = _draw_betas(
        rng_for(seed, _STAGE_ID, shard, body_id, _BETAS_STREAM),
        config["body"]["n_betas"],
        config["body"]["beta_clip"],
    )
    pose_root, pose_body, rejections = _draw_pose(
        context, rng_for(seed, _STAGE_ID, shard, body_id, _POSE_STREAM), body_id
    )
    vertices = context.body.vertices(betas, pose_root, pose_body)
    vertices = vertices - np.array([0.0, vertices[:, 1].min(), 0.0])  # stand on the floor, y = 0
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    rig = sample_rig(
        rng_for(seed, _STAGE_ID, shard, body_id, _RIG_STREAM), config["camera"], center
    )
    masks, render_flags = _render_views(context, vertices, rig)
    return _GeneratedBody(
        body_id=body_id,
        betas=betas,
        pose_root=pose_root,
        pose_body=pose_body,
        rejections=rejections,
        rig=rig,
        masks=masks,
        render_flags=render_flags,
    )


def _draw_betas(rng: np.random.Generator, n_betas: int, clip: float) -> NDArray[np.float64]:
    """Draw standard normal shape coefficients, clipped to plus or minus ``clip``."""
    return np.clip(rng.standard_normal(n_betas), -clip, clip)


def _draw_pose(
    context: _Context, rng: np.random.Generator, body_id: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Draw poses until the filter accepts one; return the pose and the rejections before it.

    The rejections are limited by ``pose.max_rejections`` (contracts/config.md). One more raises
    GenerationRefusedError, so a source that the filter almost always rejects cannot stall a run.
    """
    limit = context.config["pose"]["max_rejections"]
    rejections = 0
    while True:
        pose_root, pose_body = context.source.draw(rng)
        pose = (np.asarray(pose_root, dtype=np.float64), np.asarray(pose_body, dtype=np.float64))
        if context.pose_filter.accept(pose).accepted:
            return pose[0], pose[1], rejections
        rejections += 1
        if rejections > limit:
            raise GenerationRefusedError(
                f"body {body_id}: the pose filter rejected draws until the rejection count "
                f"{rejections} passed pose.max_rejections ({limit}); raise pose.max_rejections or "
                "widen pose.limits_deg"
            )


def _render_views(
    context: _Context, vertices: NDArray[np.float64], rig: Rig
) -> tuple[NDArray[np.uint8], int]:
    """Render one silhouette per camera from the true placement, and collect the render flags."""
    side = context.config["camera"]["image_size"]
    masks = np.empty((CAMERA_COUNT, side, side), dtype=np.uint8)
    flags = 0
    for view in range(CAMERA_COUNT):
        silhouette = render_silhouette(
            vertices, context.faces, rig.K, rig.R_true[view], rig.t_true[view], side
        )
        masks[view] = silhouette.mask
        if not silhouette.area_fraction > 0.0:
            flags |= FLAG_BITS["empty_mask"]
        if silhouette.touches_border:
            flags |= FLAG_BITS["out_of_frame"]
    return masks, flags


def _measure_canonical(context: _Context, betas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return height, chest, waist, hip, and thigh in cm, shape (bodies, 5), for each row of betas.

    The measurements depend on the shape only (FR-004): each mesh is the canonical pose of its
    shape coefficients. They run on the CPU, as the module docstring explains.
    """
    step_cm = context.config["measure"]["step_cm"]
    chunks = []
    for first in range(0, len(betas), _MEASURE_CHUNK):
        vertices, joints = canonical_mesh(context.body, betas[first : first + _MEASURE_CHUNK])
        measured = measure_batch(
            vertices, context.faces, context.part_ids, joints, step_cm, device="cpu"
        )
        chunks.append(measured.numpy())
    return np.concatenate(chunks)
