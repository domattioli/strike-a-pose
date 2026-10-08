"""Real-image evaluation: the trained model and a synthetic cell's quantiles on BodyM and SSP-3D.

This is the ``real-eval`` stage of specs/001-kill-test-mvp (FR-017 to FR-020; research R12). It
reads ``<out>/train/model_final.pt``, ``<out>/calibrate/quantiles.csv``, and
``<out>/run_record.json``, and writes ``<out>/real/<dataset>/results.csv`` and ``subjects.csv``
(contracts/artifacts.md). The results are reported only and never enter the kill verdict (FR-020).

Method. Each usable subject becomes the input of the model exactly as a synthetic body does: the
mask is padded, resized, and thresholded (``real/common.py``), the camera of each view is the
nominal camera of research R12 (BodyM: front at azimuth 0 and side at
``real.bodym.side_azimuth_deg``; SSP-3D: one view at azimuth 0), and the per-view experts are fused
by a product of experts (Wu and Goodman, NeurIPS 2018, https://arxiv.org/abs/1802.05335; research
R7). Latent samples ``mean + standard deviation * noise`` (Kingma and Welling, 2013,
https://arxiv.org/abs/1312.6114) are decoded to the decoder mean and measured on the canonical-pose
mesh with the rules of research R6, as in ``predict.py``. The median and the scaled median absolute
deviation of the samples (Rousseeuw and Croux, 1993; ``predict.summarize_samples``) give the
interval ``median -/+ q_hat * max(spread, floor)``, with the lower bound clipped at 0 cm. This is
split conformal prediction (research R8; Lei et al. 2018, https://arxiv.org/abs/1604.04173) with the
quantile of the matching synthetic cell: the 2-view cell without noise (``v2_n0``) for BodyM, and
the 1-view cell without noise (``v1_n0``) for SSP-3D. The real photographs are not exchangeable
with the synthetic calibration bodies, so the coverage is a measurement and not a guarantee.

Skipped subjects. A subject is skipped and counted, never dropped silently: BodyM subjects that the
loader cannot join (reasons of ``real/bodym.py``), subjects whose masks fail the status rules of
``real/common.py``, SSP-3D subjects whose ground truth is not finite, and subjects whose prediction
is not finite. Subjects with a mask status other than ``ok`` appear in ``subjects.csv`` with empty
measurement cells. ``n_skipped`` of ``results.csv`` is the total over all reasons for the split.
Subjects that the BodyM loader cannot join, subjects with a non-finite ground truth, and subjects
with a non-finite prediction are counted but have no row in ``subjects.csv``, because they have no
mask status.

A second run of the same dataset with another mask source replaces only the rows of that mask
source, and keeps the rows of the other source when they come from the same configuration.
"""

import csv
import io
import logging
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from strike_a_pose.assets import asset_root, require_asset
from strike_a_pose.body.base import BodyModel
from strike_a_pose.calibrate import CalibrationQuantile, read_quantiles
from strike_a_pose.camera import CAMERA_ENCODING_DIM, encode_camera
from strike_a_pose.checkpoint import atomic_write_text, input_hashes
from strike_a_pose.config import config_hash, resolve_config
from strike_a_pose.data.generate import create_body_model
from strike_a_pose.device import select_device
from strike_a_pose.evaluate import cell_identifier
from strike_a_pose.measure import MEASUREMENT_NAMES
from strike_a_pose.model.vae import ShapeVAE, ViewBatch
from strike_a_pose.predict import (
    _MeshMeasurer,
    linearized_measurements,
    summarize_samples,
)
from strike_a_pose.real.bodym import BODYM_PATH_KEY, BODYM_SPLITS, load_bodym
from strike_a_pose.real.common import mask_status, nominal_camera, prepare_mask
from strike_a_pose.real.masks import MaskBackend, ProvidedMasks, Sam2Masks
from strike_a_pose.real.ssp3d import (
    ground_truth_measurements,
    load_labels,
    load_silhouette,
    smpl_body_factory,
)
from strike_a_pose.report.tables import REAL_RESULT_COLUMNS
from strike_a_pose.runrecord import STAGES, RunRecord, RunRecordError, read_run_record
from strike_a_pose.seeding import rng_for
from strike_a_pose.train import FINAL_MODEL_FILE_NAME

__all__ = [
    "DATASETS",
    "MATCHING_CELLS",
    "REAL_DIRECTORY_NAME",
    "STAGE_REFUSED_EXIT_CODE",
    "SUBJECT_COLUMNS",
    "RealEvalError",
    "RealEvalResult",
    "RealResultRow",
    "real_directory",
    "run_real_eval",
]

logger = logging.getLogger(__name__)

REAL_DIRECTORY_NAME = "real"
STAGE_REFUSED_EXIT_CODE = 4
DATASETS = ("bodym", "ssp3d")
SSP3D_PATH_KEY = "real.ssp3d.path"
SAM2_CHECKPOINT_KEY = "real.sam2.checkpoint"
SSP3D_SPLIT = "all"
BODYM_MASK_SOURCE = "provided"
MASK_SOURCES = ("provided", "sam2")
# The synthetic cell whose quantiles each dataset uses (data-model.md, RealImageSubject), as
# (number of views, noise in degrees).
MATCHING_CELLS: Mapping[str, tuple[int, float]] = {"bodym": (2, 0.0), "ssp3d": (1, 0.0)}
# The folder of SSP-3D photographs, needed only for SAM 2 masks `[assumed; research R12]`.
SSP3D_PHOTO_FOLDER = "images"

# The columns of subjects.csv: identity, then one block per quantity over the five measurements.
SUBJECT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "split",
    "subject_id",
    "mask_source",
    "mask_status",
    *(
        f"{quantity}_{name}"
        for quantity in ("m_true", "m_median", "lower", "upper", "covered")
        for name in MEASUREMENT_NAMES
    ),
)

_STAGE_ID = STAGES.index("real_eval")
_SLOTS = 4
_BODY_BATCH = 16
_STATUS_ORDER = ("skipped_no_mask", "skipped_multi_person", "skipped_unusable", "ok")
_MASK_PIXEL_THRESHOLD = 127


class RealEvalError(ValueError):
    """The stage refuses to go on, which the sap command reports with exit code 4.

    The causes are a missing input stage, a quantile row that is absent, a configuration that
    cannot use the matching cell, and a dataset with no evaluable subject. The message names the
    cause and the file or key involved.
    """

    exit_code = STAGE_REFUSED_EXIT_CODE


@dataclass(frozen=True)
class RealResultRow:
    """One row of real/<dataset>/results.csv: a split and a measurement (contracts/artifacts.md)."""

    dataset: str
    split: str
    mask_source: str
    cell_id: str
    measurement: str
    nominal_level: float
    n_subjects: int
    n_skipped: int
    n_cal: int
    coverage: float
    median_width_cm: float
    mae_cm: float
    mean_signed_error_cm: float
    clipped_count: int
    q_hat: float


@dataclass(frozen=True)
class RealEvalResult:
    """What a call of ``run_real_eval`` wrote and counted.

    ``evaluated`` and ``skipped`` map each split to its subject counts. ``skip_reasons`` maps each
    split to the number of skipped subjects by reason: the loader reasons of ``real/bodym.py``, the
    mask statuses of ``real/common.py``, ``non_finite_truth``, and ``non_finite_prediction``.
    """

    dataset: str
    mask_source: str
    results_path: Path
    subjects_path: Path
    rows: tuple[RealResultRow, ...]
    evaluated: Mapping[str, int]
    skipped: Mapping[str, int]
    skip_reasons: Mapping[str, Mapping[str, int]]


@dataclass
class _Candidate:
    """One subject: its identity, mask status, prepared masks, and ground truth in centimeters."""

    split: str
    subject_id: str
    status: str
    masks: list[NDArray[np.uint8]] = field(default_factory=list)
    truth: NDArray[np.float64] | None = None


def real_directory(out_dir: str | os.PathLike[str]) -> Path:
    """Return the folder of this stage, ``<out>/real``."""
    return Path(out_dir) / REAL_DIRECTORY_NAME


def run_real_eval(
    config: Mapping[str, Any],
    out_dir: str | os.PathLike[str],
    dataset: str,
    *,
    mask_source: str | None = None,
    mask_backend: MaskBackend | None = None,
    body: BodyModel | None = None,
    body_factory: Callable[[str], BodyModel] | None = None,
) -> RealEvalResult:
    """Evaluate the trained model on one real dataset and write ``<out>/real/<dataset>/``.

    ``dataset`` is ``bodym`` (2 views, tape measurements as ground truth, split rows testA and
    testB) or ``ssp3d`` (1 view, ground truth from the SMPL shape in canonical pose, split ``all``).
    ``mask_source`` is ``provided`` or ``sam2``; it applies to SSP-3D only (BodyM masks are used as
    provided), and defaults to ``real.ssp3d.mask_source``. The inputs are ``train/model_final.pt``,
    ``calibrate/quantiles.csv`` with the DONE.json of both stages, and ``run_record.json``, whose
    configuration hash must match ``config``. ``mask_backend`` replaces the backend that
    ``mask_source`` names, ``body`` replaces the body model that measures the decoded shapes
    (default: the model of ``body.model``), and ``body_factory`` replaces the loader of the
    gendered SMPL bodies of the SSP-3D ground truth. They exist so that a test needs no licensed
    asset.

    Raises MissingAssetError (exit code 3) for an absent dataset, checkpoint, or model file, and
    RealEvalError (exit code 4) for a missing input stage, a run record of another configuration,
    a missing quantile row, a view or noise level of the matching cell that ``evaluate`` does not
    cover, or a dataset in which no subject can be evaluated.
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(DATASETS)}; got {dataset!r}")
    resolved = resolve_config(config)
    out = Path(out_dir)
    record = _read_record(out / "run_record.json", resolved)
    source = _resolve_mask_source(resolved, dataset, mask_source)
    inputs = input_hashes({"train": out / "train", "calibrate": out / "calibrate"})
    for stage in ("train", "calibrate"):
        if stage not in inputs:
            raise RealEvalError(
                f"the {stage} stage is not done: {out / stage / 'DONE.json'} is missing or "
                f"unreadable; run sap {stage} first"
            )

    measurements = list(resolved["evaluate"]["measurements"])
    if tuple(measurements) != MEASUREMENT_NAMES:
        raise RealEvalError(
            f"evaluate.measurements must be {list(MEASUREMENT_NAMES)} for the real-image "
            f"evaluation; got {measurements}"
        )
    views, noise_deg = MATCHING_CELLS[dataset]
    cell = cell_identifier(views, noise_deg)
    quantiles = _quantiles_of_cell(out / "calibrate" / "quantiles.csv", cell, measurements)

    device = select_device(resolved["device"]).device
    model = _load_model(resolved, out / "train" / FINAL_MODEL_FILE_NAME, device)
    body_model = body if body is not None else create_body_model(resolved)
    measurer = _MeshMeasurer(body_model, float(resolved["measure"]["step_cm"]))
    image_size = int(resolved["camera"]["image_size"])
    real = resolved["real"]
    azimuths = _view_azimuths(dataset, real)
    cameras = [
        torch.from_numpy(
            encode_camera(*nominal_camera(real, resolved["camera"], azimuth)[1:]).astype(np.float32)
        )
        for azimuth in azimuths
    ]
    root = asset_root(resolved)

    if dataset == "bodym":
        candidates, loader_skips = _bodym_candidates(resolved, root, image_size)
    else:
        candidates, loader_skips = _ssp3d_candidates(
            resolved, root, image_size, source, mask_backend, body_factory
        )

    reasons: dict[str, Counter[str]] = {
        split: Counter(skips) for split, skips in loader_skips.items()
    }
    evaluated_rows: dict[str, list[_Evaluated]] = {}
    subject_lines: list[_Evaluated] = []
    for split in loader_skips:
        subjects = [c for c in candidates if c.split == split]
        usable: list[_Candidate] = []
        for candidate in subjects:
            if candidate.status != "ok":
                reasons[split][candidate.status] += 1
                subject_lines.append(_Evaluated(candidate, None))
            elif candidate.truth is None or not np.all(np.isfinite(candidate.truth)):
                reasons[split]["non_finite_truth"] += 1
            else:
                usable.append(candidate)
        predictions = _predict(
            resolved,
            model,
            device,
            measurer,
            usable,
            cameras,
            image_size,
            quantiles,
            _seed_path(dataset, split),
        )
        kept: list[_Evaluated] = []
        for candidate, prediction in zip(usable, predictions, strict=True):
            if prediction is None:
                reasons[split]["non_finite_prediction"] += 1
                continue
            kept.append(_Evaluated(candidate, prediction))
        evaluated_rows[split] = kept
        subject_lines.extend(kept)

    rows = _result_rows(
        dataset,
        source,
        cell,
        quantiles,
        measurements,
        evaluated_rows,
        reasons,
    )
    if not rows:
        counts = {split: dict(counter) for split, counter in reasons.items()}
        raise RealEvalError(
            f"no {dataset} subject could be evaluated; skipped subjects by reason: {counts}"
        )
    results_text = _results_csv(rows, record)
    subjects_text = _subjects_csv(dataset, source, subject_lines)

    directory = real_directory(out) / dataset
    mask_label = source
    results_path = directory / "results.csv"
    subjects_path = directory / "subjects.csv"
    other_sources = _other_sources(results_path, mask_label, record.config_hash)
    atomic_write_text(
        results_path,
        _merged(results_path, REAL_RESULT_COLUMNS, results_text, mask_label, other_sources),
    )
    atomic_write_text(
        subjects_path,
        _merged(subjects_path, SUBJECT_COLUMNS, subjects_text, mask_label, other_sources),
    )
    evaluated = {split: len(kept) for split, kept in evaluated_rows.items()}
    skipped = {split: sum(counter.values()) for split, counter in reasons.items()}
    for split in evaluated:
        logger.info(
            "%s %s (%s masks): %d subjects evaluated, %d skipped %s",
            dataset,
            split,
            mask_label,
            evaluated[split],
            skipped[split],
            dict(reasons[split]),
        )
    return RealEvalResult(
        dataset=dataset,
        mask_source=mask_label,
        results_path=results_path,
        subjects_path=subjects_path,
        rows=tuple(rows),
        evaluated=evaluated,
        skipped=skipped,
        skip_reasons={split: dict(counter) for split, counter in reasons.items()},
    )


@dataclass
class _Prediction:
    """The median, spread, and interval of one subject, each of shape (5,)."""

    median: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    clipped: NDArray[np.bool_]


@dataclass
class _Evaluated:
    """A candidate with its prediction; the prediction is None for a mask-skipped subject."""

    candidate: _Candidate
    prediction: _Prediction | None


def _read_record(path: Path, config: Mapping[str, Any]) -> RunRecord:
    """Read run_record.json and check that it names the configuration given."""
    try:
        record = read_run_record(path)
    except RunRecordError as error:
        raise RealEvalError(str(error)) from error
    expected = config_hash(config)
    if record.config_hash != expected:
        raise RealEvalError(
            f"the run record '{path}' names configuration {record.config_hash[:12]}, but the "
            f"configuration given holds {expected[:12]}; real-eval reads the outputs of one run"
        )
    return record


def _resolve_mask_source(config: Mapping[str, Any], dataset: str, requested: str | None) -> str:
    """Return the mask source of the run: BodyM always uses its provided silhouettes."""
    if dataset == "bodym":
        if requested not in (None, BODYM_MASK_SOURCE):
            raise ValueError("BodyM masks are used as provided; --mask-source applies to ssp3d")
        return BODYM_MASK_SOURCE
    source = requested if requested is not None else config["real"]["ssp3d"]["mask_source"]
    if source not in MASK_SOURCES:
        raise ValueError(f"mask_source must be one of {', '.join(MASK_SOURCES)}; got {source!r}")
    return str(source)


def _quantiles_of_cell(
    path: Path, cell: str, measurements: Sequence[str]
) -> dict[str, CalibrationQuantile]:
    """Return the calibration row of each measurement in the cell, or refuse when one is absent."""
    rows = {row.measurement: row for row in read_quantiles(path) if row.cell_id == cell}
    missing = [name for name in measurements if name not in rows]
    if missing:
        raise RealEvalError(
            f"'{path}' has no row for cell {cell} and measurement(s) {', '.join(missing)}; the "
            "matching synthetic cell must be part of evaluate.views and evaluate.noise_deg"
        )
    return {name: rows[name] for name in measurements}


def _load_model(config: Mapping[str, Any], path: Path, device: torch.device) -> ShapeVAE:
    """Load ``train/model_final.pt`` into a model of this configuration, in evaluation mode."""
    if not path.is_file():
        raise FileNotFoundError(f"the trained model {path} is missing; run sap train first")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ShapeVAE.from_config(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def _view_azimuths(dataset: str, real: Mapping[str, Any]) -> list[float]:
    """Return the azimuths of the views of a dataset: BodyM front and side, SSP-3D one view."""
    if dataset == "bodym":
        return [0.0, float(real["bodym"]["side_azimuth_deg"])]
    return [0.0]


def _read_mask_file(path: Path) -> NDArray[np.uint8] | None:
    """Return a mask file as a 0 and 1 array, or None when it cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("mask %s cannot be decoded; treated as absent", path)
        return None
    return (image > _MASK_PIXEL_THRESHOLD).astype(np.uint8)


def _combined_status(statuses: Sequence[str]) -> str:
    """Return the first status of the fixed order that any view has (no mask, multi, unusable)."""
    for status in _STATUS_ORDER:
        if status in statuses:
            return status
    raise ValueError(f"unknown mask status in {list(statuses)}")


def _bodym_candidates(
    config: Mapping[str, Any], root: Path | None, image_size: int
) -> tuple[list[_Candidate], dict[str, Counter[str]]]:
    """Load the BodyM subjects of both splits, prepare their masks, and decide their statuses."""
    real = config["real"]
    folder = require_asset(str(real["bodym"]["path"] or ""), BODYM_PATH_KEY, root)
    loaded = load_bodym(folder, BODYM_SPLITS)
    candidates: list[_Candidate] = []
    skips: dict[str, Counter[str]] = {}
    for split, split_load in loaded.items():
        skips[split] = Counter({k: v for k, v in split_load.skipped.items() if v})
        for subject in split_load.subjects:
            prepared: list[NDArray[np.uint8] | None] = []
            for path in (subject.front_mask, subject.side_mask):
                raw = _read_mask_file(path)
                prepared.append(None if raw is None else prepare_mask(raw, image_size))
            status = _combined_status([mask_status(mask, real) for mask in prepared])
            candidates.append(
                _Candidate(
                    split=split,
                    subject_id=subject.subject_id,
                    status=status,
                    masks=[mask for mask in prepared if mask is not None],
                    truth=np.asarray(subject.measurements_cm, dtype=np.float64),
                )
            )
    return candidates, skips


def _ssp3d_candidates(
    config: Mapping[str, Any],
    root: Path | None,
    image_size: int,
    source: str,
    backend: MaskBackend | None,
    body_factory: Callable[[str], BodyModel] | None,
) -> tuple[list[_Candidate], dict[str, Counter[str]]]:
    """Load the SSP-3D subjects, segment them, decide statuses, and measure the ground truth."""
    real = config["real"]
    folder = require_asset(str(real["ssp3d"]["path"] or ""), SSP3D_PATH_KEY, root)
    labels = load_labels(folder)
    if backend is None:
        backend = _default_backend(config, root, folder, labels.filenames, source)
    candidates: list[_Candidate] = []
    for index, filename in enumerate(labels.filenames):
        subject_id = Path(filename).stem
        image = _photograph(folder, filename) if source == "sam2" else _NO_IMAGE
        box = None if labels.boxes is None else tuple(float(v) for v in labels.boxes[index][:4])
        mask = backend.segment(subject_id, image, box) if image is not None else None
        prepared = None if mask is None else prepare_mask(np.asarray(mask), image_size)
        status = mask_status(prepared, real)
        candidates.append(
            _Candidate(
                split=SSP3D_SPLIT,
                subject_id=subject_id,
                status=status,
                masks=[] if prepared is None else [prepared],
            )
        )
    wanted = [i for i, c in enumerate(candidates) if c.status == "ok"]
    factory = body_factory if body_factory is not None else smpl_body_factory(root)
    if wanted:
        truth = ground_truth_measurements(
            labels, factory, float(config["measure"]["step_cm"]), indices=wanted
        )
        for row, index in enumerate(wanted):
            candidates[index].truth = truth[row]
    return candidates, {SSP3D_SPLIT: Counter()}


# A stand-in photograph for backends that read the dataset silhouettes and ignore the image.
_NO_IMAGE: NDArray[np.uint8] = np.zeros((0, 0, 3), dtype=np.uint8)


def _default_backend(
    config: Mapping[str, Any],
    root: Path | None,
    folder: Path,
    filenames: Sequence[str],
    source: str,
) -> MaskBackend:
    """Return the mask backend that ``source`` names: the dataset silhouettes or SAM 2."""
    if source == "provided":
        masks = {}
        for filename in filenames:
            silhouette = load_silhouette(folder, filename)
            if silhouette is not None:
                masks[Path(filename).stem] = silhouette
        return ProvidedMasks(masks)
    checkpoint = str(config["real"]["sam2"]["checkpoint"] or "")
    device = select_device(config["device"]).device.type
    return Sam2Masks(checkpoint, root, device=device)


def _photograph(folder: Path, filename: str) -> NDArray[np.uint8] | None:
    """Return the RGB photograph of a subject for SAM 2, or None when it cannot be read."""
    path = folder / SSP3D_PHOTO_FOLDER / filename
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("photograph %s cannot be read; the subject has no mask", path)
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _seed_path(dataset: str, split: str) -> tuple[int, int]:
    """Return the integers that name the random stream of a dataset and split (research R9)."""
    splits = BODYM_SPLITS if dataset == "bodym" else (SSP3D_SPLIT,)
    return DATASETS.index(dataset), splits.index(split)


@torch.no_grad()
def _predict(
    config: Mapping[str, Any],
    model: ShapeVAE,
    device: torch.device,
    measurer: _MeshMeasurer,
    subjects: Sequence[_Candidate],
    cameras: Sequence[torch.Tensor],
    image_size: int,
    quantiles: Mapping[str, CalibrationQuantile],
    seed_path: tuple[int, int],
) -> list[_Prediction | None]:
    """Return one prediction per subject (None when not finite), in the order of ``subjects``.

    The views of a subject fill the first slots of the four-slot layout of the model, as in the
    synthetic cell of the same view count. The interval follows ``evaluate.py``: the spread is
    floored at the calibration floor, and the lower bound is clipped at 0 cm.
    """
    if not subjects:
        return []
    views = len(cameras)
    n_samples = int(config["predict"]["n_samples"])
    mode = config["predict"]["measure_mode"]
    seed = int(config["seed"])
    q_hats = np.array([quantiles[name].q_hat for name in MEASUREMENT_NAMES])
    floors = np.array([quantiles[name].spread_floor_cm for name in MEASUREMENT_NAMES])
    results: list[_Prediction | None] = []
    for first in range(0, len(subjects), _BODY_BATCH):
        group = subjects[first : first + _BODY_BATCH]
        silhouettes = torch.zeros(len(group), _SLOTS, image_size, image_size)
        encodings = torch.zeros(len(group), _SLOTS, CAMERA_ENCODING_DIM)
        view_mask = torch.zeros(len(group), _SLOTS, dtype=torch.bool)
        for row, subject in enumerate(group):
            for slot in range(views):
                silhouettes[row, slot] = torch.from_numpy(subject.masks[slot].astype(np.float32))
                encodings[row, slot] = cameras[slot]
                view_mask[row, slot] = True
        batch = ViewBatch(silhouettes.to(device), encodings.to(device), view_mask.to(device))
        mu, logvar = model.fuse(model.encode(batch), batch.view_mask)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(rng_for(seed, _STAGE_ID, *seed_path, first).integers(2**62)))
        noise = torch.randn(
            (mu.shape[0], n_samples, mu.shape[-1]),
            generator=generator,
            device=mu.device,
            dtype=mu.dtype,
        )
        latent = mu.unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * noise
        betas = model.decoder(latent).mean.double().cpu().numpy()
        if mode == "linearized":
            samples = linearized_measurements(measurer, betas)
        else:
            bodies, count, n_betas = betas.shape
            samples = measurer(betas.reshape(bodies * count, n_betas)).reshape(
                bodies, count, len(MEASUREMENT_NAMES)
            )
        median, spread = summarize_samples(samples, float(config["calibrate"]["spread_floor_cm"]))
        spread = np.maximum(spread, floors[np.newaxis, :])
        half_width = q_hats[np.newaxis, :] * spread
        lower_before_clip = median - half_width
        clipped = lower_before_clip < 0.0
        lower = np.where(clipped, 0.0, lower_before_clip)
        upper = median + half_width
        for row in range(len(group)):
            finite = all(np.all(np.isfinite(a[row])) for a in (median, spread, lower, upper))
            results.append(
                _Prediction(median[row], lower[row], upper[row], clipped[row]) if finite else None
            )
    return results


def _result_rows(
    dataset: str,
    mask_source: str,
    cell: str,
    quantiles: Mapping[str, CalibrationQuantile],
    measurements: Sequence[str],
    evaluated: Mapping[str, Sequence[_Evaluated]],
    reasons: Mapping[str, Counter[str]],
) -> list[RealResultRow]:
    """Return the aggregate rows: one per split with evaluated subjects and per measurement."""
    rows: list[RealResultRow] = []
    for split, kept in evaluated.items():
        if not kept:
            logger.warning("%s %s: no subject could be evaluated; no result rows", dataset, split)
            continue
        n_skipped = sum(reasons[split].values())
        predictions = [item.prediction for item in kept if item.prediction is not None]
        median = np.stack([prediction.median for prediction in predictions])
        lower = np.stack([prediction.lower for prediction in predictions])
        upper = np.stack([prediction.upper for prediction in predictions])
        clipped = np.stack([prediction.clipped for prediction in predictions])
        truth = np.stack(
            [item.candidate.truth for item in kept if item.candidate.truth is not None]
        )
        covered = (lower <= truth) & (truth <= upper)
        errors = median - truth
        for column, name in enumerate(measurements):
            quantile = quantiles[name]
            rows.append(
                RealResultRow(
                    dataset=dataset,
                    split=split,
                    mask_source=mask_source,
                    cell_id=cell,
                    measurement=name,
                    nominal_level=1.0 - quantile.alpha,
                    n_subjects=len(kept),
                    n_skipped=n_skipped,
                    n_cal=quantile.n_cal,
                    coverage=float(np.mean(covered[:, column])),
                    median_width_cm=float(np.median(upper[:, column] - lower[:, column])),
                    mae_cm=float(np.mean(np.abs(errors[:, column]))),
                    mean_signed_error_cm=float(np.mean(errors[:, column])),
                    clipped_count=int(np.count_nonzero(clipped[:, column])),
                    q_hat=quantile.q_hat,
                )
            )
    return rows


def _number(value: float) -> str:
    """Return a number as the shortest text that reads back to the same value."""
    return repr(float(value))


def _results_csv(rows: Sequence[RealResultRow], record: RunRecord) -> str:
    """Return results.csv with the columns of REAL_RESULT_COLUMNS and the run record values."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(REAL_RESULT_COLUMNS)
    for row in rows:
        writer.writerow(
            [
                row.dataset,
                row.split,
                row.mask_source,
                row.cell_id,
                row.measurement,
                _number(row.nominal_level),
                row.n_subjects,
                row.n_skipped,
                row.n_cal,
                _number(row.coverage),
                _number(row.median_width_cm),
                _number(row.mae_cm),
                _number(row.mean_signed_error_cm),
                row.clipped_count,
                _number(row.q_hat),
                record.seed,
                record.config_hash,
                record.code_version,
                record.hardware_class,
            ]
        )
    return buffer.getvalue()


def _subjects_csv(dataset: str, mask_source: str, items: Sequence[_Evaluated]) -> str:
    """Return subjects.csv: one row per subject with a mask status, empty values when skipped."""
    count = len(MEASUREMENT_NAMES)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SUBJECT_COLUMNS)
    for item in items:
        candidate, prediction = item.candidate, item.prediction
        head = [dataset, candidate.split, candidate.subject_id, mask_source, candidate.status]
        if prediction is None or candidate.truth is None:
            writer.writerow(head + [""] * (5 * count))
            continue
        covered = (prediction.lower <= candidate.truth) & (candidate.truth <= prediction.upper)
        writer.writerow(
            head
            + [_number(v) for v in candidate.truth]
            + [_number(v) for v in prediction.median]
            + [_number(v) for v in prediction.lower]
            + [_number(v) for v in prediction.upper]
            + ["true" if flag else "false" for flag in covered]
        )
    return buffer.getvalue()


def _read_table(path: Path, columns: Sequence[str]) -> list[list[str]] | None:
    """Return the data rows of a CSV file with exactly these columns, or None when unusable."""
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        logger.warning("%s cannot be read; it is replaced", path)
        return None
    if tuple(header) != tuple(columns):
        logger.warning("%s has another header; it is replaced", path)
        return None
    return [row for row in rows if len(row) == len(columns)]


def _other_sources(results_path: Path, mask_source: str, digest: str) -> set[str]:
    """Return the other mask sources that results.csv holds rows of from the same configuration."""
    rows = _read_table(results_path, REAL_RESULT_COLUMNS)
    if rows is None:
        return set()
    source_at = REAL_RESULT_COLUMNS.index("mask_source")
    hash_at = REAL_RESULT_COLUMNS.index("config_hash")
    return {row[source_at] for row in rows if row[hash_at] == digest} - {mask_source}


def _merged(
    path: Path,
    columns: Sequence[str],
    new_text: str,
    mask_source: str,
    other_sources: set[str],
) -> str:
    """Return the new table followed by the old rows of the mask sources in ``other_sources``.

    The rows of the mask source of this run are replaced, and the rows of every other source are
    dropped unless ``other_sources`` names it (the caller names the sources whose results come from
    the same configuration, so a table never mixes two runs).
    """
    old_rows = _read_table(path, columns) if other_sources else None
    if not old_rows:
        return new_text
    source_at = columns.index("mask_source")
    kept = [row for row in old_rows if row[source_at] in other_sources]
    if not kept:
        return new_text
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(kept)
    return new_text + buffer.getvalue()
