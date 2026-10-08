"""Predict medians and spreads of the five measurements for each split, noise level, view count.

The stage reads ``<out>/data`` and ``<out>/train`` and writes ``<out>/predict``: one file
``<split>_<cell>.npz`` per split (``cal``, ``test``) and cell (``contracts/artifacts.md``), and a
``DONE.json``. Calibration reads the ``cal`` files and evaluation the ``test`` files. Public
sources:

* Wu and Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning" (NeurIPS
  2018, https://arxiv.org/abs/1802.05335), for fusing the per-view experts of any subset of views
  by a product of experts (research R7). This module calls ``ShapeVAE.fuse``.
* Kingma and Welling, "Auto-Encoding Variational Bayes" (2013, https://arxiv.org/abs/1312.6114),
  for the reparameterized latent sample ``mean + standard deviation * noise`` (research R7).
* The measurement rules of research R6 through ``measure.measure_batch``, applied to canonical-pose
  meshes, exactly as for the ground truth of the data stage.
* Rousseeuw and Croux, "Alternatives to the Median Absolute Deviation" (Journal of the American
  Statistical Association 88, 1993), for the factor 1.4826 that makes the median absolute
  deviation a consistent estimate of a normal standard deviation (research R8).
* Taylor's first-order expansion, for the ``linearized`` measurement mode (research R13).

Reuse of the per-view posteriors. For each split and noise level, every body is encoded once on
all four cameras (``ShapeVAE.encode``), which makes 4 encodings per body and noise level instead of
7 (1 + 2 + 4). Each view-count cell then fuses the first k slots of the same four per-view experts
with ``ShapeVAE.fuse``. The fusion runs over the fixed four-slot layout in camera order 0 to 3,
with zero precision for an absent slot, so for the same experts a fewer-view fused precision never
exceeds a more-view one in floating point (``model/fusion.py``). ``latent_var_mean`` is the mean
over the latent dimensions of ``1 / fusion.fused_precision(experts.logvar_v, mask)``, which
therefore never increases with the view count, exactly (SC-004). The posterior log-variance of
``ShapeVAE.fuse`` is not used for it.

Predictive spread (orchestrator decision). Each of the K latent samples of a body is decoded to
the decoder MEAN of the shape coefficients, with no decoder noise, and that coefficient vector is
measured. The median and the scaled median absolute deviation of the K measurements therefore
follow the fused latent posterior, and so the number of views, and are not blurred by the
decoder's own noise head. The spread is floored at ``calibrate.spread_floor_cm`` (data-model.md,
MeasurementInterval).

Measurement modes (``predict.measure_mode``). ``exact`` measures the canonical-pose mesh of every
sample, in batches of bodies times K meshes. ``linearized`` measures the mesh of the per-coefficient
median of a body's K samples (the median sample) and the meshes of that sample shifted by one step
along each of the ten shape coefficients, which is 11 mesh evaluations per body and cell, and
predicts every sample by the first-order expansion
``m(c) + J (beta - c)``, where ``c`` is the median sample and column j of ``J`` is the forward
difference along coefficient j. The expansion equals the exact measurement at the median sample
(research R13).

Randomness. The latent noise of a batch comes from a generator seeded by the run seed, the split,
the noise level, the view count, and the first body of the batch, so a resumed run, and a run in
which other cells were already cached, draw the same samples.

Resume. A cell file is written atomically. With ``resume=True`` a cell file that holds the current
configuration hash and the finish time of the current training stage is reused, and only the other
cells are computed. A non-finite prediction raises ``PredictionRefusedError`` (exit code 4) that
names the body, the sample, the measurement, and the cell.
"""

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from strike_a_pose.body.base import BodyModel, canonical_mesh
from strike_a_pose.checkpoint import (
    StageStatus,
    TimeBudget,
    atomic_path,
    check_stage,
    clear_done_marker,
    input_hashes,
    read_done_marker,
    write_done_marker,
)
from strike_a_pose.config import config_hash
from strike_a_pose.data.dataset import ShapeDataset, collate_samples
from strike_a_pose.data.generate import create_body_model, data_directory
from strike_a_pose.data.splits import DataSplit
from strike_a_pose.device import select_device
from strike_a_pose.evaluate import cell_identifier
from strike_a_pose.measure import MEASUREMENT_NAMES, measure_batch
from strike_a_pose.model.fusion import fused_precision
from strike_a_pose.model.vae import ShapeVAE, ViewBatch
from strike_a_pose.runrecord import STAGES, RunRecord, current_code_version
from strike_a_pose.seeding import rng_for
from strike_a_pose.train import FINAL_MODEL_FILE_NAME, train_directory
from strike_a_pose.train import STAGE_NAME as TRAIN_STAGE

__all__ = [
    "MAD_SCALE",
    "PredictionRefusedError",
    "PredictResult",
    "STAGE_NAME",
    "STAGE_REFUSED_EXIT_CODE",
    "linearized_measurements",
    "predict_directory",
    "run_predict",
    "summarize_samples",
]

logger = logging.getLogger(__name__)

STAGE_NAME = "predict"
PREDICT_DIRECTORY_NAME = "predict"
STAGE_REFUSED_EXIT_CODE = 4

# Multiplies the median absolute deviation to estimate a normal standard deviation (research R8).
MAD_SCALE = 1.4826

# The cell splits, in the order they are written. The index is part of the random stream path.
_SPLITS = ("cal", "test")

# The number of camera slots of a rig, and the number of bodies encoded and measured together.
_SLOTS = 4
_BODY_BATCH = 16
# The number of meshes handed to the body model and the measurement code in one call.
_MESH_CHUNK = 64

# The forward-difference step of the linearized mode, in shape-coefficient units.
LINEARIZATION_STEP = 1.0

_STAGE_ID = STAGES.index(STAGE_NAME)
_COLUMNS = len(MEASUREMENT_NAMES)


class PredictionRefusedError(RuntimeError):
    """The stage refuses to go on, which the sap command reports with exit code 4.

    The cause is a prediction that is not finite, for example a degenerate slice of a decoded
    shape (research R6). The message names the body, the sample, the measurement, and the cell.
    """

    exit_code = STAGE_REFUSED_EXIT_CODE


@dataclass(frozen=True)
class PredictResult:
    """What a call of ``run_predict`` did.

    ``completed`` is False when the time budget stopped the stage between two groups of cells; the
    sap command then exits with code 7 and a later call with ``resume=True`` continues. ``skipped``
    is True when a matching DONE.json made the call do nothing. ``cells_written`` and
    ``cells_reused`` list the file names (such as ``test_v4_n0.npz``) computed and kept by this
    call.
    """

    predict_dir: Path
    completed: bool
    skipped: bool
    cells_written: tuple[str, ...]
    cells_reused: tuple[str, ...]


def predict_directory(out_dir: str | Path) -> Path:
    """Return the folder of this stage, ``<out>/predict``."""
    return Path(out_dir) / PREDICT_DIRECTORY_NAME


def summarize_samples(
    samples: NDArray[np.float64], spread_floor_cm: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the median and the floored scaled MAD over the sample axis of (bodies, K, 5) samples.

    The spread is 1.4826 times the median absolute deviation from the median, and at least
    ``spread_floor_cm`` (MeasurementInterval of data-model.md). Both results have shape (bodies, 5).
    """
    median = np.median(samples, axis=1)
    deviation = np.median(np.abs(samples - median[:, None, :]), axis=1)
    return median, np.maximum(MAD_SCALE * deviation, spread_floor_cm)


def linearized_measurements(
    measure: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    samples: NDArray[np.float64],
    step: float = LINEARIZATION_STEP,
) -> NDArray[np.float64]:
    """Return the first-order measurements of shape-coefficient samples, 11 evaluations per body.

    ``samples`` has shape (bodies, K, n_betas) and ``measure`` maps coefficients of shape
    (N, n_betas) to measurements of shape (N, 5) (it is called once with 1 + n_betas rows per body).
    For each body the expansion point ``c`` is the per-coefficient median of its K samples, and the
    result for sample ``beta`` is ``measure(c) + J (beta - c)``, where column j of ``J`` is
    ``(measure(c + step e_j) - measure(c)) / step``. The result has shape (bodies, K, 5) and equals
    ``measure(c)`` at ``beta = c``.
    """
    if not step > 0.0:
        raise ValueError(f"the linearization step must be positive, not {step!r}")
    bodies, _, n_betas = samples.shape
    center = np.median(samples, axis=1)
    points = np.repeat(center[:, None, :], n_betas + 1, axis=1)
    points[:, 1:, :] += step * np.eye(n_betas)
    measured = measure(points.reshape(bodies * (n_betas + 1), n_betas))
    measured = measured.reshape(bodies, n_betas + 1, _COLUMNS)
    base = measured[:, 0, :]
    jacobian = (measured[:, 1:, :] - base[:, None, :]) / step  # (bodies, n_betas, 5)
    offset = samples - center[:, None, :]
    return base[:, None, :] + np.einsum("bkj,bjm->bkm", offset, jacobian)


def run_predict(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    resume: bool = False,
    time_budget: TimeBudget | None = None,
    run_record: RunRecord | None = None,
) -> PredictResult:
    """Predict every cell of the calibration and test splits into ``<out_dir>/predict``.

    The cells are the combinations of ``evaluate.views`` and ``evaluate.noise_deg``.
    ``resume=False`` removes the cell files of an earlier run and computes all of them. With
    ``resume=True`` a matching DONE.json skips the stage, and a cell file of the current
    configuration and training run is kept. The stage needs the DONE.json of the generate and
    train stages. ``run_record`` supplies the code version and the hardware class for DONE.json.
    The time budget is tested before each group of cells that shares one encoding (one split
    and noise level).

    Raises FileNotFoundError when the data or train stage is not done, ConfigError (exit code 2)
    for a bad configuration, ValueError for a view count outside 1 to 4 or a model of another
    architecture, and PredictionRefusedError (exit code 4) for a split with no unflagged body or
    a non-finite prediction.
    """
    digest = config_hash(config)  # validates the configuration first
    views_list = [int(views) for views in config["evaluate"]["views"]]
    noise_list = [float(noise) for noise in config["evaluate"]["noise_deg"]]
    for views in views_list:
        if not 1 <= views <= _SLOTS:
            raise ValueError(f"evaluate.views must hold numbers from 1 to {_SLOTS}; got {views}")
    n_samples = int(config["predict"]["n_samples"])
    if n_samples < 1:
        raise ValueError(f"predict.n_samples must be at least 1; got {n_samples}")
    mode = config["predict"]["measure_mode"]
    if mode not in ("exact", "linearized"):
        raise ValueError(f"predict.measure_mode must be exact or linearized; got {mode!r}")

    predict_dir = predict_directory(out_dir)
    data_dir = data_directory(out_dir)
    train_dir = train_directory(out_dir)
    inputs = input_hashes({"generate": data_dir, TRAIN_STAGE: train_dir})
    for name, directory in (("generate", data_dir), (TRAIN_STAGE, train_dir)):
        if name not in inputs:
            raise FileNotFoundError(
                f"the {name} stage is not done: {directory / 'DONE.json'} is missing or unreadable"
            )
    train_marker = read_done_marker(train_dir)
    model_stamp = train_marker.finished_at if train_marker is not None else ""
    choice = select_device(config["device"])
    device = choice.device
    predict_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        check = check_stage(predict_dir, stage=STAGE_NAME, config_hash=digest, inputs=inputs)
        if check.status is StageStatus.DONE:
            logger.info("prediction is done and current; skipping (%s)", check.reason)
            return PredictResult(predict_dir, True, True, (), ())
    clear_done_marker(predict_dir)
    if not resume:
        for path in predict_dir.glob("*_v*_n*.npz"):
            path.unlink()

    model = _load_model(config, train_dir, device)
    body = create_body_model(config)
    measure = _MeshMeasurer(body, float(config["measure"]["step_cm"]))
    data_split = DataSplit(
        int(config["data"]["n_train"]), int(config["data"]["n_cal"]), int(config["data"]["n_test"])
    )
    context = _Context(
        config=config,
        digest=digest,
        model_stamp=model_stamp,
        model=model,
        device=device,
        measure=measure,
        mode=mode,
        n_samples=n_samples,
        spread_floor_cm=float(config["calibrate"]["spread_floor_cm"]),
        seed=int(config["seed"]),
    )

    written: list[str] = []
    reused: list[str] = []
    last_group_seconds = 0.0
    for split_index, split_name in enumerate(_SPLITS):
        for noise_index, noise_deg in enumerate(noise_list):
            names = {views: _file_name(split_name, views, noise_deg) for views in views_list}
            pending = []
            for views in views_list:
                path = predict_dir / names[views]
                if resume and _is_current(path, context, split_name, views, noise_deg):
                    reused.append(names[views])
                else:
                    pending.append(views)
            if not pending:
                continue
            if time_budget is not None and time_budget.should_stop(last_group_seconds):
                logger.warning(
                    "the time budget stopped prediction before %s noise %g; resume to finish",
                    split_name,
                    noise_deg,
                )
                return PredictResult(predict_dir, False, False, tuple(written), tuple(reused))
            began = time.monotonic()
            dataset = ShapeDataset.for_cell(
                config,
                data_dir,
                data_split.range_of(split_name),
                views=_SLOTS,
                noise_deg=noise_deg,
            )
            if len(dataset) == 0:
                raise PredictionRefusedError(
                    f"the {split_name} split holds no unflagged body, so nothing can be predicted"
                )
            cells = _predict_group(
                context, dataset, split_index, split_name, noise_index, noise_deg, pending
            )
            for views, arrays in cells.items():
                with atomic_path(predict_dir / names[views]) as temporary:
                    np.savez(temporary, **arrays)
                written.append(names[views])
            last_group_seconds = time.monotonic() - began

    record = run_record
    write_done_marker(
        predict_dir,
        stage=STAGE_NAME,
        config_hash=digest,
        seed=int(config["seed"]),
        code_version=record.code_version if record is not None else current_code_version(),
        hardware_class=record.hardware_class if record is not None else choice.hardware_class,
        inputs=inputs,
    )
    return PredictResult(predict_dir, True, False, tuple(written), tuple(reused))


@dataclass(frozen=True, eq=False)
class _Context:
    """The fixed inputs of one prediction run."""

    config: Mapping[str, Any]
    digest: str
    model_stamp: str
    model: ShapeVAE
    device: torch.device
    measure: "_MeshMeasurer"
    mode: str
    n_samples: int
    spread_floor_cm: float
    seed: int


class _MeshMeasurer:
    """Measure the canonical-pose meshes of shape coefficients, on the CPU in chunks.

    The ground truth of the data stage is measured the same way, so predictions and ground truth
    follow one rule (research R6). ``count`` is the number of meshes measured so far.
    """

    def __init__(self, body: BodyModel, step_cm: float) -> None:
        self._body = body
        self._step_cm = step_cm
        self._faces = np.asarray(body.faces)
        self._part_ids = np.asarray(body.part_ids)
        self.count = 0

    def __call__(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the five measurements in cm, shape (N, 5), of the N rows of betas."""
        betas = np.asarray(betas, dtype=np.float64)
        chunks = []
        for first in range(0, betas.shape[0], _MESH_CHUNK):
            vertices, joints = canonical_mesh(self._body, betas[first : first + _MESH_CHUNK])
            measured = measure_batch(
                vertices, self._faces, self._part_ids, joints, self._step_cm, device="cpu"
            )
            chunks.append(measured.numpy())
        self.count += betas.shape[0]
        return np.concatenate(chunks) if chunks else np.empty((0, _COLUMNS))


def _file_name(split_name: str, views: int, noise_deg: float) -> str:
    return f"{split_name}_{cell_identifier(views, noise_deg)}.npz"


def _load_model(config: Mapping[str, Any], train_dir: Path, device: torch.device) -> ShapeVAE:
    """Load ``train/model_final.pt`` into a model of this configuration, in evaluation mode.

    The model keeps working under a changed ``predict`` or ``evaluate`` key (a research R13
    mitigation changes only those), so its own configuration hash is not compared; a model of
    another architecture fails to load.
    """
    path = train_dir / FINAL_MODEL_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"the trained model {path} is missing; run sap train first")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ShapeVAE.from_config(config)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model


def _is_current(
    path: Path, context: _Context, split_name: str, views: int, noise_deg: float
) -> bool:
    """Return True when the cell file exists, is readable, and was made by this run's model."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["split"]) == split_name
                and int(archive["views"]) == views
                and float(archive["noise_deg"]) == noise_deg
                and str(archive["config_hash"]) == context.digest
                and str(archive["model_stamp"]) == context.model_stamp
            )
    except (OSError, ValueError, KeyError):
        return False


@torch.no_grad()
def _predict_group(
    context: _Context,
    dataset: ShapeDataset,
    split_index: int,
    split_name: str,
    noise_index: int,
    noise_deg: float,
    pending: list[int],
) -> dict[int, dict[str, Any]]:
    """Predict the pending view counts of one split and noise level; return the arrays per count.

    Each batch of bodies is encoded once on all four cameras, and every pending view count fuses
    the first k slots of those experts.
    """
    pieces: dict[int, list[dict[str, NDArray[Any]]]] = {views: [] for views in pending}
    for first in range(0, len(dataset), _BODY_BATCH):
        batch = collate_samples(
            [dataset[position] for position in range(first, min(first + _BODY_BATCH, len(dataset)))]
        )
        if not bool(batch.views.view_mask.all()):
            raise ValueError("a cell dataset must hold all four views of every body")
        views_on_device = ViewBatch(*(tensor.to(context.device) for tensor in batch.views))
        experts = context.model.encode(views_on_device)  # once per body and noise level
        for views in pending:
            mask = torch.zeros_like(views_on_device.view_mask)
            mask[:, :views] = True
            posterior = context.model.fuse(experts, mask)
            precision = fused_precision(experts.logvar_v, mask)
            latent_var_mean = (1.0 / precision.double()).mean(dim=-1)
            generator = torch.Generator(device=context.device)
            generator.manual_seed(
                int(
                    rng_for(
                        context.seed, _STAGE_ID, split_index, noise_index, views, first
                    ).integers(2**62)
                )
            )
            betas = _decode_samples(context, posterior, generator)
            samples = _measure_samples(context, betas)
            _require_finite(samples, batch.body_id.numpy(), split_name, views, noise_deg)
            median, spread = summarize_samples(samples, context.spread_floor_cm)
            pieces[views].append(
                {
                    "body_id": batch.body_id.numpy().astype(np.int64),
                    "m_true": batch.measurements.numpy().astype(np.float64),
                    "m_median": median,
                    "spread": spread,
                    "latent_var_mean": latent_var_mean.cpu().numpy(),
                    "samples": samples.astype(np.float32),
                }
            )
    result: dict[int, dict[str, Any]] = {}
    for views, parts in pieces.items():
        arrays: dict[str, Any] = {
            key: np.concatenate([part[key] for part in parts]) for key in parts[0]
        }
        arrays["views"] = np.int64(views)
        arrays["noise_deg"] = np.float64(noise_deg)
        arrays["split"] = np.str_(split_name)
        arrays["config_hash"] = np.str_(context.digest)
        arrays["model_stamp"] = np.str_(context.model_stamp)
        result[views] = arrays
    return result


def _decode_samples(
    context: _Context, posterior: Any, generator: torch.Generator
) -> NDArray[np.float64]:
    """Draw K latent samples per body and decode each to the decoder mean, shape (bodies, K, n)."""
    mu, logvar = posterior
    noise = torch.randn(
        (mu.shape[0], context.n_samples, mu.shape[-1]),
        generator=generator,
        device=mu.device,
        dtype=mu.dtype,
    )
    latent = mu.unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * noise
    return context.model.decoder(latent).mean.double().cpu().numpy()


def _measure_samples(context: _Context, betas: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the measurements of the decoded samples, shape (bodies, K, 5), by the measure mode."""
    bodies, count, n_betas = betas.shape
    if context.mode == "linearized":
        return linearized_measurements(context.measure, betas)
    measured = context.measure(betas.reshape(bodies * count, n_betas))
    return measured.reshape(bodies, count, _COLUMNS)


def _require_finite(
    samples: NDArray[np.float64],
    body_id: NDArray[np.int64],
    split_name: str,
    views: int,
    noise_deg: float,
) -> None:
    """Raise PredictionRefusedError naming the first sample whose measurements are not finite."""
    bad = np.argwhere(~np.isfinite(samples))
    if bad.size:
        row, sample, column = (int(value) for value in bad[0])
        raise PredictionRefusedError(
            f"non-finite prediction for body {int(body_id[row])}, sample {sample}, measurement "
            f"{MEASUREMENT_NAMES[column]}, in cell {cell_identifier(views, noise_deg)} of the "
            f"{split_name} split; the decoded shape has a degenerate slice (research R6)"
        )
