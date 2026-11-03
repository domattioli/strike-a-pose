"""Probe round-trip plausibility of VAE samples (spec 016).

Encode an image with the VAE (μ), predict the SMPL pose and ring camera with the probe, re-render
the predicted mean-shape body with the repo's NumPy rasterizer and score the silhouette IoU against
the image. A sample is *plausible* when its IoU reaches τ, the 5th percentile of the same score on
real test images (spec 016 FR-005, FR-006).
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from strike_a_pose.data import DatasetError, PoseImageDataset
from strike_a_pose.datagen.assets import MissingAssetError, require_file, sha256_file
from strike_a_pose.datagen.cameras import Camera
from strike_a_pose.datagen.render import render
from strike_a_pose.datagen.smpl import SMPLModel
from strike_a_pose.models import ConvAutoencoder
from strike_a_pose.runlog import (
    checksums_sha256,
    environment_summary,
    git_info,
    make_run_dir,
    peak_rss_mb,
    round_sig,
    seed_everything,
    utc_now,
    write_json_exact,
    write_yaml,
)

from . import probe_config, sample_config
from .config import ConfigError
from .probe import (
    Standardizer,
    _encoders,
    _no_grad,
    fit_mlp,
    load_or_extract,
    score,
    target_values,
)
from .probe_config import ProbeConfig
from .report import grid_figure
from .roundtrip_config import RoundTripConfig
from .sample import _load, _val_images, classify, decode_batches
from .sample_config import SampleConfig

ROUND_TRIP_VERSION = 1
CONTROL_STREAM = 16
QUANTILES = (5, 25, 50, 75, 95)
HIST_BINS = 20
HIST_W = 480
HIST_HEADER = 20
HIST_ROW_H = 60
SET_ORDER = ("vae_prior", "ae_prior", "vae_recon_val", "real_test")
_NOT_RENDERABLE = ("camera_jitter", "body_shape")
REFIT_ANGLE_TOL = 0.05  # degrees (spec 016 FR-004)
REFIT_ACC_TOL = 0.005
_HIST_X0 = 110  # left edge of the bars (the set name sits left of it)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """|a ∧ b| / |a ∨ b| of two bool masks; 0.0 when the union is empty (spec 016 FR-005)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def calibrate_threshold(ious: np.ndarray, percentile: float) -> float:
    """τ = ``percentile`` of the real-image IoUs, ``method="lower"`` (an order statistic, so
    ``mean(ious >= τ) >= 1 - percentile / 100``; spec 016 FR-006)."""
    ious = np.asarray(ious, dtype=np.float64)
    if ious.size == 0:
        raise ValueError("cannot calibrate on an empty IoU array")
    return float(np.percentile(ious, percentile, method="lower"))


def camera_from_dict(d: dict) -> Camera:
    """A manifest ``cameras[k]`` entry as a ``Camera`` (spec 016 FR-005)."""
    return Camera(
        name=d["name"],
        K=np.asarray(d["K"], dtype=np.float64),
        R=np.asarray(d["R"], dtype=np.float64),
        t=np.asarray(d["t"], dtype=np.float64),
        width=int(d["width"]),
        height=int(d["height"]),
        yaw_deg=float(d["yaw_deg"]),
    )


def check_renderable(manifest: dict) -> None:
    """Refuse datasets whose renders need a per-image camera or body shape (spec 016 FR-002)."""
    cfg = manifest.get("config", {})
    for key in _NOT_RENDERABLE:
        if cfg.get(key) is not None:
            raise ConfigError(f"round trip needs mean-shape ring-camera data; dataset has {key}")


def rerender_masks(
    body: SMPLModel, poses: np.ndarray, cams: list[Camera], cam_idx: np.ndarray, num_betas: int
) -> np.ndarray:
    """Silhouettes [n, H, W] (bool) of ``poses[i]`` (zero betas) seen by ``cams[cam_idx[i]]``
    (spec 016 FR-005)."""
    n = len(poses)
    out = np.zeros((n, cams[0].height, cams[0].width), dtype=bool)
    betas = np.zeros(num_betas)
    for i in range(n):
        verts, _ = body.forward(poses[i], betas)
        out[i] = render(verts, body.faces, cams[int(cam_idx[i])])[1] > 0
    return out


def summarize(
    ious: np.ndarray,
    valid: np.ndarray,
    yaw: np.ndarray,
    empty: int,
    tau: float,
    n_classes: int,
    control: np.ndarray | None = None,
) -> dict:
    """The data-model "RT" record of one set (spec 016 FR-006, FR-007, FR-008)."""
    ious = np.asarray(ious, dtype=np.float64)
    n = int(ious.size)
    valid = np.asarray(valid, dtype=bool)
    rec: dict = {"n": n}
    if n == 0:
        rec |= {
            "iou_mean": None,
            "iou_quantiles": {f"p{q}": None for q in QUANTILES},
            "plausible_fraction": None,
            "plausible_and_valid_fraction": None,
        }
    else:
        plausible = ious >= tau
        rec |= {
            "iou_mean": float(ious.mean()),
            "iou_quantiles": {f"p{q}": float(np.percentile(ious, q)) for q in QUANTILES},
            "plausible_fraction": float(plausible.mean()),
            "plausible_and_valid_fraction": float((plausible & valid).mean()),
        }
    rec["empty_silhouette"] = int(empty)
    rec["yaw_counts"] = [int(c) for c in np.bincount(np.asarray(yaw, int), minlength=n_classes)]
    if control is None or n == 0:
        rec |= {"control_iou_median": None, "control_plausible_fraction": None}
    else:
        c = np.asarray(control, dtype=np.float64)
        rec |= {
            "control_iou_median": float(np.median(c)),
            "control_plausible_fraction": float((c >= tau).mean()),
        }
    return rec


def iou_histogram_figure(dists: dict[str, np.ndarray], tau: float) -> Image.Image:
    """Grey IoU histogram per set (one row each, dict order) with a black line at τ."""
    fig = Image.new("L", (HIST_W, HIST_HEADER + HIST_ROW_H * len(dists)), 255)
    draw = ImageDraw.Draw(fig)
    draw.text((4, 4), f"round-trip IoU, tau {tau:.4f}", fill=0)
    span = HIST_W - 4 - _HIST_X0
    bin_w = span // HIST_BINS
    for r, (name, x) in enumerate(dists.items()):
        y0 = HIST_HEADER + r * HIST_ROW_H
        draw.text((4, y0 + 2), name, fill=0)
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            continue
        counts, _ = np.histogram(x, bins=HIST_BINS, range=(0, 1))
        base, top = y0 + HIST_ROW_H - 4, HIST_ROW_H - 10
        for b, c in enumerate(counts):
            h = int(round(top * c / counts.max()))
            if h:
                xl = _HIST_X0 + b * bin_w
                draw.rectangle([xl, base - h, xl + bin_w - 2, base], fill=96)
        xt = _HIST_X0 + int(round(float(tau) * bin_w * HIST_BINS))
        draw.line([xt, y0 + 2, xt, base], fill=0)
    return fig


# --------------------------------------------------------------------------- probe refit


def refit_probe(
    pcfg: ProbeConfig, metrics: dict, data: dict
) -> tuple[Callable, Callable, Standardizer, dict]:
    """Re-fit the probe run's selected ``smpl_pose`` and ``yaw`` MLPs from its cache (spec 016
    FR-004; research R1): same calls as ``probe.fit_probes``, selected weight decay."""
    train = data["train"]
    zs = Standardizer.fit(train["latents"]["trained"])
    ztr = zs(train["latents"]["trained"])
    n_classes = len(metrics["classes"])
    sel = metrics["selection"]["mlp"]
    wd_pose, wd_yaw = (float(sel[t]["value"]) for t in ("smpl_pose", "yaw"))
    ys = Standardizer.fit(target_values(train, "smpl_pose"))
    m = fit_mlp(
        ztr, ys(target_values(train, "smpl_pose")), False, pcfg.mlp_hidden, pcfg.mlp_epochs,
        pcfg.mlp_batch_size, pcfg.mlp_lr, wd_pose, pcfg.seed,
    )  # fmt: skip
    m2 = fit_mlp(
        ztr, train["camera_index"], True, pcfg.mlp_hidden, pcfg.mlp_epochs,
        pcfg.mlp_batch_size, pcfg.mlp_lr, wd_yaw, pcfg.seed, n_classes,
    )  # fmt: skip

    def pose_fn(z: torch.Tensor) -> torch.Tensor:
        return ys.inverse(_no_grad(m, z))

    def yaw_fn(z: torch.Tensor) -> torch.Tensor:
        return _no_grad(m2, z).argmax(dim=1)

    return pose_fn, yaw_fn, zs, {"smpl_pose": wd_pose, "yaw": wd_yaw}


def refit_check(pose_fn: Callable, yaw_fn: Callable, zs: Standardizer, data: dict,
                metrics: dict, split: str) -> dict:  # fmt: skip
    """Score the refit MLPs on ``split`` against the probe run's recorded scores (spec 016
    FR-004). Null fields when the split is empty or was not scored by the probe."""
    out: dict = {"split": split, "angle_deg": None, "recorded_angle_deg": None,
                 "accuracy": None, "recorded_accuracy": None, "matches": None}  # fmt: skip
    a = data.get(split)
    rec = metrics["splits"].get(split, {}).get("targets")
    if a is None or a["n"] == 0 or not rec:
        return out
    z = zs(a["latents"]["trained"])
    n_classes = len(metrics["classes"])
    out["angle_deg"] = score("smpl_pose", pose_fn(z), a, n_classes)["angle_deg"]
    out["accuracy"] = score("yaw", yaw_fn(z), a, n_classes)["accuracy"]
    out["recorded_angle_deg"] = float(rec["smpl_pose"]["mlp"]["angle_deg"])
    out["recorded_accuracy"] = float(rec["yaw"]["mlp"]["accuracy"])
    out["matches"] = bool(
        abs(out["angle_deg"] - out["recorded_angle_deg"]) <= REFIT_ANGLE_TOL
        and abs(out["accuracy"] - out["recorded_accuracy"]) <= REFIT_ACC_TOL
    )
    return out


def predict(
    vae: ConvAutoencoder, images: torch.Tensor, image_size: tuple[int, int], batch_size: int,
    zs: Standardizer, pose_fn: Callable, yaw_fn: Callable,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Pose [n, 72] (float64) and yaw class [n] (int64) predicted from μ of ``images``
    [n, H*W] (spec 016 FR-005)."""
    h, w = image_size
    vae.eval()
    poses, yaws = [], []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            x = images[i : i + batch_size].reshape(-1, 1, h, w)
            z = zs(vae.encode(x.contiguous(memory_format=torch.channels_last)))
            poses.append(pose_fn(z).double().numpy())
            yaws.append(yaw_fn(z).numpy().astype(np.int64))
    if not poses:
        return np.zeros((0, 72)), np.zeros(0, dtype=np.int64)
    return np.concatenate(poses), np.concatenate(yaws)


# --------------------------------------------------------------------------- sets


def rebuild_sets(
    scfg: SampleConfig, vae: ConvAutoencoder, ref: ConvAutoencoder | None,
    val_ds: PoseImageDataset, test_ds: PoseImageDataset,
) -> dict[str, torch.Tensor]:  # fmt: skip
    """The sample run's sets as ``run_sample`` builds them, plus ``real_test`` (the first
    ``n_recon`` test images; operator ruling), each [n, H*W] (spec 016 FR-003)."""
    seed_everything(scfg.seed)
    z = torch.randn(
        scfg.n_samples, vae.latent_dim, generator=torch.Generator().manual_seed(scfg.seed)
    )
    images = {"vae_prior": decode_batches(vae, z, scfg.batch_size)}
    if ref is not None:
        images["ae_prior"] = decode_batches(ref, z, scfg.batch_size)
    xv = _val_images(val_ds, min(scfg.n_recon, len(val_ds)), scfg.batch_size)
    bs = scfg.batch_size
    with torch.no_grad():
        mu = [vae.encode(xv[i : i + bs]) for i in range(0, xv.shape[0], bs)]
    images["vae_recon_val"] = (
        decode_batches(vae, torch.cat(mu), bs)
        if mu
        else torch.empty(0, vae.image_size[0] * vae.image_size[1])
    )
    xt = _val_images(test_ds, min(scfg.n_recon, len(test_ds)), scfg.batch_size)
    images["real_test"] = xt.reshape(xt.shape[0], -1)
    return images


def _silhouettes(images: torch.Tensor, threshold: float, image_size: tuple[int, int]):
    n = images.shape[0]
    return (images > threshold).reshape(n, *image_size).numpy()


def check_reproduced(
    images: dict[str, torch.Tensor], source: dict, image_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Per-set ``valid`` flags; ``ConfigError`` unless every rebuilt 015 set reproduces the
    source ``foreground_px_mean`` and ``valid_fraction`` (spec 016 FR-003)."""
    crit = source["criteria"]
    lo, hi = crit["area_bounds_px"]
    thr = crit["iou_threshold"]
    sets = source["sets"]
    got = {k for k in images if k != "real_test"}
    if got != set(sets):
        raise ConfigError(
            f"sample run not reproduced: sets {sorted(got)} differ from recorded {sorted(sets)}"
        )
    valid = {}
    for name, x in images.items():
        fg = _silhouettes(x, thr, image_size)
        v = np.array([classify(m, lo, hi) == "valid" for m in fg], dtype=bool)
        valid[name] = v
        if name == "real_test":
            continue
        n = fg.shape[0]
        px = round_sig(float(fg.reshape(n, -1).sum(1).mean())) if n else None
        vf = round_sig(float(v.mean())) if n else 0.0
        rec = sets[name]
        if px != rec["foreground_px_mean"] or vf != rec["valid_fraction"]:
            raise ConfigError(
                f"sample run not reproduced: {name} foreground_px_mean {px} vs "
                f"{rec['foreground_px_mean']}, valid_fraction {vf} vs {rec['valid_fraction']}"
            )
    return valid


# --------------------------------------------------------------------------- run


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise ConfigError(f"{what}: no such file {path}")
    try:
        return json.loads(path.read_text())
    except ValueError as e:
        raise ConfigError(f"{what}: cannot parse {path}: {e}") from e


def _read_yaml(path: Path, what: str) -> dict:
    if not path.is_file():
        raise ConfigError(f"{what}: no such file {path}")
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{what}: cannot parse {path}: {e}") from e


def run_round_trip(
    cfg: RoundTripConfig, argv: list[str] | None = None, log: Callable[[str], None] | None = None
) -> Path:
    """Check inputs, rebuild and verify the sample sets, refit the probe, round-trip every set,
    write the run dir (spec 016 FR-002 – FR-008). Every refusal (``ConfigError`` /
    ``DatasetError``) comes before anything is created."""
    say = log or (lambda _msg: None)
    t0 = time.monotonic()
    started = utc_now()
    srun, prun = Path(cfg.sample_run), Path(cfg.probe_run)
    source = _read_json(srun / "samples.json", "sample_run samples.json")
    scfg = sample_config.config_from_dict(_read_yaml(srun / "config.yaml", "sample_run config"))
    pcfg = probe_config.config_from_dict(_read_yaml(prun / "config.yaml", "probe_run config"))
    metrics = _read_json(prun / "metrics.json", "probe_run metrics.json")
    missing = [t for t in ("smpl_pose", "yaw") if t not in pcfg.targets]
    if missing:
        raise ConfigError(f"probe run did not probe {missing}")

    train_ds = PoseImageDataset(scfg.dataset_dir, split="train", split_file=scfg.split_file)
    val_ds = PoseImageDataset(scfg.dataset_dir, split=scfg.recon_split, split_file=scfg.split_file)
    test_ds = PoseImageDataset(scfg.dataset_dir, split="test", split_file=scfg.split_file)
    if len(train_ds) == 0:
        raise DatasetError("train split has no samples")
    if len(test_ds) == 0:
        raise DatasetError("test split has no samples (the ceiling set real_test is empty)")
    manifest = train_ds.manifest
    check_renderable(manifest)
    if metrics["dataset"]["manifest_sha256"] != train_ds.manifest_sha256:
        raise ConfigError("probe run and sample run use different datasets (manifest hash)")
    if source["dataset"]["manifest_sha256"] != train_ds.manifest_sha256:
        raise ConfigError("sample run's samples.json does not match its dataset (manifest hash)")

    vae, vae_path = _load(scfg.checkpoint, "checkpoint")
    if not vae.variational:
        raise ConfigError("checkpoint: not a VAE (variational: false)")
    image_size = tuple(train_ds.image_size)
    if tuple(vae.image_size) != image_size:
        raise ConfigError(f"checkpoint image size {vae.image_size} != dataset {image_size}")
    ck_sha = sha256_file(vae_path)
    if metrics["encoder"]["checkpoint_sha256"] != ck_sha:
        raise ConfigError("probe run is of a different checkpoint than the sample run's VAE")
    if source["checkpoint"]["sha256"] != ck_sha:
        raise ConfigError("sample run's samples.json is of a different checkpoint")
    ref = ref_path = None
    if scfg.reference_checkpoint is not None:
        ref, ref_path = _load(scfg.reference_checkpoint, "reference_checkpoint")
    try:
        body = SMPLModel.load(require_file(Path(cfg.smpl_model_path), "SMPL model file"))
    except MissingAssetError as e:
        raise ConfigError(str(e)) from e
    smpl_sha = sha256_file(Path(cfg.smpl_model_path))
    manifest_smpl = manifest.get("inputs", {}).get("smpl_model", {}).get("sha256")
    smpl_matches = smpl_sha == manifest_smpl
    num_betas = int(manifest["config"]["num_betas"])
    cams = [camera_from_dict(c) for c in manifest["cameras"]]
    n_classes = len(metrics["classes"])
    if len(cams) != n_classes:
        raise ConfigError("probe classes differ from the dataset's cameras")

    images = rebuild_sets(scfg, vae, ref, val_ds, test_ds)
    valid = check_reproduced(images, source, image_size)
    say(f"sets rebuilt and reproduced: {', '.join(f'{k} {v.shape[0]}' for k, v in images.items())}")

    # probe refit (hashes as run_probe: the cache key of the probe run)
    hashes = {
        "checkpoint_sha256": ck_sha,
        "manifest_sha256": train_ds.manifest_sha256,
        "split_sha256": sha256_file(Path(pcfg.split_file)),
    }
    pdatasets = {
        s: PoseImageDataset(pcfg.dataset_dir, split=s, split_file=pcfg.split_file)
        for s in ("train", *pcfg.eval_splits)
    }
    data, cpath, hit = load_or_extract(pcfg, pdatasets, _encoders(vae, pcfg), hashes)
    say(f"probe latents {'from cache' if hit else 'encoded'} ({cpath})")
    pose_fn, yaw_fn, zs, wds = refit_probe(pcfg, metrics, data)
    check = refit_check(pose_fn, yaw_fn, zs, data, metrics, scfg.recon_split)
    say(f"refit check on {check['split']}: {check}")
    if check["matches"] is False:
        say("WARNING refit differs from the probe run's recorded scores")
    if not smpl_matches:
        say("WARNING SMPL file differs from the dataset manifest")

    # round trip per set
    thr = source["criteria"]["iou_threshold"]
    train = data["train"]
    pose_rows = np.nonzero((train["camera_index"] == 0).numpy())[0]
    train_poses = train["pose"].numpy()[pose_rows]
    ious, yaws, ctrl, rerender, empties = {}, {}, {}, {}, {}
    for name, x in images.items():
        sil = _silhouettes(x, thr, image_size)
        poses, yaw = predict(vae, x, image_size, scfg.batch_size, zs, pose_fn, yaw_fn)
        masks = rerender_masks(body, poses, cams, yaw, num_betas)
        ious[name] = np.array([silhouette_iou(s, m) for s, m in zip(sil, masks, strict=True)])
        yaws[name], rerender[name] = yaw, masks
        empties[name] = int((sil.reshape(len(sil), -1).sum(1) == 0).sum())
        if cfg.control:
            rng = np.random.default_rng([cfg.seed, CONTROL_STREAM, SET_ORDER.index(name)])
            pick = rng.integers(0, len(train_poses), size=len(sil))
            cm = rerender_masks(body, train_poses[pick], cams, yaw, num_betas)
            ctrl[name] = np.array([silhouette_iou(s, m) for s, m in zip(sil, cm, strict=True)])
        say(f"{name}: round-tripped {len(sil)}")
    tau = calibrate_threshold(ious["real_test"], cfg.calibration_percentile)
    say(f"tau {tau:.4f} (p{cfg.calibration_percentile:g} of real_test)")

    sets = {}
    for name in images:
        rt = summarize(
            ious[name], valid[name], yaws[name], empties[name], tau, n_classes, ctrl.get(name)
        )
        if name == "real_test":
            lo, hi = source["criteria"]["area_bounds_px"]
            fg = _silhouettes(images[name], thr, image_size)
            cls = [classify(m, lo, hi) for m in fg]
            n = len(cls)
            sets[name] = {
                "n": n,
                "valid_fraction": cls.count("valid") / n if n else 0.0,
                "fail_empty": cls.count("empty"),
                "fail_area": cls.count("area"),
                "fail_components": cls.count("components"),
                "nn_mse_mean": None,
                "nn_mse_median": None,
                "foreground_px_mean": float(fg.reshape(n, -1).sum(1).mean()) if n else None,
            }
        else:
            sets[name] = dict(source["sets"][name])
        sets[name]["round_trip"] = rt
        say(f"{name}: plausible {rt['plausible_fraction']} valid {sets[name]['valid_fraction']}")

    run_dir = make_run_dir(cfg.run_root, cfg.run_name, started)
    say(f"run dir {run_dir}")
    write_yaml(run_dir / "config.yaml", cfg.to_dict())
    samples_sha = sha256_file(srun / "samples.json")
    metrics_sha = sha256_file(prun / "metrics.json")
    report = {k: v for k, v in source.items() if k != "sets"} | {
        "round_trip_version": ROUND_TRIP_VERSION,
        "source_sample_run": {"path": str(srun), "samples_sha256": samples_sha},
        "round_trip_criteria": {
            "silhouette_threshold": thr,
            "calibration": {
                "set": "real_test",
                "split": "test",
                "percentile": cfg.calibration_percentile,
                "method": "lower",
                "n": int(ious["real_test"].size),
                "tau": tau,
            },  # fmt: skip
            "probe": {
                "run": str(prun),
                "metrics_sha256": metrics_sha,
                "checkpoint_sha256": ck_sha,
                "weight_decay": wds,
                "refit": check,
            },
            "smpl_model": {
                "path": cfg.smpl_model_path,
                "sha256": smpl_sha,
                "manifest_sha256": manifest_smpl,
                "matches_manifest": smpl_matches,
            },
            "render": "SMPLModel.forward(predicted 72-d pose, zero betas); "
            "ring camera of the predicted yaw class",
            "control": {
                "enabled": cfg.control,
                "stream": CONTROL_STREAM,
                "train_poses": int(len(train_poses)),
            },
        },
        "sets": sets,
    }
    write_json_exact(run_dir / "samples.json", report)
    per_image = {
        "sets": {
            name: {
                "iou": ious[name].tolist(),
                "yaw": yaws[name].tolist(),
                "valid": valid[name].tolist(),
                "control_iou": ctrl[name].tolist() if name in ctrl else None,
            }
            for name in images
        }
    }
    write_json_exact(run_dir / "round_trip_per_image.json", per_image)
    k = min(cfg.figure_samples, images["vae_prior"].shape[0], images["real_test"].shape[0])
    sel = {
        "s": images["vae_prior"][:k],
        "r": torch.from_numpy(rerender["vae_prior"][:k].reshape(k, -1).astype(np.float32)),
        "x": images["real_test"][:k],
        "rx": torch.from_numpy(rerender["real_test"][:k].reshape(k, -1).astype(np.float32)),
    }
    grid_figure(
        [("vae prior", sel["s"]), ("re-render", sel["r"]),
         ("real test", sel["x"]), ("re-render", sel["rx"])],
        image_size,
    ).save(run_dir / "round_trip.png", format="PNG")  # fmt: skip
    iou_histogram_figure(ious, tau).save(run_dir / "round_trip_iou.png", format="PNG")
    split_sha = sha256_file(Path(scfg.split_file))
    rck = None if ref_path is None else {"path": str(ref_path), "sha256": sha256_file(ref_path)}
    info = {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "seed": cfg.seed,
        "git": git_info(),
        "env": environment_summary(),
        "sample_run": {"path": str(srun), "samples_sha256": samples_sha},
        "probe_run": {"path": str(prun), "metrics_sha256": metrics_sha},
        "dataset": {
            "dir": scfg.dataset_dir,
            "manifest_sha256": train_ds.manifest_sha256,
            "split_file_sha256": split_sha,
            "checksums_sha256": checksums_sha256(scfg.dataset_dir),
        },
        "checkpoint": {"path": str(vae_path), "sha256": ck_sha},
        "reference_checkpoint": rck,
        "smpl_model": {"path": cfg.smpl_model_path, "sha256": smpl_sha},
        "cache": {"path": str(cpath) if cpath else None, "hit": hit},
        "started_utc": started.isoformat(),
        "finished_utc": utc_now().isoformat(),
        "wall_s": round(time.monotonic() - t0, 1),
        "peak_rss_mb": peak_rss_mb(),
    }
    write_json_exact(run_dir / "run_info.json", info)
    return run_dir


__all__ = [
    "calibrate_threshold",
    "camera_from_dict",
    "check_renderable",
    "check_reproduced",
    "iou_histogram_figure",
    "predict",
    "rebuild_sets",
    "refit_check",
    "refit_probe",
    "rerender_masks",
    "run_round_trip",
    "silhouette_iou",
    "summarize",
]
