"""Best-val weight selection (spec 008): rule, scored weights, records, resume, legacy.

Training is real (committed fixture, small model). The per-epoch val loss is scripted:
the fixture's real val curve has its minimum at the last epoch, so it cannot tell
best-val from final weights (research R4).
"""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from strike_a_pose.data import PoseImageDataset, make_split, tensor_to_image, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.eval.report import figure_indices, score_images, tile_origin
from strike_a_pose.models import ConvAutoencoder, load_checkpoint
from strike_a_pose.runlog import round_sig
from strike_a_pose.train import trainer
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import TrainConfig, load_config
from strike_a_pose.train.trainer import (
    best_val,
    load_selected_weights,
    load_split,
    predict_rows,
    resume_training,
    run_training,
    to_float,
    weight_selection,
)

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
NAN = float("nan")
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}  # the rule does not depend on size
SCRIPT = [0.5, 0.3, 0.4, 0.2, 0.25, 0.6]  # new minima at epochs 1, 2, 4: epoch 4 is selected


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.fixture(scope="module")
def val_split(tmp_path_factory) -> Path:
    """Fixture split with a val pose: train 1 / val 1 / test 0 poses (8 images each)."""
    p = tmp_path_factory.mktemp("split") / "val_split.json"
    write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), p)
    return p


class ScriptedVal:
    """Test double for ``trainer.eval_loss``: epoch ``e`` gets ``script[e - 1]``.

    It counts calls, so it continues across a ``stop_after_epoch`` session and a resume
    in the same process; an empty split still gets None.
    """

    def __init__(self, script: list[float]) -> None:
        self.script, self.calls = list(script), 0

    def __call__(self, model, images, batch_size):
        if images.shape[0] == 0:
            return None
        self.calls += 1
        return self.script[self.calls - 1]


@pytest.fixture
def scripted(monkeypatch):
    def install(script: list[float]) -> ScriptedVal:
        double = ScriptedVal(script)
        monkeypatch.setattr(trainer, "eval_loss", double)
        return double

    return install


def _cfg(tmp_path: Path, root: str, split: Path | None = None, **kw) -> TrainConfig:
    """Small fixture config; with ``split``, the split with a val pose (figure on val)."""
    over = {"run_root": str(tmp_path / root), "epochs": 6, **SMALL, **kw}
    if split is not None:
        over |= {"split_file": str(split), "eval_splits": ["val", "test"], "figure_split": "val"}
    return load_config(CFG).with_overrides(**over)


def _metrics(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


def _history(d: Path) -> list[dict]:
    with (d / "history.csv").open() as fh:
        return [
            {k: r[k] for k in ("epoch", "train_loss", "val_loss", "lr")} for r in csv.DictReader(fh)
        ]


def _rows(*vals) -> list[dict]:
    return [{"epoch": i + 1, "val_loss": v} for i, v in enumerate(vals)]


def _selection(rule: str, weights: str, by: str, epoch: int, val_loss, checkpoint: str) -> dict:
    return {
        "select_weights": rule,
        "weights": weights,
        "selected_by": by,
        "epoch": epoch,
        "val_loss": val_loss,
        "checkpoint": checkpoint,
    }


def _rescore(weights: Path, cfg: TrainConfig) -> tuple[dict, list[np.ndarray]]:
    """Val scores (as written to metrics.json, regions included: spec 009) and figure tiles
    of a file's weights."""
    model, _ = load_checkpoint(weights)
    model = model.to(memory_format=torch.channels_last)  # the trainer's layout
    ds = PoseImageDataset(cfg.dataset_dir, split="val", split_file=cfg.split_file)
    a = load_split(ds, cfg.batch_size, masks=True)
    x = to_float(a["image"]).reshape(a["n"], -1)
    scores = score_images(
        lambda b: predict_rows(model, b),
        x,
        a["mask"],
        cfg.batch_size,
        cfg.iou_threshold,
        labels=(a["kp2d"], a["visible"], a["image_size"]),
    )
    h, w = a["image_size"]
    shown = predict_rows(model, x[figure_indices(a["n"], cfg.figure_samples)])
    return round_sig(scores), [tensor_to_image(r.reshape(h, w)) for r in shown]


def _ae_tiles(png: Path, rows: int, image_size: tuple[int, int] = (256, 192)) -> list:
    """The ``ae`` column (column 1) of a recon figure."""
    fig = np.asarray(Image.open(png))
    h, w = image_size
    out = []
    for r in range(rows):
        x0, y0 = tile_origin(r, 1, image_size)
        out.append(fig[y0 : y0 + h, x0 : x0 + w])
    return out


def _same_weights(a: Path, b: Path) -> bool:
    sa, sb = torch.load(a)["model"], torch.load(b)["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sb)


# ------------------------------------------------------------------ rule (FR-002)


def test_best_val_is_the_earliest_finite_minimum():
    assert best_val(_rows(0.5, 0.2, 0.3, 0.2)) == {"epoch": 2, "val_loss": 0.2}  # tie: earliest
    assert best_val(_rows(0.5, 0.4, 0.3)) == {"epoch": 3, "val_loss": 0.3}  # minimum at the end
    assert best_val(_rows(NAN, 0.4, 0.5)) == {"epoch": 2, "val_loss": 0.4}  # NaN does not pin
    assert best_val(_rows(0.4, NAN, 0.5)) == {"epoch": 1, "val_loss": 0.4}
    assert best_val(_rows(None, None)) is None and best_val(_rows(NAN, NAN)) is None


def test_weight_selection_cases():
    sel = weight_selection(_rows(0.5, 0.2, 0.3), "best_val")
    assert sel == _selection("best_val", "best_val", "val loss", 2, 0.2, "best.pt")
    sel = weight_selection(_rows(0.5, 0.2, 0.3), "final")
    assert sel == _selection("final", "final", "config", 3, 0.3, "checkpoint.pt")
    sel = weight_selection(_rows(None, None), "best_val")
    assert sel == _selection("best_val", "final", "final (empty val)", 2, None, "checkpoint.pt")
    sel = weight_selection(_rows(NAN, NAN, NAN), "best_val")
    assert sel == _selection(
        "best_val", "final", "final (no finite val loss)", 3, None, "checkpoint.pt"
    )


# ------------------------------------------------------------------ scored weights (FR-004/005)


def test_best_val_weights_are_scored_and_drawn(tmp_path, val_split, scripted):
    """US1-1: minimum at epoch 2 of 4: best.pt (epoch 2) is scored and drawn, not the
    final weights."""
    scripted([0.5, 0.2, 0.3, 0.4])
    cfg = _cfg(tmp_path, "r", val_split, epochs=4)
    d = run_training(cfg)
    m = _metrics(d)
    assert m["train_version"] == 2
    assert m["selection"] == _selection("best_val", "best_val", "val loss", 2, 0.2, "best.pt")
    assert m["training"]["best_val_epoch"] == 2 and m["training"]["final_val_loss"] == 0.4
    assert torch.load(d / "best.pt")["epoch"] == 2 and torch.load(d / "checkpoint.pt")["epoch"] == 4
    best, tiles = _rescore(d / "best.pt", cfg)
    final, _ = _rescore(d / "checkpoint.pt", cfg)
    assert m["splits"]["val"]["image"]["ae"] == best != final
    assert m["splits"]["test"] == {"samples": 0, "poses": 0}
    shown = _ae_tiles(d / "recon_val.png", len(tiles))
    assert all(np.array_equal(f, t) for f, t in zip(shown, tiles, strict=True))


def test_nan_val_loss_never_pins_the_best(tmp_path, val_split, scripted):
    """A NaN val loss at epoch 1 must not become the best (`x < nan` is always false)."""
    scripted([NAN, 0.2, 0.3])
    d = run_training(_cfg(tmp_path, "r", val_split, epochs=3))
    assert torch.load(d / "best.pt")["epoch"] == 2
    m = _metrics(d)
    assert m["selection"]["epoch"] == 2 and m["training"]["first_val_loss"] is None


def test_minimum_at_last_epoch_equals_final_run(tmp_path, val_split, scripted):
    """US1-2/3: selection never changes training; a minimum at the last epoch scores the
    final weights; best.pt lost at the last epoch is rewritten from the checkpoint."""
    scripted([0.5, 0.4, 0.3, 0.2])
    a = run_training(_cfg(tmp_path, "a", val_split, epochs=4))
    scripted([0.5, 0.4, 0.3, 0.2])
    b = run_training(_cfg(tmp_path, "b", val_split, epochs=4, select_weights="final"))
    ma, mb = _metrics(a), _metrics(b)
    assert ma["selection"] == _selection("best_val", "best_val", "val loss", 4, 0.2, "best.pt")
    assert mb["selection"] == _selection("final", "final", "config", 4, 0.2, "checkpoint.pt")
    assert ma["splits"] == mb["splits"] and ma["training"] == mb["training"]
    assert (a / "recon_val.png").read_bytes() == (b / "recon_val.png").read_bytes()
    assert _history(a) == _history(b) and _same_weights(a / "checkpoint.pt", b / "checkpoint.pt")

    c = tmp_path / "c"  # killed after the last epoch, before its best.pt write
    shutil.copytree(a, c)
    for name in ("best.pt", "metrics.json", "recon_val.png"):
        (c / name).unlink()
    assert main(["--resume", str(c)]) == 0
    assert (c / "metrics.json").read_bytes() == (a / "metrics.json").read_bytes()
    assert torch.load(c / "best.pt")["epoch"] == 4 and _same_weights(c / "best.pt", a / "best.pt")


def test_empty_val_falls_back_to_final(tmp_path):
    """US2-1: the committed fixture has no val pose: final weights, recorded; no best.pt."""
    a = run_training(_cfg(tmp_path, "a", epochs=4))
    b = run_training(_cfg(tmp_path, "b", epochs=4, select_weights="final"))
    ma, mb = _metrics(a), _metrics(b)
    assert ma["selection"] == _selection(
        "best_val", "final", "final (empty val)", 4, None, "checkpoint.pt"
    )
    assert mb["selection"] == _selection("final", "final", "config", 4, None, "checkpoint.pt")
    assert ma["splits"] == mb["splits"] and not (a / "best.pt").exists()


def test_final_rule_scores_the_final_weights(tmp_path, val_split, scripted):
    """US2-2: `final` scores the last epoch; best.pt is still written (informational)."""
    scripted([0.5, 0.2, 0.3, 0.4])
    cfg = _cfg(tmp_path, "r", val_split, epochs=4, select_weights="final")
    d = run_training(cfg)
    m = _metrics(d)
    assert m["selection"] == _selection("final", "final", "config", 4, 0.4, "checkpoint.pt")
    assert torch.load(d / "best.pt")["epoch"] == 2
    assert m["splits"]["val"]["image"]["ae"] == _rescore(d / "checkpoint.pt", cfg)[0]


def test_no_finite_val_loss_falls_back_to_final(tmp_path, val_split, scripted):
    """US2-3."""
    scripted([NAN, NAN, NAN])
    d = run_training(_cfg(tmp_path, "r", val_split, epochs=3))
    m = _metrics(d)
    assert m["selection"] == _selection(
        "best_val", "final", "final (no finite val loss)", 3, None, "checkpoint.pt"
    )
    assert not (d / "best.pt").exists() and m["training"]["best_val_epoch"] is None


# ------------------------------------------------------------------ resume (FR-006)


def test_resume_at_every_kill_point_equals_uninterrupted(tmp_path, val_split, scripted):
    """US3-1: minima at epochs 1, 2, 4 of 6; three kill points resume to the same result."""
    scripted(SCRIPT)
    full = run_training(_cfg(tmp_path, "full", val_split))
    assert _metrics(full)["selection"]["epoch"] == 4

    scripted(SCRIPT)  # killed at an epoch boundary after the best epoch
    late = run_training(_cfg(tmp_path, "late", val_split), stop_after_epoch=5)
    assert main(["--resume", str(late)]) == 0

    scripted(SCRIPT)  # killed between the checkpoint.pt and best.pt writes of epoch 4
    gap = run_training(_cfg(tmp_path, "gap", val_split), stop_after_epoch=3)
    epoch2 = (gap / "best.pt").read_bytes()
    resume_training(gap, stop_after_epoch=4)
    assert torch.load(gap / "checkpoint.pt")["epoch"] == 4
    (gap / "best.pt").write_bytes(epoch2)  # the epoch-4 best.pt write never happened
    assert main(["--resume", str(gap)]) == 0

    tail = tmp_path / "tail"  # killed after the last epoch, during evaluation
    shutil.copytree(full, tail)
    (tail / "metrics.json").unlink()
    assert main(["--resume", str(tail)]) == 0

    for d in (late, gap, tail):
        assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes(), d
        assert torch.load(d / "best.pt")["epoch"] == 4 and _same_weights(
            d / "best.pt", full / "best.pt"
        ), d


def test_resume_refuses_when_the_best_weights_are_lost(tmp_path, val_split, scripted, capsys):
    """US3-2: best epoch 4, checkpoint at 5, best.pt deleted: exit 2 before training."""
    scripted(SCRIPT)
    d = run_training(_cfg(tmp_path, "r", val_split), stop_after_epoch=5)
    (d / "best.pt").unlink()
    sessions = json.loads((d / "run_info.json").read_text())["sessions"]
    assert main(["--resume", str(d)]) == 2
    err = capsys.readouterr().err
    assert "best.pt" in err and "epoch 4" in err
    assert not (d / "metrics.json").exists()
    assert json.loads((d / "run_info.json").read_text())["sessions"] == sessions


def test_scoring_refuses_a_best_file_of_another_epoch(tmp_path, val_split, scripted):
    """FR-004: the scored file must hold the selected epoch (only a hand edit breaks it)."""
    scripted([0.5, 0.2, 0.3])
    cfg = _cfg(tmp_path, "r", val_split, epochs=3)
    d = run_training(cfg)
    sel = weight_selection(torch.load(d / "checkpoint.pt")["history"], cfg.select_weights)
    model = ConvAutoencoder((256, 192), cfg.latent_dim, cfg.channels)
    load_selected_weights(model, d, sel, cfg)  # matches
    with pytest.raises(ConfigError, match="best.pt"):
        load_selected_weights(model, d, sel | {"epoch": 3}, cfg)


# ------------------------------------------------------------------ legacy run dirs (FR-007)


def test_run_dir_from_before_008_resumes_as_final(tmp_path, val_split, scripted):
    """US3-3: no select_weights in config.yaml, checkpoint.pt or best.pt: the run resumes
    (instead of "checkpoint does not match config.yaml") and scores its final weights."""
    scripted(SCRIPT)
    d = run_training(_cfg(tmp_path, "r", val_split), stop_after_epoch=3)
    c = yaml.safe_load((d / "config.yaml").read_text())
    del c["select_weights"]
    (d / "config.yaml").write_text(yaml.safe_dump(c))
    for name in ("checkpoint.pt", "best.pt"):
        obj = torch.load(d / name)
        del obj["config"]["select_weights"]
        torch.save(obj, d / name)
    assert main(["--resume", str(d)]) == 0
    assert _metrics(d)["selection"] == _selection(
        "final", "final", "config", 6, 0.6, "checkpoint.pt"
    )
