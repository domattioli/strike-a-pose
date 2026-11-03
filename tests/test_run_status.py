"""Campaign status script (spec 012 FR-013): read-only status of a train run directory."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch
import yaml

from strike_a_pose.data import make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.train import trainer
from strike_a_pose.train.config import load_config
from strike_a_pose.train.guards import LOCK, RunAborted, boot_id
from strike_a_pose.train.trainer import run_training

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}
_spec = importlib.util.spec_from_file_location("run_status", ROOT / "scripts" / "run_status.py")
run_status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_status)


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def val_split(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("split") / "val_split.json"
    write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), p)
    return p


@pytest.fixture
def helper():
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "strike-a-pose-train"]
    )
    yield p
    p.kill()
    p.wait()


def _cfg(tmp_path: Path, root: str, split: Path | None = None, **kw):
    over = {"run_root": str(tmp_path / root), "epochs": 3, **SMALL, **kw}
    if split is not None:
        over |= {"split_file": str(split), "eval_splits": ["val", "test"], "figure_split": "val"}
    return load_config(CFG).with_overrides(**over)


def _status(d: Path, minutes_later: float = 0.0, *extra: str) -> dict:
    now = datetime.now(UTC) + timedelta(minutes=minutes_later)
    out = run_status.status(d, now)
    assert out == json.loads(json.dumps(out))  # JSON-ready
    return out


def _snapshot(d: Path) -> dict:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}


def test_finished_run(tmp_path, capsys):
    d = run_training(_cfg(tmp_path, "r"))
    before = _snapshot(d)
    assert run_status.main([str(d)]) == 0
    text = capsys.readouterr().out
    assert "finished" in text and "3 / 3" in text
    assert run_status.main([str(d), "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "finished" and (st["epochs_done"], st["epochs_planned"]) == (3, 3)
    assert st["selection"]["checkpoint"] == "checkpoint.pt" and st["best_val_epoch"] is None
    assert st["lock"] == {"pid": None, "alive": False} and st["val_image_mse"] is None
    assert st["early_stop"] is None and st["divergence"] == "off"
    assert _snapshot(d) == before  # read-only


def test_stopped_early_run_reports_the_scored_numbers(tmp_path, val_split, monkeypatch):
    vals = iter([0.5, 0.3, 0.4, 0.45])
    monkeypatch.setattr(trainer, "eval_loss", lambda m, x, b: next(vals) if x.shape[0] else None)
    d = run_training(_cfg(tmp_path, "r", val_split, epochs=4, early_stop={"patience": 2}))
    st = _status(d)
    assert st["state"] == "stopped early" and st["epochs_done"] == 4
    assert (st["best_val_epoch"], st["best_val_loss"], st["last_val_loss"]) == (2, 0.3, 0.45)
    assert st["val_vs_best"] == pytest.approx(0.5)
    assert st["early_stop"]["fired"]["epoch"] == 4
    m = json.loads((d / "metrics.json").read_text())
    ae = m["splits"]["val"]["image"]["ae"]
    assert st["val_image_mse"] == ae["image_mse"]
    regions = [ae["regions"][r]["image_mse"] for r in ("hands", "feet", "head")]
    assert st["val_region_mse_mean"] == pytest.approx(sum(regions) / 3)


def test_aborted_run(tmp_path, monkeypatch):
    real = trainer.train_epoch
    monkeypatch.setattr(
        trainer, "train_epoch", lambda *a, **k: real(*a, **k) | {"train_loss": float("nan")}
    )
    with pytest.raises(RunAborted):
        run_training(_cfg(tmp_path, "r", abort_on_divergence=True))
    (d,) = (tmp_path / "r").iterdir()
    st = _status(d)
    assert st["state"] == "aborted" and st["aborted"]["epoch"] == 1
    assert st["divergence"].startswith("would abort: train loss nan")


def test_killed_run_looks_dead_only_when_stale(tmp_path):
    d = run_training(_cfg(tmp_path, "r"), stop_after_epoch=1)  # the session ended: no lock
    st = _status(d)
    assert st["state"] == "not running" and st["epochs_done"] == 1 and st["stale_after_min"] == 10
    assert _status(d, 11)["state"] == "looks dead"
    later = datetime.now(UTC) + timedelta(seconds=2 * st["median_epoch_s"] + 1)  # no floor:
    assert run_status.status(d, later, min_stale_min=0.0)["state"] == "looks dead"  # 2 x median


def test_live_lock_owner_is_running_then_stalled(tmp_path, helper):
    d = run_training(_cfg(tmp_path, "r"), stop_after_epoch=1)
    lock = {"argv": ["strike-a-pose-train"], "boot_id": boot_id(), "pid": helper.pid,
            "started_utc": "2026-10-21T00:00:00+00:00"}  # fmt: skip
    (d / LOCK).write_text(json.dumps(lock))
    st = _status(d, 25)
    assert st["state"] == "running" and st["lock"] == {"pid": helper.pid, "alive": True}
    assert _status(d, 31)["state"] == "stalled"  # 3 x the 10-min threshold without activity
    helper.kill()
    helper.wait()
    assert _status(d, 31)["state"] == "looks dead"


def test_vae_reports_train_kl(tmp_path):
    d = run_training(_cfg(tmp_path, "r", epochs=10, variational=True, beta=0.1))
    st = _status(d)
    rows = torch.load(d / "checkpoint.pt")["history"]
    assert st["train_kl_epoch10"] == pytest.approx(rows[9]["train_kl"], rel=1e-7)
    assert st["train_kl_last"] == st["train_kl_epoch10"] > 0


def test_not_a_run_dir_exits_2(tmp_path, capsys):
    assert run_status.main([str(tmp_path)]) == 2
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "config.yaml").write_text(yaml.safe_dump({"checkpoint": "x.pt"}))
    (probe / "run_info.json").write_text("{}")
    assert run_status.main([str(probe)]) == 2
    assert "not a train run directory" in capsys.readouterr().err


def test_now_option_and_text_output(tmp_path, capsys):
    d = run_training(_cfg(tmp_path, "r"), stop_after_epoch=1)
    later = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    assert run_status.main([str(d), "--now", later]) == 0
    out = capsys.readouterr().out
    assert "looks dead" in out and "1 / 3" in out and os.fspath(d) in out
