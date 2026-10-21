"""Run guards (spec 012): early stop, divergence abort, run-dir lock, legacy resume, defaults.

Training is real (committed fixture, small model); per-epoch val and train losses are scripted
where a rule needs a particular curve (the 008 pattern, tests/test_train_selection.py).
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from strike_a_pose.data import make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.config import ConfigError
from strike_a_pose.train import trainer
from strike_a_pose.train.cli import main
from strike_a_pose.train.config import (
    LEGACY_VALUES,
    EarlyStop,
    TrainConfig,
    config_from_dict,
    load_config,
)
from strike_a_pose.train.guards import (
    ABORTED,
    LOCK,
    RunAborted,
    abort_record,
    boot_id,
    divergence_reason,
    early_stop_state,
    lock_alive,
    read_lock,
)
from strike_a_pose.train.trainer import resume_training, run_training

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
NAN, INF = float("nan"), float("inf")
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}  # the rules do not depend on size
GUARD_KEYS = {"early_stop": None, "abort_on_divergence": False, "divergence_ratio": 10.0}
BASE = {"dataset_dir": "d", "split_file": "s.json"}
SCRIPT = [0.5, 0.3, 0.4, 0.35, 0.45, 0.6]  # best epoch 2; patience 2 stops after epoch 4
REASON = "no val loss below 0.3 (epoch 2) by more than 0 for 2 epochs (patience 2, min_epochs 0)"


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # config paths are repo-relative


@pytest.fixture(scope="module")
def val_split(tmp_path_factory) -> Path:
    """Fixture split with a val pose: train 1 / val 1 / test 0 poses (8 images each)."""
    p = tmp_path_factory.mktemp("split") / "val_split.json"
    write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), p)
    return p


class Scripted:
    """Counts calls across sessions in one process; call i gets ``script[i - 1]``."""

    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), 0

    def next(self):
        self.calls += 1
        return self.script[self.calls - 1] if self.calls <= len(self.script) else None


@pytest.fixture
def scripted(monkeypatch):
    """Replace ``trainer.eval_loss`` by a scripted per-epoch val loss (None for empty val)."""

    def install(script: list) -> Scripted:
        s = Scripted(script)
        monkeypatch.setattr(
            trainer, "eval_loss", lambda m, images, b: s.next() if images.shape[0] else None
        )
        return s

    return install


@pytest.fixture
def scripted_train(monkeypatch):
    """Override the epoch train loss (the model still trains); None keeps the real value."""
    real = trainer.train_epoch

    def install(script: list) -> Scripted:
        s = Scripted(script)

        def epoch(*a, **kw):
            st = real(*a, **kw)
            v = s.next()
            return st if v is None else st | {"train_loss": v}

        monkeypatch.setattr(trainer, "train_epoch", epoch)
        return s

    return install


def _wait_for_exec(proc, needle: bytes, timeout=10.0):
    """Block until /proc/<pid>/cmdline contains needle (the child has exec'd)."""
    path = Path(f"/proc/{proc.pid}/cmdline")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if needle in path.read_bytes():
                return
        except OSError:
            pass
        time.sleep(0.01)
    raise AssertionError(f"child {proc.pid} never showed {needle!r} in its cmdline")


@pytest.fixture
def helper():
    """A live process whose command line names strike-a-pose-train (a stand-in lock owner)."""
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "strike-a-pose-train"]
    )
    try:
        _wait_for_exec(p, b"strike-a-pose-train")
        yield p
    finally:
        p.kill()
        p.wait()


def _cfg(tmp_path: Path, root: str, split: Path | None = None, **kw) -> TrainConfig:
    over = {"run_root": str(tmp_path / root), "epochs": 6, **SMALL, **kw}
    if split is not None:
        over |= {"split_file": str(split), "eval_splits": ["val", "test"], "figure_split": "val"}
    return load_config(CFG).with_overrides(**over)


def _write(tmp_path: Path, cfg: TrainConfig) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    return p


def _metrics(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


def _history(d: Path) -> list[dict]:
    with (d / "history.csv").open() as fh:
        keys = ("epoch", "train_loss", "val_loss", "lr")
        return [{k: r[k] for k in keys} for r in csv.DictReader(fh)]


def _same_weights(a: Path, b: Path) -> bool:
    sa, sb = torch.load(a)["model"], torch.load(b)["model"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sb)


def _files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in d.iterdir()}


def _rows(vals: list, train: list | None = None) -> list[dict]:
    train = train or [0.1] * len(vals)
    return [
        {"epoch": i + 1, "train_loss": t, "val_loss": v}
        for i, (t, v) in enumerate(zip(train, vals, strict=True))
    ]


def _fired(vals: list, **es):
    st = early_stop_state(_rows(vals), EarlyStop(**es))
    return None if st["fired"] is None else st["fired"]["epoch"]


# ------------------------------------------------------------------ config (FR-001)


def test_guard_defaults_and_legacy_values():
    cfg = TrainConfig("d", "s.json")
    assert {k: getattr(cfg, k) for k in GUARD_KEYS} == GUARD_KEYS
    assert {k: LEGACY_VALUES[k] for k in GUARD_KEYS} == GUARD_KEYS
    assert config_from_dict(BASE) == cfg and cfg.to_dict()["early_stop"] is None


def test_early_stop_block_is_filled_normalized_and_round_trips():
    cfg = config_from_dict(BASE | {"early_stop": {"patience": 10}})
    assert cfg.early_stop == EarlyStop(10, 0, 0.0)
    full = config_from_dict(
        BASE
        | {
            "early_stop": {"patience": 10, "min_epochs": 20, "min_delta": 0},
            "abort_on_divergence": True,
            "divergence_ratio": 4,
        }
    )
    assert full.to_dict()["early_stop"] == {"patience": 10, "min_epochs": 20, "min_delta": 0.0}
    assert isinstance(full.early_stop.min_delta, float) and full.divergence_ratio == 4.0
    assert isinstance(full.divergence_ratio, float)
    assert config_from_dict(yaml.safe_load(yaml.safe_dump(full.to_dict()))) == full
    assert full.with_overrides(seed=3).early_stop == full.early_stop


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"early_stop": 10}, "early_stop"),
        ({"early_stop": [10]}, "early_stop"),
        ({"early_stop": {}}, "early_stop.patience"),
        ({"early_stop": {"patience": 0}}, "early_stop.patience"),
        ({"early_stop": {"patience": True}}, "early_stop.patience"),
        ({"early_stop": {"patience": 2.0}}, "early_stop.patience"),
        ({"early_stop": {"patience": 2, "min_epochs": -1}}, "early_stop.min_epochs"),
        ({"early_stop": {"patience": 2, "min_epochs": 1.5}}, "early_stop.min_epochs"),
        ({"early_stop": {"patience": 2, "min_delta": -0.1}}, "early_stop.min_delta"),
        ({"early_stop": {"patience": 2, "min_delta": NAN}}, "early_stop.min_delta"),
        ({"early_stop": {"patience": 2, "min_delta": "0"}}, "early_stop.min_delta"),
        ({"early_stop": {"patience": 2, "mode": "min"}}, "early_stop: unknown"),
        ({"abort_on_divergence": 1}, "abort_on_divergence"),
        ({"abort_on_divergence": "yes"}, "abort_on_divergence"),
        ({"abort_on_divergence": True, "divergence_ratio": 1.0}, "divergence_ratio"),
        ({"abort_on_divergence": True, "divergence_ratio": INF}, "divergence_ratio"),
        ({"abort_on_divergence": True, "divergence_ratio": "10"}, "divergence_ratio"),
        ({"divergence_ratio": 5.0}, "divergence_ratio: needs abort_on_divergence"),
    ],
)
def test_invalid_guard_keys(kw, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(BASE | kw)


def test_invalid_guard_key_exits_2_before_any_run_dir(tmp_path):
    c = yaml.safe_load(CFG.read_text()) | {"run_root": str(tmp_path / "r")}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c | {"early_stop": {"patience": 0}}))
    assert main(["--config", str(p)]) == 2 and not (tmp_path / "r").exists()


# ------------------------------------------------------------------ early-stop rule (FR-002)


def test_early_stop_fires_patience_epochs_after_the_last_improvement():
    assert _fired([0.5, 0.3, 0.4, 0.35, 0.31, 0.2], patience=3) == 5
    assert _fired([0.5, 0.3, 0.4, 0.35], patience=3) is None  # not yet
    assert _fired([0.5, 0.4, 0.3, 0.2], patience=1) is None  # always improving
    assert _fired([0.5, 0.5, 0.5], patience=2) == 3  # equal is no improvement


def test_early_stop_waits_for_min_epochs():
    vals = [0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
    assert _fired(vals, patience=2) == 3
    assert _fired(vals, patience=2, min_epochs=5) == 5
    assert _fired(vals, patience=2, min_epochs=7) is None


def test_min_delta_is_measured_against_the_anchor():
    """Steps below min_delta accumulate (Keras / Lightning semantics, research R1)."""
    vals = [0.5, 0.496, 0.492, 0.488, 0.487, 0.486]
    st = early_stop_state(_rows(vals), EarlyStop(patience=3, min_delta=0.01))
    assert (st["anchor_epoch"], st["since"], st["earliest_stop"], st["fired"]) == (4, 2, 7, None)
    assert _fired(vals, patience=3) is None  # min_delta 0: every epoch improves
    assert _fired([0.5, 0.495, 0.49, 0.489], patience=3, min_delta=0.02) == 4


def test_nan_epochs_count_but_never_decide_and_empty_val_never_stops():
    assert _fired([0.3, NAN, NAN, 0.4], patience=2) == 4
    assert _fired([0.3, NAN, NAN, NAN], patience=2) is None
    assert _fired([NAN, 0.3, 0.4, 0.4], patience=2) == 4  # NaN never anchors
    assert _fired([None] * 8, patience=1) is None
    assert early_stop_state(_rows([0.1]), None) is None


def test_early_stop_state_fields_and_reason():
    st = early_stop_state(_rows([0.5, 0.3, 0.4, 0.45]), EarlyStop(patience=5, min_epochs=10))
    want = {"anchor_epoch": 2, "anchor_loss": 0.3, "since": 2, "earliest_stop": 10, "fired": None}
    assert st == want
    st = early_stop_state(_rows([0.5, 0.3, 0.4, 0.45, 0.2]), EarlyStop(patience=2))
    assert st["fired"] == {"epoch": 4, "reason": REASON} and st["since"] == 2


# ------------------------------------------------------------------ divergence rule (FR-004)


def test_divergence_reasons_and_order():
    ok = _rows([0.3] * 5, train=[0.1, 0.05, 0.04, 0.03, 0.02])
    assert divergence_reason(ok, 10.0) is None and divergence_reason([], 10.0) is None
    assert divergence_reason(_rows([0.3, 0.3], [0.1, NAN]), 10.0) == (
        "train loss nan at epoch 2 is not finite"
    )
    assert divergence_reason(_rows([0.3], [INF]), 10.0) == "train loss inf at epoch 1 is not finite"
    assert divergence_reason(_rows([0.3] * 3, [0.01, 0.5, 2.0]), 10.0) is None  # grace epochs
    assert divergence_reason(_rows([0.3] * 4, [0.1, 0.05, 0.05, 1.0]), 10.0) is None  # not above
    assert divergence_reason(_rows([0.3] * 4, [0.1, 0.05, 0.05, 1.01]), 10.0) == (
        "train loss 1.01 at epoch 4 exceeds 10 x the epoch-1 train loss 0.1"
    )
    assert divergence_reason(_rows([0.3] * 4, [0.1, 0.05, 0.05, 0.31]), 3.0) is not None
    assert divergence_reason(_rows([0.3, NAN, INF, NAN]), 10.0) == (
        "val loss not finite for 3 epochs (2-4)"
    )
    assert divergence_reason(_rows([NAN, NAN, 0.3, NAN, NAN]), 10.0) is None
    assert divergence_reason(_rows([None] * 5), 10.0) is None  # empty val never counts
    both = _rows([NAN, NAN, NAN], [0.1, 0.1, NAN])
    assert divergence_reason(both, 10.0).startswith("train loss nan")  # (a) before (c)


def test_abort_record_keeps_non_finite_values_as_strings():
    rows = _rows([0.3, 0.2, NAN, None], [0.1, 0.05, INF, NAN])
    rec = abort_record(rows, "why", 10.0)
    assert rec == {
        "epoch": 4,
        "reason": "why",
        "divergence_ratio": 10.0,
        "first_train_loss": 0.1,
        "last_losses": [
            {"epoch": 2, "train_loss": 0.05, "val_loss": 0.2},
            {"epoch": 3, "train_loss": "inf", "val_loss": "nan"},
            {"epoch": 4, "train_loss": "nan", "val_loss": None},
        ],
    }
    json.dumps(rec, allow_nan=False)  # writable


# ------------------------------------------------------------------ early stop end to end


def test_early_stop_ends_the_run_as_a_prefix_of_the_full_run(tmp_path, val_split, scripted):
    """US1-1/2: best epoch 2, patience 2: stops after epoch 4 of 6, records why; history and lr
    are the full run's first rows; selection, scores, figure and best.pt equal the full run's."""
    scripted(SCRIPT)
    full = run_training(_cfg(tmp_path, "full", val_split))
    scripted(SCRIPT)
    logs: list[str] = []
    d = run_training(_cfg(tmp_path, "es", val_split, early_stop={"patience": 2}), log=logs.append)
    m, mf = _metrics(d), _metrics(full)
    assert m["training"]["stopped_early"] == {"epoch": 4, "reason": REASON}
    assert "stopped_early" not in mf["training"]
    assert f"early stop after epoch 4/6: {REASON}" in logs
    assert torch.load(d / "checkpoint.pt")["epoch"] == 4
    h, hf = _history(d), _history(full)
    assert len(h) == 4 and h == hf[:4]  # lr included: the schedule spans the planned 6 epochs
    assert m["selection"] == mf["selection"] and m["selection"]["epoch"] == 2
    assert m["splits"] == mf["splits"] and _same_weights(d / "best.pt", full / "best.pt")
    assert (d / "recon_val.png").read_bytes() == (full / "recon_val.png").read_bytes()
    assert m["training"]["epochs"] == 6 and m["training"]["final_val_loss"] == 0.35


def test_resume_recomputes_the_same_stop(tmp_path, val_split, scripted):
    """US1-3: min_epochs holds the stop to epoch 5; kills before it and after its checkpoint
    resume to the same stop and a byte-identical metrics.json."""
    script = [0.5, 0.3, 0.4, 0.45, 0.5, 0.6]
    es = {"patience": 1, "min_epochs": 5}
    scripted(script)
    full = run_training(_cfg(tmp_path, "full", val_split, early_stop=es))
    assert _metrics(full)["training"]["stopped_early"]["epoch"] == 5  # not 3: min_epochs
    scripted(script)
    early = run_training(_cfg(tmp_path, "early", val_split, early_stop=es), stop_after_epoch=3)
    assert main(["--resume", str(early)]) == 0
    s = scripted(script)
    at = run_training(_cfg(tmp_path, "at", val_split, early_stop=es), stop_after_epoch=5)
    assert not (at / "metrics.json").exists() and s.calls == 5
    assert main(["--resume", str(at)]) == 0 and s.calls == 5  # no epoch trained
    for d in (early, at):
        assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes(), d
        assert torch.load(d / "checkpoint.pt")["epoch"] == 5


def test_empty_val_never_stops_and_says_so(tmp_path):
    logs: list[str] = []
    d = run_training(_cfg(tmp_path, "r", epochs=4, early_stop={"patience": 1}), log=logs.append)
    assert "early stop inactive: val is empty" in logs
    assert _metrics(d)["training"]["stopped_early"] is None
    assert torch.load(d / "checkpoint.pt")["epoch"] == 4


# ------------------------------------------------------------------ divergence abort (FR-005/006)


def test_non_finite_train_loss_aborts_with_exit_5(tmp_path, scripted_train, capsys):
    scripted_train([None, NAN])
    p = _write(tmp_path, _cfg(tmp_path, "r", abort_on_divergence=True))
    assert main(["--config", str(p)]) == 5
    (d,) = (tmp_path / "r").iterdir()
    rec = json.loads((d / ABORTED).read_text())
    assert rec["epoch"] == 2 and rec["reason"] == "train loss nan at epoch 2 is not finite"
    assert rec["last_losses"][-1]["train_loss"] == "nan" and rec["divergence_ratio"] == 10.0
    assert not (d / "metrics.json").exists() and not list(d.glob("*.png"))
    assert torch.load(d / "checkpoint.pt")["epoch"] == 2  # post-mortem state kept
    assert "aborted: train loss nan at epoch 2 is not finite" in capsys.readouterr().err
    assert not (d / LOCK).exists()  # released on abort


def test_train_loss_ratio_aborts_only_after_the_grace_epochs(tmp_path, scripted_train):
    scripted_train([0.01, 0.2, 0.2, 0.2])
    with pytest.raises(RunAborted, match="epoch 4 exceeds 10 x the epoch-1 train loss 0.01"):
        run_training(_cfg(tmp_path, "r", abort_on_divergence=True))
    (d,) = (tmp_path / "r").iterdir()
    assert json.loads((d / ABORTED).read_text())["epoch"] == 4


def test_three_non_finite_val_losses_abort(tmp_path, val_split, scripted):
    scripted([0.5, NAN, NAN, NAN, 0.3, 0.3])
    with pytest.raises(RunAborted, match=r"val loss not finite for 3 epochs \(2-4\)"):
        run_training(_cfg(tmp_path, "r", val_split, abort_on_divergence=True))
    (d,) = (tmp_path / "r").iterdir()
    rec = json.loads((d / ABORTED).read_text())
    assert [r["val_loss"] for r in rec["last_losses"]] == ["nan", "nan", "nan"]
    assert torch.load(d / "best.pt")["epoch"] == 1


def test_abort_wins_over_an_early_stop_at_the_same_epoch(
    tmp_path, val_split, scripted, scripted_train
):
    scripted([0.3, 0.4, 0.5])
    scripted_train([None, None, NAN])
    cfg = _cfg(tmp_path, "r", val_split, early_stop={"patience": 2}, abort_on_divergence=True)
    with pytest.raises(RunAborted):
        run_training(cfg)
    (d,) = (tmp_path / "r").iterdir()
    assert (d / ABORTED).exists() and not (d / "metrics.json").exists()


def test_aborted_run_is_not_resumable(tmp_path, scripted_train, capsys):
    scripted_train([None, NAN])
    with pytest.raises(RunAborted):
        run_training(_cfg(tmp_path, "r", abort_on_divergence=True))
    (d,) = (tmp_path / "r").iterdir()
    before = _files(d)
    assert main(["--resume", str(d)]) == 5
    err = capsys.readouterr().err
    assert "aborted at epoch 2" in err and "start a new run" in err
    assert _files(d) == before  # nothing written, no lock left


def test_kill_before_aborted_json_aborts_on_resume_without_training(tmp_path, scripted_train):
    s = scripted_train([None, None, NAN])
    d = run_training(_cfg(tmp_path, "r", abort_on_divergence=True), stop_after_epoch=3)
    assert not (d / ABORTED).exists() and s.calls == 3  # the simulated kill came first
    with pytest.raises(RunAborted, match="epoch 3"):
        resume_training(d)
    assert s.calls == 3 and json.loads((d / ABORTED).read_text())["epoch"] == 3
    assert torch.load(d / "checkpoint.pt")["epoch"] == 3


# ------------------------------------------------------------------ run-dir lock (FR-007)


def test_lock_is_held_while_running_and_removed_after(tmp_path):
    seen = []

    def log(msg: str) -> None:
        if msg.startswith("epoch 1/"):
            seen.append(read_lock(next((tmp_path / "r").iterdir()) / LOCK))

    d = run_training(_cfg(tmp_path, "r", epochs=2), log=log)
    assert seen[0]["pid"] == os.getpid() and seen[0]["boot_id"] == boot_id()
    assert not (d / LOCK).exists()


def test_live_lock_refuses_and_stale_lock_is_replaced(tmp_path, helper, capsys):
    d = run_training(_cfg(tmp_path, "r", epochs=3), stop_after_epoch=1)
    lock = {"argv": ["strike-a-pose-train"], "boot_id": boot_id(), "pid": helper.pid,
            "started_utc": "2026-10-21T00:00:00+00:00"}  # fmt: skip
    (d / LOCK).write_text(json.dumps(lock))
    assert lock_alive(lock)
    assert main(["--resume", str(d)]) == 2
    err = capsys.readouterr().err
    assert str(helper.pid) in err and LOCK in err
    assert json.loads((d / LOCK).read_text()) == lock and not (d / "metrics.json").exists()
    helper.kill()
    helper.wait()
    assert not lock_alive(lock)  # its process is gone: stale
    assert main(["--resume", str(d)]) == 0 and not (d / LOCK).exists()


def test_lock_liveness_rejects_reused_pids_and_other_boots(helper):
    assert not lock_alive({"pid": os.getpid(), "boot_id": boot_id()})  # alive, not a trainer
    assert not lock_alive({"pid": "1"}) and not lock_alive({}) and not lock_alive(None)
    assert lock_alive({"pid": helper.pid})  # no boot id recorded: the process decides
    if boot_id() is not None:
        assert not lock_alive({"pid": helper.pid, "boot_id": "another-boot"})


def test_lock_liveness_accepts_python_m_launches():
    """A trainer started as `python -m strike_a_pose.train.cli ...` is alive (its /proc cmdline
    lacks the console-script name); found in live use 2026-10-21."""
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", "-m", "strike_a_pose.train.cli"]
    )
    try:
        _wait_for_exec(p, b"strike_a_pose.train.cli")
        assert lock_alive({"pid": p.pid, "boot_id": boot_id()})
    finally:
        p.kill()
        p.wait()
    assert not lock_alive({"pid": p.pid, "boot_id": boot_id()})


def test_lock_is_released_after_an_error(tmp_path):
    d = run_training(_cfg(tmp_path, "r", epochs=3), stop_after_epoch=1)
    c = yaml.safe_load((d / "config.yaml").read_text())
    (d / "config.yaml").write_text(yaml.safe_dump(c | {"lr": 0.5}))  # config changed
    assert main(["--resume", str(d)]) == 2 and not (d / LOCK).exists()


# ------------------------------------------------------------------ legacy and defaults


def test_run_dir_from_before_012_resumes_with_the_guards_off(tmp_path, val_split, scripted):
    """FR-008: no 012 key in config.yaml, checkpoint.pt or best.pt: the run resumes and equals
    an uninterrupted run."""
    scripted(SCRIPT)
    full = run_training(_cfg(tmp_path, "full", val_split))
    scripted(SCRIPT)
    d = run_training(_cfg(tmp_path, "cut", val_split), stop_after_epoch=2)
    c = yaml.safe_load((d / "config.yaml").read_text())
    (d / "config.yaml").write_text(
        yaml.safe_dump({k: v for k, v in c.items() if k not in GUARD_KEYS})
    )
    for name in ("checkpoint.pt", "best.pt"):
        obj = torch.load(d / name)
        obj["config"] = {k: v for k, v in obj["config"].items() if k not in GUARD_KEYS}
        torch.save(obj, d / name)
    assert main(["--resume", str(d)]) == 0
    assert (d / "metrics.json").read_bytes() == (full / "metrics.json").read_bytes()


def test_default_run_writes_no_guard_records(tmp_path):
    d = run_training(_cfg(tmp_path, "r", epochs=2))
    assert "stopped_early" not in _metrics(d)["training"]
    assert not (d / ABORTED).exists() and not (d / LOCK).exists()
    assert yaml.safe_load((d / "config.yaml").read_text())["early_stop"] is None
    assert all(math.isfinite(float(r["train_loss"])) for r in _history(d))


# ------------------------------------------------------------------ campaign configs (FR-012)

CAMPAIGN = ("ae", "ae_mask", "ae_vae", "ae_reg", "ae_seed1", "ae_vae_beta0.1", "ae_vae_beta0.001")
GUARDS_ON = {
    "early_stop": {"patience": 10, "min_epochs": 20, "min_delta": 0.0},
    "abort_on_divergence": True,
    "divergence_ratio": 10.0,
}


@pytest.mark.parametrize("name", CAMPAIGN)
def test_campaign_configs_list_the_guards(name):
    raw = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    assert {k: raw[k] for k in GUARDS_ON} == GUARDS_ON
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.early_stop == EarlyStop(10, 20, 0.0) and cfg.abort_on_divergence


def test_fixture_config_has_no_guards():
    assert not set(GUARDS_ON) & set(yaml.safe_load(CFG.read_text()))
    cfg = load_config(CFG)
    assert cfg.early_stop is None and not cfg.abort_on_divergence


@pytest.mark.parametrize(
    "name, parent, diff",
    [
        ("ae_seed1", "ae", {"seed": 1, "run_name": "ae-seed1-smpl_proc_v1"}),
        ("ae_vae_beta0.1", "ae_vae", {"beta": 0.1, "run_name": "ae-vae-beta0.1-smpl_proc_v1"}),
        (
            "ae_vae_beta0.001",
            "ae_vae",
            {"beta": 0.001, "run_name": "ae-vae-beta0.001-smpl_proc_v1"},
        ),
    ],
)
def test_new_campaign_configs_differ_from_their_parent_only_in_their_knob(name, parent, diff):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml").to_dict()
    base = load_config(ROOT / "configs" / f"{parent}.yaml").to_dict()
    assert {k: v for k, v in cfg.items() if v != base[k]} == diff
