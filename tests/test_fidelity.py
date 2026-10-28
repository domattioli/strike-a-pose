"""Rescore and fidelity report (spec 020 FR-005)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from strike_a_pose.data import make_split, write_split
from strike_a_pose.datagen.io import load_manifest
from strike_a_pose.eval.fidelity import (
    MSE_COST_REL,
    REAL_GAIN_SD,
    FidelityEntry,
    fidelity_rows,
    load_entry,
    rescore,
    write_fidelity,
)
from strike_a_pose.runlog import round_sig
from strike_a_pose.train.config import load_config
from strike_a_pose.train.trainer import run_training

ROOT = Path(__file__).parents[1]
CFG = ROOT / "configs" / "ae_fixture.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "smpl_fixture"
SMALL = {"channels": [4, 4, 4, 4, 4, 4], "latent_dim": 8}


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fid")
    sp = tmp / "split.json"
    write_split(make_split(load_manifest(FIXTURE), 0, (0.5, 0.5, 0.0)), sp)
    cfg = load_config(CFG).with_overrides(
        run_root=str(tmp / "r"),
        split_file=str(sp),
        epochs=2,
        eval_splits=["val"],
        figure_split="val",
        **SMALL,
    )
    return run_training(cfg)


def test_rescore_matches_training_metrics(run_dir):
    res = rescore(run_dir, splits=("val",))
    m = json.loads((run_dir / "metrics.json").read_text())
    assert res["pose_weight"] == 0.0 and res["gradient_weight"] == 0.0
    for s in ("val",):  # the 2-pose fixture has no test pose
        want = m["splits"][s]["image"]["ae"]
        got = res["splits"][s]
        for k in ("image_mse", "silhouette_iou", "boundary_f1"):
            assert got[k] == pytest.approx(want[k], rel=1e-5)
        for name, r in want["regions"].items():
            assert got["regions"][name]["image_mse"] == pytest.approx(r["image_mse"], rel=1e-5)
    path = write_fidelity(run_dir, res)
    assert json.loads(path.read_text()) == round_sig(res)  # written at 6 significant digits
    e = load_entry(run_dir, "val")
    assert e.kind == "plain" and e.probe_mm is None and e.bf1 == res["splits"]["val"]["boundary_f1"]


def test_rescore_deterministic(run_dir):
    assert rescore(run_dir, ("val",)) == rescore(run_dir, ("val",))


def _e(name, kind, bf1, mse=0.0016, probe=None):
    reg = {k: {"image_mse": 0.01, "samples": 5} for k in ("hands", "feet", "head")}
    return FidelityEntry(name, kind, mse, 0.9, bf1, reg, probe)


def test_report_verdicts():
    seeds = [
        _e("s0", "plain", 0.92, probe=131.0),
        _e("s1", "plain", 0.93, probe=133.0),
        _e("s2", "plain", 0.94, probe=134.4),
    ]
    refs = [_e("p0", "pose", 0.86, 0.0022, 140.0), _e("p1", "pose", 0.862, 0.0022, 142.0)]
    mse0 = 0.0016
    vs = [
        _e("A", "plain", 0.96, mse0 * 1.03, 132.8 + 1.0),
        _e("B", "plain", 0.96, mse0 * 1.08, 132.8),
        _e("C", "plain", 0.935, mse0, 132.8),
        _e("D", "pose", 0.90, 0.0022, 141.0 + 10.0),
        _e("E", "plain", 0.96, mse0, None),
    ]
    rows = {r.name: r for r in fidelity_rows(seeds, refs, vs, "val")}
    assert {n: rows[n].verdict for n in "ABCDE"} == {
        "A": "SHARPER", "B": "SHARPER-COSTLY", "C": "NO-GAIN",
        "D": "SHARPER-COSTLY", "E": "NO-PROBE",
    }  # fmt: skip
    a = rows["A"]
    assert a.d_bf1 == pytest.approx(0.03, abs=1e-9)
    assert a.d_bf1_sd == pytest.approx(0.03 / 0.01, abs=1e-9)
    assert a.mse_rel == pytest.approx(0.03, abs=1e-9)
    assert a.d_probe_mm == pytest.approx(1.0, abs=1e-9)
    assert rows["D"].d_bf1 == pytest.approx(0.90 - 0.861, abs=1e-9)
    assert rows["D"].d_probe_mm == pytest.approx(10.0, abs=1e-9)
    assert rows["s0"].verdict == "—" and rows["p1"].ref == "pose-ref"
    assert REAL_GAIN_SD == 2.0 and MSE_COST_REL == 0.05


def test_report_needs_two_seeds():
    with pytest.raises(ValueError, match=">= 2"):
        fidelity_rows([_e("s0", "plain", 0.9, probe=1.0)], [], [], "val")


def _cli(*args):
    return subprocess.run(
        [sys.executable, "scripts/fidelity.py", *args], capture_output=True, text=True, cwd=ROOT
    )


def test_cli_report_exit_codes(tmp_path):
    p = _cli("report", "--seed", str(tmp_path / "a"), "--variant", f"X={tmp_path / 'b'}")
    assert p.returncode == 2
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    d1.mkdir()
    d2.mkdir()
    p = _cli("report", "--seed", str(d1), "--seed", str(d2))
    assert p.returncode == 2 and str(d1 / "fidelity.json") in p.stderr
