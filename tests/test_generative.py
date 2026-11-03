"""Goal verdict report (spec 021 FR-004)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from strike_a_pose.eval.generative import (
    COMPARE_SD,
    HEAD_MAX_MM,
    MARGIN_MIN_PTS,
    MSE_COST_REL,
    PRIOR_VALID_MIN,
    GenEntry,
    axis_checks,
    compare_prior,
    format_markdown,
    goal_line_pixel,
    goal_verdict,
    load_gen_entry,
    mean_mse,
)

ROOT = Path(__file__).parents[1]
L2 = {"val": 0.001675765, "test": 0.001653395}
R0 = {"val": 0.0015793, "test": 0.00155856}
S = ("val", "test")


def _e(name="V1", head=(67.0, 64.7), mse=(0.00170, 0.00168), valid=0.59, n=1024,
       plaus=0.60, ctrl=0.46):  # fmt: skip
    return GenEntry(
        name, {"val": head[0], "test": head[1]}, {"val": mse[0], "test": mse[1]},
        valid, n, plaus, ctrl,
    )  # fmt: skip


def test_constants():
    assert (HEAD_MAX_MM, MSE_COST_REL, PRIOR_VALID_MIN, MARGIN_MIN_PTS, COMPARE_SD) == (
        80.0, 0.05, 0.56, 13.0, 2.0,
    )  # fmt: skip


def test_margin_pts():
    e = GenEntry("x", {}, {}, None, None, 0.457031, 0.326172)
    assert e.margin_pts == pytest.approx(13.0859, abs=1e-9)
    assert GenEntry("x", {}, {}, None, None, None, 0.3).margin_pts is None
    assert GenEntry("x", {}, {}, None, None, 0.3, None).margin_pts is None


def test_goal_met():
    assert goal_verdict(axis_checks(_e(), L2, S)) == ("GOAL-MET", [], "")


def test_bounds_inclusive():
    e = _e(head=(80.0, 80.0), mse=(0.001675765 * 1.05, 0.001653395 * 1.05),
           valid=0.56, plaus=0.59, ctrl=0.46)  # fmt: skip
    assert goal_verdict(axis_checks(e, L2, S)) == ("GOAL-MET", [], "")


def test_partial_pose():
    got = goal_verdict(axis_checks(_e(head=(79.0, 80.1)), L2, S))
    assert got == ("PARTIAL", ["pose"], "beta 0.03")


def test_partial_valid():
    got = goal_verdict(axis_checks(_e(valid=0.5599), L2, S))
    assert got == ("PARTIAL", ["valid"], "head on z")


def test_partial_pixel_and_margin():
    e = _e(mse=(0.00176, 0.00168), plaus=0.55, ctrl=0.46)
    got = goal_verdict(axis_checks(e, L2, S))
    # operator ruling 2026-11-03: the generation step ("head on z") comes first
    assert got == ("PARTIAL", ["pixel", "margin"], "head on z; beta 0.03")


def test_partial_pose_and_pixel_single_step():
    e = _e(head=(81.0, 81.0), mse=(0.0018, 0.0018))
    assert goal_verdict(axis_checks(e, L2, S)) == ("PARTIAL", ["pose", "pixel"], "beta 0.03")


def test_partial_all_fail_head_on_z_first():
    e = _e(head=(90.0, 90.0), mse=(0.0019, 0.0019), valid=0.3, plaus=0.4, ctrl=0.4)
    got = goal_verdict(axis_checks(e, L2, S))
    assert got == ("PARTIAL", ["pose", "pixel", "valid", "margin"], "head on z; beta 0.03")


def test_incomplete():
    assert goal_verdict(axis_checks(_e(valid=None), L2, S)) == ("INCOMPLETE", ["valid"], "")
    e = _e()
    e = GenEntry(e.name, {"val": None, "test": 64.7}, e.mse, e.prior_valid, e.prior_n,
                 e.rt_plausible, e.rt_control)  # fmt: skip
    v, failing, _ = goal_verdict(axis_checks(e, L2, S))
    assert v == "INCOMPLETE" and "pose" in failing


def test_compare_prior():
    label, diff, sd = compare_prior(_e(valid=0.62), _e(valid=0.59))
    assert label == "NO-DIFFERENCE"
    assert diff == pytest.approx(0.03, abs=1e-12)
    assert sd == pytest.approx(0.0215941796, abs=1e-9)
    label, _, sd = compare_prior(_e(valid=0.65), _e(valid=0.59))
    assert label == "HELPS" and sd == pytest.approx(0.0214102414, abs=1e-9)
    assert compare_prior(_e(valid=0.59), _e(valid=0.65))[0] == "HURTS"
    label, diff, sd = compare_prior(_e(valid=None), _e())
    assert label == "INCOMPLETE" and math.isnan(diff) and math.isnan(sd)


def test_goal_line_pixel():
    assert goal_line_pixel(_e(), R0, S) is False
    assert goal_line_pixel(_e(mse=(0.00165, 0.00163)), R0, S) is True


def _write(p: Path, obj) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    f = p / ("metrics.json" if "splits" in obj else "samples.json")
    f.write_text(json.dumps(obj))
    return p


def _files(tmp_path: Path):
    def m(v, t, head=True):
        sp = {
            "val": {"image": {"ae": {"image_mse": v}}},
            "test": {"image": {"ae": {"image_mse": t}}},
        }
        if head:
            sp["val"]["pose_head"] = {"mpjpe_mm": 66.9733}
            sp["test"]["pose_head"] = {"mpjpe_mm": 64.7}
        return {"splits": sp}

    t1 = _write(tmp_path / "t1", m(0.00166906, 0.0016472))
    t2 = _write(tmp_path / "t2", m(0.00168247, 0.00165959, head=False))
    s = _write(tmp_path / "s", {"sets": {"vae_prior": {"valid_fraction": 0.589844, "n": 1024}}})
    r = _write(
        tmp_path / "r",
        {"sets": {"vae_prior": {"round_trip": {
            "plausible_fraction": 0.457031, "control_plausible_fraction": 0.326172}}}},
    )  # fmt: skip
    return t1, t2, s, r


def test_mean_mse_and_load_entry(tmp_path):
    t1, t2, s, r = _files(tmp_path)
    got = mean_mse([t1, t2], S)
    assert got["val"] == pytest.approx(0.001675765, rel=1e-12)
    assert got["test"] == pytest.approx(0.001653395, rel=1e-12)
    e = load_gen_entry("V1", t1, s, r)
    assert e.name == "V1" and e.head_mm == {"val": 66.9733, "test": 64.7}
    assert e.mse == {"val": 0.00166906, "test": 0.0016472}
    assert (e.prior_valid, e.prior_n) == (0.589844, 1024)
    assert (e.rt_plausible, e.rt_control) == (0.457031, 0.326172)
    x = load_gen_entry("x", t2)
    assert x.head_mm == {"val": None, "test": None}
    assert x.prior_valid is None and x.prior_n is None
    assert x.rt_plausible is None and x.rt_control is None
    with pytest.raises(FileNotFoundError, match="nope"):
        load_gen_entry("x", tmp_path / "nope")
    with pytest.raises(ValueError, match=">= 1"):
        mean_mse([], S)


def test_format_markdown(tmp_path):
    t1, _, s, r = _files(tmp_path)
    md = format_markdown([load_gen_entry("V1", t1, s, r), _e("bare", valid=None)], L2, R0, "val")
    lines = md.splitlines()
    assert lines[0] == (
        "| model | head mm | MSE | MSE / L2 | pixel (≤ L2 +5 %) | vs R0 64-d +5 % "
        "| prior valid | RT margin pts |"
    )
    assert "67.0" in lines[2] and "0.590" in lines[2] and "13.1" in lines[2]
    assert "n/a" in lines[3]


def _cli(*args):
    return subprocess.run(
        [sys.executable, "scripts/generative.py", *args], capture_output=True, text=True, cwd=ROOT
    )


def test_cli_report(tmp_path):
    t1, t2, s, r = _files(tmp_path)
    out = tmp_path / "v.json"
    args = ["report", "--v1", t1, s, r, "--v2", t1, s, r, "--l2", t1, "--l2", t2, "--r0", t2,
            "--split", "both", "--json", out]  # fmt: skip
    p = _cli(*map(str, args))
    assert p.returncode == 0, p.stderr
    assert "## val" in p.stdout and "## test" in p.stdout
    assert "V2 vs V1 prior: NO-DIFFERENCE" in p.stdout
    assert p.stdout.strip().splitlines()[-1] == "VERDICT GOAL-MET failing=- next=-"
    assert "QUEUE_DONE" not in p.stdout
    j = json.loads(out.read_text())
    assert j["verdict"] == "GOAL-MET" and j["generative_version"] == 1
    assert set(j["entries"]) == {"V1", "V2"}
    p = _cli(*map(str, args[:2] + [t1, s, tmp_path / "gone"] + args[5:]))
    assert p.returncode == 2 and "gone" in p.stderr
