"""Pose-track ranking and report (spec 018 FR-006)."""

import importlib.util
import json
from pathlib import Path

import pytest

from strike_a_pose.eval.pose_track import PoseEntry, rank, second_seed, seed_stats

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("pose_track_script", ROOT / "scripts/pose_track.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

SEEDS = [
    PoseEntry("s0", 131.0, None, 0.001578),
    PoseEntry("s1", 134.4, None, 0.001587),
    PoseEntry("s2", 133.1, None, 0.001573),
]


def test_seed_stats():
    mean, sd = seed_stats([131.0, 134.4, 133.1])
    assert mean == pytest.approx(132.8333, abs=1e-4)
    assert sd == pytest.approx(1.71561, abs=1e-5)
    with pytest.raises(ValueError):
        seed_stats([131.0])
    with pytest.raises(ValueError):
        seed_stats([1.0, 1.0])


def test_rank_orders_and_flags():
    ents = [
        PoseEntry("R1-v3", 124.5, None, 0.00181),
        PoseEntry("P1", 78.0, 70.0, 0.00170),
        PoseEntry("P2", 130.0, 85.0, 0.00160),
    ]
    r = {x.name: x for x in rank(ents, SEEDS)}
    assert [x.name for x in rank(ents, SEEDS)] == ["P1", "R1-v3", "P2"]
    assert r["R1-v3"].d_mm == pytest.approx(-8.3333, abs=1e-4)
    assert r["R1-v3"].z == pytest.approx(-4.8573, abs=1e-4)
    assert r["R1-v3"].real is True and r["R1-v3"].head_ok is None
    assert r["P2"].d_mm == pytest.approx(-2.8333, abs=1e-4)
    assert r["P2"].z == pytest.approx(-1.6515, abs=1e-4)
    assert r["P2"].real is False and r["P2"].head_ok is False
    assert r["P1"].probe_ok is True and r["P1"].head_ok is True
    assert r["P1"].d_mse_rel == pytest.approx(0.00170 / 0.0015793333 - 1, rel=1e-6)


def test_second_seed_rule():
    sd = seed_stats([s.probe_mm for s in SEEDS])[1]
    far = rank([PoseEntry("A", 120.0, None, 0.0016), PoseEntry("B", 131.0, None, 0.0016)], SEEDS)
    assert second_seed(far, sd)[0] is False  # 11 mm apart (> 3 sd), 40 mm from the line
    close = rank([PoseEntry("A", 120.0, None, 0.0016), PoseEntry("B", 123.0, None, 0.0016)], SEEDS)
    assert second_seed(close, sd)[0] is True  # 3 mm apart (< 3 sd = 5.1 mm)
    line = rank([PoseEntry("A", 90.0, 81.0, 0.0016), PoseEntry("B", 120.0, None, 0.0016)], SEEDS)
    need, why = second_seed(line, sd)
    assert need is True and "head" in why  # head 1 mm from 80 mm
    assert second_seed([], sd)[0] is False


def _write_run(d: Path, split_mse: dict, head: dict | None = None) -> str:
    d.mkdir(parents=True)
    splits = {}
    for sp, mse in split_mse.items():
        splits[sp] = {"image": {"ae": {"image_mse": mse}}}
        if head and sp in head:
            splits[sp]["pose_head"] = {"target": "joints_world", "mpjpe_mm": head[sp]}
    (d / "metrics.json").write_text(json.dumps({"splits": splits}))
    return str(d)


def _write_probe(d: Path, mm: dict) -> str:
    d.mkdir(parents=True)
    splits = {sp: {"targets": {"joints_world": {"mlp": {"mpjpe_mm": v}}}} for sp, v in mm.items()}
    (d / "metrics.json").write_text(json.dumps({"splits": splits}))
    return str(d)


def test_script_tables(tmp_path, capsys):
    args = []
    for i, (mm, mse) in enumerate([(131.0, 0.001578), (134.4, 0.001587), (133.1, 0.001573)]):
        r = _write_run(tmp_path / f"s{i}", {"val": mse, "test": mse})
        p = _write_probe(tmp_path / f"sp{i}", {"val": mm, "test": mm + 1})
        args += ["--seed", r, p]
    r1 = _write_run(tmp_path / "v1", {"val": 0.0017, "test": 0.0017}, {"val": 70.0, "test": 72.0})
    p1 = _write_probe(tmp_path / "vp1", {"val": 78.0, "test": 79.0})
    r2 = _write_run(tmp_path / "v2", {"val": 0.0016, "test": 0.0016})
    p2 = _write_probe(tmp_path / "vp2", {"val": 125.0, "test": 126.0})
    argv = [*args, "--variant", "P1", r1, p1, "--variant", "P2", r2, p2, "--split", "both"]
    assert script.main(argv) == 0
    out = capsys.readouterr().out
    h = "| Rank | Variant | MSE vs seeds | Probe MPJPE {} (mm) | Δ vs seeds (mm) | Δ / sd | "
    h += "Real (< −2 sd) | Head MPJPE {} (mm) | Probe ≤ 80 | Head ≤ 80 |"
    assert h.format("val", "val") in out and h.format("test", "test") in out
    assert out.count("seeds: n=3 mean 132.8 mm sd 1.7 mm") == 1  # val
    assert "seeds: n=3 mean 133.8 mm sd 1.7 mm (test)" in out
    p2_row = next(line for line in out.splitlines() if "| P2 |" in line)
    assert "| – | no | – |" in p2_row
    assert "second seed for P1" in out
    assert script.main(["--seed", r1, p1, "--variant", "P1", r1, p1]) == 2
    assert "need >= 2 --seed" in capsys.readouterr().err
    assert script.main([*args, "--variant", "X", str(tmp_path / "nope"), p1]) == 2
