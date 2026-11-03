"""Promotion rule (spec 015 FR-006): pixel win or latent win; supersedes the 012 rule."""

import importlib.util
import json
from pathlib import Path

import pytest

from strike_a_pose.eval.promotion import Measures, pose_track_verdict, verdict

ROOT = Path(__file__).parents[1]
# rounded README numbers (analyze A6); the real verdicts come from scripts/promotion.py
R0 = Measures(0.00218, 174.8)
R0B = Measures(0.00216, 172.0)


def _label(m: Measures, variational: bool = False) -> str:
    return verdict(m, R0, R0B, variational=variational).label


def test_v1_campaign_verdicts():
    v = verdict(Measures(0.00236, 162.4), R0, R0B)
    assert v.label == "DROPPED"
    assert v.d_mse_rel == pytest.approx(0.00018 / 0.00218)
    assert v.d_mpjpe_mm == pytest.approx(-12.4)
    assert v.probe_gap_mm == pytest.approx(2.8)
    assert v.seed_gap_rel == pytest.approx(0.00002 / 0.00218)
    assert _label(Measures(0.00218, 174.4, train_kl=274.0), variational=True) == "REPORTED"
    assert _label(Measures(0.00221, 184.8)) == "REPORTED"


def test_pixel_win():
    assert _label(Measures(0.00200, 176.0)) == "WIN-pixel"
    assert _label(Measures(0.00200, 178.0)) == "REPORTED"


def test_latent_win():
    assert _label(Measures(0.00228, 168.0)) == "WIN-latent"
    assert _label(Measures(0.00228, 169.5)) == "REPORTED"
    assert _label(Measures(0.00229, 160.0)) == "DROPPED"


def test_boundaries():
    assert _label(Measures(0.00218 * 1.05, 160.0)) == "WIN-latent"  # +5 % inclusive
    exact = R0.mse * (1 - abs(R0B.mse - R0.mse) / R0.mse)
    assert _label(Measures(exact, 174.8)) != "WIN-pixel"  # strict


def test_pixel_wins_over_latent_when_both():
    assert _label(Measures(0.00200, 160.0)) == "WIN-pixel"


def test_vae_needs_kl():
    v = verdict(Measures(0.00200, 170.0, train_kl=0.5), R0, R0B, variational=True)
    assert v.label == "REPORTED" and "KL" in v.reason
    with pytest.raises(ValueError):
        verdict(Measures(0.00200, 170.0), R0, R0B, variational=True)


def test_zero_gaps():
    assert verdict(Measures(0.00217, 174.8), R0, R0).label == "WIN-pixel"
    assert verdict(Measures(0.00218, 174.8), R0, R0).label == "REPORTED"


def _script():
    spec = importlib.util.spec_from_file_location(
        "promotion_script", ROOT / "scripts" / "promotion.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(
    tmp: Path, name: str, mse: float, kl: list[str] | None = None, test_mse: float | None = None
) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "metrics.json").write_text(
        json.dumps(
            {"splits": {sp: {"image": {"ae": {"image_mse": m}}} for sp, m in _both(mse, test_mse)}}
        )
    )
    (d / "config.yaml").write_text(f"variational: {'true' if kl else 'false'}\n")
    rows = ["epoch,train_loss,val_loss,train_kl"]
    rows += [f"{i + 1},0.1,0.1,{k}" for i, k in enumerate(kl or ["", ""])]
    (d / "history.csv").write_text("\n".join(rows) + "\n")
    return d


def _both(val: float, test: float | None) -> list[tuple[str, float]]:
    return [("val", val), ("test", val if test is None else test)]


def _probe(tmp: Path, name: str, mp: float, test_mp: float | None = None) -> Path:
    d = tmp / name
    d.mkdir()
    splits = {
        sp: {"targets": {"joints_world": {"mlp": {"mpjpe_mm": m}}}} for sp, m in _both(mp, test_mp)
    }
    (d / "metrics.json").write_text(json.dumps({"splits": splits}))
    return d


def test_script_table(tmp_path, capsys):
    r0, p0 = _run(tmp_path, "r0", 0.00218), _probe(tmp_path, "p0", 174.8)
    r0b, p0b = _run(tmp_path, "r0b", 0.00216), _probe(tmp_path, "p0b", 172.0)
    a = [str(x) for x in (r0, p0, r0b, p0b)]
    r1, p1 = _run(tmp_path, "r1", 0.00236), _probe(tmp_path, "p1", 162.4)
    r2, p2 = _run(tmp_path, "r2", 0.00218, kl=["240.0", "274.0"]), _probe(tmp_path, "p2", 174.4)
    argv = ["--r0", a[0], "--r0-probe", a[1], "--r0b", a[2], "--r0b-probe", a[3],
            "--variant", "R1", str(r1), str(p1), "--variant", "R2", str(r2), str(p2)]  # fmt: skip
    assert _script().main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    rows = [line for line in out if line.startswith("| R")]
    assert len(rows) == 2
    assert "DROPPED" in rows[0] and "+8.3 %" in rows[0]
    assert "REPORTED" in rows[1]
    assert _script().main([*argv[:8], "--variant", "X", str(tmp_path / "no"), str(p1)]) == 2


def _split_argv(tmp_path: Path) -> list[str]:
    """Variant V: val says DROPPED (+8.3 % MSE, latent win too small), test says WIN-pixel."""
    r0, p0 = _run(tmp_path, "r0", 0.00218, test_mse=0.00300), _probe(tmp_path, "p0", 174.8, 190.0)
    r0b, p0b = (
        _run(tmp_path, "r0b", 0.00216, test_mse=0.00298),
        _probe(tmp_path, "p0b", 172.0, 188.0),
    )
    r1, p1 = _run(tmp_path, "r1", 0.00236, test_mse=0.00280), _probe(tmp_path, "p1", 174.0, 190.0)
    return ["--r0", str(r0), "--r0-probe", str(p0), "--r0b", str(r0b), "--r0b-probe", str(p0b),
            "--variant", "V", str(r1), str(p1)]  # fmt: skip


def test_split_val_is_default(tmp_path, capsys):
    argv = _split_argv(tmp_path)
    assert _script().main(argv) == 0
    default = capsys.readouterr().out
    assert _script().main([*argv, "--split", "val"]) == 0
    assert capsys.readouterr().out == default
    assert default.splitlines()[0].startswith("| Variant | Val MSE vs R0 | Probe MPJPE val (mm) |")
    assert "DROPPED" in default


def test_split_test_reads_test_numbers(tmp_path, capsys):
    assert _script().main([*_split_argv(tmp_path), "--split", "test"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("| Variant | Test MSE vs R0 | Probe MPJPE test (mm) |")
    assert "WIN-pixel" in out[2] and "190.0" in out[2]


def test_split_both_agreement(tmp_path, capsys):
    assert _script().main([*_split_argv(tmp_path), "--split", "both"]) == 0
    out = capsys.readouterr().out.split("\n\n")
    assert len(out) == 3
    assert "Val MSE" in out[0] and "Test MSE" in out[1]
    last = out[2].splitlines()
    assert last[0] == "| Variant | Verdict val | Verdict test | Agree |"
    assert last[2] == "| V | DROPPED | WIN-pixel | **no** |"


SEEDS = [170.0, 174.0, 178.0]  # mean 174, sd 4 -> bar 166


def test_pose_track_verdict():
    assert pose_track_verdict(165.9, SEEDS)
    assert not pose_track_verdict(166.5, SEEDS)
    assert not pose_track_verdict(166.0, SEEDS)  # exactly at the bar
    with pytest.raises(ValueError):
        pose_track_verdict(100.0, [170.0])


def _seed_argv(tmp_path: Path) -> list[str]:
    out = []
    for i, (mse, mp, tmp_) in enumerate([(0.0020, 170.0, 190.0), (0.0024, 174.0, 194.0),
                                         (0.0022, 178.0, 198.0)]):  # fmt: skip
        out += [
            "--seed",
            str(_run(tmp_path, f"s{i}", mse)),
            str(_probe(tmp_path, f"sp{i}", mp, tmp_)),
        ]
    return out


def test_script_seed_columns(tmp_path, capsys):
    argv = [*_split_argv(tmp_path), *_seed_argv(tmp_path)]
    assert _script().main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("| Pose track | Pixel cost |")
    assert out[2].endswith("| — | +7.3 % |")  # 174.0 not below 166; 0.00236 vs mean 0.0022


def test_script_seed_both_requires_both(tmp_path, capsys):
    argv = [*_split_argv(tmp_path), *_seed_argv(tmp_path)]
    # val probe 174.0 -> no; make a variant better on val only
    v = _probe(tmp_path, "pv", 160.0, 199.0)
    argv[argv.index("--variant") + 3] = str(v)
    assert _script().main([*argv, "--split", "both"]) == 0
    out = capsys.readouterr().out.split("\n\n")
    assert out[0].splitlines()[2].split("|")[-3].strip() == "PROMOTED (pose)"
    assert out[1].splitlines()[2].split("|")[-3].strip() == "—"
    last = out[2].splitlines()
    assert last[0].endswith("| Pose track val / test |")
    assert last[2].endswith("| — (PROMOTED (pose) / —) |")


def test_script_no_seed_unchanged(tmp_path, capsys):
    argv = [*_split_argv(tmp_path), "--split", "both"]
    assert _script().main(argv) == 0
    plain = capsys.readouterr().out
    assert "Pose track" not in plain and "Pixel cost" not in plain
    assert _script().main([*argv, "--seed", "x", "y"]) == 0  # a single seed is ignored
    assert capsys.readouterr().out == plain
