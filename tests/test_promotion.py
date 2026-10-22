"""Promotion rule (spec 015 FR-006): pixel win or latent win; supersedes the 012 rule."""

import importlib.util
import json
from pathlib import Path

import pytest

from strike_a_pose.eval.promotion import Measures, verdict

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


def _run(tmp: Path, name: str, mse: float, kl: list[str] | None = None) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "metrics.json").write_text(
        json.dumps({"splits": {"val": {"image": {"ae": {"image_mse": mse}}}}})
    )
    (d / "config.yaml").write_text(f"variational: {'true' if kl else 'false'}\n")
    rows = ["epoch,train_loss,val_loss,train_kl"]
    rows += [f"{i + 1},0.1,0.1,{k}" for i, k in enumerate(kl or ["", ""])]
    (d / "history.csv").write_text("\n".join(rows) + "\n")
    return d


def _probe(tmp: Path, name: str, mp: float) -> Path:
    d = tmp / name
    d.mkdir()
    t = {"joints_world": {"mlp": {"mpjpe_mm": mp}}}
    (d / "metrics.json").write_text(json.dumps({"splits": {"val": {"targets": t}}}))
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
