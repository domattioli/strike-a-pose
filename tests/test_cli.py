from pathlib import Path

import yaml

from strike_a_pose.datagen.cli import main


def _write(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    return str(p)


def test_missing_assets_writes_nothing(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    out = tmp_path / "out"
    p.write_text(yaml.safe_dump({
        "dataset_name": "x", "amass_subsets": ["A"], "output_root": str(out),
        "smpl_model_path": str(tmp_path / "missing.npz"), "amass_root": str(tmp_path),
    }))  # fmt: skip
    assert main(["generate", "--config", str(p), "--seed", "0"]) == 3
    assert "docs/data-assets.md" in capsys.readouterr().err
    assert not out.exists()


def test_generate_exists_validate(small_cfg, tmp_path):
    cfg = _write(tmp_path, small_cfg)
    assert main(["generate", "--config", cfg, "--seed", "1"]) == 0
    assert main(["generate", "--config", cfg, "--seed", "1"]) == 4
    assert main(["generate", "--config", cfg, "--seed", "1", "--overwrite"]) == 0
    ds = Path(small_cfg.output_root) / small_cfg.dataset_name
    assert main(["validate", str(ds)]) == 0
    assert main(["validate", str(tmp_path / "nothing")]) == 1


def test_config_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset_name: x\npose_source: amass\n")  # no amass_subsets
    assert main(["generate", "--config", str(p), "--seed", "0"]) == 2
    assert main(["generate", "--config", str(tmp_path / "nope.yaml"), "--seed", "0"]) == 2


FIXTURE = Path(__file__).parent / "fixtures" / "smpl_fixture"
FIXTURE_SPLIT = Path(__file__).parents[1] / "splits" / "smpl_fixture.json"
FIXTURE_SPLIT_ARGS = ["--seed", "0", "--fractions", "0.5", "0", "0.5"]


def test_split_frozen_and_matches_committed(tmp_path, capsys):
    out = tmp_path / "s.json"
    args = ["split", str(FIXTURE), *FIXTURE_SPLIT_ARGS, "--out", str(out)]
    assert main(args) == 0
    assert out.read_bytes() == FIXTURE_SPLIT.read_bytes()  # SC-002
    assert main(args) == 4  # frozen
    assert "--overwrite" in capsys.readouterr().err
    assert main([*args, "--overwrite"]) == 0


def test_split_error_exits(tmp_path):
    out = tmp_path / "s.json"
    # default fractions: 2 poses cannot give val, test and train >= 1 each
    assert main(["split", str(FIXTURE), "--seed", "0", "--out", str(out)]) == 2
    bad = ["--fractions", "0.5", "0.5", "0.5"]
    assert main(["split", str(FIXTURE), "--seed", "0", *bad, "--out", str(out)]) == 2
    assert main(["split", str(tmp_path / "nothing"), "--seed", "0", "--out", str(out)]) == 1
    assert not out.exists()
