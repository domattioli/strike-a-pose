"""Tests for cli.py: sap info, exit codes, the strict check, the parser, and sap run (T037)."""

import json
import re
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from strike_a_pose import cli
from strike_a_pose.cli import build_parser, check_pinned_versions, main
from strike_a_pose.runrecord import read_run_record
from strike_a_pose.verdict import VerdictRefusedError

FULL_CONFIGURATION = Path(__file__).resolve().parent.parent / "configs" / "full.yaml"

# One valid argument list for each subcommand of contracts/cli.md.
CONTRACT_COMMANDS = [
    ["info", "--strict"],
    ["generate", "--config", "c.yaml", "--out", "o", "--resume"],
    ["train", "--config", "c.yaml", "--out", "o", "--resume", "--time-budget", "8.5h"],
    ["predict", "--config", "c.yaml", "--out", "o", "--resume"],
    ["calibrate", "--config", "c.yaml", "--out", "o"],
    ["evaluate", "--config", "c.yaml", "--out", "o"],
    ["verdict", "--config", "c.yaml", "--out", "o"],
    ["report", "--config", "c.yaml", "--out", "o"],
    ["run", "--config", "c.yaml", "--out", "o", "--time-budget", "90m", "--seed-check", "ref"],
    ["real-eval", "--config", "c.yaml", "--out", "o", "--dataset", "bodym"],
    ["verify", "--out", "o"],
]


@pytest.fixture(autouse=True)
def no_device_or_output_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variables that change a command's device and output directory."""
    monkeypatch.delenv("SAP_DEVICE", raising=False)
    monkeypatch.delenv("SAP_OUTPUT_DIR", raising=False)


def _write_json(path: Path, document: dict) -> Path:
    """Write a configuration. JSON is also YAML, so a path needs no quoting."""
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_pins(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sap_info_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "numpy: " in output
    assert "device: cpu" in output
    assert "smplx/SMPLX_NEUTRAL.npz: key body.model, missing" in output


def test_sap_info_reads_the_tiny_configuration(
    tiny_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", "--config", str(tiny_config_path)]) == 0
    assert "config hash" in capsys.readouterr().out


def test_unknown_override_key_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--set", "bogus.key=1"]) == 2
    error = capsys.readouterr().err
    assert error.startswith("sap: error: ")
    assert "bogus.key" in error


def test_unknown_key_in_a_configuration_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = _write_json(tmp_path / "config.yaml", {"seed": 1, "nonsense": True})
    assert main(["generate", "--config", str(configuration), "--out", str(tmp_path / "out")]) == 2
    assert "'nonsense'" in capsys.readouterr().err


def test_missing_asset_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["generate", "--config", str(FULL_CONFIGURATION), "--out", str(tmp_path / "out")])
    assert code == 3
    error = capsys.readouterr().err
    assert "SMPLX_NEUTRAL.npz" in error
    assert "'body.model'" in error


def test_missing_file_under_the_configured_asset_root_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    configuration = _write_json(
        tmp_path / "config.json",
        {"seed": 1, "assets": {"root": str(root)}, "body": {"model": "smplx"}},
    )
    assert main(["run", "--config", str(configuration), "--out", str(tmp_path / "out")]) == 3
    assert str(root / "smplx" / "SMPLX_NEUTRAL.npz") in capsys.readouterr().err


def test_stage_without_an_output_directory_exits_2(
    tiny_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--config", str(tiny_config_path)]) == 2
    assert "SAP_OUTPUT_DIR" in capsys.readouterr().err


def test_strict_check_passes_when_the_installed_version_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pins = _write_pins(tmp_path / "constraints.txt", f"numpy=={metadata.version('numpy')}")
    monkeypatch.setattr(cli, "CONSTRAINTS_PATH", pins)
    assert main(["info", "--strict"]) == 0
    assert "strict: 1 installed package(s) match" in capsys.readouterr().out


def test_strict_check_exits_5_when_a_version_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pins = _write_pins(tmp_path / "constraints.txt", "numpy==0.0.1")
    monkeypatch.setattr(cli, "CONSTRAINTS_PATH", pins)
    assert main(["info", "--strict"]) == 5
    error = capsys.readouterr().err
    assert error.startswith("sap: error: strict version check failed")
    assert f"numpy is {metadata.version('numpy')} but constraints.txt pins 0.0.1" in error


def test_strict_check_exits_5_when_the_constraints_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "CONSTRAINTS_PATH", tmp_path / "absent.txt")
    assert main(["info", "--strict"]) == 5
    assert "cannot read the constraints file" in capsys.readouterr().err


def test_strict_check_skips_pins_for_packages_that_are_not_installed(tmp_path: Path) -> None:
    pins = _write_pins(tmp_path / "constraints.txt", "not-an-installed-package==1.0.0")
    assert check_pinned_versions(pins) == 0


def test_local_version_label_satisfies_a_plain_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = _write_pins(tmp_path / "constraints.txt", "torch==2.14.1")
    monkeypatch.setattr(cli, "_installed_version", lambda name: "2.14.1+cpu")
    assert check_pinned_versions(pins) == 1


def test_a_line_that_is_not_an_exact_pin_is_refused(tmp_path: Path) -> None:
    pins = _write_pins(tmp_path / "constraints.txt", "numpy>=2.0")
    with pytest.raises(cli.StrictVersionError, match="not a name==version pin"):
        check_pinned_versions(pins)


def test_time_budget_is_read_in_seconds() -> None:
    hours = build_parser().parse_args(["run", "--config", "c.yaml", "--time-budget", "8.5h"])
    minutes = build_parser().parse_args(["run", "--config", "c.yaml", "--time-budget", "90m"])
    assert hours.time_budget == 8.5 * 3600
    assert minutes.time_budget == 90 * 60


def test_bad_time_budget_exits_2(
    tiny_config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["run", "--config", str(tiny_config_path), "--out", str(tmp_path)]
    assert main([*argv, "--time-budget", "soon"]) == 2
    assert "--time-budget" in capsys.readouterr().err


def test_unknown_device_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--device", "tpu"]) == 2
    assert "--device" in capsys.readouterr().err


def test_repeated_set_options_are_collected_in_order() -> None:
    sets = ["--set", "evaluate.views=[1,4]", "--set", "seed=3"]
    arguments = build_parser().parse_args(["info", *sets])
    assert arguments.overrides == ["evaluate.views=[1,4]", "seed=3"]


@pytest.mark.parametrize("argv", CONTRACT_COMMANDS, ids=lambda argv: argv[0])
def test_each_contract_subcommand_parses(argv: list[str]) -> None:
    build_parser().parse_args(argv)


def test_a_missing_subcommand_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert capsys.readouterr().err.startswith("sap: error: ")


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "COMMAND" in capsys.readouterr().out


# The stages of sap run, in order, and the directory each one writes under the output directory.
PIPELINE_STAGES = ("generate", "train", "predict", "calibrate", "evaluate", "verdict", "report")
STAGE_DIRECTORIES = ("data", "train", "predict", "calibrate", "evaluate", "verdict", "report")
# The extra overrides that tests/test_train.py and tests/test_predict.py use with small_config.
RUN_OVERRIDES: dict[str, object] = {"camera.focal_px": 48, "data.min_unflagged": 0}
VERDICT_LINE = re.compile(r"^VERDICT: (PASS|KILL) median_ratio=", re.MULTILINE)


def _set_arguments(overrides: dict[str, object]) -> list[str]:
    """Return one --set option for each override, with the value written as JSON (also YAML)."""
    arguments: list[str] = []
    for key, value in overrides.items():
        arguments += ["--set", f"{key}={json.dumps(value)}"]
    return arguments


def _run_arguments(
    config_path: Path, output: Path, small_config: dict[str, object], command: str = "run"
) -> list[str]:
    return [
        command,
        "--config",
        str(config_path),
        "--out",
        str(output),
        *_set_arguments({**small_config, **RUN_OVERRIDES}),
    ]


def test_sap_run_writes_every_stage_directory(
    tiny_config_path: Path,
    small_config: dict[str, object],
    output_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_arguments(tiny_config_path, output_dir, small_config)) == 0
    output = capsys.readouterr().out
    assert VERDICT_LINE.search(output) is not None
    for directory in STAGE_DIRECTORIES:
        assert (output_dir / directory).is_dir(), directory
    assert (output_dir / "report" / "results.md").is_file()
    assert (output_dir / "report" / "coverage_vs_views.png").is_file()
    assert (output_dir / "report" / "width_vs_views.png").is_file()
    assert (output_dir / "verdict" / "verdict.md").is_file()
    record = read_run_record(output_dir / "run_record.json")
    assert record.finished_at is not None
    assert all(record.timings[stage] is not None for stage in PIPELINE_STAGES)


def test_sap_run_resume_skips_finished_stages(
    tiny_config_path: Path,
    small_config: dict[str, object],
    output_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = _run_arguments(tiny_config_path, output_dir, small_config)
    assert main(arguments) == 0
    capsys.readouterr()
    assert main([*arguments, "--resume"]) == 0
    assert VERDICT_LINE.search(capsys.readouterr().out) is not None
    record = read_run_record(output_dir / "run_record.json")
    assert record.finished_at is not None


def test_single_stage_command_needs_the_earlier_stages(
    tiny_config_path: Path,
    small_config: dict[str, object],
    output_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_arguments(tiny_config_path, output_dir, small_config, "generate")) == 0
    capsys.readouterr()
    assert main(_run_arguments(tiny_config_path, output_dir, small_config, "evaluate")) == 4
    assert capsys.readouterr().err.startswith("sap: error: ")


def test_evaluate_without_a_run_record_exits_4(
    tiny_config_path: Path, output_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["evaluate", "--config", str(tiny_config_path), "--out", str(output_dir)]
    assert main(argv) == 4
    assert "run_record.json" in capsys.readouterr().err


def _stub_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace each stage runner by a function that records its name, and does nothing else."""
    for stage in PIPELINE_STAGES:
        monkeypatch.setitem(
            cli._STAGE_RUNNERS, stage, lambda session, name=stage: calls.append(name)
        )


def test_a_kill_verdict_exits_0_and_run_continues_to_the_report(
    tiny_config_path: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)
    verdict = SimpleNamespace(line="VERDICT: KILL median_ratio=0.900", exit_code=0)
    monkeypatch.setitem(
        cli._STAGE_RUNNERS,
        "verdict",
        lambda session: print(verdict.line) or calls.append("verdict"),
    )
    argv = ["run", "--config", str(tiny_config_path), "--out", str(output_dir)]
    assert main(argv) == 0
    assert calls == list(PIPELINE_STAGES)
    assert "VERDICT: KILL" in capsys.readouterr().out


def test_a_refused_verdict_exits_4_and_stops_run_before_the_report(
    tiny_config_path: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _stub_stages(monkeypatch, calls)
    verdict = SimpleNamespace(line="VERDICT: KILL median_ratio=nan")
    refusal = VerdictRefusedError("the width ratio of chest is not finite", verdict)

    def refuse(session: object) -> None:
        print(verdict.line)
        raise refusal

    monkeypatch.setitem(cli._STAGE_RUNNERS, "verdict", refuse)
    argv = ["run", "--config", str(tiny_config_path), "--out", str(output_dir)]
    assert main(argv) == 4
    captured = capsys.readouterr()
    assert "VERDICT: KILL median_ratio=nan" in captured.out
    assert "not finite" in captured.err
    assert "report" not in calls
    assert calls == list(PIPELINE_STAGES[:-2])


def test_a_stage_stopped_by_the_time_budget_exits_7_with_the_resume_command(
    tiny_config_path: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unfinished = SimpleNamespace(completed=False, shards_generated=1, shards_total=4)
    monkeypatch.setattr(cli, "generate_dataset", lambda *args, **kwargs: unfinished)
    argv = [
        "run",
        "--config",
        str(tiny_config_path),
        "--out",
        str(output_dir),
        "--set",
        "seed=3",
        "--time-budget",
        "90m",
    ]
    assert main(argv) == 7
    error = capsys.readouterr().err
    assert error.startswith("sap: error: time budget reached in generate after 1 of 4 shards")
    assert "resume with: sap run --config" in error
    assert "--set seed=3" in error
    assert error.rstrip().endswith("--resume")
    assert read_run_record(output_dir / "run_record.json").timings["generate"] is not None


def test_seed_check_is_not_implemented_yet(
    tiny_config_path: Path, output_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["run", "--config", str(tiny_config_path), "--out", str(output_dir)]
    assert main([*argv, "--seed-check", str(output_dir)]) == 1
    assert "--seed-check" in capsys.readouterr().err


def test_the_device_option_overrides_the_configuration(
    tiny_config_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setitem(
        cli._STAGE_RUNNERS, "generate", lambda session: seen.append(session.configuration["device"])
    )
    argv = ["generate", "--config", str(tiny_config_path), "--out", str(output_dir)]
    assert main([*argv, "--device", "auto"]) == 0
    monkeypatch.setenv("SAP_DEVICE", "auto")
    assert main(argv) == 0
    assert seen == ["auto", "auto"]


def test_info_reads_the_constraints_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pins = _write_pins(tmp_path / "bundled.txt", f"numpy=={metadata.version('numpy')}")
    assert main(["info", "--strict", "--constraints", str(pins)]) == 0
    assert "strict: 1 installed package(s) match bundled.txt" in capsys.readouterr().out
    pins = _write_pins(tmp_path / "wrong.txt", "numpy==0.0.1")
    assert main(["info", "--strict", "--constraints", str(pins)]) == 5
