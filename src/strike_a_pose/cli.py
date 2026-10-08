"""The sap command: argument parsing, exit codes, configuration and device checks, and sap info.

The subcommands and options follow specs/001-kill-test-mvp/contracts/cli.md, and the exit codes are
that contract's exit-code table. sap info prints versions, the device, and asset presence, and
with --strict it compares each installed package with its pin in a constraints file (research
R11). The stage subcommands generate, train, predict, calibrate, evaluate, verdict, report, and run
load their configuration, check their device request, output directory, and licensed assets
(FR-021, exit 3), and call their stage. sap run calls the stages in order. A KILL verdict exits 0
and the run goes on to the report; a verdict that refuses its comparison (exit 4) stops the run
after verdict.json and verdict.md exist. A time budget that ends a stage early exits 7 and prints
the command that resumes the run. The real-eval command checks its dataset folder and, for SAM 2
silhouettes, the SAM 2 checkpoint, then calls the real-image stage (FR-017 to FR-020). The verify
and --seed-check commands recompute the verdict and compare a second run with a reference output
(SC-003, SC-005); both exit 6 on any difference. This module implements no published method, so
it cites no algorithm source. The pin
comparison follows the local version rule of PEP 440 (https://peps.python.org/pep-0440/).
"""

import argparse
import logging
import os
import re
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, NoReturn

from strike_a_pose.assets import MissingAssetError, asset_root, list_assets, require_asset
from strike_a_pose.calibrate import CalibrationRefusedError, calibrate_quantiles
from strike_a_pose.checkpoint import TimeBudget, parse_time_budget
from strike_a_pose.config import ConfigError, config_hash, load_config, resolve_config
from strike_a_pose.data.generate import GenerationRefusedError, generate_dataset
from strike_a_pose.device import (
    DEVICE_CHOICES,
    DeviceChoice,
    requested_device_from_environment,
    select_device,
)
from strike_a_pose.evaluate import run_evaluate
from strike_a_pose.predict import PredictionRefusedError, run_predict
from strike_a_pose.real.evaluate_real import run_real_eval
from strike_a_pose.report.plots import write_plots
from strike_a_pose.report.tables import write_results_markdown
from strike_a_pose.runrecord import (
    RunRecord,
    current_code_version,
    current_library_versions,
    read_run_record,
    start_run_record,
    write_run_record,
)
from strike_a_pose.train import train_model
from strike_a_pose.verdict import (
    ReproductionDifferenceError,
    VerdictRefusedError,
    check_second_run,
    run_verdict,
    verify_output,
)

__all__ = [
    "CONSTRAINTS_FILE_NAME",
    "CONSTRAINTS_PATH",
    "EXIT_CONFIGURATION",
    "EXIT_FAILURE",
    "EXIT_MISSING_ASSET",
    "EXIT_OK",
    "EXIT_STAGE_REFUSED",
    "EXIT_STRICT_VERSION",
    "EXIT_TIME_BUDGET",
    "EXIT_VERIFY_DIFFERENCE",
    "LOG_LEVELS",
    "OUTPUT_DIRECTORY_VARIABLE",
    "SUBCOMMANDS",
    "CommandNotImplementedError",
    "StrictVersionError",
    "TimeBudgetReachedError",
    "UsageError",
    "build_parser",
    "check_pinned_versions",
    "main",
]

# The exit codes of contracts/cli.md. A usage error that argparse finds exits with the configuration
# code 2, as argparse itself does.
EXIT_OK = 0
EXIT_CONFIGURATION = 2
EXIT_MISSING_ASSET = 3
EXIT_STAGE_REFUSED = 4
EXIT_STRICT_VERSION = 5
EXIT_VERIFY_DIFFERENCE = 6
EXIT_TIME_BUDGET = 7
# Not a code of contracts/cli.md. A subcommand whose stage is not implemented yet exits with it.
EXIT_FAILURE = 1

SUBCOMMANDS = (
    "info",
    "generate",
    "train",
    "predict",
    "calibrate",
    "evaluate",
    "verdict",
    "report",
    "run",
    "real-eval",
    "verify",
)
# The subcommands that run a stage from a configuration file and write under an output directory.
_STAGE_SUBCOMMANDS = (
    "generate",
    "train",
    "predict",
    "calibrate",
    "evaluate",
    "verdict",
    "report",
    "run",
    "real-eval",
)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
OUTPUT_DIRECTORY_VARIABLE = "SAP_OUTPUT_DIR"
CONSTRAINTS_FILE_NAME = "constraints.txt"
# The pins file at the repository root. The editable install in the development container reads it
# from there. A wheel installed elsewhere has no such file beside it, so --strict exits 5 there.
CONSTRAINTS_PATH = Path(__file__).resolve().parents[2] / CONSTRAINTS_FILE_NAME
# One pin of constraints.txt: a distribution name, two equals signs, and one exact version.
_PIN_PATTERN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9._+!-]+)")
# config.py requires a seed in every configuration. Without a configuration file, sap info validates
# its --set overrides and reads the defaults with this placeholder. Nothing that sap info prints
# depends on the seed, and sap info does not print a configuration hash without a file.
_PLACEHOLDER_SEED = 0


class UsageError(Exception):
    """A command line that argparse cannot check alone, such as a run with no output directory."""


class CommandNotImplementedError(Exception):
    """A subcommand is registered, but its stage is not implemented in this build."""


class StrictVersionError(Exception):
    """The strict check of sap info failed: a version differs, or the pins cannot be read."""


class TimeBudgetReachedError(Exception):
    """The time budget ended a stage early: the run is partial and resumable (exit 7)."""


class _SapParser(argparse.ArgumentParser):
    """An argument parser that reports each usage error on one line that starts with sap: error:."""

    def error(self, message: str) -> NoReturn:
        self.exit(EXIT_CONFIGURATION, f"sap: error: {message}\n")


def _time_budget(text: str) -> float:
    """Read a --time-budget value such as 8.5h or 90m, and return it in seconds."""
    try:
        return parse_time_budget(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add the options every subcommand accepts: the device and the log level."""
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=None,
        help="compute device; without it, SAP_DEVICE, then the configuration key device",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="INFO",
        help="log level of the strike_a_pose loggers (default INFO)",
    )


def _add_configuration(parser: argparse.ArgumentParser, *, required: bool) -> None:
    """Add the configuration file and the repeatable --set overrides of contracts/config.md."""
    parser.add_argument(
        "--config",
        type=Path,
        required=required,
        metavar="C",
        help="configuration file (contracts/config.md)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override one configuration key with a YAML value, such as evaluate.views=[1,4]",
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    """Add the output directory; SAP_OUTPUT_DIR supplies it when --out is absent."""
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="O",
        help=f"output directory (default: the {OUTPUT_DIRECTORY_VARIABLE} variable)",
    )


def _add_time_budget(parser: argparse.ArgumentParser) -> None:
    """Add --time-budget, after which the run stops cleanly at the next checkpoint."""
    parser.add_argument(
        "--time-budget",
        type=_time_budget,
        default=None,
        metavar="T",
        help="budget as <hours>h or <minutes>m, such as 8.5h; the run stops at a checkpoint",
    )


def _add_resume(parser: argparse.ArgumentParser) -> None:
    """Add --resume: a stage whose DONE.json matches the current run is skipped."""
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip each stage whose DONE.json matches the current configuration and inputs",
    )


def _add_stage_options(parser: argparse.ArgumentParser) -> None:
    """Add the options of a stage subcommand: the common options, configuration, and output."""
    _add_common_options(parser)
    _add_configuration(parser, required=True)
    _add_output(parser)
    _add_time_budget(parser)


def build_parser() -> argparse.ArgumentParser:
    """Return the sap parser, with one subparser for each subcommand of contracts/cli.md."""
    parser = _SapParser(
        prog="sap",
        description="Kill-test MVP for calibrated multi-view body-measurement uncertainty.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    info = commands.add_parser("info", help="print versions, the device, and asset presence")
    _add_common_options(info)
    _add_configuration(info, required=False)
    info.add_argument(
        "--strict",
        action="store_true",
        help="exit 5 when an installed version differs from the constraints file",
    )
    info.add_argument(
        "--constraints",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"constraints file for --strict (default: {CONSTRAINTS_FILE_NAME} at the root)",
    )

    generate = commands.add_parser("generate", help="generate the synthetic bodies")
    _add_stage_options(generate)
    _add_resume(generate)

    train = commands.add_parser("train", help="train the multi-view model")
    _add_stage_options(train)
    _add_resume(train)

    predict = commands.add_parser("predict", help="predict medians and spreads for each cell")
    _add_stage_options(predict)
    _add_resume(predict)

    calibrate = commands.add_parser(
        "calibrate", help="calibrate intervals on the calibration split"
    )
    _add_stage_options(calibrate)

    evaluate = commands.add_parser("evaluate", help="evaluate coverage and width on the test split")
    _add_stage_options(evaluate)

    verdict = commands.add_parser(
        "verdict", help="apply the FR-014 kill rule and print the verdict"
    )
    _add_stage_options(verdict)

    report = commands.add_parser("report", help="write the results table and the plots")
    _add_stage_options(report)

    run = commands.add_parser("run", help="run every stage in order and print the verdict line")
    _add_stage_options(run)
    _add_resume(run)
    run.add_argument(
        "--seed-check",
        type=Path,
        metavar="O_REF",
        help="make this run the second run, and compare it with the reference output O_REF",
    )

    real_eval = commands.add_parser("real-eval", help="evaluate on a real dataset (reported only)")
    _add_stage_options(real_eval)
    real_eval.add_argument(
        "--dataset",
        required=True,
        choices=("bodym", "ssp3d"),
        help="the real dataset to evaluate",
    )
    real_eval.add_argument(
        "--mask-source",
        choices=("provided", "sam2"),
        default=None,
        help="silhouette source; without it, the configuration key real.ssp3d.mask_source",
    )

    verify = commands.add_parser(
        "verify", help="recompute evaluate and verdict from predict outputs"
    )
    _add_common_options(verify)
    _add_output(verify)
    return parser


def _load_configuration(arguments: argparse.Namespace) -> dict[str, Any]:
    """Load the configuration file with its --set overrides, or the defaults without a file."""
    overrides = arguments.overrides or []
    if arguments.config is None:
        return resolve_config({"seed": _PLACEHOLDER_SEED}, overrides)
    return load_config(arguments.config, overrides)


def _resolve_device(
    arguments: argparse.Namespace, configuration: Mapping[str, Any]
) -> tuple[DeviceChoice, str]:
    """Return the run's device and how it was requested: --device, SAP_DEVICE, or the config."""
    if arguments.device is not None:
        requested, origin = arguments.device, "--device"
    else:
        from_environment = requested_device_from_environment()
        if from_environment is not None:
            requested, origin = from_environment, "SAP_DEVICE"
        else:
            requested, origin = configuration["device"], "configuration key device"
    return select_device(requested), f"requested {requested} from {origin}"


def _output_directory(arguments: argparse.Namespace) -> Path:
    """Return --out, else the SAP_OUTPUT_DIR variable. Without either, the command is refused."""
    if arguments.out is not None:
        return arguments.out
    from_environment = os.environ.get(OUTPUT_DIRECTORY_VARIABLE, "")
    if from_environment:
        return Path(from_environment)
    raise UsageError(f"no output directory; give --out or set {OUTPUT_DIRECTORY_VARIABLE}")


def _required_assets(command: str, configuration: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return (asset path, configuration key) for each licensed asset that the command reads.

    generate and run read the body model when body.model is smplx, and the pose folders when
    pose.source is amass (contracts/cli.md). The other stages read only files in the output
    directory. A path that starts with <root> lies under the asset root (assets.py).
    """
    if command not in ("generate", "run"):
        return []
    required: list[tuple[str, str]] = []
    if configuration["body"]["model"] == "smplx":
        required.append(("<root>/smplx/SMPLX_NEUTRAL.npz", "body.model"))
    if configuration["pose"]["source"] == "amass":
        subsets = configuration["pose"]["amass_subsets"]
        if subsets:
            required.extend((f"<root>/amass/{subset}", "pose.source") for subset in subsets)
        else:
            required.append(("<root>/amass", "pose.source"))
    return required


def _real_eval_assets(
    configuration: Mapping[str, Any], dataset: str, mask_source: str | None
) -> list[tuple[str, str]]:
    """Return (asset path, configuration key) for each licensed asset that real-eval reads.

    real-eval reads the folder of its dataset, which defaults to <root>/bodym or <root>/ssp3d as in
    assets.py. It reads the SAM 2 checkpoint only for SSP-3D silhouettes from SAM 2: the value of
    --mask-source, else the configuration key real.ssp3d.mask_source (contracts/cli.md). BodyM
    always uses its provided silhouettes.
    """
    real = configuration["real"]
    required = [(real[dataset]["path"] or f"<root>/{dataset}", f"real.{dataset}.path")]
    source = mask_source if mask_source is not None else real["ssp3d"]["mask_source"]
    if dataset == "ssp3d" and source == "sam2":
        checkpoint = real["sam2"]["checkpoint"] or "<root>/sam2/sam2.1_hiera_base_plus.pt"
        required.append((checkpoint, "real.sam2.checkpoint"))
    return required


def _installed_version(name: str) -> str | None:
    """Return the installed version of a distribution, or None when it is not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _same_version(installed: str, pinned: str) -> bool:
    """Return True when an installed version satisfies an exact pin.

    Under PEP 440, a pin without a local label matches any local label, so torch 2.14.1+cpu
    satisfies the pin torch==2.14.1. A pin that names a local label must match it exactly.
    """
    if "+" in pinned:
        return installed == pinned
    return installed.split("+", 1)[0] == pinned


def _strict_failure(detail: str) -> StrictVersionError:
    """Return the error of a failed strict check, with the detail that names what failed."""
    return StrictVersionError(f"strict version check failed: {detail}")


def _read_pins(path: Path) -> dict[str, str]:
    """Read the exact pins of a constraints file: name, then the version after two equals signs."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        reason = getattr(error, "strerror", None) or error
        raise _strict_failure(f"cannot read the constraints file '{path}': {reason}") from error
    pins: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PIN_PATTERN.fullmatch(line)
        if match is None:
            raise _strict_failure(
                f"line {number} of '{path}' is not a name==version pin: {raw_line.strip()!r}"
            )
        pins[match.group(1)] = match.group(2)
    return pins


def check_pinned_versions(constraints: Path) -> int:
    """Compare each installed package with its pin in a constraints file, and count the comparisons.

    A pin for a package that is not installed is skipped, because only installed versions can
    differ. Raise StrictVersionError (exit 5) that names each installed version which differs from
    its pin, or the file itself when it cannot be read or holds a line that is not a pin.
    """
    compared = 0
    differences: list[str] = []
    for name, pinned in _read_pins(constraints).items():
        installed = _installed_version(name)
        if installed is None:
            continue
        compared += 1
        if not _same_version(installed, pinned):
            differences.append(f"{name} is {installed} but {CONSTRAINTS_FILE_NAME} pins {pinned}")
    if differences:
        raise _strict_failure("; ".join(differences))
    return compared


def _run_info(arguments: argparse.Namespace) -> int:
    """Print the code version, the library versions, the device, and each asset's presence."""
    configuration = _load_configuration(arguments)
    choice, request = _resolve_device(arguments, configuration)
    print(f"code version: {current_code_version()}")
    if arguments.config is None:
        print("configuration: defaults (no configuration file)")
    else:
        print(f"configuration: {arguments.config} (config hash {config_hash(configuration)[:12]})")
    print("versions:")
    for name, version in current_library_versions().items():
        print(f"  {name}: {version or 'not installed'}")
    print(
        f"device: {choice.device} ({request}; hardware class {choice.hardware_class}; "
        f"device name {choice.device_name})"
    )
    print("assets:")
    for status in list_assets(configuration):
        state = "present" if status.present else "missing"
        where = "no asset root is set" if status.location is None else str(status.location)
        print(f"  {status.name}: key {status.key}, {state} ({where})")
    if arguments.strict:
        constraints = CONSTRAINTS_PATH if arguments.constraints is None else arguments.constraints
        compared = check_pinned_versions(constraints)
        print(
            f"strict: {compared} installed package(s) match {constraints.name}; "
            "packages that are not installed are not compared"
        )
    return EXIT_OK


RUN_RECORD_NAME = "run_record.json"
# The stage subcommands that call a stage, in run order. real-eval runs on its own, so it is not
# among them: sap run never evaluates a real dataset.
_PIPELINE_STAGES = (
    "generate",
    "train",
    "predict",
    "calibrate",
    "evaluate",
    "verdict",
    "report",
)


@dataclass
class _Session:
    """What every stage of one command shares: configuration, output, record, and time budget."""

    arguments: argparse.Namespace
    configuration: dict[str, Any]
    choice: DeviceChoice
    out: Path
    record: RunRecord
    budget: TimeBudget | None
    resume: bool


def _prepare_stage_command(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], DeviceChoice, Path]:
    """Check a stage command's inputs, and return its configuration, device, and output directory.

    The device that --device, SAP_DEVICE, or the configuration requests is written into the
    configuration under the key device, because each stage selects its device from that key.
    """
    out = _output_directory(arguments)
    configuration = _load_configuration(arguments)
    choice, _ = _resolve_device(arguments, configuration)
    if choice.requested != configuration["device"]:
        configuration = resolve_config({**configuration, "device": choice.requested})
    root = asset_root(configuration)
    if arguments.command == "real-eval":
        required = _real_eval_assets(configuration, arguments.dataset, arguments.mask_source)
    else:
        required = _required_assets(arguments.command, configuration)
    for path, key in required:
        require_asset(path, key, root)
    return configuration, choice, out


def _open_run_record(
    configuration: Mapping[str, Any],
    choice: DeviceChoice,
    out: Path,
    *,
    fresh: bool,
    creatable: bool,
) -> RunRecord:
    """Return the run record of the output directory, creating and writing it when allowed.

    A fresh record replaces any file. Otherwise an existing file is read back, so that stage
    timings add up across commands (runrecord.py). A missing file is created when creatable, and a
    missing file raises RunRecordError when not (evaluate, verdict, and report read the file
    themselves and need the run that wrote it). A record of another configuration is not replaced
    here: the stage that reads it refuses it and names both hashes.
    """
    path = out / RUN_RECORD_NAME
    if not fresh and path.is_file():
        return read_run_record(path)
    if not fresh and not creatable:
        return read_run_record(path)  # raises RunRecordError that names the missing file
    record = start_run_record(
        configuration, hardware_class=choice.hardware_class, device_name=choice.device_name
    )
    write_run_record(record, path)
    return record


def _stage_generate(session: _Session) -> None:
    """Generate the bodies, and stop with exit 7 when the time budget ends the stage early."""
    result = generate_dataset(
        session.configuration,
        session.out,
        resume=session.resume,
        time_budget=session.budget,
        run_record=session.record,
    )
    if not result.completed:
        raise TimeBudgetReachedError(
            f"time budget reached in generate after {result.shards_generated} of "
            f"{result.shards_total} shards"
        )


def _stage_train(session: _Session) -> None:
    """Train the model, and stop with exit 7 when the time budget ends the stage early."""
    result = train_model(
        session.configuration,
        session.out,
        resume=session.resume,
        time_budget=session.budget,
        run_record=session.record,
    )
    if not result.completed:
        raise TimeBudgetReachedError(
            f"time budget reached in train after {result.steps_done} of {result.total_steps} steps"
        )


def _stage_predict(session: _Session) -> None:
    """Predict every cell, and stop with exit 7 when the time budget ends the stage early."""
    result = run_predict(
        session.configuration,
        session.out,
        resume=session.resume,
        time_budget=session.budget,
        run_record=session.record,
    )
    if not result.completed:
        raise TimeBudgetReachedError(
            f"time budget reached in predict after {len(result.cells_written)} new cell file(s)"
        )


def _stage_calibrate(session: _Session) -> None:
    """Calibrate the intervals on the calibration split."""
    calibrate_quantiles(session.configuration, session.out, run_record=session.record)


def _stage_evaluate(session: _Session) -> None:
    """Evaluate coverage and width on the test split."""
    run_evaluate(session.configuration, session.out)


def _stage_verdict(session: _Session) -> None:
    """Apply the FR-014 rule and print the verdict line.

    A KILL verdict returns normally, so the run goes on to the report. A refused comparison prints
    its verdict line first, and then raises the error for exit 4 after the verdict files exist.
    """
    try:
        verdict = run_verdict(session.configuration, session.out)
    except VerdictRefusedError as error:
        print(error.verdict.line)
        raise
    print(verdict.line)


def _stage_report(session: _Session) -> None:
    """Write the two plots and the results table under report/."""
    write_plots(session.out / "evaluate" / "results.csv", session.out / "report")
    write_results_markdown(session.out)


def _stage_real_eval(session: _Session) -> None:
    """Evaluate the trained model on the real dataset of --dataset and write real/<dataset>/.

    The results are reported only and never enter the kill verdict (FR-020). The run record is
    read, not created: the earlier stages must have written it.
    """
    result = run_real_eval(
        session.configuration,
        session.out,
        session.arguments.dataset,
        mask_source=session.arguments.mask_source,
    )
    print(f"real-eval {result.dataset} ({result.mask_source} masks): wrote {result.results_path}")


_STAGE_RUNNERS: dict[str, Callable[[_Session], None]] = {
    "generate": _stage_generate,
    "train": _stage_train,
    "predict": _stage_predict,
    "calibrate": _stage_calibrate,
    "evaluate": _stage_evaluate,
    "verdict": _stage_verdict,
    "report": _stage_report,
    "real_eval": _stage_real_eval,
}


def _resume_command(arguments: argparse.Namespace, command: str) -> str:
    """Return the command line that continues a run that the time budget stopped."""
    words = ["sap", command, "--config", str(arguments.config), "--out", str(arguments.out)]
    for override in arguments.overrides or []:
        words += ["--set", override]
    if arguments.device is not None:
        words += ["--device", arguments.device]
    words.append("--resume")
    return " ".join(shlex.quote(word) for word in words)


def _execute_stages(session: _Session, stages: Sequence[str]) -> None:
    """Run the stages in order. Each stage is timed, and the run record is written after it."""
    record_path = session.out / RUN_RECORD_NAME
    for stage in stages:
        print(f"stage {stage}: started")
        stopped: TimeBudgetReachedError | None = None
        with session.record.time_stage(stage):
            try:
                _STAGE_RUNNERS[stage](session)
            except TimeBudgetReachedError as error:
                stopped = error  # the stage is partial, but its time still counts
        write_run_record(session.record, record_path)
        if stopped is not None:
            resume = _resume_command(session.arguments, session.arguments.command)
            raise TimeBudgetReachedError(f"{stopped}; resume with: {resume}") from stopped
        print(f"stage {stage}: done")
    if stages[-1] == "report":
        session.record.finish()
        write_run_record(session.record, record_path)


def _run_stage(arguments: argparse.Namespace) -> int:
    """Run one pipeline stage, or every stage in order for sap run."""
    command = arguments.command
    if command == "real-eval":
        configuration, choice, out = _prepare_stage_command(arguments)
        record = _open_run_record(configuration, choice, out, fresh=False, creatable=False)
        session = _Session(arguments, configuration, choice, out, record, None, resume=False)
        _execute_stages(session, ("real_eval",))
        return EXIT_OK
    seed_check = arguments.seed_check if command == "run" else None
    if seed_check is not None:
        _check_seed_reference(seed_check, _output_directory(arguments))
    configuration, choice, out = _prepare_stage_command(arguments)
    is_first = command in ("generate", "run")
    resume = bool(getattr(arguments, "resume", False))
    record = _open_run_record(
        configuration,
        choice,
        out,
        fresh=is_first and not resume,
        creatable=command not in ("evaluate", "verdict", "report"),
    )
    if is_first and resume and record.config_hash != config_hash(configuration):
        # The configuration changed since the earlier session: its stages are stale, so start over.
        record = _open_run_record(configuration, choice, out, fresh=True, creatable=True)
    budget = None if arguments.time_budget is None else TimeBudget(arguments.time_budget)
    session = _Session(arguments, configuration, choice, out, record, budget, resume)
    stages = _PIPELINE_STAGES if command == "run" else (command,)
    _execute_stages(session, stages)
    if seed_check is not None:
        check_second_run(out, seed_check)
        print(f"seed check: '{out}' matches the reference '{seed_check}' (SC-005)")
    return EXIT_OK


def _check_seed_reference(reference: Path, out: Path) -> None:
    """Refuse a --seed-check reference that is the run's own output, or is not a directory.

    Both checks run before any stage, so that a refused command never writes into the reference.
    """
    if reference.resolve() == out.resolve():
        raise UsageError("--seed-check names the output of this run; give the reference output")
    if not reference.is_dir():
        raise UsageError(f"--seed-check reference '{reference}' is not a directory")


def _run_verify(arguments: argparse.Namespace) -> int:
    """Recompute the verdict of --out, compare it with the stored verdict files, and print it."""
    verdict = verify_output(_output_directory(arguments))
    print(verdict.line)
    return EXIT_OK


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "info": _run_info,
    "verify": _run_verify,
    **dict.fromkeys(_STAGE_SUBCOMMANDS, _run_stage),
}


def _refusal_code(error: Exception) -> int:
    """Return the exit code of a stage that refuses its inputs: the error's own code, else 4."""
    code = getattr(error, "exit_code", EXIT_STAGE_REFUSED)
    return code if isinstance(code, int) else EXIT_STAGE_REFUSED


def _fail(status: int, error: Exception) -> int:
    """Print the error on stderr as one line that starts with sap: error:, and return its status."""
    print(f"sap: error: {' '.join(str(error).split())}", file=sys.stderr)
    return status


def _exit_status(code: object) -> int:
    """Return the status of a SystemExit that argparse raised: None or 0 for --help, 2 for usage."""
    if code is None:
        return EXIT_OK
    return code if isinstance(code, int) else EXIT_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    """Run one sap subcommand on argv (sys.argv[1:] when None) and return its exit code.

    The console script passes the result to sys.exit. Each failure prints one line on stderr that
    starts with sap: error:, and returns the exit code of contracts/cli.md for that kind of failure.
    """
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as stop:  # argparse exits for --help, and for every usage error
        return _exit_status(stop.code)
    logging.basicConfig(format="sap: %(levelname)s %(name)s: %(message)s")
    logging.getLogger("strike_a_pose").setLevel(arguments.log_level)
    try:
        return _HANDLERS[arguments.command](arguments)
    except (ConfigError, UsageError) as error:
        return _fail(EXIT_CONFIGURATION, error)
    except MissingAssetError as error:
        return _fail(EXIT_MISSING_ASSET, error)
    except StrictVersionError as error:
        return _fail(EXIT_STRICT_VERSION, error)
    except TimeBudgetReachedError as error:
        return _fail(EXIT_TIME_BUDGET, error)
    except ReproductionDifferenceError as error:
        return _fail(EXIT_VERIFY_DIFFERENCE, error)
    except CommandNotImplementedError as error:
        return _fail(EXIT_FAILURE, error)
    except (
        CalibrationRefusedError,
        GenerationRefusedError,
        PredictionRefusedError,
        ValueError,
        FileNotFoundError,
    ) as error:
        # Stage refusals (GenerationRefusedError, VerdictError, EvaluateError, ReportError,
        # AmassDataError, RunRecordError, and the other errors that a stage raises for bad inputs).
        return _fail(_refusal_code(error), error)
