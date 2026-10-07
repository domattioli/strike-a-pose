"""The sap command: argument parsing, exit codes, configuration and device checks, and sap info.

The subcommands and options follow specs/001-kill-test-mvp/contracts/cli.md, and the exit codes are
that contract's exit-code table. This skeleton implements sap info in full, including --strict,
which compares each installed package with its pin in constraints.txt (research R11). Every other
subcommand loads its configuration, checks its device request and output directory, and checks the
licensed assets its stage reads (FR-021, exit 3). Then it stops with a message that the command is
not implemented yet, and a later task replaces that stop with the stage itself. This module
implements no published method, so it cites no algorithm source. The pin comparison follows the
local version rule of PEP 440 (https://peps.python.org/pep-0440/).
"""

import argparse
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, NoReturn

from strike_a_pose.assets import MissingAssetError, asset_root, list_assets, require_asset
from strike_a_pose.checkpoint import parse_time_budget
from strike_a_pose.config import ConfigError, config_hash, load_config, resolve_config
from strike_a_pose.device import (
    DEVICE_CHOICES,
    DeviceChoice,
    requested_device_from_environment,
    select_device,
)
from strike_a_pose.runrecord import current_code_version, current_library_versions

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
        help="exit 5 when an installed version differs from constraints.txt",
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
        compared = check_pinned_versions(CONSTRAINTS_PATH)
        print(
            f"strict: {compared} installed package(s) match {CONSTRAINTS_FILE_NAME}; "
            "packages that are not installed are not compared"
        )
    return EXIT_OK


def _run_stage(arguments: argparse.Namespace) -> int:
    """Check a stage subcommand's inputs, then stop: its stage is not implemented in this build."""
    _output_directory(arguments)
    configuration = _load_configuration(arguments)
    _resolve_device(arguments, configuration)
    root = asset_root(configuration)
    for path, key in _required_assets(arguments.command, configuration):
        require_asset(path, key, root)
    raise CommandNotImplementedError(f"the '{arguments.command}' command is not implemented yet")


def _run_verify(arguments: argparse.Namespace) -> int:
    """Check the output directory of verify, then stop: its recomputation is not implemented yet."""
    _output_directory(arguments)
    raise CommandNotImplementedError("the 'verify' command is not implemented yet")


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "info": _run_info,
    "verify": _run_verify,
    **dict.fromkeys(_STAGE_SUBCOMMANDS, _run_stage),
}


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
    except CommandNotImplementedError as error:
        return _fail(EXIT_FAILURE, error)
