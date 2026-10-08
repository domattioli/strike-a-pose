"""Checks of scripts/cpu_smoke.sh: bash syntax, dry-run part lists, and the seed-check part."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "cpu_smoke.sh"


def _run_bash(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the smoke script under bash with the given arguments and capture its output."""
    return subprocess.run(
        ["bash", str(SMOKE_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _part_header_lines(output: str) -> list[str]:
    """Return the lines that name a part (they start with 'part ' and are not indented)."""
    return [line for line in output.splitlines() if line.startswith("part ")]


def test_smoke_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dry_run_lists_three_parts_by_default() -> None:
    result = _run_bash("--dry-run")
    assert result.returncode == 0, result.stderr
    headers = _part_header_lines(result.stdout)
    assert len(headers) == 3
    assert headers[0].startswith("part 1: ruff")
    assert headers[1].startswith("part 2: pytest")
    assert headers[2].startswith("part 3: tiny end-to-end run")
    assert "seed" not in result.stdout


def test_dry_run_with_seed_check_lists_four_parts() -> None:
    result = _run_bash("--dry-run", "--seed-check")
    assert result.returncode == 0, result.stderr
    headers = _part_header_lines(result.stdout)
    assert len(headers) == 4
    assert headers[3].startswith("part 4: seed check")


def test_seed_check_part_runs_second_tiny_run_against_part_three_output() -> None:
    result = _run_bash("--dry-run", "--seed-check")
    assert result.returncode == 0, result.stderr
    seed_command = [
        line.strip() for line in result.stdout.splitlines() if "--seed-check" in line
    ]
    assert len(seed_command) == 1
    command = seed_command[0]
    assert "timeout 600 " in command
    assert "sap run --config configs/tiny.yaml" in command
    assert '--out "$tmp/out2"' in command
    assert '--seed-check "$tmp/out"' in command


def test_seed_check_flag_is_not_passed_without_the_flag() -> None:
    result = _run_bash("--dry-run")
    assert "--seed-check" not in result.stdout


def test_unknown_argument_exits_with_usage_error() -> None:
    result = _run_bash("--no-such-flag")
    assert result.returncode == 2
    assert "usage: bash scripts/cpu_smoke.sh [--dry-run] [--seed-check]" in result.stderr
