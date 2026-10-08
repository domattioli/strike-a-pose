#!/usr/bin/env bash
# CPU pre-commit gate (constitution Principle VI): ruff, the pytest suite, and the tiny end-to-end run.
#
# Usage: bash scripts/cpu_smoke.sh [--dry-run]
#
# Each part runs under `timeout 600`; the script fails when any part fails or exceeds 600 s.
# --dry-run prints the parts and exits 0 without running them.
# When SAP_PYTHON is set, it names the Python of the project environment, and ruff, pytest, and
# sap are taken from the same bin directory. Otherwise python, ruff, pytest, and sap come from PATH.
set -euo pipefail

PART_TIMEOUT_SECONDS=600

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
for argument in "$@"; do
    case "$argument" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            echo "cpu_smoke.sh: unknown argument: $argument" >&2
            echo "usage: bash scripts/cpu_smoke.sh [--dry-run]" >&2
            exit 2
            ;;
    esac
done

# tool_path NAME prints the command for a tool of the project environment.
tool_path() {
    if [[ -n "${SAP_PYTHON:-}" ]]; then
        echo "$(dirname "$SAP_PYTHON")/$1"
    else
        echo "$1"
    fi
}

PYTHON_BIN="${SAP_PYTHON:-python}"
RUFF_BIN="$(tool_path ruff)"
PYTEST_BIN="$(tool_path pytest)"
SAP_BIN="$(tool_path sap)"

PART_1_NAME="part 1: ruff"
PART_1_COMMAND="timeout $PART_TIMEOUT_SECONDS $RUFF_BIN check ."
PART_2_NAME="part 2: pytest"
PART_2_COMMAND="timeout $PART_TIMEOUT_SECONDS $PYTEST_BIN --durations=10"
PART_3_NAME="part 3: tiny end-to-end run"
PART_3_COMMANDS=(
    "timeout $PART_TIMEOUT_SECONDS $SAP_BIN run --config configs/tiny.yaml --out \"\$tmp/out\" | tee \"\$tmp/stdout.txt\""
    "$PYTHON_BIN scripts/check_tiny_run.py \"\$tmp/out\""
    "grep -E '^VERDICT: ' \"\$tmp/stdout.txt\""
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "$PART_1_NAME"
    echo "    $PART_1_COMMAND"
    echo "$PART_2_NAME"
    echo "    $PART_2_COMMAND"
    echo "$PART_3_NAME (tmp=\$(mktemp -d))"
    for command_line in "${PART_3_COMMANDS[@]}"; do
        echo "    $command_line"
    done
    exit 0
fi

run_part() {
    local name="$1"
    shift
    echo "== $name: starting"
    local status=0
    "$@" || status=$?
    if [[ "$status" -eq 0 ]]; then
        echo "== $name: passed"
    elif [[ "$status" -eq 124 ]]; then
        echo "== $name: failed, exceeded $PART_TIMEOUT_SECONDS s" >&2
        exit 1
    else
        echo "== $name: failed with exit status $status" >&2
        exit 1
    fi
}

run_part "$PART_1_NAME" timeout "$PART_TIMEOUT_SECONDS" "$RUFF_BIN" check .
run_part "$PART_2_NAME" timeout "$PART_TIMEOUT_SECONDS" "$PYTEST_BIN" --durations=10

tmp="$(mktemp -d)"
cleanup() {
    rm -rf "$tmp"
}
trap cleanup EXIT

# Inside run_part, set -e is off, so every step returns its status explicitly.
run_tiny_run() {
    timeout "$PART_TIMEOUT_SECONDS" "$SAP_BIN" run --config configs/tiny.yaml --out "$tmp/out" \
        | tee "$tmp/stdout.txt" || return $?
    "$PYTHON_BIN" scripts/check_tiny_run.py "$tmp/out" || return $?
    grep -E '^VERDICT: ' "$tmp/stdout.txt" || return $?
}

run_part "$PART_3_NAME" run_tiny_run
