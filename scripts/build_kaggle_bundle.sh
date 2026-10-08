#!/usr/bin/env bash
# Build the Kaggle upload bundle: the wheel, constraints.txt, and configs/ in dist/kaggle-bundle/ (quickstart section 4 step 1).
# The wheel is built offline with setuptools' build_meta (setuptools 70.1 or later, no separate wheel package).
# Set SAP_PYTHON to the interpreter to use; it defaults to python3.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SAP_PYTHON:-python3}"
bundle="$repo/dist/kaggle-bundle"
staging="$repo/dist/kaggle-wheel-staging"

rm -rf "$bundle" "$staging"
mkdir -p "$bundle" "$staging"

cd "$repo"
"$python_bin" - "$staging" <<'PYEOF'
import sys
from setuptools.build_meta import build_wheel
print("built", build_wheel(sys.argv[1]))
PYEOF

wheels=("$staging"/strike_a_pose-*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
    echo "build_kaggle_bundle.sh: expected exactly one strike_a_pose wheel, found ${#wheels[@]}" >&2
    exit 1
fi
cp "${wheels[0]}" "$bundle/"
rm -rf "$staging"

cp "$repo/constraints.txt" "$bundle/constraints.txt"
mkdir -p "$bundle/configs"
cp "$repo"/configs/*.yaml "$bundle/configs/"

echo "bundle written to $bundle:"
find "$bundle" -type f | sort
