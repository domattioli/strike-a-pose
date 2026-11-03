"""Licensed-asset checks. The tool never downloads SMPL or AMASS."""

import hashlib
from pathlib import Path

DOC = "docs/data-assets.md"


class MissingAssetError(Exception):
    """A required licensed asset (SMPL model / AMASS data) is missing or unusable."""

    def __init__(self, path: Path | str, what: str, hint: str = "") -> None:
        msg = f"Missing {what}: {path}. See {DOC} for how to obtain it and where to put it."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)


def require_file(path: Path, what: str) -> Path:
    if not Path(path).is_file():
        raise MissingAssetError(path, what)
    return Path(path)


def require_dir(path: Path, what: str) -> Path:
    if not Path(path).is_dir():
        raise MissingAssetError(path, what)
    return Path(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
