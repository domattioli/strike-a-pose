"""Licensed-asset root, asset paths, missing-asset errors, and the asset listing of sap info.

The asset root is the configuration key assets.root, or the environment variable SAP_ASSET_ROOT when
that key is null (contracts/cli.md). Configured asset paths may start with the placeholder <root>.
A missing asset stops the command with an error that names the file and the configuration key that
needs it (FR-021, exit 3). Nothing in this module downloads an asset (constitution Principle II).
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ASSET_ROOT_ENV",
    "ASSET_ROOT_KEY",
    "ROOT_PLACEHOLDER",
    "AssetStatus",
    "MissingAssetError",
    "asset_root",
    "list_assets",
    "require_asset",
]

ASSET_ROOT_ENV = "SAP_ASSET_ROOT"
ASSET_ROOT_KEY = "assets.root"
ROOT_PLACEHOLDER = "<root>"


class MissingAssetError(FileNotFoundError):
    """A licensed asset is absent. The message names the asset and the configuration key."""

    def __init__(self, asset: str | Path, key: str, *, root_missing: bool = False) -> None:
        self.asset = str(asset)
        self.key = key
        if root_missing:
            reason = f"no asset root is set; set {ASSET_ROOT_KEY} or {ASSET_ROOT_ENV}"
        else:
            reason = (
                "the path is absent, and the package never downloads assets; place the asset "
                f"under the asset root ({ASSET_ROOT_KEY} or {ASSET_ROOT_ENV})"
            )
        super().__init__(f"missing asset '{self.asset}' for configuration key '{key}': {reason}")


@dataclass(frozen=True)
class AssetStatus:
    """One row of the asset listing of sap info."""

    name: str  # path under the asset root, for example "smplx/SMPLX_NEUTRAL.npz"
    key: str  # configuration key that names the asset, for example "body.model"
    location: Path | None  # where the asset is looked up; None when no asset root is set
    present: bool  # True when that location exists


def asset_root(
    config: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Return assets.root from the configuration, else SAP_ASSET_ROOT, else None.

    The configuration wins when it sets a root. An empty value counts as not set.
    """
    configured = _value_at(config, ASSET_ROOT_KEY)
    if isinstance(configured, str) and configured:
        return Path(configured)
    variables = os.environ if environment is None else environment
    from_environment = variables.get(ASSET_ROOT_ENV, "")
    return Path(from_environment) if from_environment else None


def require_asset(value: str, key: str, root: Path | None) -> Path:
    """Return the asset that a configured path names. Raise MissingAssetError when it is absent.

    A value that starts with <root> lies under the asset root. Any other value is a plain path,
    absolute or relative to the current directory. The error names the resolved path and the key.
    """
    if not value:
        raise MissingAssetError(value, key)
    location = _expand(value, root)
    if location is None:
        raise MissingAssetError(value, key, root_missing=True)
    if not location.exists():
        raise MissingAssetError(location, key)
    return location


def list_assets(
    config: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[AssetStatus]:
    """Return the asset root and each licensed asset that sap reads, with its presence.

    The configuration sets the names: one folder per amass subset, and an explicit location
    when real.bodym.path, real.ssp3d.path, or real.sam2.checkpoint holds one. Presence means
    that the location exists. No asset is opened, and nothing is downloaded.
    """
    root = asset_root(config, environment)
    listing = [
        AssetStatus(ROOT_PLACEHOLDER, ASSET_ROOT_KEY, root, root is not None and root.is_dir())
    ]
    for name, key, path_key in _catalogue(config):
        location = _location(name, path_key, config, root)
        present = location is not None and location.exists()
        listing.append(AssetStatus(name, key, location, present))
    return listing


def _catalogue(config: Mapping[str, Any] | None) -> list[tuple[str, str, str | None]]:
    """Return (name, key, path key) for each asset; the path key is None when no path applies.

    The SMPL files serve SSP-3D ground truth only (research R5), so the SSP-3D key names them.
    """
    subsets = _value_at(config, "pose.amass_subsets")
    amass_names = [f"amass/{subset}" for subset in subsets] if subsets else ["amass"]
    return [
        ("smplx/SMPLX_NEUTRAL.npz", "body.model", None),
        ("smpl/SMPL_MALE.pkl", "real.ssp3d.path", None),
        ("smpl/SMPL_FEMALE.pkl", "real.ssp3d.path", None),
        ("smpl/SMPL_NEUTRAL.pkl", "real.ssp3d.path", None),
        *((name, "pose.source", None) for name in amass_names),
        ("bodym", "real.bodym.path", "real.bodym.path"),
        ("ssp3d", "real.ssp3d.path", "real.ssp3d.path"),
        ("sam2/sam2.1_hiera_base_plus.pt", "real.sam2.checkpoint", "real.sam2.checkpoint"),
    ]


def _location(
    name: str,
    path_key: str | None,
    config: Mapping[str, Any] | None,
    root: Path | None,
) -> Path | None:
    """Return an asset's explicit configured path, else its name under the root."""
    configured = _value_at(config, path_key) if path_key is not None else None
    if isinstance(configured, str) and configured:
        return _expand(configured, root)
    return None if root is None else root / name


def _expand(value: str, root: Path | None) -> Path | None:
    """Return the path a configured value names; None when it needs an unset asset root."""
    if value == ROOT_PLACEHOLDER or value.startswith(f"{ROOT_PLACEHOLDER}/"):
        if root is None:
            return None
        return root / value[len(ROOT_PLACEHOLDER) :].lstrip("/")
    return Path(value)


def _value_at(config: Mapping[str, Any] | None, dotted_key: str) -> Any:
    """Return the value at a dotted configuration key, or None when the configuration lacks it."""
    node: Any = config
    for name in dotted_key.split("."):
        if not isinstance(node, Mapping) or name not in node:
            return None
        node = node[name]
    return node
