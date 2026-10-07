"""Smoke tests for assets.py: root precedence, missing-asset errors, listing, and no download."""

from pathlib import Path
from typing import Any

import pytest

from strike_a_pose import assets
from strike_a_pose.assets import (
    ASSET_ROOT_ENV,
    AssetStatus,
    MissingAssetError,
    asset_root,
    list_assets,
    require_asset,
)
from strike_a_pose.config import load_config, resolve_config


def _config(document: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a resolved configuration. config.py requires a seed, so every test supplies one."""
    return resolve_config({"seed": 1, **(document or {})})


def _by_name(statuses: list[AssetStatus]) -> dict[str, AssetStatus]:
    return {status.name: status for status in statuses}


def test_configured_root_wins_over_the_environment(tmp_path: Path) -> None:
    config = _config({"assets": {"root": str(tmp_path / "configured")}})
    environment = {ASSET_ROOT_ENV: str(tmp_path / "environment")}
    assert asset_root(config, environment) == tmp_path / "configured"


def test_environment_root_is_the_default_when_the_configured_root_is_null(tmp_path: Path) -> None:
    assert asset_root(_config(), {ASSET_ROOT_ENV: str(tmp_path)}) == tmp_path


def test_no_root_when_neither_source_sets_one() -> None:
    assert asset_root(_config(), {}) is None
    assert asset_root(_config(), {ASSET_ROOT_ENV: ""}) is None


def test_without_arguments_the_process_environment_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path))
    assert asset_root() == tmp_path


def test_missing_asset_error_names_the_file_and_the_key(tmp_path: Path) -> None:
    with pytest.raises(MissingAssetError) as caught:
        require_asset("<root>/smplx/SMPLX_NEUTRAL.npz", "body.model", tmp_path)
    expected = tmp_path / "smplx" / "SMPLX_NEUTRAL.npz"
    assert caught.value.asset == str(expected)
    assert caught.value.key == "body.model"
    assert str(expected) in str(caught.value)
    assert "'body.model'" in str(caught.value)
    assert isinstance(caught.value, FileNotFoundError)


def test_missing_root_error_names_the_placeholder_path_and_the_key() -> None:
    with pytest.raises(MissingAssetError) as caught:
        require_asset("<root>/bodym", "real.bodym.path", None)
    message = str(caught.value)
    assert "'<root>/bodym'" in message
    assert "'real.bodym.path'" in message
    assert ASSET_ROOT_ENV in message
    assert "assets.root" in message


def test_present_assets_under_the_root_are_returned(tmp_path: Path) -> None:
    (tmp_path / "smplx").mkdir()
    model = tmp_path / "smplx" / "SMPLX_NEUTRAL.npz"
    model.write_bytes(b"")
    (tmp_path / "bodym").mkdir()
    assert require_asset("<root>/smplx/SMPLX_NEUTRAL.npz", "body.model", tmp_path) == model
    assert require_asset("<root>/bodym", "real.bodym.path", tmp_path) == tmp_path / "bodym"


def test_plain_path_needs_no_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"")
    assert require_asset(str(checkpoint), "real.sam2.checkpoint", None) == checkpoint
    with pytest.raises(MissingAssetError, match="real.sam2.checkpoint"):
        require_asset(str(tmp_path / "absent.pt"), "real.sam2.checkpoint", None)


def test_listing_reports_presence_for_each_key(tmp_path: Path) -> None:
    (tmp_path / "smplx").mkdir()
    (tmp_path / "smplx" / "SMPLX_NEUTRAL.npz").write_bytes(b"")
    (tmp_path / "bodym").mkdir()
    statuses = _by_name(list_assets(_config({"assets": {"root": str(tmp_path)}}), {}))
    assert statuses["<root>"] == AssetStatus("<root>", "assets.root", tmp_path, True)
    assert statuses["smplx/SMPLX_NEUTRAL.npz"].present
    assert statuses["smplx/SMPLX_NEUTRAL.npz"].key == "body.model"
    assert statuses["bodym"].present
    assert statuses["bodym"].key == "real.bodym.path"
    assert not statuses["ssp3d"].present
    assert statuses["ssp3d"].location == tmp_path / "ssp3d"
    assert not statuses["sam2/sam2.1_hiera_base_plus.pt"].present


def test_listing_without_a_root_marks_every_asset_absent() -> None:
    statuses = list_assets(_config(), {})
    assert statuses[0] == AssetStatus("<root>", "assets.root", None, False)
    assert all(not status.present and status.location is None for status in statuses)


def test_tiny_configuration_lists_no_present_asset(tiny_config_path: Path) -> None:
    statuses = list_assets(load_config(tiny_config_path), {})
    assert not any(status.present for status in statuses)


def test_explicit_bodym_path_replaces_the_root_for_that_asset(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    config = _config(
        {"assets": {"root": str(tmp_path / "root")}, "real": {"bodym": {"path": str(elsewhere)}}}
    )
    status = _by_name(list_assets(config, {}))["bodym"]
    assert status == AssetStatus("bodym", "real.bodym.path", elsewhere, True)


def test_placeholder_bodym_path_resolves_under_the_root(tmp_path: Path) -> None:
    config = _config(
        {"assets": {"root": str(tmp_path)}, "real": {"bodym": {"path": "<root>/bodym"}}}
    )
    assert _by_name(list_assets(config, {}))["bodym"].location == tmp_path / "bodym"


def test_amass_subsets_are_listed_as_folders(tmp_path: Path) -> None:
    root = {"assets": {"root": str(tmp_path)}}
    with_subsets = _config({**root, "pose": {"amass_subsets": ["ACCAD", "CMU"]}})
    amass = [status for status in list_assets(with_subsets, {}) if status.key == "pose.source"]
    assert [status.name for status in amass] == ["amass/ACCAD", "amass/CMU"]
    assert amass[0].location == tmp_path / "amass" / "ACCAD"
    without_subsets = _config(root)
    folders = [status for status in list_assets(without_subsets, {}) if status.key == "pose.source"]
    assert [status.name for status in folders] == ["amass"]


def test_the_module_has_no_download_path() -> None:
    names = [name.lower() for name in vars(assets)]
    assert not [name for name in names if "download" in name or "fetch" in name]
    for network_module in ("urllib", "http", "requests", "socket", "ftplib"):
        assert network_module not in vars(assets)
