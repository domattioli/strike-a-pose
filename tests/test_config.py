"""Smoke tests for config.py: defaults, validation, --set overrides, canonical dump, and hash."""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from strike_a_pose.config import (
    FIXED_KEYS,
    ConfigError,
    canonical_dump,
    config_hash,
    load_config,
    parse_override,
    resolve_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIMIT_GROUPS = (
    "spine",
    "neck",
    "head",
    "collar",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
)

# Every key of contracts/config.md, as a dotted path. The shipped files must name all of them.
CONTRACT_KEYS = frozenset(
    {
        "seed",
        "device",
        "assets.root",
        "body.model",
        "body.n_betas",
        "body.beta_clip",
        "pose.source",
        "pose.amass_subsets",
        *(f"pose.limits_deg.{group}" for group in LIMIT_GROUPS),
        "pose.capsule_overlap_cm",
        "pose.max_rejections",
        "camera.n_cameras",
        "camera.image_size",
        "camera.focal_px",
        "camera.distance_m",
        "camera.height_m",
        "camera.min_separation_deg",
        "camera.lookat_jitter_m",
        "data.n_train",
        "data.n_cal",
        "data.n_test",
        "data.min_unflagged",
        "data.shard_size",
        "measure.step_cm",
        "model.latent_dim",
        "model.channels",
        "model.camera_embed_dim",
        "train.epochs",
        "train.batch_size",
        "train.lr",
        "train.kl_weight",
        "train.views_train",
        "train.noise_train_deg",
        "train.checkpoint_every",
        "predict.n_samples",
        "predict.measure_mode",
        "calibrate.alpha",
        "calibrate.min_cal",
        "calibrate.spread_floor_cm",
        "evaluate.views",
        "evaluate.noise_deg",
        "evaluate.band",
        "evaluate.measurements",
        "verdict.threshold",
        "verdict.measurements",
        "verdict.noise_deg",
        "verdict.compare_views",
        "real.bodym.path",
        "real.bodym.side_azimuth_deg",
        "real.ssp3d.path",
        "real.ssp3d.mask_source",
        "real.sam2.checkpoint",
        "real.nominal_camera.distance_m",
        "real.nominal_camera.height_m",
        "real.nominal_camera.lookat_height_m",
        "real.mask_min_area_fraction",
        "real.mask_min_component_fraction",
    }
)


def _write(directory: Path, text: str, name: str = "config.yaml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _nested(path: tuple[str, ...], value: object) -> dict[str, Any]:
    """Return a document holding the seed and one key set to the value, at the given path."""
    document: dict[str, Any] = {"seed": 1}
    node = document
    for name in path[:-1]:
        node = node.setdefault(name, {})
    node[path[-1]] = value
    return document


def _leaf_keys(mapping: Mapping[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name, value in mapping.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            keys |= _leaf_keys(value, path)
        else:
            keys.add(path)
    return keys


def test_a_seed_alone_resolves_to_the_cpu_smoke_defaults():
    config = resolve_config({"seed": 7})
    assert config["seed"] == 7
    assert config["device"] == "cpu"
    assert config["body"] == {"model": "standin", "n_betas": 10, "beta_clip": 3.0}
    assert config["pose"]["source"] == "limits"
    assert config["camera"]["image_size"] == 64
    assert config["evaluate"]["views"] == [1, 2, 4]
    assert config["calibrate"]["min_cal"] == 32


def test_defaults_equal_the_shipped_tiny_configuration():
    assert resolve_config({"seed": 1}) == load_config(REPO_ROOT / "configs" / "tiny.yaml")


def test_resolved_configuration_holds_every_key_of_the_contract():
    assert _leaf_keys(resolve_config({"seed": 1})) == CONTRACT_KEYS


@pytest.mark.parametrize("name", ["tiny.yaml", "full.yaml"])
def test_shipped_configuration_files_write_every_key_and_load(name):
    path = REPO_ROOT / "configs" / name
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _leaf_keys(written) == CONTRACT_KEYS
    load_config(path)


def test_full_configuration_keeps_the_asset_placeholders_as_written():
    config = load_config(REPO_ROOT / "configs" / "full.yaml")
    assert config["real"]["bodym"]["path"] == "<root>/bodym"
    assert config["real"]["ssp3d"]["path"] == "<root>/ssp3d"
    assert config["real"]["sam2"]["checkpoint"] == "<root>/sam2/sam2.1_hiera_base_plus.pt"


def test_fixed_keys_are_the_fr014_constants_and_the_measurement_order():
    assert FIXED_KEYS == {
        "evaluate.band": [0.87, 0.93],
        "evaluate.measurements": ["height", "chest", "waist", "hip", "thigh"],
        "verdict.threshold": 0.7,
        "verdict.measurements": ["chest", "waist", "hip", "thigh"],
        "verdict.noise_deg": 0.0,
        "verdict.compare_views": [1, 4],
    }


def test_missing_seed_is_refused_by_name():
    with pytest.raises(ConfigError, match="'seed'") as info:
        resolve_config({})
    assert info.value.key == "seed"


def test_unknown_keys_are_refused_and_named_with_their_path():
    with pytest.raises(ConfigError, match="'sed'") as info:
        resolve_config({"seed": 1, "sed": 2})
    assert info.value.key == "sed"
    with pytest.raises(ConfigError, match=r"camera\.image_sise") as info:
        resolve_config({"seed": 1, "camera": {"image_sise": 32}})
    assert info.value.key == "camera.image_sise"
    with pytest.raises(ConfigError, match=r"pose\.limits_deg\.tail"):
        resolve_config({"seed": 1, "pose": {"limits_deg": {"tail": 10}}})


def test_a_section_given_as_a_scalar_is_refused():
    with pytest.raises(ConfigError, match="'camera'"):
        resolve_config({"seed": 1, "camera": 5})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), "1"),
        (("seed",), 1.5),
        (("seed",), True),
        (("device",), "gpu"),
        (("camera", "image_size"), "64"),
        (("camera", "image_size"), 64.5),
        (("camera", "focal_px"), True),
        (("camera", "distance_m"), [2.5]),
        (("body", "beta_clip"), "3.0"),
        (("assets", "root"), 5),
        (("pose", "amass_subsets"), "ACCAD"),
        (("pose", "limits_deg"), ["spine"]),
        (("evaluate", "views"), 4),
        (("evaluate", "measurements"), ["height", "chest", "waist", "hip", "thigh", "x"]),
        (("train", "lr"), "fast"),
        (("predict", "measure_mode"), "linearized2"),
    ],
)
def test_wrong_types_and_choices_are_refused_by_key(path, value):
    with pytest.raises(ConfigError, match=re.escape(f"'{'.'.join(path)}'")) as info:
        resolve_config(_nested(path, value))
    assert info.value.key == ".".join(path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), -1),
        (("calibrate", "alpha"), 0),
        (("calibrate", "alpha"), 1.0),
        (("camera", "n_cameras"), 0),
        (("camera", "distance_m"), [4.0, 2.5]),
        (("camera", "focal_px"), float("nan")),
        (("camera", "focal_px"), 10**400),
        (("train", "noise_train_deg"), [-1.0, 5.0]),
        (("train", "views_train"), []),
        (("model", "channels"), [8, 0]),
        (("evaluate", "noise_deg"), [-2.0]),
        (("measure", "step_cm"), 0),
        (("body", "beta_clip"), float("inf")),
        (("pose", "limits_deg", "spine"), 0),
        (("real", "mask_min_area_fraction"), 1.5),
    ],
)
def test_values_outside_their_rule_are_refused_by_key(path, value):
    with pytest.raises(ConfigError) as info:
        resolve_config(_nested(path, value))
    assert info.value.key == ".".join(path)


def test_view_counts_may_not_exceed_the_cameras_in_a_rig():
    with pytest.raises(ConfigError, match="evaluate.views") as info:
        resolve_config({"seed": 1, "camera": {"n_cameras": 2}, "train": {"views_train": [1, 2]}})
    assert info.value.key == "evaluate.views"
    assert resolve_config({"seed": 1})["evaluate"]["views"] == [1, 2, 4]


def test_fr014_keys_accept_their_fixed_values_and_refuse_any_other():
    fixed_values = {
        "seed": 1,
        "verdict": {"threshold": 0.7, "noise_deg": 0},
        "evaluate": {"band": [0.87, 0.93]},
    }
    resolve_config(fixed_values)
    with pytest.raises(ConfigError, match="FR-014") as info:
        resolve_config({"seed": 1, "verdict": {"threshold": 0.8}})
    assert info.value.key == "verdict.threshold"
    with pytest.raises(ConfigError, match=r"evaluate\.band"):
        resolve_config({"seed": 1, "evaluate": {"band": [0.85, 0.95]}})
    with pytest.raises(ConfigError, match=r"verdict\.compare_views"):
        resolve_config({"seed": 1, "verdict": {"compare_views": [1, 2]}})


def test_evaluate_measurements_is_fixed_in_order():
    with pytest.raises(ConfigError, match="fixed in order"):
        resolve_config({"seed": 1, "evaluate": {"measurements": ["chest", "height"]}})
    with pytest.raises(ConfigError, match="fixed in order"):
        resolve_config({"seed": 1}, ["evaluate.measurements=[chest,height,waist,hip,thigh]"])


@pytest.mark.parametrize(
    "text",
    [
        "verdict.threshold=0.7",
        "verdict.threshold=0.8",
        "verdict.noise_deg=0",
        "verdict.compare_views=[1,4]",
        "verdict.measurements=[chest,waist,hip,thigh]",
        "evaluate.band=[0.87,0.93]",
        "evaluate.band=[0.85,0.95]",
    ],
)
def test_fr014_keys_refuse_every_override_even_with_the_fixed_value(text):
    with pytest.raises(ConfigError, match="FR-014") as info:
        resolve_config({"seed": 1}, [text])
    assert info.value.key == text.partition("=")[0]


def test_set_override_changes_the_value_and_the_hash():
    base = resolve_config({"seed": 1})
    changed = resolve_config({"seed": 1}, ["camera.image_size=32"])
    assert changed["camera"]["image_size"] == 32
    assert config_hash(changed) != config_hash(base)


def test_set_override_value_is_parsed_as_yaml():
    assert resolve_config({"seed": 1}, ["evaluate.views=[1,4]"])["evaluate"]["views"] == [1, 4]
    cleared = resolve_config({"seed": 1, "assets": {"root": "/x"}}, ["assets.root=null"])
    assert cleared["assets"]["root"] is None
    assert parse_override("assets.root=/data/a=b") == ("assets.root", "/data/a=b")


def test_set_override_of_one_joint_group_keeps_the_other_groups():
    config = resolve_config({"seed": 1}, ["pose.limits_deg.spine=30"])
    assert config["pose"]["limits_deg"]["spine"] == 30.0
    assert config["pose"]["limits_deg"]["neck"] == 50.0
    partial = resolve_config({"seed": 1, "pose": {"limits_deg": {"spine": 30}}})
    assert partial["pose"]["limits_deg"]["neck"] == 50.0


@pytest.mark.parametrize(
    "text",
    [
        "camera.image_sise=32",
        "camera=5",
        "camera.image_size",
        "=5",
        "camera.image_size.x=3",
        "evaluate.views=[1,",
    ],
)
def test_malformed_or_unknown_overrides_are_refused(text):
    with pytest.raises(ConfigError):
        resolve_config({"seed": 1}, [text])


def test_the_same_key_may_not_be_overridden_twice():
    with pytest.raises(ConfigError, match=r"camera\.image_size") as info:
        resolve_config({"seed": 1}, ["camera.image_size=32", "camera.image_size=64"])
    assert info.value.key == "camera.image_size"


def test_integer_and_decimal_spellings_of_one_value_give_one_hash(tmp_path):
    as_integer = resolve_config({"seed": 1}, ["camera.focal_px=32"])
    as_decimal = resolve_config({"seed": 1}, ["camera.focal_px=32.0"])
    assert as_integer["camera"]["focal_px"] == 32.0
    assert isinstance(as_integer["camera"]["focal_px"], float)
    assert config_hash(as_integer) == config_hash(as_decimal)
    as_list = resolve_config({"seed": 1, "evaluate": {"noise_deg": [0]}})
    as_floats = resolve_config({"seed": 1, "evaluate": {"noise_deg": [0.0]}})
    assert config_hash(as_list) == config_hash(as_floats)
    integer_file = load_config(_write(tmp_path, "seed: 1\ncamera:\n  focal_px: 32\n", "i.yaml"))
    decimal_file = load_config(_write(tmp_path, "seed: 1\ncamera:\n  focal_px: 32.0\n", "d.yaml"))
    assert config_hash(integer_file) == config_hash(decimal_file) == config_hash(as_integer)


def test_exponent_notation_reads_as_a_float_and_hashes_like_the_decimal(tmp_path):
    exponent = load_config(_write(tmp_path, "seed: 1\ntrain:\n  lr: 1e-3\n", "e.yaml"))
    decimal = load_config(_write(tmp_path, "seed: 1\ntrain:\n  lr: 0.001\n", "f.yaml"))
    assert isinstance(exponent["train"]["lr"], float)
    assert exponent["train"]["lr"] == 0.001
    assert config_hash(exponent) == config_hash(decimal)
    assert resolve_config({"seed": 1}, ["train.lr=1e-3"])["train"]["lr"] == 0.001


def test_quoted_exponent_text_is_refused_for_a_float_key(tmp_path):
    with pytest.raises(ConfigError, match="train.lr") as info:
        load_config(_write(tmp_path, 'seed: 1\ntrain:\n  lr: "1e-3"\n'))
    assert info.value.key == "train.lr"


def test_placeholder_paths_are_kept_as_written(tmp_path):
    config = load_config(_write(tmp_path, "seed: 1\nreal:\n  bodym:\n    path: <root>/bodym\n"))
    assert config["real"]["bodym"]["path"] == "<root>/bodym"


def test_file_layout_and_key_order_do_not_change_the_hash(tmp_path):
    block = _write(tmp_path, "seed: 4  # a comment\ncamera:\n  image_size: 32\n  focal_px: 32\n")
    flow = _write(tmp_path, "camera: {focal_px: 32.0, image_size: 32}\nseed: 4\n", "flow.yaml")
    assert canonical_dump(load_config(block)) == canonical_dump(load_config(flow))
    assert config_hash(load_config(block)) == config_hash(load_config(flow))


def test_canonical_dump_is_sorted_and_reads_back_to_the_same_configuration():
    config = resolve_config({"seed": 3, "pose": {"amass_subsets": ["1e3", "ACCAD"]}})
    text = canonical_dump(config)
    assert text.splitlines()[0] == "assets:"
    assert "beta_clip: 3.0" in text
    assert "'1e3'" in text
    assert yaml.safe_load(text) == config
    assert config_hash(yaml.safe_load(text)) == config_hash(config)


def test_config_hash_is_the_sha256_of_the_canonical_dump():
    config = resolve_config({"seed": 1})
    digest = config_hash(config)
    assert digest == hashlib.sha256(canonical_dump(config).encode("utf-8")).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_hash_is_stable_for_one_input_and_changes_with_the_seed():
    assert config_hash(resolve_config({"seed": 1})) == config_hash(resolve_config({"seed": 1}))
    assert config_hash(resolve_config({"seed": 1})) != config_hash(resolve_config({"seed": 2}))


def test_load_config_matches_resolve_config_with_overrides(tmp_path):
    text = "seed: 9\ncamera:\n  image_size: 48\n"
    loaded = load_config(_write(tmp_path, text), ["camera.focal_px=48"])
    expected = resolve_config(yaml.safe_load(text), ["camera.focal_px=48"])
    assert loaded == expected
    assert config_hash(loaded) == config_hash(expected)


def test_defaults_are_copied_so_one_call_cannot_change_the_next():
    first = resolve_config({"seed": 1})
    first["pose"]["amass_subsets"].append("ACCAD")
    first["pose"]["limits_deg"]["spine"] = 1.0
    second = resolve_config({"seed": 1})
    assert second["pose"]["amass_subsets"] == []
    assert second["pose"]["limits_deg"]["spine"] == 35.0


def test_empty_or_non_mapping_documents_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="'seed'"):
        load_config(_write(tmp_path, ""))
    with pytest.raises(ConfigError, match="mapping of keys"):
        load_config(_write(tmp_path, "- 1\n- 2\n", "list.yaml"))


def test_unreadable_or_invalid_files_are_configuration_errors(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.yaml")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(_write(tmp_path, "seed: [1,\n", "broken.yaml"))
