"""Configuration for sap runs: YAML load, defaults, validation, canonical dump, SHA-256 hash.

The keys and rules follow specs/001-kill-test-mvp/contracts/config.md. The configuration hash is the
SHA-256 of the canonical YAML dump (research R9), so every table and run record names the exact
configuration that produced it. Defaults are the CPU smoke values of the tiny column, so a file that
leaves a key out never needs a licensed asset or a full-size run.
"""

import copy
import hashlib
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "FIXED_KEYS",
    "ConfigError",
    "Setting",
    "canonical_dump",
    "config_hash",
    "load_config",
    "parse_override",
    "resolve_config",
]


class ConfigError(ValueError):
    """A configuration error, which the CLI reports with exit code 2. The message names the key."""

    def __init__(self, message: str, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key


# Marks a setting that the specification does not fix.
_NOT_FIXED = object()


@dataclass(frozen=True)
class Setting:
    """One configuration key: its kind, its default, and the rule its value must meet.

    Kinds: int, float, path (text or null), enum, text_list, int_list, float_list, float_pair (a
    [min, max] pair), and float_map (joint group to number; a group left out keeps its default).
    A fixed setting holds a constant that the specification fixes, and its value must equal it.
    """

    kind: str
    default: Any = None
    rule: str = ""
    check: Callable[[Any], bool] | None = None
    choices: tuple[str, ...] = ()
    min_length: int = 0
    required: bool = False
    fixed: Any = _NOT_FIXED
    fixed_reason: str = ""


def _at_least(bound: float) -> Callable[[Any], bool]:
    return lambda value: value >= bound


def _above(bound: float) -> Callable[[Any], bool]:
    return lambda value: value > bound


def _is_open_unit(value: Any) -> bool:
    return 0 < value < 1


def _is_fraction(value: Any) -> bool:
    return 0 <= value <= 1


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _leaf_settings(schema: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Setting]]:
    """Yield each setting of a schema together with its dotted key path."""
    for name, entry in schema.items():
        path = _join(prefix, name)
        if isinstance(entry, Setting):
            yield path, entry
        else:
            yield from _leaf_settings(entry, path)


_LIMITS_DEG = {
    "spine": 35.0,
    "neck": 50.0,
    "head": 50.0,
    "collar": 20.0,
    "shoulder": 150.0,
    "elbow": 150.0,
    "wrist": 60.0,
    "hip": 120.0,
    "knee": 150.0,
    "ankle": 45.0,
}
_MEASUREMENTS = ["height", "chest", "waist", "hip", "thigh"]
_FR014 = "fixed by FR-014"

_SCHEMA: dict[str, Any] = {
    "seed": Setting("int", required=True, check=_at_least(0), rule="a non-negative integer"),
    "device": Setting(
        "enum", "cpu", choices=("auto", "cpu", "cuda"), rule="one of auto, cpu, cuda"
    ),
    "assets": {
        "root": Setting("path", None, rule="a path or null"),
    },
    "body": {
        "model": Setting(
            "enum", "standin", choices=("standin", "smplx"), rule="one of standin, smplx"
        ),
        "n_betas": Setting("int", 10, check=_at_least(1), rule="an integer of at least 1"),
        "beta_clip": Setting("float", 3.0, check=_above(0), rule="a positive number"),
    },
    "pose": {
        "source": Setting(
            "enum", "limits", choices=("limits", "amass"), rule="one of limits, amass"
        ),
        "amass_subsets": Setting("text_list", [], rule="a list of folder names"),
        "limits_deg": Setting(
            "float_map",
            _LIMITS_DEG,
            check=_above(0),
            rule="a map from joint group to a positive number of degrees",
        ),
        "capsule_overlap_cm": Setting(
            "float", 1.0, check=_at_least(0), rule="a number of at least 0"
        ),
        "max_rejections": Setting("int", 100, check=_at_least(0), rule="an integer of at least 0"),
    },
    "camera": {
        "n_cameras": Setting("int", 4, check=_at_least(1), rule="an integer of at least 1"),
        "image_size": Setting("int", 64, check=_at_least(1), rule="an integer of at least 1"),
        "focal_px": Setting("float", 64.0, check=_above(0), rule="a positive number"),
        "distance_m": Setting(
            "float_pair",
            [2.5, 4.0],
            check=_above(0),
            rule="a pair [min, max] of positive numbers, with min not above max",
        ),
        "height_m": Setting(
            "float_pair",
            [0.8, 1.8],
            check=_at_least(0),
            rule="a pair [min, max] of numbers of at least 0, with min not above max",
        ),
        "min_separation_deg": Setting(
            "float", 20.0, check=_at_least(0), rule="a number of at least 0"
        ),
        "lookat_jitter_m": Setting(
            "float", 0.05, check=_at_least(0), rule="a number of at least 0"
        ),
    },
    "data": {
        "n_train": Setting("int", 256, check=_at_least(1), rule="an integer of at least 1"),
        "n_cal": Setting("int", 64, check=_at_least(1), rule="an integer of at least 1"),
        "n_test": Setting("int", 64, check=_at_least(1), rule="an integer of at least 1"),
        "min_unflagged": Setting("int", 32, check=_at_least(0), rule="an integer of at least 0"),
        "shard_size": Setting("int", 64, check=_at_least(1), rule="an integer of at least 1"),
    },
    "measure": {
        "step_cm": Setting("float", 0.5, check=_above(0), rule="a positive number"),
    },
    "model": {
        "latent_dim": Setting("int", 8, check=_at_least(1), rule="an integer of at least 1"),
        "channels": Setting(
            "int_list",
            [8, 16, 32, 32, 32],
            check=_at_least(1),
            min_length=1,
            rule="a non-empty list of integers, each at least 1",
        ),
        "camera_embed_dim": Setting("int", 16, check=_at_least(1), rule="an integer of at least 1"),
    },
    "train": {
        "epochs": Setting("int", 2, check=_at_least(1), rule="an integer of at least 1"),
        "batch_size": Setting("int", 32, check=_at_least(1), rule="an integer of at least 1"),
        "lr": Setting("float", 1e-3, check=_above(0), rule="a positive number"),
        "kl_weight": Setting("float", 1e-3, check=_at_least(0), rule="a number of at least 0"),
        "views_train": Setting(
            "int_list",
            [1, 2, 3, 4],
            check=_at_least(1),
            min_length=1,
            rule="a non-empty list of integers, each from 1 to camera.n_cameras",
        ),
        "noise_train_deg": Setting(
            "float_pair",
            [0.0, 5.0],
            check=_at_least(0),
            rule="a pair [min, max] of numbers of at least 0, with min not above max",
        ),
        "checkpoint_every": Setting("int", 10, check=_at_least(1), rule="an integer of at least 1"),
    },
    "predict": {
        "n_samples": Setting("int", 8, check=_at_least(1), rule="an integer of at least 1"),
        "measure_mode": Setting(
            "enum", "exact", choices=("exact", "linearized"), rule="one of exact, linearized"
        ),
    },
    "calibrate": {
        "alpha": Setting("float", 0.10, check=_is_open_unit, rule="a number above 0 and below 1"),
        "min_cal": Setting("int", 32, check=_at_least(1), rule="an integer of at least 1"),
        "spread_floor_cm": Setting("float", 0.1, check=_at_least(0), rule="a number of at least 0"),
    },
    "evaluate": {
        "views": Setting(
            "int_list",
            [1, 2, 4],
            check=_at_least(1),
            min_length=1,
            rule="a non-empty list of integers, each from 1 to camera.n_cameras",
        ),
        "noise_deg": Setting(
            "float_list",
            [0.0, 2.0, 5.0],
            check=_at_least(0),
            min_length=1,
            rule="a non-empty list of numbers of at least 0",
        ),
        "band": Setting(
            "float_pair",
            [0.87, 0.93],
            rule="the pair [0.87, 0.93]",
            fixed=[0.87, 0.93],
            fixed_reason=_FR014,
        ),
        "measurements": Setting(
            "text_list",
            list(_MEASUREMENTS),
            rule="the list height, chest, waist, hip, thigh",
            fixed=list(_MEASUREMENTS),
            fixed_reason="fixed in order",
        ),
    },
    "verdict": {
        "threshold": Setting("float", 0.7, rule="a number", fixed=0.7, fixed_reason=_FR014),
        "measurements": Setting(
            "text_list",
            ["chest", "waist", "hip", "thigh"],
            rule="the list chest, waist, hip, thigh",
            fixed=["chest", "waist", "hip", "thigh"],
            fixed_reason=_FR014,
        ),
        "noise_deg": Setting("float", 0.0, rule="a number", fixed=0.0, fixed_reason=_FR014),
        "compare_views": Setting(
            "int_list",
            [1, 4],
            rule="the list 1, 4",
            fixed=[1, 4],
            fixed_reason=_FR014,
        ),
    },
    "real": {
        "bodym": {
            "path": Setting("path", None, rule="a path or null"),
            "side_azimuth_deg": Setting("float", 90.0, rule="a number"),
        },
        "ssp3d": {
            "path": Setting("path", None, rule="a path or null"),
            "mask_source": Setting(
                "enum", "provided", choices=("provided", "sam2"), rule="one of provided, sam2"
            ),
        },
        "sam2": {
            "checkpoint": Setting("path", None, rule="a path or null"),
        },
        "nominal_camera": {
            "distance_m": Setting("float", 3.0, check=_above(0), rule="a positive number"),
            "height_m": Setting("float", 1.2, check=_at_least(0), rule="a number of at least 0"),
            "lookat_height_m": Setting(
                "float", 0.9, check=_at_least(0), rule="a number of at least 0"
            ),
        },
        "mask_min_area_fraction": Setting(
            "float", 0.02, check=_is_fraction, rule="a number from 0 to 1"
        ),
        "mask_min_component_fraction": Setting(
            "float", 0.90, check=_is_fraction, rule="a number from 0 to 1"
        ),
    },
}

# The keys whose values the specification fixes. Any --set override is refused, and a file value
# must equal the fixed value. verdict.py holds the verdict constants and must agree with these.
FIXED_KEYS: dict[str, Any] = {
    path: copy.deepcopy(setting.fixed)
    for path, setting in _leaf_settings(_SCHEMA)
    if setting.fixed is not _NOT_FIXED
}

# YAML 1.2 float notation such as 1e-3. PyYAML reads it as text unless a decimal point is present,
# so the loader and the dumper both add this pattern, which keeps 1e-3 and 0.001 the same number.
_EXPONENT_FLOAT = re.compile(r"^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?$")
_FLOAT_TAG = "tag:yaml.org,2002:float"
_FLOAT_FIRST_CHARACTERS = list("-+0123456789.")


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that reads exponent notation such as 1e-3 as a float."""


class _ConfigDumper(yaml.SafeDumper):
    """Safe YAML dumper that quotes text which would otherwise read back as a float."""


_ConfigLoader.add_implicit_resolver(_FLOAT_TAG, _EXPONENT_FLOAT, _FLOAT_FIRST_CHARACTERS)
_ConfigDumper.add_implicit_resolver(_FLOAT_TAG, _EXPONENT_FLOAT, _FLOAT_FIRST_CHARACTERS)


def _one_line(error: Exception) -> str:
    return " ".join(str(error).split())


def _refuse(key: str, setting: Setting, value: Any) -> ConfigError:
    return ConfigError(f"configuration key '{key}' must be {setting.rule}; got {value!r}", key)


def _is_list(value: Any, setting: Setting) -> bool:
    return isinstance(value, list) and len(value) >= setting.min_length


def _is_text(item: Any) -> bool:
    return isinstance(item, str) and item != ""


def _as_int(key: str, setting: Setting, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse(key, setting, value)
    if setting.check is not None and not setting.check(value):
        raise _refuse(key, setting, value)
    return value


def _as_float(key: str, setting: Setting, value: Any) -> float:
    """Return the value as a float. An integer is converted, so 32 and 32.0 are the same setting."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _refuse(key, setting, value)
    try:
        number = float(value)
    except OverflowError as error:  # an integer too large for a float
        raise _refuse(key, setting, value) from error
    if not math.isfinite(number):
        raise _refuse(key, setting, value)
    if setting.check is not None and not setting.check(number):
        raise _refuse(key, setting, value)
    return number


def _coerce(key: str, setting: Setting, value: Any) -> Any:
    """Check one configured value against its setting and return it in canonical form."""
    kind = setting.kind
    if kind == "int":
        result: Any = _as_int(key, setting, value)
    elif kind == "float":
        result = _as_float(key, setting, value)
    elif kind == "path":
        if value is not None and not isinstance(value, str):
            raise _refuse(key, setting, value)
        # Placeholders such as <root>/bodym are kept as written; assets.py resolves them later.
        result = value
    elif kind == "enum":
        if not isinstance(value, str) or value not in setting.choices:
            raise _refuse(key, setting, value)
        result = value
    elif kind == "text_list":
        if not (_is_list(value, setting) and all(_is_text(item) for item in value)):
            raise _refuse(key, setting, value)
        result = list(value)
    elif kind == "int_list":
        if not _is_list(value, setting):
            raise _refuse(key, setting, value)
        result = [_as_int(key, setting, item) for item in value]
    elif kind == "float_list":
        if not _is_list(value, setting):
            raise _refuse(key, setting, value)
        result = [_as_float(key, setting, item) for item in value]
    elif kind == "float_pair":
        if not isinstance(value, list) or len(value) != 2:
            raise _refuse(key, setting, value)
        low, high = [_as_float(key, setting, item) for item in value]
        if low > high:
            raise _refuse(key, setting, value)
        result = [low, high]
    elif kind == "float_map":
        if not isinstance(value, Mapping):
            raise _refuse(key, setting, value)
        result = copy.deepcopy(setting.default)
        for name, item in value.items():
            entry_key = f"{key}.{name}"
            if name not in setting.default:
                raise ConfigError(f"unknown configuration key '{entry_key}'", entry_key)
            result[name] = _as_float(entry_key, setting, item)
    else:
        raise ValueError(f"unknown setting kind {kind!r}")
    if setting.fixed is not _NOT_FIXED and result != setting.fixed:
        raise ConfigError(
            f"configuration key '{key}' is {setting.fixed_reason} at {setting.fixed!r}; "
            f"got {value!r}",
            key,
        )
    return result


def _resolve_section(
    schema: Mapping[str, Any], given: Mapping[Any, Any], prefix: str
) -> dict[str, Any]:
    """Validate the keys of one section, and fill in the defaults of the keys it leaves out."""
    unknown = sorted(str(name) for name in given if name not in schema)
    if unknown:
        keys = [_join(prefix, name) for name in unknown]
        noun = "keys" if len(keys) > 1 else "key"
        listed = ", ".join(f"'{key}'" for key in keys)
        raise ConfigError(f"unknown configuration {noun} {listed}", keys[0])
    resolved: dict[str, Any] = {}
    for name, entry in schema.items():
        key = _join(prefix, name)
        if isinstance(entry, Setting):
            if name in given:
                resolved[name] = _coerce(key, entry, given[name])
            elif entry.required:
                raise ConfigError(f"missing required configuration key '{key}'", key)
            else:
                resolved[name] = copy.deepcopy(entry.default)
        else:
            section = given.get(name, {})
            if not isinstance(section, Mapping):
                raise ConfigError(
                    f"configuration section '{key}' must be a mapping of keys; got {section!r}",
                    key,
                )
            resolved[name] = _resolve_section(entry, section, key)
    return resolved


def _check_view_counts(resolved: Mapping[str, Any]) -> None:
    """Refuse a view count above the cameras in a rig. The rule spans two keys, so it runs last."""
    cameras = resolved["camera"]["n_cameras"]
    for section, name in (("train", "views_train"), ("evaluate", "views")):
        key = f"{section}.{name}"
        for views in resolved[section][name]:
            if views > cameras:
                raise ConfigError(
                    f"configuration key '{key}' holds {views}, but camera.n_cameras is {cameras}; "
                    "a view count cannot exceed the cameras in a rig",
                    key,
                )


def _locate(path: str) -> tuple[Setting, str | None]:
    """Return the setting a dotted key path names, and the map key for one float_map entry."""
    parts = path.split(".")
    node: Any = _SCHEMA
    for position, name in enumerate(parts):
        if isinstance(node, Setting):
            if node.kind == "float_map" and position == len(parts) - 1 and name in node.default:
                return node, name
            raise ConfigError(f"unknown configuration key '{path}'", path)
        if name not in node:
            raise ConfigError(f"unknown configuration key '{path}'", path)
        node = node[name]
    if not isinstance(node, Setting):
        raise ConfigError(
            f"configuration key '{path}' is a section; give one of its keys instead", path
        )
    return node, None


def _apply_overrides(document: dict[str, Any], overrides: Iterable[str]) -> None:
    """Store each --set value in the document, after the checks that only overrides need."""
    seen: set[str] = set()
    for text in overrides:
        path, value = parse_override(text)
        if path in seen:
            raise ConfigError(f"configuration key '{path}' is set more than once with --set", path)
        seen.add(path)
        setting, map_key = _locate(path)
        if setting.fixed is not _NOT_FIXED and map_key is None:
            raise ConfigError(
                f"configuration key '{path}' is {setting.fixed_reason} and cannot be overridden "
                "with --set",
                path,
            )
        *sections, leaf = path.split(".")
        node = document
        walked = ""
        for name in sections:
            walked = _join(walked, name)
            if name not in node:
                node[name] = {}
            elif not isinstance(node[name], dict):
                raise ConfigError(
                    f"configuration section '{walked}' must be a mapping of keys", walked
                )
            node = node[name]
        node[leaf] = value


def resolve_config(raw: Any, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Validate a parsed configuration document, apply the --set overrides, and fill in defaults.

    Raises ConfigError (exit code 2) for an unknown key, a missing required key, a value of the
    wrong type or outside its rule, an override of a fixed key, or a view count above the cameras.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"the configuration must be a mapping of keys; got {raw!r}")
    document = copy.deepcopy(dict(raw))
    _apply_overrides(document, overrides)
    resolved = _resolve_section(_SCHEMA, document, "")
    _check_view_counts(resolved)
    return resolved


def parse_override(text: str) -> tuple[str, Any]:
    """Split one --set argument, key.path=value, and parse its value as YAML.

    The argument is split at the first equals sign, so a value may itself contain one.
    """
    key, separator, value_text = text.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ConfigError(f"--set expects key=value; got {text!r}")
    try:
        value = yaml.load(value_text, Loader=_ConfigLoader)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"the value for '{key}' is not valid YAML: {_one_line(error)}", key
        ) from error
    return key, value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Read a YAML configuration file, apply the --set overrides, and return the resolved config."""
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            f"cannot read the configuration file '{location}': {error.strerror or error}"
        ) from error
    try:
        raw = yaml.load(text, Loader=_ConfigLoader)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"the configuration file '{location}' is not valid YAML: {_one_line(error)}"
        ) from error
    return resolve_config(raw, overrides)


def canonical_dump(config: Mapping[str, Any]) -> str:
    """Return the canonical YAML text of a configuration: sorted keys, resolved values, no wrapping.

    The input is validated first, so the same values written in different spellings (32 and 32.0,
    1e-3 and 0.001) give the same text.
    """
    return yaml.dump(
        resolve_config(config),
        Dumper=_ConfigDumper,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=float("inf"),
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest (64 lowercase characters) of the canonical dump."""
    return hashlib.sha256(canonical_dump(config).encode("utf-8")).hexdigest()
