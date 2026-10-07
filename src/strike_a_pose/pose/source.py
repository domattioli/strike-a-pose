"""Pose sources: the PoseSource protocol and the registry of the amass and limits sources (FR-001).

A pose source draws the pose of one synthetic body from the seeded generator of that body (research
R9). ``pose.source`` names the source. ``limits`` is the asset-free sampler of pose/limits.py
(research R2 and R3). ``amass`` reads AMASS motion-capture files from the asset root, in
pose/amass.py (research R2). Every source returns a zero root orientation, so the body stands
upright, and a pose_body in the joint order of body/base.py.

A source does not apply the pose filters. The caller passes each draw to pose/filters.py and redraws
a rejected pose (FR-001), so each source keeps only its own distribution.

Public sources. The protocol is a structural interface in the sense of PEP 544 (Protocols:
Structural subtyping, https://peps.python.org/pep-0544/). The registry is a read-only mapping from
each name to its factory. The design is research R2 (two sources behind one interface) and R9
(seeded generators).
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from strike_a_pose.assets import asset_root, require_asset
from strike_a_pose.body.base import BodyPose
from strike_a_pose.config import ConfigError
from strike_a_pose.pose.limits import LimitPoseSource

__all__ = ["POSE_SOURCE_NAMES", "PoseSource", "create_pose_source"]


@runtime_checkable
class PoseSource(Protocol):
    """A source of body poses. ``name`` is the value of pose.source that selects the source."""

    name: str

    def draw(self, rng: np.random.Generator) -> BodyPose:
        """Return ``(pose_root, pose_body)`` for one body, with angles in radians.

        ``pose_root`` is zero, shape (3,). ``pose_body`` holds the axis-angle rotations of joints 1
        to 21, shape (63,). The draw consumes ``rng``, the generator of that body.
        """


def _limits_source(config: Mapping[str, Any]) -> PoseSource:
    """Build the limits source from the joint-limit table of the configuration."""
    return LimitPoseSource(config["pose"]["limits_deg"])


def _amass_source(config: Mapping[str, Any]) -> PoseSource:
    """Build the amass source from the folders that pose.amass_subsets names under the asset root.

    Without subsets, the whole amass folder is used. The folders are checked first, so a missing
    asset stops the run with MissingAssetError before the loader module is imported.
    """
    root = asset_root(config)
    names = [f"<root>/amass/{subset}" for subset in config["pose"]["amass_subsets"]]
    folders = [require_asset(name, "pose.source", root) for name in names or ["<root>/amass"]]
    # Imported here, so that a limits run never loads the AMASS loader.
    from strike_a_pose.pose.amass import AmassPoseSource

    return AmassPoseSource(folders)


# The registry: each pose.source name maps to the factory that builds that source.
_REGISTRY: Mapping[str, Callable[[Mapping[str, Any]], PoseSource]] = MappingProxyType(
    {"amass": _amass_source, "limits": _limits_source}
)

POSE_SOURCE_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def create_pose_source(config: Mapping[str, Any]) -> PoseSource:
    """Return the source that ``pose.source`` names, built from a resolved configuration.

    An unregistered name raises ConfigError with the key pose.source (exit 2). The amass source
    raises MissingAssetError (exit 3) when the asset root is unset or a folder is absent.
    """
    name = config["pose"]["source"]
    factory = _REGISTRY.get(name)
    if factory is None:
        names = ", ".join(POSE_SOURCE_NAMES)
        raise ConfigError(
            f"configuration key 'pose.source' must be one of {names}; got {name!r}", "pose.source"
        )
    return factory(config)
