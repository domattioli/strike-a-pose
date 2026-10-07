"""Compute device selection (auto, cpu, cuda) with CPU fallback, and the hardware class (FR-028).

A run records its hardware class, cpu or gpu, and its device name in the run record (data-model
RunRecord), so every result names the hardware that produced it.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from strike_a_pose.config import ConfigError

__all__ = [
    "CPU_DEVICE_NAME",
    "CPU_HARDWARE_CLASS",
    "DEVICE_CHOICES",
    "DEVICE_ENVIRONMENT_VARIABLE",
    "GPU_HARDWARE_CLASS",
    "DeviceChoice",
    "requested_device_from_environment",
    "select_device",
]

logger = logging.getLogger(__name__)

# The values of the configuration key device, the --device option, and SAP_DEVICE
# (contracts/cli.md).
DEVICE_CHOICES = ("auto", "cpu", "cuda")
DEVICE_ENVIRONMENT_VARIABLE = "SAP_DEVICE"

# The hardware classes and the device name of the CPU, as the run record stores them.
CPU_HARDWARE_CLASS = "cpu"
GPU_HARDWARE_CLASS = "gpu"
CPU_DEVICE_NAME = "cpu"


@dataclass(frozen=True)
class DeviceChoice:
    """The device a run uses, with the request it answers and the values the run record stores.

    The device is the CPU when the request was cpu, and also when the request was auto or cuda and
    no GPU is present. Otherwise it is the first CUDA device.
    """

    requested: str
    device: torch.device
    hardware_class: str
    device_name: str


def select_device(requested: str = "auto") -> DeviceChoice:
    """Return the device for a request of auto, cpu, or cuda, using the CPU when no GPU is present.

    An auto or cuda request uses the first CUDA device when one is available. A cuda request without
    a GPU logs a warning and uses the CPU, and an auto request without a GPU logs a note. Any other
    request raises ConfigError, which the CLI reports with exit code 2 and names the key device.
    """
    if requested not in DEVICE_CHOICES:
        raise ConfigError(
            f"configuration key 'device' must be one of auto, cpu, cuda; got {requested!r}",
            "device",
        )
    if requested != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", 0)
        name = torch.cuda.get_device_name(device)
        return DeviceChoice(requested, device, GPU_HARDWARE_CLASS, name)
    if requested == "cuda":
        logger.warning("CUDA was requested, but no GPU is present; the run uses the CPU")
    elif requested == "auto":
        logger.info("no GPU is present; the run uses the CPU")
    return DeviceChoice(requested, torch.device("cpu"), CPU_HARDWARE_CLASS, CPU_DEVICE_NAME)


def requested_device_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the device named by SAP_DEVICE, or None when the variable is unset or empty.

    Any other value that is not auto, cpu, or cuda raises ConfigError, which names SAP_DEVICE. The
    process environment is read when no mapping is given.
    """
    source = os.environ if environ is None else environ
    value = source.get(DEVICE_ENVIRONMENT_VARIABLE, "")
    if value == "":
        return None
    if value not in DEVICE_CHOICES:
        raise ConfigError(
            f"environment variable {DEVICE_ENVIRONMENT_VARIABLE} must be one of auto, cpu, cuda; "
            f"got {value!r}",
            DEVICE_ENVIRONMENT_VARIABLE,
        )
    return value
