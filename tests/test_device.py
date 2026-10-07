"""Smoke tests for device.py: auto, cpu, and cuda selection, the CPU fallback, and SAP_DEVICE."""

import logging

import pytest
import torch

from strike_a_pose.config import ConfigError
from strike_a_pose.device import (
    CPU_DEVICE_NAME,
    CPU_HARDWARE_CLASS,
    DEVICE_CHOICES,
    DEVICE_ENVIRONMENT_VARIABLE,
    GPU_HARDWARE_CLASS,
    DeviceChoice,
    requested_device_from_environment,
    select_device,
)

FAKE_GPU_NAME = "Tesla T4"


@pytest.fixture
def cuda_unavailable(monkeypatch):
    """Simulate a machine without a usable GPU, whatever hardware the test runs on."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def one_cuda_gpu(monkeypatch):
    """Simulate one CUDA device named Tesla T4, so no test needs a real GPU."""

    def device_name(device=None):
        assert device == torch.device("cuda", 0)
        return FAKE_GPU_NAME

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)


def test_the_device_choices_are_auto_cpu_and_cuda():
    assert DEVICE_CHOICES == ("auto", "cpu", "cuda")
    assert CPU_HARDWARE_CLASS == "cpu"
    assert GPU_HARDWARE_CLASS == "gpu"


def test_auto_falls_back_to_the_cpu_when_cuda_is_unavailable(cuda_unavailable):
    choice = select_device("auto")
    assert choice == DeviceChoice("auto", torch.device("cpu"), "cpu", CPU_DEVICE_NAME)


def test_cuda_request_falls_back_to_the_cpu_and_warns_when_cuda_is_unavailable(
    cuda_unavailable, caplog
):
    with caplog.at_level(logging.WARNING, logger="strike_a_pose.device"):
        choice = select_device("cuda")
    assert choice == DeviceChoice("cuda", torch.device("cpu"), "cpu", CPU_DEVICE_NAME)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "CUDA" in warnings[0].getMessage()


def test_cpu_request_uses_the_cpu_even_when_a_gpu_is_present(one_cuda_gpu):
    choice = select_device("cpu")
    assert choice == DeviceChoice("cpu", torch.device("cpu"), "cpu", CPU_DEVICE_NAME)


def test_auto_selects_the_first_cuda_device_when_one_is_present(one_cuda_gpu):
    choice = select_device("auto")
    assert choice == DeviceChoice("auto", torch.device("cuda", 0), "gpu", FAKE_GPU_NAME)


def test_cuda_selects_the_first_cuda_device_when_one_is_present(one_cuda_gpu):
    choice = select_device("cuda")
    assert choice == DeviceChoice("cuda", torch.device("cuda", 0), "gpu", FAKE_GPU_NAME)


def test_auto_is_the_default_request(cuda_unavailable):
    assert select_device() == select_device("auto")


def test_auto_resolves_consistently_on_the_machine_running_the_test():
    choice = select_device("auto")
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert choice.device.type == expected_type
    assert choice.hardware_class == ("gpu" if expected_type == "cuda" else "cpu")


@pytest.mark.parametrize("requested", ["gpu", "CUDA", "mps", "", None, 0])
def test_unknown_requests_are_refused_and_name_the_device_key(requested):
    with pytest.raises(ConfigError, match="'device'") as info:
        select_device(requested)
    assert info.value.key == "device"


def test_the_cpu_choice_holds_tensors_on_its_device(cuda_unavailable):
    choice = select_device("auto")
    tensor = torch.zeros(2, 3, device=choice.device)
    assert tensor.device == choice.device


@pytest.mark.parametrize("environ", [{}, {DEVICE_ENVIRONMENT_VARIABLE: ""}])
def test_an_unset_or_empty_device_variable_names_no_device(environ):
    assert requested_device_from_environment(environ) is None


@pytest.mark.parametrize("value", DEVICE_CHOICES)
def test_the_device_variable_names_each_choice(value):
    assert requested_device_from_environment({DEVICE_ENVIRONMENT_VARIABLE: value}) == value


@pytest.mark.parametrize("value", ["gpu", "CPU", " cpu"])
def test_other_device_variable_values_are_refused_and_name_the_variable(value):
    with pytest.raises(ConfigError, match=DEVICE_ENVIRONMENT_VARIABLE) as info:
        requested_device_from_environment({DEVICE_ENVIRONMENT_VARIABLE: value})
    assert info.value.key == DEVICE_ENVIRONMENT_VARIABLE


def test_the_device_variable_is_read_from_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(DEVICE_ENVIRONMENT_VARIABLE, "cpu")
    assert requested_device_from_environment() == "cpu"
    monkeypatch.delenv(DEVICE_ENVIRONMENT_VARIABLE)
    assert requested_device_from_environment() is None
