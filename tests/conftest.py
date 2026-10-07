"""Shared pytest fixtures: tiny config, test overrides, output directory, asset root, network."""

import copy
import ipaddress
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"

# The test overrides of contracts/config.md. They shrink tiny.yaml so that every test
# finishes within the 60-second limit of constitution Principle VI.
SMALL_CONFIG_OVERRIDES: dict[str, object] = {
    "data.n_train": 64,
    "data.n_cal": 32,
    "data.n_test": 32,
    "data.shard_size": 32,
    "data.min_unflagged": 16,
    "calibrate.min_cal": 16,
    "train.epochs": 1,
    "predict.n_samples": 4,
    "camera.image_size": 32,
    "camera.focal_px": 32,
    "evaluate.views": [1, 4],
    "evaluate.noise_deg": [0],
}

LOCAL_HOST_NAMES = frozenset(
    {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
)


class NetworkBlockedError(OSError):
    """A test tried to reach a host other than this machine."""


def _is_local_host(host: object) -> bool:
    """Return True for empty, wildcard, loopback, or localhost names; False otherwise."""
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if host is None or host == "":
        return True
    if not isinstance(host, str):
        return False
    if host.lower() in LOCAL_HOST_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _refuse_remote_socket(sock: socket.socket, address: object, action: str) -> None:
    """Raise NetworkBlockedError when an IPv4 or IPv6 socket addresses a host off this machine."""
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    if isinstance(address, tuple) and address and not _is_local_host(address[0]):
        raise NetworkBlockedError(
            f"network access is blocked in the test suite: {action} to {address[0]!r}"
        )


def _refuse_remote_name(host: object) -> None:
    """Raise NetworkBlockedError when a name lookup asks for a host off this machine."""
    if not _is_local_host(host):
        raise NetworkBlockedError(
            f"network access is blocked in the test suite: name lookup for {host!r}"
        )


@pytest.fixture
def tiny_config_path() -> Path:
    """Path of configs/tiny.yaml, the CPU end-to-end configuration (FR-027)."""
    if not TINY_CONFIG.is_file():
        pytest.fail(f"missing configuration file: {TINY_CONFIG}")
    return TINY_CONFIG


@pytest.fixture
def small_config() -> dict[str, object]:
    """The test overrides of contracts/config.md, keyed by dotted configuration path."""
    return copy.deepcopy(SMALL_CONFIG_OVERRIDES)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """An output directory inside the test's temporary directory. The command creates it."""
    return tmp_path / "out"


@pytest.fixture(autouse=True)
def unset_asset_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset SAP_ASSET_ROOT for every test, so no test can read a licensed asset (FR-021)."""
    monkeypatch.delenv("SAP_ASSET_ROOT", raising=False)


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every socket connection, datagram send, and name lookup that leaves this machine."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname

    def connect(self, address):
        _refuse_remote_socket(self, address, "connect")
        return real_connect(self, address)

    def connect_ex(self, address):
        _refuse_remote_socket(self, address, "connect")
        return real_connect_ex(self, address)

    def sendto(self, data, *args):
        _refuse_remote_socket(self, args[-1] if args else None, "send")
        return real_sendto(self, data, *args)

    def getaddrinfo(host, port, *args, **kwargs):
        _refuse_remote_name(host)
        return real_getaddrinfo(host, port, *args, **kwargs)

    def gethostbyname(hostname):
        _refuse_remote_name(hostname)
        return real_gethostbyname(hostname)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", gethostbyname)
