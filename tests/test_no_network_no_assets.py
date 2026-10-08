"""Check that the suite runs with the network blocker and the asset root unset (FR-021, FR-022)."""

import os
import socket

import pytest

REMOTE_ADDRESS = ("203.0.113.7", 9)
BLOCKED_MESSAGE = "network access is blocked in the test suite"


def test_network_blocker_refuses_remote_connect() -> None:
    """A TCP connect to a host off this machine is refused before any packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match=BLOCKED_MESSAGE):
            sock.connect(REMOTE_ADDRESS)


def test_network_blocker_refuses_remote_connect_ex() -> None:
    """The non-raising connect_ex is refused the same way, so no caller can bypass the blocker."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError, match=BLOCKED_MESSAGE):
            sock.connect_ex(REMOTE_ADDRESS)


def test_network_blocker_refuses_remote_name_lookup() -> None:
    """A name lookup for a host off this machine is refused."""
    with pytest.raises(OSError, match=BLOCKED_MESSAGE):
        socket.getaddrinfo("remote.example.invalid", 443)


def test_network_blocker_allows_loopback_connection() -> None:
    """A connection to this machine still works, so local sockets in the suite are not broken."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            assert client.connect_ex(("127.0.0.1", port)) == 0


def test_asset_root_is_unset_during_the_suite() -> None:
    """SAP_ASSET_ROOT is unset, so no test can read a licensed asset (constitution Principle II)."""
    assert "SAP_ASSET_ROOT" not in os.environ
