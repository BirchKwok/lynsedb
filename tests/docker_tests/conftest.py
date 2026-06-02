"""Fixtures for end-to-end remote HTTP API smoke tests."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ServerHandle:
    base_url: str
    process: subprocess.Popen | None
    api_key: str | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(base_url: str, api_key: str | None = None, timeout: float = 15.0) -> None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(base_url, headers=headers, timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.1)

    raise RuntimeError(f"Timed out waiting for server at {base_url}: {last_error}")


def _start_server(root_path: Path, api_key: str | None = None) -> ServerHandle:
    port = _find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "lynse.server",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--root",
        str(root_path),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        _wait_for_server(base_url, api_key=api_key)
    except Exception:
        process.kill()
        output = ""
        try:
            output, _ = process.communicate(timeout=5)
        except Exception:
            process.wait(timeout=5)
        raise RuntimeError(f"Failed to start LynseDB server.\nCommand: {cmd}\nOutput:\n{output}")

    return ServerHandle(base_url=base_url, process=process, api_key=api_key)


def _stop_server(handle: ServerHandle) -> None:
    if handle.process is None:
        return

    if handle.process.poll() is not None:
        return

    handle.process.terminate()
    try:
        handle.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=5)


@pytest.fixture(scope="session")
def remote_server(tmp_path_factory):
    base_url = os.environ.get("LYNSE_REMOTE_BASE_URL")
    if base_url:
        _wait_for_server(base_url)
        handle = ServerHandle(base_url=base_url, process=None)
    else:
        handle = _start_server(tmp_path_factory.mktemp("lynsedb-remote"))
    try:
        yield handle
    finally:
        _stop_server(handle)


@pytest.fixture(scope="session")
def remote_server_with_auth(tmp_path_factory):
    handle = _start_server(
        tmp_path_factory.mktemp("lynsedb-remote-auth"),
        api_key="remote-test-secret",
    )
    try:
        yield handle
    finally:
        _stop_server(handle)


@pytest.fixture
def unique_name():
    return f"test_{uuid.uuid4().hex[:8]}"
