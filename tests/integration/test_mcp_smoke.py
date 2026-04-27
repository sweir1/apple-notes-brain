"""End-to-end smoke test: subprocess-spawn the server and run JSON-RPC handshake.

Mirrors obsidian-brain's `scripts/mcp-smoke.ts`. Spawns the actual server via
the `apple-notes-brain` console script (installed by uv), sends an
`initialize` request + `tools/list` over stdin, and verifies the responses
come back over stdout.

These tests:
- Are slower than unit tests (process spawn ~1-3s)
- Do NOT require Apple Notes / FDA — the server boots, registers tools, and
  responds to MCP protocol calls without hitting any real I/O. We don't
  invoke any tools; we just verify the protocol surface.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Default 30-second timeout for any single test (process spawn + handshake)
PROC_TIMEOUT_S = 30


def _resolve_server_command() -> list[str] | None:
    """Find the right command to invoke the server.

    Priority:
    1. The console script installed in the active venv (apple-notes-brain)
    2. python -m apple_notes_brain via the venv's python
    3. Skip if neither found
    """
    venv_bin = Path(sys.executable).parent
    binary = venv_bin / "apple-notes-brain"
    if binary.exists() and os.access(binary, os.X_OK):
        return [str(binary)]
    # Fallback: python -m
    return [sys.executable, "-m", "apple_notes_brain"]


def _send_request(proc: subprocess.Popen, msg: dict) -> dict:
    """Send a JSON-RPC request, return the parsed response."""
    payload = json.dumps(msg) + "\n"
    proc.stdin.write(payload)
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        # process died
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"server exited without responding; stderr:\n{stderr}")
    return json.loads(line)


@pytest.fixture
def server_proc():
    """Spawn the server as a subprocess. Tear down cleanly on exit.

    Sets APPLE_NOTES_BRAIN_NO_PREWARM=1 + NOTES_MCP_AUTO_REFRESH=0 so the
    server boots without trying to talk to Notes.app via AppleScript.
    On CI runners (no Notes.app, no Automation permission, no UI), the
    osascript ping would otherwise hang for the 30s prewarm budget and
    every smoke test would time out before the first JSON-RPC reply.
    """
    cmd = _resolve_server_command()
    if cmd is None:
        pytest.skip("no apple-notes-brain command available in venv")

    env = os.environ.copy()
    env["APPLE_NOTES_BRAIN_NO_PREWARM"] = "1"
    env["NOTES_MCP_AUTO_REFRESH"] = "0"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        env=env,
    )
    try:
        yield proc
    finally:
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(PROC_TIMEOUT_S)
def test_server_responds_to_initialize(server_proc):
    """Send `initialize`, expect protocolVersion + serverInfo back."""
    response = _send_request(
        server_proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "0.0.1"},
            },
        },
    )
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 0
    assert "result" in response, f"no result in {response}"
    result = response["result"]
    assert "serverInfo" in result
    assert result["serverInfo"]["name"] == "apple-notes-brain"
    assert "capabilities" in result


@pytest.mark.timeout(PROC_TIMEOUT_S)
def test_server_lists_all_tools(server_proc):
    """After init, tools/list returns the 12 documented tools."""
    # Initialize
    _send_request(
        server_proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "x", "version": "0"}},
        },
    )
    # Send initialized notification
    server_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    server_proc.stdin.flush()
    # List tools
    response = _send_request(
        server_proc,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert "result" in response
    tools = response["result"]["tools"]
    names = sorted(t["name"] for t in tools)
    assert names == sorted([
        "list_folders", "list_notes", "search_notes", "get_note",
        "create_note", "update_note", "rename_note", "move_note",
        "create_folder", "rename_folder", "delete_folder", "delete_note",
    ])


@pytest.mark.timeout(PROC_TIMEOUT_S)
def test_server_lists_overview_prompt(server_proc):
    """prompts/list returns notes_server_overview."""
    _send_request(
        server_proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "x", "version": "0"}},
        },
    )
    server_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    server_proc.stdin.flush()
    response = _send_request(
        server_proc,
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/list", "params": {}},
    )
    assert "result" in response
    names = [p["name"] for p in response["result"]["prompts"]]
    assert "notes_server_overview" in names


@pytest.mark.timeout(PROC_TIMEOUT_S)
def test_server_lists_resources_without_error(server_proc):
    """resources/list returns successfully (may be empty if no NoteStore)."""
    _send_request(
        server_proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "x", "version": "0"}},
        },
    )
    server_proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    server_proc.stdin.flush()
    response = _send_request(
        server_proc,
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
    )
    assert "result" in response
    assert "resources" in response["result"]
    assert isinstance(response["result"]["resources"], list)


@pytest.mark.timeout(PROC_TIMEOUT_S)
def test_server_clean_shutdown_via_stdin_close(server_proc):
    """Closing stdin triggers clean server exit (no zombie process)."""
    _send_request(
        server_proc,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "x", "version": "0"}},
        },
    )
    server_proc.stdin.close()
    # Server should exit within a few seconds of stdin close.
    return_code = server_proc.wait(timeout=10)
    assert return_code == 0, f"unclean exit: {return_code}"
