"""Timeout behaviour of `apple_notes_brain.applescript.run`.

Reproduces the wedge scenario from the post-merge live-use review: a single
stuck osascript invocation must NOT propagate `subprocess.TimeoutExpired`
(callers catch `AppleScriptError`, not the bare subprocess exception), and
must clean up so that subsequent pure-Python tools (e.g. `list_folders`)
still work.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from apple_notes_brain import applescript
from apple_notes_brain.applescript import (
    AppleScriptError,
    AppleScriptTimeoutError,
    run,
)


def _fake_popen(timeout_on_communicate: bool = False, returncode: int = 0,
                stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a MagicMock that quacks like subprocess.Popen for our run()."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode
    if timeout_on_communicate:
        # First call raises (the timeout); second call is the drain after kill.
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["osascript", "-"], timeout=5.0),
            ("", ""),
        ]
    else:
        proc.communicate.return_value = (stdout, stderr)
    proc.kill.return_value = None
    proc.wait.return_value = 0
    return proc


def test_timeout_raises_appplescript_timeout_error() -> None:
    """A timed-out osascript must raise AppleScriptTimeoutError, NOT the
    bare subprocess.TimeoutExpired. Callers catch AppleScriptError; a bare
    TimeoutExpired would leak through every layer."""
    proc = _fake_popen(timeout_on_communicate=True)
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc), \
         patch("apple_notes_brain.applescript.os.killpg") as killpg_mock, \
         patch("apple_notes_brain.applescript.os.getpgid", return_value=12345):
        with pytest.raises(AppleScriptTimeoutError, match="timed out"):
            run("tell application \"Notes\" to count notes", timeout=0.05)
        # We must have at least attempted to kill the process group (not just
        # the lone process) — Notes.app's AS host may spawn helpers.
        killpg_mock.assert_called()


def test_timeout_error_is_applescript_error_subclass() -> None:
    """Existing call sites catch `AppleScriptError` — the timeout subclass
    must still be caught by those `except AppleScriptError` paths."""
    assert issubclass(AppleScriptTimeoutError, AppleScriptError)


def test_no_timeout_returns_stdout_cleanly() -> None:
    proc = _fake_popen(stdout="hello\n", returncode=0)
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc):
        out = run("...", timeout=5.0)
    assert out == "hello\n"


def test_nonzero_exit_raises_applescript_error_not_timeout() -> None:
    proc = _fake_popen(returncode=1, stderr="syntax error")
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc):
        with pytest.raises(AppleScriptError, match="syntax error"):
            run("...", timeout=5.0)


def test_default_timeout_applied_when_none_passed() -> None:
    """When timeout is omitted, the module's DEFAULT_TIMEOUT is used.
    Regression guard: a future refactor that drops the default would
    re-introduce the wedge mode (calls hang forever)."""
    proc = _fake_popen(stdout="", returncode=0)
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc):
        run("...")
    # Verify communicate received the module default.
    args, kwargs = proc.communicate.call_args
    assert kwargs.get("timeout") == applescript.DEFAULT_TIMEOUT


def test_explicit_timeout_passed_through_to_communicate() -> None:
    proc = _fake_popen(stdout="", returncode=0)
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc):
        run("...", timeout=7.5)
    args, kwargs = proc.communicate.call_args
    assert kwargs.get("timeout") == 7.5


def test_read_only_timeout_constant_is_smaller_than_default() -> None:
    """Read probes should have a shorter ceiling than write operations —
    they have no IO and any hang is true wedge, not legitimate slowness."""
    assert applescript.READ_ONLY_TIMEOUT < applescript.DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# End-to-end wedge scenario: timed-out create_note must NOT prevent pure-
# Python tools from working afterwards.
# ---------------------------------------------------------------------------

def test_timeout_does_not_wedge_pure_python_tools(mocker) -> None:
    """End-to-end reproduction of the live-use wedge:

      1. create_note(format='html', body='<p>x</p>') triggers a simulated
         osascript timeout.
      2. AppleScriptTimeoutError surfaces cleanly (wrapped as ValueError by
         the @_safe_tool decorator — that's the model-facing surface).
      3. The pure-Python `list_folders` tool still works afterwards.
    """
    from apple_notes_brain import tools

    # First call: create_note hits the simulated timeout.
    proc = _fake_popen(timeout_on_communicate=True)
    with patch("apple_notes_brain.applescript.subprocess.Popen", return_value=proc), \
         patch("apple_notes_brain.applescript.os.killpg"), \
         patch("apple_notes_brain.applescript.os.getpgid", return_value=12345), \
         patch("apple_notes_brain.tools._resolve_folder_uri", return_value=None):
        with pytest.raises(ValueError) as excinfo:
            tools.create_note(title="t", body="<p>x</p>", format="html")
        # The @_safe_tool wrapper translates AppleScript failures into
        # ValueError; make sure the timeout origin survives in the chain.
        chain = []
        cur = excinfo.value
        while cur is not None:
            chain.append(type(cur))
            cur = cur.__cause__
        assert AppleScriptTimeoutError in chain or any(
            issubclass(c, AppleScriptError) for c in chain
        )

    # Pure-Python tool — must still work after the timeout. We mock the
    # SQLite reader so it never touches a real DB.
    mocker.patch(
        "apple_notes_brain.tools.db.list_folders",
        return_value=[
            {"id": "f1", "path": "Inbox", "is_trash": False,
             "account": "iCloud", "shared": False},
        ],
    )
    folders = tools.list_folders()
    assert len(folders) == 1
    assert folders[0].path == "Inbox"
