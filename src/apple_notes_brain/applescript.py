"""Thin wrapper around `osascript` for running AppleScript from Python.

Defence-in-depth against the "wedged server" failure mode:
  - Every `run()` call uses a hard wall-clock timeout via `subprocess.run`'s
    own `timeout=` argument. On expiry we kill the osascript process group
    (in case it spawned children — Notes.app's AS host has been observed to
    fork internally), then raise `AppleScriptTimeoutError`. We NEVER let
    `subprocess.TimeoutExpired` propagate, because callers and the outer
    `@_safe_tool` decorator catch `AppleScriptError` specifically.
  - Default timeout is conservative (30s) — large enough to absorb
    well-behaved write-path commits including the Notes.app→CloudKit
    save pipeline, small enough that one stuck call cannot wedge the
    MCP event loop indefinitely.
  - Read-only osascript calls (e.g. addressability probes) should pass
    a lower `timeout` (~10s) since they have no excuse to hang.
"""
from __future__ import annotations

import os
import signal
import subprocess

# Delimiters used inside AppleScript output. AppleScript emits these via
# `character id 30` (RS) and `character id 31` (US) — both control chars
# that do not appear in normal note content.
RECORD_SEP = "\x1e"
UNIT_SEP = "\x1f"

# Default wall-clock cap on any osascript invocation. Write-path commands
# (create/update/move/rename/delete) are the slowest because Notes.app's
# CoreData→CloudKit save pipeline runs inline; 30s comfortably accommodates
# the observed worst-case write under iCloud backpressure. Read-only probes
# should pass a shorter override.
DEFAULT_TIMEOUT = 30.0
# Suggested override for read-only AppleScript pings (existence probes,
# addressability checks). They have no IO and should never legitimately
# take more than a few seconds.
READ_ONLY_TIMEOUT = 10.0


class AppleScriptError(RuntimeError):
    """Raised when osascript exits non-zero."""


class AppleScriptTimeoutError(AppleScriptError):
    """Raised when osascript exceeds its wall-clock timeout.

    Distinct from AppleScriptError so callers can choose to retry or surface
    a more specific message. Inherits from AppleScriptError so blanket
    `except AppleScriptError` paths still catch it without code changes.
    """


def quote(s: str) -> str:
    """Return an AppleScript string literal for `s` with proper escaping."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def as_list(items: list[str]) -> str:
    """Format a list of strings as an AppleScript list literal."""
    return "{" + ", ".join(quote(x) for x in items) + "}"


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Best-effort: kill the osascript process group so any child processes
    forked by Notes.app's AppleScript host don't survive.

    Silent on every failure — by the time we get here the parent is already
    in trouble and we just want to clean up what we can."""
    try:
        # On POSIX we created the child in its own process group via
        # `start_new_session=True` below; kill the whole group via -pid.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=1.0)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run(script: str, timeout: float | None = None) -> str:
    """Run an AppleScript and return stdout.

    Args:
        script: the AppleScript source to feed osascript via stdin.
        timeout: wall-clock seconds before we kill the process group and
            raise. Default `DEFAULT_TIMEOUT` (30s). Pass `READ_ONLY_TIMEOUT`
            (10s) or lower for pure read probes.

    Raises:
        AppleScriptTimeoutError: timeout exceeded; process group has been
            killed.
        AppleScriptError: osascript returned a non-zero exit status.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    # Use Popen + communicate so we can explicitly kill the process group on
    # timeout (subprocess.run also does this but doesn't expose the group
    # semantics on macOS reliably across Python versions). start_new_session
    # puts the child in a fresh process group so SIGKILL reaches any
    # osascript-spawned helpers.
    try:
        proc = subprocess.Popen(  # noqa: S603 — osascript is trusted
            ["osascript", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise AppleScriptError(f"failed to launch osascript: {exc}") from exc

    try:
        stdout, stderr = proc.communicate(input=script, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        # Drain any output that did emerge so the pipes close.
        try:
            proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise AppleScriptTimeoutError(
            f"osascript timed out after {timeout}s (process group killed)"
        ) from exc
    except Exception:
        # Any other communicate() failure — make sure we don't leak the child.
        _kill_process_group(proc)
        raise

    if proc.returncode != 0:
        raise AppleScriptError(
            f"osascript failed (exit {proc.returncode}): {(stderr or '').strip()}"
        )
    return stdout or ""


def parse_records(output: str) -> list[list[str]]:
    """Split RECORD_SEP-delimited output into rows, each split on UNIT_SEP."""
    records = []
    for rec in output.split(RECORD_SEP):
        if not rec.strip("\n"):
            continue
        records.append(rec.lstrip("\n").split(UNIT_SEP))
    return records
