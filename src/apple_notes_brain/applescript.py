"""Thin wrapper around `osascript` for running AppleScript from Python."""
from __future__ import annotations

import subprocess

# Delimiters used inside AppleScript output. AppleScript emits these via
# `character id 30` (RS) and `character id 31` (US) — both control chars
# that do not appear in normal note content.
RECORD_SEP = "\x1e"
UNIT_SEP = "\x1f"


class AppleScriptError(RuntimeError):
    """Raised when osascript exits non-zero or times out."""


def quote(s: str) -> str:
    """Return an AppleScript string literal for `s` with proper escaping."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def as_list(items: list[str]) -> str:
    """Format a list of strings as an AppleScript list literal."""
    return "{" + ", ".join(quote(x) for x in items) + "}"


def run(script: str, timeout: float = 60.0) -> str:
    """Run an AppleScript and return stdout. Raises AppleScriptError on failure."""
    try:
        result = subprocess.run(
            ["osascript", "-"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AppleScriptError(f"osascript timed out after {timeout}s") from exc
    if result.returncode != 0:
        raise AppleScriptError(
            f"osascript failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def parse_records(output: str) -> list[list[str]]:
    """Split RECORD_SEP-delimited output into rows, each split on UNIT_SEP."""
    records = []
    for rec in output.split(RECORD_SEP):
        if not rec.strip("\n"):
            continue
        records.append(rec.lstrip("\n").split(UNIT_SEP))
    return records
