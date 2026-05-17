"""Debug logging infrastructure for the semantic subsystem.

Mirrors obsidian-brain's `src/util/debug-log.ts` + `src/util/logger.ts`
pattern, adapted to Python's stdlib logging module.

Design notes:

- **Gate on env var, not config field.** `APPLE_NOTES_BRAIN_DEBUG=1`
  is read DIRECTLY via `os.environ.get(...)` at log time, not from a
  cached config object. This matters because debug mode is most useful
  before the first config-load (e.g. inside `get_state()` itself).
- **Elapsed milliseconds since process start** in every line. Matches
  obsidian-brain's `[+{elapsedMs}ms]` format so log lines from the two
  -brain servers correlate naturally on a shared user box.
- **Sync write.** `logging.debug(...)` is sync; we don't add any extra
  async buffering. Debug lines that land in the log right before a
  crash are the whole point.
- **Failures swallowed.** A logger misconfig must not kill the caller.
  Every call is wrapped in `try/except`.
- **Idempotent setup.** `setup_logging()` is safe to call repeatedly
  from any entry point (the first call wins; subsequent calls verify
  the level matches and otherwise no-op).
"""
from __future__ import annotations

import logging
import os
import time

ENV_DEBUG = "APPLE_NOTES_BRAIN_DEBUG"

_LOGGER_NAME = "apple-notes-brain"

# Captured at module import — used to compute elapsed-ms for every debug
# call. Module is imported once per process, so this is effectively
# "process start" minus a few ms for the import.
_PROCESS_START_MS = time.monotonic_ns() // 1_000_000


def _elapsed_ms() -> int:
    """Milliseconds since this module was first imported."""
    return (time.monotonic_ns() // 1_000_000) - _PROCESS_START_MS


def is_debug_enabled() -> bool:
    """Return True iff `APPLE_NOTES_BRAIN_DEBUG=1` in the environment.

    Reads the env var fresh on every call — tests can toggle via
    monkeypatch without resetting any cached state.
    """
    return os.environ.get(ENV_DEBUG) == "1"


def setup_logging() -> None:
    """Configure the apple-notes-brain logger based on the debug env var.

    If `APPLE_NOTES_BRAIN_DEBUG=1` → set the logger level to DEBUG.
    Otherwise leave the current level alone (existing code calls
    `logging.basicConfig(level=logging.INFO)` at server boot so the
    default is INFO).

    Idempotent — safe to call from every entry point that might be the
    first.
    """
    try:
        logger = logging.getLogger(_LOGGER_NAME)
        if is_debug_enabled():
            logger.setLevel(logging.DEBUG)
        # else: don't force any level — defer to whoever else configured
        # the root logger (typically server.py's logging.basicConfig).
    except Exception:
        # A failure to set the level shouldn't crash boot. Worst case the
        # debug logs simply don't appear.
        pass


def debug_log(msg: str, **fields: object) -> None:
    """Emit a DEBUG-level message via the apple-notes-brain logger.

    Format: `[+{elapsed_ms}ms] {msg}` with optional `key=value` suffixes
    for each kwarg. Example::

        debug_log("indexer: pass start", count=42)
        # → "[+1234ms] indexer: pass start count=42"

    Calls are silently dropped if logging itself fails (e.g. logger
    misconfigured), so a debug line can never kill the caller.
    """
    try:
        logger = logging.getLogger(_LOGGER_NAME)
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if fields:
            suffix = " " + " ".join(f"{k}={v}" for k, v in fields.items())
        else:
            suffix = ""
        logger.debug("[+%dms] %s%s", _elapsed_ms(), msg, suffix)
    except Exception:
        # Drop silently — debug logs must never crash the caller.
        pass
