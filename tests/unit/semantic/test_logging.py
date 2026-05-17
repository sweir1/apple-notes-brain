"""Tests for `semantic/_logging.py` — the debug-mode gate + helper."""
from __future__ import annotations

import logging
import re

import pytest

from apple_notes_brain.semantic import _logging as dbg


_LOGGER_NAME = "apple-notes-brain"


@pytest.fixture(autouse=True)
def _reset_logger():
    """Restore the apple-notes-brain logger level after each test so a
    DEBUG flip in one case doesn't bleed into the next."""
    logger = logging.getLogger(_LOGGER_NAME)
    original = logger.level
    yield
    # Use the unbound method directly so a test that monkey-patched
    # `Logger.setLevel` doesn't break teardown (monkeypatch unwinds AFTER
    # this fixture, so we have to be defensive).
    try:
        logging.Logger.setLevel(logger, original)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# is_debug_enabled
# ---------------------------------------------------------------------------

def test_is_debug_enabled_true_when_env_one(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    assert dbg.is_debug_enabled() is True


def test_is_debug_enabled_false_when_env_zero(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "0")
    assert dbg.is_debug_enabled() is False


def test_is_debug_enabled_false_when_env_absent(monkeypatch):
    monkeypatch.delenv(dbg.ENV_DEBUG, raising=False)
    assert dbg.is_debug_enabled() is False


def test_is_debug_enabled_false_when_env_empty(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "")
    assert dbg.is_debug_enabled() is False


def test_is_debug_enabled_false_when_env_truthy_but_not_one(monkeypatch):
    """Explicit `"1"` only — `"true"` / `"yes"` etc. do NOT enable.
    Matches obsidian-brain's gate (strict equality)."""
    for val in ("true", "yes", "on", "TRUE", "2"):
        monkeypatch.setenv(dbg.ENV_DEBUG, val)
        assert dbg.is_debug_enabled() is False, f"value {val!r} should not enable"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

def test_setup_logging_flips_to_debug_when_env_set(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    dbg.setup_logging()
    assert logger.level == logging.DEBUG


def test_setup_logging_leaves_level_when_env_absent(monkeypatch):
    monkeypatch.delenv(dbg.ENV_DEBUG, raising=False)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    dbg.setup_logging()
    assert logger.level == logging.INFO


def test_setup_logging_leaves_level_when_env_zero(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "0")
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.WARNING)
    dbg.setup_logging()
    assert logger.level == logging.WARNING


def test_setup_logging_idempotent(monkeypatch):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    logger = logging.getLogger(_LOGGER_NAME)
    dbg.setup_logging()
    dbg.setup_logging()
    dbg.setup_logging()
    assert logger.level == logging.DEBUG


def test_setup_logging_swallows_exceptions(monkeypatch):
    """Even a logger-setLevel failure must not propagate."""
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")

    def boom(*args, **kwargs):
        raise RuntimeError("logger broken")

    monkeypatch.setattr(logging.Logger, "setLevel", boom)
    dbg.setup_logging()  # should NOT raise


# ---------------------------------------------------------------------------
# debug_log
# ---------------------------------------------------------------------------

def test_debug_log_writes_via_logger_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    dbg.setup_logging()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        dbg.debug_log("hello world")
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
    assert any("hello world" in m for m in msgs)


def test_debug_log_no_record_when_disabled(monkeypatch):
    """When env var is absent + logger is at INFO, the debug call must NOT
    emit a record. We assert via a manually attached handler — caplog
    would force DEBUG level on its own and defeat the gate."""
    monkeypatch.delenv(dbg.ENV_DEBUG, raising=False)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    h = _Capture(level=logging.DEBUG)
    logger.addHandler(h)
    try:
        dbg.debug_log("nope")
    finally:
        logger.removeHandler(h)
    assert not any("nope" in r.getMessage() for r in captured)


def test_debug_log_elapsed_ms_format(monkeypatch, caplog):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    dbg.setup_logging()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        dbg.debug_log("formatted")
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
    assert any(re.search(r"^\[\+\d+ms\] formatted", m) for m in msgs)


def test_debug_log_kwargs_formatted_as_key_value(monkeypatch, caplog):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    dbg.setup_logging()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        dbg.debug_log("event", count=42, name="foo")
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
    full = "\n".join(msgs)
    assert "count=42" in full
    assert "name=foo" in full
    assert "event" in full


def test_debug_log_swallows_exceptions(monkeypatch):
    """A misconfigured logger must not propagate exceptions up."""
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    dbg.setup_logging()

    def boom(*args, **kwargs):
        raise RuntimeError("logger broken")

    monkeypatch.setattr(logging.Logger, "debug", boom)
    # Should NOT raise.
    dbg.debug_log("safe")


def test_debug_log_handles_no_kwargs(monkeypatch, caplog):
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    dbg.setup_logging()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        dbg.debug_log("plain")
    # No trailing space when there are no kwargs.
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER_NAME]
    assert any(m.endswith("plain") for m in msgs)


def test_elapsed_ms_monotonic_nondecreasing():
    """Successive calls return non-decreasing elapsed values."""
    a = dbg._elapsed_ms()
    b = dbg._elapsed_ms()
    assert b >= a


def test_process_start_ms_is_int():
    assert isinstance(dbg._PROCESS_START_MS, int)
    assert dbg._PROCESS_START_MS > 0


def test_debug_log_skips_when_logger_below_debug(monkeypatch):
    """If env var is set BUT logger was not configured (level > DEBUG),
    `debug_log` should bail early (isEnabledFor check). Uses a manual
    handler — caplog would force DEBUG level and defeat the gate."""
    monkeypatch.setenv(dbg.ENV_DEBUG, "1")
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.WARNING)  # explicitly above DEBUG
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    h = _Capture(level=logging.DEBUG)
    logger.addHandler(h)
    try:
        dbg.debug_log("should not appear")
    finally:
        logger.removeHandler(h)
    assert not any("should not appear" in r.getMessage() for r in captured)
