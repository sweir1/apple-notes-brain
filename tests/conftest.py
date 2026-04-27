"""Shared fixtures for the apple-notes-brain test suite.

Organisation:
- Subprocess + AppleScript isolation
- SQLite isolation (in-memory + mock-connection patterns)
- Schema sample factories (NoteDetail, NoteSummary, Folder)
- Time + threading control (frozen monotonic, disabled background refresh)
- Cache state reset (so tests don't leak global state into each other)

These are deliberately conservative — only the most-repeated patterns are
fixtures; one-off mocks stay inline in the test that needs them.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from apple_notes_brain import cache, schemas


# ---------------------------------------------------------------------------
# Cache state isolation — autouse so EVERY test starts with clean overlays.
# Without this, tombstones/renames/count-deltas leak between tests and
# pytest-randomly will surface flakes.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_cache_state() -> Iterator[None]:
    """Reset cache module-level overlays before each test."""
    # Stop the background refresh thread if it's running (started at import in
    # production; tests must NOT have it ticking unless they explicitly start it).
    try:
        cache.stop_background_refresh(join_timeout_s=0.5)
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass

    with cache._tomb_lock:  # type: ignore[attr-defined]
        cache._tombstones.clear()  # type: ignore[attr-defined]
        cache._renames.clear()  # type: ignore[attr-defined]
        cache._count_deltas.clear()  # type: ignore[attr-defined]
    cache._recover_last_attempt = 0.0  # type: ignore[attr-defined]
    cache._last_data_version = 0  # type: ignore[attr-defined]
    cache._bg_last_refresh_ms = 0  # type: ignore[attr-defined]
    cache._bg_tick_count = 0  # type: ignore[attr-defined]
    cache._bg_skip_count = 0  # type: ignore[attr-defined]
    cache._last_activity_monotonic = 0.0  # type: ignore[attr-defined]
    yield


# ---------------------------------------------------------------------------
# AppleScript isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_applescript_run(mocker):
    """Patch applescript.run with a MagicMock returning empty string by default.

    Tests can set `.return_value` or `.side_effect` to control behaviour.
    """
    return mocker.patch("apple_notes_brain.applescript.run", return_value="")


@pytest.fixture
def mock_subprocess_run(mocker):
    """Patch subprocess.run module-wide so cache.* helpers don't actually
    invoke osascript / pgrep / pkill / open.
    """
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = ""
    proc.stderr = ""
    return mocker.patch("subprocess.run", return_value=proc)


# ---------------------------------------------------------------------------
# SQLite mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sqlite_conn(mocker):
    """Mock a sqlite3.Connection.

    Returns a MagicMock with .execute() returning a cursor whose
    fetchone()/fetchall()/fetchmany() return None/[] by default.
    Tests can override via cursor.execute.return_value.fetchall.return_value = ...
    """
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.fetchmany.return_value = []
    conn.cursor.return_value = cursor
    conn.execute.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=None)
    return conn


# ---------------------------------------------------------------------------
# Sample schema factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_folder() -> schemas.Folder:
    """A canonical Folder for tests that need one without caring about specifics."""
    return schemas.Folder(
        id="f7",
        path="Work/Projects",
        note_count=12,
        is_trash=False,
        account="iCloud",
        shared=False,
    )


@pytest.fixture
def sample_note_summary() -> schemas.NoteSummary:
    """A canonical NoteSummary."""
    return schemas.NoteSummary(
        id="p160",
        title="Test Note",
        folder="Notes",
        modified="2026-04-26 12:00",
        snippets=["…matched text…"],
        match_count=1,
        body_preview=None,
        pinned=False,
        locked=False,
        account="iCloud",
        attachments=0,
        shared=False,
    )


@pytest.fixture
def sample_note_detail() -> schemas.NoteDetail:
    """A canonical NoteDetail with non-empty body."""
    return schemas.NoteDetail(
        id="p160",
        title="Test Note",
        folder="Notes",
        modified="2026-04-26 12:00",
        body="# Heading\n\nBody paragraph.\n\n- bullet 1\n- bullet 2",
        format="markdown",
        pinned=False,
        locked=False,
        account="iCloud",
        attachments=0,
        shared=False,
    )


@pytest.fixture
def sample_mutation_result() -> schemas.MutationResult:
    """A canonical 'created' MutationResult."""
    return schemas.MutationResult(id="p999", action="created", error=None)


# ---------------------------------------------------------------------------
# Time control (for cache TTL / overlay tests)
# ---------------------------------------------------------------------------

class _FrozenMonotonic:
    """Replace time.monotonic with a controllable clock.

    Use as: clock.advance(60); assert clock.now() == 60.0
    Patches time.monotonic in the cache module specifically.
    """

    def __init__(self, start: float = 1000.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture
def frozen_monotonic(mocker) -> _FrozenMonotonic:
    """Freeze time.monotonic in the cache module.

    Yields a _FrozenMonotonic instance with .now() and .advance(seconds).
    """
    clock = _FrozenMonotonic(start=1000.0)
    mocker.patch("apple_notes_brain.cache.time.monotonic", side_effect=clock.now)
    return clock


# ---------------------------------------------------------------------------
# Background refresh disable (autouse — see _reset_cache_state above for the
# stop call. This fixture only matters for tests that import cache and
# accidentally trigger _bg_loop via start_background_refresh — they should
# explicitly opt into that fixture.)
# ---------------------------------------------------------------------------

@pytest.fixture
def disable_background_refresh(mocker):
    """Patch start_background_refresh to a no-op for tests that import server.py
    indirectly and would otherwise spawn the refresher thread.
    """
    return mocker.patch(
        "apple_notes_brain.cache.start_background_refresh",
        return_value=False,
    )


# ---------------------------------------------------------------------------
# Random seed reproducibility — pytest-randomly handles this, but we expose
# the seed as a fixture for tests that want to log it.
# ---------------------------------------------------------------------------

@pytest.fixture
def random_seed(request) -> int:
    """The seed pytest-randomly used for this test run.

    Useful for printing in error messages so flakes are reproducible.
    """
    seed = request.config.getoption("--randomly-seed", default=None)
    if seed is None:
        # Fallback if pytest-randomly isn't loaded
        return int(time.time())
    return int(seed)
