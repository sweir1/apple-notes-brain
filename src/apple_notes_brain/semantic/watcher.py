"""Background watcher — polls PRAGMA data_version, reindexes on change.

The lazy gate (data_version) is what makes this cheap: an idle Notes.app
ticks no work. When a note is added/edited, `data_version` increments
and the watcher kicks off a full `IndexPipeline.index_all()` pass on
the next tick (which is cheap on its own because content-hash dedup
skips everything that hasn't actually changed).

Mirrors obsidian-brain's `src/pipeline/watcher.ts` but adapted to the
Apple Notes model (no chokidar; the source is a SQLite DB Apple owns).

Lifecycle:
  - start()        — fire-and-forget; spawns the daemon thread
  - request_stop() — signals via threading.Event
  - join(timeout)  — wait for the thread to exit cleanly
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .source import NotesSource

_log = logging.getLogger("apple-notes-brain")


# A small jitter avoids thundering-herd if multiple watchers run on the
# same DB (e.g. two MCP clients) — they pick different observation times.
_JITTER_FRAC = 0.05


class SemanticWatcher:
    """Background thread + a stop event. Inert until start()."""

    def __init__(
        self,
        *,
        indexer: Any,                            # IndexPipeline (avoid circular import)
        source: NotesSource,
        data_version_fn: Callable[[], int],
        interval_s: float = 30.0,
        no_watch: bool = False,
    ):
        self._indexer = indexer
        self._source = source
        self._data_version_fn = data_version_fn
        # Floor at 10ms — config-layer validation enforces ≥1s for real
        # use; this floor exists only to prevent zero/negative intervals
        # busy-looping the host. Unit tests use ~20ms so we can run them
        # in single-digit ms.
        self._interval_s = max(0.01, interval_s)
        self._no_watch = no_watch

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_seen_dv: int | None = None
        self._tick_count = 0
        self._reindex_count = 0
        self._error_count = 0

    def start(self) -> bool:
        """Spawn the daemon thread. Returns False if `no_watch=True` or
        the thread already started."""
        if self._no_watch:
            _log.info("apple-notes-brain: semantic watcher disabled (no_watch=1)")
            return False
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="anb-semantic-watcher",
            daemon=True,
        )
        self._thread.start()
        return True

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def reindex_count(self) -> int:
        return self._reindex_count

    @property
    def error_count(self) -> int:
        return self._error_count

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            self._last_seen_dv = self._data_version_fn()
        except Exception as exc:
            _log.warning(
                "apple-notes-brain: semantic watcher couldn't read data_version "
                "at boot (%s); will retry on first tick", exc,
            )
            self._last_seen_dv = None

        while not self._stop.is_set():
            wait_s = self._interval_s * (1 + _JITTER_FRAC * 0)  # static jitter slot
            self._stop.wait(wait_s)
            if self._stop.is_set():
                break
            self._tick_count += 1
            try:
                current_dv = self._data_version_fn()
            except Exception as exc:
                self._error_count += 1
                _log.warning(
                    "apple-notes-brain: semantic watcher data_version probe failed: %s",
                    exc,
                )
                continue
            if self._last_seen_dv is not None and current_dv == self._last_seen_dv:
                # Nothing changed since the last tick — cheap path.
                continue
            self._last_seen_dv = current_dv
            try:
                self._indexer.index_all(self._source)
                self._reindex_count += 1
            except Exception as exc:
                self._error_count += 1
                _log.warning(
                    "apple-notes-brain: semantic watcher index pass raised: %s",
                    exc,
                )
                # Continue ticking — a single bad pass shouldn't kill the
                # daemon. The next data_version bump triggers a retry.
