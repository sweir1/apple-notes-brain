"""Tests for the semantic watcher."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from apple_notes_brain.semantic.watcher import SemanticWatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(
    *,
    data_versions: list[int] | None = None,
    interval_s: float = 0.05,
    no_watch: bool = False,
):
    """Construct a watcher with a programmable data_version sequence."""
    seq = iter(data_versions or [1, 1, 1])

    def dv_fn() -> int:
        try:
            return next(seq)
        except StopIteration:
            return -1

    indexer = MagicMock()
    indexer.index_all.return_value = MagicMock()
    return (
        SemanticWatcher(
            indexer=indexer,
            source=MagicMock(),
            data_version_fn=dv_fn,
            interval_s=interval_s,
            no_watch=no_watch,
        ),
        indexer,
    )


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------

def test_no_watch_env_disables_start():
    w, indexer = _build(no_watch=True)
    started = w.start()
    assert started is False
    # No tick fires.
    time.sleep(0.05)
    assert indexer.index_all.call_count == 0


# ---------------------------------------------------------------------------
# Unchanged data_version — no reindex
# ---------------------------------------------------------------------------

def test_unchanged_data_version_skips_reindex():
    """When the data_version probe returns the same value every tick,
    the watcher does NOT call index_all (cheap-path)."""
    w, indexer = _build(data_versions=[5] * 10, interval_s=0.02)
    w.start()
    time.sleep(0.1)  # ≥ 3 ticks
    w.request_stop()
    w.join(timeout=1.0)
    assert w.tick_count >= 2
    assert w.reindex_count == 0
    assert indexer.index_all.call_count == 0


# ---------------------------------------------------------------------------
# Changed data_version — reindex fires
# ---------------------------------------------------------------------------

def test_change_triggers_reindex():
    """Boot dv=1, then dv=2 → at least one reindex."""
    w, indexer = _build(data_versions=[1, 2, 2, 2, 2, 2], interval_s=0.02)
    w.start()
    time.sleep(0.15)
    w.request_stop()
    w.join(timeout=1.0)
    assert w.reindex_count >= 1
    assert indexer.index_all.call_count == w.reindex_count


def test_multiple_changes_in_a_row_each_reindex():
    """Successive distinct dv values each trigger a reindex."""
    w, indexer = _build(
        data_versions=[1, 2, 3, 4, 5, 5, 5], interval_s=0.02
    )
    w.start()
    time.sleep(0.2)
    w.request_stop()
    w.join(timeout=1.0)
    assert w.reindex_count >= 3


# ---------------------------------------------------------------------------
# Error tolerance
# ---------------------------------------------------------------------------

def test_data_version_probe_error_is_recoverable():
    """A failed data_version probe increments error count but doesn't
    kill the thread; subsequent ticks still try."""
    seq = iter([Exception("fail"), 2, 2])

    def dv_fn():
        nxt = next(seq)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    indexer = MagicMock()
    w = SemanticWatcher(
        indexer=indexer,
        source=MagicMock(),
        data_version_fn=dv_fn,
        interval_s=0.02,
    )
    w.start()
    time.sleep(0.1)
    w.request_stop()
    w.join(timeout=1.0)
    assert w.error_count >= 1


def test_index_pass_error_is_recoverable():
    """When IndexPipeline.index_all raises, the watcher logs and keeps
    ticking. The thread does NOT die."""
    indexer = MagicMock()
    indexer.index_all.side_effect = [RuntimeError("oh no"), None]
    seq = iter([1, 2, 3, 3])

    def dv_fn():
        return next(seq)

    w = SemanticWatcher(
        indexer=indexer,
        source=MagicMock(),
        data_version_fn=dv_fn,
        interval_s=0.02,
    )
    w.start()
    time.sleep(0.15)
    w.request_stop()
    w.join(timeout=1.0)
    # At least one error counted; thread cleanly exited.
    assert w.error_count >= 1
    assert w._thread is not None and not w._thread.is_alive()


# ---------------------------------------------------------------------------
# Clean shutdown
# ---------------------------------------------------------------------------

def test_request_stop_breaks_loop_quickly():
    """After request_stop, the thread exits within ~1 interval."""
    w, _ = _build(data_versions=[1] * 50, interval_s=0.5)  # long interval
    w.start()
    time.sleep(0.05)
    t0 = time.time()
    w.request_stop()
    w.join(timeout=1.0)
    elapsed = time.time() - t0
    assert elapsed < 1.0
    assert w._thread is not None and not w._thread.is_alive()


def test_double_start_is_noop():
    w, indexer = _build(data_versions=[1] * 10)
    assert w.start() is True
    assert w.start() is False
    w.request_stop()
    w.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Interval clamping
# ---------------------------------------------------------------------------

def test_interval_clamped_to_min_floor():
    """Zero/negative intervals are clamped to the 10ms floor — protects
    against busy-loops. Real use cases enforce ≥1s in config.py."""
    w, _ = _build(interval_s=0.0)
    assert w._interval_s == 0.01
    w2, _ = _build(interval_s=-1.0)
    assert w2._interval_s == 0.01
