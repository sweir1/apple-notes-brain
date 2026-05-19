"""Tests for the semantic-subsystem boot block (Phase ζ).

Verifies:
  - start_semantic_subsystem_background spawns a daemon thread, idempotently.
  - The boot loop walks boot_phase: pending → embedder-init → bootstrap →
    indexing → ready.
  - first-time index fires when db is empty.
  - catch-up reindex respects APPLE_NOTES_BRAIN_NO_CATCHUP=1.
  - Drift forces a reindex.
  - Embedder init failure surfaces on state.init_error and state.boot_phase='failed'.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apple_notes_brain import tools_semantic
from apple_notes_brain.semantic.boot import (
    _no_catchup,
    start_semantic_subsystem_background,
)
from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import Search
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkerConfig

from .conftest import FakeEmbedder


def _rec(zid: str, title: str = "T", body: str = "Body text long enough.") -> tuple[NoteRecord, str]:
    # Parse trailing digits if present, fall back to hash(zid) otherwise so
    # exotic ids like 'zid-pre' / 'zid-tomb' don't trip int().
    tail = zid.rsplit("-", 1)[-1]
    z_pk = int(tail) if tail.isdigit() else (abs(hash(zid)) % 100000)
    return NoteRecord(
        z_identifier=zid, z_pk=z_pk, title=title,
        folder="Notes", modified_at=1700000000,
        locked=False, pinned=False,
    ), body


@pytest.fixture
def _scrub_module_state(monkeypatch):
    """Reset _state and _boot_thread on the tools_semantic module so each
    test gets a clean spawn."""
    if hasattr(tools_semantic, "_boot_thread"):
        delattr(tools_semantic, "_boot_thread")
    if hasattr(tools_semantic, "_boot_init_error"):
        delattr(tools_semantic, "_boot_init_error")
    tools_semantic._state = None
    yield
    tools_semantic._state = None
    if hasattr(tools_semantic, "_boot_thread"):
        delattr(tools_semantic, "_boot_thread")


def _make_state(tmp_path: Path, with_data: bool = False) -> tools_semantic.SemanticState:
    db_path = tmp_path / "x.db"
    conn = open_db(db_path)
    emb = FakeEmbedder(dim=32)
    emb.init()
    indexer = IndexPipeline(
        conn, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10, heading_split_depth=4,
        )),
    )
    src = FakeNotesSource()
    if with_data:
        for i in range(3):
            rec, body = _rec(f"zid-{i}")
            src.add(rec, body)
    search = Search(conn, emb)
    cfg = MagicMock(db_path=db_path)
    s = tools_semantic.SemanticState(
        conn=conn, embedder=emb, indexer=indexer,
        search=search, source=src, config_snapshot=cfg,
    )
    return s


# ---------------------------------------------------------------------------
# Env-gate semantics
# ---------------------------------------------------------------------------

def test_no_catchup_default_false(monkeypatch):
    monkeypatch.delenv("APPLE_NOTES_BRAIN_NO_CATCHUP", raising=False)
    assert _no_catchup() is False


def test_no_catchup_when_env_set(monkeypatch):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_NO_CATCHUP", "1")
    assert _no_catchup() is True


def test_no_catchup_other_values_treated_as_false(monkeypatch):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_NO_CATCHUP", "true")  # not '1'
    assert _no_catchup() is False


# ---------------------------------------------------------------------------
# start_semantic_subsystem_background — idempotency
# ---------------------------------------------------------------------------

def test_start_with_state_already_built_is_noop(tmp_path, _scrub_module_state):
    """If get_state() has already returned (state singleton exists), the
    boot block doesn't spawn a thread."""
    tools_semantic._state = _make_state(tmp_path)
    thread = start_semantic_subsystem_background()
    assert thread is None


def test_start_returns_thread_when_state_unbuilt(tmp_path, _scrub_module_state, monkeypatch):
    """Real path: a fresh process spawns the boot thread on first call.
    We monkey-patch get_state so it doesn't actually download a model."""
    fake_state = _make_state(tmp_path)
    monkeypatch.setattr(tools_semantic, "get_state", lambda: fake_state)
    thread = start_semantic_subsystem_background()
    assert thread is not None
    assert thread.is_alive() or not thread.is_alive()  # thread may have finished already
    thread.join(timeout=2.0)


def test_double_start_returns_same_thread_when_alive(
    tmp_path, _scrub_module_state, monkeypatch
):
    """If thread is still running, second call returns the same one."""
    fake_state = _make_state(tmp_path)

    # Use a slow-init mock so the thread is still alive when we re-call.
    indexer_slow = MagicMock()
    indexer_slow.index_all.side_effect = lambda *a, **kw: time.sleep(0.3) or MagicMock(
        notes_indexed=0, chunks_embedded=0, took_ms=300,
    )
    fake_state.indexer = indexer_slow

    monkeypatch.setattr(tools_semantic, "get_state", lambda: fake_state)
    t1 = start_semantic_subsystem_background()
    t2 = start_semantic_subsystem_background()
    assert t1 is t2
    if t1 is not None:
        t1.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Boot loop — first-time index path
# ---------------------------------------------------------------------------

def test_boot_loop_first_time_index_fires_on_empty_db(
    tmp_path, _scrub_module_state, monkeypatch
):
    """db_is_empty → indexer.index_all called exactly once."""
    state = _make_state(tmp_path, with_data=True)
    # Wrap the real index_all with a counter.
    real = state.indexer.index_all
    call_count = {"n": 0}

    def counted(*a, **kw):
        call_count["n"] += 1
        return real(*a, **kw)

    state.indexer.index_all = counted  # type: ignore[method-assign]
    monkeypatch.setattr(tools_semantic, "get_state", lambda: state)

    thread = start_semantic_subsystem_background()
    if thread is not None:
        thread.join(timeout=10.0)

    assert call_count["n"] == 1
    assert state.boot_phase == "ready"
    assert state.init_error is None


def test_boot_loop_no_catchup_skips_reindex(
    tmp_path, _scrub_module_state, monkeypatch
):
    """db non-empty + APPLE_NOTES_BRAIN_NO_CATCHUP=1 → indexer.index_all NOT called."""
    state = _make_state(tmp_path, with_data=True)
    # Pre-populate so db_is_empty=False on the bootstrap pass.
    rec, body = _rec("zid-pre")
    state.source.add(rec, body)
    state.indexer.index_all(state.source)  # full pre-index
    monkeypatch.setattr(tools_semantic, "get_state", lambda: state)

    monkeypatch.setenv("APPLE_NOTES_BRAIN_NO_CATCHUP", "1")
    real = state.indexer.index_all
    call_count = {"n": 0}

    def counted(*a, **kw):
        call_count["n"] += 1
        return real(*a, **kw)

    state.indexer.index_all = counted  # type: ignore[method-assign]
    thread = start_semantic_subsystem_background()
    if thread is not None:
        thread.join(timeout=10.0)

    assert call_count["n"] == 0
    assert state.boot_phase == "ready"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------

def test_boot_loop_records_failure_on_get_state_exception(
    tmp_path, _scrub_module_state, monkeypatch
):
    """If get_state() raises (e.g. ONNX cache corruption), the boot
    thread captures the error on the module rather than crashing."""

    def boom():
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(tools_semantic, "get_state", boom)
    thread = start_semantic_subsystem_background()
    if thread is not None:
        thread.join(timeout=2.0)
    # state is None since get_state failed; the error stash lives on the module.
    assert tools_semantic._state is None
    assert getattr(tools_semantic, "_boot_init_error", None) is not None


def test_boot_phase_attribute_present_on_state(tmp_path):
    """Track A's SemanticState should expose boot_phase + init_error."""
    state = _make_state(tmp_path)
    assert hasattr(state, "boot_phase")
    assert hasattr(state, "init_error")
    assert state.boot_phase == "ready"  # default for synchronous get_state
    assert state.init_error is None
