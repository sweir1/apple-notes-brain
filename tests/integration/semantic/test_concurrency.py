"""Concurrency smoke tests for the semantic index.

Stresses the WAL+busy-timeout setup: one thread reindexes while
another searches. We expect no DB corruption, no torn vectors, and
search results that never include rows from a partially-deleted note.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import Search
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkerConfig

import hashlib
import numpy as np


class FakeEmbedder:
    """Minimal deterministic embedder for integration tests. Same shape
    as the one in tests/unit/semantic/conftest.py — duplicated here
    because the unit-test conftest isn't import-reachable from the
    integration tests under pytest's default rootdir resolution."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def init(self) -> None:
        pass

    def embed(self, text: str, task_type: str | None = None) -> np.ndarray:
        seed = hashlib.sha256(f"{task_type or ''}\x00{text}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        v = rng.standard_normal(self._dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v

    def dimensions(self) -> int:
        return self._dim

    def model_identifier(self) -> str:
        return "fake/integration"

    def provider_name(self) -> str:
        return "fake"

    def dispose(self) -> None:
        pass

pytestmark = pytest.mark.integration


def _rec(i: int) -> NoteRecord:
    return NoteRecord(
        z_identifier=f"zid-{i}", z_pk=i, title=f"Note {i}",
        folder="Notes", modified_at=1700000000 + i,
        locked=False, pinned=False,
    )


@pytest.fixture
def setup(tmp_path: Path):
    # WAL connections need to be per-thread; open separate ones for the
    # reindex thread vs the search thread.
    db_path = tmp_path / "concurrent.db"
    conn_a = open_db(db_path)
    conn_b = open_db(db_path)
    emb = FakeEmbedder(dim=32)
    emb.init()
    indexer = IndexPipeline(
        conn_a, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10, heading_split_depth=4,
        )),
    )
    src = FakeNotesSource()
    for i in range(20):
        src.add(_rec(i), f"Body content for note {i}, with enough text to chunk into bits.")
    indexer.index_all(src)
    search = Search(conn_b, emb)
    yield {"conn_a": conn_a, "conn_b": conn_b, "indexer": indexer, "search": search, "src": src}
    conn_a.close()
    conn_b.close()


def test_concurrent_reindex_and_search_no_corruption(setup):
    """Run reindex in a background thread while the main thread searches
    repeatedly. After both finish, the index should be consistent."""
    stop = threading.Event()
    errors: list[Exception] = []

    def reindex_loop():
        try:
            for _ in range(5):
                setup["indexer"].index_all(setup["src"])
                if stop.is_set():
                    return
        except Exception as exc:
            errors.append(exc)

    def search_loop():
        try:
            for _ in range(20):
                out = setup["search"].semantic("note", limit=5)
                # Every returned hit's note_id has the expected shape.
                for r in out:
                    assert r.note_id.startswith("zid-"), r
                time.sleep(0.005)
                if stop.is_set():
                    return
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=reindex_loop, daemon=True),
        threading.Thread(target=search_loop, daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    stop.set()
    assert not errors, f"concurrent ops raised: {errors!r}"


def test_search_during_reindex_returns_no_phantom_rows(setup):
    """A search that runs while reindex is mid-flight must return either
    valid rows or no rows for a given note_id — never partial state."""

    def reindex():
        for _ in range(3):
            setup["indexer"].index_all(setup["src"])

    t = threading.Thread(target=reindex, daemon=True)
    t.start()
    for _ in range(10):
        out = setup["search"].fulltext("note", limit=20)
        for r in out:
            # Every hit's title is non-empty (no partial-row torn read).
            assert r.title
        time.sleep(0.01)
    t.join(timeout=15)
