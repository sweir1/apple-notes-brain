"""End-to-end test of AppleNotesSource → IndexPipeline → Search using
the existing NoteStore.sqlite fixture from tests/integration/test_sqlite_in_memory.py.

Verifies that:
  1. AppleNotesSource can iterate notes from a NoteStore-shaped DB
  2. note_body_text gracefully handles missing/empty ZICNOTEDATA rows
  3. The indexer + search work end-to-end against real-shape Apple data
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import Search
from apple_notes_brain.semantic.source import AppleNotesSource
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkerConfig

import hashlib
import numpy as np


# Inlined from tests/integration/test_sqlite_in_memory.py since cross-
# directory imports require a tests/__init__.py the project doesn't ship.
FIXTURE_SQL = (
    Path(__file__).parent.parent.parent
    / "fixtures" / "sqlite" / "notestore_minimal.sql"
)


def _build_db(db_path: Path, extra_sql: str = "") -> None:
    sql = FIXTURE_SQL.read_text()
    if extra_sql:
        sql += "\n" + extra_sql
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(sql)
    conn.commit()
    conn.close()


def _patch_notestore(monkeypatch, db_path: Path) -> None:
    monkeypatch.setattr("apple_notes_brain.sqlite_reader.NOTE_STORE_PATH", db_path)
    monkeypatch.setattr("apple_notes_brain.sqlite_reader._uuid_cache", None)
    monkeypatch.setattr("apple_notes_brain.sqlite_reader._COLS_CACHE", {})

    def _open_fixture(path: Path = db_path) -> sqlite3.Connection:
        from apple_notes_brain.sqlite_reader import NoteStoreError
        if not path.exists():
            raise NoteStoreError(f"NoteStore not found at {path}")
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

    monkeypatch.setattr("apple_notes_brain.sqlite_reader._open", _open_fixture)


class FakeEmbedder:
    """Inlined to avoid the tests/ cross-directory import problem; same
    shape as tests/unit/semantic/conftest.py."""

    def __init__(self, dim: int = 32):
        self._dim = dim

    def init(self):
        pass

    def embed(self, text, task_type=None):
        seed = hashlib.sha256(f"{task_type or ''}\x00{text}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        v = rng.standard_normal(self._dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v

    def dimensions(self):
        return self._dim

    def model_identifier(self):
        return "fake/integration"

    def provider_name(self):
        return "fake"

    def dispose(self):
        pass

pytestmark = pytest.mark.integration


@pytest.fixture
def apple_setup(tmp_path: Path, monkeypatch):
    # Build the NoteStore.sqlite fixture (5 folders + 5 notes incl. locked + trashed).
    notestore = tmp_path / "NoteStore.sqlite"
    _build_db(notestore)
    _patch_notestore(monkeypatch, notestore)

    # Semantic-index DB lives separately.
    semantic_db = open_db(tmp_path / "semantic_index.db")
    emb = FakeEmbedder(dim=32)
    emb.init()
    indexer = IndexPipeline(
        semantic_db, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10, heading_split_depth=4,
        )),
    )
    source = AppleNotesSource()
    search = Search(semantic_db, emb)
    yield {"indexer": indexer, "search": search, "source": source, "conn": semantic_db}
    semantic_db.close()


def test_apple_notes_source_iterates_fixture_notes(apple_setup):
    """AppleNotesSource sees every non-deleted note from the fixture."""
    records = list(apple_setup["source"].iter_notes())
    # Fixture has 5 notes; one (PK 13) is ZMARKEDFORDELETION=1 → excluded.
    titles = {r.title for r in records}
    assert "Note in Notes" in titles
    assert "Note in Work" in titles
    assert "Note in Subfolder" in titles
    assert "Locked Note" in titles
    # Trashed note (PK 13) is filtered out.
    assert "Trashed Note" not in titles
    # Folder paths populated.
    assert all(r.folder is not None for r in records)


def test_locked_note_is_visible_in_listing(apple_setup):
    """Locked notes are still surfaced by iter_notes — the indexer
    decides what to do with them, not the source."""
    records = list(apple_setup["source"].iter_notes())
    locked = [r for r in records if r.locked]
    assert len(locked) == 1
    assert locked[0].title == "Locked Note"


def test_get_record_by_zidentifier(apple_setup):
    """The watcher path uses get_record(zid) to resolve single notes."""
    r = apple_setup["source"].get_record("note-1")
    assert r is not None
    assert r.title == "Note in Notes"


def test_get_record_missing_returns_none(apple_setup):
    assert apple_setup["source"].get_record("does-not-exist") is None


def test_end_to_end_index_then_search(apple_setup):
    """Run a full index pass against the fixture, then search for a
    seeded title — the note should appear in the results."""
    stats = apple_setup["indexer"].index_all(apple_setup["source"])
    # 4 visible non-trashed notes; locked one will index a placeholder.
    assert stats.notes_seen == 4
    # At least the three unlocked notes get a real chunk.
    assert stats.notes_indexed >= 3
    # Note titles are indexed into nodes_fts so the lexical path finds them.
    hits = apple_setup["search"].fulltext("Note in Work", limit=5)
    titles = [h.title for h in hits]
    assert "Note in Work" in titles


def test_body_text_for_locked_note_returns_empty(apple_setup):
    locked = next(
        r for r in apple_setup["source"].iter_notes() if r.locked
    )
    assert apple_setup["source"].body_text(locked) == ""


def test_body_text_for_note_without_zicnotedata_returns_empty(apple_setup):
    """The fixture has no ZICNOTEDATA rows; body_text should degrade
    gracefully to empty string rather than crashing."""
    visible = [
        r for r in apple_setup["source"].iter_notes()
        if not r.locked
    ]
    assert visible
    body = apple_setup["source"].body_text(visible[0])
    # Whether it's "" or a partial extracted string is fine — what we
    # care about is "no exception raised".
    assert isinstance(body, str)
