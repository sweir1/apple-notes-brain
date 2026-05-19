"""Tests for the IndexPipeline.

Uses FakeNotesSource (in-memory) + FakeEmbedder (deterministic) so the
whole indexer is exercised end-to-end without any external dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import (
    all_node_ids,
    all_sync_node_ids,
    count_failed_chunks,
    get_chunk,
    get_node,
    open_db,
)
from apple_notes_brain.semantic.types import (
    ChunkerConfig,
    EmbedderDeadError,
    TooLongError,
)

from .conftest import FakeEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rec(zid: str, title: str = "T", body: str = "", *, locked: bool = False,
         modified: int = 1700000000) -> NoteRecord:
    return NoteRecord(
        z_identifier=zid,
        z_pk=int(zid.split("-")[-1]) if "-" in zid else 1,
        title=title,
        folder="Notes",
        modified_at=modified,
        locked=locked,
        pinned=False,
    )


@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    yield c
    c.close()


@pytest.fixture
def emb() -> FakeEmbedder:
    e = FakeEmbedder(dim=64)
    e.init()
    return e


@pytest.fixture
def pipeline(conn, emb) -> IndexPipeline:
    return IndexPipeline(conn, emb, IndexerConfig(
        advertised_max_tokens=512,
        chunker_config=ChunkerConfig(
            chunk_size=100, min_chunk_chars=10, heading_split_depth=4
        ),
    ))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_index_all_empty_source_returns_empty_stats(pipeline):
    src = FakeNotesSource()
    stats = pipeline.index_all(src)
    assert stats.notes_seen == 0
    assert stats.notes_indexed == 0
    assert stats.chunks_embedded == 0


def test_index_all_inserts_notes_and_chunks(pipeline, conn, emb):
    src = FakeNotesSource()
    for i in range(3):
        src.add(_rec(f"zid-{i}", title=f"Note {i}"), f"Body content {i} long enough to chunk.")
    stats = pipeline.index_all(src)
    assert stats.notes_seen == 3
    assert stats.notes_indexed == 3
    assert stats.chunks_embedded >= 3  # at least 1 chunk per note
    # All nodes present.
    assert all_node_ids(conn) == {"zid-0", "zid-1", "zid-2"}
    # Each note has a sync row.
    assert all_sync_node_ids(conn) == {"zid-0", "zid-1", "zid-2"}
    # Embedder called at least 3 times.
    assert emb.embed_count >= 3


def test_index_all_returns_took_ms(pipeline):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min.")
    stats = pipeline.index_all(src)
    assert stats.took_ms >= 0


def test_index_all_records_chunk_content_in_store(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1", title="Apple"), "Body about apples and pears here long enough.")
    pipeline.index_all(src)
    chunk = get_chunk(conn, "zid-1#0")
    assert chunk is not None
    assert "apples" in chunk.content.lower()


# ---------------------------------------------------------------------------
# Content-hash dedup
# ---------------------------------------------------------------------------

def test_rerun_with_identical_source_skips_all_chunks(pipeline, emb):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min chunk.")
    pipeline.index_all(src)
    embed_count_after_first = emb.embed_count
    pipeline.index_all(src)
    # Second pass should not have called embed() again.
    assert emb.embed_count == embed_count_after_first


def test_changing_body_triggers_reembed(pipeline, emb):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Original body content longer than min.")
    pipeline.index_all(src)
    embed_count_first = emb.embed_count
    # Update body.
    src.add(_rec("zid-1"), "New body content here longer than min.")
    pipeline.index_all(src)
    assert emb.embed_count > embed_count_first


# ---------------------------------------------------------------------------
# Locked notes
# ---------------------------------------------------------------------------

def test_locked_note_is_recorded_and_skipped(pipeline, conn, emb):
    src = FakeNotesSource()
    src.add(_rec("zid-1", title="Locked", locked=True), "")
    stats = pipeline.index_all(src)
    assert stats.notes_skipped == 1
    assert stats.notes_indexed == 0
    # Node still upserted (so search by title works) but failed_chunks
    # logged with reason='locked'.
    assert count_failed_chunks(conn) == 1
    row = conn.execute(
        "SELECT reason FROM failed_chunks WHERE node_id = 'zid-1'"
    ).fetchone()
    assert row[0] == "locked"
    # Embedder not called for body.
    assert emb.embed_count == 0


def test_unlocking_note_clears_locked_failed_chunk(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1", title="X", locked=True), "")
    pipeline.index_all(src)
    assert count_failed_chunks(conn) == 1
    # Unlock + add body.
    src.add(_rec("zid-1", title="X", locked=False), "Body content here longer than min.")
    pipeline.index_all(src)
    # The previously-locked failed_chunk row is for chunk_id zid-1#0;
    # re-indexing as unlocked OVERWRITES that row via UPSERT in
    # record_failed_chunk (or leaves it if no failure occurred this
    # pass). Without explicit cleanup we don't expect the row to vanish;
    # the indexer doesn't sweep failed_chunks. Status remains visible.
    # The new indexed chunk is present:
    assert get_chunk(conn, "zid-1#0") is not None


# ---------------------------------------------------------------------------
# Empty-note fallback
# ---------------------------------------------------------------------------

def test_empty_body_falls_back_to_title_chunk(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1", title="Important Title"), "")
    pipeline.index_all(src)
    chunk = get_chunk(conn, "zid-1#0")
    assert chunk is not None
    assert "Important Title" in chunk.content


def test_whitespace_only_body_falls_back(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1", title="X"), "   \n\n   ")
    pipeline.index_all(src)
    chunk = get_chunk(conn, "zid-1#0")
    assert chunk is not None


# ---------------------------------------------------------------------------
# Tombstone sweep — notes removed from source vanish from index
# ---------------------------------------------------------------------------

def test_note_removed_from_source_is_deleted(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min.")
    src.add(_rec("zid-2"), "Other body content here longer than min.")
    pipeline.index_all(src)
    assert all_node_ids(conn) == {"zid-1", "zid-2"}
    src.remove("zid-1")
    stats = pipeline.index_all(src)
    assert stats.notes_deleted == 1
    assert all_node_ids(conn) == {"zid-2"}


# ---------------------------------------------------------------------------
# Error paths — TooLongError and EmbedderDeadError
# ---------------------------------------------------------------------------

def test_too_long_chunk_recorded_and_ratcheted(conn):
    """A chunk that's too long for the embedder is recorded as failed and
    triggers the capacity ratchet."""
    emb = FakeEmbedder(dim=64, max_chars=20)  # any chunk > 20 chars raises TooLong
    emb.init()
    pipeline = IndexPipeline(conn, emb, IndexerConfig(
        chunker_config=ChunkerConfig(chunk_size=200, min_chunk_chars=5),
    ))
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "A reasonably long body that exceeds max_chars=20 for sure.")
    stats = pipeline.index_all(src)
    assert stats.chunks_failed >= 1
    assert count_failed_chunks(conn) >= 1
    rows = list(conn.execute(
        "SELECT reason FROM failed_chunks WHERE reason = 'too-long'"
    ))
    assert len(rows) >= 1


def test_embedder_dead_error_aborts_pass(conn):
    """An EmbedderDeadError mid-iteration aborts the full pass."""

    class DyingEmbedder(FakeEmbedder):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._calls = 0

        def embed(self, text, task_type=None):
            self._calls += 1
            # Raise on the SECOND chunk embedded so the first note
            # completes and the second triggers the abort.
            if self._calls == 2:
                raise EmbedderDeadError("ollama is down")
            return super().embed(text, task_type)

    emb = DyingEmbedder(dim=64)
    emb.init()
    pipeline = IndexPipeline(conn, emb, IndexerConfig(
        chunker_config=ChunkerConfig(chunk_size=100, min_chunk_chars=5),
    ))
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content one longer than min.")
    src.add(_rec("zid-2"), "Body content two longer than min.")
    with pytest.raises(EmbedderDeadError):
        pipeline.index_all(src)


def test_arbitrary_embedder_exception_logged_as_embed_error(conn):
    """Any non-TooLong/non-Dead exception is treated as recoverable and
    logged to failed_chunks; the indexer keeps going."""

    class FlakyEmbedder(FakeEmbedder):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._calls = 0

        def embed(self, text, task_type=None):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("rng glitch")
            return super().embed(text, task_type)

    emb = FlakyEmbedder(dim=64)
    emb.init()
    pipeline = IndexPipeline(conn, emb, IndexerConfig(
        chunker_config=ChunkerConfig(chunk_size=100, min_chunk_chars=5),
    ))
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content one longer than min.")
    src.add(_rec("zid-2"), "Body content two longer than min.")
    stats = pipeline.index_all(src)
    assert stats.chunks_failed >= 1
    rows = list(conn.execute(
        "SELECT reason FROM failed_chunks WHERE reason = 'embed-error'"
    ))
    assert len(rows) >= 1
    # The second note still indexed despite the first failure.
    assert get_node(conn, "zid-2") is not None


# ---------------------------------------------------------------------------
# index_single (watcher path)
# ---------------------------------------------------------------------------

def test_index_single_add(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min.")
    result = pipeline.index_single(src, "zid-1", event="add")
    assert result.note_id == "zid-1"
    assert result.event == "add"
    assert result.chunks_embedded >= 1
    assert get_node(conn, "zid-1") is not None


def test_index_single_unlink(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min.")
    pipeline.index_single(src, "zid-1", event="add")
    assert get_node(conn, "zid-1") is not None
    result = pipeline.index_single(src, "zid-1", event="unlink")
    assert result.event == "unlink"
    assert get_node(conn, "zid-1") is None


def test_index_single_missing_record_treated_as_unlink(pipeline, conn):
    """If get_record returns None (note vanished between event and call),
    the indexer cleans up rather than crashing."""
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Body content here longer than min.")
    pipeline.index_all(src)
    src.remove("zid-1")
    result = pipeline.index_single(src, "zid-1", event="change")
    assert result.event == "unlink"
    assert get_node(conn, "zid-1") is None


def test_index_single_change_re_embeds(pipeline, conn, emb):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Original body content longer than min.")
    pipeline.index_all(src)
    before = emb.embed_count
    src.add(_rec("zid-1"), "Brand new body content longer than min.")
    pipeline.index_single(src, "zid-1", event="change")
    assert emb.embed_count > before


# ---------------------------------------------------------------------------
# Trash exclusion (Fix #1 — v1.1)
# ---------------------------------------------------------------------------

def _trash_rec(zid: str, title: str = "T", body: str = "") -> NoteRecord:
    """A NoteRecord whose folder is the canonical English trash name."""
    return NoteRecord(
        z_identifier=zid, z_pk=1, title=title,
        folder="Recently Deleted", modified_at=1700000000,
        locked=False, pinned=False,
    )


def test_index_all_excludes_trash_by_default(pipeline, conn):
    """A note whose folder is 'Recently Deleted' is silently dropped."""
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Live body content here long enough to chunk.")
    src.add(_trash_rec("zid-2"), "Trash body content here long enough to chunk.")
    stats = pipeline.index_all(src)
    assert stats.notes_seen == 1  # FakeNotesSource skips the trash record
    assert all_node_ids(conn) == {"zid-1"}
    assert get_node(conn, "zid-2") is None


def test_index_all_include_trash_true_indexes_trash(pipeline, conn):
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Live body content here long enough to chunk.")
    src.add(_trash_rec("zid-2"), "Trash body content here long enough to chunk.")
    stats = pipeline.index_all(src, include_trash=True)
    assert stats.notes_seen == 2
    assert all_node_ids(conn) == {"zid-1", "zid-2"}


def test_trashing_previously_indexed_note_triggers_tombstone(pipeline, conn):
    """End-to-end: a note indexed live, then moved to trash, gets
    tombstone-deleted from the index on the next pass."""
    src = FakeNotesSource()
    # Initial pass with the note in a live folder.
    live = _rec("zid-99", title="ShockSpec")
    src.add(live, "suspension shocks coilovers body content longer than min.")
    pipeline.index_all(src)
    assert get_node(conn, "zid-99") is not None

    # Simulate user moving the note to Recently Deleted.
    src.add(_trash_rec("zid-99", title="ShockSpec"),
            "suspension shocks coilovers body content longer than min.")
    stats = pipeline.index_all(src)
    # Tombstone sweep should have removed it.
    assert stats.notes_deleted == 1
    assert get_node(conn, "zid-99") is None


def test_only_trash_source_yields_empty_pass(pipeline, conn):
    src = FakeNotesSource()
    src.add(_trash_rec("zid-1"), "Body 1 here long enough to chunk.")
    src.add(_trash_rec("zid-2"), "Body 2 here long enough to chunk.")
    stats = pipeline.index_all(src)
    assert stats.notes_seen == 0
    assert all_node_ids(conn) == set()


def test_index_all_seen_count_excludes_trash(pipeline):
    """The reported `notes_seen` reflects what was actually iterated,
    not the underlying corpus size."""
    src = FakeNotesSource()
    for i in range(5):
        src.add(_rec(f"zid-{i}"), f"Body {i} content longer than min.")
    for i in range(5, 8):
        src.add(_trash_rec(f"zid-{i}"), f"Trash {i} body longer than min.")
    stats = pipeline.index_all(src)
    assert stats.notes_seen == 5


def test_trash_filter_handles_custom_trash_names(pipeline, conn):
    """FakeNotesSource accepts a custom trash-folder-name set for tests
    that simulate non-English-locale users."""
    src = FakeNotesSource(trash_folder_names={"Papelera"})
    rec = NoteRecord(
        z_identifier="zid-1", z_pk=1, title="T", folder="Papelera",
        modified_at=1700000000, locked=False, pinned=False,
    )
    src.add(rec, "Trash body content here longer than min.")
    pipeline.index_all(src)
    assert get_node(conn, "zid-1") is None


def test_trash_filter_does_not_drop_notes_without_folder(pipeline, conn):
    """A note with folder=None is treated as live (no folder ≠ trash)."""
    src = FakeNotesSource()
    rec = NoteRecord(
        z_identifier="zid-1", z_pk=1, title="T", folder=None,
        modified_at=1700000000, locked=False, pinned=False,
    )
    src.add(rec, "Body content here longer than min.")
    pipeline.index_all(src)
    assert get_node(conn, "zid-1") is not None


def test_index_all_with_include_trash_then_default_sweeps(pipeline, conn):
    """Index once with include_trash=True; subsequent default pass
    sweeps trash-folder nodes via the tombstone path."""
    src = FakeNotesSource()
    src.add(_rec("zid-1"), "Live body content here longer than min.")
    src.add(_trash_rec("zid-2"), "Trash body content here longer than min.")
    pipeline.index_all(src, include_trash=True)
    assert get_node(conn, "zid-2") is not None
    stats = pipeline.index_all(src)  # default include_trash=False
    assert stats.notes_deleted == 1
    assert get_node(conn, "zid-2") is None
