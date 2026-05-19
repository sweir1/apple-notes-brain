"""Tests for node + chunk CRUD + sync table + failed_chunks log."""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.store import (
    all_node_ids,
    all_sync_node_ids,
    clear_failed_chunks,
    count_failed_chunks,
    count_failed_chunks_by_reason,
    get_chunk,
    get_node,
    get_sync,
    index_status,
    list_failed_chunk_ids,
    open_db,
    record_failed_chunk,
    set_sync,
    upsert_chunk,
    upsert_node,
)


@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# upsert_node + get_node
# ---------------------------------------------------------------------------

def test_upsert_node_inserts_new(conn):
    rowid = upsert_node(
        conn, node_id="zid-1", z_pk=10, title="My Note",
        folder="Notes", modified_at=1700000000,
        locked=False, pinned=False, content_hash="abc",
    )
    assert rowid >= 1
    row = get_node(conn, "zid-1")
    assert row is not None
    assert row.id == "zid-1"
    assert row.z_pk == 10
    assert row.title == "My Note"
    assert row.folder == "Notes"
    assert row.locked is False
    assert row.pinned is False
    assert row.content_hash == "abc"


def test_upsert_node_updates_existing(conn):
    upsert_node(
        conn, node_id="zid-1", z_pk=10, title="Old",
        folder=None, modified_at=1, locked=False, pinned=False, content_hash="h1",
    )
    upsert_node(
        conn, node_id="zid-1", z_pk=10, title="New",
        folder="Notes", modified_at=2, locked=True, pinned=True, content_hash="h2",
    )
    row = get_node(conn, "zid-1")
    assert row.title == "New"
    assert row.folder == "Notes"
    assert row.locked is True
    assert row.pinned is True
    assert row.content_hash == "h2"
    assert row.modified_at == 2


def test_get_node_returns_none_for_missing(conn):
    assert get_node(conn, "nope") is None


def test_upsert_node_keeps_zidentifier_stable_across_z_pk_change(conn):
    """ZIDENTIFIER is our primary key; Z_PK can change (CloudKit can
    reassign). Upsert by ZIDENTIFIER updates the cached Z_PK."""
    upsert_node(
        conn, node_id="zid-1", z_pk=10, title="t",
        folder=None, modified_at=1, locked=False, pinned=False, content_hash=None,
    )
    upsert_node(
        conn, node_id="zid-1", z_pk=99, title="t",
        folder=None, modified_at=1, locked=False, pinned=False, content_hash=None,
    )
    row = get_node(conn, "zid-1")
    assert row.z_pk == 99


def test_all_node_ids(conn):
    for i in range(3):
        upsert_node(
            conn, node_id=f"zid-{i}", z_pk=i, title=f"t{i}",
            folder=None, modified_at=None,
            locked=False, pinned=False, content_hash=None,
        )
    assert all_node_ids(conn) == {"zid-0", "zid-1", "zid-2"}


# ---------------------------------------------------------------------------
# nodes_fts synced on upsert
# ---------------------------------------------------------------------------

def test_nodes_fts_synced_with_title_on_insert(conn):
    upsert_node(
        conn, node_id="zid-1", z_pk=1, title="Apple Pie Recipe",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    matches = list(conn.execute(
        "SELECT title FROM nodes_fts WHERE nodes_fts MATCH ?", ("apple",),
    ))
    assert len(matches) == 1
    assert matches[0][0] == "Apple Pie Recipe"


def test_nodes_fts_resyncs_on_title_update(conn):
    upsert_node(
        conn, node_id="zid-1", z_pk=1, title="Original Title",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    upsert_node(
        conn, node_id="zid-1", z_pk=1, title="Updated Heading",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    matches = list(conn.execute(
        "SELECT title FROM nodes_fts WHERE nodes_fts MATCH ?", ("updated",),
    ))
    assert len(matches) == 1
    no_old = list(conn.execute(
        "SELECT title FROM nodes_fts WHERE nodes_fts MATCH ?", ("original",),
    ))
    assert no_old == []


# ---------------------------------------------------------------------------
# upsert_chunk + get_chunk
# ---------------------------------------------------------------------------

def test_upsert_chunk_inserts(conn):
    upsert_node(
        conn, node_id="n", z_pk=1, title="t",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    rowid = upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading="Section",
        heading_level=2,
        content="Body text.",
        content_hash="hash-0",
        start_line=5,
        end_line=8,
    )
    assert rowid >= 1
    row = get_chunk(conn, "n#0")
    assert row.heading == "Section"
    assert row.heading_level == 2
    assert row.content == "Body text."
    assert row.content_hash == "hash-0"
    assert row.start_line == 5
    assert row.end_line == 8


def test_upsert_chunk_updates_on_conflict(conn):
    upsert_node(
        conn, node_id="n", z_pk=1, title="t",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading="H1", heading_level=1,
        content="old", content_hash="ho",
        start_line=1, end_line=1,
    )
    upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading="H2", heading_level=2,
        content="new", content_hash="hn",
        start_line=2, end_line=4,
    )
    row = get_chunk(conn, "n#0")
    assert row.heading == "H2"
    assert row.heading_level == 2
    assert row.content == "new"
    assert row.content_hash == "hn"
    assert row.start_line == 2


def test_get_chunk_returns_none_for_missing(conn):
    assert get_chunk(conn, "no-such") is None


# ---------------------------------------------------------------------------
# sync table
# ---------------------------------------------------------------------------

def test_set_sync_then_get(conn):
    upsert_node(
        conn, node_id="n", z_pk=1, title="t",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    set_sync(conn, "n", modified_at=100, indexed_at=200)
    assert get_sync(conn, "n") == (100, 200)


def test_set_sync_overwrites(conn):
    upsert_node(
        conn, node_id="n", z_pk=1, title="t",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    set_sync(conn, "n", modified_at=100, indexed_at=200)
    set_sync(conn, "n", modified_at=110, indexed_at=210)
    assert get_sync(conn, "n") == (110, 210)


def test_get_sync_missing_returns_none(conn):
    assert get_sync(conn, "nope") is None


def test_all_sync_node_ids(conn):
    for i in range(3):
        upsert_node(
            conn, node_id=f"n{i}", z_pk=i, title="t",
            folder=None, modified_at=None,
            locked=False, pinned=False, content_hash=None,
        )
        set_sync(conn, f"n{i}", modified_at=i, indexed_at=i)
    assert all_sync_node_ids(conn) == {"n0", "n1", "n2"}


# ---------------------------------------------------------------------------
# failed_chunks log
# ---------------------------------------------------------------------------

def test_record_failed_chunk_inserts(conn):
    record_failed_chunk(
        conn, chunk_id="n#0", node_id="n",
        reason="too-long", error_message="exceeds 512 tokens",
    )
    row = conn.execute(
        "SELECT reason, error_message FROM failed_chunks WHERE chunk_id = ?",
        ("n#0",),
    ).fetchone()
    assert row[0] == "too-long"
    assert row[1] == "exceeds 512 tokens"


def test_record_failed_chunk_overwrites_on_same_id(conn):
    record_failed_chunk(conn, chunk_id="n#0", node_id="n", reason="too-long")
    record_failed_chunk(
        conn, chunk_id="n#0", node_id="n",
        reason="embed-error", error_message="boom",
    )
    assert count_failed_chunks(conn) == 1
    row = conn.execute(
        "SELECT reason FROM failed_chunks WHERE chunk_id = 'n#0'"
    ).fetchone()
    assert row[0] == "embed-error"


def test_count_failed_chunks_empty(conn):
    assert count_failed_chunks(conn) == 0


# ---------------------------------------------------------------------------
# v1.1 Part 4 Phase 1 — locked-vs-failed partition at the store layer
# ---------------------------------------------------------------------------

def test_count_failed_chunks_by_reason(conn):
    for i in range(3):
        record_failed_chunk(conn, chunk_id=f"l-{i}", node_id=f"n-l-{i}", reason="locked")
    record_failed_chunk(conn, chunk_id="tl-0", node_id="n-tl-0", reason="too-long")
    record_failed_chunk(conn, chunk_id="ee-0", node_id="n-ee-0", reason="embed-error")
    by_reason = count_failed_chunks_by_reason(conn)
    assert by_reason == {"locked": 3, "too-long": 1, "embed-error": 1}


def test_count_failed_chunks_by_reason_empty(conn):
    assert count_failed_chunks_by_reason(conn) == {}


def test_clear_failed_chunks_returns_real_failure_count_only(conn):
    """clear_failed_chunks deletes ALL rows (so the next pass can
    re-validate locked notes) but reports only the real-failure count
    so callers don't see a misleading 'cleared 9' when none of those
    9 were actually failures."""
    for i in range(4):
        record_failed_chunk(conn, chunk_id=f"l-{i}", node_id=f"n-l-{i}", reason="locked")
    record_failed_chunk(conn, chunk_id="tl-0", node_id="n-tl-0", reason="too-long")
    record_failed_chunk(conn, chunk_id="tl-1", node_id="n-tl-1", reason="too-long")
    real = clear_failed_chunks(conn)
    assert real == 2  # only the two too-long, not the 4 locked
    assert count_failed_chunks(conn) == 0  # but all 6 rows are gone


def test_index_status_partitions_locked_from_failed(conn):
    """index_status separates locked_notes from total_failed_chunks."""
    for i in range(5):
        record_failed_chunk(conn, chunk_id=f"l-{i}", node_id=f"n-l-{i}", reason="locked")
    record_failed_chunk(conn, chunk_id="tl-0", node_id="n-tl-0", reason="too-long")
    s = index_status(conn)
    assert s.locked_notes == 5
    assert s.total_failed_chunks == 1
    assert s.failed_chunks_by_reason == {"too-long": 1}


def test_list_failed_chunk_ids_excludes_locked_by_default(conn):
    for i in range(3):
        record_failed_chunk(conn, chunk_id=f"l-{i}", node_id=f"n-l-{i}", reason="locked")
    record_failed_chunk(conn, chunk_id="tl-0", node_id="n-tl-0", reason="too-long")
    ids = list_failed_chunk_ids(conn)
    assert ids == ["tl-0"]


def test_list_failed_chunk_ids_with_exclude_locked_false_returns_all(conn):
    record_failed_chunk(conn, chunk_id="l-0", node_id="n-l-0", reason="locked")
    record_failed_chunk(conn, chunk_id="tl-0", node_id="n-tl-0", reason="too-long")
    ids = list_failed_chunk_ids(conn, exclude_locked=False)
    assert set(ids) == {"l-0", "tl-0"}
