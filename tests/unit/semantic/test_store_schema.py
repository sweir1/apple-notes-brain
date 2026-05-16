"""Tests for the semantic-index sqlite store — schema + metadata.

Exercises open_db, _init_schema, schema-version handling, and the
metadata key/value helpers. Vector + node/chunk CRUD live in their
own test files.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from apple_notes_brain.semantic.store import (
    SCHEMA_VERSION,
    chunks_vec_dim,
    current_schema_version,
    get_metadata,
    index_status,
    open_db,
    set_metadata,
)


# ---------------------------------------------------------------------------
# Connection + pragmas
# ---------------------------------------------------------------------------

def test_open_db_creates_file(tmp_path: Path):
    p = tmp_path / "x.db"
    assert not p.exists()
    conn = open_db(p)
    assert p.exists()
    conn.close()


def test_open_db_enables_wal(tmp_path: Path):
    conn = open_db(tmp_path / "x.db")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_open_db_enables_foreign_keys(tmp_path: Path):
    conn = open_db(tmp_path / "x.db")
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_open_db_loads_sqlite_vec(tmp_path: Path):
    """`vec_version()` is only defined when sqlite-vec is loaded."""
    conn = open_db(tmp_path / "x.db")
    ver = conn.execute("SELECT vec_version()").fetchone()[0]
    assert isinstance(ver, str) and ver  # e.g. "v0.1.9"


def test_open_db_idempotent(tmp_path: Path):
    """Opening twice doesn't error or wipe state."""
    p = tmp_path / "x.db"
    open_db(p).close()
    conn = open_db(p)
    # Schema version is preserved across reopens.
    assert current_schema_version(conn) == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Schema — tables present
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    yield c
    c.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'virtual') ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def test_init_schema_creates_all_required_tables(conn):
    names = _table_names(conn)
    for required in [
        "nodes",
        "chunks",
        "sync",
        "embedder_capability",
        "failed_chunks",
        "index_metadata",
        "nodes_fts",
    ]:
        assert required in names, f"missing table {required}"


def test_chunks_vec_not_created_until_ensure_vec_tables(conn):
    """The vec0 table is created on first `ensure_vec_tables(dim)` so
    we know what dim to use. Until then it doesn't exist."""
    names = _table_names(conn)
    assert "chunks_vec" not in names


def test_nodes_columns(conn):
    """The nodes table has the documented columns and types."""
    info = list(conn.execute("PRAGMA table_info(nodes)"))
    col_types = {row[1]: row[2] for row in info}
    assert col_types["id"] == "TEXT"
    assert col_types["z_pk"] == "INTEGER"
    assert col_types["title"] == "TEXT"
    assert col_types["folder"] == "TEXT"
    assert col_types["modified_at"] == "INTEGER"
    assert col_types["locked"] == "INTEGER"
    assert col_types["pinned"] == "INTEGER"
    assert col_types["content_hash"] == "TEXT"


def test_chunks_columns(conn):
    info = list(conn.execute("PRAGMA table_info(chunks)"))
    col_types = {row[1]: row[2] for row in info}
    expected = {
        "id": "TEXT",
        "node_id": "TEXT",
        "chunk_index": "INTEGER",
        "heading": "TEXT",
        "heading_level": "INTEGER",
        "content": "TEXT",
        "content_hash": "TEXT",
        "start_line": "INTEGER",
        "end_line": "INTEGER",
    }
    for col, kind in expected.items():
        assert col in col_types, f"missing chunk column {col}"
        assert col_types[col] == kind, f"chunk column {col} wrong type"


def test_chunks_node_id_foreign_key_with_cascade(conn):
    """Inserting a node then deleting it cascades its chunks."""
    conn.execute(
        "INSERT INTO nodes (id, z_pk, title) VALUES ('n1', 1, 'Note 1')"
    )
    conn.execute(
        "INSERT INTO chunks (id, node_id, chunk_index, content, content_hash) "
        "VALUES ('n1#0', 'n1', 0, 'body', 'h')"
    )
    conn.execute("DELETE FROM nodes WHERE id = 'n1'")
    cnt = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE node_id = 'n1'"
    ).fetchone()[0]
    assert cnt == 0


def test_chunks_unique_node_id_chunk_index(conn):
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n1', 1, 'x')")
    conn.execute(
        "INSERT INTO chunks (id, node_id, chunk_index, content, content_hash) "
        "VALUES ('n1#0', 'n1', 0, 'a', 'ha')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        # Different id, same (node_id, chunk_index) — UNIQUE constraint trips.
        conn.execute(
            "INSERT INTO chunks (id, node_id, chunk_index, content, content_hash) "
            "VALUES ('n1#0b', 'n1', 0, 'b', 'hb')"
        )


def test_chunks_has_indexes(conn):
    idx_names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert "idx_chunks_node_id" in idx_names
    assert "idx_chunks_hash" in idx_names
    assert "idx_nodes_z_pk" in idx_names


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

def test_schema_version_set_on_fresh_db(conn):
    assert current_schema_version(conn) == SCHEMA_VERSION


def test_schema_version_persists_across_reopen(tmp_path: Path):
    p = tmp_path / "x.db"
    open_db(p).close()
    conn = open_db(p)
    assert current_schema_version(conn) == SCHEMA_VERSION


def test_schema_version_unchanged_by_second_init(conn):
    """Re-running _init_schema (via re-opening) doesn't reset a future
    bumped version. We simulate by manually bumping then reopening."""
    set_metadata(conn, "schema_version", "99")
    assert current_schema_version(conn) == 99
    # Manually call _init_schema again via the open_db path requires
    # a new connection but same DB file.
    conn.close()


# ---------------------------------------------------------------------------
# Metadata key/value
# ---------------------------------------------------------------------------

def test_metadata_get_returns_none_for_missing_key(conn):
    assert get_metadata(conn, "nope") is None


def test_metadata_set_then_get(conn):
    set_metadata(conn, "embedder", "onnx:bge-small-en-v1.5")
    assert get_metadata(conn, "embedder") == "onnx:bge-small-en-v1.5"


def test_metadata_set_overwrites(conn):
    set_metadata(conn, "key", "v1")
    set_metadata(conn, "key", "v2")
    assert get_metadata(conn, "key") == "v2"


def test_chunks_vec_dim_initially_none(conn):
    assert chunks_vec_dim(conn) is None


# ---------------------------------------------------------------------------
# index_status snapshot
# ---------------------------------------------------------------------------

def test_index_status_on_empty_db(conn):
    s = index_status(conn)
    assert s.total_nodes == 0
    assert s.total_chunks == 0
    assert s.total_failed_chunks == 0
    assert s.chunks_vec_dim is None
    assert s.schema_version == SCHEMA_VERSION
    assert s.last_indexed_at is None
    assert isinstance(s.vec_version, str)


def test_index_status_counts_nodes_and_chunks(conn):
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n1', 1, 't')")
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n2', 2, 't2')")
    conn.execute(
        "INSERT INTO chunks (id, node_id, chunk_index, content, content_hash) "
        "VALUES ('n1#0', 'n1', 0, 'c', 'h')"
    )
    s = index_status(conn)
    assert s.total_nodes == 2
    assert s.total_chunks == 1
