"""Tests for vector storage + kNN lookups in the store.

Uses random unit vectors for the round-trip and a hand-built corpus of
3 vectors for the MATCH-ordering test. No FakeEmbedder here — we want
to confirm raw sqlite-vec behaviour, not the embedder chain.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from apple_notes_brain.semantic.store import (
    DimensionMismatchError,
    chunks_vec_dim,
    delete_chunks_for_node,
    delete_node,
    ensure_vec_tables,
    open_db,
    search_chunk_vectors,
    upsert_chunk,
    upsert_chunk_vector,
    upsert_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_corpus(conn: sqlite3.Connection, dim: int = 8) -> dict[str, int]:
    """Insert 3 hand-built vectors at known unit directions so kNN ordering
    is deterministic. Returns {chunk_id: rowid}.

    Vectors:
      A: [1, 0, 0, ...]
      B: [0, 1, 0, ...]
      C: [1/√2, 1/√2, 0, ...]   (close to both A and B)
    """
    ensure_vec_tables(conn, dim)
    upsert_node(
        conn, node_id="na", z_pk=1, title="A",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    upsert_node(
        conn, node_id="nb", z_pk=2, title="B",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )
    upsert_node(
        conn, node_id="nc", z_pk=3, title="C",
        folder=None, modified_at=None, locked=False, pinned=False, content_hash=None,
    )

    def _vec(values: list[float]) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for i, x in enumerate(values):
            v[i] = x
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 0 else v

    out: dict[str, int] = {}
    for note, vals in [("na", [1, 0, 0]), ("nb", [0, 1, 0]), ("nc", [1, 1, 0])]:
        rowid = upsert_chunk(
            conn,
            chunk_id=f"{note}#0",
            node_id=note,
            chunk_index=0,
            heading=None,
            heading_level=None,
            content=f"content {note}",
            content_hash="h",
            start_line=1,
            end_line=1,
        )
        upsert_chunk_vector(conn, rowid, _vec(vals))
        out[f"{note}#0"] = rowid
    return out


@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# ensure_vec_tables — dim lifecycle
# ---------------------------------------------------------------------------

def test_ensure_vec_tables_creates_at_dim(conn):
    ensure_vec_tables(conn, 16)
    assert chunks_vec_dim(conn) == 16
    # Verify the table actually accepts dim-16 vectors.
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('x', 1, 't')")
    rowid = upsert_chunk(
        conn, chunk_id="x#0", node_id="x", chunk_index=0,
        heading=None, heading_level=None,
        content="c", content_hash="h", start_line=1, end_line=1,
    )
    upsert_chunk_vector(conn, rowid, np.ones(16, dtype=np.float32) / 4.0)


def test_ensure_vec_tables_noop_when_dim_matches(conn):
    ensure_vec_tables(conn, 16)
    ensure_vec_tables(conn, 16)  # should be a no-op, not raise
    assert chunks_vec_dim(conn) == 16


def test_ensure_vec_tables_recreates_when_empty_dim_mismatch(conn):
    ensure_vec_tables(conn, 16)
    ensure_vec_tables(conn, 32)
    assert chunks_vec_dim(conn) == 32


def test_ensure_vec_tables_refuses_when_rows_present(conn):
    _seed_corpus(conn, dim=8)
    with pytest.raises(DimensionMismatchError, match="dim=8"):
        ensure_vec_tables(conn, 16)


# ---------------------------------------------------------------------------
# Vector round-trip
# ---------------------------------------------------------------------------

def test_vector_round_trip_preserves_dim(conn):
    ensure_vec_tables(conn, 4)
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n', 1, 't')")
    rowid = upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading=None, heading_level=None,
        content="c", content_hash="h", start_line=1, end_line=1,
    )
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    upsert_chunk_vector(conn, rowid, v)
    # The bytes survive a round-trip.
    payload = conn.execute(
        "SELECT embedding FROM chunks_vec WHERE rowid = ?", (rowid,)
    ).fetchone()[0]
    recovered = np.frombuffer(payload, dtype=np.float32)
    np.testing.assert_array_equal(recovered, v)


def test_upsert_chunk_vector_casts_to_float32(conn):
    """Passing a float64 array works — store casts internally."""
    ensure_vec_tables(conn, 4)
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n', 1, 't')")
    rowid = upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading=None, heading_level=None,
        content="c", content_hash="h", start_line=1, end_line=1,
    )
    v = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    upsert_chunk_vector(conn, rowid, v)
    payload = conn.execute(
        "SELECT embedding FROM chunks_vec WHERE rowid = ?", (rowid,)
    ).fetchone()[0]
    recovered = np.frombuffer(payload, dtype=np.float32)
    np.testing.assert_allclose(recovered, v.astype(np.float32))


def test_upsert_chunk_vector_replaces_on_same_rowid(conn):
    ensure_vec_tables(conn, 4)
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n', 1, 't')")
    rowid = upsert_chunk(
        conn, chunk_id="n#0", node_id="n", chunk_index=0,
        heading=None, heading_level=None,
        content="c", content_hash="h", start_line=1, end_line=1,
    )
    upsert_chunk_vector(conn, rowid, np.ones(4, dtype=np.float32) / 2.0)
    upsert_chunk_vector(conn, rowid, np.array([1, 0, 0, 0], dtype=np.float32))
    cnt = conn.execute(
        "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (rowid,)
    ).fetchone()[0]
    assert cnt == 1


# ---------------------------------------------------------------------------
# search_chunk_vectors — kNN ordering
# ---------------------------------------------------------------------------

def test_knn_returns_results_sorted_by_distance(conn):
    _seed_corpus(conn, dim=8)
    # Query close to "A" direction.
    query = np.zeros(8, dtype=np.float32)
    query[0] = 1.0
    results = search_chunk_vectors(conn, query, limit=3)
    assert len(results) == 3
    # A should be the top hit; then C (cosine ≈ 0.707); then B.
    ordered = [r.chunk_id for r in results]
    assert ordered[0] == "na#0"
    assert ordered[1] == "nc#0"
    assert ordered[2] == "nb#0"


def test_knn_score_is_one_minus_distance(conn):
    _seed_corpus(conn, dim=8)
    query = np.zeros(8, dtype=np.float32)
    query[0] = 1.0
    results = search_chunk_vectors(conn, query, limit=3)
    top = results[0]
    assert top.score == pytest.approx(1.0, abs=1e-5)


def test_knn_respects_limit(conn):
    _seed_corpus(conn, dim=8)
    query = np.zeros(8, dtype=np.float32)
    query[0] = 1.0
    assert len(search_chunk_vectors(conn, query, limit=1)) == 1
    assert len(search_chunk_vectors(conn, query, limit=2)) == 2


def test_knn_returns_note_metadata(conn):
    _seed_corpus(conn, dim=8)
    query = np.zeros(8, dtype=np.float32)
    query[0] = 1.0
    top = search_chunk_vectors(conn, query, limit=1)[0]
    assert top.note_title == "A"
    assert top.chunk_index == 0
    assert top.content == "content na"


def test_knn_empty_db_returns_empty_list(conn):
    """Asking sqlite-vec for kNN against an empty vec table — must not crash."""
    from apple_notes_brain.semantic.store import ensure_vec_tables
    ensure_vec_tables(conn, 8)
    out = search_chunk_vectors(conn, np.ones(8, dtype=np.float32) / 8**0.5, limit=10)
    assert out == []


# ---------------------------------------------------------------------------
# Cascade deletes propagate to chunks_vec
# ---------------------------------------------------------------------------

def test_delete_node_cascades_to_chunks_and_vectors(conn):
    rowids = _seed_corpus(conn, dim=8)
    delete_node(conn, "na")
    # nodes row gone.
    assert conn.execute(
        "SELECT 1 FROM nodes WHERE id = 'na'"
    ).fetchone() is None
    # chunks rows gone via FK cascade.
    assert conn.execute(
        "SELECT 1 FROM chunks WHERE node_id = 'na'"
    ).fetchone() is None
    # chunks_vec rows for those rowids gone too.
    assert conn.execute(
        "SELECT 1 FROM chunks_vec WHERE rowid = ?", (rowids["na#0"],)
    ).fetchone() is None


def test_delete_chunks_for_node_keep_indices(conn):
    """Delete every chunk for a node *except* those in keep_indices."""
    ensure_vec_tables(conn, 4)
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n', 1, 't')")
    for i in range(3):
        rowid = upsert_chunk(
            conn, chunk_id=f"n#{i}", node_id="n", chunk_index=i,
            heading=None, heading_level=None,
            content=f"c{i}", content_hash=f"h{i}", start_line=1, end_line=1,
        )
        upsert_chunk_vector(conn, rowid, np.eye(4, dtype=np.float32)[i % 4])

    removed = delete_chunks_for_node(conn, "n", keep_indices={1})
    assert removed == 2
    remaining = [r[0] for r in conn.execute(
        "SELECT chunk_index FROM chunks WHERE node_id = 'n' ORDER BY chunk_index"
    )]
    assert remaining == [1]


def test_delete_chunks_for_node_keep_indices_empty_removes_all(conn):
    ensure_vec_tables(conn, 4)
    conn.execute("INSERT INTO nodes (id, z_pk, title) VALUES ('n', 1, 't')")
    for i in range(2):
        upsert_chunk(
            conn, chunk_id=f"n#{i}", node_id="n", chunk_index=i,
            heading=None, heading_level=None,
            content="c", content_hash="h", start_line=1, end_line=1,
        )
    removed = delete_chunks_for_node(conn, "n", keep_indices=set())
    assert removed == 2
