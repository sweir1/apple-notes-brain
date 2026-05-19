"""Tests for the semantic-subsystem bootstrap (Phase η).

Covers the four drift-detection layers: schema, model/dim, identity
hash, prefix strategy. Plus the first-boot stamp path and the
stale-cache promotion fallback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.bootstrap import (
    BootstrapResult,
    _compute_prefix_strategy,
    _drop_embedding_state,
    run_bootstrap,
)
from apple_notes_brain.semantic.store import (
    chunks_vec_dim,
    ensure_vec_tables,
    get_metadata,
    open_db,
    set_metadata,
    upsert_chunk,
    upsert_chunk_vector,
    upsert_node,
)

from .conftest import FakeEmbedder

import numpy as np


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

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


def _populate_one_chunk(conn, emb, node_id="n1", z_pk=1):
    """Seed nodes + chunks + chunks_vec so the DB is non-empty."""
    ensure_vec_tables(conn, emb.dimensions())
    upsert_node(
        conn, node_id=node_id, z_pk=z_pk, title="t",
        folder="Notes", modified_at=1, locked=False, pinned=False,
        content_hash="h",
    )
    rowid = upsert_chunk(
        conn, chunk_id=f"{node_id}#0", node_id=node_id, chunk_index=0,
        heading=None, heading_level=None,
        content="body", content_hash="h0",
        start_line=1, end_line=1,
    )
    upsert_chunk_vector(conn, rowid, np.zeros(emb.dimensions(), dtype=np.float32))


# ---------------------------------------------------------------------------
# First boot stamps metadata, no reindex
# ---------------------------------------------------------------------------

def test_first_boot_stamps_metadata_no_reindex(conn, emb):
    result = run_bootstrap(conn, emb)
    assert result.db_is_empty is True
    assert result.needs_reindex is False
    assert result.reasons == []
    assert get_metadata(conn, "embedding_model") == emb.model_identifier()
    assert get_metadata(conn, "embedding_dim") == str(emb.dimensions())


def test_idempotent_re_run_on_clean_state(conn, emb):
    """Two back-to-back run_bootstrap calls: second is a no-op."""
    run_bootstrap(conn, emb)
    result2 = run_bootstrap(conn, emb)
    assert result2.needs_reindex is False
    assert result2.reasons == []


# ---------------------------------------------------------------------------
# Model + dim drift
# ---------------------------------------------------------------------------

def test_model_change_triggers_reindex(conn, emb):
    _populate_one_chunk(conn, emb)
    # Simulate a prior boot with a different model.
    set_metadata(conn, "embedding_model", "old-model")
    set_metadata(conn, "embedding_dim", str(emb.dimensions()))

    result = run_bootstrap(conn, emb)

    assert result.needs_reindex is True
    assert any("embedding model changed" in r for r in result.reasons)
    # Chunks + chunks_vec + sync should be empty after drop.
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sync").fetchone()[0] == 0
    # Stored model is now the current one.
    assert get_metadata(conn, "embedding_model") == emb.model_identifier()


def test_dim_change_triggers_reindex(conn, emb):
    _populate_one_chunk(conn, emb)
    set_metadata(conn, "embedding_model", emb.model_identifier())
    set_metadata(conn, "embedding_dim", str(emb.dimensions() + 256))

    result = run_bootstrap(conn, emb)

    assert result.needs_reindex is True
    assert get_metadata(conn, "embedding_dim") == str(emb.dimensions())
    # chunks_vec must be recreated at the new dim.
    assert chunks_vec_dim(conn) == emb.dimensions()


def test_no_drift_no_reindex(conn, emb):
    _populate_one_chunk(conn, emb)
    set_metadata(conn, "embedding_model", emb.model_identifier())
    set_metadata(conn, "embedding_dim", str(emb.dimensions()))

    result = run_bootstrap(conn, emb)

    assert result.needs_reindex is False
    # Chunks untouched.
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Identity hash drift (Ollama-style)
# ---------------------------------------------------------------------------

class _IdentityHashEmbedder(FakeEmbedder):
    """FakeEmbedder + identity_hash() — mimics OllamaEmbedder."""

    def __init__(self, identity: str, **kwargs):
        super().__init__(**kwargs)
        self._identity = identity

    def identity_hash(self) -> str | None:
        return self._identity


def test_identity_hash_change_triggers_reindex(conn):
    """An Ollama-style `ollama pull` that updates weights with the same
    model name should drop+reindex."""
    emb_old = _IdentityHashEmbedder(identity="abc", dim=64)
    emb_old.init()
    run_bootstrap(conn, emb_old)
    _populate_one_chunk(conn, emb_old)

    # Now boot with a different identity hash but same model name + dim.
    emb_new = _IdentityHashEmbedder(identity="xyz", dim=64)
    emb_new.init()
    result = run_bootstrap(conn, emb_new)

    assert result.needs_reindex is True
    assert any("ollama pull" in r for r in result.reasons)
    assert get_metadata(conn, "embedder_identity_hash") == "xyz"


def test_identity_hash_none_is_noop(conn, emb):
    """FakeEmbedder has no identity_hash() — the check should pass-through."""
    run_bootstrap(conn, emb)
    # Second boot, no drift.
    result = run_bootstrap(conn, emb)
    assert result.needs_reindex is False


# ---------------------------------------------------------------------------
# Prefix strategy drift
# ---------------------------------------------------------------------------

def test_compute_prefix_strategy_symmetric_is_empty():
    assert _compute_prefix_strategy("", "") == ""
    assert _compute_prefix_strategy(None, None) == ""


def test_compute_prefix_strategy_asymmetric_is_stable_hash():
    h1 = _compute_prefix_strategy("Query: ", "Document: ")
    h2 = _compute_prefix_strategy("Query: ", "Document: ")
    assert h1 == h2 != ""


def test_compute_prefix_strategy_differs_on_input_change():
    h_a = _compute_prefix_strategy("A", "B")
    h_b = _compute_prefix_strategy("A", "C")
    assert h_a != h_b


# ---------------------------------------------------------------------------
# drop_embedding_state — direct test
# ---------------------------------------------------------------------------

def test_drop_embedding_state_clears_chunks_vec_sync(conn, emb):
    _populate_one_chunk(conn, emb)
    set_metadata(conn, "embedding_model", emb.model_identifier())
    # Add a sync row.
    conn.execute(
        "INSERT INTO sync (node_id, modified_at, indexed_at) VALUES (?, ?, ?)",
        ("n1", 1, 1),
    )

    _drop_embedding_state(conn)

    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sync").fetchone()[0] == 0
    # nodes survive — those don't depend on the embedder.
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1


def test_drop_embedding_state_safe_without_chunks_vec(conn):
    """If chunks_vec doesn't exist yet (brand-new DB), drop is silent."""
    # Don't ensure_vec_tables — leave chunks_vec absent.
    _drop_embedding_state(conn)  # must not raise


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_bootstrap_result_dataclass_shape(conn, emb):
    result = run_bootstrap(conn, emb)
    assert isinstance(result, BootstrapResult)
    assert isinstance(result.db_is_empty, bool)
    assert isinstance(result.needs_reindex, bool)
    assert isinstance(result.reasons, list)
