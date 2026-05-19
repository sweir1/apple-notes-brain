"""Tests for ``semantic.metadata_cache``.

The cache is a thin SQL wrapper over the v7 columns on
``embedder_capability``. The tests pin the load/upsert/invalidate
contract, especially the v6→v7 promotion semantics (rows with
``fetched_at IS NULL`` are misses, not bad data).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from apple_notes_brain.semantic import store
from apple_notes_brain.semantic.metadata_cache import (
    CachedMetadata,
    invalidate_cache,
    load_cached,
    upsert_cache,
)

from .conftest import FakeEmbedder


@pytest.fixture
def db(tmp_path: Path):
    """Open a fresh, schema-initialised SQLite DB for each test."""
    conn = store.open_db(tmp_path / "cache.db")
    yield conn
    conn.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(model_id="fake/cache-test")


def _make_meta(**overrides) -> CachedMetadata:
    base = dict(
        model_id="fake/cache-test",
        dim=384,
        max_tokens=512,
        query_prefix="query: ",
        document_prefix="passage: ",
        prefix_source="seed",
        base_model=None,
        size_bytes=None,
        fetched_at=int(time.time()),
    )
    base.update(overrides)
    return CachedMetadata(**base)


# ─── load_cached ────────────────────────────────────────────────────────


def test_load_cached_returns_none_when_table_empty(db, embedder):
    assert load_cached(db, embedder) is None


def test_load_cached_returns_none_when_other_embedder_present(db, embedder):
    other = FakeEmbedder(model_id="different/model")
    upsert_cache(db, other, _make_meta(model_id="different/model"))
    assert load_cached(db, embedder) is None


def test_load_cached_v6_row_is_miss(db, embedder):
    """A row written by the v6 capacity helper has fetched_at NULL and
    no v7 columns populated; the resolver re-fills it."""
    # Simulate a pre-resolver row.
    db.execute(
        """
        INSERT INTO embedder_capability
          (embedder_id, model_hash, advertised_max_tokens)
        VALUES (?, ?, ?)
        """,
        (embedder.model_identifier(), embedder.model_identifier(), 512),
    )
    assert load_cached(db, embedder) is None


def test_load_cached_hits_after_upsert(db, embedder):
    meta = _make_meta()
    upsert_cache(db, embedder, meta)
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.model_id == embedder.model_identifier()
    assert loaded.dim == 384
    assert loaded.max_tokens == 512
    assert loaded.query_prefix == "query: "
    assert loaded.document_prefix == "passage: "
    assert loaded.prefix_source == "seed"
    assert loaded.fetched_at is not None


def test_load_cached_preserves_null_fields(db, embedder):
    """A real HF-resolved row may have base_model=None / size_bytes=None."""
    meta = _make_meta(base_model=None, size_bytes=None)
    upsert_cache(db, embedder, meta)
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.base_model is None
    assert loaded.size_bytes is None


def test_load_cached_returns_empty_string_prefixes_distinctly_from_null(db, embedder):
    """An asymmetric model can have query_prefix='query: ' AND
    document_prefix='' (e.g. instruction-only prepending). The empty
    string must round-trip distinctly from None."""
    meta = _make_meta(query_prefix="Instruct: ", document_prefix="")
    upsert_cache(db, embedder, meta)
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.query_prefix == "Instruct: "
    assert loaded.document_prefix == ""


# ─── upsert_cache ───────────────────────────────────────────────────────


def test_upsert_inserts_then_updates(db, embedder):
    upsert_cache(db, embedder, _make_meta(max_tokens=512))
    upsert_cache(db, embedder, _make_meta(max_tokens=1024))
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.max_tokens == 1024


def test_upsert_writes_v7_columns(db, embedder):
    upsert_cache(db, embedder, _make_meta(prefix_source="metadata", dim=768))
    row = db.execute(
        "SELECT dim, prefix_source, query_prefix, fetched_at "
        "FROM embedder_capability WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    ).fetchone()
    assert row is not None
    assert row[0] == 768
    assert row[1] == "metadata"
    assert row[2] == "query: "
    assert row[3] is not None


def test_upsert_seeds_discovered_max_tokens_on_insert(db, embedder):
    """First write seeds discovered_max_tokens to the advertised value
    so the capacity ratchet has a starting point."""
    upsert_cache(db, embedder, _make_meta(max_tokens=512))
    row = db.execute(
        "SELECT advertised_max_tokens, discovered_max_tokens "
        "FROM embedder_capability WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    ).fetchone()
    assert row[0] == 512
    assert row[1] == 512


def test_upsert_preserves_discovered_capacity_on_update(db, embedder):
    """The v6 columns (discovered_max_tokens) shouldn't be reset by a
    metadata-cache update — adaptive capacity is independent of metadata."""
    upsert_cache(db, embedder, _make_meta(max_tokens=512))
    # Simulate the ratchet kicking in.
    db.execute(
        "UPDATE embedder_capability SET discovered_max_tokens = 384 "
        "WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    )
    # Now bump the advertised value through the metadata cache.
    upsert_cache(db, embedder, _make_meta(max_tokens=1024))
    row = db.execute(
        "SELECT advertised_max_tokens, discovered_max_tokens "
        "FROM embedder_capability WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    ).fetchone()
    assert row[0] == 1024
    assert row[1] == 384  # preserved


def test_upsert_stamps_fetched_at_when_none(db, embedder):
    """Passing fetched_at=None lets the upsert stamp int(time.time())."""
    before = int(time.time()) - 1
    upsert_cache(db, embedder, _make_meta(fetched_at=None))
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.fetched_at is not None
    assert loaded.fetched_at >= before


def test_upsert_respects_explicit_fetched_at(db, embedder):
    explicit = 1_700_000_000
    upsert_cache(db, embedder, _make_meta(fetched_at=explicit))
    loaded = load_cached(db, embedder)
    assert loaded is not None
    assert loaded.fetched_at == explicit


# ─── invalidate_cache ───────────────────────────────────────────────────


def test_invalidate_single_clears_v7_columns(db, embedder):
    upsert_cache(db, embedder, _make_meta())
    assert load_cached(db, embedder) is not None
    rows = invalidate_cache(db, embedder)
    assert rows >= 1
    assert load_cached(db, embedder) is None


def test_invalidate_preserves_v6_columns(db, embedder):
    upsert_cache(db, embedder, _make_meta(max_tokens=512))
    db.execute(
        "UPDATE embedder_capability SET discovered_max_tokens = 256 "
        "WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    )
    invalidate_cache(db, embedder)
    row = db.execute(
        "SELECT advertised_max_tokens, discovered_max_tokens, dim, query_prefix "
        "FROM embedder_capability WHERE embedder_id = ?",
        (embedder.model_identifier(),),
    ).fetchone()
    # v6 columns preserved; v7 columns NULLed.
    assert row[0] == 512
    assert row[1] == 256
    assert row[2] is None
    assert row[3] is None


def test_invalidate_all_clears_every_row(db):
    a = FakeEmbedder(model_id="fake/a")
    b = FakeEmbedder(model_id="fake/b")
    upsert_cache(db, a, _make_meta(model_id="fake/a"))
    upsert_cache(db, b, _make_meta(model_id="fake/b"))
    invalidate_cache(db, embedder=None)
    assert load_cached(db, a) is None
    assert load_cached(db, b) is None


def test_invalidate_missing_row_is_noop(db, embedder):
    """No row → invalidate returns 0; doesn't raise."""
    rows = invalidate_cache(db, embedder)
    assert rows == 0
