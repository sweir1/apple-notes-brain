"""Query-time trash-folder filter on Search (defence-in-depth).

The primary filter is at index time (NotesSource.iter_notes). This file
exercises the secondary filter that drops trash-folder hits from kNN
output even when a stale index still contains them — e.g. a user runs
v1.0 of the index, upgrades to v1.1, and queries before re-indexing.

We populate the store DIRECTLY (bypassing the indexer's trash filter)
to simulate the stale-index scenario, then verify Search drops those
hits at query time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import (
    Search,
    _is_trash_folder,
    _TRASH_FOLDER_NAMES,
)
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkerConfig

from .conftest import FakeEmbedder


def _rec(zid: str, folder: str, body_seed: str) -> NoteRecord:
    return NoteRecord(
        z_identifier=zid, z_pk=hash(zid) & 0xFFFF,
        title=body_seed.split()[0].capitalize() if body_seed else zid,
        folder=folder, modified_at=1700000000, locked=False, pinned=False,
    )


@pytest.fixture
def stale_index(tmp_path: Path):
    """Build an index where one note lives in 'Recently Deleted'.

    We pass include_trash=True so the indexer doesn't filter — this is
    the stale-index simulation. Then we hand the connection to Search.
    """
    conn = open_db(tmp_path / "x.db")
    emb = FakeEmbedder(dim=64)
    emb.init()
    pipeline = IndexPipeline(
        conn, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10, heading_split_depth=4,
        )),
    )
    src = FakeNotesSource()
    src.add(_rec("zid-live", "Notes",
                 "suspension shocks coilovers tuning notes"),
            "suspension shocks coilovers tuning notes for offroad rigs")
    src.add(_rec("zid-trash", "Recently Deleted",
                 "suspension shocks coilovers tuning"),
            "suspension shocks coilovers tuning trashed long ago by accident")
    pipeline.index_all(src, include_trash=True)
    search = Search(conn, emb)
    yield search, conn, emb
    conn.close()


# ---------------------------------------------------------------------------
# Unit: _is_trash_folder helper
# ---------------------------------------------------------------------------

def test_is_trash_folder_recognises_recently_deleted():
    assert _is_trash_folder("Recently Deleted") is True


def test_is_trash_folder_treats_none_as_live():
    assert _is_trash_folder(None) is False


def test_is_trash_folder_treats_empty_as_live():
    assert _is_trash_folder("") is False


def test_is_trash_folder_case_sensitive():
    """English-locale macOS uses exactly 'Recently Deleted'; lowered
    or hyphenated variants are NOT considered trash (would catch a
    user-created folder named 'recently-deleted')."""
    assert _is_trash_folder("recently deleted") is False
    assert _is_trash_folder("Recently-Deleted") is False


def test_trash_folder_names_constant_is_immutable():
    assert isinstance(_TRASH_FOLDER_NAMES, frozenset)
    assert "Recently Deleted" in _TRASH_FOLDER_NAMES


# ---------------------------------------------------------------------------
# Semantic — defence-in-depth
# ---------------------------------------------------------------------------

def test_semantic_chunks_excludes_trash_by_default(stale_index):
    search, _, _ = stale_index
    out = search.semantic_chunks("suspension shocks coilovers", limit=10)
    ids = {r.note_id for r in out}
    assert "zid-trash" not in ids
    # zid-live should still surface.
    assert "zid-live" in ids


def test_semantic_chunks_include_trash_true_exposes_trash(stale_index):
    search, _, _ = stale_index
    out = search.semantic_chunks(
        "suspension shocks coilovers", limit=10, include_trash=True,
    )
    ids = {r.note_id for r in out}
    assert "zid-trash" in ids
    assert "zid-live" in ids


def test_semantic_top_level_helper_respects_include_trash(stale_index):
    search, _, _ = stale_index
    out_default = search.semantic("suspension shocks", limit=10)
    out_include = search.semantic("suspension shocks", limit=10, include_trash=True)
    assert "zid-trash" not in {r.note_id for r in out_default}
    assert "zid-trash" in {r.note_id for r in out_include}


def test_semantic_chunks_only_trash_index_returns_empty_by_default(tmp_path: Path):
    """When the entire index is trash, the default query returns nothing
    rather than leaking everything."""
    conn = open_db(tmp_path / "y.db")
    emb = FakeEmbedder(dim=64)
    emb.init()
    pipeline = IndexPipeline(
        conn, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10,
        )),
    )
    src = FakeNotesSource()
    src.add(_rec("zid-1", "Recently Deleted", "abc def"),
            "trash body content here longer than min chunk size.")
    pipeline.index_all(src, include_trash=True)
    search = Search(conn, emb)
    assert search.semantic_chunks("abc def", limit=10) == []


# ---------------------------------------------------------------------------
# Fulltext — defence-in-depth
# ---------------------------------------------------------------------------

def test_fulltext_excludes_trash_by_default(stale_index):
    search, _, _ = stale_index
    out = search.fulltext("coilovers", limit=10)
    ids = {r.note_id for r in out}
    assert "zid-trash" not in ids
    assert "zid-live" in ids


def test_fulltext_include_trash_true_exposes_trash(stale_index):
    search, _, _ = stale_index
    out = search.fulltext("coilovers", limit=10, include_trash=True)
    ids = {r.note_id for r in out}
    assert "zid-trash" in ids


# ---------------------------------------------------------------------------
# Hybrid — defence-in-depth
# ---------------------------------------------------------------------------

def test_hybrid_excludes_trash_by_default(stale_index):
    search, _, _ = stale_index
    out = search.hybrid("suspension shocks coilovers", limit=10)
    ids = {r.note_id for r in out}
    assert "zid-trash" not in ids


def test_hybrid_include_trash_true_exposes_trash(stale_index):
    search, _, _ = stale_index
    out = search.hybrid(
        "suspension shocks coilovers", limit=10, include_trash=True,
    )
    ids = {r.note_id for r in out}
    assert "zid-trash" in ids
