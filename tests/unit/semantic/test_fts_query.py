"""Tests for `search_full_text` against a real sqlite-with-FTS5 store."""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.fts import search_full_text
from apple_notes_brain.semantic.store import open_db, upsert_node


@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    # Seed a small corpus.
    upsert_node(
        c, node_id="apple", z_pk=1, title="Apple Pie Recipe",
        folder=None, modified_at=None, locked=False, pinned=False,
        content_hash=None, body_text="Mix flour and sugar. Bake the pie.",
    )
    upsert_node(
        c, node_id="berry", z_pk=2, title="Berry Smoothie Notes",
        folder=None, modified_at=None, locked=False, pinned=False,
        content_hash=None, body_text="Blend frozen berries with yoghurt.",
    )
    upsert_node(
        c, node_id="pasta", z_pk=3, title="Quick Pasta",
        folder=None, modified_at=None, locked=False, pinned=False,
        content_hash=None,
        body_text="Pasta with garlic, olive oil, parsley. Twenty minutes.",
    )
    yield c
    c.close()


def test_query_with_no_match_returns_empty(conn):
    assert search_full_text(conn, "doesnotexist", 10) == []


def test_query_matches_title(conn):
    hits = search_full_text(conn, "apple", 10)
    assert any(h.node_id == "apple" for h in hits)


def test_query_matches_body(conn):
    hits = search_full_text(conn, "yoghurt", 10)
    assert any(h.node_id == "berry" for h in hits)


def test_title_hits_outrank_body_hits(conn):
    """`pasta` is in the TITLE of "Quick Pasta" and in the BODY of itself —
    a title-only hit elsewhere should outrank a body-only hit because of
    the BM25 column weights (title 5x)."""
    upsert_node(
        conn, node_id="title-pasta", z_pk=10, title="Pasta",
        folder=None, modified_at=None, locked=False, pinned=False,
        content_hash=None, body_text="(no pasta here)",
    )
    upsert_node(
        conn, node_id="body-pasta", z_pk=11, title="(none)",
        folder=None, modified_at=None, locked=False, pinned=False,
        content_hash=None, body_text="just a body that mentions pasta",
    )
    hits = search_full_text(conn, "pasta", 10)
    title_pasta = next((h for h in hits if h.node_id == "title-pasta"), None)
    body_pasta = next((h for h in hits if h.node_id == "body-pasta"), None)
    assert title_pasta is not None and body_pasta is not None
    assert title_pasta.score > body_pasta.score


def test_empty_query_returns_empty(conn):
    assert search_full_text(conn, "", 10) == []
    assert search_full_text(conn, "   ", 10) == []


def test_snippet_marks_match(conn):
    hits = search_full_text(conn, "berries", 10)
    matched = [h for h in hits if h.node_id == "berry"]
    assert matched and ">>>" in matched[0].excerpt or "<<<" in matched[0].excerpt


def test_limit_respected(conn):
    """With three notes containing 'a' (very common), the limit applies."""
    hits = search_full_text(conn, "the", 1)
    assert len(hits) <= 1


def test_results_returned_with_score_descending(conn):
    """Internally bm25 is ascending; we negate so the returned list is
    sorted by score DESCENDING (higher = better)."""
    hits = search_full_text(conn, "the OR a", 10)
    if len(hits) >= 2:
        # Scores monotonic descending (we sort by bm25 ASC, negate, so
        # output is implicitly desc by score).
        for i in range(len(hits) - 1):
            assert hits[i].score >= hits[i + 1].score - 1e-6
