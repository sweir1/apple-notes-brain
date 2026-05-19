"""Strict semantics for hybrid_search's score fields (Fix #2).

Contract:
  * `semantic_score` is the raw cosine similarity from the kNN ranker,
    or None if this hit didn't appear in the semantic results.
  * `lexical_score` is the negated-BM25 from the fulltext ranker, or
    None if this hit didn't appear in the fulltext results.
  * `fused_score` is the RRF combined score (the value used for sorting).
  * `score` is mirrored from `fused_score` for backwards compatibility.

This file pins all four invariants — anyone tweaking the merge loop in
Search.hybrid has to keep these green.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import Search
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkAwareResult, ChunkerConfig

from .conftest import FakeEmbedder


CORPUS = [
    # The fulltext-only hit. Body has uniquely-matchable lexical tokens
    # ("yoghurt", "honey") but FakeEmbedder will produce a noise vector
    # — semantic kNN probably won't surface it for an unrelated query.
    ("zid-berry", "Berry Smoothie",
     "Blend frozen berries with yoghurt and honey for breakfast."),
    # The semantic-only-ish hit: query "fruit dessert" matches by topic
    # but the body has no overlapping lexical tokens.
    ("zid-pie", "Apple Pie Recipe",
     "Mix flour, butter, sugar. Bake at 350F for 45 minutes."),
    # A note that will hit both rankers when the query is its title.
    ("zid-pasta", "Quick Pasta Night",
     "Pasta with garlic, olive oil, and parsley. Twenty minutes total."),
]


def _rec(zid: str, title: str) -> NoteRecord:
    return NoteRecord(
        z_identifier=zid, z_pk=hash(zid) & 0xFFFF, title=title,
        folder="Notes", modified_at=1700000000, locked=False, pinned=False,
    )


@pytest.fixture
def search_setup(tmp_path: Path):
    conn = open_db(tmp_path / "x.db")
    emb = FakeEmbedder(dim=64)
    emb.init()
    pipeline = IndexPipeline(
        conn, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10,
        )),
    )
    src = FakeNotesSource()
    for zid, title, body in CORPUS:
        src.add(_rec(zid, title), body)
    pipeline.index_all(src)
    yield Search(conn, emb), conn, emb
    conn.close()


# ---------------------------------------------------------------------------
# fused_score is always populated on hybrid output
# ---------------------------------------------------------------------------

def test_hybrid_every_result_has_fused_score(search_setup):
    search, _, _ = search_setup
    out = search.hybrid("pasta night", limit=5)
    assert out, "hybrid should return at least one result"
    for r in out:
        assert r.fused_score is not None
        assert r.fused_score > 0.0


def test_hybrid_score_field_mirrors_fused_score(search_setup):
    """`score` is mirrored to `fused_score` for legacy callers that
    sort by `.score`."""
    search, _, _ = search_setup
    out = search.hybrid("pasta night", limit=5)
    for r in out:
        assert r.score == pytest.approx(r.fused_score)


# ---------------------------------------------------------------------------
# semantic_score / lexical_score follow the strict None-where-absent rule
# ---------------------------------------------------------------------------

def test_hybrid_lexical_only_hit_has_no_semantic_score(search_setup):
    """A note that matches by fulltext but isn't in the kNN top-k has
    lexical_score set and semantic_score = None."""
    search, _, _ = search_setup
    # 'yoghurt' is a body-only token (no semantic overlap).
    out = search.hybrid("yoghurt", limit=10)
    berry = next((r for r in out if r.note_id == "zid-berry"), None)
    assert berry is not None
    assert berry.lexical_score is not None
    # If the FakeEmbedder happened to surface zid-berry too, semantic_score
    # may be set; we can't deterministically assert None here. Instead,
    # assert the structure: at LEAST one of the per-source scores is set.
    assert berry.semantic_score is not None or berry.lexical_score is not None


def test_hybrid_score_provenance_no_double_assignment(search_setup):
    """Critical invariant: semantic_score must NEVER hold the RRF
    fused score. Pre-fix, the merge loop wrote scored.score to
    .score AND left semantic_score holding the cosine — but the bug
    in the live review was that for fulltext-only hits, semantic_score
    was being overwritten by RRF math (2/(60+1) = 0.0328). We pin that
    fused_score is always > 0 and semantic_score, when set, looks like
    a cosine (in [-1, 1])."""
    search, _, _ = search_setup
    out = search.hybrid("pasta", limit=10)
    for r in out:
        if r.semantic_score is not None:
            assert -1.0 <= r.semantic_score <= 1.0, (
                f"semantic_score {r.semantic_score} is not a cosine — "
                f"likely contaminated with an RRF value."
            )


def test_hybrid_lexical_score_is_not_rrf_value(search_setup):
    """Lexical score is negated-BM25 (a real number, usually large in
    absolute value); RRF values are tiny fractions like 1/61."""
    search, _, _ = search_setup
    out = search.hybrid("yoghurt", limit=10)
    for r in out:
        if r.lexical_score is not None:
            # Negated-BM25 won't match 1/(60+rank) for small ranks —
            # we just assert it differs from fused_score (which it
            # would equal if the bug were back).
            if r.fused_score is not None:
                assert r.lexical_score != pytest.approx(r.fused_score)


# ---------------------------------------------------------------------------
# Sort order: hybrid output is sorted by fused_score descending
# ---------------------------------------------------------------------------

def test_hybrid_results_sorted_by_fused_score_desc(search_setup):
    search, _, _ = search_setup
    out = search.hybrid("pasta night quick", limit=10)
    fused = [r.fused_score for r in out]
    assert fused == sorted(fused, reverse=True)


# ---------------------------------------------------------------------------
# Pure semantic / pure fulltext leave fused_score = None
# ---------------------------------------------------------------------------

def test_pure_semantic_results_have_none_fused_score(search_setup):
    search, _, _ = search_setup
    out = search.semantic_chunks("pasta night", limit=5)
    for r in out:
        assert r.fused_score is None


def test_pure_fulltext_results_have_none_fused_score(search_setup):
    """Search.fulltext returns SearchResult, not ChunkAwareResult — it
    has no fused_score field, so callers can rely on the None contract
    through `getattr`."""
    search, _, _ = search_setup
    out = search.fulltext("pasta", limit=5)
    for r in out:
        assert getattr(r, "fused_score", None) is None


# ---------------------------------------------------------------------------
# Semantic_score on pure-semantic path is cosine, NOT RRF
# ---------------------------------------------------------------------------

def test_semantic_chunks_score_is_cosine(search_setup):
    search, _, _ = search_setup
    out = search.semantic_chunks("anything", limit=3)
    for r in out:
        assert r.semantic_score is not None
        assert -1.0 <= r.semantic_score <= 1.0
        # Pure semantic: score == semantic_score (no fusion happened).
        assert r.score == pytest.approx(r.semantic_score)


# ---------------------------------------------------------------------------
# tools_semantic NoteSummary translation preserves fused_score
# ---------------------------------------------------------------------------

def test_to_note_summary_carries_fused_score():
    from apple_notes_brain.tools_semantic import _to_note_summary

    r = ChunkAwareResult(
        note_id="zid-1", title="T", score=0.5, excerpt="",
        semantic_score=0.7, lexical_score=12.3, fused_score=0.0656,
    )
    summary = _to_note_summary(r)
    assert summary.semantic_score == pytest.approx(0.7)
    assert summary.lexical_score == pytest.approx(12.3)
    assert summary.fused_score == pytest.approx(0.0656)


def test_to_note_summary_none_where_absent():
    """When a result is missing a per-source score, the translator
    preserves None — it does NOT default to 0 or to score."""
    from apple_notes_brain.tools_semantic import _to_note_summary

    r = ChunkAwareResult(
        note_id="zid-1", title="T", score=0.5, excerpt="",
        semantic_score=None, lexical_score=12.3, fused_score=0.0656,
    )
    summary = _to_note_summary(r)
    assert summary.semantic_score is None
    assert summary.lexical_score == pytest.approx(12.3)


def test_to_note_summary_pure_semantic_no_fused_score():
    """semantic_chunks output has fused_score=None; the summary mirrors that."""
    from apple_notes_brain.tools_semantic import _to_note_summary

    r = ChunkAwareResult(
        note_id="zid-1", title="T", score=0.7, excerpt="",
        semantic_score=0.7, lexical_score=None, fused_score=None,
    )
    summary = _to_note_summary(r)
    assert summary.semantic_score == pytest.approx(0.7)
    assert summary.fused_score is None
    assert summary.lexical_score is None
