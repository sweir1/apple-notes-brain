"""Tests for the four MCP-facing semantic tool functions in `tools_semantic`.

State is injected via `set_state_for_tests` so the entire pipeline runs
against a FakeEmbedder + FakeNotesSource — no model download, no
NoteStore.sqlite. The shapes returned match what the MCP server will
expose.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain import tools_semantic
from apple_notes_brain.schemas import NoteSummary, SearchPage
from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
from apple_notes_brain.semantic.search import Search
from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord
from apple_notes_brain.semantic.store import open_db
from apple_notes_brain.semantic.types import ChunkerConfig

from .conftest import FakeEmbedder


# ---------------------------------------------------------------------------
# Per-test state injection
# ---------------------------------------------------------------------------

CORPUS = [
    ("zid-apple", "Apple Pie Recipe",
     "Mix flour, butter, and sugar. Bake the pie at 350F for 45 minutes."),
    ("zid-berry", "Berry Smoothie",
     "Blend frozen berries with yoghurt and honey for a quick breakfast."),
    ("zid-pasta", "Quick Pasta Night",
     "Pasta with garlic, olive oil, and parsley. Twenty minutes total."),
]


def _rec(zid: str, title: str) -> NoteRecord:
    return NoteRecord(
        z_identifier=zid, z_pk=hash(zid) & 0xFFFF, title=title,
        folder="Notes", modified_at=1700000000, locked=False, pinned=False,
    )


@pytest.fixture
def state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    cfg = load_config()
    conn = open_db(cfg.db_path)
    emb = FakeEmbedder(dim=64)
    emb.init()
    indexer = IndexPipeline(
        conn, emb,
        IndexerConfig(chunker_config=ChunkerConfig(
            chunk_size=200, min_chunk_chars=10, heading_split_depth=4,
        )),
    )
    src = FakeNotesSource()
    for zid, title, body in CORPUS:
        src.add(_rec(zid, title), body)
    indexer.index_all(src)
    search = Search(conn, emb)
    s = tools_semantic.SemanticState(
        conn=conn,
        embedder=emb,
        indexer=indexer,
        search=search,
        source=src,
        config_snapshot=cfg,
    )
    tools_semantic.set_state_for_tests(s)
    yield s
    tools_semantic.reset_state_for_tests()


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

def test_semantic_search_returns_search_page(state):
    out = tools_semantic.semantic_search("apple", limit=5)
    assert isinstance(out, SearchPage)
    assert out.returned == len(out.results)
    assert out.has_more is False
    assert out.next_cursor is None


def test_semantic_search_populates_semantic_score(state):
    out = tools_semantic.semantic_search("apple", limit=5)
    assert all(isinstance(r, NoteSummary) for r in out.results)
    assert any(r.semantic_score is not None for r in out.results)


def test_semantic_search_enriches_folder_and_modified(state):
    out = tools_semantic.semantic_search("apple", limit=5)
    # All seeded notes have folder='Notes' and modified_at=1700000000.
    for r in out.results:
        assert r.folder == "Notes"
        assert r.modified


def test_semantic_search_empty_query_returns_empty_page(state):
    out = tools_semantic.semantic_search("", limit=5)
    assert isinstance(out, SearchPage)
    assert out.results == []
    assert out.returned == 0


def test_semantic_search_limit_clamped_to_100(state):
    out = tools_semantic.semantic_search("anything", limit=999)
    # Just verify no crash + envelope shape.
    assert isinstance(out, SearchPage)


def test_semantic_search_limit_clamped_to_minimum_one(state):
    out = tools_semantic.semantic_search("anything", limit=0)
    assert isinstance(out, SearchPage)


def test_semantic_search_unique_chunks_returns_chunk_metadata(state):
    out = tools_semantic.semantic_search("apple pie", limit=5, unique="chunks")
    if out.results:
        assert any(r.chunk_excerpt for r in out.results)


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

def test_hybrid_search_returns_search_page(state):
    out = tools_semantic.hybrid_search("pasta", limit=5)
    assert isinstance(out, SearchPage)


def test_hybrid_search_finds_lexical_match(state):
    out = tools_semantic.hybrid_search("yoghurt", limit=5)
    ids = [r.id for r in out.results]
    assert "zid-berry" in ids


def test_hybrid_search_attaches_both_scores(state):
    out = tools_semantic.hybrid_search("pasta", limit=5)
    # At least one result should have lexical_score set (matched 'pasta').
    assert any(r.lexical_score is not None for r in out.results)


def test_hybrid_search_empty_query_returns_empty(state):
    out = tools_semantic.hybrid_search("", limit=5)
    assert out.results == []


# ---------------------------------------------------------------------------
# reindex_semantic
# ---------------------------------------------------------------------------

def test_reindex_returns_stats_dict(state):
    stats = tools_semantic.reindex_semantic()
    assert isinstance(stats, dict)
    for key in (
        "notes_seen", "notes_indexed", "notes_skipped", "notes_deleted",
        "chunks_embedded", "chunks_skipped", "chunks_failed", "took_ms",
        "failures",
    ):
        assert key in stats


def test_reindex_idempotent_via_content_hash(state):
    """Running reindex twice with no source changes: second pass embeds
    zero chunks because content_hash dedup catches everything."""
    first = tools_semantic.reindex_semantic()
    second = tools_semantic.reindex_semantic()
    # First pass already ran in the fixture so notes_indexed may be 3;
    # the second pass should still index notes but embed 0 chunks.
    assert second["chunks_embedded"] == 0


def test_reindex_force_flag_accepted(state):
    """`force=True` is accepted today even though dedup means it
    behaves the same as the default — surface area for a future
    drop-and-rebuild path."""
    stats = tools_semantic.reindex_semantic(force=True)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# semantic_index_status
# ---------------------------------------------------------------------------

def test_status_returns_keys(state):
    status = tools_semantic.semantic_index_status()
    expected_keys = {
        "schema_version", "total_nodes", "total_chunks",
        "total_failed_chunks", "chunks_vec_dim", "last_indexed_at",
        "vec_version", "embedder_provider", "embedder_model",
        "embedder_dim", "onnx_providers", "data_dir", "db_path",
    }
    assert expected_keys.issubset(status.keys())


def test_status_reports_indexed_corpus(state):
    status = tools_semantic.semantic_index_status()
    assert status["total_nodes"] == 3
    assert status["total_chunks"] >= 3
    assert status["embedder_provider"] == "fake"
    assert status["embedder_dim"] == 64


# ---------------------------------------------------------------------------
# Missing-extras error envelope (simulated via flag flip)
# ---------------------------------------------------------------------------

def test_missing_extras_envelope_shape(monkeypatch):
    """When [semantic] is missing, every tool returns the structured
    error envelope rather than crashing. We simulate via flag-flip
    (the real venv-without-extras case is covered by a separate
    integration test under tests/integration/semantic/)."""
    monkeypatch.setattr(tools_semantic, "HAVE_SEMANTIC", False)
    err1 = tools_semantic.semantic_search("query")
    err2 = tools_semantic.hybrid_search("query")
    err3 = tools_semantic.reindex_semantic()
    err4 = tools_semantic.semantic_index_status()
    for err in (err1, err2, err3, err4):
        assert isinstance(err, dict)
        assert err.get("code") == "missing-extras"
        assert "error" in err
        assert "[semantic]" in err["error"] or "semantic" in err["error"].lower()


def test_state_singleton_reused_between_calls(state):
    """Once initialised, the state survives across calls — no re-init,
    no re-download."""
    assert tools_semantic.get_state() is state
    assert tools_semantic.get_state() is state


# ---------------------------------------------------------------------------
# Schema extension
# ---------------------------------------------------------------------------

def test_note_summary_accepts_new_fields():
    n = NoteSummary(
        id="z", title="T", folder="Notes", modified="now",
        semantic_score=0.85, lexical_score=12.3,
        chunk_excerpt="...matched span...", chunk_heading="Section",
    )
    assert n.semantic_score == 0.85
    assert n.lexical_score == 12.3
    assert n.chunk_excerpt == "...matched span..."
    assert n.chunk_heading == "Section"


def test_note_summary_back_compat_no_semantic_fields():
    """Existing callers that don't pass the new fields still parse."""
    n = NoteSummary(id="z", title="T", folder="Notes", modified="now")
    assert n.semantic_score is None
    assert n.lexical_score is None
    assert n.chunk_excerpt is None
    assert n.chunk_heading is None
    assert n.fused_score is None


# ---------------------------------------------------------------------------
# v1.1 reindex force=True clears failed_chunks
# ---------------------------------------------------------------------------

def test_reindex_force_clears_failed_chunks(state):
    """force=True should drop every row in failed_chunks at the start."""
    from apple_notes_brain.semantic.store import record_failed_chunk, count_failed_chunks

    # Seed a few stale failures.
    for i in range(5):
        record_failed_chunk(
            state.conn,
            chunk_id=f"zid-stale-{i}#0",
            node_id=f"zid-stale-{i}",
            reason="locked",
            error_message="stale failure",
        )
    assert count_failed_chunks(state.conn) == 5

    stats = tools_semantic.reindex_semantic(force=True)
    assert stats["failed_chunks_cleared"] == 5
    # After the pass, no stale failures remain (the indexed corpus has no
    # locked notes either, so the count is 0).
    assert count_failed_chunks(state.conn) == 0


def test_reindex_default_does_not_clear_failed_chunks(state):
    """force=False (default) must leave the failed_chunks table alone."""
    from apple_notes_brain.semantic.store import record_failed_chunk, count_failed_chunks

    record_failed_chunk(
        state.conn,
        chunk_id="zid-stale#0",
        node_id="zid-stale",
        reason="locked",
        error_message="stale failure",
    )
    assert count_failed_chunks(state.conn) == 1

    stats = tools_semantic.reindex_semantic()  # force defaults to False
    assert stats.get("failed_chunks_cleared", 0) == 0
    # Row still present.
    assert count_failed_chunks(state.conn) == 1


def test_reindex_force_with_empty_table_returns_zero_cleared(state):
    from apple_notes_brain.semantic.store import count_failed_chunks

    assert count_failed_chunks(state.conn) == 0
    stats = tools_semantic.reindex_semantic(force=True)
    assert stats["failed_chunks_cleared"] == 0


def test_reindex_force_then_status_shows_zero_failed(state):
    """End-to-end: stale failures, force-reindex, status shows clean."""
    from apple_notes_brain.semantic.store import record_failed_chunk

    for i in range(3):
        record_failed_chunk(
            state.conn,
            chunk_id=f"chunk-{i}", node_id=f"node-{i}",
            reason="locked", error_message="x",
        )
    status_before = tools_semantic.semantic_index_status()
    assert status_before["total_failed_chunks"] == 3

    tools_semantic.reindex_semantic(force=True)
    status_after = tools_semantic.semantic_index_status()
    assert status_after["total_failed_chunks"] == 0
    assert status_after["failed_chunk_ids"] == []


# ---------------------------------------------------------------------------
# semantic_index_status — new fields
# ---------------------------------------------------------------------------

def test_status_includes_embedder_warm(state):
    """The state singleton was injected by the fixture, so
    embedder_warm is True on every call from inside this test."""
    status = tools_semantic.semantic_index_status()
    assert "embedder_warm" in status
    assert status["embedder_warm"] is True


def test_status_includes_failed_chunk_ids(state):
    status = tools_semantic.semantic_index_status()
    assert "failed_chunk_ids" in status
    assert isinstance(status["failed_chunk_ids"], list)


def test_status_failed_chunk_ids_truncated_at_50():
    """When >50 failures exist, the list is capped and the last entry
    is suffixed with ' (truncated)'."""
    # Build a fresh state isolated from the shared fixture (which
    # currently injects 0 failures).
    from pathlib import Path
    import tempfile

    from apple_notes_brain.semantic.config import load_config
    from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
    from apple_notes_brain.semantic.search import Search
    from apple_notes_brain.semantic.source import FakeNotesSource
    from apple_notes_brain.semantic.store import open_db, record_failed_chunk
    from apple_notes_brain.semantic.types import ChunkerConfig

    with tempfile.TemporaryDirectory() as td:
        import os

        os.environ["APPLE_NOTES_BRAIN_DATA_DIR"] = td
        try:
            cfg = load_config()
            conn = open_db(cfg.db_path)
            emb = FakeEmbedder(dim=64)
            emb.init()
            indexer = IndexPipeline(
                conn, emb, IndexerConfig(chunker_config=ChunkerConfig(
                    chunk_size=200, min_chunk_chars=10,
                )),
            )
            # Build chunks_vec table by running an empty index pass.
            indexer.prepare()
            for i in range(60):
                record_failed_chunk(
                    conn, chunk_id=f"chunk-{i:03d}",
                    node_id=f"node-{i:03d}",
                    reason="locked", error_message="x",
                )
            search = Search(conn, emb)
            src = FakeNotesSource()
            s = tools_semantic.SemanticState(
                conn=conn, embedder=emb, indexer=indexer,
                search=search, source=src, config_snapshot=cfg,
            )
            tools_semantic.set_state_for_tests(s)

            status = tools_semantic.semantic_index_status()
            assert status["total_failed_chunks"] == 60
            assert len(status["failed_chunk_ids"]) == 50
            assert status["failed_chunk_ids"][-1].endswith(" (truncated)")
        finally:
            tools_semantic.reset_state_for_tests()
            os.environ.pop("APPLE_NOTES_BRAIN_DATA_DIR", None)


def test_status_failed_chunk_ids_empty_when_none_recorded(state):
    status = tools_semantic.semantic_index_status()
    assert status["failed_chunk_ids"] == []


def test_status_embedder_warm_false_before_first_state_construction(monkeypatch, tmp_path):
    """The first call to semantic_index_status (or any tool) constructs
    the state. We exercise that by resetting the singleton FIRST and
    then asserting embedder_warm reports False on the construction call."""
    # Reset shared singleton.
    tools_semantic.reset_state_for_tests()

    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    # Mock create_embedder to return a FakeEmbedder so we don't trigger
    # an ONNX model download in CI.
    from apple_notes_brain.semantic import embedder as embedder_module
    from apple_notes_brain import tools_semantic as ts_module

    def _fake_factory(cfg):
        e = FakeEmbedder(dim=64)
        return e

    monkeypatch.setattr(ts_module, "create_embedder", _fake_factory)

    # Now call status — this is the FIRST tool call, so embedder_warm
    # should report False (the snapshot happens before get_state runs).
    status = tools_semantic.semantic_index_status()
    assert status["embedder_warm"] is False
    # The cached state is now ready; a second call reports True.
    status2 = tools_semantic.semantic_index_status()
    assert status2["embedder_warm"] is True

    tools_semantic.reset_state_for_tests()


# ---------------------------------------------------------------------------
# Trash plumbing — semantic_search / hybrid_search include_trash
# ---------------------------------------------------------------------------

def test_semantic_search_default_excludes_trash(state):
    """End-to-end: a previously-indexed note that's now in trash should
    NOT appear in default semantic_search output. We force the note
    into the index via include_trash=True, then verify the tool drops it."""
    from apple_notes_brain.semantic.source import NoteRecord

    # Add a trash note via include_trash=True on the indexer to bypass
    # the index-time filter.
    trash_rec = NoteRecord(
        z_identifier="zid-tomb", z_pk=999, title="Tombstone",
        folder="Recently Deleted", modified_at=1700000000,
        locked=False, pinned=False,
    )
    state.source.add(trash_rec, "Pasta with garlic, olive oil — old recipe.")
    state.indexer.index_all(state.source, include_trash=True)
    # Default semantic_search must NOT return the trash hit.
    out = tools_semantic.semantic_search("pasta garlic olive", limit=10)
    ids = [r.id for r in out.results]
    assert "zid-tomb" not in ids


def test_semantic_search_include_trash_true_returns_trash(state):
    from apple_notes_brain.semantic.source import NoteRecord

    trash_rec = NoteRecord(
        z_identifier="zid-tomb", z_pk=999, title="Tombstone",
        folder="Recently Deleted", modified_at=1700000000,
        locked=False, pinned=False,
    )
    state.source.add(trash_rec, "Pasta with garlic, olive oil — old recipe.")
    state.indexer.index_all(state.source, include_trash=True)
    out = tools_semantic.semantic_search(
        "pasta garlic olive", limit=10, include_trash=True,
    )
    ids = [r.id for r in out.results]
    assert "zid-tomb" in ids


def test_hybrid_search_default_excludes_trash(state):
    from apple_notes_brain.semantic.source import NoteRecord

    trash_rec = NoteRecord(
        z_identifier="zid-tomb", z_pk=999, title="Tombstone",
        folder="Recently Deleted", modified_at=1700000000,
        locked=False, pinned=False,
    )
    state.source.add(trash_rec, "Pasta with garlic, olive oil — old recipe.")
    state.indexer.index_all(state.source, include_trash=True)
    out = tools_semantic.hybrid_search("pasta garlic olive", limit=10)
    ids = [r.id for r in out.results]
    assert "zid-tomb" not in ids


def test_hybrid_search_results_carry_fused_score(state):
    """hybrid_search results expose fused_score on the NoteSummary."""
    out = tools_semantic.hybrid_search("pasta", limit=5)
    assert isinstance(out, SearchPage)
    assert any(r.fused_score is not None for r in out.results)


def test_hybrid_search_results_sorted_by_fused_score(state):
    out = tools_semantic.hybrid_search("pasta", limit=10)
    fused = [r.fused_score or 0.0 for r in out.results]
    assert fused == sorted(fused, reverse=True)


def test_semantic_search_match_count_is_one(state):
    """Per the v1.1 contract change, semantic hits set match_count=1."""
    out = tools_semantic.semantic_search("apple", limit=5)
    for r in out.results:
        assert r.match_count == 1
