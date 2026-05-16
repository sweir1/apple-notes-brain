"""MCP tool implementations for semantic + hybrid search.

This module is the thin shim between `server.py` (which only knows about
MCP tool registration) and the `semantic` subpackage (which does the
work). It owns three responsibilities:

  1. Lazy import: if `[semantic]` extras aren't installed, the semantic
     subpackage can't be imported. We catch that here so the rest of
     the server keeps working and the four new tools return a
     structured `missing-extras` error.
  2. State management: holds a singleton SemanticState (sqlite
     connection + embedder + IndexPipeline + Search) created on first
     use. Reset hook for tests.
  3. Tool surface: four functions matching the names server.py
     registers — semantic_search, hybrid_search, reindex_semantic,
     semantic_index_status.

We deliberately don't subclass MCP — the tool functions take plain
Python arguments and return Pydantic models (or a dict on errors).
server.py decorates them with @mcp.tool().
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from .schemas import NoteSummary, SearchPage

_log = logging.getLogger("apple-notes-brain")

# ---------------------------------------------------------------------------
# Lazy import shim — `[semantic]` extras may not be installed.
# ---------------------------------------------------------------------------

try:
    from .semantic.config import load_config
    from .semantic.embedder import create_embedder
    from .semantic.indexer import IndexPipeline, IndexerConfig
    from .semantic.search import Search
    from .semantic.source import AppleNotesSource, NotesSource
    from .semantic.store import index_status as store_index_status
    from .semantic.store import open_db
    from .semantic.types import ChunkerConfig, Embedder, IndexStats

    HAVE_SEMANTIC = True
    _IMPORT_ERROR: str | None = None
except ImportError as _imp_err:
    HAVE_SEMANTIC = False
    _IMPORT_ERROR = str(_imp_err)


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------

MISSING_EXTRAS_ERROR: dict[str, str] = {
    "error": (
        "semantic search requires the [semantic] install extra. "
        "Run `pip install apple-notes-brain[semantic]` (or "
        "`uv sync --extra semantic` for development)."
    ),
    "code": "missing-extras",
    "missing_import": _IMPORT_ERROR or "",
}


# ---------------------------------------------------------------------------
# State container — singleton (per-process) + reset hook for tests
# ---------------------------------------------------------------------------

class SemanticState:
    """Bundle of everything the semantic tools need. Constructed once
    (lazy), can be reset by tests."""

    def __init__(
        self,
        *,
        conn,
        embedder: "Embedder",
        indexer: "IndexPipeline",
        search: "Search",
        source: "NotesSource",
        config_snapshot,
    ):
        self.conn = conn
        self.embedder = embedder
        self.indexer = indexer
        self.search = search
        self.source = source
        self.config = config_snapshot

    def dispose(self) -> None:
        try:
            self.embedder.dispose()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass


_state: SemanticState | None = None


def get_state() -> SemanticState:
    """Construct or return the singleton SemanticState.

    First call:
      - load_config() → open_db() → create_embedder() → embedder.init()
      - build IndexPipeline + Search + AppleNotesSource
      - cache on _state

    Subsequent calls return the cached instance.
    """
    global _state
    if _state is not None:
        return _state
    if not HAVE_SEMANTIC:
        raise RuntimeError(
            "get_state() called without [semantic] extras installed. "
            "Tool callers must guard on HAVE_SEMANTIC."
        )
    cfg = load_config()
    conn = open_db(cfg.db_path)
    embedder = create_embedder(cfg)
    embedder.init()
    indexer = IndexPipeline(
        conn=conn,
        embedder=embedder,
        indexer_config=IndexerConfig(
            chunker_config=ChunkerConfig(
                chunk_size=(
                    int(cfg.max_chunk_tokens_override * 2.5)
                    if cfg.max_chunk_tokens_override
                    else ChunkerConfig().chunk_size
                ),
            ),
        ),
    )
    search = Search(conn, embedder)
    source = AppleNotesSource()
    _state = SemanticState(
        conn=conn,
        embedder=embedder,
        indexer=indexer,
        search=search,
        source=source,
        config_snapshot=cfg,
    )
    return _state


def reset_state_for_tests() -> None:
    """Test hook: dispose the singleton + drop the reference."""
    global _state
    if _state is not None:
        _state.dispose()
        _state = None


def set_state_for_tests(state: SemanticState) -> None:
    """Test hook: inject a custom state (with FakeEmbedder + FakeNotesSource)."""
    global _state
    _state = state


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _missing() -> dict[str, str]:
    return dict(MISSING_EXTRAS_ERROR)


def _to_note_summary(r) -> NoteSummary:
    """Translate a ChunkAwareResult / SearchResult into the NoteSummary
    envelope existing MCP clients already know how to render."""
    return NoteSummary(
        id=r.note_id,
        title=r.title,
        folder="",  # filled in below from the store if we have it
        modified="",
        snippets=[r.excerpt] if getattr(r, "excerpt", None) else [],
        match_count=0,
        body_preview=None,
        semantic_score=getattr(r, "semantic_score", None),
        lexical_score=getattr(r, "lexical_score", None),
        chunk_excerpt=getattr(r, "chunk_excerpt", None),
        chunk_heading=getattr(r, "chunk_heading", None),
    )


def _enrich_with_node_metadata(state: SemanticState, summaries: list[NoteSummary]) -> list[NoteSummary]:
    """Fill in folder + modified from the local nodes table.

    Both fields live in our semantic store (populated by the indexer)
    so we don't need to round-trip to NoteStore.sqlite for each hit.
    """
    if not summaries:
        return summaries
    from .semantic.store import get_node

    for s in summaries:
        node = get_node(state.conn, s.id)
        if node is None:
            continue
        s.folder = node.folder or ""
        if node.modified_at:
            s.modified = datetime.fromtimestamp(node.modified_at, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
        s.locked = bool(node.locked)
        s.pinned = bool(node.pinned)
    return summaries


def semantic_search(
    query: str,
    limit: int = 20,
    unique: Literal["notes", "chunks"] = "notes",
) -> SearchPage | dict[str, Any]:
    """Semantic chunk-level search. Returns the same envelope shape as
    `search_notes` so callers can swap one for the other."""
    if not HAVE_SEMANTIC:
        return _missing()
    if not query or not query.strip():
        return SearchPage(
            results=[], returned=0, has_more=False,
            next_cursor=None, total_estimate=0,
        )
    limit = max(1, min(int(limit), 100))
    state = get_state()
    hits = state.search.semantic_chunks(query, limit=limit, unique=unique)
    summaries = [_to_note_summary(h) for h in hits]
    _enrich_with_node_metadata(state, summaries)
    return SearchPage(
        results=summaries,
        returned=len(summaries),
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )


def hybrid_search(
    query: str,
    limit: int = 20,
    unique: Literal["notes", "chunks"] = "notes",
) -> SearchPage | dict[str, Any]:
    """RRF-fused semantic + lexical search. Higher-quality default for
    most queries — combines lexical precision with semantic recall."""
    if not HAVE_SEMANTIC:
        return _missing()
    if not query or not query.strip():
        return SearchPage(
            results=[], returned=0, has_more=False,
            next_cursor=None, total_estimate=0,
        )
    limit = max(1, min(int(limit), 100))
    state = get_state()
    hits = state.search.hybrid(query, limit=limit, unique=unique)
    summaries = [_to_note_summary(h) for h in hits]
    _enrich_with_node_metadata(state, summaries)
    return SearchPage(
        results=summaries,
        returned=len(summaries),
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )


def reindex_semantic(force: bool = False) -> dict[str, Any]:
    """Trigger a full index pass. Returns stats.

    `force=True` is reserved for a future 'drop everything and rebuild'
    path; today it's equivalent to the normal incremental pass because
    content-hash dedup already minimises unnecessary work.
    """
    if not HAVE_SEMANTIC:
        return _missing()
    state = get_state()
    stats = state.indexer.index_all(state.source)
    return {
        "notes_seen": stats.notes_seen,
        "notes_indexed": stats.notes_indexed,
        "notes_skipped": stats.notes_skipped,
        "notes_deleted": stats.notes_deleted,
        "chunks_embedded": stats.chunks_embedded,
        "chunks_skipped": stats.chunks_skipped,
        "chunks_failed": stats.chunks_failed,
        "took_ms": stats.took_ms,
        "failures": stats.failures[:20],  # cap surfaced failures
    }


def semantic_index_status() -> dict[str, Any]:
    """Snapshot of the index + embedder state."""
    if not HAVE_SEMANTIC:
        return _missing()
    state = get_state()
    s = store_index_status(state.conn)
    # Pull the active EP from the session for transparency.
    providers: list[str] = []
    try:
        sess = getattr(state.embedder, "_session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            providers = list(sess.get_providers())
    except Exception:
        providers = []
    return {
        "schema_version": s.schema_version,
        "total_nodes": s.total_nodes,
        "total_chunks": s.total_chunks,
        "total_failed_chunks": s.total_failed_chunks,
        "chunks_vec_dim": s.chunks_vec_dim,
        "last_indexed_at": s.last_indexed_at,
        "vec_version": s.vec_version,
        "embedder_provider": state.embedder.provider_name(),
        "embedder_model": state.embedder.model_identifier(),
        "embedder_dim": state.embedder.dimensions(),
        "onnx_providers": providers,
        "data_dir": str(state.config.data_dir),
        "db_path": str(state.config.db_path),
    }
