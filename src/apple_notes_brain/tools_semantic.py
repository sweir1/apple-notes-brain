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
    from .semantic._logging import debug_log, setup_logging
    from .semantic.config import load_config
    from .semantic.embedder import create_embedder
    from .semantic.indexer import IndexPipeline, IndexerConfig
    from .semantic.search import Search
    from .semantic.source import AppleNotesSource, NotesSource
    from .semantic.store import (
        clear_failed_chunks,
        index_status as store_index_status,
        list_failed_chunk_ids,
        open_db,
    )
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
        # v1.1 Phase ζ — boot lifecycle tracking. Either string phase
        # (pending → embedder-init → bootstrap → indexing → ready) or
        # 'failed' with init_error populated.
        self.boot_phase: str = "ready"  # default for synchronous get_state()
        self.init_error: BaseException | None = None

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
    # Wire debug logging FIRST so subsequent steps can emit DEBUG-level
    # lines. Idempotent — safe to call from every entry point.
    setup_logging()
    debug_log("semantic: initialising state singleton")
    cfg = load_config()
    conn = open_db(cfg.db_path)
    embedder = create_embedder(cfg)
    embedder.init()
    _log.info(
        "semantic: embedder ready: provider=%s model=%s dim=%d",
        embedder.provider_name(),
        embedder.model_identifier(),
        embedder.dimensions(),
    )
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
    _log.info("semantic: state singleton ready")
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


def _to_note_summary(state: "SemanticState", r) -> NoteSummary:
    """Translate a ChunkAwareResult / SearchResult into the NoteSummary
    envelope existing MCP clients already know how to render.

    The semantic store keys notes by ZIDENTIFIER (a UUID), but the rest
    of the MCP tool surface (search_notes, get_note, etc.) keys by the
    short `pN` form. We translate here so MCP clients can round-trip
    results between the lexical and semantic families. The original
    ZIDENTIFIER stays on the `z_identifier` field for callers that need
    cross-session stability.

    Score-field semantics (v1.1):
      * `semantic_score` is copied directly from the source ranker — None
        if this hit didn't appear in the semantic results.
      * `lexical_score` is copied directly from the source ranker — None
        if this hit didn't appear in the fulltext results.
      * `fused_score` is the RRF combined score for hybrid results; None
        for pure-semantic / pure-lexical paths.
      * `match_count` is set to 1 for semantic / hybrid hits — the
        lexical search tool overloads it with a token-hit count, but
        for chunk-level / cosine matching it's tautological.
    """
    from .semantic.store import get_node

    z_identifier = r.note_id
    id_str = z_identifier
    try:
        node = get_node(state.conn, z_identifier)
        if node is not None:
            id_str = f"p{node.z_pk}"
    except Exception:
        # If the lookup fails we fall back to the ZIDENTIFIER so the
        # caller still gets *something* it can correlate against.
        pass
    return NoteSummary(
        id=id_str,
        title=r.title,
        folder="",  # filled in below from the store if we have it
        modified="",
        snippets=[r.excerpt] if getattr(r, "excerpt", None) else [],
        match_count=1,
        body_preview=None,
        semantic_score=getattr(r, "semantic_score", None),
        lexical_score=getattr(r, "lexical_score", None),
        fused_score=getattr(r, "fused_score", None),
        chunk_excerpt=getattr(r, "chunk_excerpt", None),
        chunk_heading=getattr(r, "chunk_heading", None),
        z_identifier=z_identifier,
    )


def _enrich_with_node_metadata(state: SemanticState, summaries: list[NoteSummary]) -> list[NoteSummary]:
    """Fill in folder + modified from the local nodes table.

    Both fields live in our semantic store (populated by the indexer)
    so we don't need to round-trip to NoteStore.sqlite for each hit.

    Lookup keys on `z_identifier` (the UUID) when available — `id` now
    carries the short `pN` form which the nodes table doesn't index.
    """
    if not summaries:
        return summaries
    from .semantic.store import get_node

    for s in summaries:
        lookup_key = s.z_identifier or s.id
        node = get_node(state.conn, lookup_key)
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


_EMPTY_INDEX_HINT = (
    "Semantic index is empty. Call `reindex_semantic` or wait for the "
    "background indexer to finish — see `semantic_index_status` for progress."
)


def _empty_index_hint(state: SemanticState) -> str | None:
    """Return the empty-index advisory if total_chunks == 0, else None.

    Callers use this to populate `SearchPage.hint` when results are empty
    so MCP clients can distinguish "no matches" from "index not built
    yet" without a second tool round-trip.
    """
    try:
        status = store_index_status(state.conn)
        if status.total_chunks == 0:
            return _EMPTY_INDEX_HINT
    except Exception:
        # Status lookup shouldn't ever fail, but if it does we'd rather
        # return no hint than crash the search response.
        return None
    return None


def semantic_search(
    query: str,
    limit: int = 20,
    unique: Literal["notes", "chunks"] = "notes",
) -> SearchPage | dict[str, Any]:
    """Semantic chunk-level search. Returns the same envelope shape as
    `search_notes` so callers can swap one for the other.

    Notes in Recently Deleted (trash) are never returned. They aren't
    indexed in the first place — even if you trash a note that was
    previously indexed, the next pass removes its chunks from the store.
    Use `search_notes(include_trash=True)` if you need to grep trash for
    a recoverable note (lexical only).
    """
    if not HAVE_SEMANTIC:
        return _missing()
    if not query or not query.strip():
        return SearchPage(
            results=[], returned=0, has_more=False,
            next_cursor=None, total_estimate=0,
        )
    limit = max(1, min(int(limit), 100))
    state = get_state()
    hits = state.search.semantic_chunks(
        query, limit=limit, unique=unique,
    )
    summaries = [_to_note_summary(state, h) for h in hits]
    _enrich_with_node_metadata(state, summaries)
    hint = _empty_index_hint(state) if not summaries else None
    return SearchPage(
        results=summaries,
        returned=len(summaries),
        has_more=False,
        next_cursor=None,
        total_estimate=None,
        hint=hint,
    )


def hybrid_search(
    query: str,
    limit: int = 20,
    unique: Literal["notes", "chunks"] = "notes",
) -> SearchPage | dict[str, Any]:
    """RRF-fused semantic + lexical search. Higher-quality default for
    most queries — combines lexical precision with semantic recall.

    Notes in Recently Deleted (trash) are never returned (same policy
    as `semantic_search`).

    Each result's `semantic_score`, `lexical_score`, and `fused_score`
    fields carry strict provenance:
      * `semantic_score` is set when (and only when) the hit appeared
        in the kNN ranker output — i.e. it was a semantic match.
      * `lexical_score` is set when (and only when) the hit appeared
        in the fulltext ranker output — i.e. it was a BM25 match.
      * `fused_score` is the RRF combined score and is set on every
        hybrid result.
    """
    if not HAVE_SEMANTIC:
        return _missing()
    if not query or not query.strip():
        return SearchPage(
            results=[], returned=0, has_more=False,
            next_cursor=None, total_estimate=0,
        )
    limit = max(1, min(int(limit), 100))
    state = get_state()
    hits = state.search.hybrid(
        query, limit=limit, unique=unique,
    )
    summaries = [_to_note_summary(state, h) for h in hits]
    _enrich_with_node_metadata(state, summaries)
    # Final defensive sort: order by fused_score descending so the
    # response respects the documented RRF ordering even if a future
    # caller appends entries to `summaries` out of order.
    summaries.sort(
        key=lambda s: -(
            s.fused_score if s.fused_score is not None
            else (s.semantic_score or s.lexical_score or 0.0)
        )
    )
    hint = _empty_index_hint(state) if not summaries else None
    return SearchPage(
        results=summaries,
        returned=len(summaries),
        has_more=False,
        next_cursor=None,
        total_estimate=None,
        hint=hint,
    )


def reindex_semantic(force: bool = False) -> dict[str, Any]:
    """Trigger a full index pass. Returns stats.

    `force=True` clears the persistent `failed_chunks` table at the
    start of the pass so that previously-recorded failures don't keep
    inflating `semantic_index_status.total_failed_chunks` after the
    underlying issue has been resolved. Without `force`, the table is
    left intact (incremental passes only add entries on new failure).

    `prior_failures_cleared` in the response counts the real failures
    cleared by `force=True` (too-long, embed-error). Locked-note
    placeholders are also wiped under the hood but they re-create on
    the next pass and don't count here.
    """
    if not HAVE_SEMANTIC:
        return _missing()
    state = get_state()
    cleared = 0
    if force:
        cleared = clear_failed_chunks(state.conn)
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
        "prior_failures_cleared": cleared,
    }


def semantic_index_status() -> dict[str, Any]:
    """Snapshot of the index + embedder state.

    Includes `embedder_warm: bool` — True iff the embedder has been
    initialised (i.e. the SemanticState singleton already existed
    before this call). Callers can use this to warn users that the
    first query will be slow while the ONNX runtime spins up.

    `locked_notes` counts password-protected notes that can't be
    indexed without unlocking — this is expected, not a failure.
    `total_failed_chunks` only counts real failures (`too-long`,
    `embed-error`); `failed_chunks_by_reason` breaks them down.

    Also includes `failed_chunk_ids: list[str]` — the IDs of up to 50
    real-failure chunks, most-recent first. Locked-note placeholders
    are excluded from this list (they correspond to `locked_notes`).
    If `total_failed_chunks > 50`, the list is truncated and the final
    entry is suffixed with ' (truncated)'.
    """
    if not HAVE_SEMANTIC:
        return _missing()
    # Snapshot warmth BEFORE get_state() runs init() so the very first
    # call reports embedder_warm=False (matches the documented contract
    # — "first query will be slow").
    warm = _state is not None
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
    failed_ids = list_failed_chunk_ids(state.conn, limit=50)
    if s.total_failed_chunks > 50 and failed_ids:
        failed_ids = list(failed_ids)
        failed_ids[-1] = failed_ids[-1] + " (truncated)"
    return {
        "schema_version": s.schema_version,
        "total_nodes": s.total_nodes,
        "total_chunks": s.total_chunks,
        "total_failed_chunks": s.total_failed_chunks,
        "failed_chunks_by_reason": s.failed_chunks_by_reason,
        "locked_notes": s.locked_notes,
        "failed_chunk_ids": failed_ids,
        "chunks_vec_dim": s.chunks_vec_dim,
        "last_indexed_at": s.last_indexed_at,
        "vec_version": s.vec_version,
        "embedder_provider": state.embedder.provider_name(),
        "embedder_model": state.embedder.model_identifier(),
        "embedder_dim": state.embedder.dimensions(),
        "embedder_warm": warm,
        "onnx_providers": providers,
        "data_dir": str(state.config.data_dir),
        "db_path": str(state.config.db_path),
    }
