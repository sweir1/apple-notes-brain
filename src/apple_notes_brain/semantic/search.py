"""Search class — semantic, fulltext, and hybrid (RRF fused).

Mirrors `obsidian-brain/src/search/unified.ts`. The API splits along
two axes:
  * granularity — note-level (`semantic`, `fulltext`) vs chunk-aware
    (`semantic_chunks`, `hybrid`)
  * scoring — semantic (cosine), fulltext (negated BM25), hybrid (RRF)

`reciprocal_rank_fusion(lists, key_fn, k=60)` is exposed standalone so
the unit tests can pin the formula without spinning up a Search.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .fts import search_full_text
from .store import search_chunk_vectors
from .types import (
    ChunkAwareResult,
    Embedder,
    SearchResult,
    SearchUnique,
)

# Folder names treated as trash for the query-time defence-in-depth
# filter. Apple's English-locale localised name is the only one we
# can identify without joining NoteStore.sqlite at query time (which
# we deliberately don't do — Search runs against the semantic store
# only). Non-English locales remain on the index-time filter path.
_TRASH_FOLDER_NAMES: frozenset[str] = frozenset({"Recently Deleted"})


def _is_trash_folder(folder: str | None) -> bool:
    """Return True iff the folder name matches a known trash folder.

    None / empty strings are treated as live (a note with no folder
    isn't in trash). Heuristic-only; the source-level filter is the
    primary line of defence.
    """
    if not folder:
        return False
    return folder in _TRASH_FOLDER_NAMES


# ---------------------------------------------------------------------------
# RRF — standalone for testability
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class RRFScored(Generic[T]):
    item: T
    score: float


def reciprocal_rank_fusion(
    lists: list[list[T]], key_fn: Callable[[T], str], k: int = 60
) -> list[RRFScored[T]]:
    """Cormack/Clarke/Büttcher 2009 fusion.

    For each ranked list, item at zero-indexed rank `i` contributes
    1 / (k + i + 1) to its identity score. Identities are normalised
    by `key_fn`. The output is sorted descending by accumulated score.

    Empty lists are tolerated (they contribute nothing). When all
    inputs are empty the result is [].
    """
    scores: dict[str, float] = defaultdict(float)
    representative: dict[str, T] = {}
    for ranked in lists:
        for idx, item in enumerate(ranked):
            key = key_fn(item)
            scores[key] += 1.0 / (k + idx + 1)
            representative.setdefault(key, item)
    return sorted(
        (RRFScored(item=representative[key], score=s) for key, s in scores.items()),
        key=lambda x: -x.score,
    )


# ---------------------------------------------------------------------------
# Search class
# ---------------------------------------------------------------------------

class Search:
    """High-level query API. Stateless beyond (conn, embedder)."""

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder):
        self._conn = conn
        self._embedder = embedder

    # -- Semantic -------------------------------------------------------

    def semantic(
        self, query: str, limit: int = 20, *, include_trash: bool = False
    ) -> list[SearchResult]:
        """Note-level semantic search. Dedups multi-chunk hits per note,
        keeping the best-scoring chunk for each."""
        chunks = self.semantic_chunks(
            query, limit=limit, unique="notes", include_trash=include_trash
        )
        # ChunkAwareResult already has note-level fields populated.
        return [
            SearchResult(
                note_id=r.note_id, title=r.title, score=r.score,
                excerpt=r.excerpt, folder=r.folder,
            )
            for r in chunks
        ]

    def semantic_chunks(
        self,
        query: str,
        limit: int = 20,
        unique: SearchUnique = "notes",
        *,
        include_trash: bool = False,
    ) -> list[ChunkAwareResult]:
        """Chunk-grained semantic search.

        When unique='notes', results are deduplicated to one per note
        (the highest-scoring chunk wins). When unique='chunks', every
        matched chunk is its own result row.

        `include_trash=False` (the default) is a defence-in-depth filter
        that drops any kNN hit whose `note_folder` is recognised as a
        trash folder. The primary defence is in the indexer (trash notes
        never get embedded). This filter is here so a stale index that
        still contains a trash-folder hit doesn't leak it to callers.
        """
        if not query or not query.strip():
            return []
        vec = self._embedder.embed(query, task_type="query")
        # Over-fetch to leave headroom for dedup + trash filtering.
        # Trash hits will be silently dropped, so we pad by a constant
        # to keep the post-filter result count near the requested limit.
        trash_overhead = 0 if include_trash else 16
        if unique == "notes":
            raw_limit = limit * 4 + trash_overhead
        else:
            raw_limit = limit + trash_overhead
        hits = search_chunk_vectors(self._conn, vec, raw_limit)
        if not hits:
            return []
        # Defence-in-depth: filter trash-folder hits at query time.
        if not include_trash:
            hits = [h for h in hits if not _is_trash_folder(h.note_folder)]
        if not hits:
            return []
        results: list[ChunkAwareResult] = []
        if unique == "chunks":
            for h in hits[:limit]:
                results.append(
                    ChunkAwareResult(
                        note_id=h.node_id,
                        title=h.note_title,
                        score=h.score,
                        excerpt=h.content[:200],
                        folder=h.note_folder,
                        chunk_id=h.chunk_id,
                        chunk_heading=h.heading,
                        chunk_start_line=h.start_line,
                        chunk_end_line=h.end_line,
                        chunk_excerpt=h.content[:200],
                        semantic_score=h.score,
                    )
                )
            return results
        # unique='notes': keep best chunk per note.
        best_by_note: dict[str, ChunkAwareResult] = {}
        for h in hits:
            candidate = ChunkAwareResult(
                note_id=h.node_id,
                title=h.note_title,
                score=h.score,
                excerpt=h.content[:200],
                folder=h.note_folder,
                chunk_id=h.chunk_id,
                chunk_heading=h.heading,
                chunk_start_line=h.start_line,
                chunk_end_line=h.end_line,
                chunk_excerpt=h.content[:200],
                semantic_score=h.score,
            )
            existing = best_by_note.get(h.node_id)
            if existing is None or candidate.score > existing.score:
                best_by_note[h.node_id] = candidate
        # Sort by score descending and apply limit.
        ranked = sorted(best_by_note.values(), key=lambda r: -r.score)
        return ranked[:limit]

    # -- Fulltext -------------------------------------------------------

    def fulltext(
        self, query: str, limit: int = 20, *, include_trash: bool = False
    ) -> list[SearchResult]:
        """BM25 search over `nodes_fts`. Returns note-level hits.

        `include_trash=False` drops fulltext hits whose folder is a
        recognised trash folder — same defence-in-depth as the semantic
        path.
        """
        if not query or not query.strip():
            return []
        # Over-fetch when trash-filtering so a fully-stale index doesn't
        # collapse to an empty result list.
        trash_overhead = 0 if include_trash else 16
        rows = search_full_text(self._conn, query, limit=limit + trash_overhead)
        if not include_trash:
            rows = [r for r in rows if not _is_trash_folder(r.folder)]
        rows = rows[:limit]
        return [
            SearchResult(
                note_id=r.node_id, title=r.title, score=r.score,
                excerpt=r.excerpt, folder=r.folder,
            )
            for r in rows
        ]

    # -- Hybrid (RRF) ---------------------------------------------------

    def hybrid(
        self,
        query: str,
        limit: int = 20,
        unique: SearchUnique = "notes",
        *,
        include_trash: bool = False,
    ) -> list[ChunkAwareResult]:
        """Reciprocal-rank-fused semantic + fulltext results.

        Both sub-queries over-fetch by 4x so RRF has signal to work with
        even when one source has very few hits.

        Score-field semantics (post v1.1 fix):
          * `semantic_score` = raw cosine similarity from the kNN ranker,
            or None if this doc only matched via fulltext.
          * `lexical_score` = negated-BM25 from the fulltext ranker,
            or None if this doc only matched via the kNN ranker.
          * `fused_score` = the RRF sum (the value used for ranking).
          * `score` = `fused_score` (so existing sort-by-score callers
            still see the fused ordering).
        """
        if not query or not query.strip():
            return []
        overfetch = max(limit * 4, 20)
        semantic_hits = self.semantic_chunks(
            query, limit=overfetch, unique="chunks", include_trash=include_trash,
        )
        fulltext_hits = self.fulltext(
            query, limit=overfetch, include_trash=include_trash
        )
        # Build a note-id keyed map of semantic + fulltext scores so we
        # can stamp both onto the merged record.
        sem_score_by_note: dict[str, ChunkAwareResult] = {}
        for h in semantic_hits:
            existing = sem_score_by_note.get(h.note_id)
            if existing is None or h.score > existing.score:
                sem_score_by_note[h.note_id] = h
        lex_score_by_note: dict[str, SearchResult] = {
            h.note_id: h for h in fulltext_hits
        }
        # RRF over the two ranked lists, keyed by note_id (when unique='notes')
        # or chunk_id (when unique='chunks').
        if unique == "chunks":
            # For chunk-level fusion, fulltext (which is note-level) needs
            # to be projected onto its best matching chunk. We keep it
            # simple: only the semantic hits' chunk-ids participate; the
            # fulltext list contributes via note_id matching to bump
            # those same chunks. Equivalent to obsidian-brain's strategy.
            chunk_list = semantic_hits
            note_list_as_chunk_keys = [
                # Map each fulltext hit onto the chunk-of-this-note from
                # the semantic list (if any) so RRF can fuse on chunk_id.
                ChunkAwareResult(
                    note_id=lex.note_id, title=lex.title, score=lex.score,
                    excerpt=lex.excerpt, folder=lex.folder,
                    chunk_id=(
                        sem_score_by_note[lex.note_id].chunk_id
                        if lex.note_id in sem_score_by_note else None
                    ),
                    chunk_heading=None,
                    chunk_start_line=None,
                    chunk_end_line=None,
                    chunk_excerpt=lex.excerpt,
                    semantic_score=None,
                    lexical_score=lex.score,
                )
                for lex in fulltext_hits
            ]
            fused = reciprocal_rank_fusion(
                [chunk_list, note_list_as_chunk_keys],
                key_fn=lambda r: r.chunk_id or f"_note_only::{r.note_id}",
            )
        else:
            # Note-level fusion is the simpler, default path.
            # Project both lists onto ChunkAwareResults keyed by note_id.
            sem_proj = list(sem_score_by_note.values())
            lex_proj = [
                ChunkAwareResult(
                    note_id=h.note_id, title=h.title, score=h.score,
                    excerpt=h.excerpt, folder=h.folder,
                    chunk_id=None, chunk_heading=None,
                    chunk_start_line=None, chunk_end_line=None,
                    chunk_excerpt=h.excerpt,
                    semantic_score=None, lexical_score=h.score,
                )
                for h in fulltext_hits
            ]
            fused = reciprocal_rank_fusion(
                [sem_proj, lex_proj], key_fn=lambda r: r.note_id
            )

        out: list[ChunkAwareResult] = []
        seen_keys: set[str] = set()
        for scored in fused:
            r = scored.item
            key = r.chunk_id if unique == "chunks" and r.chunk_id else r.note_id
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # Strict score-provenance semantics:
            #   semantic_score := raw cosine from semantic ranker (None
            #     if this doc didn't appear in the semantic results).
            #   lexical_score  := negated-BM25 from fulltext ranker
            #     (None if this doc didn't appear in the fulltext results).
            #   fused_score    := RRF sum (always set for hybrid output).
            #   score          := fused_score (preserved as the ordering
            #     key so existing `sort by -r.score` consumers Just Work).
            r.semantic_score = (
                sem_score_by_note[r.note_id].score
                if r.note_id in sem_score_by_note else None
            )
            r.lexical_score = (
                lex_score_by_note[r.note_id].score
                if r.note_id in lex_score_by_note else None
            )
            r.fused_score = scored.score
            r.score = scored.score
            out.append(r)
            if len(out) >= limit:
                break
        # Defensive resort by fused_score descending — the RRF helper
        # already returns its output sorted, but downstream readers
        # shouldn't rely on iteration order matching RRF order if the
        # underlying merge ever changes.
        out.sort(key=lambda r: -(r.fused_score or r.score))
        return out
