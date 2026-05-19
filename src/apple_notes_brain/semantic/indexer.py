"""Indexing pipeline.

Glues together the source, chunker, embedder, capacity ratchet, and
store. Two entry points: `index_all()` for full passes (idempotent
modulo content-hash dedup) and `index_single()` for the watcher.

Mirrors obsidian-brain's `src/pipeline/indexer/index.ts` algorithm:
  - iterate source
  - per-note: skip if (locked) → fallback chunk for empty body; otherwise
    chunk → check content-hash → embed dirty chunks
  - on TooLongError: log to failed_chunks, ratchet capacity, continue
  - on EmbedderDeadError: abort the pass, don't corrupt partial state
  - on any other exception: log to failed_chunks with reason='embed-error'

Stats are returned to the caller for telemetry; `failed_chunks` table
keeps a longer-lived record so the status MCP tool can surface 'we
silently failed 3 chunks last pass' visibly.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass

from .capacity import (
    approx_tokens_for,
    get_capacity,
    initialise_capacity,
    reduce_discovered_max_tokens,
)
from .chunker import build_chunk_embedding_text, chunk_id, chunk_markdown
from .source import NoteRecord, NotesSource
from .store import (
    delete_chunks_for_node,
    delete_node,
    ensure_vec_tables,
    get_chunk,
    record_failed_chunk,
    set_metadata,
    set_sync,
    upsert_chunk,
    upsert_chunk_vector,
    upsert_node,
)
from .types import (
    Chunk,
    ChunkerConfig,
    DEFAULT_CHUNKER_CONFIG,
    Embedder,
    EmbedderDeadError,
    IndexStats,
    SingleNoteResult,
    TooLongError,
)

_log = logging.getLogger("apple-notes-brain")


@dataclass(frozen=True)
class IndexerConfig:
    """Indexer-specific tuning (chunker config is separate)."""
    advertised_max_tokens: int = 512
    chunker_config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG


class IndexPipeline:
    """The indexing loop. Owns no state beyond what's passed in; the
    db connection + embedder are injected so callers can dispose them
    deterministically."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        indexer_config: IndexerConfig | None = None,
    ):
        self._conn = conn
        self._embedder = embedder
        self._cfg = indexer_config or IndexerConfig()

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Ensure vec table exists at the embedder's dim + record the
        embedder identity so a future run can detect a model swap."""
        ensure_vec_tables(self._conn, self._embedder.dimensions())
        initialise_capacity(
            self._conn,
            self._embedder,
            advertised_max_tokens=self._cfg.advertised_max_tokens,
            dim=self._embedder.dimensions(),
        )
        set_metadata(self._conn, "embedder_identifier", self._embedder.model_identifier())
        set_metadata(self._conn, "embedder_provider", self._embedder.provider_name())

    # ------------------------------------------------------------------
    # Full-pass index
    # ------------------------------------------------------------------

    def index_all(
        self, source: NotesSource, *, include_trash: bool = False
    ) -> IndexStats:
        """Walk every note from the source, embedding dirty chunks.

        Notes present in the index but absent from the source are
        deleted (cascade removes chunks + vec rows). The default
        `include_trash=False` filters trashed notes at the source —
        the tombstone sweep below then naturally cleans any
        previously-indexed notes that have since been trashed (they
        stop appearing in `iter_notes()` so they're treated like any
        other deleted note).
        """
        self.prepare()
        start_ms = int(time.time() * 1000)
        stats = IndexStats()
        seen_zids: set[str] = set()

        for record in source.iter_notes(include_trash=include_trash):
            stats.notes_seen += 1
            seen_zids.add(record.z_identifier)
            try:
                outcome = self._index_record(record, source)
            except EmbedderDeadError as exc:
                _log.error(
                    "apple-notes-brain: embedder died mid-pass — aborting "
                    "after %d/%d notes (%s)",
                    stats.notes_seen, stats.notes_seen, exc,
                )
                stats.took_ms = int(time.time() * 1000) - start_ms
                stats.failures.append({
                    "z_identifier": record.z_identifier,
                    "reason": "embedder-dead",
                    "error": str(exc),
                })
                # Re-raise so the caller knows the pass aborted; partial
                # state is preserved (sync table not advanced for this note).
                raise
            stats.notes_indexed += outcome["indexed"]
            stats.notes_skipped += outcome["skipped"]
            stats.chunks_embedded += outcome["chunks_embedded"]
            stats.chunks_skipped += outcome["chunks_skipped"]
            stats.chunks_failed += outcome["chunks_failed"]
            if outcome["failures"]:
                stats.failures.extend(outcome["failures"])

        # Tombstone sweep: notes in the index that aren't in the source.
        from .store import all_node_ids

        indexed = all_node_ids(self._conn)
        stale = indexed - seen_zids
        for zid in stale:
            delete_node(self._conn, zid)
            stats.notes_deleted += 1

        stats.took_ms = int(time.time() * 1000) - start_ms
        return stats

    def index_single(
        self,
        source: NotesSource,
        z_identifier: str,
        event: str = "change",
    ) -> SingleNoteResult:
        """Per-note index call used by the watcher."""
        self.prepare()
        if event == "unlink":
            delete_node(self._conn, z_identifier)
            return SingleNoteResult(note_id=z_identifier, event="unlink")
        record = source.get_record(z_identifier)
        if record is None:
            # Record vanished between event firing and the index call;
            # treat as a delete to keep state consistent.
            delete_node(self._conn, z_identifier)
            return SingleNoteResult(note_id=z_identifier, event="unlink")
        outcome = self._index_record(record, source)
        return SingleNoteResult(
            note_id=z_identifier,
            event=event,  # type: ignore[arg-type]
            chunks_embedded=outcome["chunks_embedded"],
            chunks_skipped=outcome["chunks_skipped"],
            chunks_failed=outcome["chunks_failed"],
        )

    # ------------------------------------------------------------------
    # Per-note logic
    # ------------------------------------------------------------------

    def _index_record(self, record: NoteRecord, source: NotesSource) -> dict:
        """Index a single record. Returns a dict of per-note counters.

        Raises EmbedderDeadError up so the caller can abort the whole
        pass. All other failures are logged into `failed_chunks` and the
        method returns counters reflecting the partial outcome.
        """
        outcome = {
            "indexed": 0,
            "skipped": 0,
            "chunks_embedded": 0,
            "chunks_skipped": 0,
            "chunks_failed": 0,
            "failures": [],
        }

        body = source.body_text(record)
        body_hash = _hash_body(record.title, body)

        # Upsert the node row + nodes_fts.
        upsert_node(
            self._conn,
            node_id=record.z_identifier,
            z_pk=record.z_pk,
            title=record.title,
            folder=record.folder,
            modified_at=record.modified_at,
            locked=record.locked,
            pinned=record.pinned,
            content_hash=body_hash,
            body_text=body,
        )

        if record.locked:
            # Locked notes have inaccessible bodies. Record the placeholder
            # in failed_chunks so status reports surface the count.
            record_failed_chunk(
                self._conn,
                chunk_id=chunk_id(record.z_identifier, 0),
                node_id=record.z_identifier,
                reason="locked",
                error_message="note is password-protected; body inaccessible",
            )
            # Remove any pre-existing chunks (e.g. user just locked a note).
            delete_chunks_for_node(self._conn, record.z_identifier, keep_indices=set())
            set_sync(self._conn, record.z_identifier, record.modified_at, int(time.time()))
            outcome["skipped"] = 1
            return outcome

        # Chunk the body. If empty, synthesise a single fallback chunk
        # from title + folder so the note is still findable semantically.
        chunks = chunk_markdown(body, self._cfg.chunker_config)
        if not chunks:
            chunks = [self._synthesise_fallback_chunk(record)]

        kept_indices: set[int] = set()
        for chunk in chunks:
            kept_indices.add(chunk.chunk_index)
            cid = chunk_id(record.z_identifier, chunk.chunk_index)
            existing = get_chunk(self._conn, cid)
            if existing is not None and existing.content_hash == chunk.content_hash:
                outcome["chunks_skipped"] += 1
                continue
            rowid = upsert_chunk(
                self._conn,
                chunk_id=cid,
                node_id=record.z_identifier,
                chunk_index=chunk.chunk_index,
                heading=chunk.heading,
                heading_level=chunk.heading_level,
                content=chunk.content,
                content_hash=chunk.content_hash,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            try:
                vec = self._embedder.embed(
                    build_chunk_embedding_text(chunk), task_type="document"
                )
            except TooLongError as exc:
                record_failed_chunk(
                    self._conn,
                    chunk_id=cid,
                    node_id=record.z_identifier,
                    reason="too-long",
                    error_message=str(exc),
                )
                reduce_discovered_max_tokens(
                    self._conn,
                    self._embedder,
                    approx_tokens_for(build_chunk_embedding_text(chunk)),
                )
                outcome["chunks_failed"] += 1
                outcome["failures"].append({
                    "chunk_id": cid,
                    "reason": "too-long",
                    "error": str(exc),
                })
                continue
            except EmbedderDeadError:
                # Re-raise — the indexer pass aborts cleanly above.
                raise
            except Exception as exc:
                record_failed_chunk(
                    self._conn,
                    chunk_id=cid,
                    node_id=record.z_identifier,
                    reason="embed-error",
                    error_message=str(exc)[:500],
                )
                outcome["chunks_failed"] += 1
                outcome["failures"].append({
                    "chunk_id": cid,
                    "reason": "embed-error",
                    "error": str(exc)[:200],
                })
                continue
            upsert_chunk_vector(self._conn, rowid, vec)
            outcome["chunks_embedded"] += 1

        # Drop chunks no longer in the current chunk list.
        removed = delete_chunks_for_node(
            self._conn, record.z_identifier, keep_indices=kept_indices
        )
        if removed:
            outcome["chunks_skipped"] = max(0, outcome["chunks_skipped"] - removed)

        set_sync(self._conn, record.z_identifier, record.modified_at, int(time.time()))
        outcome["indexed"] = 1
        return outcome

    def _synthesise_fallback_chunk(self, record: NoteRecord) -> Chunk:
        """When a note's chunker output is empty (locked, attachment-only,
        title-only), synthesise a chunk from title + folder + minimal
        scaffolding. Mirrors obsidian-brain's empty-note fallback so
        notes never silently disappear from the index."""
        parts = [record.title]
        if record.folder:
            parts.append(f"folder: {record.folder}")
        content = "\n".join(parts) or "(empty)"
        return Chunk(
            chunk_index=0,
            heading=None,
            heading_level=None,
            content=content,
            content_hash=_hash_body(record.title, content),
            start_line=1,
            end_line=1,
        )


def _hash_body(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\n\n{body}".encode("utf-8")).hexdigest()[:32]
