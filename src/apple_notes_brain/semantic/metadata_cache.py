"""Persistent cache layer for resolved embedding metadata.

Mirrors obsidian-brain's `src/embeddings/metadata-cache.ts`. A thin
wrapper over the v7 columns on `embedder_capability`:

  ``dim``, ``query_prefix``, ``document_prefix``, ``prefix_source``,
  ``base_model``, ``size_bytes``, ``fetched_at``.

Cache lifetime is forever, until explicit invalidation — the fields we
cache (dim, prefixes, ONNX size) are immutable for a given HF model
revision, so silently re-fetching would just burn HF quota. Users
invalidate explicitly via a future ``apple-notes-brain models
refresh-cache`` CLI command (or a tool wrapper on the MCP side).

SQL-only — does not call HF, does not load the bundled seed, does not
apply user overrides. All higher-level orchestration lives in
:mod:`metadata_resolver`.

The (embedder_id, model_hash) pair forms the primary key. notes-mcp,
like the existing capacity helper, uses the embedder.model_identifier()
value for BOTH columns — model_hash isn't a separate sha256 here
because we already have a stable model identifier. The schema stays
compatible with obsidian-brain's shape so future schema-sync work has
something to talk to.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._logging import debug_log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .types import Embedder

_log = logging.getLogger("apple-notes-brain")


@dataclass(frozen=True)
class CachedMetadata:
    """Persisted, resolved metadata for a single embedder model.

    Mirrors obsidian-brain's ``CachedMetadata`` shape. ``fetched_at`` is
    the canonical "this row was populated by the resolver" marker —
    pre-resolver (v6-shape) rows have ``fetched_at IS NULL`` and are
    treated as a cache miss so the resolver re-fills them.

    All "no value known" cases are represented as ``None`` so the
    resolver layer can layer overrides + defaults on top.
    """

    model_id: str
    dim: int | None
    max_tokens: int | None
    query_prefix: str | None
    document_prefix: str | None
    prefix_source: str
    base_model: str | None
    size_bytes: int | None
    fetched_at: int | None


def _embedder_key(embedder: "Embedder") -> str:
    """Return the stable identifier used as both embedder_id and model_hash.

    Matches the convention in :mod:`semantic.capacity` so a single row
    per (provider, model) is consistently keyed across all layers.
    """
    return embedder.model_identifier()


def load_cached(
    db: sqlite3.Connection, embedder: "Embedder"
) -> CachedMetadata | None:
    """Read the cache row for ``embedder``. Returns ``None`` on a miss.

    Treats v6-shape rows (``fetched_at IS NULL``) as a miss — those rows
    were written by an older notes-mcp before the resolver chain
    existed, and don't carry the metadata fields the resolver needs.
    """
    key = _embedder_key(embedder)
    row = db.execute(
        """
        SELECT advertised_max_tokens, dim, query_prefix, document_prefix,
               prefix_source, base_model, size_bytes, fetched_at
          FROM embedder_capability
         WHERE embedder_id = ? AND model_hash = ?
        """,
        (key, key),
    ).fetchone()
    if row is None:
        debug_log("metadata-cache: miss (no row)", model=key)
        return None
    (
        max_tokens,
        dim,
        query_prefix,
        document_prefix,
        prefix_source,
        base_model,
        size_bytes,
        fetched_at,
    ) = row
    if fetched_at is None:
        # v6-shape row written before the resolver existed — treat as a miss.
        debug_log("metadata-cache: miss (fetched_at NULL)", model=key)
        return None
    debug_log(
        "metadata-cache: hit",
        model=key,
        dim=dim,
        prefix_source=prefix_source,
        fetched_at=fetched_at,
    )
    return CachedMetadata(
        model_id=key,
        dim=int(dim) if dim is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        prefix_source=str(prefix_source) if prefix_source is not None else "none",
        base_model=base_model,
        size_bytes=int(size_bytes) if size_bytes is not None else None,
        fetched_at=int(fetched_at),
    )


def upsert_cache(
    db: sqlite3.Connection,
    embedder: "Embedder",
    metadata: CachedMetadata,
) -> None:
    """Insert-or-update the resolver-owned v7 columns for this embedder.

    Preserves the v6 capacity columns (``discovered_max_tokens``,
    ``discovered_at``, ``method``) — those track adaptive-capacity drift
    and shouldn't be reset by a metadata refresh. On a fresh insert we
    seed ``discovered_max_tokens`` to the advertised value so the
    capacity ratchet has a sensible starting point.
    """
    key = _embedder_key(embedder)
    now = metadata.fetched_at if metadata.fetched_at is not None else int(time.time())
    db.execute(
        """
        INSERT INTO embedder_capability (
            embedder_id, model_hash,
            advertised_max_tokens, discovered_max_tokens, discovered_at, method,
            dim, query_prefix, document_prefix, prefix_source,
            base_model, size_bytes, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(embedder_id, model_hash) DO UPDATE SET
            advertised_max_tokens = excluded.advertised_max_tokens,
            dim                   = excluded.dim,
            query_prefix          = excluded.query_prefix,
            document_prefix       = excluded.document_prefix,
            prefix_source         = excluded.prefix_source,
            base_model            = excluded.base_model,
            size_bytes            = excluded.size_bytes,
            fetched_at            = excluded.fetched_at
        """,
        (
            key,
            key,
            metadata.max_tokens,
            metadata.max_tokens,
            now,
            "metadata-cache",
            metadata.dim,
            metadata.query_prefix,
            metadata.document_prefix,
            metadata.prefix_source,
            metadata.base_model,
            metadata.size_bytes,
            now,
        ),
    )
    debug_log(
        "metadata-cache: upsert",
        model=key,
        dim=metadata.dim,
        prefix_source=metadata.prefix_source,
        max_tokens=metadata.max_tokens,
    )


def invalidate_cache(
    db: sqlite3.Connection, embedder: "Embedder" | None = None
) -> int:
    """Clear the v7 metadata columns. Returns the number of rows touched.

    When ``embedder`` is ``None``, clears every entry (used by a future
    ``refresh_cache`` admin tool). When set, only the matching row is
    cleared. The v6 capacity columns are preserved — adaptive capacity
    is a separate concern from metadata staleness.
    """
    set_columns = (
        "dim = NULL, query_prefix = NULL, document_prefix = NULL, "
        "prefix_source = NULL, base_model = NULL, size_bytes = NULL, "
        "fetched_at = NULL"
    )
    if embedder is None:
        cur = db.execute(f"UPDATE embedder_capability SET {set_columns}")
        changes = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
        debug_log("metadata-cache: invalidate-all", rows=changes)
        return int(changes)
    key = _embedder_key(embedder)
    cur = db.execute(
        f"UPDATE embedder_capability SET {set_columns} WHERE embedder_id = ?",
        (key,),
    )
    changes = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
    debug_log("metadata-cache: invalidate", model=key, rows=changes)
    return int(changes)
