"""Bootstrap: detect first boot, model/dim/prefix/schema drift, force reindex.

Mirrors obsidian-brain's `src/pipeline/bootstrap.ts:126–296`. Called once
per process from the boot block (`semantic.boot._boot_loop`) after the
embedder finishes `init()`. Returns a `BootstrapResult` the boot block
uses to decide between first-time index / catch-up / drift-forced reindex.

Detection layers (each independently sets `needs_reindex=True`):

  1. **Schema migration ladder** — currently empty (we ship at v1) but
     the loop shape is in place so future-us can `INSERT INTO
     SCHEMA_MIGRATIONS` and let the ladder walk forward.
  2. **Model id + dim** — if `index_metadata.embedding_model` or
     `embedding_dim` differs from the live embedder, drop every chunk
     row + every chunks_vec row + the sync table, recreate chunks_vec
     at the new dim, stamp new metadata.
  3. **Embedder identity hash** — only meaningful for Ollama
     (`OllamaEmbedder.identity_hash()` returns the manifest digest);
     for ONNX the method returns None and this check is a no-op.
  4. **Prefix strategy hash** — sha256 of (query_prefix || '\\n' ||
     document_prefix). If the resolved metadata's prefixes changed
     between boots (model with same name but new asymmetric prompts,
     or a user override switching them), re-embed. Symmetric → empty
     hash; switching symmetric ↔ asymmetric always reindexes.
  5. **Stale-cache promotion** — pre-resolver `embedder_capability`
     rows with NULL prefixes but a known seed entry get filled from
     the seed. Doesn't force a reindex (yet); just patches the cache
     so the resolver chain returns correct values on next access.

A clean first boot stamps everything but doesn't set `needs_reindex` —
the boot block will trigger the first-time index based on `db_is_empty`.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from . import _logging as anb_logging
from .metadata_resolver import resolve_model_metadata
from .seed_loader import lookup as seed_lookup
from .store import (
    all_node_ids,
    ensure_vec_tables,
    get_metadata,
    set_metadata,
)
from .types import Embedder, EmbedderMetadata

_log = logging.getLogger("apple-notes-brain")


# Schema migration ladder — populated when SCHEMA_VERSION bumps require
# data fix-ups (e.g. v1 → v2 adds a column, ladder entry runs ALTER TABLE
# + backfill). Each entry's `apply` MUST be idempotent.
@dataclass(frozen=True)
class _Migration:
    to_version: int
    apply: Callable[[sqlite3.Connection], None]


SCHEMA_MIGRATIONS: list[_Migration] = []


@dataclass
class BootstrapResult:
    """What `run_bootstrap()` tells the boot block.

    The boot block consults `db_is_empty` and `needs_reindex` to pick
    one of three paths:
      * `db_is_empty=True` → run a first-time index (always).
      * `needs_reindex=True` (db not empty) → drift forced a re-embed;
        the sync table has been cleared, so the next `index_all` will
        re-embed every chunk.
      * Otherwise → catch-up reindex (cheap; content-hash dedup) unless
        `APPLE_NOTES_BRAIN_NO_CATCHUP=1`.

    `reasons` is surfaced via stderr INFO logs on every boot so an
    operator can see *why* a heavy reindex was triggered.
    """
    db_is_empty: bool
    needs_reindex: bool
    reasons: list[str] = field(default_factory=list)


def run_bootstrap(conn: sqlite3.Connection, embedder: Embedder) -> BootstrapResult:
    """Reconcile stored state against the live embedder. See module docstring."""
    anb_logging.debug_log("bootstrap: starting")
    reasons: list[str] = []
    needs_reindex = False

    db_is_empty = len(all_node_ids(conn)) == 0
    anb_logging.debug_log(
        "bootstrap: db_is_empty=%s" % db_is_empty
    )

    # 1. Schema migration ladder.
    _run_schema_migrations(conn)

    # 2. Model + dim change detection.
    stored_model = get_metadata(conn, "embedding_model")
    stored_dim_raw = get_metadata(conn, "embedding_dim")
    stored_dim = int(stored_dim_raw) if stored_dim_raw else 0
    current_model = embedder.model_identifier()
    current_dim = embedder.dimensions()

    if not stored_model:
        # First-boot stamp. Don't force a reindex — db_is_empty already
        # tells the boot block to run the first-time index.
        anb_logging.debug_log(
            "bootstrap: first-ever boot — stamping model=%s dim=%d"
            % (current_model, current_dim)
        )
        set_metadata(conn, "embedding_model", current_model)
        set_metadata(conn, "embedding_dim", str(current_dim))
    elif stored_model != current_model or stored_dim != current_dim:
        reason = (
            f"embedding model changed: {stored_model} → {current_model} "
            f"(dim {stored_dim} → {current_dim}) — re-embedding all notes"
        )
        reasons.append(reason)
        _log.info("bootstrap: %s", reason)
        needs_reindex = True
        _drop_embedding_state(conn)
        ensure_vec_tables(conn, current_dim)
        set_metadata(conn, "embedding_model", current_model)
        set_metadata(conn, "embedding_dim", str(current_dim))

    # 3. Identity hash change (Ollama-only). For ONNX this is a no-op.
    current_identity = _safe_call(getattr(embedder, "identity_hash", None))
    if current_identity is not None:
        stored_identity = get_metadata(conn, "embedder_identity_hash")
        if stored_identity and stored_identity != current_identity:
            reason = (
                f"model weights for {current_model} were updated "
                "(probably an `ollama pull`) — re-embedding to match"
            )
            reasons.append(reason)
            _log.info("bootstrap: %s", reason)
            needs_reindex = True
            _drop_embedding_state(conn)
            ensure_vec_tables(conn, current_dim)
        if stored_identity != current_identity:
            set_metadata(conn, "embedder_identity_hash", current_identity)

    # 4. Resolve current metadata + compute prefix-strategy hash.
    metadata = resolve_model_metadata(conn, embedder)
    # Stamp the resolved metadata onto the embedder so embed() applies
    # the correct prefixes per task_type going forward.
    try:
        embedder.set_metadata(_to_embedder_metadata(metadata))
    except Exception as exc:  # set_metadata is best-effort
        anb_logging.debug_log(
            "bootstrap: embedder.set_metadata raised %r (continuing)" % exc
        )

    current_strategy = _compute_prefix_strategy(metadata.query_prefix, metadata.document_prefix)
    stored_strategy = get_metadata(conn, "embedder_prefix_strategy") or ""

    if not stored_model:
        # First boot: just stamp.
        set_metadata(conn, "embedder_prefix_strategy", current_strategy)
    elif stored_strategy != current_strategy:
        if current_strategy == "" and stored_strategy != "":
            reason = (
                "embedding model now treats queries and documents the same way "
                "— re-embedding to match"
            )
        elif current_strategy != "":
            reason = (
                f"embedding model {current_model} uses different query/document "
                "prefixes than before — re-embedding for accurate search"
            )
        else:
            reason = "embedding prefix strategy changed — re-embedding"
        reasons.append(reason)
        _log.info("bootstrap: %s", reason)
        needs_reindex = True
        set_metadata(conn, "embedder_prefix_strategy", current_strategy)

    # 5. Stale-cache promotion (best-effort, no reindex impact).
    try:
        _promote_from_seed_if_stale(conn, embedder)
    except Exception as exc:
        anb_logging.debug_log(
            "bootstrap: stale-cache promotion raised %r (continuing)" % exc
        )

    # If any drift forced a reindex, clear the sync table so the next
    # index_all pass recomputes every chunk's content_hash and re-embeds.
    if needs_reindex:
        conn.execute("DELETE FROM sync")
        anb_logging.debug_log("bootstrap: sync table cleared (force reindex)")

    anb_logging.debug_log(
        "bootstrap: complete — db_is_empty=%s needs_reindex=%s reasons=%d"
        % (db_is_empty, needs_reindex, len(reasons))
    )
    return BootstrapResult(
        db_is_empty=db_is_empty,
        needs_reindex=needs_reindex,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    """Walk SCHEMA_MIGRATIONS from `index_metadata.schema_version` forward.

    Helpers are required to be idempotent. We bump the stored version
    atomically after each helper succeeds; a crash mid-chain leaves us
    able to resume on next boot from the next un-applied migration."""
    if not SCHEMA_MIGRATIONS:
        return
    stored = int(get_metadata(conn, "schema_version") or "0")
    for migration in SCHEMA_MIGRATIONS:
        if stored < migration.to_version:
            anb_logging.debug_log(
                "bootstrap: applying schema migration → v%d" % migration.to_version
            )
            migration.apply(conn)
            set_metadata(conn, "schema_version", str(migration.to_version))
            stored = migration.to_version


def _drop_embedding_state(conn: sqlite3.Connection) -> None:
    """Clear every byte that's downstream of an old embedder/dim.

    Keeps nodes + nodes_fts (those don't depend on the embedder) so the
    catch-up reindex doesn't have to re-walk the source for note
    identities — only re-chunk + re-embed.
    """
    # chunks_vec must be dropped before chunks because the FK cascade on
    # chunks → chunks_vec would otherwise leave orphaned vec rows; the
    # cleanest sequence is just DELETE both explicitly.
    try:
        conn.execute("DELETE FROM chunks_vec")
    except sqlite3.OperationalError:
        # chunks_vec may not exist yet on a brand-new DB; harmless.
        pass
    conn.execute("DELETE FROM chunks")
    # Sync table cleared after drift is committed; keep it here too so
    # the operation is self-contained (helps test isolation).
    conn.execute("DELETE FROM sync")
    anb_logging.debug_log(
        "bootstrap: dropped chunks + chunks_vec + sync (model/dim/identity change)"
    )


def _compute_prefix_strategy(query_prefix: str | None, document_prefix: str | None) -> str:
    """Stable short hash of the (query, document) prefix pair.

    Symmetric models (both empty) collapse to the empty string. Any
    change to either prefix flips the hash → bootstrap triggers a
    re-embed on next boot.
    """
    q = query_prefix or ""
    d = document_prefix or ""
    if not q and not d:
        return ""
    payload = f"{q}\n{d}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _to_embedder_metadata(resolved) -> EmbedderMetadata:
    """Bridge from `ResolvedMetadata` (resolver-internal) to
    `EmbedderMetadata` (the embedder Protocol's contract)."""
    return EmbedderMetadata(
        model_id=resolved.model_id,
        dim=resolved.dim,
        max_tokens=resolved.max_tokens,
        query_prefix=resolved.query_prefix or "",
        document_prefix=resolved.document_prefix or "",
        prefix_source=resolved.prefix_source,
        base_model=resolved.base_model,
        size_bytes=resolved.size_bytes,
    )


def _promote_from_seed_if_stale(conn: sqlite3.Connection, embedder: Embedder) -> None:
    """Detect pre-resolver embedder_capability rows where the prefix
    columns are NULL but the bundled seed has them. Fill from seed.

    Best-effort; failures here don't block bootstrap."""
    from .metadata_cache import load_cached, upsert_cache  # local import

    cached = load_cached(conn, embedder)
    if cached is None:
        return
    if cached.query_prefix not in (None, "") or cached.document_prefix not in (None, ""):
        return  # Cache is already populated; nothing to promote.
    seed_entry = seed_lookup(embedder.model_identifier())
    if seed_entry is None:
        return
    if not seed_entry.query_prefix and not seed_entry.document_prefix:
        return  # Seed has symmetric prefixes — nothing useful to promote.

    anb_logging.debug_log(
        "bootstrap: promoting seed prefixes into stale embedder_capability row"
    )
    from .metadata_cache import CachedMetadata
    import time

    promoted = CachedMetadata(
        model_id=cached.model_id,
        dim=cached.dim,
        max_tokens=cached.max_tokens,
        query_prefix=seed_entry.query_prefix,
        document_prefix=seed_entry.document_prefix,
        prefix_source="seed",
        base_model=cached.base_model,
        size_bytes=cached.size_bytes,
        fetched_at=int(time.time()),
    )
    upsert_cache(conn, embedder, promoted)


def _safe_call(fn):
    """Call a possibly-None / possibly-raising method, returning the
    result or None. Used for optional embedder methods like
    `identity_hash()` that ONNX doesn't implement."""
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None
