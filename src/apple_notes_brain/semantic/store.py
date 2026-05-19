"""SQLite store for the semantic index — schema, opening, kNN queries.

This is the Python equivalent of obsidian-brain's `src/store/db.ts` plus
`src/store/chunks.ts` plus `src/store/embeddings.ts`, collapsed into one
module since the surface area is small enough that splitting hurts more
than it helps.

Design choices worth flagging:

- **Schema version starts at 1.** This is a fresh database — we don't
  inherit obsidian-brain's v7 schema, even though field names match,
  because the migration ladder is a different shape (no node-level
  legacy `nodes_vec`, no graph tables, no Louvain communities, no
  obsidian wiki-link edges).
- **`chunks_vec` dim is decided at first init from the embedder** and
  stored in `index_metadata.chunks_vec_dim`. A subsequent boot with a
  different embedder is detected: if the table is empty we drop+recreate
  at the new dim; if it has rows we refuse and ask the caller to delete
  the DB or run reindex with --force-recreate.
- **WAL + 5s busy timeout** so the background watcher and on-demand
  reindex don't trip each other.
- **sqlite-vec is loaded per-connection** (extension loading is connection-
  scoped in sqlite). Every `open_db` call enables it.
- **Vectors are stored as raw float32 bytes**, identical encoding to
  obsidian-brain — sqlite-vec accepts `bytes(np.ndarray.astype(float32))`.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from ._logging import debug_log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEFAULT_DIM = 384  # bge-small-en-v1.5 / MiniLM-L6-v2 default

_BUSY_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class StoreError(Exception):
    """Base for store errors."""


class DimensionMismatchError(StoreError):
    """The embedder's dim doesn't match what the DB was built for and the
    DB has rows we'd lose by recreating. Caller must reindex to recover."""


# ---------------------------------------------------------------------------
# Connection open + schema init
# ---------------------------------------------------------------------------

def open_db(path: Path | str) -> sqlite3.Connection:
    """Open / create the semantic-index DB. Loads sqlite-vec, sets WAL,
    initialises schema if needed.

    Caller owns the connection lifecycle — close it when done. The store
    functions all accept a `conn` so a long-lived connection can be
    shared across calls.
    """
    # check_same_thread=False lets the background watcher thread and the
    # MCP tool-dispatch thread share a connection. WAL mode + busy_timeout
    # serialise writers; readers are concurrent.
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,
        timeout=5.0,
        check_same_thread=False,
    )
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # Load sqlite-vec — required for `vec0` virtual tables and the
    # `vec_version()` function. enable_load_extension must be on first.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent: creates every table + virtual table + index if absent.

    On a fresh DB this is the only call needed; on an existing DB at the
    current schema version it's a no-op (CREATE IF NOT EXISTS everywhere).
    """
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS nodes (
            id            TEXT PRIMARY KEY,
            z_pk          INTEGER NOT NULL,
            title         TEXT NOT NULL,
            folder        TEXT,
            modified_at   INTEGER,
            locked        INTEGER NOT NULL DEFAULT 0,
            pinned        INTEGER NOT NULL DEFAULT 0,
            content_hash  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_nodes_z_pk ON nodes(z_pk);

        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            node_id       TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            chunk_index   INTEGER NOT NULL,
            heading       TEXT,
            heading_level INTEGER,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            start_line    INTEGER,
            end_line      INTEGER,
            UNIQUE(node_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_node_id ON chunks(node_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

        CREATE TABLE IF NOT EXISTS sync (
            node_id     TEXT PRIMARY KEY,
            modified_at INTEGER NOT NULL,
            indexed_at  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embedder_capability (
            embedder_id            TEXT NOT NULL,
            model_hash             TEXT NOT NULL,
            advertised_max_tokens  INTEGER,
            discovered_max_tokens  INTEGER,
            discovered_at          INTEGER,
            method                 TEXT,
            dim                    INTEGER,
            query_prefix           TEXT,
            document_prefix        TEXT,
            prefix_source          TEXT,
            base_model             TEXT,
            size_bytes             INTEGER,
            fetched_at             INTEGER,
            PRIMARY KEY (embedder_id, model_hash)
        );

        CREATE TABLE IF NOT EXISTS failed_chunks (
            chunk_id      TEXT PRIMARY KEY,
            node_id       TEXT NOT NULL,
            reason        TEXT NOT NULL,
            error_message TEXT,
            failed_at     INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS index_metadata (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        -- Standalone FTS5 — NOT using external-content because `nodes` doesn't
        -- store the full body (chunks own that). upsert_node/delete_node
        -- maintain nodes_fts directly via rowid alignment.
        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            title,
            content,
            tokenize='porter unicode61'
        );
        """
    )
    # SCHEMA_VERSION write — only on a totally fresh DB. We use INSERT OR
    # IGNORE so re-running init() doesn't clobber a future migration that
    # bumped the version.
    conn.execute(
        "INSERT OR IGNORE INTO index_metadata (key, value, updated_at) "
        "VALUES ('schema_version', ?, ?)",
        (str(SCHEMA_VERSION), int(time.time())),
    )
    debug_log(f"store: schema initialised at version={SCHEMA_VERSION}")


def current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM index_metadata WHERE key = 'schema_version'"
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# chunks_vec — dim-aware lifecycle
# ---------------------------------------------------------------------------

def chunks_vec_dim(conn: sqlite3.Connection) -> int | None:
    """Return the dim the chunks_vec virtual table was created for, or
    None if it hasn't been created yet."""
    row = conn.execute(
        "SELECT value FROM index_metadata WHERE key = 'chunks_vec_dim'"
    ).fetchone()
    return int(row[0]) if row else None


def ensure_vec_tables(conn: sqlite3.Connection, dim: int) -> None:
    """Make sure chunks_vec exists and is at `dim`. Behaviour:

      * not yet created → create at `dim`, write dim to metadata.
      * exists at `dim` → no-op.
      * exists at other dim, table empty → drop and recreate at `dim`.
      * exists at other dim, table has rows → raise DimensionMismatchError
        so the caller can decide (typically: delete the DB + reindex).
    """
    stored = chunks_vec_dim(conn)
    if stored is None:
        _create_chunks_vec(conn, dim)
        return
    if stored == dim:
        return
    # Mismatch — check row count.
    count = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    if count == 0:
        debug_log(
            f"store: vec dim mismatch stored={stored} embedder={dim} "
            f"empty=True → recreating"
        )
        conn.execute("DROP TABLE chunks_vec")
        _create_chunks_vec(conn, dim)
        return
    debug_log(
        f"store: vec dim mismatch stored={stored} embedder={dim} "
        f"populated={count} → raising"
    )
    raise DimensionMismatchError(
        f"chunks_vec was built for dim={stored} but the current embedder "
        f"emits dim={dim}. The table has {count} rows — refusing to drop. "
        f"Delete the DB or rebuild the index after switching models."
    )


def _create_chunks_vec(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"embedding float[{int(dim)}])"
    )
    _set_metadata(conn, "chunks_vec_dim", str(int(dim)))
    debug_log(f"store: chunks_vec created at dim={int(dim)}")


def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO index_metadata (key, value, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, value, int(time.time())),
    )


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM index_metadata WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    _set_metadata(conn, key, value)


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeRow:
    id: str
    z_pk: int
    title: str
    folder: str | None
    modified_at: int | None
    locked: bool
    pinned: bool
    content_hash: str | None


def upsert_node(
    conn: sqlite3.Connection,
    *,
    node_id: str,
    z_pk: int,
    title: str,
    folder: str | None,
    modified_at: int | None,
    locked: bool,
    pinned: bool,
    content_hash: str | None,
    body_text: str = "",
) -> int:
    """Insert or update a node row. Returns its rowid (FTS-anchor).

    `body_text` is the full note body to index in `nodes_fts(content)`.
    Pass an empty string when you haven't materialised the body (a
    title-only index still helps lexical search; `Search.fulltext()`
    queries match titles even when content is empty).
    """
    conn.execute(
        """
        INSERT INTO nodes (id, z_pk, title, folder, modified_at, locked, pinned, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            z_pk         = excluded.z_pk,
            title        = excluded.title,
            folder       = excluded.folder,
            modified_at  = excluded.modified_at,
            locked       = excluded.locked,
            pinned       = excluded.pinned,
            content_hash = excluded.content_hash
        """,
        (
            node_id, z_pk, title, folder, modified_at,
            int(locked), int(pinned), content_hash,
        ),
    )
    rowid = conn.execute(
        "SELECT rowid FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()[0]
    # Standalone FTS — delete + insert keeps it in sync without depending
    # on triggers (FTS5 contentless triggers wouldn't carry the body text).
    conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (int(rowid),))
    conn.execute(
        "INSERT INTO nodes_fts (rowid, title, content) VALUES (?, ?, ?)",
        (int(rowid), title, body_text),
    )
    return int(rowid)


def get_node(conn: sqlite3.Connection, node_id: str) -> NodeRow | None:
    row = conn.execute(
        "SELECT id, z_pk, title, folder, modified_at, locked, pinned, content_hash "
        "FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    if row is None:
        return None
    return NodeRow(
        id=row[0], z_pk=row[1], title=row[2], folder=row[3],
        modified_at=row[4], locked=bool(row[5]), pinned=bool(row[6]),
        content_hash=row[7],
    )


def delete_node(conn: sqlite3.Connection, node_id: str) -> None:
    """Delete a node and cascade to its chunks. Also cleans nodes_fts and
    chunks_vec rows whose rowids reference the deleted chunks."""
    rowid_row = conn.execute(
        "SELECT rowid FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if rowid_row is None:
        return
    rowid = int(rowid_row[0])
    # Gather chunk rowids before cascade so we can clean chunks_vec.
    chunk_rowids = [
        int(r[0])
        for r in conn.execute(
            "SELECT rowid FROM chunks WHERE node_id = ?", (node_id,)
        )
    ]
    conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
    conn.execute("DELETE FROM nodes_fts WHERE rowid = ?", (rowid,))
    conn.execute("DELETE FROM sync WHERE node_id = ?", (node_id,))
    if chunk_rowids:
        # IN list — small, so direct interpolation of placeholders is fine.
        placeholders = ",".join("?" * len(chunk_rowids))
        conn.execute(
            f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})",
            chunk_rowids,
        )


def all_node_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT id FROM nodes")}


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkRow:
    id: str
    node_id: str
    chunk_index: int
    heading: str | None
    heading_level: int | None
    content: str
    content_hash: str
    start_line: int | None
    end_line: int | None


def upsert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    node_id: str,
    chunk_index: int,
    heading: str | None,
    heading_level: int | None,
    content: str,
    content_hash: str,
    start_line: int | None,
    end_line: int | None,
) -> int:
    """Insert / update a chunk. Returns its rowid (for chunks_vec anchor)."""
    conn.execute(
        """
        INSERT INTO chunks (id, node_id, chunk_index, heading, heading_level,
                            content, content_hash, start_line, end_line)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            chunk_index   = excluded.chunk_index,
            heading       = excluded.heading,
            heading_level = excluded.heading_level,
            content       = excluded.content,
            content_hash  = excluded.content_hash,
            start_line    = excluded.start_line,
            end_line      = excluded.end_line
        """,
        (chunk_id, node_id, chunk_index, heading, heading_level,
         content, content_hash, start_line, end_line),
    )
    row = conn.execute("SELECT rowid FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    return int(row[0])


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> ChunkRow | None:
    row = conn.execute(
        "SELECT id, node_id, chunk_index, heading, heading_level, content, "
        "       content_hash, start_line, end_line "
        "FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return ChunkRow(*row)


def delete_chunks_for_node(
    conn: sqlite3.Connection, node_id: str, *, keep_indices: set[int] | None = None
) -> int:
    """Delete chunks belonging to node_id whose chunk_index isn't in
    keep_indices (or all of them if keep_indices is None). Returns the
    number of rows removed. Also cleans the chunks_vec rows by rowid."""
    if keep_indices is None:
        rows = list(
            conn.execute("SELECT rowid FROM chunks WHERE node_id = ?", (node_id,))
        )
    else:
        if not keep_indices:
            rows = list(
                conn.execute("SELECT rowid FROM chunks WHERE node_id = ?", (node_id,))
            )
        else:
            placeholders = ",".join("?" * len(keep_indices))
            params = [node_id, *keep_indices]
            rows = list(
                conn.execute(
                    f"SELECT rowid FROM chunks "
                    f"WHERE node_id = ? AND chunk_index NOT IN ({placeholders})",
                    params,
                )
            )
    if not rows:
        return 0
    rowids = [int(r[0]) for r in rows]
    placeholders = ",".join("?" * len(rowids))
    conn.execute(f"DELETE FROM chunks WHERE rowid IN ({placeholders})", rowids)
    conn.execute(f"DELETE FROM chunks_vec WHERE rowid IN ({placeholders})", rowids)
    return len(rowids)


# ---------------------------------------------------------------------------
# Vector ops
# ---------------------------------------------------------------------------

def upsert_chunk_vector(
    conn: sqlite3.Connection, rowid: int, vector: np.ndarray
) -> None:
    """Write a float32 vector at the given rowid. Existing row replaced.

    vec0 virtual tables don't support `INSERT OR REPLACE` semantics in the
    sqlite-vec dialect we target, so this is a DELETE + INSERT pair.
    """
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    payload = vector.tobytes()
    conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (int(rowid),))
    conn.execute(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        (int(rowid), payload),
    )


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    node_id: str
    chunk_index: int
    heading: str | None
    heading_level: int | None
    content: str
    start_line: int | None
    end_line: int | None
    note_title: str
    score: float
    note_folder: str | None = None


def search_chunk_vectors(
    conn: sqlite3.Connection, vector: np.ndarray, limit: int
) -> list[ChunkHit]:
    """kNN over chunks_vec, joined back to chunks + nodes.

    Score = 1 - distance (cosine similarity). vec0's MATCH operator
    returns rows sorted by distance ascending, so higher score == better
    after the transformation.
    """
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    payload = vector.reshape(-1).tobytes()
    rows = conn.execute(
        """
        SELECT v.rowid, v.distance,
               c.id, c.node_id, c.chunk_index, c.heading, c.heading_level,
               c.content, c.start_line, c.end_line,
               n.title, n.folder
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        JOIN nodes  n ON n.id    = c.node_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (payload, int(limit)),
    ).fetchall()
    return [
        ChunkHit(
            chunk_id=r[2],
            node_id=r[3],
            chunk_index=r[4],
            heading=r[5],
            heading_level=r[6],
            content=r[7],
            start_line=r[8],
            end_line=r[9],
            note_title=r[10],
            score=1.0 - float(r[1]),
            note_folder=r[11],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Sync table (per-note last-indexed tracking)
# ---------------------------------------------------------------------------

def get_sync(conn: sqlite3.Connection, node_id: str) -> tuple[int, int] | None:
    row = conn.execute(
        "SELECT modified_at, indexed_at FROM sync WHERE node_id = ?", (node_id,)
    ).fetchone()
    return (int(row[0]), int(row[1])) if row else None


def set_sync(
    conn: sqlite3.Connection, node_id: str, modified_at: int, indexed_at: int
) -> None:
    conn.execute(
        "INSERT INTO sync (node_id, modified_at, indexed_at) VALUES (?, ?, ?) "
        "ON CONFLICT(node_id) DO UPDATE SET "
        "modified_at = excluded.modified_at, indexed_at = excluded.indexed_at",
        (node_id, modified_at, indexed_at),
    )


def all_sync_node_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT node_id FROM sync")}


# ---------------------------------------------------------------------------
# Failed chunks log
# ---------------------------------------------------------------------------

def record_failed_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    node_id: str,
    reason: str,
    error_message: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO failed_chunks (chunk_id, node_id, reason, error_message, failed_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(chunk_id) DO UPDATE SET "
        "reason = excluded.reason, error_message = excluded.error_message, "
        "failed_at = excluded.failed_at",
        (chunk_id, node_id, reason, error_message, int(time.time())),
    )


def count_failed_chunks(conn: sqlite3.Connection) -> int:
    """Total failed_chunks rows, including `reason='locked'` placeholders.

    Prefer `count_failed_chunks_by_reason` + the split in
    `IndexStatus.total_failed_chunks` / `IndexStatus.locked_notes` for
    user-facing reporting — locked notes aren't failures.
    """
    return int(conn.execute("SELECT COUNT(*) FROM failed_chunks").fetchone()[0])


def count_failed_chunks_by_reason(
    conn: sqlite3.Connection,
) -> dict[str, int]:
    rows = conn.execute(
        "SELECT reason, COUNT(*) FROM failed_chunks GROUP BY reason"
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def clear_failed_chunks(conn: sqlite3.Connection) -> int:
    """Delete every row from failed_chunks. Returns the row count
    deleted, excluding `reason='locked'` placeholders (locked notes
    are an expected state, not a failure to "clear").

    Used by `reindex_semantic(force=True)` to reset the persistent
    failure counter that otherwise sticks around even after the
    original failure has been resolved by a subsequent successful
    pass. The cleared-count returned to the caller maps to
    `prior_failures_cleared` in the tool response — locked-row deletes
    are not user-visible because they re-create on the next pass.
    """
    real_failures = int(
        conn.execute(
            "SELECT COUNT(*) FROM failed_chunks WHERE reason != 'locked'"
        ).fetchone()[0]
    )
    conn.execute("DELETE FROM failed_chunks")
    return real_failures


def list_failed_chunk_ids(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    exclude_locked: bool = True,
) -> list[str]:
    """Return up to `limit` failed chunk IDs, most-recent first.

    Used by `semantic_index_status` to surface which chunks are
    currently failing so the caller can correlate with the
    `total_failed_chunks` counter. By default excludes
    `reason='locked'` rows — those aren't failures, they're expected
    placeholders for password-protected notes (surfaced under
    `locked_notes` instead).
    """
    if limit <= 0:
        return []
    if exclude_locked:
        sql = (
            "SELECT chunk_id FROM failed_chunks WHERE reason != 'locked' "
            "ORDER BY failed_at DESC, chunk_id ASC LIMIT ?"
        )
    else:
        sql = (
            "SELECT chunk_id FROM failed_chunks "
            "ORDER BY failed_at DESC, chunk_id ASC LIMIT ?"
        )
    rows = conn.execute(sql, (int(limit),)).fetchall()
    return [str(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Status / inspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexStatus:
    total_nodes: int
    total_chunks: int
    total_failed_chunks: int  # excludes reason='locked'
    locked_notes: int  # reason='locked' placeholders only
    failed_chunks_by_reason: dict[str, int]  # excludes 'locked' key
    chunks_vec_dim: int | None
    schema_version: int
    last_indexed_at: int | None
    vec_version: str | None


def index_status(conn: sqlite3.Connection) -> IndexStatus:
    total_nodes = int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
    total_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    last_indexed = conn.execute("SELECT MAX(indexed_at) FROM sync").fetchone()[0]
    try:
        vec_version = conn.execute("SELECT vec_version()").fetchone()[0]
    except sqlite3.OperationalError:
        vec_version = None
    by_reason = count_failed_chunks_by_reason(conn)
    locked_notes = by_reason.pop("locked", 0)
    real_failures = sum(by_reason.values())
    return IndexStatus(
        total_nodes=total_nodes,
        total_chunks=total_chunks,
        total_failed_chunks=real_failures,
        locked_notes=locked_notes,
        failed_chunks_by_reason=by_reason,
        chunks_vec_dim=chunks_vec_dim(conn),
        schema_version=current_schema_version(conn),
        last_indexed_at=int(last_indexed) if last_indexed else None,
        vec_version=vec_version,
    )
