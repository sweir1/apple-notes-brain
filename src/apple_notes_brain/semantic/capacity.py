"""Adaptive capacity ratchet + failed_chunks fallback.

Mirrors obsidian-brain's `src/embeddings/capacity.ts`. The idea: every
embedder advertises a max_tokens (e.g. 512 for BGE-small), but real-
world chunks can exceed that if the chunker's `chunk_size` * `CHARS_
PER_TOKEN` heuristic was too optimistic. When the embedder rejects a
chunk for being too long, we ratchet down the *discovered* max_tokens
in `embedder_capability`, and the next indexing pass uses a smaller
char budget so future chunks fit.

The floor at 256 tokens prevents an infinite ratchet (one outlier
shouldn't drag every chunk down to 32 tokens forever). When the floor
is hit, the chunk goes to failed_chunks with reason='too-long' and
the indexer keeps going.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from .types import Embedder

# Cross-model average — BGE/MiniLM/E5 all sit around 2.5 chars/token in
# English text. Multilingual models drop to ~1.5; future tuning per
# embedder lives in the preset registry, not here.
CHARS_PER_TOKEN = 2.5

# The ratchet won't go below this many tokens. Picked so even a 5x
# under-shoot still produces useful 256-token chunks.
MIN_DISCOVERED_TOKENS = 256

# When nothing is known about the embedder's max length yet, fall back
# to a safe default. 512 matches BGE-small / MiniLM-L6-v2.
FALLBACK_MAX_TOKENS = 512


@dataclass(frozen=True)
class EmbedderCapacity:
    """Effective capacity snapshot used by the indexer to size chunks."""
    embedder_id: str
    advertised_max_tokens: int | None
    discovered_max_tokens: int | None
    effective_max_tokens: int
    chunk_budget_chars: int


def _embedder_id(embedder: Embedder) -> str:
    """Stable identifier — used as the key into embedder_capability."""
    return embedder.model_identifier()


def get_capacity(conn: sqlite3.Connection, embedder: Embedder) -> EmbedderCapacity:
    """Read the embedder's capacity row, or synthesise defaults if absent.

    `effective_max_tokens` = min(advertised, discovered) if both present,
    else whichever is present, else FALLBACK_MAX_TOKENS.
    """
    embedder_id = _embedder_id(embedder)
    row = conn.execute(
        "SELECT advertised_max_tokens, discovered_max_tokens "
        "FROM embedder_capability WHERE embedder_id = ? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (embedder_id,),
    ).fetchone()
    advertised = int(row[0]) if row and row[0] is not None else None
    discovered = int(row[1]) if row and row[1] is not None else None
    effective = _effective_max_tokens(advertised, discovered)
    return EmbedderCapacity(
        embedder_id=embedder_id,
        advertised_max_tokens=advertised,
        discovered_max_tokens=discovered,
        effective_max_tokens=effective,
        chunk_budget_chars=int(effective * CHARS_PER_TOKEN),
    )


def _effective_max_tokens(
    advertised: int | None, discovered: int | None
) -> int:
    if advertised is not None and discovered is not None:
        return min(advertised, discovered)
    if discovered is not None:
        return discovered
    if advertised is not None:
        return advertised
    return FALLBACK_MAX_TOKENS


def initialise_capacity(
    conn: sqlite3.Connection,
    embedder: Embedder,
    *,
    advertised_max_tokens: int,
    dim: int | None = None,
) -> None:
    """First-time row for this embedder. Idempotent — re-running just
    updates advertised and fetched_at, leaves discovered untouched."""
    embedder_id = _embedder_id(embedder)
    conn.execute(
        """
        INSERT INTO embedder_capability (
            embedder_id, model_hash, advertised_max_tokens, dim, fetched_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(embedder_id, model_hash) DO UPDATE SET
            advertised_max_tokens = excluded.advertised_max_tokens,
            dim = COALESCE(excluded.dim, embedder_capability.dim),
            fetched_at = excluded.fetched_at
        """,
        (embedder_id, embedder_id, advertised_max_tokens, dim, int(time.time())),
    )


def reduce_discovered_max_tokens(
    conn: sqlite3.Connection, embedder: Embedder, observed_tokens: int
) -> int:
    """Ratchet `discovered_max_tokens` down to `observed_tokens - margin`.

    Returns the new effective value. Never goes below MIN_DISCOVERED_TOKENS.
    A 10%-of-observed margin gives us breathing room so the next round
    of chunks doesn't oscillate at the boundary.
    """
    target = max(MIN_DISCOVERED_TOKENS, int(observed_tokens * 0.9))
    embedder_id = _embedder_id(embedder)
    # Upsert pattern.
    conn.execute(
        """
        INSERT INTO embedder_capability (
            embedder_id, model_hash, discovered_max_tokens,
            discovered_at, method, fetched_at
        ) VALUES (?, ?, ?, ?, 'ratchet', ?)
        ON CONFLICT(embedder_id, model_hash) DO UPDATE SET
            discovered_max_tokens = MIN(
                COALESCE(embedder_capability.discovered_max_tokens, ?),
                excluded.discovered_max_tokens
            ),
            discovered_at = excluded.discovered_at,
            method = 'ratchet'
        """,
        (
            embedder_id, embedder_id, target, int(time.time()),
            int(time.time()), target,
        ),
    )
    return target


def approx_tokens_for(text: str) -> int:
    """Cheap upper bound on token count — used to feed the ratchet when
    the embedder said 'too long' but didn't tell us how long.

    Apple Notes bodies are mostly English; the CHARS_PER_TOKEN average
    is close to accurate. We bias up by 10% for safety because we'd
    rather over-ratchet (smaller next chunks) than under (loop forever)."""
    if not text:
        return 1
    return max(1, int(len(text) / CHARS_PER_TOKEN * 1.1))
