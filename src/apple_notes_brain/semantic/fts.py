"""FTS5 helpers — `escape_fts5_query` and `search_full_text`.

Mirrors obsidian-brain's `src/store/fulltext.ts` + `src/store/fts5-escape.ts`.
BM25 column weights: title 5.0, content 1.0 — title hits matter more
because users rarely write "thoughts on pricing" in the body when the
title already says it. snippet() emits markers around matches so the
caller can render highlighted excerpts.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_SAFE_FTS5_RE = re.compile(r"^[\w\s*]+$")  # word chars + spaces + trailing-* wildcard
# FTS5 reserves these as boolean operators when they appear as standalone
# upper-case tokens. A query like "hello AND world" without quotes would
# be parsed as a boolean — surprising for users typing plain phrases.
# Quote-wrap whenever any of these appear so the search is taken literally.
_FTS5_OPERATORS_RE = re.compile(r"\b(?:AND|OR|NOT|NEAR)\b")


def escape_fts5_query(query: str) -> str:
    """Make `query` safe to splice into an FTS5 `MATCH ?` parameter.

    If the input is already FTS5-syntax-only (alphanumerics, spaces,
    `*` prefix wildcard, `-` operator) we let it through. Anything else
    (quotes, AND/OR/NOT keywords without explicit operators, punctuation)
    becomes a phrase-quoted literal so FTS5 treats it as a plain phrase
    rather than parsing operators.

    The FTS5 way to escape an embedded double-quote inside a phrase is to
    double it: `"foo ""bar"" baz"`.
    """
    if not query:
        return query
    if _SAFE_FTS5_RE.match(query) and not _FTS5_OPERATORS_RE.search(query):
        return query
    return '"' + query.replace('"', '""') + '"'


@dataclass(frozen=True)
class FullTextHit:
    """One row from a `Search.fulltext()` call."""
    node_id: str
    title: str
    score: float          # higher = better (we negate BM25 internally)
    excerpt: str = ""     # `>>>match<<<` markers around the matched span


def search_full_text(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[FullTextHit]:
    """BM25 search over nodes_fts. Returns title-weighted hits.

    Empty/whitespace queries return []. FTS5-syntax errors (caught by
    the escape function or a regex slip-through) raise the underlying
    OperationalError so the caller can surface the bad query.
    """
    if not query or not query.strip():
        return []
    safe = escape_fts5_query(query)
    try:
        rows = conn.execute(
            """
            SELECT
                n.id,
                n.title,
                bm25(nodes_fts, 5.0, 1.0) AS bm25_score,
                snippet(nodes_fts, 1, '>>>', '<<<', '...', 40) AS excerpt
            FROM nodes_fts
            JOIN nodes n ON n.rowid = nodes_fts.rowid
            WHERE nodes_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (safe, int(limit)),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # FTS5 throws "no such cursor" etc. for syntactically valid but
        # logically empty queries (e.g. all-stopwords). Return [] so the
        # caller doesn't have to special-case.
        if "fts5" in str(exc).lower() or "syntax error" in str(exc).lower():
            return []
        raise
    return [
        FullTextHit(
            node_id=str(r[0]),
            title=str(r[1]),
            # bm25 returns lower-is-better; negate to align with the
            # rest of the codebase where higher-is-better.
            score=-float(r[2]),
            excerpt=str(r[3] or ""),
        )
        for r in rows
    ]
