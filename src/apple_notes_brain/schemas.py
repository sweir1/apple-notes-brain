from __future__ import annotations

from pydantic import BaseModel, Field


class Folder(BaseModel):
    """A Notes folder with optional note count."""

    id: str
    path: str
    note_count: int | None = None
    is_trash: bool = False
    account: str | None = None
    shared: bool = False  # CloudKit collaborative folder — write ops via AppleScript may silently fail


class NoteSummary(BaseModel):
    """A condensed note row returned by list_notes / search_notes /
    semantic_search / hybrid_search.

    The five trailing fields (semantic_score, lexical_score, chunk_*)
    are only populated by the semantic + hybrid search tools — they
    stay `None` for the lexical path so older clients continue to
    parse the schema unchanged."""

    id: str
    title: str
    folder: str
    modified: str
    snippets: list[str] = Field(default_factory=list)
    match_count: int = 0
    body_preview: str | None = None
    pinned: bool = False
    locked: bool = False
    account: str | None = None
    attachments: int = 0
    shared: bool = False  # CloudKit shared note — owner can edit/delete; read-only participants silently fail

    # Semantic / hybrid search additions (v1.1). All None for lexical paths.
    semantic_score: float | None = None  # raw cosine similarity from the
                                         # semantic ranker; None when the
                                         # hit only matched via fulltext.
    lexical_score: float | None = None   # negated BM25 from the fulltext
                                         # ranker; None when the hit only
                                         # matched via the semantic ranker.
    fused_score: float | None = None     # RRF combined score for hybrid
                                         # results; None for pure-semantic
                                         # / pure-lexical paths. Higher
                                         # is better.
    chunk_excerpt: str | None = None     # ~200 chars from the matched chunk
    chunk_heading: str | None = None     # heading of the matched chunk
    # ZIDENTIFIER UUID of the underlying note. Stable across sessions and
    # safe to round-trip through the Apple Notes APIs even when local pK
    # values shift on rebuild. `id` (above) carries the short `pN` form
    # for consistency with lexical search_notes.
    z_identifier: str | None = None


class NoteDetail(BaseModel):
    """Full note content returned by get_note and get_notes."""

    id: str
    title: str
    folder: str
    modified: str
    body: str
    format: str
    pinned: bool = False
    locked: bool = False
    account: str | None = None
    attachments: int = 0
    shared: bool = False  # CloudKit shared note — owner can edit/delete; read-only participants silently fail


class SearchPage(BaseModel):
    """Paginated envelope for search_notes results."""

    results: list[NoteSummary]
    returned: int
    has_more: bool
    next_cursor: str | None
    total_estimate: int | None
    # Optional advisory message — populated by semantic_search / hybrid_search
    # when an empty result list could surprise the caller (e.g. the index
    # hasn't been built yet). Always None for lexical search_notes.
    hint: str | None = None


class ListPage(BaseModel):
    """Paginated envelope for list_notes results."""

    results: list[NoteSummary]
    returned: int
    has_more: bool
    next_cursor: str | None
    total_estimate: int | None


class NoteCreateSpec(BaseModel):
    """One note to create in a batch create_note call.

    Use the top-level title/body arguments for a single note. Pass `notes=[...]`
    with a list of these specs to create many in one call. All notes in a batch
    share the call's `folder_path` and `format`.
    """

    title: str
    body: str = ""


class MutationResult(BaseModel):
    """Outcome of a write tool call.

    action: "created" | "updated" | "renamed" | "moved" | "deleted" | "skipped".
    error: present and non-null only on per-item failures inside a batch call
    (rename_note / move_note with list input). Single-note calls raise on
    failure rather than returning a skipped result.
    """

    id: str
    action: str
    error: str | None = None
