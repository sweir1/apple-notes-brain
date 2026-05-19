from __future__ import annotations

from pydantic import BaseModel, Field


class AttachmentBucket(BaseModel):
    """One bucket of attachments classified by type.

    `destructive=True` means an AppleScript `set body` overwrite will
    annihilate the content (images, sketches, scans, audio files, and
    unknown-type attachments). `destructive=False` means the content
    is rebuilt from the new HTML/markdown body (tables).
    """

    count: int = 0
    destructive: bool = True
    utis: list[str] = Field(default_factory=list)
    filenames: list[str] = Field(default_factory=list)


class AttachmentSummary(BaseModel):
    """Per-note attachment summary with nested per-bucket detail.

    `total_destructive` is the count that the update_note guard fires
    on — only attachment types that get destroyed by a body overwrite.
    `total_reconstructable` covers tables (rebuilt from the new body).

    The `by_type` dict always has all six keys populated (even with
    `count=0`) so callers can index without missing-key handling. The
    six buckets:

      - image   ← public.jpeg, public.png, public.heic, public.svg-image
      - sketch  ← com.apple.drawing.2, com.apple.paper (PaperKit)
      - scan    ← com.apple.notes.scan, com.apple.notes.gallery,
                  or any row with non-null ZFALLBACKPDFGENERATION
      - audio   ← public.audio, public.mpeg-4-audio, com.apple.m4a-audio
      - file    ← any unknown ZTYPEUTI (user-attached PDFs, ZIPs, docs)
      - table   ← com.apple.notes.table  (NON-destructive — only one)
    """

    total_destructive: int = 0
    total_reconstructable: int = 0
    by_type: dict[str, AttachmentBucket] = Field(default_factory=dict)


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
    # Number of DESTRUCTIVE attachments (image/sketch/scan/audio/file).
    # Tables are reconstructable from the new body and do NOT count
    # toward this number — see `attachments_detail.by_type.table` for
    # the raw table count.
    # Contract shift in v1.1: this used to be the total Z_ENT=5 row
    # count (including tables), but tables triggered false positives
    # in the update_note guard.
    attachments: int = 0
    # Optional nested breakdown — populated by get_note / list_notes
    # paths that opt in. None for list contexts that want a lightweight
    # row.
    attachments_detail: "AttachmentSummary | None" = None
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
    # Destructive-only attachment count (image/sketch/scan/audio/file).
    # Contract shift in v1.1 — see NoteSummary.attachments above.
    attachments: int = 0
    # Nested per-type breakdown so the model can distinguish e.g.
    # "1 image" from "1 table" instead of seeing a bare "1".
    attachments_detail: "AttachmentSummary | None" = None
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
    verified: False iff the operation succeeded at the AppleScript layer
    but couldn't be confirmed against the SQLite store within the
    timeout window (Notes.app's MOC commits asynchronously under load).
    The change is almost certainly in flight and will commit shortly;
    callers can retry verification rather than treating it as a hard
    failure. Defaults to True.
    warning: human-readable explanation when verified=False; otherwise None.
    """

    id: str
    action: str
    error: str | None = None
    verified: bool = True
    warning: str | None = None
