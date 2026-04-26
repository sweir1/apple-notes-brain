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
    """A condensed note row returned by list_notes and search_notes."""

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
