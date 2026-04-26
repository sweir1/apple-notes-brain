"""FastMCP wiring for the Apple Notes MCP server.

All logic lives in `tools.py`; this file registers the MCP tools, resources,
prompts, and wires up capability annotations. Tool descriptions below are
written for LLM consumption — they are the only prose the model sees when
deciding to call a tool, so they explicitly document every parameter,
edge case, and known gotcha (permission prompts, attachment destruction,
locked notes, pagination flow).
"""
from __future__ import annotations

import logging
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FunctionResource
from mcp.types import ToolAnnotations
from pydantic import AnyUrl

from . import cache
from . import sqlite_reader as db
from . import tools
from .schemas import Folder, ListPage, MutationResult, NoteCreateSpec, NoteDetail, SearchPage

log = logging.getLogger("apple-notes-brain")

mcp = FastMCP("apple-notes-brain")


READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY)
def list_folders(include_counts: bool = False, include_trash: bool = False) -> list[Folder]:
    """List Notes folders. Returns: id, path (slash-joined for nested), is_trash, account, shared.

    Recently Deleted is hidden by default — pass include_trash=True to include it.
    include_counts=True adds note_count per folder.

    Fields:
    - account: iCloud / On My Mac / etc. Non-default accounts may fail writes (-1719); check before writing.
    - shared: true = collaborative (CloudKit shared zone). Writes may silently fail for read-only participants.
    """
    return tools.list_folders(include_counts=include_counts, include_trash=include_trash)


@mcp.tool(annotations=READ_ONLY)
def list_notes(
    folder_path: str | None = None,
    limit: int = tools.DEFAULT_LIST_LIMIT,
    cursor: str | None = None,
    include_trash: bool = False,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> ListPage:
    """List notes, most-recently-modified first. Bodies NOT returned (call get_note).

    Returns {results, returned, has_more, next_cursor, total_estimate}.

    - folder_path: scope to one folder + descendants (case-insensitive). Omit for all.
    - limit: 1-500 (default 20).
    - cursor: pass next_cursor from previous response to paginate.
    - include_trash: default False (excludes Recently Deleted).
    - modified_after / modified_before: ISO-8601 date or datetime, inclusive.
    """
    return tools.list_notes(
        folder_path, limit, cursor,
        include_trash=include_trash,
        modified_after=modified_after,
        modified_before=modified_before,
    )


@mcp.tool(annotations=READ_ONLY)
def search_notes(
    query: str,
    folder_path: str | None = None,
    search_body: bool = True,
    fuzzy: bool = False,
    mode: str = "substring",
    limit: int = tools.DEFAULT_SEARCH_LIMIT,
    cursor: str | None = None,
    include_body: bool = False,
    max_body_chars: int = 1200,
    include_trash: bool = False,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> SearchPage:
    """Search notes by title and/or body. Matches on extracted plaintext (formatting stripped).

    Returns the same envelope as list_notes plus `snippets` (up to 3 spans per hit) and
    `match_count`. total_estimate is None (full count too expensive).

    Modes (combine as needed):
    - mode='substring' (default) | 'regex' (Python re, IGNORECASE; invalid → ValueError).
    - fuzzy=True: token-based, order-insensitive, every token must appear. Overrides mode.
    - search_body=True (default): match against body. Set False for title-only (faster).

    Output controls:
    - include_body=True: bundles first `max_body_chars` (default 1200, max 2000) of body
      plaintext as `body_preview` on the TOP 5 results only.
    - limit: 1-100 (default 10), clamped silently.
    - cursor: pass next_cursor to paginate.
    - include_trash: default False.
    - modified_after / modified_before: ISO bounds, inclusive.

    Locked notes match by title only.
    """
    return tools.search_notes(
        query=query,
        folder_path=folder_path,
        search_body=search_body,
        fuzzy=fuzzy,
        mode=mode,  # type: ignore[arg-type]
        limit=limit,
        cursor=cursor,
        include_body=include_body,
        max_body_chars=max_body_chars,
        include_trash=include_trash,
        modified_after=modified_after,
        modified_before=modified_before,
    )


@mcp.tool(annotations=READ_ONLY)
def get_note(
    note_id: str | list[str],
    format: str = "markdown",
    fast: bool = False,
) -> NoteDetail | list[NoteDetail | MutationResult]:
    """Read one or more notes' full content.

    Single (str note_id): returns NoteDetail. Raises on failure.
    Batch (list, max 20): returns list[NoteDetail | MutationResult]. Per-item failures
    interleave as {action: 'skipped', error: str}. Batch never raises mid-way.

    format:
    - 'markdown' (default): full fidelity — headings, bold/italic/strike, lists,
      `- [x]`/`- [ ]` checklists, links, code, tables, attachment placeholders.
    - 'text': plaintext, formatting stripped.
    - 'html': raw HTML.

    fast=True: SQLite-only, <100ms, but PLAINTEXT only (no formatting). Requires
    format='text'. Default fast=False uses AppleScript (~500ms-1s/note, full fidelity).

    Response fields beyond the obvious (id/title/folder/modified/body/format):
    - pinned, locked, attachments (count), account.
    - shared: true = collaborative note (CloudKit). If you OWN the share, edits/deletes
      affect collaborators; delete_note requires confirm_shared_delete=True.
      If you're a participant, edits to read-only shares silently fail.

    Locked notes return body='[locked — unlock this note in Notes.app to read its
    contents]'. Notes in Recently Deleted are readable; include_trash is irrelevant here.
    """
    return tools.get_note(note_id, format=format, fast=fast)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=WRITE)
def create_note(
    title: str | None = None,
    body: str | None = None,
    folder_path: str | None = None,
    format: str = "markdown",
    notes: list[NoteCreateSpec] | None = None,
) -> MutationResult | list[MutationResult]:
    """Create one note OR many notes in a single call.

    Single mode: pass `title` (required) and optionally `body`. Returns one
    MutationResult.

    Batch mode: pass `notes=[{"title": ..., "body": ...}, ...]`. All notes go to
    `folder_path` with `format`. Returns list[MutationResult] in input order.
    Single AppleScript invocation regardless of N — much cheaper than calling
    create_note in a loop. Apple's CloudKit pipeline still saves each note
    individually (~150-300ms each), so wall-clock scales linearly with N, but
    bridge stress is O(1) instead of O(N).

    folder_path: target folder (case-insensitive; raises if not found). Omit for
    the default Notes folder. Cannot target Recently Deleted.

    format='markdown' (default) | 'html' | 'text'. Markdown supports headings (h1-h3),
    bold/italic/strike, ordered/unordered lists, `- [ ]/[x]` checklists, links,
    code, tables. HTML is sanitized (script/iframe/style/form stripped).

    Apple Notes round-trip caveats:
    - Checklists `- [ ]/- [x]` write as plain bullets — Apple's interactive
      checklists aren't creatable via AppleScript. (Read-back recovers state
      via the protobuf reader.)
    - Link hrefs are silently dropped — inline the URL in text if needed.
    - Inline code and fenced code blocks store identically; on read-back both
      become fenced blocks.
    - Fenced-code language hints are stripped on write.
    """
    return tools.create_note(
        title=title, body=body, folder_path=folder_path,
        format=format, notes=notes,  # type: ignore[arg-type]
    )


@mcp.tool(annotations=WRITE)
def update_note(
    note_id: str,
    body: str,
    append: bool = False,
    format: str = "markdown",
    allow_attachment_loss: bool = False,
) -> MutationResult:
    """Replace or append to a note's body. Returns {id, action: 'updated'}.

    append=False (default) replaces. append=True appends to existing body.
    format semantics + caveats match create_note (markdown/html/text; checklists,
    links, code-block round-trip).

    ⚠️ ATTACHMENT GUARD: Apple's `set body of note` silently destroys ALL
    attachments (images, sketches, PDFs). REFUSED when the note has attachments
    unless allow_attachment_loss=True (confirm with user first).

    ⚠️ TITLE side effect: Apple derives the displayed title from the first body
    line. If your new body's first line differs from the current title, the
    title silently changes. To preserve, prepend the title as the first line.

    Refuses locked notes.
    """
    return tools.update_note(
        note_id, body, append=append, format=format,  # type: ignore[arg-type]
        allow_attachment_loss=allow_attachment_loss,
    )


@mcp.tool(annotations=WRITE)
def rename_note(
    note_id: str | list[str],
    new_title: str | list[str],
) -> MutationResult | list[MutationResult]:
    """Rename one or many notes (title only — body untouched).

    Single (both str) → MutationResult. Batch (both equal-length lists, max 20)
    → list[MutationResult]; per-item failures return action='skipped' with error.
    Mixed shapes or length mismatch raise ValueError.

    ⚠️ Apple Notes derives the displayed title from the first body line, so the
    rename may revert at any subsequent operation. For a durable rename, also
    call update_note to set the first body line to match.

    Refuses locked notes.
    """
    return tools.rename_note(note_id, new_title)


@mcp.tool(annotations=WRITE)
def move_note(
    note_id: str | list[str],
    folder_path: str,
) -> MutationResult | list[MutationResult]:
    """Move one or many notes to a folder (body + attachments untouched).

    Single (str note_id) → MutationResult. Batch (list, max 20) → list[MutationResult]
    where all notes go to the SAME folder_path; per-item failures return
    action='skipped' with error.

    folder_path is case-insensitive. Refuses moves into Recently Deleted (use
    delete_note instead) and refuses locked notes.
    """
    return tools.move_note(note_id, folder_path)


@mcp.tool(annotations=WRITE)
def create_folder(name: str, parent_folder_path: str | None = None) -> MutationResult:
    """Create a folder. Returns {id, action: 'created'}.

    name: non-empty, cannot contain '/'.
    parent_folder_path: optional parent (must exist, not trash). Omit for top-level.
    Refuses on duplicate names within the same parent.
    """
    return tools.create_folder(name, parent_folder_path)


@mcp.tool(annotations=WRITE)
def rename_folder(folder_id: str, new_name: str) -> MutationResult:
    """Rename a folder. Returns {id, action: 'renamed'}.

    folder_id: short form (fNNN) or ICFolder URI. new_name: non-empty, no '/'.
    Refuses the trash folder. Notes inside are untouched.
    """
    return tools.rename_folder(folder_id, new_name)


@mcp.tool(annotations=DESTRUCTIVE)
def delete_folder(
    folder_id: str,
    allow_non_empty: bool = False,
    note_disposition: Literal["trash", "preserve"] = "trash",
    allow_orphaned_subfolders: bool = False,
    recursive: bool = False,
) -> MutationResult:
    """Delete a folder. Returns {id, action: 'deleted'}.

    ⚠️ DESTRUCTIVE for contained notes.

    Empty folder: deletes cleanly.
    Non-empty: REFUSED unless allow_non_empty=True (or recursive=True). With
    note_disposition:
      - 'trash' (default): contained notes → Recently Deleted (matches UI).
      - 'preserve': contained notes → default 'Notes' folder.

    Subfolders: by default REFUSED. Three options:
      - allow_orphaned_subfolders=True: subfolders survive as top-level folders.
      - recursive=True: walk the subtree bottom-up — every descendant folder's
        notes are moved per note_disposition, every descendant folder is deleted,
        then the target itself. Implies allow_non_empty. Capped at depth 8.
      - Delete subfolders manually first.

    Confirm with the user before recursive=True or allow_non_empty=True; state
    what's being moved and where.

    Refuses: trash folder, default 'Notes' folder.

    ⚠️ iCloud sync: deleted folders can reappear within seconds due to iCloud
    conflict resolution. If a folder returns after deletion, call delete_folder
    again, or remove it manually in Notes.app.
    """
    return tools.delete_folder(
        folder_id,
        allow_non_empty=allow_non_empty,
        note_disposition=note_disposition,
        allow_orphaned_subfolders=allow_orphaned_subfolders,
        recursive=recursive,
    )


@mcp.tool(annotations=DESTRUCTIVE)
def delete_note(note_id: str, confirm_shared_delete: bool = False) -> MutationResult:
    """Move a note to Recently Deleted. Returns {id, action: 'deleted'}.

    🔒 NEVER permanently destroys data. Every call moves the note to Recently
    Deleted (Apple auto-purges after 30 days). There is no parameter to bypass
    this — only the user can permanently empty Recently Deleted via Notes.app.

    Shared notes (NoteSummary.shared / NoteDetail.shared = true):
      - If you OWN the share, deleting moves the note to YOUR Recently Deleted
        (recoverable 30d), but tears down the share so every collaborator
        loses access. Pass confirm_shared_delete=True after asking the user.
      - If you're a participant, delete only removes it from your view; the
        owner keeps it. confirm_shared_delete is not required.

    Refuses: notes already in Recently Deleted, locked notes, unknown note_id,
    shared-note owner-delete without confirm_shared_delete=True.
    """
    return tools.delete_note(note_id, confirm_shared_delete=confirm_shared_delete)


# ---------------------------------------------------------------------------
# Resources — URI-addressable note bodies. Clients that support resources
# (Claude Desktop, Claude Code, Cursor, Continue) can @-mention these without
# burning a tool call.
# ---------------------------------------------------------------------------

@mcp.resource("notes://note/{note_id}", mime_type="text/markdown")
def note_markdown_resource(note_id: str) -> str:
    """A single Apple Notes note rendered as Markdown."""
    detail = tools.get_note(note_id, format="markdown", fast=False)
    header = f"# {detail.title}\n\n*Folder: {detail.folder} · Modified: {detail.modified}*\n\n"
    return header + (detail.body or "")


@mcp.resource("notes://note/{note_id}/html", mime_type="text/html")
def note_html_resource(note_id: str) -> str:
    """The raw HTML body of a note, as emitted by Apple Notes."""
    detail = tools.get_note(note_id, format="html", fast=False)
    return detail.body or ""


@mcp.resource("notes://note/{note_id}/text", mime_type="text/plain")
def note_text_resource(note_id: str) -> str:
    """Plaintext body (HTML stripped) of a note."""
    detail = tools.get_note(note_id, format="text", fast=False)
    return detail.body or ""


def _populate_recent_resources(count: int = 50) -> int:
    """Register the N most recent non-locked, non-trash notes as resources."""
    try:
        rows = db.recent_notes(limit=count)
    except Exception as exc:
        log.warning("could not pre-populate recent notes: %s", exc)
        return 0

    registered = 0
    for row in rows:
        if row.get("locked"):
            continue
        short = row["id"]
        title = (row.get("title") or "Untitled").strip() or "Untitled"

        def _reader(nid: str = short) -> str:
            try:
                detail = tools.get_note(nid, format="markdown", fast=False)
                header = f"# {detail.title}\n\n*Folder: {detail.folder} · Modified: {detail.modified}*\n\n"
                return header + (detail.body or "")
            except Exception as exc:  # noqa: BLE001
                return f"[could not read note {nid}: {exc}]"

        try:
            resource = FunctionResource(
                uri=AnyUrl(f"notes://recent/{short}"),
                name=title,
                title=title,
                description=f"Apple Notes: {title}",
                mime_type="text/markdown",
                fn=_reader,
            )
            mcp.add_resource(resource)
            registered += 1
        except Exception as exc:  # noqa: BLE001
            log.debug("skipping %s: %s", short, exc)
    return registered


# ---------------------------------------------------------------------------
# Prompt — a self-describing overview the LLM can surface on demand.
# ---------------------------------------------------------------------------

@mcp.prompt(title="Apple Notes server overview")
def notes_server_overview() -> str:
    """How apple-notes-brain works — architecture, data flow, ID format, limits."""
    fts = db.fts_available()
    return (
        "# apple-notes-brain — how it works\n\n"
        "**Storage.** Apple Notes keeps its live data in a SQLite DB at\n"
        "`~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite`.\n"
        "Note bodies are stored as gzip-compressed protobuf blobs (no published schema).\n\n"
        "**Reads.** `list_folders`, `list_notes`, `search_notes` query SQLite in read-only\n"
        "WAL mode — concurrent-safe with Notes.app. Sub-100ms.\n"
        f"FTS fast-path available on this machine: **{fts}**.\n\n"
        "**Body reads.** `get_note` / `get_notes` with `fast=false` (default) go through\n"
        "AppleScript — body comes back as HTML, server renders to Markdown by default.\n"
        "`fast=true` uses SQLite protobuf extraction (text-only, lossy).\n\n"
        "**Writes.** Always AppleScript. First write per session may prompt macOS for\n"
        "Automation permission — approve to avoid a 60-second timeout.\n"
        "`create_note` / `update_note` accept Markdown (default), HTML, or plain text.\n"
        "`rename_note`, `move_note`, `create_folder` are dedicated — do not try to do\n"
        "these via update_note.\n\n"
        "**Known gotchas:**\n"
        "  - `update_note` on a note with attachments silently destroys every attachment\n"
        "    unless `allow_attachment_loss=true` is passed (Apple bug). Server refuses by default.\n"
        "  - Locked notes: body never decrypted; surface as `locked: true` with an explanatory\n"
        "    sentinel; writes refused.\n"
        "  - Recently Deleted (`ZFOLDERTYPE=1`) excluded from list/search by default; pass\n"
        "    `include_trash=true` to see it.\n"
        "  - Cache staleness: fully automatic. A background thread pings Notes.app every 4s while you're actively using MCP tools, auto-pauses after 5min idle (resumes instantly on next tool call), skips ticks if Notes.app is closed (CloudKit daemons keep SQLite fresh independently), and freezes during lid-closed sleep. Every write also auto-flushes. Tunable via NOTES_MCP_REFRESH_INTERVAL, NOTES_MCP_IDLE_THRESHOLD, NOTES_MCP_AUTO_REFRESH=0.\n\n"
        "**IDs.** Short form `pNNN` (notes) and `fNNN` (folders). Full x-coredata URIs\n"
        "also accepted. Folder paths are `/`-joined; `Work` matches `Work/Clients/Acme`.\n\n"
        "**Pagination.** Both `list_notes` and `search_notes` return an envelope\n"
        "`{results, returned, has_more, next_cursor, total_estimate}`. Pass `cursor` to page.\n"
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup_log() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        log.info("apple-notes-brain starting")
        ok = cache.prewarm(timeout_s=30.0)
        log.info("apple-notes-brain AppleScript prewarm: %s", ok)
        n = _populate_recent_resources()
        log.info("apple-notes-brain registered %d recent notes as resources", n)
        started = cache.start_background_refresh()
        log.info("apple-notes-brain background auto-refresh: %s", "started" if started else "disabled or already running")
    except Exception as exc:  # noqa: BLE001
        log.warning("apple-notes-brain startup probe failed: %s", exc)


_startup_log()
