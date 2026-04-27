"""Tool implementations — framework-free, typed Pydantic returns.

Hybrid data strategy:
  - SQLite (read-only, sub-100ms) for folder listing, note listing, search filter,
    metadata, cursor pagination.
  - AppleScript (osascript) for full-fidelity body read (returns HTML → Markdown)
    and every write (create / update / delete). Writes via SQLite would desync
    iCloud and corrupt the Core Data store.
"""
from __future__ import annotations

import html as _htmlmod
import re
import signal as _signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from typing import Any, Literal

from . import applescript as aps
from . import scripts
from . import search as search_mod
from . import sqlite_reader as db
from .html_text import count_matches, html_to_text, snippets
from .markdown import html_to_markdown, markdown_to_html
from .schemas import (
    Folder,
    ListPage,
    MutationResult,
    NoteCreateSpec,
    NoteDetail,
    NoteSummary,
    SearchPage,
)

DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MAX_BATCH_NOTES = 20
MAX_BODY_PREVIEW_CHARS = 2000
MAX_INCLUDE_BODY_ROWS = 5
BATCH_WORKERS = 5

# Apple Core Data stores times as seconds since 2001-01-01 UTC.
CORE_DATA_EPOCH_OFFSET = 978_307_200

BodyFormat = Literal["markdown", "text", "html"]
SearchMode = Literal["substring", "regex"]


def _tracks_activity(fn):
    """Decorator: call cache.mark_activity() at entry. Used on every public
    tool function so the background refresher can pause when MCP has been idle."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            from . import cache
            cache.mark_activity()
        except Exception:
            pass
        return fn(*args, **kwargs)
    return wrapper


# Hard wall-clock budget for any tool invocation. Defends against AppleScript
# stalls (Notes.app unresponsive, iCloud sync), SQLite hangs, and any other
# blocking call. Uses SIGALRM on POSIX main thread (which is where FastMCP
# dispatches sync tools per func_metadata.py) — the signal interrupts blocking
# C calls (subprocess.run, sqlite3) cleanly. Markdown converters have their
# own pebble-process budget for GIL-bound regex pathologies; this is the
# outer guarantee for everything else.
# Outer SIGALRM tool wall-clock cap. Most tools finish in <2s; the budget is
# defensive headroom for verify polls and bridge recovery. MUST be larger than
# MOC_COMMIT_TIMEOUT_S + setup/teardown overhead, otherwise the SIGALRM fires
# before the inner verify can finish on a stressed pipeline.
TOOL_BUDGET_S = 90
# delete_folder needs longer than typical tools for large cascades + recursive
# walks. Hard-capped at 180s (3 min) — any operation that exceeds this is
# truly stuck (iCloud unreachable, system hang, etc.) and should surface as
# an error rather than block the user/model indefinitely.
#
# Trade-off: in rare pathological cases (deeply-nested recursive delete on
# freshly-created folders during severe iCloud backpressure), we may hit the
# cap and surface a timeout. Retry usually succeeds because state has settled.
DELETE_FOLDER_BUDGET_S = 180

# Global wait-for-Notes-MOC-to-commit timeout. Notes.app's CoreData→CloudKit
# pipeline takes ~5-7s to commit a delete/move/rename to SQLite for freshly-
# created/just-modified objects, but can spike to 25-45s under stress (recent
# create + rename + write sequence on the same record/zone). 60s accommodates
# the v14 audit's worst-observed cases without surfacing user-visible timeouts.
#
# Used by every helper that polls SQLite waiting for a write to land.
#
# Distinct from:
#   - TOOL_BUDGET_S / DELETE_FOLDER_BUDGET_S: outer SIGALRM tool wall-clock
#   - subprocess timeouts in cache.py: how long we wait for osascript ITSELF
#     to respond (not for Notes.app to commit anything)
MOC_COMMIT_TIMEOUT_S = 60.0


class _ToolTimeout(BaseException):
    """Not Exception — so try/except Exception inside tools doesn't swallow it."""


def _alarm_handler(signum, frame):
    raise _ToolTimeout()


def _safe_tool(fn):
    """Decorator: convert any unexpected exception into a clean ValueError.

    FastMCP returns the string representation of any raised exception to the
    client, so a raw `AttributeError`, `TypeError`, or `AppleScriptError`
    leaking out shows up as an unhelpful internal-looking error to the model.

    This wrapper:
      - lets ValueError pass through unchanged (these are our intentional,
        informative messages — model-actionable)
      - lets _ToolTimeout pass through (the budget timeout already wraps it)
      - wraps everything else with the tool name and a helpful prefix so the
        model gets actionable context instead of a stack trace fragment

    Apply OUTSIDE @_with_tool_timeout so the timeout's own ValueError gets
    surfaced cleanly without being re-wrapped.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, _ToolTimeout):
            raise
        except aps.AppleScriptError as exc:
            # AS error that wasn't translated upstream — surface cleanly.
            raise ValueError(
                f"tool {fn.__name__!r}: AppleScript failure — {exc}. "
                "Notes.app may be unresponsive or the operation may not be "
                "supported in your current Notes state. Retry, or check "
                "Notes.app for any blocking dialogs."
            ) from exc
        except (TypeError, AttributeError) as exc:
            # Internal bug — surface with enough context to file a fix.
            raise ValueError(
                f"tool {fn.__name__!r}: internal error ({type(exc).__name__}: {exc}). "
                "This is a bug in the MCP server. Please report the tool name "
                "and arguments you used."
            ) from exc
        except Exception as exc:  # noqa: BLE001 — defensive top-level catch
            raise ValueError(
                f"tool {fn.__name__!r}: unexpected {type(exc).__name__}: {exc}. "
                "The operation may have partially completed; check Notes.app "
                "for state and retry if appropriate."
            ) from exc
    return wrapper


def _with_tool_timeout(fn=None, *, budget_s: int = TOOL_BUDGET_S):
    """Decorator: enforce wall-clock budget on a tool. Use bare or with kwargs:
        @_with_tool_timeout                 # default 30s
        @_with_tool_timeout(budget_s=120)   # override
    """
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            installed = False
            try:
                old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
                installed = True
            except (ValueError, OSError):
                return f(*args, **kwargs)
            _signal.alarm(budget_s)
            try:
                return f(*args, **kwargs)
            except _ToolTimeout:
                raise ValueError(
                    f"tool {f.__name__!r} exceeded the {budget_s}s timeout — "
                    "Apple Notes is likely unresponsive (long-running iCloud sync, "
                    "stuck UI, or a CPU-bound conversion edge case). The call has "
                    "been abandoned. Try again in a moment; if the issue persists, "
                    "restart Notes.app."
                ) from None
            finally:
                _signal.alarm(0)
                if installed:
                    try:
                        _signal.signal(_signal.SIGALRM, old_handler)
                    except (ValueError, OSError):
                        pass
        return wrapper
    if fn is not None:
        return decorate(fn)
    return decorate


def _iso_to_core_data_epoch(iso: str | None) -> float | None:
    """Parse ISO-8601 date or datetime → Core Data epoch seconds. Naive inputs
    are interpreted as local time (consistent with our 'YYYY-MM-DD HH:MM' output).
    Returns None when `iso` is falsy. Raises ValueError on invalid input.
    """
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(f"invalid ISO date: {iso!r} ({exc})")
    return dt.timestamp() - CORE_DATA_EPOCH_OFFSET


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_time(epoch: float) -> str:
    if not epoch:
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError):
        return ""


def _is_bridge_corruption(exc: aps.AppleScriptError) -> bool:
    """Detect the 'Invalid index on ICNote/ICFolder' signature that indicates
    the AppleScript bridge's NSXPCConnection state is wedged."""
    err = str(exc).lower()
    return "invalid index" in err and ("icnote" in err or "icfolder" in err)


def _aps_run_with_recovery(script: str) -> str:
    """Run an AppleScript with two-tier recovery for 'Invalid index' errors.

    Tier 1 — MOC backpressure (cheap): right after a create/move/delete,
    Notes.app's CoreData→CloudKit pipeline is busy indexing the new objects.
    AS lookups on those just-mutated objects return 'Invalid index' even
    though the URIs are valid. Wait briefly + retry once — usually enough.
    This is the root cause of v10-v12 audits' "delete failed on freshly-created
    folder" issues.

    Tier 2 — true bridge corruption (expensive): if the retry also fails,
    escalate to a full Notes.app restart via cache.recover_bridge().

    Tier 1 doesn't burn the recover_bridge() 60s cooldown, so subsequent
    operations can still recover from genuine corruption later.
    """
    try:
        return aps.run(script)
    except aps.AppleScriptError as exc:
        if not _is_bridge_corruption(exc):
            raise
        # Tier 1: MOC backpressure retry. Brief settle, no Notes.app restart.
        time.sleep(0.7)
        try:
            return aps.run(script)
        except aps.AppleScriptError as exc2:
            if not _is_bridge_corruption(exc2):
                raise
            # Tier 2: real bridge corruption. Restart Notes.app.
            from . import cache as _cache
            if not _cache.recover_bridge():
                raise
            return aps.run(script)


def _wait_until_as_addressable(uri: str, kind: str, max_wait_s: float = MOC_COMMIT_TIMEOUT_S) -> bool:
    """Poll AppleScript until the just-created object's URI is addressable.

    Notes.app's CoreData→CloudKit pipeline takes time to fully index a new
    folder/note. During that window, subsequent AS calls on the new object
    return 'Invalid index' even though the URI is valid (this is the v10-v13
    audit failure mode for create-then-immediately-delete patterns).

    This poll exits AS SOON AS Notes.app reports the URI is findable. When
    indexing is fast (settled MOC), it exits in ~50ms. When indexing is slow
    (busy MOC, iCloud uploading), it waits up to max_wait_s. Either way the
    caller is guaranteed the object is usable when this returns True — no
    arbitrary fixed sleep needed.

    `kind` should be "folder" or "note" (the AppleScript class name).
    Returns True if addressable, False on timeout. On any other AS error
    returns True (we tried — caller proceeds).
    """
    if not uri:
        return True
    probe = f'tell application "Notes" to get id of (first {kind} whose id is {aps.quote(uri)})'
    deadline = time.monotonic() + max_wait_s
    sleep_s = 0.05
    while time.monotonic() < deadline:
        try:
            aps.run(probe)
            return True  # AS can now address the new object
        except aps.AppleScriptError as exc:
            if "invalid index" not in str(exc).lower():
                return True  # different error — not an indexing issue, caller proceeds
            time.sleep(sleep_s)
            sleep_s = min(0.3, sleep_s * 1.5)  # gentle backoff
    return False


def _wait_for_state(check_fn, timeout_s: float = MOC_COMMIT_TIMEOUT_S, sleep_s: float = 0.3,
                    max_pings: int = 6) -> bool:
    """Poll `check_fn()` until it returns True or `timeout_s` elapses.

    Bridge-friendly: uses PRAGMA data_version to skip osascript pings when
    SQLite hasn't ticked, and HARD-CAPS the total number of osascript pings
    at `max_pings` regardless of how many poll iterations run. Replaces the
    legacy 25-iteration unconditional cache.refresh() loops that were
    triggering NSXPCConnection corruption (v12 audit BRIDGE-RETRY-LOOP).
    """
    if check_fn():
        return True
    deadline = time.monotonic() + timeout_s
    last_dv = -1
    pings = 0
    try:
        last_dv = db.data_version()
    except Exception:
        pass
    while time.monotonic() < deadline:
        time.sleep(sleep_s)
        try:
            cur_dv = db.data_version()
        except Exception:
            cur_dv = -1
        # Only ping the bridge when SQLite has changed AND we're under the cap.
        # On the first iteration last_dv may equal cur_dv (no writes yet) — that's
        # fine; we wait for a change rather than blindly pinging.
        if cur_dv != -1 and cur_dv != last_dv:
            last_dv = cur_dv
        elif pings < max_pings:
            try:
                from . import cache as _cache
                _cache.refresh()
            except Exception:
                pass
            pings += 1
            try:
                last_dv = db.data_version()
            except Exception:
                pass
        if check_fn():
            return True
    return check_fn()


def _translate_apple_error(
    exc: aps.AppleScriptError,
    note_id: str | None = None,
    folder_path: str | None = None,
) -> None:
    """Translate known Apple error patterns into clean ValueError. Raises if matched.
    Caller re-raises the original if this returns without raising."""
    err = str(exc).lower()
    if "invalid index" in err and ("icfolder" in err or folder_path):
        # Differentiate ghost (folder deleted on another device, cache lagging)
        # from cross-account (folder exists but in an account AppleScript cannot
        # address). The signal is the folder's account in the local cache.
        is_primary_icloud = False
        if folder_path:
            try:
                needle = folder_path.strip("/").lower()
                for f in db.list_folders():
                    if (f.get("path") or "").lower() == needle:
                        acct = (f.get("account") or "").lower()
                        is_primary_icloud = ("icloud" in acct)
                        break
            except Exception:
                pass

        if is_primary_icloud:
            raise ValueError(
                f"folder {folder_path!r} exists in the local cache but Apple Notes "
                "no longer recognises it. It was likely deleted on another device — "
                "the local cache will catch up automatically once iCloud syncs "
                "(usually within a few minutes). Try a different folder, or wait and retry."
            ) from exc
        raise ValueError(
            f"folder {folder_path!r} is in an account that AppleScript cannot reach "
            "(non-default iCloud account, shared CloudKit zone, or another account). "
            "Try a folder in your primary iCloud account."
        ) from exc
    if "password" in err or "locked" in err or "protected" in err:
        raise ValueError(f"note {note_id!r} is locked — unlock in Notes.app first") from exc
    if "recently deleted" in err or "-10000" in err:
        raise ValueError(f"note {note_id!r} cannot be modified (likely in Recently Deleted)") from exc
    if "duplicate folder name" in err:
        raise ValueError("folder already exists (per Apple Notes)") from exc
    # ICNote Invalid index — only reached if auto-recovery (restart Notes.app)
    # already ran and the retry still failed. Note likely still exists.
    if "invalid index" in err and "icnote" in err:
        raise ValueError(
            f"note {note_id!r} is unreachable via AppleScript even after a bridge restart. "
            "The note likely still exists; retry in a few seconds."
        ) from exc


def _folder_pks_for_path(folders: list[dict], folder_path: str) -> set[int]:
    needle = folder_path.strip("/").lower()
    if not needle:
        return set()
    out: set[int] = set()
    for f in folders:
        path_l = (f.get("path") or "").lower()
        if path_l == needle or path_l.startswith(needle + "/"):
            try:
                out.add(int(f["id"][1:]))
            except (ValueError, KeyError):
                pass
    return out


def _find_folder_exact(folders: list[dict], folder_path: str) -> dict | None:
    needle = folder_path.strip("/").lower()
    for f in folders:
        if (f.get("path") or "").lower() == needle:
            return f
    return None


def _folder_name_map(folders: list[dict]) -> dict[int, str]:
    out: dict[int, str] = {}
    for f in folders:
        try:
            out[int(f["id"][1:])] = f["path"]
        except (ValueError, KeyError):
            pass
    return out


def _body_to_html(body: str, format: BodyFormat) -> str:
    if not body:
        return ""
    if format == "markdown":
        return markdown_to_html(body)
    if format == "html":
        from .html_validate import normalize_html
        return normalize_html(body)
    if format == "text":
        escaped = _htmlmod.escape(body).replace("\n", "<br>")
        return f"<div>{escaped}</div>"
    raise ValueError(f"invalid format: {format!r}")


_LEADING_EMPTY_HEADING = re.compile(r"^\s*#+\s*\n+")


def _render_body(body_html: str, format: BodyFormat, pk: int | None = None) -> str:
    if format == "html":
        return body_html
    if format == "text":
        return html_to_text(body_html)
    if format == "markdown":
        md = html_to_markdown(body_html)
        # Apple Notes bodies often start with an <h1> that represents the title
        # (sometimes empty, sometimes duplicating the note's title). The title
        # is already returned as a separate field — strip the redundancy.
        md = _LEADING_EMPTY_HEADING.sub("", md)
        md = md.lstrip("\n")
        # Protobuf augmentation: recover what AppleScript HTML loses
        # (checklist state, monospace runs). Best-effort, silent on failure.
        if pk is not None:
            try:
                from . import protobuf_reader as _pb
                from .markdown import apply_style_runs
                blob = db.note_protobuf_blob(pk)
                if blob:
                    note = _pb.decode_note_protobuf(blob)
                    if note is not None:
                        runs = _pb.extract_style_runs(note)
                        text = _pb.extract_plain_text(note)
                        md = apply_style_runs(md, runs, text)
            except Exception:
                pass
        return md
    raise ValueError(f"invalid format: {format!r}")


_REGEX_PREFILTER = re.compile(r"[A-Za-z0-9]{2,}")


def _regex_prefilter_seed(pattern: str) -> str:
    """Pick a literal alphanumeric substring from a regex to pre-filter via SQL LIKE.

    Returns empty string if nothing useful can be extracted — caller should
    fall back to an unfiltered scan.
    """
    m = _REGEX_PREFILTER.search(pattern)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def _folders_with_overlays(include_counts: bool = False) -> list[dict]:
    """Internal: return raw folder dicts with cache overlays applied.

    Use this for any internal path-resolution lookup (create_note, move_note,
    create_folder, etc.) so freshly-renamed folders resolve under their NEW
    path immediately. The decorated `list_folders` tool wraps this — but
    internal callers go through here directly to avoid signal-handler nesting
    AND to ensure the overlay is consistently applied. v8 had a regression
    where internal lookups bypassed the overlay; this helper closes that gap."""
    rows = db.list_folders(include_counts=include_counts)
    try:
        from . import cache as _cache
        rows = [r for r in rows if not _cache.is_tombstoned(int(r["id"][1:]))]
        rows = _cache.apply_rename_overlay(rows)
    except Exception:
        pass
    return rows


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def list_folders(include_counts: bool = False, include_trash: bool = False) -> list[Folder]:
    rows = _folders_with_overlays(include_counts=include_counts)
    if not include_trash:
        rows = [r for r in rows if not r.get("is_trash")]
    return [
        Folder(
            id=r["id"],
            path=r.get("path") or "",
            note_count=r.get("note_count") if include_counts else None,
            is_trash=bool(r.get("is_trash")),
            account=r.get("account"),
            shared=bool(r.get("shared")),
        )
        for r in rows
    ]


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def list_notes(
    folder_path: str | None,
    limit: int,
    cursor: str | None = None,
    include_trash: bool = False,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> ListPage:
    limit = max(1, min(limit, MAX_LIST_LIMIT))

    all_folders = _folders_with_overlays()
    if folder_path:
        pks = _folder_pks_for_path(all_folders, folder_path)
        if not pks:
            raise ValueError(
                f"folder not found: {folder_path!r}. Use list_folders to see "
                "available folder paths (case-insensitive, slash-joined)."
            )
    else:
        pks = None

    fname = _folder_name_map(all_folders)
    rows, has_more, next_cursor, total = db.list_notes(
        pks, limit, cursor,
        include_trash=include_trash,
        modified_after_cd=_iso_to_core_data_epoch(modified_after),
        modified_before_cd=_iso_to_core_data_epoch(modified_before),
    )

    results = [
        NoteSummary(
            id=n["id"],
            title=n.get("title") or "",
            folder=fname.get(n.get("folder_pk") or -1, ""),
            modified=_fmt_time(n.get("modified") or 0),
            pinned=bool(n.get("pinned")),
            locked=bool(n.get("locked")),
            shared=bool(n.get("shared")),
        )
        for n in rows
    ]
    return ListPage(
        results=results,
        returned=len(results),
        has_more=has_more,
        next_cursor=next_cursor,
        total_estimate=total,
    )


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def search_notes(
    query: str,
    folder_path: str | None,
    search_body: bool = True,
    fuzzy: bool = False,
    mode: SearchMode = "substring",
    limit: int = DEFAULT_SEARCH_LIMIT,
    cursor: str | None = None,
    include_body: bool = False,
    max_body_chars: int = 1200,
    include_trash: bool = False,
    modified_after: str | None = None,
    modified_before: str | None = None,
) -> SearchPage:
    q = query.strip()
    if not q:
        return SearchPage(results=[], returned=0, has_more=False, next_cursor=None, total_estimate=None)
    if mode not in ("substring", "regex"):
        raise ValueError(f"invalid mode: {mode!r}. Supported: 'substring', 'regex'.")

    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    body_chars = max(0, min(max_body_chars, MAX_BODY_PREVIEW_CHARS))

    all_folders = _folders_with_overlays()
    if folder_path:
        pks = _folder_pks_for_path(all_folders, folder_path)
        if not pks:
            raise ValueError(
                f"folder not found: {folder_path!r}. Use list_folders to see "
                "available folder paths (case-insensitive, slash-joined)."
            )
    else:
        pks = None
    fname = _folder_name_map(all_folders)

    # Decide the DB pre-filter query and the Python-side matcher.
    tokens: list[str] = []
    matcher = None
    pool_query: str

    if fuzzy:
        tokens = search_mod.tokenize(q)
        if not tokens:
            return SearchPage(results=[], returned=0, has_more=False, next_cursor=None, total_estimate=None)
        pool_query = search_mod.selective_token(tokens)
    elif mode == "regex":
        try:
            matcher = search_mod.compile_matcher(q, "regex")
        except ValueError as exc:
            raise ValueError(str(exc))
        # Regex is applied after SQLite returns. We can't safely pre-filter via
        # SQL `LIKE` because extracting a literal seed from an arbitrary regex
        # is unreliable and produces false negatives. Pass empty string to
        # make sqlite_reader skip pre-filtering; Python regex does the real work.
        pool_query = ""
    else:
        # substring or phrase
        matcher = search_mod.compile_matcher(q, mode)
        pool_query = q

    # Fetch a pool from SQLite with some headroom for ranking / post-filter.
    # Regex has no SQL pre-filter, so it needs to scan effectively everything.
    if mode == "regex":
        pool_limit = 10_000
    else:
        pool_limit = max(limit * 3, limit + 5)
    candidates, has_more_raw, next_cursor_raw, _ = db.search_notes(
        pool_query, pks, search_body, pool_limit, cursor,
        include_trash=include_trash,
        modified_after_cd=_iso_to_core_data_epoch(modified_after),
        modified_before_cd=_iso_to_core_data_epoch(modified_before),
    )

    include_body_remaining = MAX_INCLUDE_BODY_ROWS if include_body else 0

    scored: list[tuple[float, NoteSummary]] = []
    for note, body_text in candidates:
        title = note.get("title") or ""
        hay = f"{title}\n{body_text}"

        # Relevance filter / score
        if fuzzy:
            s = search_mod.score(tokens, title, body_text)
            if s <= 0:
                continue
            highlight = search_mod.selective_token(tokens)
        elif mode == "regex":
            assert matcher is not None
            if not matcher.test(hay):
                continue
            s = 1.0
            # For regex, use the first actual match as the snippet seed so the
            # snippet window is centred on a real hit, not a literal of the pattern.
            highlight = matcher.first_match(hay) or q
        else:
            assert matcher is not None
            if not matcher.test(hay):
                continue
            s = 1.0
            highlight = matcher.literal or q

        is_locked = bool(note.get("locked"))
        if is_locked:
            # Body is encrypted and never read; match reason is title only.
            snip_list = ["[locked — title matched; body encrypted]"]
            mc = count_matches(title, highlight) if highlight else 0
        else:
            snip_list = snippets(body_text or title, highlight) if highlight else []
            mc = count_matches(hay, highlight) if highlight else 0

        preview: str | None = None
        if include_body_remaining > 0 and body_chars > 0 and body_text and not is_locked:
            preview = body_text[:body_chars]
            include_body_remaining -= 1

        scored.append((s, NoteSummary(
            id=note["id"],
            title=title,
            folder=fname.get(note.get("folder_pk") or -1, ""),
            modified=_fmt_time(note.get("modified") or 0),
            snippets=snip_list,
            match_count=mc,
            body_preview=preview,
            pinned=bool(note.get("pinned")),
            locked=bool(note.get("locked")),
            shared=bool(note.get("shared")),
        )))

    scored.sort(key=lambda x: -x[0])
    results = [row for _, row in scored[:limit]]

    return SearchPage(
        results=results,
        returned=len(results),
        has_more=has_more_raw,
        next_cursor=next_cursor_raw,
        total_estimate=None,
    )


def _get_one_note(note_id: str, format: BodyFormat, fast: bool) -> NoteDetail:
    kind, pk = db.resolve_id(note_id)
    if kind != "note":
        raise ValueError(f"not a note id: {note_id!r}")

    if fast and format != "text":
        raise ValueError("fast=True is only compatible with format='text'")

    if fast:
        return _get_note_sqlite(pk, format="text")

    return _get_note_applescript(pk, format=format)


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def get_note(
    note_id: str | list[str],
    format: BodyFormat = "markdown",
    fast: bool = False,
) -> NoteDetail | list[NoteDetail | MutationResult]:
    """Fetch one OR many notes. Accepts a single id string or a list of ids.

    - Single: pass a str → returns one NoteDetail. Raises on failure.
    - Batch: pass a list → returns list[NoteDetail | MutationResult]. Max 20 ids
      per batch. Empty list → []. Per-item failures come back as
      MutationResult(action="skipped", error=...). AppleScript path fans out
      across 5 concurrent workers; SQLite path runs sequentially (already
      sub-100ms per call).
    """
    if isinstance(note_id, list):
        if not note_id:
            return []
        if len(note_id) > MAX_BATCH_NOTES:
            raise ValueError(f"get_note: too many ids ({len(note_id)} > {MAX_BATCH_NOTES})")
        if fast:
            def _safe_fast(nid: str) -> NoteDetail | MutationResult:
                try:
                    return _get_one_note(nid, format="text", fast=True)
                except Exception as exc:
                    return MutationResult(id=nid, action="skipped", error=str(exc))
            return [_safe_fast(nid) for nid in note_id]

        def _safe_slow(nid: str) -> NoteDetail | MutationResult:
            try:
                return _get_one_note(nid, format=format, fast=False)
            except Exception as exc:
                return MutationResult(id=nid, action="skipped", error=str(exc))
        with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(note_id))) as pool:
            return list(pool.map(_safe_slow, note_id))
    return _get_one_note(note_id, format=format, fast=fast)


_LOCKED_BODY = "[locked — unlock this note in Notes.app to read its contents]"


def _locked_detail(meta: dict, pk: int, format: BodyFormat) -> NoteDetail:
    fmap = _folder_name_map(_folders_with_overlays())
    return NoteDetail(
        id=db.short_id(pk),
        title=meta.get("title") or "",
        folder=fmap.get(meta.get("folder_pk") or -1, ""),
        modified=_fmt_time(meta.get("modified") or 0),
        body=_LOCKED_BODY,
        format=format,
        pinned=bool(meta.get("pinned")),
        locked=True,
        shared=bool(meta.get("shared")),
    )


def _get_note_sqlite(pk: int, format: BodyFormat) -> NoteDetail:
    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: p{pk}")
    if meta.get("locked"):
        return _locked_detail(meta, pk, "text")
    fmap = _folder_name_map(_folders_with_overlays())
    body_text = db.note_body_text(pk)
    return NoteDetail(
        id=db.short_id(pk),
        title=meta.get("title") or "",
        folder=fmap.get(meta.get("folder_pk") or -1, ""),
        modified=_fmt_time(meta.get("modified") or 0),
        body=body_text,
        format="text",
        pinned=bool(meta.get("pinned")),
        locked=False,
        shared=bool(meta.get("shared")),
    )


def _get_note_applescript(pk: int, format: BodyFormat) -> NoteDetail:
    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: p{pk}")
    if meta.get("locked"):
        return _locked_detail(meta, pk, format)

    full_uri = db.to_uri(pk, db.store_uuid(), "ICNote")
    script = scripts.fill(scripts.GET_NOTE, NOTE_ID=aps.quote(full_uri))
    # Transient "Invalid index" (-1719) can follow a recent delete_folder while
    # SQLite still shows the note as live — retry with backoff before surfacing.
    for attempt in range(4):
        try:
            out = aps.run(script).rstrip("\n")
            break
        except aps.AppleScriptError as exc:
            msg = str(exc).lower()
            transient = "invalid index" in msg or "-1719" in msg
            if transient and attempt < 3 and db.note_meta(pk):
                time.sleep(0.25 * (2 ** attempt))
                continue
            raise
    parts = out.split(aps.UNIT_SEP)
    if len(parts) < 5:
        raise aps.AppleScriptError(f"unexpected get_note output: {out!r}")
    body_html = aps.UNIT_SEP.join(parts[4:])

    fmap = _folder_name_map(_folders_with_overlays())
    return NoteDetail(
        id=db.short_id(pk),
        title=meta.get("title") or "",
        folder=fmap.get(meta.get("folder_pk") or -1, ""),
        modified=_fmt_time(meta.get("modified") or 0),
        body=_render_body(body_html, format, pk=pk),
        format=format,
        pinned=bool(meta.get("pinned")),
        locked=False,
        attachments=db.attachment_count(pk),
        shared=bool(meta.get("shared")),
    )


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------

def _resolve_folder_uri(folder_path: str | None) -> str | None:
    """Resolve a folder path to its URI, or return None for default-folder placement."""
    if not folder_path:
        return None
    exact = _find_folder_exact(_folders_with_overlays(), folder_path)
    if exact is None:
        raise ValueError(f"folder not found: {folder_path!r}")
    return db.to_uri(int(exact["id"][1:]), db.store_uuid(), "ICFolder")


def _post_create_uri_to_short(uri: str) -> str:
    try:
        _, pk = db.resolve_id(uri)
        return db.short_id(pk)
    except ValueError:
        return uri


def _create_note_single(
    title: str,
    body: str,
    folder_path: str | None,
    format: BodyFormat,
) -> MutationResult:
    """Single-note create — one AppleScript call, returns one MutationResult."""
    html_body = _body_to_html(body, format)
    folder_uri = _resolve_folder_uri(folder_path)
    if folder_uri:
        script = scripts.fill(
            scripts.CREATE_NOTE_IN_FOLDER,
            FOLDER_ID=aps.quote(folder_uri),
            TITLE=aps.quote(title),
            BODY=aps.quote(html_body),
        )
    else:
        script = scripts.fill(
            scripts.CREATE_NOTE_DEFAULT,
            TITLE=aps.quote(title),
            BODY=aps.quote(html_body),
        )
    try:
        uri = _aps_run_with_recovery(script).strip()
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id=None, folder_path=folder_path)
        raise
    # Wait for AS-addressability AFTER create. Empirical: removing this took
    # the audit's create→immediate-delete scenario from 11s → 22s. Each AS
    # probe forces Notes.app to come up for air and process pending saves,
    # so the wait is ALSO pre-flushing the save queue — not just polling.
    # Cost: ~85ms for the typical settled case. Benefit: halves the wall-clock
    # of any create-then-immediate-mutate workflow by clearing the save queue
    # before the next operation hits the bridge.
    _wait_until_as_addressable(uri, "note", max_wait_s=MOC_COMMIT_TIMEOUT_S)
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    return MutationResult(id=_post_create_uri_to_short(uri), action="created")


def _create_notes_bulk(
    specs: list[NoteCreateSpec],
    folder_path: str | None,
    format: BodyFormat,
) -> list[MutationResult]:
    """Batch-create N notes in one AppleScript invocation.

    All notes go to the same `folder_path` and use the same `format` (per-note
    folder/format would defeat the single-tell-block batching). Returns a list
    of MutationResult, one per input spec, in the same order.
    """
    if not specs:
        return []

    titles = [s.title for s in specs]
    bodies = [_body_to_html(s.body or "", format) for s in specs]
    folder_uri = _resolve_folder_uri(folder_path)

    if folder_uri:
        script = scripts.fill(
            scripts.BULK_CREATE_NOTES_IN_FOLDER,
            FOLDER_ID=aps.quote(folder_uri),
            TITLES=aps.as_list(titles),
            BODIES=aps.as_list(bodies),
        )
    else:
        script = scripts.fill(
            scripts.BULK_CREATE_NOTES_DEFAULT,
            TITLES=aps.as_list(titles),
            BODIES=aps.as_list(bodies),
        )
    try:
        out = _aps_run_with_recovery(script)
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id=None, folder_path=folder_path)
        raise

    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass

    # Output is RECORD_SEP-joined URIs (one per successfully created note).
    uris = [u for u in out.split(aps.RECORD_SEP) if u.strip()]

    # No addressability wait — empirical testing showed it's redundant with
    # cache.sync_after_write() above. Tier 1 retry in _aps_run_with_recovery
    # handles any rare Invalid index transient on follow-up operations.

    results: list[MutationResult] = [
        MutationResult(id=_post_create_uri_to_short(u), action="created") for u in uris
    ]
    # If AppleScript silently dropped some (e.g. permission failure mid-loop),
    # pad with failure markers so the caller sees the count mismatch.
    while len(results) < len(specs):
        idx = len(results)
        results.append(MutationResult(
            id="", action="skipped",
            error=f"note {idx}: AppleScript bulk create returned no id (create likely failed)",
        ))
    return results


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def create_note(
    title: str | None = None,
    body: str | None = None,
    folder_path: str | None = None,
    format: BodyFormat = "markdown",
    notes: list[NoteCreateSpec] | None = None,
) -> MutationResult | list[MutationResult]:
    """Unified single-or-batch create.

    Single mode (default): pass `title` and `body`. Returns one MutationResult.
    Batch mode: pass `notes=[NoteCreateSpec(title=..., body=...), ...]`. All notes
    are created in `folder_path` with `format`. Returns a list[MutationResult].
    Both modes go through ONE AppleScript subprocess invocation regardless of N.
    """
    if notes is not None:
        if len(notes) == 0:
            raise ValueError(
                "notes list cannot be empty; pass at least one NoteCreateSpec "
                "or use single-note mode"
            )
        return _create_notes_bulk(notes, folder_path, format)
    if title is None:
        raise ValueError(
            "create_note requires either single-note args (title, body) or "
            "batch args (notes=[NoteCreateSpec(...), ...])"
        )
    return _create_note_single(title, body or "", folder_path, format)


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def update_note(
    note_id: str,
    body: str,
    append: bool = False,
    format: BodyFormat = "markdown",
    allow_attachment_loss: bool = False,
) -> MutationResult:
    kind, pk = db.resolve_id(note_id)
    if kind != "note":
        raise ValueError(f"not a note id: {note_id!r}")

    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: {note_id!r}")

    # SAFETY: AppleScript's `set body of note` silently deletes every attachment
    # (images, sketches, scans, PDFs) on the note. Refuse unless caller explicitly
    # acknowledges the loss. Append mode has the same underlying behaviour.
    # Check FIRST (before lock check) so a locked-AND-attachmented note shows
    # the more dangerous attachment warning; otherwise the user could see
    # "locked", unlock-and-retry, and silently lose attachments.
    if not allow_attachment_loss:
        n_att = db.attachment_count(pk)
        if n_att > 0:
            raise ValueError(
                f"refusing to update note {note_id!r}: it has {n_att} attachment(s) which "
                "Apple's AppleScript 'body' setter would silently delete. "
                "Pass allow_attachment_loss=True to override (only after confirming with the user)."
            )

    # Lock check AFTER attachment guard. Recoverable: user unlocks, retries.
    locked = meta.get("locked")
    if locked is None:
        raise ValueError(
            f"cannot determine lock state of note {note_id!r}; "
            "the SQLite read may have failed or the schema is incomplete. "
            "Refusing to update."
        )
    if locked:
        raise ValueError(f"refusing to write to locked note {note_id!r} — unlock in Notes.app first")

    # NEW: trash pre-check
    trash_pks = db.trash_folder_pks()
    if meta.get("folder_pk") in trash_pks:
        raise ValueError(
            f"refusing to update note {note_id!r} — it is in Recently Deleted. "
            "Restore it first (move it to a folder)."
        )

    html_body = _body_to_html(body, format)
    full_uri = db.to_uri(pk, db.store_uuid(), "ICNote")
    template = scripts.UPDATE_NOTE_APPEND if append else scripts.UPDATE_NOTE_REPLACE
    script = scripts.fill(
        template,
        NOTE_ID=aps.quote(full_uri),
        BODY=aps.quote(html_body),
    )
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id)
        raise
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    return MutationResult(id=db.short_id(pk), action="updated")


def _rename_one(note_id: str, new_title: str) -> MutationResult:
    kind, pk = db.resolve_id(note_id)
    if kind != "note":
        raise ValueError(f"not a note id: {note_id!r}")
    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: {note_id!r}")
    if meta.get("locked"):
        raise ValueError(f"refusing to rename locked note {note_id!r} — unlock in Notes.app first")
    trash_pks = db.trash_folder_pks()
    if meta.get("folder_pk") in trash_pks:
        raise ValueError(
            f"refusing to rename note {note_id!r} — it is in Recently Deleted. "
            "Restore it first."
        )
    if not new_title or not new_title.strip():
        raise ValueError("new_title must be non-empty")

    full_uri = db.to_uri(pk, db.store_uuid(), "ICNote")
    script = scripts.fill(
        scripts.RENAME_NOTE,
        NOTE_ID=aps.quote(full_uri),
        TITLE=aps.quote(new_title),
    )
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id)
        raise
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    return MutationResult(id=db.short_id(pk), action="renamed")


def _move_one(note_id: str, folder: dict) -> MutationResult:
    """Move one note to the already-resolved folder dict. Caller handles folder lookup once."""
    kind, pk = db.resolve_id(note_id)
    if kind != "note":
        raise ValueError(f"not a note id: {note_id!r}")
    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: {note_id!r}")
    if meta.get("locked"):
        raise ValueError(f"refusing to move locked note {note_id!r} — unlock in Notes.app first")
    trash_pks = db.trash_folder_pks()
    if meta.get("folder_pk") in trash_pks:
        raise ValueError(
            f"refusing to move note {note_id!r} — it is in Recently Deleted. "
            "Restore it first."
        )

    full_note_uri = db.to_uri(pk, db.store_uuid(), "ICNote")
    folder_uri = db.to_uri(int(folder["id"][1:]), db.store_uuid(), "ICFolder")
    script = scripts.fill(
        scripts.MOVE_NOTE,
        NOTE_ID=aps.quote(full_note_uri),
        FOLDER_ID=aps.quote(folder_uri),
    )
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id, folder_path=folder.get("path"))
        raise
    try:
        src_pk = meta.get("folder_pk") if meta else None
        dst_pk = int(folder["id"][1:]) if folder else None
        from . import cache as _cache
        if src_pk:
            _cache.adjust_note_count(int(src_pk), -1)
        if dst_pk:
            _cache.adjust_note_count(dst_pk, +1)
    except Exception:
        pass
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    return MutationResult(id=db.short_id(pk), action="moved")


def _safe(fn, fallback_id: str) -> MutationResult:
    """Wrap a single-note mutation so batch callers get a skipped/error row
    instead of an exception that kills the whole batch.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return MutationResult(id=fallback_id, action="skipped", error=str(exc))


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def rename_note(
    note_id: str | list[str],
    new_title: str | list[str],
) -> MutationResult | list[MutationResult]:
    """Rename one or many notes.

    - Single: pass strs for both; returns one MutationResult, raises on error.
    - Batch: pass lists of equal length; returns list[MutationResult] with per-item
      outcomes. Failed items come back with action='skipped' and error populated;
      the batch itself never raises mid-way.
    """
    # Shape validation
    id_is_list = isinstance(note_id, list)
    title_is_list = isinstance(new_title, list)
    if id_is_list != title_is_list:
        raise ValueError(
            "note_id and new_title must have the same shape — "
            "either both strings (single rename) or both lists of equal length (batch)"
        )

    if not id_is_list:
        return _rename_one(note_id, new_title)  # type: ignore[arg-type]

    if len(note_id) != len(new_title):  # type: ignore[arg-type]
        raise ValueError(
            f"batch rename: note_id has {len(note_id)} items, new_title has {len(new_title)} — "  # type: ignore[arg-type]
            "lengths must match"
        )
    if len(note_id) > MAX_BATCH_NOTES:  # type: ignore[arg-type]
        raise ValueError(f"batch rename: too many notes ({len(note_id)} > {MAX_BATCH_NOTES})")  # type: ignore[arg-type]
    if len(note_id) == 0:  # type: ignore[arg-type]
        return []

    pairs = list(zip(note_id, new_title))  # type: ignore[arg-type]
    with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(pairs))) as pool:
        return list(pool.map(
            lambda pair: _safe(lambda: _rename_one(pair[0], pair[1]), fallback_id=pair[0]),
            pairs,
        ))


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def move_note(
    note_id: str | list[str],
    folder_path: str,
) -> MutationResult | list[MutationResult]:
    """Move one or many notes to the same folder.

    - Single: pass a str for note_id; returns one MutationResult, raises on error.
    - Batch: pass a list; returns list[MutationResult] with per-item outcomes.
      Failed items come back with action='skipped' and error populated.

    folder_path is always a single destination — all notes in the batch land in
    the same folder.
    """
    folder = _find_folder_exact(_folders_with_overlays(), folder_path)
    if folder is None:
        raise ValueError(f"folder not found: {folder_path!r}")
    if folder.get("is_trash"):
        raise ValueError(f"refusing to move into trash ({folder_path!r}) — use delete_note instead")

    if isinstance(note_id, str):
        return _move_one(note_id, folder)

    if len(note_id) > MAX_BATCH_NOTES:
        raise ValueError(f"batch move: too many notes ({len(note_id)} > {MAX_BATCH_NOTES})")
    if len(note_id) == 0:
        return []

    with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(note_id))) as pool:
        return list(pool.map(
            lambda nid: _safe(lambda: _move_one(nid, folder), fallback_id=nid),
            note_id,
        ))


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def create_folder(name: str, parent_folder_path: str | None = None) -> MutationResult:
    """Create a new folder. AppleScript `make new folder [at parent] with properties {name:...}`.

    When parent_folder_path is given, the new folder is nested under it.
    Otherwise it is created at the top level of the default Notes account.
    """
    if not name or not name.strip():
        raise ValueError("folder name must be non-empty")
    if "/" in name:
        raise ValueError("folder name cannot contain '/' — use parent_folder_path for nesting")

    all_folders = _folders_with_overlays()
    if parent_folder_path:
        parent = _find_folder_exact(all_folders, parent_folder_path)
        if parent is None:
            raise ValueError(f"parent folder not found: {parent_folder_path!r}")
        if parent.get("is_trash"):
            raise ValueError("cannot create a folder under Recently Deleted")
        parent_path = (parent.get("path") or "").strip("/")
        target_path = f"{parent_path}/{name}" if parent_path else name
        parent_uri = db.to_uri(int(parent["id"][1:]), db.store_uuid(), "ICFolder")
        script = scripts.fill(
            scripts.CREATE_FOLDER_IN_FOLDER,
            PARENT_ID=aps.quote(parent_uri),
            NAME=aps.quote(name),
        )
    else:
        parent_path = ""
        target_path = name
        script = scripts.fill(
            scripts.CREATE_FOLDER_DEFAULT,
            NAME=aps.quote(name),
        )

    # Sibling-scope duplicate pre-check (case-insensitive). Avoids the raw
    # osascript "Duplicate folder name" error bubbling up to MCP clients.
    name_cf = name.casefold()
    for f in all_folders:
        path = (f.get("path") or "").strip("/")
        if not path:
            continue
        if parent_path:
            prefix = parent_path + "/"
            if not path.lower().startswith(prefix.lower()):
                continue
            tail = path[len(prefix):]
        else:
            tail = path
        if "/" in tail:
            continue
        if tail.casefold() == name_cf:
            raise ValueError(f"folder already exists at that path: {target_path!r}")

    try:
        uri = _aps_run_with_recovery(script).strip()
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, name)
        raise
    # No addressability wait — cache.sync_after_write() (next) triggers Notes.app's
    # MOC flush; empirical testing confirmed list_folders/create_folder-under-this/
    # create_note-inside-this all succeed without an extra wait here. Rare
    # Invalid index transient on follow-up is handled by Tier 1 retry.
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    try:
        _, pk = db.resolve_id(uri)
        short = db.short_folder_id(pk)
    except ValueError:
        short = uri
    return MutationResult(id=short, action="created")


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def rename_folder(folder_id: str, new_name: str) -> MutationResult:
    """Rename a folder. Refused for trash folder, empty/slashed names, note IDs."""
    kind, pk = db.resolve_id(folder_id)
    if kind != "folder":
        raise ValueError(f"not a folder id: {folder_id!r}")
    if not new_name or not new_name.strip():
        raise ValueError("new_name must be non-empty")
    if "/" in new_name:
        raise ValueError("folder name cannot contain '/'")

    target = None
    for f in _folders_with_overlays():
        if f["id"] == db.short_folder_id(pk):
            target = f
            break
    if target is None:
        raise ValueError(f"folder not found: {folder_id!r}")
    if target.get("is_trash"):
        raise ValueError(f"refusing to rename the trash folder ({folder_id!r})")

    folder_uri = db.to_uri(pk, db.store_uuid(), "ICFolder")
    script = scripts.fill(
        scripts.RENAME_FOLDER,
        FOLDER_ID=aps.quote(folder_uri),
        NAME=aps.quote(new_name),
    )
    _aps_run_with_recovery(script)
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass
    # Cache-layer rename overlay: surface the new name (and cascade to descendants)
    # immediately so list_folders() reflects the change before SQLite catches up.
    try:
        old_path = (target.get("path") or "").strip("/")
        parts = old_path.rsplit("/", 1)
        if len(parts) == 2:
            new_full_path = parts[0] + "/" + new_name
        else:
            new_full_path = new_name
        from . import cache as _cache
        _cache.rename_path_overlay(pk, new_full_path)
    except Exception:
        pass
    return MutationResult(id=db.short_folder_id(pk), action="renamed")


def _cascade_note_to_trash(note_zid: str, source_folder_pk: int, timeout_s: float = MOC_COMMIT_TIMEOUT_S) -> tuple[bool, str | None]:
    """Move one note to Recently Deleted; idempotent. Returns (succeeded, error_msg).

    Uses MOVE_NOTE (not DELETE_NOTE) because retrying `delete note` on a trash
    note permanently destroys it.
    """
    trash_pks = db.trash_folder_pks()
    if source_folder_pk in trash_pks:
        # Caller asked us to move a trash note to trash — already done, refuse-as-success.
        return True, None

    state = db.note_state_by_zid(note_zid)
    if state is None:
        return True, None  # already gone
    if state["folder_pk"] != source_folder_pk:
        return True, None  # already moved out
    if state["folder_pk"] in trash_pks:
        # Already in trash via some other path — do not re-issue, do not retry.
        return True, None

    if not trash_pks:
        return False, "Recently Deleted folder not found in SQLite"
    trash_pk = next(iter(trash_pks))

    note_uri = db.to_uri(state["pk"], db.store_uuid(), "ICNote")
    trash_uri = db.to_uri(trash_pk, db.store_uuid(), "ICFolder")
    script = scripts.fill(
        scripts.MOVE_NOTE,
        NOTE_ID=aps.quote(note_uri),
        FOLDER_ID=aps.quote(trash_uri),
    )
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError:
        pass  # ignore — AppleScript errors can still have side effects; verify regardless

    def _moved_out() -> bool:
        st = db.note_state_by_zid(note_zid)
        return st is None or st["folder_pk"] != source_folder_pk

    if _wait_for_state(_moved_out, timeout_s=timeout_s):
        return True, None
    return False, (
        f"note {note_zid!r} did not leave the source folder after {timeout_s}s. "
        "AppleScript reported success but SQLite hasn't reflected the move. "
        "The bridge may be stalled; do NOT retry immediately."
    )


def _cascade_note_to_folder(note_zid: str, source_folder_pk: int, dst_folder_pk: int,
                            timeout_s: float = MOC_COMMIT_TIMEOUT_S) -> tuple[bool, str | None]:
    """Move one note to dst_folder_pk via AppleScript, verify against SQLite."""
    state = db.note_state_by_zid(note_zid)
    if state is None:
        return False, "note no longer exists"
    if state["folder_pk"] == dst_folder_pk:
        return True, None  # already there
    note_uri = db.to_uri(state["pk"], db.store_uuid(), "ICNote")
    folder_uri = db.to_uri(dst_folder_pk, db.store_uuid(), "ICFolder")
    script = scripts.fill(scripts.MOVE_NOTE, NOTE_ID=aps.quote(note_uri), FOLDER_ID=aps.quote(folder_uri))
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError:
        pass

    def _arrived() -> bool:
        st = db.note_state_by_zid(note_zid)
        return st is not None and st["folder_pk"] == dst_folder_pk

    if _wait_for_state(_arrived, timeout_s=timeout_s):
        return True, None
    if db.note_state_by_zid(note_zid) is None:
        return False, "note vanished during move"
    return False, f"note {note_zid!r} did not arrive at destination after {timeout_s}s"


def _cascade_notes_bulk(note_pks: list[int], source_folder_pk: int, dst_folder_pk: int,
                        timeout_s: float | None = None) -> tuple[bool, str | None]:
    """Bulk-move every note in source_folder_pk to dst_folder_pk in ONE AppleScript call.

    Replaces N per-note osascript invocations with one. Single bulk verification
    poll via _wait_for_state — capped osascript pings, data_version gated.
    Idempotent: re-running on already-moved notes is a no-op.

    Timeout scales with note count: Notes.app commits ZFOLDER changes one-at-a-time
    through its CloudKit pipeline (~1-2s per note empirically). Fixed budget of
    base 6s + 1.5s/note covers reasonable folder sizes; clamped to a 60s ceiling
    so a runaway cascade can't block the tool's outer 30s timeout indefinitely.
    """
    if not note_pks:
        return True, None
    if source_folder_pk == dst_folder_pk:
        return True, None

    n = len(note_pks)
    if timeout_s is None:
        # MOC_COMMIT_TIMEOUT_S baseline + scale with note count, ceiling 90s.
        # Covers Apple's MOC-commit floor (~5-7s base + 1.5s per additional note).
        timeout_s = min(90.0, MOC_COMMIT_TIMEOUT_S + 1.5 * max(0, n - 1))
    # Allow more pings for larger cascades, still capped well below the
    # bridge-corruption threshold (>40 osascript calls in <10s).
    max_pings = min(12, 4 + n // 3)

    note_uris = [db.to_uri(pk, db.store_uuid(), "ICNote") for pk in note_pks]
    folder_uri = db.to_uri(dst_folder_pk, db.store_uuid(), "ICFolder")
    script = scripts.fill(
        scripts.BULK_MOVE_NOTES,
        FOLDER_ID=aps.quote(folder_uri),
        NOTE_IDS=aps.as_list(note_uris),
    )
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError:
        pass  # ignore — verify against SQLite regardless

    # Tighter sleep_s (0.2s vs default 0.3s): when SQLite IS ticking, we want to
    # catch the count==0 moment quickly. Doesn't add bridge stress because pings
    # are independently rate-limited by max_pings.
    if _wait_for_state(
        lambda: db.count_notes_in_folder(source_folder_pk) == 0,
        timeout_s=timeout_s,
        sleep_s=0.2,
        max_pings=max_pings,
    ):
        return True, None

    remaining = db.count_notes_in_folder(source_folder_pk)
    return False, (
        f"{remaining} of {n} note(s) still in source folder after {timeout_s:.1f}s. "
        "Some may have moved; check Notes.app."
    )


def _attempt_folder_delete(folder_zid: str, folder_pk: int, folder_name: str,
                           timeout_s: float = MOC_COMMIT_TIMEOUT_S) -> tuple[bool, str | None]:
    """Delete an empty folder via AppleScript, trying two approaches.

    AppleScript folder delete is unreliable: id-based may error -1728 silently,
    name-based may return 'DELETED' while no-oping. Try both, verify SQLite
    after each. Success = row gone, ZMARKEDFORDELETION=1, or ACHANGE recorded
    a delete commit (the same signal Notes.app's UI uses).

    Verify timeout is 8s by default (was 5s) — a freshly-created folder's MOC
    save can be queued behind its create save, and the delete commit then takes
    5-7s to propagate to SQLite. With 8s headroom Approach 1 usually verifies
    without falling through to Approach 2, halving the worst-case wall-clock.

    v12 BRIDGE-RETRY-LOOP fix: each verification uses _wait_for_state with a
    hard cap on osascript pings (5 per attempt → max ~12 osascript calls total).
    """
    def verify() -> bool:
        st = db.folder_state_by_zid(folder_zid)
        if st is None or st.get("marked", 0) == 1:
            return True
        try:
            return db.folder_has_delete_change(int(st["pk"]))
        except Exception:
            return False

    if verify():
        return True, None

    # Approach 1: id-based (most specific). Refresh URI from current Z_PK.
    state = db.folder_state_by_zid(folder_zid)
    if state:
        uri = db.to_uri(state["pk"], db.store_uuid(), "ICFolder")
        script = scripts.fill(scripts.DELETE_FOLDER, FOLDER_ID=aps.quote(uri))
        try:
            _aps_run_with_recovery(script)
        except aps.AppleScriptError:
            pass
        if _wait_for_state(verify, timeout_s=timeout_s, max_pings=5):
            return True, None

    # Settle pause between approaches — gives Notes.app's MOC time to commit
    # whatever Approach 1 did before we hit the bridge again.
    time.sleep(0.5)

    # Approach 2: predicate by name (silent-lie risk, verify regardless)
    fallback = f'''tell application "Notes"
    try
        delete (every folder of account "iCloud" whose name is {aps.quote(folder_name)})
    end try
end tell'''
    try:
        _aps_run_with_recovery(fallback)
    except aps.AppleScriptError:
        pass
    if _wait_for_state(verify, timeout_s=timeout_s, max_pings=5):
        return True, None

    # Approach 3 (last resort): proactive bridge recovery. v14 audit found
    # that when both AS approaches' verifies time out, the manual retry
    # eventually triggers recover_bridge() via Invalid index detection. Pre-empt
    # that — quit + relaunch Notes.app, then verify once more. Saves the user
    # from having to retry manually. Rate-limited (60s) inside recover_bridge.
    try:
        from . import cache as _cache
        if _cache.recover_bridge():
            time.sleep(1.0)
            if verify():
                return True, None
            # One last AS attempt after the restart, in case the delete needs
            # to be re-issued post-restart (some pending operations may have
            # been dropped).
            try:
                state = db.folder_state_by_zid(folder_zid)
                if state:
                    uri = db.to_uri(state["pk"], db.store_uuid(), "ICFolder")
                    script = scripts.fill(scripts.DELETE_FOLDER, FOLDER_ID=aps.quote(uri))
                    _aps_run_with_recovery(script)
                    if _wait_for_state(verify, timeout_s=5.0, max_pings=3):
                        return True, None
            except Exception:
                pass
    except Exception:
        pass

    return False, (
        f"folder {folder_name!r} could not be deleted — Notes.app accepted the "
        "delete request but its CoreData/CloudKit save hasn't propagated to SQLite "
        f"within the {2 * timeout_s + 0.5:.0f}s budget, even after a bridge restart. "
        "This usually means iCloud is currently saturated. Wait ~30s and retry, "
        "or delete manually via the Notes UI."
    )


def _delete_one_folder(
    pk: int,
    folder_path_for_msg: str,
    note_disposition: Literal["trash", "preserve"],
    dst_pk_override: int | None,
) -> None:
    """Cascade notes out of `pk` (bulk) then delete the (now-empty) folder. Raises on failure.

    Resolves the destination folder for cascade once: trash folder for 'trash',
    default 'Notes' folder for 'preserve' (or `dst_pk_override` for recursive
    siblings that share the same dst). Single bulk AppleScript call + single
    verification poll, instead of N per-note loops.
    """
    notes = db.notes_in_folder(pk)
    if notes:
        if note_disposition == "preserve":
            dst_pk = dst_pk_override if dst_pk_override is not None else db.default_folder_pk()
            if dst_pk is None:
                raise ValueError(
                    "note_disposition='preserve' requires a default 'Notes' folder, "
                    "but none was found. Try 'trash' instead."
                )
            if dst_pk == pk:
                raise ValueError(
                    "cannot 'preserve' notes — destination folder is the folder being deleted"
                )
        else:
            trash_pks = db.trash_folder_pks()
            if not trash_pks:
                raise ValueError("Recently Deleted folder not found in SQLite")
            dst_pk = next(iter(trash_pks))

        ok, err = _cascade_notes_bulk([n["pk"] for n in notes], pk, dst_pk)
        if not ok:
            raise ValueError(
                f"folder {folder_path_for_msg!r} NOT deleted: {err}. Folder left intact. "
                "Some notes may have been moved already — check Notes.app and retry."
            )

        try:
            from . import cache as _cache
            _cache.adjust_note_count(pk, -len(notes))
            _cache.adjust_note_count(dst_pk, +len(notes))
        except Exception:
            pass

    folder_zid = db.folder_zid_by_pk(pk)
    if folder_zid is None:
        return  # already gone

    folder_name = folder_path_for_msg.split("/")[-1] or folder_path_for_msg
    ok, err = _attempt_folder_delete(folder_zid, pk, folder_name)
    if not ok:
        raise ValueError(err or f"folder {folder_path_for_msg!r} deletion verification failed")

    try:
        from . import cache as _cache
        _cache.tombstone_folder(pk)
        _cache.sync_after_write()
    except Exception:
        pass


_RECURSIVE_DEPTH_CAP = 8


def _walk_folder_tree_post_order(root_pk: int, depth: int = 0) -> list[int]:
    """Return descendants + root in post-order (leaves first). Caps at _RECURSIVE_DEPTH_CAP."""
    if depth > _RECURSIVE_DEPTH_CAP:
        raise ValueError(
            f"folder nesting exceeds depth cap ({_RECURSIVE_DEPTH_CAP}). "
            "Delete deeper subtrees manually first, or in stages."
        )
    out: list[int] = []
    for child in db.child_folder_pks(root_pk):
        out.extend(_walk_folder_tree_post_order(child, depth + 1))
    out.append(root_pk)
    return out


@_safe_tool
@_with_tool_timeout(budget_s=DELETE_FOLDER_BUDGET_S)
@_tracks_activity
def delete_folder(
    folder_id: str,
    allow_non_empty: bool = False,
    note_disposition: Literal["trash", "preserve"] = "trash",
    allow_orphaned_subfolders: bool = False,
    recursive: bool = False,
) -> MutationResult:
    """Delete a folder. See server.py description for full destructive-operation contract.

    Cascade strategy: AppleScript's `delete folder` does not cascade notes
    (CoreData Folder→Notes Deny rule). We move every contained note out FIRST
    via a single bulk AppleScript call, verify against SQLite, then delete the
    folder via id-based AppleScript with name-predicate fallback.

    Modes:
      recursive=False (default):
        - Refuses if subfolders present unless allow_orphaned_subfolders=True
        - With allow_orphaned_subfolders=True, subfolders survive as top-level
      recursive=True:
        - Walks the subtree post-order (leaves first), cascade-and-deletes every
          descendant folder before removing the target. Implies allow_non_empty
          and supersedes allow_orphaned_subfolders. Hard-capped at depth 8.
    """
    kind, pk = db.resolve_id(folder_id)
    if kind != "folder":
        raise ValueError(f"not a folder id: {folder_id!r}")

    target = None
    for f in _folders_with_overlays():
        if f["id"] == db.short_folder_id(pk):
            target = f
            break
    if target is None:
        raise ValueError(f"folder not found: {folder_id!r}")
    if target.get("is_trash"):
        raise ValueError(f"refusing to delete the trash folder ({folder_id!r})")
    if db.is_default_folder(pk):
        raise ValueError(
            f"refusing to delete the default 'Notes' folder ({folder_id!r}) — "
            "this is the system-fixture top-level folder and cannot be removed."
        )
    if db.folder_is_shared(pk):
        raise ValueError(
            f"folder {folder_id!r} is a shared folder (CloudKit collaborative zone) "
            "and cannot be deleted via AppleScript — Notes.app silently no-ops the "
            "operation. Manage sharing in Notes.app: stop sharing the folder, then "
            "delete it manually, or remove yourself from the share."
        )
    if note_disposition not in ("trash", "preserve"):
        raise ValueError(
            f"invalid note_disposition {note_disposition!r}; must be 'trash' or 'preserve'"
        )

    target_path = target.get("path") or ""

    # ----- Recursive mode: bottom-up DFS -----
    if recursive:
        order = _walk_folder_tree_post_order(pk)
        # Resolve destination once for the whole walk.
        if note_disposition == "preserve":
            dst_pk = db.default_folder_pk()
            if dst_pk is None:
                raise ValueError(
                    "note_disposition='preserve' requires a default 'Notes' folder, "
                    "but none was found. Try 'trash' instead."
                )
            if dst_pk in order:
                raise ValueError(
                    "cannot 'preserve' notes — default folder is inside the subtree being deleted"
                )
        else:
            dst_pk = None  # _delete_one_folder resolves trash itself

        for sub_pk in order:
            sub_target = next(
                (f for f in _folders_with_overlays() if f["id"] == db.short_folder_id(sub_pk)),
                None,
            )
            sub_path = (sub_target.get("path") or "") if sub_target else f"f{sub_pk}"
            _delete_one_folder(sub_pk, sub_path, note_disposition, dst_pk)
            # No blind refresh between iterations — _delete_one_folder already
            # called sync_after_write at success. v12 audit BRIDGE-RETRY-LOOP:
            # blind refreshes here added 8+ extra osascript calls per recursive
            # delete on a deep tree, contributing to bridge corruption.

        return MutationResult(id=db.short_folder_id(pk), action="deleted")

    # ----- Non-recursive mode -----
    if not allow_non_empty:
        _, _, _, total = db.list_notes({pk}, 1, None)
        try:
            from . import cache as _cache
            delta = _cache.get_count_delta(pk)
            total = max(0, (total or 0) + delta)
        except Exception:
            pass
        if total and total > 0:
            raise ValueError(
                f"folder {folder_id!r} contains {total} note(s). "
                "Pass allow_non_empty=True after confirming with the user. "
                "ALL contained notes will be moved per note_disposition: "
                "'trash' (default) sends them to Recently Deleted; "
                "'preserve' moves them to the default 'Notes' folder."
            )

    children = db.child_folder_pks(pk)
    if children and not allow_orphaned_subfolders:
        raise ValueError(
            f"folder {folder_id!r} has {len(children)} child subfolder(s). "
            "Pass allow_orphaned_subfolders=True to orphan them (they survive as "
            "top-level folders), or pass recursive=True to delete the entire subtree."
        )

    _delete_one_folder(pk, target_path, note_disposition, None)

    # Strip the deleted parent's path prefix from orphaned children so
    # list_folders() reflects their new top-level position immediately.
    if children and allow_orphaned_subfolders:
        parent_path = target_path.strip("/")
        if parent_path:
            try:
                from . import cache as _cache
                for f in _folders_with_overlays():
                    try:
                        child_pk = int(f["id"][1:])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if child_pk not in children:
                        continue
                    child_path = f.get("path") or ""
                    if child_path.startswith(parent_path + "/"):
                        new_path = child_path[len(parent_path) + 1:]
                        _cache.rename_path_overlay(child_pk, new_path)
            except Exception:
                pass

    return MutationResult(id=db.short_folder_id(pk), action="deleted")


@_safe_tool
@_with_tool_timeout
@_tracks_activity
def delete_note(note_id: str, confirm_shared_delete: bool = False) -> MutationResult:
    """Move a note to Recently Deleted. NEVER permanently destroys data.

    This MCP server intentionally does not expose a permanent-delete path:
    notes always go to Recently Deleted, where Apple Notes auto-purges them
    after 30 days. If a note is already in Recently Deleted, this tool
    refuses — the only way to permanently remove a note is for the user to
    manually empty Recently Deleted in Notes.app.

    Shared-note safety:
      - If you OWN the share and call delete_note, the share is torn down for
        EVERY collaborator — they lose the note from their devices. Requires
        confirm_shared_delete=True to proceed.
      - If you're a PARTICIPANT, the note is removed from your view only;
        the owner keeps it. SQLite reflects this as the row being purged
        outright (not moved to trash) — our verifier accepts row-absence
        as success."""
    kind, pk = db.resolve_id(note_id)
    if kind != "note":
        raise ValueError(f"not a note id: {note_id!r}")
    meta = db.note_meta(pk)
    if not meta:
        raise ValueError(f"note not found: {note_id!r}")
    if meta.get("locked"):
        raise ValueError(f"refusing to delete locked note {note_id!r} — unlock in Notes.app first")

    # Hard safety: this MCP NEVER permanently deletes notes. AppleScript's
    # `delete note` on a note already in Recently Deleted permanently destroys
    # it — we refuse before we get there. Permanent purge is a user-only,
    # Notes.app-only action.
    trash_pks = db.trash_folder_pks()
    if meta.get("folder_pk") in trash_pks:
        raise ValueError(
            f"note {note_id!r} is already in Recently Deleted. This MCP server "
            "does not permanently delete notes — to remove it permanently, "
            "manually empty Recently Deleted in Notes.app (or wait for Apple's "
            "auto-purge after 30 days)."
        )

    # Shared-note guard: owner-of-share delete tears down the share for collaborators.
    share_role = db.note_share_role(pk)
    if share_role == "owner" and not confirm_shared_delete:
        raise ValueError(
            f"note {note_id!r} is shared with others and you are the OWNER. "
            "The note itself goes to YOUR Recently Deleted (recoverable for 30 days), "
            "but the share is torn down immediately and every collaborator loses "
            "access — the note disappears from their devices. Even if you restore "
            "from trash, you'd need to re-share to give them access again. "
            "Re-call delete_note with confirm_shared_delete=True to proceed, "
            "or stop sharing the note in Notes.app first to drop collaborators "
            "without affecting your copy."
        )

    full_uri = db.to_uri(pk, db.store_uuid(), "ICNote")
    script = scripts.fill(scripts.DELETE_NOTE, NOTE_ID=aps.quote(full_uri))
    try:
        _aps_run_with_recovery(script)
    except aps.AppleScriptError as exc:
        _translate_apple_error(exc, note_id)
        raise

    # Look up the note's stable ZIDENTIFIER for the verification poll. We use
    # ZIDENTIFIER (not Z_PK) because CloudKit can reassign Z_PKs mid-operation.
    note_zid = None
    try:
        with db._open() as _conn:  # internal access — we just need the ZID
            _cur = _conn.execute(
                "SELECT ZIDENTIFIER FROM ZICCLOUDSYNCINGOBJECT WHERE Z_PK = ?",
                (pk,),
            )
            _row = _cur.fetchone()
            note_zid = _row[0] if _row else None
    except Exception:
        pass

    source_folder_pk = meta.get("folder_pk")

    try:
        if source_folder_pk:
            from . import cache as _cache
            _cache.adjust_note_count(int(source_folder_pk), -1)
    except Exception:
        pass
    try:
        from . import cache
        cache.sync_after_write()
    except Exception:
        pass

    # Verify the move actually happened against SQLite. v12 audit found that on
    # a corrupted bridge, AppleScript "succeeds" silently without moving the note.
    # Without this verification we'd return {action:"deleted"} while the note is
    # still live in its original folder. Raise instead of lying.
    #
    # Three success signals (any one is enough):
    #   1. Note's row is gone from SQLite (participant-delete on shared notes)
    #   2. Note's ZFOLDER changed away from source (move to trash committed)
    #   3. ACHANGE has a delete row for the note PK (MOC may be backed up but
    #      the AS transaction committed — same trick we use for folders)
    if note_zid and source_folder_pk is not None:
        trash_pks = db.trash_folder_pks()

        def _delete_commited() -> bool:
            st = db.note_state_by_zid(note_zid)
            if st is None:
                return True  # row gone (participant-style)
            if st["folder_pk"] in trash_pks or st["folder_pk"] != source_folder_pk:
                return True  # moved to trash or anywhere else
            try:
                if db.note_has_delete_change(int(st["pk"])):
                    return True  # ACHANGE recorded the delete commit
            except Exception:
                pass
            return False

        if not _wait_for_state(_delete_commited, timeout_s=MOC_COMMIT_TIMEOUT_S, max_pings=8):
            cur = db.note_state_by_zid(note_zid)
            if cur is not None and cur["folder_pk"] == source_folder_pk:
                raise ValueError(
                    f"note {note_id!r}: AppleScript reported success but the note "
                    f"did not move to Recently Deleted within {MOC_COMMIT_TIMEOUT_S:.0f}s. "
                    "Notes.app's MOC may be busy with other operations; the move may "
                    "complete shortly. If you've just done a bulk create/move, wait a "
                    "few seconds and retry."
                )

    return MutationResult(id=db.short_id(pk), action="deleted")


