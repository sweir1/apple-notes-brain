"""Direct read access to the Apple Notes NoteStore SQLite DB.

Fast path for list and search. Writes never go through here — Core Data
consistency, iCloud sync tokens, and the full-text index all require going
through Notes.app (AppleScript).

Opens the live DB read-only. SQLite WAL mode lets us read while Notes.app is
running without locking it.

Requires Full Disk Access for whichever process runs this module (Claude Desktop
when wired in as an MCP server, or the terminal when invoked directly).
"""
from __future__ import annotations

import base64
import gzip
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

CORE_DATA_EPOCH_OFFSET = 978_307_200  # seconds between 1970-01-01 and 2001-01-01

NOTE_STORE_PATH = Path.home() / "Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"


class NoteStoreError(RuntimeError):
    """Raised when the NoteStore DB is missing or inaccessible."""


@dataclass(frozen=True)
class FolderRow:
    pk: int
    id: str            # short id: "f<Z_PK>"
    name: str
    parent_pk: int | None
    path: str          # "/"-joined full nested path


@dataclass(frozen=True)
class NoteRow:
    pk: int
    id: str            # short id: "p<Z_PK>"
    title: str
    folder_pk: int | None
    modified: float    # Unix epoch seconds


# ---------------------------------------------------------------------------
# Short-ID helpers
# ---------------------------------------------------------------------------

def short_id(z_pk: int) -> str:
    """Return the short note ID for a given Z_PK, e.g. 'p160'."""
    return f"p{z_pk}"


def short_folder_id(z_pk: int) -> str:
    """Return the short folder ID for a given Z_PK, e.g. 'f7'."""
    return f"f{z_pk}"


def resolve_id(id_or_uri: str) -> tuple[str, int]:
    """Parse a note or folder identifier into (kind, z_pk).

    Accepted forms:
      - "p160"                             → ("note", 160)
      - "f7"                               → ("folder", 7)
      - "x-coredata://UUID/ICNote/p42"     → ("note", 42)
      - "x-coredata://UUID/ICFolder/p7"    → ("folder", 7)

    Raises ValueError on unrecognised input.
    """
    s = id_or_uri.strip()

    # Short note form: p<int>
    m = re.fullmatch(r"p(\d+)", s)
    if m:
        return ("note", int(m.group(1)))

    # Short folder form: f<int>
    m = re.fullmatch(r"f(\d+)", s)
    if m:
        return ("folder", int(m.group(1)))

    # Full x-coredata URI: x-coredata://<uuid>/<Entity>/p<pk>
    m = re.search(r"/(ICNote|ICFolder)/p(\d+)$", s)
    if m:
        entity = m.group(1)
        pk = int(m.group(2))
        kind = "note" if entity == "ICNote" else "folder"
        return (kind, pk)

    raise ValueError(f"invalid note id: {id_or_uri!r}")


def to_uri(z_pk: int, store_uuid: str, entity: str = "ICNote") -> str:
    """Build the full x-coredata URI from a Z_PK (for AppleScript consumption)."""
    return f"x-coredata://{store_uuid}/{entity}/p{z_pk}"


# ---------------------------------------------------------------------------
# Cursor pagination helpers
# ---------------------------------------------------------------------------

# Upper bound on cursor offsets. Anyone whose Apple Notes library has a
# billion notes has bigger problems than pagination. Rejecting offsets above
# this cap is defence-in-depth: a malicious or malformed cursor cannot force
# SQLite into an expensive full-index walk via a multi-billion OFFSET.
MAX_CURSOR_OFFSET = 10**9


def encode_cursor(offset: int) -> str:
    """Encode an integer offset as a URL-safe base64 cursor string.

    Rejects negative offsets — cursors are produced by our own pagination code
    and a negative offset is always a programming error.
    """
    if offset < 0:
        raise ValueError(f"invalid cursor: negative offset {offset}")
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def decode_cursor(cursor: str | None) -> int:
    """Decode a cursor string back to an integer offset.

    Returns 0 for None. Raises ValueError("invalid cursor") on bad input.

    Bounds: 0 <= offset <= MAX_CURSOR_OFFSET (10^9). Out-of-bounds cursors are
    rejected to prevent a malicious cursor from forcing SQLite into an
    expensive full-index walk via a multi-billion OFFSET.
    """
    if cursor is None:
        return 0
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
        offset = int(decoded)
    except Exception:
        raise ValueError("invalid cursor")
    if offset < 0:
        raise ValueError(f"invalid cursor: negative offset {offset}")
    if offset > MAX_CURSOR_OFFSET:
        raise ValueError(
            f"invalid cursor: offset {offset} exceeds maximum {MAX_CURSOR_OFFSET}"
        )
    return offset


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def _open(path: Path = NOTE_STORE_PATH) -> sqlite3.Connection:
    if not path.exists():
        raise NoteStoreError(f"NoteStore not found at {path}")
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    except sqlite3.OperationalError as exc:
        raise NoteStoreError(
            f"cannot open NoteStore ({exc}). If running via Claude Desktop, grant "
            "Full Disk Access to Claude in System Settings → Privacy & Security."
        ) from exc


def _store_uuid(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT Z_UUID FROM Z_METADATA LIMIT 1").fetchone()
    if not row:
        raise NoteStoreError("NoteStore Z_METADATA is empty")
    return row[0]


_uuid_cache: str | None = None


def store_uuid() -> str:
    """Cached NoteStore UUID. Needed to rebuild full x-coredata URIs for AppleScript."""
    global _uuid_cache
    if _uuid_cache is None:
        with _open() as conn:
            _uuid_cache = _store_uuid(conn)
    return _uuid_cache


def full_uri(short_id: str) -> str:
    """Expand a short id ('p160' / 'f7') back to its full x-coredata URI."""
    kind, pk = resolve_id(short_id)
    entity = "ICNote" if kind == "note" else "ICFolder"
    return to_uri(pk, store_uuid(), entity)


def _uri(store_uuid: str, entity: str, pk: int) -> str:
    return f"x-coredata://{store_uuid}/{entity}/p{pk}"


# ---------------------------------------------------------------------------
# FTS probe
# ---------------------------------------------------------------------------

def _probe_fts(conn: sqlite3.Connection) -> bool:
    """Return True if the DB has any FTS table or view."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','view') AND name LIKE '%fts%' COLLATE NOCASE"
    ).fetchall()
    return len(rows) > 0


def has_fts(conn: sqlite3.Connection) -> bool:
    """Return whether *conn* has FTS tables available."""
    return _probe_fts(conn)


def fts_available() -> bool:
    """Open a fresh connection and return whether FTS tables are present.

    Intended for the server startup log.
    """
    try:
        with _open() as conn:
            return _probe_fts(conn)
    except NoteStoreError:
        return False


# ---------------------------------------------------------------------------
# Column existence probe — for bonus metadata columns that vary across macOS
# ---------------------------------------------------------------------------

_COLS_CACHE: dict[str, set[str]] = {}


def _columns_of(conn: sqlite3.Connection, table: str) -> set[str]:
    if table not in _COLS_CACHE:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            rows = []
        _COLS_CACHE[table] = {r[1] for r in rows}
    return _COLS_CACHE[table]


def _select_optional(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    alias: str | None = None,
    table_alias: str | None = None,
) -> str:
    """Return a SELECT fragment that yields NULL when *column* is absent on *table*.

    Pass `table_alias` when the FROM clause uses an alias (e.g. `FROM FOO f`),
    so the generated fragment references `f.col` instead of `FOO.col`.
    """
    cols = _columns_of(conn, table)
    name = alias or column
    qualifier = table_alias or table
    if column in cols:
        return f"{qualifier}.{column} AS {name}"
    return f"NULL AS {name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trash_folder_pks() -> set[int]:
    """Z_PKs of folders marked as trash (ZFOLDERTYPE=1). 'Recently Deleted' on Apple Notes."""
    with _open() as conn:
        if "ZFOLDERTYPE" not in _columns_of(conn, "ZICCLOUDSYNCINGOBJECT"):
            return set()
        rows = conn.execute(
            "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT WHERE ZTITLE2 IS NOT NULL AND ZFOLDERTYPE = 1"
        ).fetchall()
    return {int(r[0]) for r in rows}


def _folder_account_column(conn: sqlite3.Connection) -> str | None:
    """Return the ZACCOUNT* column name used by folder rows on this macOS version.

    Apple Notes shards the FK across differently-numbered columns between macOS
    releases (ZACCOUNT2, ZACCOUNT3, ..., ZACCOUNT8). Probe the schema and pick
    the first one that exists *and* has a non-null value on a folder row.
    """
    cols = _columns_of(conn, "ZICCLOUDSYNCINGOBJECT")
    for candidate in ("ZACCOUNT8", "ZACCOUNT7", "ZACCOUNT6", "ZACCOUNT5",
                      "ZACCOUNT4", "ZACCOUNT3", "ZACCOUNT2", "ZACCOUNT1", "ZACCOUNT"):
        if candidate not in cols:
            continue
        row = conn.execute(
            f"SELECT 1 FROM ZICCLOUDSYNCINGOBJECT "
            f"WHERE ZTITLE2 IS NOT NULL AND {candidate} IS NOT NULL LIMIT 1"
        ).fetchone()
        if row:
            return candidate
    return None


def _account_name_column(conn: sqlite3.Connection) -> str | None:
    """Return the column holding account display names on ICAccount rows (Z_ENT=14)."""
    cols = _columns_of(conn, "ZICCLOUDSYNCINGOBJECT")
    for candidate in ("ZNAME", "ZNAME2", "ZACCOUNTNAMEFORACCOUNTLISTSORTING"):
        if candidate in cols:
            return candidate
    return None


def data_version() -> int:
    """Return SQLite's `PRAGMA data_version` — increments on every committed write
    by any process to NoteStore.sqlite. Per https://www.sqlite.org/pragma.html#pragma_data_version
    this is the canonical cross-process change-detection primitive (cheaper than
    re-running queries or AppleScript pings)."""
    with _open() as conn:
        cur = conn.execute("PRAGMA data_version")
        row = cur.fetchone()
        return int(row[0]) if row else 0


def count_notes_in_folder(folder_pk: int) -> int:
    """Count live notes (non-trash, non-deleted) in a folder."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE ZFOLDER = ? "
            "AND ZTITLE1 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0",
            (folder_pk,),
        )
        return int(cur.fetchone()[0])


def notes_in_folder(folder_pk: int) -> list[dict]:
    """Return live notes inside a folder as [{pk, zid, title}, ...].

    Used by delete_folder's cascade to enumerate notes that need to be
    moved or deleted before the folder itself can be removed (Apple's
    AppleScript `delete folder` does NOT cascade — it silently no-ops
    when the folder has notes pointing at it via the Folder→Notes
    Deny relationship rule)."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT Z_PK, ZIDENTIFIER, COALESCE(ZTITLE1, '') "
            "FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE ZFOLDER = ? "
            "AND ZTITLE1 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0",
            (folder_pk,),
        )
        return [{"pk": int(r[0]), "zid": r[1], "title": r[2]} for r in cur.fetchall()]


def default_folder_pk() -> int | None:
    """Return the Z_PK of the user's default 'Notes' folder.

    Identified by the ZIDENTIFIER prefix 'DefaultFolder-' (e.g. 'DefaultFolder-CloudKit').
    This is locale-agnostic — works whether the folder is named 'Notes', 'Notas', etc.
    Returns None if no default folder is found (rare; means iCloud Notes not set up)."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE ZIDENTIFIER LIKE 'DefaultFolder-%' "
            "AND ZTITLE2 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0 "
            "ORDER BY Z_PK LIMIT 1"
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def is_default_folder(folder_pk: int) -> bool:
    """True if the folder is the user's default 'Notes' folder (DefaultFolder-* zid)."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT 1 FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE Z_PK = ? AND ZIDENTIFIER LIKE 'DefaultFolder-%'",
            (folder_pk,),
        )
        return cur.fetchone() is not None


def note_state_by_zid(zid: str) -> dict | None:
    """Look up a note by stable ZIDENTIFIER (UUID), returning current state.

    Returns {pk, folder_pk, marked} or None if not found. Used by delete_folder's
    cascade to verify each AppleScript step actually moved the note — the Z_PK
    can be reassigned by CloudKit mid-operation, but ZIDENTIFIER is stable."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT Z_PK, ZFOLDER, COALESCE(ZMARKEDFORDELETION, 0) "
            "FROM ZICCLOUDSYNCINGOBJECT WHERE ZIDENTIFIER = ? AND ZTITLE1 IS NOT NULL",
            (zid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"pk": int(row[0]), "folder_pk": row[1], "marked": int(row[2])}


def folder_zid_by_pk(folder_pk: int) -> str | None:
    """Return the stable ZIDENTIFIER (UUID) for a folder by its current Z_PK,
    or None if the folder doesn't exist. Used to anchor verification against
    Z_PK reassignment by CloudKit."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT ZIDENTIFIER FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE Z_PK = ? AND ZTITLE2 IS NOT NULL",
            (folder_pk,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def folder_state_by_zid(zid: str) -> dict | None:
    """Look up a folder by stable ZIDENTIFIER, returning current state.

    Returns {pk, marked} or None if the folder no longer exists in SQLite.
    Used by delete_folder verification — Z_PK can be reassigned, ZIDENTIFIER is stable."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT Z_PK, COALESCE(ZMARKEDFORDELETION, 0) "
            "FROM ZICCLOUDSYNCINGOBJECT WHERE ZIDENTIFIER = ? AND ZTITLE2 IS NOT NULL",
            (zid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"pk": int(row[0]), "marked": int(row[1])}


# Z_ENT for ICFolder per Z_PRIMARYKEY. Same on every macOS version observed.
_ENT_ICFOLDER = 15
# NSPersistentHistoryChangeType.delete raw value.
_PHCT_DELETE = 2


def _live_folder_subquery(conn: sqlite3.Connection) -> str:
    """Subquery returning Z_PKs of currently-live folders.

    Combines: ZTITLE2 NOT NULL, ZMARKEDFORDELETION=0, AND no ACHANGE delete
    commit. Used by list_notes / search_notes / child_folder_pks to filter
    out notes/children whose parent folder was just AppleScript-deleted but
    SQLite hasn't propagated yet. Matches Notes.app's UI semantics."""
    try:
        conn.execute("SELECT 1 FROM ACHANGE WHERE ZENTITY=15 AND ZCHANGETYPE=2 LIMIT 1")
        # Skip ACHANGE-delete filter for folders that have a CloudKit server record
        # (ZSERVERRECORDDATA NOT NULL) — those are confirmed-live by the cloud, so
        # any stale delete-commit in ACHANGE is leftover migration noise. See
        # list_folders() for the full rationale.
        return (
            "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT outer_f "
            "WHERE outer_f.ZTITLE2 IS NOT NULL "
            "AND COALESCE(outer_f.ZMARKEDFORDELETION, 0) = 0 "
            f"AND (outer_f.ZSERVERRECORDDATA IS NOT NULL "
            f"     OR NOT EXISTS (SELECT 1 FROM ACHANGE a "
            f"                    WHERE a.ZENTITY = {_ENT_ICFOLDER} AND a.ZCHANGETYPE = {_PHCT_DELETE} "
            f"                    AND a.ZENTITYPK = outer_f.Z_PK))"
        )
    except Exception:
        return (
            "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE ZTITLE2 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0"
        )


def note_share_role(note_pk: int) -> str:
    """Return 'unshared', 'owner', or 'participant' for a note's CKShare role.

    Heuristic per Apple Notes scripting research:
      - ZSERVERSHAREDATA IS NULL              → 'unshared'
      - ZSERVERSHAREDATA NOT NULL, ZZONEOWNERNAME IS NULL  → 'owner' (you created the share)
      - ZSERVERSHAREDATA NOT NULL, ZZONEOWNERNAME NOT NULL → 'participant' (someone else owns the zone)

    Critical for delete_note safety:
      - 'owner' delete via AppleScript → moves to YOUR trash + tears down the share for ALL collaborators
      - 'participant' delete via AppleScript → local removal only; owner keeps the note; row is purged outright (no trash row)
    """
    try:
        with _open() as conn:
            cols = _columns_of(conn, "ZICCLOUDSYNCINGOBJECT")
            zone_col = "ZZONEOWNERNAME" if "ZZONEOWNERNAME" in cols else "NULL"
            row = conn.execute(
                f"SELECT (ZSERVERSHAREDATA IS NOT NULL) AS shared, "
                f"({zone_col} IS NOT NULL) AS in_others_zone "
                "FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE Z_PK = ? AND ZTITLE1 IS NOT NULL",
                (note_pk,),
            ).fetchone()
            if not row:
                return "unshared"
            shared, in_others_zone = row
            if not shared:
                return "unshared"
            return "participant" if in_others_zone else "owner"
    except Exception:
        return "unshared"  # degrade safely — caller proceeds via existing flow


def note_is_shared(note_pk: int) -> bool:
    """True if this note is shared via CloudKit (CKShare).

    Shared notes can be read by any participant; writes succeed for owner +
    read-write participants but silently fail for read-only participants.
    Apple Notes provides no AppleScript API to query the role, so we expose
    `shared=True` on the note response and let callers decide whether to
    proceed (and rely on our post-write verification to catch silent fails).
    """
    try:
        with _open() as conn:
            cur = conn.execute(
                "SELECT 1 FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE Z_PK = ? AND ZTITLE1 IS NOT NULL "
                "AND ZSERVERSHAREDATA IS NOT NULL",
                (note_pk,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def folder_is_shared(folder_pk: int) -> bool:
    """True if this folder is a CloudKit shared zone (collaborative folder).

    Shared folders cannot be deleted, renamed, or have notes moved out via
    AppleScript — Notes.app's scripting layer silently no-ops on participant
    writes and read-only shares. Detect upfront and refuse with a clear
    error rather than letting the verification poll surface a confusing
    "X notes still in source" failure.

    Signals (per Apple Notes research):
      - ZSERVERSHAREDATA: present on the share anchor itself (the shared folder)
      - ZSHARETARGETMANAGEDOBJECTID: FK on descendants pointing to the anchor
    Either non-null → folder is part of a shared zone.
    """
    try:
        with _open() as conn:
            cols = _columns_of(conn, "ZICCLOUDSYNCINGOBJECT")
            tgt_clause = (
                "OR ZSHARETARGETMANAGEDOBJECTID IS NOT NULL"
                if "ZSHARETARGETMANAGEDOBJECTID" in cols
                else ""
            )
            cur = conn.execute(
                f"SELECT 1 FROM ZICCLOUDSYNCINGOBJECT "
                f"WHERE Z_PK = ? AND ZTITLE2 IS NOT NULL "
                f"AND (ZSERVERSHAREDATA IS NOT NULL {tgt_clause})",
                (folder_pk,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False  # column absent or query failed — degrade silently


_ENT_ICNOTE = 12  # per Z_PRIMARYKEY: ICNote


def note_has_delete_change(note_pk: int) -> bool:
    """Mirror of folder_has_delete_change for notes (Z_ENT=12=ICNote).

    True if Apple's NSPersistentHistory log records a delete commit for this
    note. Useful as a fallback verification signal when the MOC save is
    queued behind other operations and ZFOLDER hasn't yet flipped to trash.
    """
    try:
        with _open() as conn:
            cur = conn.execute(
                "SELECT 1 FROM ACHANGE WHERE ZENTITY = ? AND ZENTITYPK = ? "
                "AND ZCHANGETYPE = ? LIMIT 1",
                (_ENT_ICNOTE, note_pk, _PHCT_DELETE),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def folder_has_delete_change(folder_pk: int) -> bool:
    """True if Apple's NSPersistentHistory log (ACHANGE) records a delete commit
    for this folder PK. This is the same signal Notes.app's UI consumes via
    NSFetchedResultsController, so it agrees with what the user sees in the app
    well before ZMARKEDFORDELETION flips on the row.

    Apple prunes ACHANGE after ~7 days; long-deleted folders rely on the
    existing ZMARKEDFORDELETION filter instead. Returns False on any error
    (older macOS without ACHANGE, locked DB, etc) — caller falls back to the
    existing filters and accepts the visibility lag.
    """
    try:
        with _open() as conn:
            cur = conn.execute(
                "SELECT 1 FROM ACHANGE WHERE ZENTITY = ? AND ZENTITYPK = ? "
                "AND ZCHANGETYPE = ? LIMIT 1",
                (_ENT_ICFOLDER, folder_pk, _PHCT_DELETE),
            )
            return cur.fetchone() is not None
    except Exception:  # OperationalError, schema drift, locked DB, IO — degrade silently
        return False


def child_folder_pks(parent_pk: int) -> list[int]:
    """Return Z_PKs of folder rows whose ZPARENT == parent_pk and currently live.

    Excludes children with ACHANGE delete commits (Apple's UI hides them already)
    so a parent whose only children were just AppleScript-deleted doesn't trigger
    the "has subfolders" guard during cascade.
    """
    with _open() as conn:
        try:
            conn.execute("SELECT 1 FROM ACHANGE WHERE ZENTITY=15 AND ZCHANGETYPE=2 LIMIT 1")
            history_filter = (
                f" AND (ZICCLOUDSYNCINGOBJECT.ZSERVERRECORDDATA IS NOT NULL "
                f"      OR NOT EXISTS (SELECT 1 FROM ACHANGE a "
                f"                     WHERE a.ZENTITY = {_ENT_ICFOLDER} AND a.ZCHANGETYPE = {_PHCT_DELETE} "
                f"                     AND a.ZENTITYPK = ZICCLOUDSYNCINGOBJECT.Z_PK))"
            )
        except Exception:
            history_filter = ""
        try:
            cur = conn.execute(
                "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE ZPARENT = ? "
                "AND ZTITLE2 IS NOT NULL "
                "AND COALESCE(ZMARKEDFORDELETION, 0) = 0"
                + history_filter,
                (parent_pk,),
            )
            return [int(r[0]) for r in cur.fetchall()]
        except Exception:
            # Fallback if the history_filter clause caused trouble
            cur = conn.execute(
                "SELECT Z_PK FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE ZPARENT = ? "
                "AND ZTITLE2 IS NOT NULL "
                "AND COALESCE(ZMARKEDFORDELETION, 0) = 0",
                (parent_pk,),
            )
            return [int(r[0]) for r in cur.fetchall()]


def list_ghost_folders() -> list[dict]:
    """Return folders that exist as SQLite rows but Notes.app's UI/AppleScript can't see.

    Identified by ZSERVERRECORDDATA IS NULL AND ZNEEDSINITIALFETCHFROMCLOUD=1
    — i.e. local placeholders waiting for a CloudKit fetch that will never come
    (server-side record was lost, account toggled, share revoked, etc.). They're
    filtered out of list_folders() so they don't pollute normal output, but
    surfaced here for transparency / cleanup awareness."""
    with _open() as conn:
        cur = conn.execute(
            "SELECT Z_PK, ZTITLE2, ZIDENTIFIER FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE ZTITLE2 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0 "
            "AND ZSERVERRECORDDATA IS NULL "
            "AND COALESCE(ZNEEDSINITIALFETCHFROMCLOUD, 0) = 1"
        )
        return [
            {"id": short_folder_id(int(r[0])), "name": r[1], "zid": r[2]}
            for r in cur.fetchall()
        ]


def list_folders(include_counts: bool = False) -> list[dict]:
    """Return every non-deleted folder with its full nested path.

    Each dict has keys: id (short 'f<pk>'), name, parent_pk, path, is_trash,
    account (display name of the owning ICAccount, or None if unresolvable).
    When include_counts=True, also includes note_count.
    """
    with _open() as conn:
        acct_col = _folder_account_column(conn)
        acct_name_col = _account_name_column(conn)
        acct_select = f"f.{acct_col}" if acct_col else "NULL"

        # Ghost-row filter: a folder row whose ZSERVERRECORDDATA is NULL AND
        # ZNEEDSINITIALFETCHFROMCLOUD=1 is a stale CloudKit placeholder — Notes.app's
        # MOC hides it (waiting for a server fetch that will never arrive), AppleScript
        # inherits the same filter, the user can't see it in the UI. Mirror that.
        # Locally-created not-yet-synced folders have ZNEEDSINITIALFETCHFROMCLOUD=0
        # so they pass the filter while waiting on the upstream sync.
        ghost_filter = (
            "AND (f.ZSERVERRECORDDATA IS NOT NULL "
            "     OR COALESCE(f.ZNEEDSINITIALFETCHFROMCLOUD, 0) = 0)"
        )
        # Persistent-history filter: hide folders that have a delete commit
        # recorded in ACHANGE but haven't yet had ZMARKEDFORDELETION set on
        # the row (the lag window where Notes.app's UI already hides them but
        # our SQLite-direct read would surface a ghost). ACHANGE is the same
        # NSPersistentHistory log Apple's UI consumes. Degrades gracefully on
        # older macOS where the table is absent or has a different schema.
        history_filter = ""
        try:
            conn.execute("SELECT 1 FROM ACHANGE WHERE ZENTITY=15 AND ZCHANGETYPE=2 LIMIT 1")
            # Apply ACHANGE-delete filter only to folders WITHOUT a CloudKit server
            # record. If ZSERVERRECORDDATA is set, the folder is confirmed to exist
            # on the CloudKit server side, so any stale delete-commit in ACHANGE is
            # a leftover from a past migration/account-toggle, not a real deletion.
            # Without this guard, system folders (DefaultFolder-CloudKit /
            # TrashFolder-CloudKit) and any user folder that survived an iCloud
            # reset get incorrectly hidden — observed on real-world databases.
            history_filter = (
                f"AND (f.ZSERVERRECORDDATA IS NOT NULL "
                f"     OR NOT EXISTS (SELECT 1 FROM ACHANGE a "
                f"                    WHERE a.ZENTITY = {_ENT_ICFOLDER} "
                f"                    AND a.ZENTITYPK = f.Z_PK "
                f"                    AND a.ZCHANGETYPE = {_PHCT_DELETE}))"
            )
        except Exception:
            pass

        def _query(hist: str) -> list:
            if include_counts:
                return conn.execute(f"""
                    SELECT f.Z_PK, f.ZTITLE2, f.ZPARENT,
                           COALESCE(cnt.note_count, 0) AS note_count,
                           {acct_select} AS account_pk,
                           (f.ZSERVERSHAREDATA IS NOT NULL) AS shared
                    FROM ZICCLOUDSYNCINGOBJECT f
                    LEFT JOIN (
                        SELECT ZFOLDER, COUNT(*) AS note_count
                        FROM ZICCLOUDSYNCINGOBJECT
                        WHERE ZTITLE1 IS NOT NULL
                          AND COALESCE(ZMARKEDFORDELETION, 0) = 0
                        GROUP BY ZFOLDER
                    ) cnt ON cnt.ZFOLDER = f.Z_PK
                    WHERE f.ZTITLE2 IS NOT NULL
                      AND COALESCE(f.ZMARKEDFORDELETION, 0) = 0
                      {ghost_filter}
                      {hist}
                """).fetchall()
            return conn.execute(f"""
                SELECT f.Z_PK, f.ZTITLE2, f.ZPARENT,
                       {acct_select} AS account_pk,
                       (f.ZSERVERSHAREDATA IS NOT NULL) AS shared
                FROM ZICCLOUDSYNCINGOBJECT f
                WHERE f.ZTITLE2 IS NOT NULL
                  AND COALESCE(f.ZMARKEDFORDELETION, 0) = 0
                  {ghost_filter}
                  {hist}
            """).fetchall()

        try:
            raw = _query(history_filter)
        except Exception:
            # If the history_filter clause caused trouble (schema drift, etc.)
            # fall back to the un-history-filtered query so list_folders never
            # breaks. We may surface a ghost folder in the lag window, which
            # is the prior behaviour — strictly no regression.
            raw = _query("") if history_filter else []

        # Build account_pk -> name map (Z_ENT=14 is ICAccount on all known versions).
        account_names: dict[int, str] = {}
        if acct_name_col:
            for ap, name in conn.execute(
                f"SELECT Z_PK, {acct_name_col} FROM ZICCLOUDSYNCINGOBJECT "
                "WHERE Z_ENT = 14"
            ).fetchall():
                if name is not None:
                    account_names[int(ap)] = str(name)

    by_pk: dict[int, tuple] = {r[0]: r for r in raw}
    # Row shape:
    # include_counts=True  → (pk, name, parent, note_count, account_pk, shared)
    # include_counts=False → (pk, name, parent, account_pk, shared)
    acct_idx = 4 if include_counts else 3
    shared_idx = 5 if include_counts else 4

    trash_pks = trash_folder_pks()

    def path_of(pk: int, seen: set[int] | None = None) -> str:
        seen = seen or set()
        if pk in seen or pk not in by_pk:
            return ""
        seen.add(pk)
        row = by_pk[pk]
        name = row[1]
        parent = row[2]
        if parent and parent in by_pk:
            return path_of(parent, seen) + "/" + name
        return name

    result = []
    for pk, row in by_pk.items():
        name = row[1]
        parent = row[2]
        account_pk = row[acct_idx]
        account = account_names.get(int(account_pk)) if account_pk is not None else None
        d: dict = {
            "id": short_folder_id(pk),
            "name": name,
            "parent_pk": parent,
            "path": path_of(pk),
            "is_trash": pk in trash_pks,
            "account": account,
            "shared": bool(row[shared_idx] or 0),
        }
        if include_counts:
            d["note_count"] = row[3]
        result.append(d)
    return result


def list_notes(
    folder_pks: set[int] | None,
    limit: int,
    cursor: str | None = None,
    include_trash: bool = False,
    modified_after_cd: float | None = None,
    modified_before_cd: float | None = None,
) -> tuple[list[dict], bool, str | None, int | None]:
    """Most-recently-modified notes first, optionally scoped to a folder set.

    Returns (rows, has_more, next_cursor, total_estimate).
    Each row dict has keys: id (short 'p<pk>'), title, folder_pk, modified,
    pinned (bool), locked (bool). pinned/locked default to False if the
    underlying columns aren't present on this macOS version.

    include_trash=False (default) excludes notes whose folder is the 'Recently
    Deleted' trash container (ZFOLDERTYPE=1).
    modified_after_cd / modified_before_cd are inclusive bounds in Core Data
    epoch (seconds since 2001-01-01 UTC).
    """
    offset = decode_cursor(cursor)
    fetch_limit = limit + 1
    params_where: list[object] = []

    with _open() as conn:
        # Orphan-note filter: require ZFOLDER to reference a currently-LIVE folder.
        # Uses the centralized helper which also excludes ACHANGE-deleted folders
        # so notes inside a just-AppleScript-deleted folder don't surface as live.
        live_folder_sub = _live_folder_subquery(conn)
        base_where = f"""
            WHERE ZTITLE1 IS NOT NULL
              AND COALESCE(ZMARKEDFORDELETION, 0) = 0
              AND ZFOLDER IN ({live_folder_sub})
        """
        if folder_pks:
            placeholders = ",".join("?" * len(folder_pks))
            base_where += f" AND ZFOLDER IN ({placeholders})"
            params_where.extend(folder_pks)
        elif not include_trash:
            trash = trash_folder_pks()
            if trash:
                placeholders = ",".join("?" * len(trash))
                base_where += f" AND (ZFOLDER IS NULL OR ZFOLDER NOT IN ({placeholders}))"
                params_where.extend(trash)
        if modified_after_cd is not None:
            base_where += " AND ZMODIFICATIONDATE1 >= ?"
            params_where.append(modified_after_cd)
        if modified_before_cd is not None:
            base_where += " AND ZMODIFICATIONDATE1 <= ?"
            params_where.append(modified_before_cd)

        count_sql = f"SELECT COUNT(*) FROM ZICCLOUDSYNCINGOBJECT {base_where}"

        pinned_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPINNED", "pinned")
        pp_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPASSWORDPROTECTED", "lock_pp")
        iv_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZCRYPTOINITIALIZATIONVECTOR", "lock_iv")
        mode_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZLOCKEDNOTESMODE", "lock_mode")
        data_sql = f"""
            SELECT Z_PK, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1,
                   {pinned_col}, {pp_col}, {iv_col}, {mode_col},
                   (ZSERVERSHAREDATA IS NOT NULL) AS shared
            FROM ZICCLOUDSYNCINGOBJECT
            {base_where}
            ORDER BY ZMODIFICATIONDATE1 DESC
            LIMIT ? OFFSET ?
        """
        total: int = conn.execute(count_sql, params_where).fetchone()[0]
        rows_raw = conn.execute(
            data_sql, params_where + [fetch_limit, offset]
        ).fetchall()

    has_more = len(rows_raw) > limit
    if has_more:
        rows_raw = rows_raw[:limit]

    next_cursor: str | None = encode_cursor(offset + limit) if has_more else None

    rows = [
        {
            "id": short_id(pk),
            "title": title,
            "folder_pk": folder_pk,
            "modified": (modified or 0) + CORE_DATA_EPOCH_OFFSET,
            "pinned": bool(pinned or 0),
            "locked": bool(lock_pp or 0) or (lock_iv is not None) or (lock_mode is not None),
            "shared": bool(shared or 0),
        }
        for pk, title, folder_pk, modified, pinned, lock_pp, lock_iv, lock_mode, shared in rows_raw
    ]

    return rows, has_more, next_cursor, total


def search_notes(
    query: str,
    folder_pks: set[int] | None,
    search_body: bool,
    limit: int,
    cursor: str | None = None,
    include_trash: bool = False,
    modified_after_cd: float | None = None,
    modified_before_cd: float | None = None,
) -> tuple[list[tuple[dict, str]], bool, str | None, int | None]:
    """Return (row_dict, body_text) pairs for matches.

    `body_text` is the decompressed plaintext extracted from the protobuf blob —
    good for snippets, not full rendering (use AppleScript `get_note` for that).

    Returns (matches, has_more, next_cursor, total_estimate).
    total_estimate is None — a full-scan count would be as expensive as the search.

    # TODO: when has_fts(conn) is True, prefer a MATCH query against the FTS
    # table for speed instead of the current full-scan approach. Requires
    # verifying the FTS table name and column layout at runtime.
    """
    offset = decode_cursor(cursor)

    params: list[object] = []
    needle = query.lower()
    needle_bytes = needle.encode("utf-8")

    # Collect all matches (no DB-level LIMIT — we need to filter in Python)
    all_matches: list[tuple[dict, str]] = []

    with _open() as conn:
        pinned_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPINNED", "pinned", table_alias="obj")
        pp_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPASSWORDPROTECTED", "lock_pp", table_alias="obj")
        iv_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZCRYPTOINITIALIZATIONVECTOR", "lock_iv", table_alias="obj")
        mode_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZLOCKEDNOTESMODE", "lock_mode", table_alias="obj")
        live_folder_sub = _live_folder_subquery(conn)
        sql = f"""
            SELECT obj.Z_PK, obj.ZTITLE1, obj.ZFOLDER, obj.ZMODIFICATIONDATE1, data.ZDATA,
                   {pinned_col}, {pp_col}, {iv_col}, {mode_col},
                   (obj.ZSERVERSHAREDATA IS NOT NULL) AS shared
            FROM ZICCLOUDSYNCINGOBJECT obj
            LEFT JOIN ZICNOTEDATA data ON data.Z_PK = obj.ZNOTEDATA
            WHERE obj.ZTITLE1 IS NOT NULL
              AND COALESCE(obj.ZMARKEDFORDELETION, 0) = 0
              AND obj.ZFOLDER IN ({live_folder_sub})
        """
        if folder_pks:
            placeholders = ",".join("?" * len(folder_pks))
            sql += f" AND obj.ZFOLDER IN ({placeholders})"
            params.extend(folder_pks)
        elif not include_trash:
            trash = trash_folder_pks()
            if trash:
                placeholders = ",".join("?" * len(trash))
                sql += f" AND (obj.ZFOLDER IS NULL OR obj.ZFOLDER NOT IN ({placeholders}))"
                params.extend(trash)
        if modified_after_cd is not None:
            sql += " AND obj.ZMODIFICATIONDATE1 >= ?"
            params.append(modified_after_cd)
        if modified_before_cd is not None:
            sql += " AND obj.ZMODIFICATIONDATE1 <= ?"
            params.append(modified_before_cd)
        sql += " ORDER BY obj.ZMODIFICATIONDATE1 DESC"

        for pk, title, folder_pk, modified, blob, pinned, lock_pp, lock_iv, lock_mode, shared in conn.execute(sql, params):
            is_locked = bool(lock_pp or 0) or (lock_iv is not None) or (lock_mode is not None)
            title_lower = (title or "").lower()
            title_match = needle in title_lower

            body_text = ""
            body_match = False
            # Never peek inside encrypted bodies of locked notes.
            if search_body and blob and not is_locked:
                try:
                    raw = gzip.decompress(blob)
                except Exception:
                    raw = b""
                if needle_bytes in raw.lower():
                    body_text = _extract_body_text(raw)
                    body_match = True

            if not (title_match or body_match):
                continue

            all_matches.append((
                {
                    "id": short_id(pk),
                    "title": title,
                    "folder_pk": folder_pk,
                    "modified": (modified or 0) + CORE_DATA_EPOCH_OFFSET,
                    "pinned": bool(pinned or 0),
                    "locked": is_locked,
                    "shared": bool(shared or 0),
                },
                body_text or title or "",
            ))

    # Apply cursor-based pagination over the filtered result set
    fetch_limit = limit + 1
    page = all_matches[offset: offset + fetch_limit]

    has_more = len(page) > limit
    if has_more:
        page = page[:limit]

    next_cursor: str | None = encode_cursor(offset + limit) if has_more else None

    return page, has_more, next_cursor, None


def note_meta(note_pk: int) -> dict | None:
    """Fetch one note's metadata by Z_PK. Returns None if not found."""
    with _open() as conn:
        pinned_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPINNED", "pinned")
        pp_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZISPASSWORDPROTECTED", "lock_pp")
        iv_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZCRYPTOINITIALIZATIONVECTOR", "lock_iv")
        mode_col = _select_optional(conn, "ZICCLOUDSYNCINGOBJECT", "ZLOCKEDNOTESMODE", "lock_mode")
        row = conn.execute(
            f"SELECT Z_PK, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1, "
            f"{pinned_col}, {pp_col}, {iv_col}, {mode_col}, "
            f"(ZSERVERSHAREDATA IS NOT NULL) AS shared "
            "FROM ZICCLOUDSYNCINGOBJECT "
            "WHERE Z_PK = ? AND ZTITLE1 IS NOT NULL "
            "AND COALESCE(ZMARKEDFORDELETION, 0) = 0",
            (note_pk,),
        ).fetchone()
    if not row:
        return None
    pk, title, folder_pk, modified, pinned, lock_pp, lock_iv, lock_mode, shared = row
    return {
        "id": short_id(pk),
        "title": title,
        "folder_pk": folder_pk,
        "modified": (modified or 0) + CORE_DATA_EPOCH_OFFSET,
        "pinned": bool(pinned or 0),
        "locked": bool(lock_pp or 0) or (lock_iv is not None) or (lock_mode is not None),
        "shared": bool(shared or 0),
    }


def recent_notes(limit: int = 200) -> list[dict]:
    """Most-recently-modified notes, for MCP resources/list population."""
    rows, _, _, _ = list_notes(None, limit, None)
    return rows


def attachment_count(note_pk: int) -> int:
    """Total Z_ENT=5 (ICAttachment) rows for a note.

    This counts EVERYTHING Apple Notes treats as an attachment,
    including reconstructable widgets like tables. Prefer
    `destructive_attachment_count` for the update_note guard — tables
    are rebuilt from the new body, so refusing to overwrite a
    table-only note is a false positive.
    """
    with _open() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 5 AND ZNOTE = ?",
            (note_pk,),
        ).fetchone()
    return int(row[0]) if row else 0


# Bucket mapping for attachment ZTYPEUTIs. Ground-truth from a live
# NoteStore.sqlite scan: tables are 'com.apple.notes.table' (CRDT widget
# rebuilt from new body — non-destructive), everything else in Z_ENT=5
# either is destructive (image/sketch/scan/audio) or unknown (treated as
# destructive by default). Links/hashtags/mentions live inline in the
# protobuf body and never appear in Z_ENT=5.
#
# Exact UTIs we know about. Future variants:
#   - any `public.*-image` is treated as image (prefix fallback)
#   - any `public.*-audio` is treated as audio (prefix fallback)

_UTI_BUCKETS_EXACT: dict[str, str] = {
    # Images (destructive — body overwrite annihilates the file)
    "public.jpeg": "image",
    "public.png": "image",
    "public.heic": "image",
    "public.svg-image": "image",
    "com.apple.notes.gallery": "scan",  # see note below — gallery groups scanned pages
    # Sketches (destructive — PaperKit content lost on overwrite)
    "com.apple.drawing.2": "sketch",  # legacy
    "com.apple.paper": "sketch",  # iOS 17+ PaperKit
    # Scans (destructive — OCR'd PDFs lost on overwrite)
    "com.apple.notes.scan": "scan",
    # Audio (destructive — recording lost on overwrite)
    "public.audio": "audio",
    "public.mpeg-4-audio": "audio",
    "com.apple.m4a-audio": "audio",
    # Tables (NON-destructive — rebuilt from new HTML/markdown body)
    "com.apple.notes.table": "table",
}

# Buckets where overwriting the note body annihilates the content.
_DESTRUCTIVE_BUCKETS = frozenset({"image", "sketch", "scan", "audio", "file"})


def _bucket_for_uti(uti: str | None) -> str:
    """Classify a ZTYPEUTI into a destructive/reconstructable bucket."""
    if uti is None:
        return "file"  # unknown → conservative (destructive)
    if uti in _UTI_BUCKETS_EXACT:
        return _UTI_BUCKETS_EXACT[uti]
    # Prefix fallbacks for future image/audio variants.
    if uti.startswith("public.") and uti.endswith("-image"):
        return "image"
    if uti.startswith("public.") and uti.endswith("-audio"):
        return "audio"
    return "file"


def attachment_breakdown(note_pk: int) -> dict[str, dict]:
    """Per-bucket attachment detail for a note. Returns a dict with
    fixed keys (`image`, `sketch`, `scan`, `audio`, `file`, `table`)
    so callers can index without missing-key handling. Each value is
    `{count: int, destructive: bool, utis: list[str], filenames: list[str]}`.

    Sourced from `Z_ENT=5` rows; ZTYPEUTI is the discriminator.
    Links / hashtags / mentions are stored inline in the protobuf
    body and don't appear here.

    `filenames` deduplicates and drops NULLs; many attachment rows
    (especially tables and inline images) have no filename in Apple's
    schema.

    Scan-fallback: a row with non-null `ZFALLBACKPDFGENERATION` is
    bucketed as `scan` regardless of its ZTYPEUTI (some legacy iOS
    versions stored scans without a distinct UTI).
    """
    # Initialise all 6 buckets so callers can do
    # `attachments.by_type.audio.count` without missing-key handling.
    buckets: dict[str, dict] = {
        b: {
            "count": 0,
            "destructive": b in _DESTRUCTIVE_BUCKETS,
            "utis": [],
            "filenames": [],
        }
        for b in ("image", "sketch", "scan", "audio", "file", "table")
    }
    with _open() as conn:
        rows = conn.execute(
            "SELECT ZTYPEUTI, ZFILENAME, ZFALLBACKPDFGENERATION "
            "FROM ZICCLOUDSYNCINGOBJECT WHERE Z_ENT = 5 AND ZNOTE = ?",
            (note_pk,),
        ).fetchall()
    for row in rows:
        uti = row[0] if row[0] else None
        filename = row[1] if row[1] else None
        has_pdf = row[2] is not None
        if has_pdf:
            bucket = "scan"
        else:
            bucket = _bucket_for_uti(uti)
        b = buckets[bucket]
        b["count"] += 1
        if uti and uti not in b["utis"]:
            b["utis"].append(uti)
        if filename and filename not in b["filenames"]:
            b["filenames"].append(filename)
    return buckets


def destructive_attachment_count(note_pk: int) -> int:
    """Count of attachment rows that an AppleScript `set body` will
    annihilate. Excludes `table` (rebuilt from new HTML/markdown body).

    Use this for the update_note guard — `attachment_count` returns
    the total including tables, which produces false positives on
    notes that contain only a markdown table.
    """
    buckets = attachment_breakdown(note_pk)
    return sum(b["count"] for name, b in buckets.items() if b["destructive"])


def note_protobuf_blob(pk: int) -> bytes | None:
    """Return the gzipped protobuf body for a note, or None if absent.

    Tries ZICCLOUDSYNCINGOBJECT.ZMERGEABLEDATA1 first (newer schemas),
    falls back to ZICNOTEDATA.ZDATA (older schemas)."""
    with _open() as conn:
        # Modern path: ZMERGEABLEDATA1
        try:
            cur = conn.execute(
                "SELECT ZMERGEABLEDATA1 FROM ZICCLOUDSYNCINGOBJECT WHERE Z_PK = ?",
                (pk,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return bytes(row[0])
        except Exception:
            pass
        # Fallback: ZICNOTEDATA.ZDATA (joined via ZICCLOUDSYNCINGOBJECT.ZNOTEDATA)
        try:
            cur = conn.execute(
                "SELECT ZDATA FROM ZICNOTEDATA WHERE Z_PK = ("
                "  SELECT ZNOTEDATA FROM ZICCLOUDSYNCINGOBJECT WHERE Z_PK = ?"
                ")",
                (pk,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return bytes(row[0])
        except Exception:
            pass
        return None


def note_body_text(note_pk: int) -> str:
    """Decompressed plaintext for one note. Lossy — strips formatting, preserves
    text runs. Use AppleScript `body of note` for full HTML fidelity."""
    with _open() as conn:
        row = conn.execute(
            "SELECT ZDATA FROM ZICNOTEDATA WHERE Z_PK = ("
            "  SELECT ZNOTEDATA FROM ZICCLOUDSYNCINGOBJECT WHERE Z_PK = ?"
            ")",
            (note_pk,),
        ).fetchone()
    if not row or not row[0]:
        return ""
    try:
        raw = gzip.decompress(row[0])
    except Exception:
        return ""
    return _extract_body_text(raw)


# --- protobuf walker: extract printable text strings without the real schema ---

def _extract_body_text(raw: bytes) -> str:
    strings = _walk_strings(raw)
    return max(strings, key=len, default="")


def _walk_strings(buf: bytes) -> list[str]:
    out: list[str] = []
    pos = 0
    n = len(buf)
    while pos < n:
        try:
            tag, pos = _read_varint(buf, pos)
        except Exception:
            break
        wire_type = tag & 0x07
        if wire_type == 0:       # varint
            try:
                _, pos = _read_varint(buf, pos)
            except Exception:
                break
        elif wire_type == 1:     # 64-bit
            pos += 8
        elif wire_type == 5:     # 32-bit
            pos += 4
        elif wire_type == 2:     # length-delimited
            try:
                length, pos = _read_varint(buf, pos)
            except Exception:
                break
            val = buf[pos:pos + length]
            pos += length
            try:
                s = val.decode("utf-8")
                if s and sum(c.isprintable() or c in "\n\r\t" for c in s) / len(s) > 0.9:
                    out.append(s)
                else:
                    out.extend(_walk_strings(val))
            except UnicodeDecodeError:
                out.extend(_walk_strings(val))
        else:
            break
    return out


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
