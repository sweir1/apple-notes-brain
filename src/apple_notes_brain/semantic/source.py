"""NotesSource abstraction — decouples the indexer from Apple Notes.

The IndexPipeline doesn't care WHERE notes come from. It cares about
an iterable of `NoteRecord`s plus a way to fetch `body_text()` and
`get_record(zid)`. Apple Notes implementation reads NoteStore.sqlite;
unit tests use FakeNotesSource (in-memory dict).

This is the seam future ports cross. If someone wants to point the
semantic stack at a markdown vault, they implement NotesSource over
the filesystem and feed it to IndexPipeline.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Core Data uses 2001-01-01 as the epoch; unix time uses 1970-01-01.
# Offset = (31 years × 365.25 days) — Apple uses an exact constant.
_CORE_DATA_EPOCH_OFFSET = 978307200


@dataclass(frozen=True)
class NoteRecord:
    """One indexable note. `body_text` lives behind `source.body_text()`
    so the iterator can stream millions of records without materialising
    every body in memory."""
    z_identifier: str
    z_pk: int
    title: str
    folder: str | None
    modified_at: int     # unix epoch seconds
    locked: bool
    pinned: bool


@runtime_checkable
class NotesSource(Protocol):
    """The contract the indexer reads against."""

    def iter_notes(self) -> Iterator[NoteRecord]:
        """Yield every indexable note exactly once. Order doesn't matter
        for correctness; sorting by modified_at descending is nice for
        responsive incremental indexing UIs."""

    def body_text(self, record: NoteRecord) -> str:
        """Return the plaintext body for a record. Empty string for
        locked notes (the indexer treats those specially)."""

    def get_record(self, z_identifier: str) -> NoteRecord | None:
        """Resolve a single record by stable UUID. Used by the watcher
        for per-note updates."""


# ---------------------------------------------------------------------------
# AppleNotesSource — production implementation, talks to NoteStore.sqlite
# ---------------------------------------------------------------------------

class AppleNotesSource:
    """Backed by Apple's `NoteStore.sqlite`. Read-only.

    Body text comes from `sqlite_reader.note_body_text(z_pk)` — that
    function handles the protobuf decompression and walks the wire
    format. Listing uses our own SQL because we need ZIDENTIFIER which
    `sqlite_reader.list_notes` doesn't expose.
    """

    def __init__(self):
        # Imports deferred to function bodies so a unit test that injects
        # a FakeNotesSource never has to pay sqlite_reader's import cost.
        pass

    def iter_notes(self) -> Iterator[NoteRecord]:
        from apple_notes_brain import sqlite_reader as sr

        with sr._open() as conn:  # type: ignore[attr-defined]
            yield from self._iter_from_conn(conn)

    def _iter_from_conn(self, conn: sqlite3.Connection) -> Iterator[NoteRecord]:
        """Run the listing query against an already-open connection."""
        # Note rows are Z_ENT=12 with a title; the column names we need.
        # ZACCOUNT8 isn't on every macOS — but for listing it doesn't matter,
        # we just don't surface the account here.
        rows = conn.execute(
            """
            SELECT
                ZICCLOUDSYNCINGOBJECT.ZIDENTIFIER,
                ZICCLOUDSYNCINGOBJECT.Z_PK,
                ZICCLOUDSYNCINGOBJECT.ZTITLE1,
                folder.ZTITLE2,
                ZICCLOUDSYNCINGOBJECT.ZMODIFICATIONDATE1,
                COALESCE(ZICCLOUDSYNCINGOBJECT.ZISPASSWORDPROTECTED, 0) AS locked,
                COALESCE(ZICCLOUDSYNCINGOBJECT.ZISPINNED, 0) AS pinned
            FROM ZICCLOUDSYNCINGOBJECT
            LEFT JOIN ZICCLOUDSYNCINGOBJECT AS folder
                ON folder.Z_PK = ZICCLOUDSYNCINGOBJECT.ZFOLDER
                AND folder.Z_ENT = 15
            WHERE ZICCLOUDSYNCINGOBJECT.ZTITLE1 IS NOT NULL
              AND COALESCE(ZICCLOUDSYNCINGOBJECT.ZMARKEDFORDELETION, 0) = 0
              AND ZICCLOUDSYNCINGOBJECT.ZIDENTIFIER IS NOT NULL
            ORDER BY ZICCLOUDSYNCINGOBJECT.ZMODIFICATIONDATE1 DESC
            """
        )
        for r in rows:
            yield NoteRecord(
                z_identifier=str(r[0]),
                z_pk=int(r[1]),
                title=str(r[2] or "(untitled)"),
                folder=str(r[3]) if r[3] is not None else None,
                modified_at=_coredata_to_unix(r[4]),
                locked=bool(r[5]),
                pinned=bool(r[6]),
            )

    def body_text(self, record: NoteRecord) -> str:
        if record.locked:
            return ""
        from apple_notes_brain import sqlite_reader as sr

        try:
            return sr.note_body_text(record.z_pk) or ""
        except Exception:
            # Mirrors sqlite_reader's own defensive degradation — body
            # extraction is best-effort. Indexer's empty-note fallback
            # handles the empty case downstream.
            return ""

    def get_record(self, z_identifier: str) -> NoteRecord | None:
        from apple_notes_brain import sqlite_reader as sr

        with sr._open() as conn:  # type: ignore[attr-defined]
            row = conn.execute(
                """
                SELECT
                    ZICCLOUDSYNCINGOBJECT.ZIDENTIFIER,
                    ZICCLOUDSYNCINGOBJECT.Z_PK,
                    ZICCLOUDSYNCINGOBJECT.ZTITLE1,
                    folder.ZTITLE2,
                    ZICCLOUDSYNCINGOBJECT.ZMODIFICATIONDATE1,
                    COALESCE(ZICCLOUDSYNCINGOBJECT.ZISPASSWORDPROTECTED, 0),
                    COALESCE(ZICCLOUDSYNCINGOBJECT.ZISPINNED, 0)
                FROM ZICCLOUDSYNCINGOBJECT
                LEFT JOIN ZICCLOUDSYNCINGOBJECT AS folder
                    ON folder.Z_PK = ZICCLOUDSYNCINGOBJECT.ZFOLDER
                    AND folder.Z_ENT = 15
                WHERE ZICCLOUDSYNCINGOBJECT.ZIDENTIFIER = ?
                  AND ZICCLOUDSYNCINGOBJECT.ZTITLE1 IS NOT NULL
                  AND COALESCE(ZICCLOUDSYNCINGOBJECT.ZMARKEDFORDELETION, 0) = 0
                """,
                (z_identifier,),
            ).fetchone()
            if row is None:
                return None
            return NoteRecord(
                z_identifier=str(row[0]),
                z_pk=int(row[1]),
                title=str(row[2] or "(untitled)"),
                folder=str(row[3]) if row[3] is not None else None,
                modified_at=_coredata_to_unix(row[4]),
                locked=bool(row[5]),
                pinned=bool(row[6]),
            )


def _coredata_to_unix(value) -> int:
    """Core Data epoch (2001-01-01 UTC) → unix epoch (1970-01-01 UTC)."""
    if value is None:
        return 0
    return int(float(value) + _CORE_DATA_EPOCH_OFFSET)


# ---------------------------------------------------------------------------
# FakeNotesSource — for unit tests
# ---------------------------------------------------------------------------

class FakeNotesSource:
    """Pure-memory source for unit tests. Construct with a dict of
    {z_identifier: (record, body_text)}."""

    def __init__(
        self, notes: dict[str, tuple[NoteRecord, str]] | None = None
    ) -> None:
        self._notes: dict[str, tuple[NoteRecord, str]] = dict(notes or {})

    def add(self, record: NoteRecord, body: str) -> None:
        self._notes[record.z_identifier] = (record, body)

    def remove(self, z_identifier: str) -> None:
        self._notes.pop(z_identifier, None)

    def iter_notes(self) -> Iterator[NoteRecord]:
        # Sort by modified_at desc to match real-world ordering, gives
        # deterministic test output.
        records = sorted(
            (rec for rec, _ in self._notes.values()),
            key=lambda r: r.modified_at,
            reverse=True,
        )
        for r in records:
            yield r

    def body_text(self, record: NoteRecord) -> str:
        if record.locked:
            return ""
        entry = self._notes.get(record.z_identifier)
        return entry[1] if entry else ""

    def get_record(self, z_identifier: str) -> NoteRecord | None:
        entry = self._notes.get(z_identifier)
        return entry[0] if entry else None
