"""Trash-folder exclusion in NotesSource.

Two surfaces:
  * `FakeNotesSource.iter_notes` (in-memory, used by tests + future
    non-Apple-Notes ports).
  * `AppleNotesSource._iter_from_conn` (production path; we build a
    minimal NoteStore-shaped SQLite DB in-memory to exercise the SQL).

Both must default to `include_trash=False` and both must accept the
override.
"""
from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

from apple_notes_brain.semantic.source import (
    AppleNotesSource,
    FakeNotesSource,
    NoteRecord,
)


# ---------------------------------------------------------------------------
# FakeNotesSource
# ---------------------------------------------------------------------------

def _rec(zid: str, folder: str | None = "Notes") -> NoteRecord:
    return NoteRecord(
        z_identifier=zid, z_pk=int(zid.split("-")[-1]),
        title=f"Title-{zid}", folder=folder, modified_at=1700000000,
        locked=False, pinned=False,
    )


def test_fake_source_default_excludes_recently_deleted():
    src = FakeNotesSource()
    src.add(_rec("zid-1", "Notes"), "body")
    src.add(_rec("zid-2", "Recently Deleted"), "body")
    out = list(src.iter_notes())
    assert [r.z_identifier for r in out] == ["zid-1"]


def test_fake_source_include_trash_true_yields_all():
    src = FakeNotesSource()
    src.add(_rec("zid-1", "Notes"), "body")
    src.add(_rec("zid-2", "Recently Deleted"), "body")
    out = list(src.iter_notes(include_trash=True))
    assert {r.z_identifier for r in out} == {"zid-1", "zid-2"}


def test_fake_source_yields_no_trash_with_only_trash():
    src = FakeNotesSource()
    src.add(_rec("zid-1", "Recently Deleted"), "body")
    assert list(src.iter_notes()) == []


def test_fake_source_handles_none_folder():
    """folder=None is treated as live."""
    src = FakeNotesSource()
    src.add(_rec("zid-1", None), "body")
    out = list(src.iter_notes())
    assert [r.z_identifier for r in out] == ["zid-1"]


def test_fake_source_custom_trash_names():
    src = FakeNotesSource(trash_folder_names={"Trash", "Bin"})
    src.add(_rec("zid-1", "Notes"), "")
    src.add(_rec("zid-2", "Trash"), "")
    src.add(_rec("zid-3", "Bin"), "")
    src.add(_rec("zid-4", "Recently Deleted"), "")
    out = list(src.iter_notes())
    ids = {r.z_identifier for r in out}
    # 'Recently Deleted' isn't in the custom set, so it's NOT filtered.
    assert ids == {"zid-1", "zid-4"}


def test_fake_source_empty_trash_names_disables_filter():
    src = FakeNotesSource(trash_folder_names=set())
    src.add(_rec("zid-1", "Notes"), "")
    src.add(_rec("zid-2", "Recently Deleted"), "")
    out = list(src.iter_notes())
    assert {r.z_identifier for r in out} == {"zid-1", "zid-2"}


# ---------------------------------------------------------------------------
# AppleNotesSource — drive _iter_from_conn against an in-memory SQLite that
# mimics NoteStore.sqlite's relevant columns. We don't need full fidelity,
# only the columns the iterator query reads.
# ---------------------------------------------------------------------------

def _build_fake_notestore() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            Z_ENT INTEGER,
            ZIDENTIFIER TEXT,
            ZTITLE1 TEXT,
            ZTITLE2 TEXT,
            ZFOLDER INTEGER,
            ZMODIFICATIONDATE1 REAL,
            ZISPASSWORDPROTECTED INTEGER DEFAULT 0,
            ZISPINNED INTEGER DEFAULT 0,
            ZMARKEDFORDELETION INTEGER DEFAULT 0,
            ZFOLDERTYPE INTEGER
        );
        """
    )
    # Folders (Z_ENT=15): one live, one trash.
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZTITLE2, ZFOLDERTYPE) "
        "VALUES (100, 15, 'Notes', 0)"
    )
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZTITLE2, ZFOLDERTYPE) "
        "VALUES (101, 15, 'Recently Deleted', 1)"
    )
    # Notes (Z_ENT=12).
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZIDENTIFIER, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1) "
        "VALUES (1, 12, 'zid-live', 'Live Note', 100, 700000000)"
    )
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZIDENTIFIER, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1) "
        "VALUES (2, 12, 'zid-trash', 'Trashed Note', 101, 700000001)"
    )
    return conn


def test_apple_notes_source_default_excludes_trash_folder():
    """`include_trash=False` (default) emits a NOT IN clause that drops
    notes whose ZFOLDER is a trash-folder pk."""
    conn = _build_fake_notestore()
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value={101},
    ):
        out = list(src._iter_from_conn(conn))
    ids = [r.z_identifier for r in out]
    assert "zid-live" in ids
    assert "zid-trash" not in ids


def test_apple_notes_source_include_trash_true_emits_all():
    conn = _build_fake_notestore()
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value={101},
    ):
        out = list(src._iter_from_conn(conn, include_trash=True))
    ids = {r.z_identifier for r in out}
    assert ids == {"zid-live", "zid-trash"}


def test_apple_notes_source_no_trash_pks_falls_back_to_no_filter():
    """If sqlite_reader.trash_folder_pks() returns an empty set (very
    old macOS without ZFOLDERTYPE), the SQL has no extra WHERE and
    every note is emitted."""
    conn = _build_fake_notestore()
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value=set(),
    ):
        out = list(src._iter_from_conn(conn))
    ids = {r.z_identifier for r in out}
    assert ids == {"zid-live", "zid-trash"}


def test_apple_notes_source_trash_pks_exception_fails_open():
    """If trash_folder_pks raises, the iterator still works — better
    to over-include than to crash the index pass."""
    conn = _build_fake_notestore()
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        side_effect=RuntimeError("schema mismatch"),
    ):
        out = list(src._iter_from_conn(conn))
    ids = {r.z_identifier for r in out}
    # Fail-open: both notes emitted.
    assert ids == {"zid-live", "zid-trash"}


def test_apple_notes_source_zmarkedfordeletion_still_filtered():
    """The new trash-folder filter is ADDITIVE: ZMARKEDFORDELETION=1
    rows are still excluded even when include_trash=True."""
    conn = _build_fake_notestore()
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZIDENTIFIER, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1, "
        " ZMARKEDFORDELETION) "
        "VALUES (3, 12, 'zid-marked', 'Marked', 100, 700000002, 1)"
    )
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value={101},
    ):
        out = list(src._iter_from_conn(conn, include_trash=True))
    ids = {r.z_identifier for r in out}
    assert "zid-marked" not in ids


def test_apple_notes_source_multiple_trash_pks():
    """Multiple distinct trash folder PKs all filtered."""
    conn = _build_fake_notestore()
    # Add a second trash folder + a note inside it.
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZTITLE2, ZFOLDERTYPE) "
        "VALUES (102, 15, 'Trash-Acct2', 1)"
    )
    conn.execute(
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZIDENTIFIER, ZTITLE1, ZFOLDER, ZMODIFICATIONDATE1) "
        "VALUES (4, 12, 'zid-trash-2', 'Other trash', 102, 700000003)"
    )
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value={101, 102},
    ):
        out = list(src._iter_from_conn(conn))
    ids = {r.z_identifier for r in out}
    assert ids == {"zid-live"}


def test_apple_notes_source_records_folder_name_for_live_notes():
    """The emitted NoteRecord carries the folder NAME, not the PK —
    we rely on this for the query-time defence in Search."""
    conn = _build_fake_notestore()
    src = AppleNotesSource()
    with mock.patch(
        "apple_notes_brain.sqlite_reader.trash_folder_pks",
        return_value={101},
    ):
        out = list(src._iter_from_conn(conn))
    [rec] = out
    assert rec.folder == "Notes"
