"""Integration tests for sqlite_reader against a synthetic in-memory NoteStore.

The fixture DB is loaded from tests/fixtures/sqlite/notestore_minimal.sql
and patched in via NOTE_STORE_PATH. These tests exercise the actual SQL
queries against a controlled schema rather than mocking the connection.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from apple_notes_brain import sqlite_reader as db

pytestmark = pytest.mark.integration

FIXTURE_SQL = Path(__file__).parent.parent / "fixtures" / "sqlite" / "notestore_minimal.sql"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_db(db_path: Path, extra_sql: str = "") -> None:
    """Build a fresh NoteStore.sqlite at db_path from the fixture SQL."""
    sql = FIXTURE_SQL.read_text()
    if extra_sql:
        sql += "\n" + extra_sql
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(sql)
    conn.commit()
    conn.close()


def _patch_notestore(monkeypatch, db_path: Path) -> None:
    """Patch module globals to point at the fixture DB and reset caches.

    NOTE: ``_open(path=NOTE_STORE_PATH)`` binds NOTE_STORE_PATH as a default
    argument at function-definition time, so monkeypatching the module
    constant alone is NOT sufficient — every public function in the reader
    calls ``_open()`` with no args. We therefore replace ``_open`` itself
    with a closure that opens the fixture DB read-only.
    """
    monkeypatch.setattr("apple_notes_brain.sqlite_reader.NOTE_STORE_PATH", db_path)
    monkeypatch.setattr("apple_notes_brain.sqlite_reader._uuid_cache", None)
    monkeypatch.setattr("apple_notes_brain.sqlite_reader._COLS_CACHE", {})

    def _open_fixture(path: Path = db_path) -> sqlite3.Connection:
        if not path.exists():
            from apple_notes_brain.sqlite_reader import NoteStoreError
            raise NoteStoreError(f"NoteStore not found at {path}")
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)

    monkeypatch.setattr("apple_notes_brain.sqlite_reader._open", _open_fixture)


@pytest.fixture
def notestore_path(tmp_path, monkeypatch):
    """Build a NoteStore.sqlite from the fixture SQL and patch NOTE_STORE_PATH.

    Yields the absolute path to the fixture DB. Module-level UUID + column
    caches are cleared so probes re-run against this DB.
    """
    db_path = tmp_path / "NoteStore.sqlite"
    _build_db(db_path)
    _patch_notestore(monkeypatch, db_path)
    return db_path


@pytest.fixture
def empty_notestore_path(tmp_path, monkeypatch):
    """A NoteStore with the schema but zero data rows (other than Z_METADATA)."""
    db_path = tmp_path / "NoteStore.sqlite"
    sql = FIXTURE_SQL.read_text()
    # Strip every INSERT statement except Z_METADATA so the schema exists but no
    # account/folder/note rows are present.
    lines = sql.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith("INSERT INTO") and not stripped.startswith("INSERT INTO Z_METADATA"):
            continue
        if stripped.startswith("UPDATE ZICCLOUDSYNCINGOBJECT"):
            continue
        if stripped.startswith("(") and "VALUES" not in stripped:
            # continuation line of a multi-row INSERT we just skipped
            continue
        kept.append(line)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript("\n".join(kept))
    conn.commit()
    conn.close()
    _patch_notestore(monkeypatch, db_path)
    return db_path


# ---------------------------------------------------------------------------
# store_uuid + data_version
# ---------------------------------------------------------------------------

def test_store_uuid_returns_metadata_value(notestore_path):
    assert db.store_uuid() == "TEST-UUID-12345-67890"


def test_store_uuid_is_cached(notestore_path, monkeypatch):
    # First call populates cache.
    first = db.store_uuid()
    # Move the DB out from under us — second call should still return cached.
    monkeypatch.setattr(
        "apple_notes_brain.sqlite_reader.NOTE_STORE_PATH",
        notestore_path.parent / "does-not-exist.sqlite",
    )
    assert db.store_uuid() == first


def test_data_version_is_positive_int(notestore_path):
    v = db.data_version()
    assert isinstance(v, int)
    assert v >= 1


# ---------------------------------------------------------------------------
# list_folders
# ---------------------------------------------------------------------------

def test_list_folders_excludes_trash_by_default_count(notestore_path):
    folders = db.list_folders()
    # Notes, Work, Personal, Subfolder, Recently Deleted — all returned, but
    # Recently Deleted is flagged is_trash=True. list_folders does NOT exclude
    # the trash folder itself; only trash_folder_pks marks it.
    names = sorted(f["name"] for f in folders)
    assert names == ["Notes", "Personal", "Recently Deleted", "Subfolder", "Work"]


def test_list_folders_marks_trash(notestore_path):
    folders = db.list_folders()
    by_name = {f["name"]: f for f in folders}
    assert by_name["Recently Deleted"]["is_trash"] is True
    assert by_name["Notes"]["is_trash"] is False
    assert by_name["Work"]["is_trash"] is False


def test_list_folders_with_counts(notestore_path):
    folders = db.list_folders(include_counts=True)
    by_name = {f["name"]: f for f in folders}
    # Notes folder has PK=2 with notes 10 + 14 (locked is still counted; trashed
    # PK=13 is in folder 3, not 2).
    assert by_name["Notes"]["note_count"] == 2
    assert by_name["Work"]["note_count"] == 1
    assert by_name["Subfolder"]["note_count"] == 1
    assert by_name["Personal"]["note_count"] == 0


def test_list_folders_nested_path(notestore_path):
    folders = db.list_folders()
    by_name = {f["name"]: f for f in folders}
    assert by_name["Subfolder"]["path"] == "Work/Subfolder"
    assert by_name["Work"]["path"] == "Work"


def test_list_folders_account_resolved(notestore_path):
    folders = db.list_folders()
    for f in folders:
        assert f["account"] == "iCloud", f"folder {f['name']!r} missing account"


def test_list_folders_short_id_format(notestore_path):
    folders = db.list_folders()
    for f in folders:
        assert f["id"].startswith("f")
        assert int(f["id"][1:]) > 0


def test_list_folders_shared_flag_default_false(notestore_path):
    folders = db.list_folders()
    for f in folders:
        assert f["shared"] is False


# ---------------------------------------------------------------------------
# trash_folder_pks / default_folder_pk / is_default_folder
# ---------------------------------------------------------------------------

def test_trash_folder_pks(notestore_path):
    assert db.trash_folder_pks() == {3}


def test_default_folder_pk(notestore_path):
    assert db.default_folder_pk() == 2


def test_is_default_folder_true(notestore_path):
    assert db.is_default_folder(2) is True


def test_is_default_folder_false(notestore_path):
    assert db.is_default_folder(4) is False
    assert db.is_default_folder(99999) is False


# ---------------------------------------------------------------------------
# list_notes
# ---------------------------------------------------------------------------

def test_list_notes_no_filter(notestore_path):
    rows, has_more, cursor, total = db.list_notes(None, limit=100)
    # PK 10 (Notes), 11 (Work), 12 (Subfolder), 14 (Notes, locked).
    # PK 13 trashed → excluded by include_trash=False default.
    pks = sorted(int(r["id"][1:]) for r in rows)
    assert pks == [10, 11, 12, 14]
    assert total == 4
    assert has_more is False
    assert cursor is None


def test_list_notes_scoped_to_one_folder(notestore_path):
    rows, _, _, total = db.list_notes({4}, limit=100)
    assert total == 1
    assert rows[0]["id"] == "p11"
    assert rows[0]["folder_pk"] == 4


def test_list_notes_scoped_to_multiple_folders(notestore_path):
    rows, _, _, total = db.list_notes({4, 6}, limit=100)
    assert total == 2
    pks = sorted(int(r["id"][1:]) for r in rows)
    assert pks == [11, 12]


def test_list_notes_include_trash(notestore_path):
    rows, _, _, total = db.list_notes(None, limit=100, include_trash=True)
    pks = sorted(int(r["id"][1:]) for r in rows)
    # ZMARKEDFORDELETION=1 still excludes PK 13. include_trash only relaxes
    # the trash-folder filter — but since the trashed note is also marked,
    # it stays excluded.
    assert 13 not in pks
    assert pks == [10, 11, 12, 14]
    assert total == 4


def test_list_notes_modified_after_filter(notestore_path):
    # ZMODIFICATIONDATE1 values: 7700000, 7710000, 7720000, 7740000.
    rows, _, _, total = db.list_notes(None, limit=100, modified_after_cd=7715000)
    pks = sorted(int(r["id"][1:]) for r in rows)
    # Strictly inclusive lower bound: keep ≥ 7715000 → PKs 12, 14.
    assert pks == [12, 14]
    assert total == 2


def test_list_notes_modified_before_filter(notestore_path):
    rows, _, _, total = db.list_notes(None, limit=100, modified_before_cd=7715000)
    pks = sorted(int(r["id"][1:]) for r in rows)
    # ≤ 7715000 → 7700000, 7710000 → PKs 10, 11.
    assert pks == [10, 11]
    assert total == 2


def test_list_notes_orders_modified_desc(notestore_path):
    rows, _, _, _ = db.list_notes(None, limit=100)
    mod_values = [r["modified"] for r in rows]
    assert mod_values == sorted(mod_values, reverse=True)


def test_list_notes_pagination_first_page(notestore_path):
    rows, has_more, cursor, total = db.list_notes(None, limit=2)
    assert len(rows) == 2
    assert has_more is True
    assert cursor is not None
    assert total == 4


def test_list_notes_cursor_round_trip(notestore_path):
    page1, has_more, cursor, _ = db.list_notes(None, limit=2)
    assert has_more is True
    page2, more2, cur2, _ = db.list_notes(None, limit=2, cursor=cursor)
    assert len(page2) == 2
    # No overlap between pages.
    page1_ids = {r["id"] for r in page1}
    page2_ids = {r["id"] for r in page2}
    assert page1_ids.isdisjoint(page2_ids)
    assert more2 is False
    assert cur2 is None


def test_list_notes_limit_one(notestore_path):
    rows, has_more, cursor, _ = db.list_notes(None, limit=1)
    assert len(rows) == 1
    assert has_more is True
    assert cursor is not None


def test_list_notes_limit_equals_total(notestore_path):
    rows, has_more, cursor, total = db.list_notes(None, limit=4)
    assert len(rows) == 4
    assert has_more is False
    assert cursor is None
    assert total == 4


def test_list_notes_locked_flag(notestore_path):
    rows, _, _, _ = db.list_notes(None, limit=100)
    by_id = {r["id"]: r for r in rows}
    assert by_id["p14"]["locked"] is True
    assert by_id["p10"]["locked"] is False


# ---------------------------------------------------------------------------
# search_notes
# ---------------------------------------------------------------------------

def test_search_notes_title_match(notestore_path):
    matches, has_more, cursor, total = db.search_notes(
        "Note in", folder_pks=None, search_body=False, limit=100
    )
    titles = sorted(m[0]["title"] for m in matches)
    assert titles == ["Note in Notes", "Note in Subfolder", "Note in Work"]
    assert total is None
    assert has_more is False


def test_search_notes_no_match(notestore_path):
    matches, _, _, _ = db.search_notes(
        "doesnotexist", folder_pks=None, search_body=False, limit=100
    )
    assert matches == []


def test_search_notes_folder_scope(notestore_path):
    matches, _, _, _ = db.search_notes(
        "Note", folder_pks={4}, search_body=False, limit=100
    )
    assert len(matches) == 1
    assert matches[0][0]["title"] == "Note in Work"


def test_search_notes_excludes_trashed_by_default(notestore_path):
    matches, _, _, _ = db.search_notes(
        "Trashed", folder_pks=None, search_body=False, limit=100
    )
    # PK 13 is in trash AND marked-for-deletion, so excluded.
    assert matches == []


def test_search_notes_pagination(notestore_path):
    page1, has_more, cursor, _ = db.search_notes(
        "Note", folder_pks=None, search_body=False, limit=1
    )
    assert len(page1) == 1
    assert has_more is True
    assert cursor is not None


# ---------------------------------------------------------------------------
# note_meta
# ---------------------------------------------------------------------------

def test_note_meta_existing(notestore_path):
    meta = db.note_meta(10)
    assert meta is not None
    assert meta["id"] == "p10"
    assert meta["title"] == "Note in Notes"
    assert meta["folder_pk"] == 2
    assert meta["locked"] is False
    assert meta["pinned"] is False


def test_note_meta_locked(notestore_path):
    meta = db.note_meta(14)
    assert meta is not None
    assert meta["locked"] is True


def test_note_meta_missing_pk(notestore_path):
    assert db.note_meta(99999) is None


def test_note_meta_excludes_trashed(notestore_path):
    # PK 13 is marked-for-deletion → note_meta refuses to surface it.
    assert db.note_meta(13) is None


# ---------------------------------------------------------------------------
# zid lookups
# ---------------------------------------------------------------------------

def test_note_state_by_zid(notestore_path):
    state = db.note_state_by_zid("note-1")
    assert state == {"pk": 10, "folder_pk": 2, "marked": 0}


def test_note_state_by_zid_missing(notestore_path):
    assert db.note_state_by_zid("nonexistent-zid") is None


def test_folder_zid_by_pk(notestore_path):
    assert db.folder_zid_by_pk(4) == "work-folder"
    assert db.folder_zid_by_pk(2) == "DefaultFolder-CloudKit"


def test_folder_zid_by_pk_missing(notestore_path):
    assert db.folder_zid_by_pk(99999) is None


def test_folder_state_by_zid(notestore_path):
    state = db.folder_state_by_zid("work-folder")
    assert state == {"pk": 4, "marked": 0}


def test_folder_state_by_zid_missing(notestore_path):
    assert db.folder_state_by_zid("no-such-folder") is None


# ---------------------------------------------------------------------------
# child_folder_pks
# ---------------------------------------------------------------------------

def test_child_folder_pks_with_children(notestore_path):
    children = db.child_folder_pks(4)
    assert children == [6]


def test_child_folder_pks_no_children(notestore_path):
    assert db.child_folder_pks(2) == []
    assert db.child_folder_pks(6) == []


# ---------------------------------------------------------------------------
# recent_notes
# ---------------------------------------------------------------------------

def test_recent_notes_returns_descending(notestore_path):
    rows = db.recent_notes(5)
    pks = [int(r["id"][1:]) for r in rows]
    # 7740000 (PK14) > 7720000 (PK12) > 7710000 (PK11) > 7700000 (PK10).
    assert pks == [14, 12, 11, 10]


def test_recent_notes_respects_limit(notestore_path):
    rows = db.recent_notes(2)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# attachment_count
# ---------------------------------------------------------------------------

def test_attachment_count_zero(notestore_path):
    assert db.attachment_count(10) == 0


def test_attachment_count_with_attachments(notestore_path, tmp_path, monkeypatch):
    # Insert two attachment rows (Z_ENT=5) referencing note PK=10.
    extra = """
    INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNOTE) VALUES (100, 5, 10);
    INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNOTE) VALUES (101, 5, 10);
    """
    db_path = tmp_path / "NoteStore_attach.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)
    assert db.attachment_count(10) == 2
    assert db.attachment_count(11) == 0


# ---------------------------------------------------------------------------
# Schema-drift fallback for _folder_account_column
# ---------------------------------------------------------------------------

def test_folder_account_column_falls_back_to_zaccount1(tmp_path, monkeypatch):
    """If the schema only has ZACCOUNT1 (older macOS), the probe should pick it up."""
    db_path = tmp_path / "NoteStore_old.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE Z_METADATA (Z_VERSION INTEGER PRIMARY KEY, Z_UUID VARCHAR, Z_PLIST BLOB);
        INSERT INTO Z_METADATA (Z_VERSION, Z_UUID) VALUES (1, 'OLD-UUID');
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            Z_ENT INTEGER,
            ZTITLE2 TEXT,
            ZNAME TEXT,
            ZPARENT INTEGER,
            ZIDENTIFIER TEXT,
            ZMARKEDFORDELETION INTEGER DEFAULT 0,
            ZFOLDERTYPE INTEGER,
            ZACCOUNT1 INTEGER,
            ZSERVERSHAREDATA BLOB,
            ZSERVERRECORDDATA BLOB,
            ZNEEDSINITIALFETCHFROMCLOUD INTEGER DEFAULT 0
        );
        INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNAME, ZIDENTIFIER, ZSERVERRECORDDATA)
        VALUES (1, 14, 'iCloud', 'acct', X'01');
        INSERT INTO ZICCLOUDSYNCINGOBJECT
            (Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT1, ZSERVERRECORDDATA)
        VALUES (2, 15, 'OldFolder', 'old-folder', 0, 1, X'01');
    """)
    conn.commit()
    conn.close()
    _patch_notestore(monkeypatch, db_path)

    folders = db.list_folders()
    assert len(folders) == 1
    assert folders[0]["name"] == "OldFolder"
    assert folders[0]["account"] == "iCloud"


# ---------------------------------------------------------------------------
# Empty database — graceful degradation
# ---------------------------------------------------------------------------

def test_empty_db_list_folders(empty_notestore_path):
    assert db.list_folders() == []


def test_empty_db_list_notes(empty_notestore_path):
    rows, has_more, cursor, total = db.list_notes(None, limit=10)
    assert rows == []
    assert has_more is False
    assert cursor is None
    assert total == 0


def test_empty_db_search_notes(empty_notestore_path):
    matches, has_more, cursor, total = db.search_notes(
        "anything", folder_pks=None, search_body=False, limit=10
    )
    assert matches == []
    assert has_more is False


def test_empty_db_default_folder_pk_none(empty_notestore_path):
    assert db.default_folder_pk() is None


def test_empty_db_trash_folder_pks_empty(empty_notestore_path):
    assert db.trash_folder_pks() == set()


def test_empty_db_recent_notes_empty(empty_notestore_path):
    assert db.recent_notes(10) == []
