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
    # attachment_count is the RAW Z_ENT=5 total (including tables) —
    # back-compat hatch for callers that want everything.
    assert db.attachment_count(10) == 2
    assert db.attachment_count(11) == 0


# ---------------------------------------------------------------------------
# v1.1 Part 4 Phase 3 — nested attachment breakdown
# ---------------------------------------------------------------------------

def _attachment_breakdown_fixture(tmp_path, monkeypatch, note_pk: int, rows: list[tuple]):
    """Build a NoteStore with arbitrary (uti, filename, has_pdf) attachment rows."""
    inserts = []
    for i, (uti, filename, has_pdf) in enumerate(rows, start=200):
        uti_val = f"'{uti}'" if uti else "NULL"
        fn_val = f"'{filename}'" if filename else "NULL"
        pdf_val = "X'01'" if has_pdf else "NULL"
        inserts.append(
            f"INSERT INTO ZICCLOUDSYNCINGOBJECT "
            f"(Z_PK, Z_ENT, ZNOTE, ZTYPEUTI, ZFILENAME, ZFALLBACKPDFGENERATION) "
            f"VALUES ({i}, 5, {note_pk}, {uti_val}, {fn_val}, {pdf_val});"
        )
    extra = "\n".join(inserts)
    db_path = tmp_path / f"NoteStore_breakdown_{note_pk}.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)


def test_attachment_breakdown_empty(notestore_path):
    out = db.attachment_breakdown(10)
    # All 6 buckets initialised with count=0 even when there are no attachments.
    assert set(out.keys()) == {"image", "sketch", "scan", "audio", "file", "table"}
    assert all(b["count"] == 0 for b in out.values())
    assert out["table"]["destructive"] is False
    assert out["image"]["destructive"] is True


def test_attachment_breakdown_image_jpegs(tmp_path, monkeypatch):
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("public.jpeg", "photo.jpg", False),
        ("public.jpeg", "photo2.jpg", False),
    ])
    out = db.attachment_breakdown(10)
    assert out["image"]["count"] == 2
    assert out["image"]["utis"] == ["public.jpeg"]
    assert out["image"]["filenames"] == ["photo.jpg", "photo2.jpg"]


def test_attachment_breakdown_all_image_variants(tmp_path, monkeypatch):
    """jpeg/png/heic/svg-image all bucket as 'image'."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("public.jpeg", None, False),
        ("public.png", None, False),
        ("public.heic", None, False),
        ("public.svg-image", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["image"]["count"] == 4
    assert set(out["image"]["utis"]) == {
        "public.jpeg", "public.png", "public.heic", "public.svg-image",
    }


def test_attachment_breakdown_sketch_uses_apple_paper(tmp_path, monkeypatch):
    """iOS 17+ PaperKit sketches show up as com.apple.paper, mapping to 'sketch'."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.paper", None, False),
        ("com.apple.drawing.2", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["sketch"]["count"] == 2
    assert out["sketch"]["destructive"] is True


def test_attachment_breakdown_audio_variants(tmp_path, monkeypatch):
    """m4a-audio, mpeg-4-audio, generic public.audio all bucket as 'audio'."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.m4a-audio", "rec.m4a", False),
        ("public.mpeg-4-audio", "song.mp4", False),
        ("public.audio", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["audio"]["count"] == 3


def test_attachment_breakdown_scan_via_gallery_and_scan_uti(tmp_path, monkeypatch):
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.notes.gallery", None, False),
        ("com.apple.notes.scan", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["scan"]["count"] == 2


def test_attachment_breakdown_scan_via_pdf_fallback(tmp_path, monkeypatch):
    """A row with non-null ZFALLBACKPDFGENERATION buckets as scan even
    when its ZTYPEUTI doesn't match the scan UTIs (legacy schema)."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("some-legacy-uti", None, True),  # has_pdf=True
    ])
    out = db.attachment_breakdown(10)
    assert out["scan"]["count"] == 1


def test_attachment_breakdown_table_is_non_destructive(tmp_path, monkeypatch):
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.notes.table", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["table"]["count"] == 1
    assert out["table"]["destructive"] is False


def test_attachment_breakdown_unknown_uti_buckets_as_file(tmp_path, monkeypatch):
    """User-attached PDFs / ZIPs / etc. with unknown UTIs default to
    'file' bucket (destructive — conservative)."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.adobe.pdf", "doc.pdf", False),
        ("public.zip-archive", "archive.zip", False),
        (None, "no_uti.bin", False),
    ])
    out = db.attachment_breakdown(10)
    assert out["file"]["count"] == 3
    assert out["file"]["destructive"] is True


def test_destructive_attachment_count_excludes_table(tmp_path, monkeypatch):
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("public.jpeg", None, False),
        ("public.jpeg", None, False),
        ("com.apple.notes.table", None, False),
    ])
    assert db.destructive_attachment_count(10) == 2
    # And the raw count still sees all three rows (back-compat).
    assert db.attachment_count(10) == 3


def test_destructive_attachment_count_zero_when_table_only(tmp_path, monkeypatch):
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.notes.table", None, False),
    ])
    assert db.destructive_attachment_count(10) == 0
    assert db.attachment_count(10) == 1  # raw count still 1


def test_attachment_breakdown_mixed_full_corpus(tmp_path, monkeypatch):
    """Replicates the user's live 'Claude attachment test note': 9
    attachments across 4 destructive buckets + 1 table."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("com.apple.notes.table", None, False),
        ("com.apple.m4a-audio", None, False),
        ("public.mpeg-4-audio", None, False),
        ("public.png", None, False),
        ("public.heic", None, False),
        ("public.svg-image", None, False),
        ("com.apple.paper", None, False),
        ("public.jpeg", None, False),
        ("com.apple.notes.gallery", None, False),
    ])
    out = db.attachment_breakdown(10)
    assert out["image"]["count"] == 4  # jpeg/png/heic/svg
    assert out["sketch"]["count"] == 1  # paper
    assert out["audio"]["count"] == 2  # m4a + mpeg-4
    assert out["scan"]["count"] == 1  # gallery
    assert out["table"]["count"] == 1
    assert out["file"]["count"] == 0  # no unknown UTIs in this set
    assert db.destructive_attachment_count(10) == 8
    assert db.attachment_count(10) == 9


def test_attachment_breakdown_filenames_deduped(tmp_path, monkeypatch):
    """Duplicate filenames across rows collapse — list_failed_chunk_ids
    style dedup. NULL filenames drop out."""
    _attachment_breakdown_fixture(tmp_path, monkeypatch, 10, [
        ("public.jpeg", "photo.jpg", False),
        ("public.jpeg", "photo.jpg", False),
        ("public.png", None, False),  # NULL filename — must not crash, must not appear
    ])
    out = db.attachment_breakdown(10)
    assert out["image"]["count"] == 3
    assert out["image"]["filenames"] == ["photo.jpg"]
    # png with NULL filename doesn't add an empty string.
    assert "" not in out["image"]["filenames"]


def test_attachment_breakdown_z_ent_9_inline_excluded(tmp_path, monkeypatch):
    """Z_ENT=9 (inline hashtag/mention) rows must not appear in the count."""
    extra = """
    INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNOTE, ZTYPEUTI)
    VALUES (200, 9, 10, 'com.apple.notes.inlinetextattachment.hashtag');
    INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNOTE, ZTYPEUTI)
    VALUES (201, 5, 10, 'public.jpeg');
    """
    db_path = tmp_path / "NoteStore_z9.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)
    out = db.attachment_breakdown(10)
    assert out["image"]["count"] == 1  # the jpeg
    assert sum(b["count"] for b in out.values()) == 1  # only one row total
    assert db.attachment_count(10) == 1  # raw count agrees


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


# ---------------------------------------------------------------------------
# ZSERVERRECORDDATA guard on the ACHANGE-delete filter (issue #1, PR #2).
#
# Apple writes stale ZCHANGETYPE=2 rows to ACHANGE during account migrations
# and CloudKit re-syncs WITHOUT actually deleting the folder. The pre-fix
# filter then incorrectly hid those folders (reporter observed 956 notes in
# the default Notes folder become invisible). The fix: skip the ACHANGE
# filter when ZSERVERRECORDDATA IS NOT NULL, because CloudKit has confirmed
# the folder is live on the server. True lag-window ghosts (no server
# record yet, awaiting CloudKit ack) keep ZSERVERRECORDDATA NULL and remain
# correctly filtered.
#
# These tests cover all three call sites that apply the guard:
#   - list_folders()
#   - child_folder_pks()
#   - _live_folder_subquery() (via list_notes / search_notes)
# ---------------------------------------------------------------------------

# A row that simulates a stale Apple-Core-Data delete-commit for folder Z_PK = ?
# without the row actually being marked-for-deletion on the syncing object.
def _achange_delete_for(folder_pk: int, achange_pk: int) -> str:
    return (
        f"INSERT INTO ACHANGE (Z_PK, ZENTITY, ZCHANGETYPE, ZENTITYPK) "
        f"VALUES ({achange_pk}, 15, 2, {folder_pk});"
    )


def test_list_folders_keeps_default_when_stale_achange_and_server_record(tmp_path, monkeypatch):
    """Issue #1 regression: default 'Notes' folder must remain visible despite a
    stale ACHANGE delete-commit, because ZSERVERRECORDDATA is set on it."""
    db_path = tmp_path / "NoteStore_default_with_stale.sqlite"
    _build_db(db_path, extra_sql=_achange_delete_for(folder_pk=2, achange_pk=900))
    _patch_notestore(monkeypatch, db_path)

    folders = db.list_folders()
    names = sorted(f["name"] for f in folders)
    # 'Notes' (Z_PK=2) has ZSERVERRECORDDATA NOT NULL → guard fires → visible.
    assert "Notes" in names, f"Default folder hidden by stale ACHANGE: {names!r}"


def test_list_folders_keeps_system_folders_when_both_have_stale_achange(tmp_path, monkeypatch):
    """Real-world scenario from the PR: BOTH DefaultFolder-CloudKit and
    TrashFolder-CloudKit have stale ACHANGE entries from CloudKit re-syncs.
    Both must remain visible because both have ZSERVERRECORDDATA set."""
    extra = (
        _achange_delete_for(folder_pk=2, achange_pk=900)
        + _achange_delete_for(folder_pk=3, achange_pk=901)
    )
    db_path = tmp_path / "NoteStore_system_folders_stale.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    folders = db.list_folders()
    names = sorted(f["name"] for f in folders)
    assert "Notes" in names
    assert "Recently Deleted" in names


def test_list_folders_hides_user_folder_with_stale_achange_and_no_server_record(tmp_path, monkeypatch):
    """Negative branch: a folder WITHOUT a CloudKit server record AND with a
    stale ACHANGE delete-commit must remain hidden — the guard does not save
    folders the cloud hasn't acknowledged."""
    # Create a brand-new folder with ZSERVERRECORDDATA NULL and ZNEEDSINITIALFETCHFROMCLOUD=0
    # (so it is NOT a ghost — the existing ghost_filter doesn't hide it).
    extra = (
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT8, "
        " ZSERVERRECORDDATA, ZNEEDSINITIALFETCHFROMCLOUD) "
        "VALUES (50, 15, 'NewLocalFolder', 'new-local-folder', 0, 1, NULL, 0);"
        + _achange_delete_for(folder_pk=50, achange_pk=910)
    )
    db_path = tmp_path / "NoteStore_user_folder_no_record.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    folders = db.list_folders()
    names = sorted(f["name"] for f in folders)
    # Without the new guard saving it, the ACHANGE filter must hide it.
    assert "NewLocalFolder" not in names, (
        f"Folder without server record should stay hidden when ACHANGE marks it deleted: {names!r}"
    )
    # Sanity: the original folders are still visible.
    assert "Notes" in names
    assert "Work" in names


def test_list_folders_keeps_ghost_filter_for_folders_with_no_server_record_and_pending_fetch(
    tmp_path, monkeypatch
):
    """Adjacent invariant: a true ghost (ZSERVERRECORDDATA NULL +
    ZNEEDSINITIALFETCHFROMCLOUD=1) without any ACHANGE entry must still be
    hidden by the existing ghost_filter — proving the new guard didn't
    accidentally widen the gate for placeholders the cloud will never fetch."""
    extra = (
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT8, "
        " ZSERVERRECORDDATA, ZNEEDSINITIALFETCHFROMCLOUD) "
        "VALUES (51, 15, 'GhostFolder', 'ghost-folder', 0, 1, NULL, 1);"
    )
    db_path = tmp_path / "NoteStore_ghost.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    folders = db.list_folders()
    names = [f["name"] for f in folders]
    assert "GhostFolder" not in names


def test_child_folder_pks_keeps_subfolder_when_stale_achange_and_server_record(
    tmp_path, monkeypatch
):
    """Subfolder under Work (PK=4) with ZSERVERRECORDDATA NOT NULL must
    survive a stale ACHANGE delete entry."""
    # Subfolder PK=6 already exists under Work in the base fixture, with
    # ZSERVERRECORDDATA = X'01'. Just stamp a stale ACHANGE on it.
    extra = _achange_delete_for(folder_pk=6, achange_pk=920)
    db_path = tmp_path / "NoteStore_subfolder_stale.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    children = db.child_folder_pks(4)
    assert children == [6]


def test_child_folder_pks_hides_subfolder_with_stale_achange_and_no_server_record(
    tmp_path, monkeypatch
):
    """A subfolder WITHOUT a server record AND with a stale ACHANGE row
    must be filtered out by child_folder_pks — confirms the negative
    branch of the new guard."""
    extra = (
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT8, "
        " ZSERVERRECORDDATA, ZNEEDSINITIALFETCHFROMCLOUD, ZPARENT) "
        "VALUES (52, 15, 'LocalChild', 'local-child', 0, 1, NULL, 0, 4);"
        + _achange_delete_for(folder_pk=52, achange_pk=921)
    )
    db_path = tmp_path / "NoteStore_local_child_stale.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    children = db.child_folder_pks(4)
    # Existing Subfolder (PK=6, server record set) survives.
    assert 6 in children
    # The local child without a server record is hidden.
    assert 52 not in children


def test_list_notes_keeps_notes_when_parent_has_stale_achange_and_server_record(
    tmp_path, monkeypatch
):
    """End-to-end check that _live_folder_subquery (called inside list_notes)
    also benefits from the guard. Parent folder Work (PK=4) carries a
    stale ACHANGE delete; its note (PK=11) must still surface."""
    extra = _achange_delete_for(folder_pk=4, achange_pk=930)
    db_path = tmp_path / "NoteStore_work_stale.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    rows, _, _, total = db.list_notes({4}, limit=100)
    assert total == 1
    assert rows[0]["id"] == "p11"


def test_search_notes_keeps_results_when_parent_has_stale_achange_and_server_record(
    tmp_path, monkeypatch
):
    """search_notes also relies on _live_folder_subquery; same guarantee."""
    extra = _achange_delete_for(folder_pk=2, achange_pk=940)
    db_path = tmp_path / "NoteStore_default_search_stale.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    matches, _, _, _ = db.search_notes(
        "Note in Notes", folder_pks=None, search_body=False, limit=100
    )
    titles = sorted(m[0]["title"] for m in matches)
    assert "Note in Notes" in titles


def test_list_notes_excludes_notes_in_parent_with_no_server_record_and_stale_achange(
    tmp_path, monkeypatch
):
    """Mirror negative case for the live-folder subquery: notes in a parent
    that genuinely fails the live-folder filter must NOT be returned."""
    # New folder with ZSERVERRECORDDATA NULL (and NOT a ghost) + stale ACHANGE
    # + a note inside it.
    extra = (
        "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT8, "
        " ZSERVERRECORDDATA, ZNEEDSINITIALFETCHFROMCLOUD) "
        "VALUES (60, 15, 'OrphanLocal', 'orphan-local', 0, 1, NULL, 0);"
        + _achange_delete_for(folder_pk=60, achange_pk=950)
        + "INSERT INTO ZICCLOUDSYNCINGOBJECT "
        "(Z_PK, Z_ENT, ZTITLE1, ZIDENTIFIER, ZFOLDER, ZMODIFICATIONDATE1) "
        "VALUES (70, 12, 'Orphan Note', 'orphan-note', 60, 7800000);"
    )
    db_path = tmp_path / "NoteStore_orphan_local.sqlite"
    _build_db(db_path, extra_sql=extra)
    _patch_notestore(monkeypatch, db_path)

    rows, _, _, total = db.list_notes({60}, limit=100)
    # Parent fails the live-folder subquery → its note is filtered out.
    assert total == 0
    assert rows == []
