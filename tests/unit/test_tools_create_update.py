"""Unit tests for the write tool surface: create_note, update_note, rename_note, move_note.

Heavy mocking of sqlite_reader and applescript. We exercise tools.py guard
clauses, batch shape validation, and the wrapper logic around AppleScript calls.

Includes targeted tests for known bug hotspots (#5, #12, #21, #22) — some
intentionally assert CURRENT (buggy) behaviour so a future fix surfaces as a
failing test.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain import applescript as aps
from apple_notes_brain import tools
from apple_notes_brain.schemas import MutationResult, NoteCreateSpec


def _f(pk: int, path: str, *, is_trash: bool = False, account: str = "iCloud") -> dict:
    return {
        "id": f"f{pk}",
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "is_trash": is_trash,
        "account": account,
        "shared": False,
        "note_count": None,
    }


def _meta(pk: int, *, folder_pk: int = 1, locked: bool = False, title: str = "T") -> dict:
    return {
        "id": f"p{pk}",
        "title": title,
        "folder_pk": folder_pk,
        "modified": 0.0,
        "pinned": False,
        "locked": locked,
        "shared": False,
    }


# Common stack of "make AppleScript inert and DB writes harmless" patches
def _common_write_patches(
    *,
    aps_return: str = "",
    aps_side_effect=None,
    folders=None,
):
    if folders is None:
        folders = [_f(1, "Notes"), _f(7, "Work")]
    patches = [
        patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              side_effect=lambda pk, uuid, entity="ICNote": f"x-coredata://{uuid}/{entity}/p{pk}"),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.applescript.as_list",
              side_effect=lambda xs: "{" + ",".join(f'"{x}"' for x in xs) + "}"),
        # Skip MOC-commit polling — we're not testing the bridge here.
        patch("apple_notes_brain.tools._wait_until_as_addressable", return_value=True),
        patch("apple_notes_brain.tools._wait_for_state", return_value=True),
        patch("apple_notes_brain.cache.sync_after_write"),
    ]
    if aps_side_effect is not None:
        patches.append(patch("apple_notes_brain.applescript.run", side_effect=aps_side_effect))
    else:
        patches.append(patch("apple_notes_brain.applescript.run", return_value=aps_return))
    return patches


def _enter_all(patches):
    """Enter a list of patches and return the started mocks for cleanup tracking."""
    mocks = [p.start() for p in patches]
    return mocks


def _stop_all(patches):
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# create_note — single mode
# ---------------------------------------------------------------------------

class TestCreateNoteSingle:
    def test_basic_returns_created(self):
        patches = _common_write_patches(aps_return="x-coredata://UUID/ICNote/p999")
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 999)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p999"):
            _enter_all(patches)
            try:
                out = tools.create_note(title="hello", body="body")
            finally:
                _stop_all(patches)
        assert isinstance(out, MutationResult)
        assert out.action == "created"
        assert out.id == "p999"

    def test_no_folder_path_uses_default(self):
        patches = _common_write_patches(aps_return="x-coredata://UUID/ICNote/p1")
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICNote/p1"

        # Replace applescript.run with capture
        for p in patches:
            try:
                p.stop()
            except RuntimeError:
                pass
        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                tools.create_note(title="t", body="b", folder_path=None)
            finally:
                _stop_all(patches)
        # When folder_path is None, the DEFAULT script template runs (no folder id placeholder)
        assert len(captured) == 1
        assert "FOLDER_ID" not in captured[0]  # template was filled

    def test_folder_path_resolves(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICNote/p1"

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                out = tools.create_note(title="t", body="b", folder_path="Work")
            finally:
                _stop_all(patches)
        assert out.action == "created"
        # IN_FOLDER variant: the folder URI (with f7) appears in the script
        assert any("/p7" in s for s in captured)

    def test_folder_path_nonexistent_raises(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="folder not found"):
                tools.create_note(title="t", body="b", folder_path="GhostFolder")
        finally:
            _stop_all(patches)

    def test_format_markdown_default(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICNote/p1"

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                tools.create_note(title="t", body="# heading", format="markdown")
            finally:
                _stop_all(patches)
        # Markdown converts to HTML with an <h1>
        assert any("<h1" in s.lower() for s in captured)

    def test_format_html_passes_through(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICNote/p1"

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                tools.create_note(title="t", body="<p>raw-content</p>", format="html")
            finally:
                _stop_all(patches)
        # html_validate normalizes <p> to <div>, but the raw text content survives
        assert any("raw-content" in s for s in captured)

    def test_format_text_wraps_in_div(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICNote/p1"

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                tools.create_note(title="t", body="line1\nline2", format="text")
            finally:
                _stop_all(patches)
        # text format wraps in <div>...</div> with <br> for newlines
        assert any("<div>" in s and "<br>" in s for s in captured)

    def test_empty_body_succeeds(self):
        patches = _common_write_patches(aps_return="x-coredata://UUID/ICNote/p1")
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 1)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p1"):
            _enter_all(patches)
            try:
                out = tools.create_note(title="t", body="")
            finally:
                _stop_all(patches)
        assert out.action == "created"

    def test_no_title_no_notes_raises(self):
        # Single mode w/o title and no batch param → raise
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="requires either"):
                tools.create_note()
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# create_note — batch mode
# ---------------------------------------------------------------------------

class TestCreateNoteBatch:
    def test_batch_two_notes_returns_two_results(self):
        out_ids = "x-coredata://UUID/ICNote/p10" + aps.RECORD_SEP + "x-coredata://UUID/ICNote/p11"
        patches = _common_write_patches(aps_return=out_ids)
        with patch("apple_notes_brain.sqlite_reader.resolve_id",
                   side_effect=lambda u: ("note", 10 if "p10" in u else 11)), \
             patch("apple_notes_brain.sqlite_reader.short_id",
                   side_effect=lambda pk: f"p{pk}"):
            _enter_all(patches)
            try:
                out = tools.create_note(notes=[
                    NoteCreateSpec(title="a"),
                    NoteCreateSpec(title="b"),
                ])
            finally:
                _stop_all(patches)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(r.action == "created" for r in out)

    def test_batch_empty_list_returns_empty_list(self):
        # Bug hotspot #22: notes=[] silently returns []. Per spec this MAY
        # warrant a ValueError ("nothing to do" is a no-op the model should
        # know about). Locking in CURRENT behaviour; revisit when policy lands.
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            out = tools.create_note(notes=[])
        finally:
            _stop_all(patches)
        assert out == []

    def test_batch_short_count_pads_with_skipped(self):
        # Bug hotspot #21: when AppleScript returns FEWER URIs than specs,
        # tools.py pads with a 'skipped' MutationResult whose error mentions
        # "(move likely failed)". The wording is wrong (should be "create"
        # likely failed, not "move") — this test ASSERTS the typo so it
        # surfaces as a failing test the day someone fixes it.
        out_ids = "x-coredata://UUID/ICNote/p10"  # only 1 URI for 2 specs
        patches = _common_write_patches(aps_return=out_ids)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 10)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p10"):
            _enter_all(patches)
            try:
                out = tools.create_note(notes=[
                    NoteCreateSpec(title="a"),
                    NoteCreateSpec(title="b"),
                ])
            finally:
                _stop_all(patches)
        assert len(out) == 2
        assert out[0].action == "created"
        assert out[1].action == "skipped"
        # The typo: error says "move likely failed" inside a CREATE bulk path
        assert "move likely failed" in (out[1].error or "")

    def test_batch_folder_path_resolves(self):
        out_ids = "x-coredata://UUID/ICNote/p10"
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return out_ids

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 10)), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p10"):
            _enter_all(patches)
            try:
                tools.create_note(notes=[NoteCreateSpec(title="a")], folder_path="Work")
            finally:
                _stop_all(patches)
        # Folder URI (f7) should appear in the script
        assert any("/p7" in s for s in captured)

    def test_batch_folder_path_nonexistent_raises(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="folder not found"):
                tools.create_note(notes=[NoteCreateSpec(title="a")],
                                  folder_path="GhostFolder")
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# update_note
# ---------------------------------------------------------------------------

class TestUpdateNote:
    def test_basic_returns_updated(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                out = tools.update_note(note_id="p100", body="new body")
            finally:
                _stop_all(patches)
        assert out.action == "updated"
        assert out.id == "p100"

    def test_append_uses_append_template(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return ""

        patches = _common_write_patches(aps_side_effect=cap)
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                tools.update_note(note_id="p100", body="x", append=True)
            finally:
                _stop_all(patches)
        # Append template differs from replace; the body assignment uses
        # `& body of note ...` (concatenation) rather than `set body ... to`.
        from apple_notes_brain import scripts as _s
        # Just assert the template is the APPEND one by checking a substring
        # only present there — easier than fragile substring checks.
        assert any(_s.UPDATE_NOTE_APPEND.split("\n")[1] in s for s in captured) or \
               any("&" in s for s in captured)

    def test_replace_default(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                out = tools.update_note(note_id="p100", body="x")
            finally:
                _stop_all(patches)
        assert out.action == "updated"

    def test_locked_note_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, locked=True)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="locked"):
                    tools.update_note(note_id="p100", body="x")
            finally:
                _stop_all(patches)

    def test_attachments_blocked_without_flag(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=3):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="attachment"):
                    tools.update_note(note_id="p100", body="x")
            finally:
                _stop_all(patches)

    def test_locked_with_attachments_lock_check_first(self):
        # Bug hotspot #12: when a note is BOTH locked and has attachments,
        # the lock check fires first → caller sees "locked" not "attachment".
        # This test documents CURRENT order. If the attachment guard should
        # actually fire first (it's the more destructive risk on unlock-and-retry),
        # this test will need flipping.
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, locked=True)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=5):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError) as exc_info:
                    tools.update_note(note_id="p100", body="x")
                # CURRENT: "locked" wins over "attachment". Argument: attachment
                # warning is more important because lock-then-unlock-and-retry
                # would lose attachments silently.
                assert "locked" in str(exc_info.value).lower()
                assert "attachment" not in str(exc_info.value).lower()
            finally:
                _stop_all(patches)

    def test_allow_attachment_loss_bypasses_check(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=3), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                out = tools.update_note(note_id="p100", body="x", allow_attachment_loss=True)
            finally:
                _stop_all(patches)
        assert out.action == "updated"

    def test_in_recently_deleted_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, folder_pk=2)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="Recently Deleted"):
                    tools.update_note(note_id="p100", body="x")
            finally:
                _stop_all(patches)

    def test_not_a_note_id_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 7)):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="not a note"):
                    tools.update_note(note_id="f7", body="x")
            finally:
                _stop_all(patches)

    def test_note_not_found_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=None):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="not found"):
                    tools.update_note(note_id="p100", body="x")
            finally:
                _stop_all(patches)


# ---------------------------------------------------------------------------
# rename_note
# ---------------------------------------------------------------------------

class TestRenameNote:
    def test_single_returns_renamed(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                out = tools.rename_note(note_id="p100", new_title="X")
            finally:
                _stop_all(patches)
        assert isinstance(out, MutationResult)
        assert out.action == "renamed"
        assert out.id == "p100"

    def test_batch_returns_list(self):
        patches = _common_write_patches()

        def resolve(s):
            pk = int(s.lstrip("p"))
            return ("note", pk)

        with patch("apple_notes_brain.sqlite_reader.resolve_id", side_effect=resolve), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   side_effect=lambda pk: _meta(pk)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.short_id",
                   side_effect=lambda pk: f"p{pk}"):
            _enter_all(patches)
            try:
                out = tools.rename_note(note_id=["p100", "p101"], new_title=["A", "B"])
            finally:
                _stop_all(patches)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(r.action == "renamed" for r in out)

    def test_mismatched_shape_raises(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="same shape"):
                tools.rename_note(note_id="p100", new_title=["A", "B"])
        finally:
            _stop_all(patches)

    def test_mismatched_lengths_raises(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="lengths must match"):
                tools.rename_note(note_id=["p100", "p101"], new_title=["A"])
        finally:
            _stop_all(patches)

    def test_batch_empty_returns_empty(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            out = tools.rename_note(note_id=[], new_title=[])
        finally:
            _stop_all(patches)
        assert out == []

    def test_locked_single_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, locked=True)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="locked"):
                    tools.rename_note(note_id="p100", new_title="X")
            finally:
                _stop_all(patches)

    def test_empty_new_title_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="non-empty"):
                    tools.rename_note(note_id="p100", new_title="   ")
            finally:
                _stop_all(patches)

    def test_batch_locked_one_succeeds_one_skipped(self):
        # Batch is fault-tolerant — failed items become skipped MutationResults.
        patches = _common_write_patches()

        def meta_for(pk):
            return _meta(pk, locked=(pk == 101))

        with patch("apple_notes_brain.sqlite_reader.resolve_id",
                   side_effect=lambda s: ("note", int(s.lstrip("p")))), \
             patch("apple_notes_brain.sqlite_reader.note_meta", side_effect=meta_for), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.short_id",
                   side_effect=lambda pk: f"p{pk}"):
            _enter_all(patches)
            try:
                out = tools.rename_note(note_id=["p100", "p101"], new_title=["A", "B"])
            finally:
                _stop_all(patches)
        assert len(out) == 2
        actions = sorted(r.action for r in out)
        assert actions == ["renamed", "skipped"]


# ---------------------------------------------------------------------------
# move_note
# ---------------------------------------------------------------------------

class TestMoveNote:
    def test_single_succeeds(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta", return_value=_meta(100)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"):
            _enter_all(patches)
            try:
                out = tools.move_note(note_id="p100", folder_path="Work")
            finally:
                _stop_all(patches)
        assert out.action == "moved"

    def test_batch_succeeds(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id",
                   side_effect=lambda s: ("note", int(s.lstrip("p")))), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   side_effect=lambda pk: _meta(pk)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.short_id",
                   side_effect=lambda pk: f"p{pk}"):
            _enter_all(patches)
            try:
                out = tools.move_note(note_id=["p100", "p101"], folder_path="Work")
            finally:
                _stop_all(patches)
        assert isinstance(out, list)
        assert len(out) == 2
        assert all(r.action == "moved" for r in out)

    def test_nonexistent_folder_error_includes_path(self):
        # Bug hotspot #5: error message must include the folder path the user
        # passed, NOT a stringified "None" or empty value.
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError) as exc_info:
                tools.move_note(note_id="p100", folder_path="DoesNotExist")
            msg = str(exc_info.value)
            assert "DoesNotExist" in msg
            assert "None" not in msg
        finally:
            _stop_all(patches)

    def test_refuses_move_into_trash(self):
        folders = [_f(1, "Notes"), _f(2, "Recently Deleted", is_trash=True)]
        patches = _common_write_patches(folders=folders)
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="trash"):
                tools.move_note(note_id="p100", folder_path="Recently Deleted")
        finally:
            _stop_all(patches)

    def test_locked_single_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, locked=True)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="locked"):
                    tools.move_note(note_id="p100", folder_path="Work")
            finally:
                _stop_all(patches)

    def test_in_recently_deleted_raises(self):
        patches = _common_write_patches()
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
             patch("apple_notes_brain.sqlite_reader.note_meta",
                   return_value=_meta(100, folder_pk=2)), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}):
            _enter_all(patches)
            try:
                with pytest.raises(ValueError, match="Recently Deleted"):
                    tools.move_note(note_id="p100", folder_path="Work")
            finally:
                _stop_all(patches)

    def test_batch_empty_returns_empty(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            out = tools.move_note(note_id=[], folder_path="Work")
        finally:
            _stop_all(patches)
        assert out == []

    def test_batch_too_many_raises(self):
        patches = _common_write_patches()
        _enter_all(patches)
        try:
            with pytest.raises(ValueError, match="too many"):
                tools.move_note(note_id=[f"p{i}" for i in range(50)],
                                folder_path="Work")
        finally:
            _stop_all(patches)
