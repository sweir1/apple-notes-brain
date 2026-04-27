"""Unit tests for folder write tools: create_folder, rename_folder,
plus delete_folder paths NOT covered by tests/test_delete_folder_cascade.py
(empty/non-empty/default/trash refusals + tombstone application).

Heavy mocking of sqlite_reader and applescript.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain import cache
from apple_notes_brain import tools
from apple_notes_brain.schemas import MutationResult


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


# ---------------------------------------------------------------------------
# create_folder
# ---------------------------------------------------------------------------

class TestCreateFolder:
    def test_basic_top_level(self):
        # Default-account create returns a URI; resolve_id maps it to a pk
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.run",
                   return_value="x-coredata://UUID/ICFolder/p99"), \
             patch("apple_notes_brain.tools._wait_until_as_addressable", return_value=True), \
             patch("apple_notes_brain.cache.sync_after_write"), \
             patch("apple_notes_brain.sqlite_reader.resolve_id",
                   return_value=("folder", 99)), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id",
                   return_value="f99"):
            out = tools.create_folder(name="Inbox")
        assert isinstance(out, MutationResult)
        assert out.action == "created"
        assert out.id == "f99"

    def test_with_parent_folder(self):
        captured: list[str] = []

        def cap(script):
            captured.append(script)
            return "x-coredata://UUID/ICFolder/p100"

        folders = [_f(7, "Work")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.run", side_effect=cap), \
             patch("apple_notes_brain.tools._wait_until_as_addressable", return_value=True), \
             patch("apple_notes_brain.cache.sync_after_write"), \
             patch("apple_notes_brain.sqlite_reader.resolve_id",
                   return_value=("folder", 100)), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f100"):
            out = tools.create_folder(name="Sub", parent_folder_path="Work")
        assert out.action == "created"
        # The IN_FOLDER variant places the parent URI (f7) in the script
        assert any("/p7" in s for s in captured)

    def test_parent_nonexistent_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="parent folder not found"):
                tools.create_folder(name="x", parent_folder_path="GhostParent")

    def test_name_with_slash_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match=r"cannot contain '/'"):
                tools.create_folder(name="bad/name")

    def test_empty_name_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="non-empty"):
                tools.create_folder(name="")

    def test_whitespace_only_name_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="non-empty"):
                tools.create_folder(name="   \t  ")

    def test_refuses_creating_under_recently_deleted(self):
        folders = [_f(2, "Recently Deleted", is_trash=True)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders):
            with pytest.raises(ValueError, match="Recently Deleted"):
                tools.create_folder(name="x", parent_folder_path="Recently Deleted")

    def test_duplicate_sibling_name_refused(self):
        # Sibling-scope dup pre-check (case-insensitive)
        folders = [_f(7, "Work"), _f(8, "Work/Inbox")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders):
            with pytest.raises(ValueError, match="already exists"):
                tools.create_folder(name="inbox", parent_folder_path="Work")


# ---------------------------------------------------------------------------
# rename_folder
# ---------------------------------------------------------------------------

class TestRenameFolder:
    def test_basic(self):
        folders = [_f(7, "Work")]
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 7)), \
             patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id",
                   side_effect=lambda pk: f"f{pk}"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.run", return_value=""), \
             patch("apple_notes_brain.cache.sync_after_write"):
            out = tools.rename_folder(folder_id="f7", new_name="WorkRenamed")
        assert isinstance(out, MutationResult)
        assert out.action == "renamed"
        assert out.id == "f7"

    def test_not_found_raises(self):
        # resolve_id passes (folder, 99), but list_folders has no such folder
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 99)), \
             patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(7, "Work")]), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f99"):
            with pytest.raises(ValueError, match="folder not found"):
                tools.rename_folder(folder_id="f99", new_name="X")

    def test_empty_new_name_raises(self):
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 7)):
            with pytest.raises(ValueError, match="non-empty"):
                tools.rename_folder(folder_id="f7", new_name="")

    def test_whitespace_only_new_name_raises(self):
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 7)):
            with pytest.raises(ValueError, match="non-empty"):
                tools.rename_folder(folder_id="f7", new_name="   ")

    def test_slash_in_new_name_raises(self):
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 7)):
            with pytest.raises(ValueError, match=r"cannot contain '/'"):
                tools.rename_folder(folder_id="f7", new_name="a/b")

    def test_not_a_folder_id_raises(self):
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)):
            with pytest.raises(ValueError, match="not a folder"):
                tools.rename_folder(folder_id="p100", new_name="X")

    def test_refuses_trash_folder(self):
        folders = [_f(2, "Recently Deleted", is_trash=True)]
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 2)), \
             patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f2"):
            with pytest.raises(ValueError, match="trash"):
                tools.rename_folder(folder_id="f2", new_name="X")


# ---------------------------------------------------------------------------
# delete_folder — non-cascade paths
# ---------------------------------------------------------------------------

class TestDeleteFolderNonCascade:
    def _common(self, *, pk: int = 99, path: str = "Test", is_trash: bool = False,
                is_default: bool = False, is_shared: bool = False):
        folders = [_f(pk, path, is_trash=is_trash)]
        return [
            patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", pk)),
            patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders),
            patch("apple_notes_brain.sqlite_reader.is_default_folder",
                  return_value=is_default),
            patch("apple_notes_brain.sqlite_reader.folder_is_shared",
                  return_value=is_shared),
            patch("apple_notes_brain.sqlite_reader.short_folder_id",
                  side_effect=lambda p: f"f{p}"),
        ]

    def _enter(self, patches):
        return [p.start() for p in patches]

    def _stop(self, patches):
        for p in patches:
            try:
                p.stop()
            except RuntimeError:
                pass

    def test_empty_folder_succeeds(self):
        patches = self._common()
        with patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)), \
             patch("apple_notes_brain.sqlite_reader.child_folder_pks", return_value=[]), \
             patch("apple_notes_brain.sqlite_reader.notes_in_folder", return_value=[]), \
             patch("apple_notes_brain.sqlite_reader.folder_zid_by_pk", return_value="zid-99"), \
             patch("apple_notes_brain.sqlite_reader.folder_state_by_zid", return_value=None), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.run", return_value=""), \
             patch("apple_notes_brain.tools._wait_for_state", return_value=True), \
             patch("apple_notes_brain.cache.tombstone_folder") as mock_tomb, \
             patch("apple_notes_brain.cache.sync_after_write"):
            self._enter(patches)
            try:
                out = tools.delete_folder("f99")
            finally:
                self._stop(patches)
        assert out.action == "deleted"
        assert out.id == "f99"
        # Tombstone applied so list_folders hides it immediately
        mock_tomb.assert_called_once_with(99)

    def test_non_empty_no_flag_raises(self):
        patches = self._common()
        with patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 5)), \
             patch("apple_notes_brain.cache.get_count_delta", return_value=0):
            self._enter(patches)
            try:
                with pytest.raises(ValueError) as exc_info:
                    tools.delete_folder("f99")
                msg = str(exc_info.value)
                assert "5 note" in msg
                assert "trash" in msg.lower()
                assert "preserve" in msg.lower()
            finally:
                self._stop(patches)

    def test_non_empty_with_flag_cascades_to_trash(self):
        patches = self._common()
        # Two notes in the folder; cascade them all to trash (pk=2), then delete
        notes = [{"pk": 100}, {"pk": 101}]
        with patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 2)), \
             patch("apple_notes_brain.cache.get_count_delta", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.child_folder_pks", return_value=[]), \
             patch("apple_notes_brain.sqlite_reader.notes_in_folder", return_value=notes), \
             patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
             patch("apple_notes_brain.sqlite_reader.count_notes_in_folder", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.folder_zid_by_pk", return_value="zid-99"), \
             patch("apple_notes_brain.sqlite_reader.folder_state_by_zid", return_value=None), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.as_list",
                   side_effect=lambda xs: "{" + ",".join(xs) + "}"), \
             patch("apple_notes_brain.applescript.run", return_value=""), \
             patch("apple_notes_brain.tools._wait_for_state", return_value=True), \
             patch("apple_notes_brain.cache.tombstone_folder"), \
             patch("apple_notes_brain.cache.adjust_note_count"), \
             patch("apple_notes_brain.cache.sync_after_write"):
            self._enter(patches)
            try:
                out = tools.delete_folder("f99", allow_non_empty=True)
            finally:
                self._stop(patches)
        assert out.action == "deleted"

    def test_default_notes_folder_refused(self):
        patches = self._common(path="Notes", is_default=True)
        self._enter(patches)
        try:
            with pytest.raises(ValueError, match="default 'Notes' folder"):
                tools.delete_folder("f99", allow_non_empty=True)
        finally:
            self._stop(patches)

    def test_trash_folder_refused(self):
        patches = self._common(path="Recently Deleted", is_trash=True)
        self._enter(patches)
        try:
            with pytest.raises(ValueError, match="trash folder"):
                tools.delete_folder("f99", allow_non_empty=True)
        finally:
            self._stop(patches)

    def test_shared_folder_refused(self):
        patches = self._common(is_shared=True)
        self._enter(patches)
        try:
            with pytest.raises(ValueError, match="shared folder"):
                tools.delete_folder("f99", allow_non_empty=True)
        finally:
            self._stop(patches)

    def test_invalid_disposition_rejected(self):
        patches = self._common()
        self._enter(patches)
        try:
            with pytest.raises(ValueError, match="invalid note_disposition"):
                tools.delete_folder("f99",
                                    note_disposition="bogus")  # type: ignore[arg-type]
        finally:
            self._stop(patches)

    def test_not_a_folder_id_raises(self):
        with patch("apple_notes_brain.sqlite_reader.resolve_id",
                   return_value=("note", 100)):
            with pytest.raises(ValueError, match="not a folder"):
                tools.delete_folder("p100")

    def test_subfolders_without_orphan_flag_refused(self):
        patches = self._common()
        with patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)), \
             patch("apple_notes_brain.cache.get_count_delta", return_value=0), \
             patch("apple_notes_brain.sqlite_reader.child_folder_pks",
                   return_value=[200, 201]):
            self._enter(patches)
            try:
                with pytest.raises(ValueError, match="subfolder"):
                    tools.delete_folder("f99")
            finally:
                self._stop(patches)

    def test_tombstone_applied_after_success(self):
        # Verify tombstone_folder is called so list_folders hides immediately.
        patches = self._common()
        with patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)), \
             patch("apple_notes_brain.sqlite_reader.child_folder_pks", return_value=[]), \
             patch("apple_notes_brain.sqlite_reader.notes_in_folder", return_value=[]), \
             patch("apple_notes_brain.sqlite_reader.folder_zid_by_pk", return_value="zid-99"), \
             patch("apple_notes_brain.sqlite_reader.folder_state_by_zid", return_value=None), \
             patch("apple_notes_brain.sqlite_reader.to_uri",
                   side_effect=lambda pk, uuid, entity: f"x-coredata://{uuid}/{entity}/p{pk}"), \
             patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="UUID"), \
             patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'), \
             patch("apple_notes_brain.applescript.run", return_value=""), \
             patch("apple_notes_brain.tools._wait_for_state", return_value=True), \
             patch("apple_notes_brain.cache.sync_after_write"), \
             patch("apple_notes_brain.cache.tombstone_folder") as mock_tomb:
            self._enter(patches)
            try:
                tools.delete_folder("f99")
            finally:
                self._stop(patches)
        # Verify tombstone applied with correct pk
        mock_tomb.assert_called_once_with(99)
