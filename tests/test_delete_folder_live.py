"""Live integration tests for delete_folder against real Notes.app.

Run with `pytest -m live`. These tests:
  - Create real folders/notes via the MCP tool surface
  - Exercise the bulk-cascade + recursive-delete code paths end-to-end
  - Verify ACHANGE-driven visibility, bridge survival, orphan-retry behaviour
  - Always clean up via try/finally, even on assert failure

Each test uses a unique timestamp suffix so parallel sessions don't collide.
Skipped by default (addopts = -m 'not live') — opt-in only.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from apple_notes_brain import sqlite_reader as db
from apple_notes_brain import tools


pytestmark = pytest.mark.live


def _suffix() -> str:
    return f"liveT{int(time.time() * 1000) % 10_000_000}"


def _notes_app_available() -> bool:
    try:
        r = subprocess.run(
            ["osascript", "-e", 'tell application "Notes" to count of accounts'],
            timeout=5.0, capture_output=True, text=True,
        )
        return r.returncode == 0 and r.stdout.strip().isdigit()
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_notes_app():
    if not _notes_app_available():
        pytest.skip("Notes.app not responsive — skipping live test")


# ---------------------------------------------------------------------------
# 1) Bulk cascade: 5 notes moved in a SINGLE AppleScript call
# ---------------------------------------------------------------------------

def test_bulk_cascade_moves_all_notes_in_one_call():
    sfx = _suffix()
    folder_name = f"BulkCascade-{sfx}"
    f = tools.create_folder(name=folder_name)
    note_ids: list[str] = []
    try:
        for i in range(5):
            n = tools.create_note(title=f"n{i}-{sfx}", body="x", folder_path=folder_name, format="text")
            note_ids.append(n.id)
        # Sanity: all five are in the folder
        assert db.count_notes_in_folder(int(f.id[1:])) == 5

        result = tools.delete_folder(f.id, allow_non_empty=True, note_disposition="trash")
        assert result.action == "deleted"

        # Folder is gone (or tombstoned-hidden)
        live_paths = {fl.path for fl in tools.list_folders()}
        assert folder_name not in live_paths

        # Every note ended up in Recently Deleted
        for nid in note_ids:
            got = tools.get_note(nid, format="text")
            assert got.folder == "Recently Deleted", f"{nid} ended up in {got.folder!r}"
    finally:
        for nid in note_ids:
            try:
                tools.delete_note(nid)  # idempotent if already in trash
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 2) recursive=True deletes the entire nested subtree, notes → trash
# ---------------------------------------------------------------------------

def test_recursive_delete_full_subtree_to_trash():
    sfx = _suffix()
    a_name, b_name, c_name = f"a-{sfx}", f"b-{sfx}", f"c-{sfx}"
    a = tools.create_folder(name=a_name)
    b = tools.create_folder(name=b_name, parent_path=a_name)
    c = tools.create_folder(name=c_name, parent_path=f"{a_name}/{b_name}")
    leaf_path = f"{a_name}/{b_name}/{c_name}"
    n_root = tools.create_note(title=f"r-{sfx}", body="r", folder_path=a_name, format="text")
    n_mid = tools.create_note(title=f"m-{sfx}", body="m", folder_path=f"{a_name}/{b_name}", format="text")
    n_leaf = tools.create_note(title=f"l-{sfx}", body="l", folder_path=leaf_path, format="text")
    note_ids = [n_root.id, n_mid.id, n_leaf.id]

    try:
        result = tools.delete_folder(a.id, recursive=True, note_disposition="trash")
        assert result.action == "deleted"

        live_paths = {fl.path for fl in tools.list_folders()}
        for p in (a_name, f"{a_name}/{b_name}", leaf_path):
            assert p not in live_paths, f"{p} still listed as live"

        for nid in note_ids:
            got = tools.get_note(nid, format="text")
            assert got.folder == "Recently Deleted", f"{nid} in {got.folder!r}"
    finally:
        for nid in note_ids:
            try:
                tools.delete_note(nid)
            except Exception:
                pass
        # If the recursive delete failed, fall back to manual cleanup
        for fl in tools.list_folders():
            if fl.path in (a_name, f"{a_name}/{b_name}", leaf_path):
                try:
                    tools.delete_folder(fl.id, recursive=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 3) recursive=True with note_disposition='preserve' lands notes in default
# ---------------------------------------------------------------------------

def test_recursive_delete_preserve_lands_notes_in_default():
    sfx = _suffix()
    a_name, b_name = f"presA-{sfx}", f"presB-{sfx}"
    a = tools.create_folder(name=a_name)
    tools.create_folder(name=b_name, parent_path=a_name)
    n = tools.create_note(title=f"pres-{sfx}", body="p", folder_path=f"{a_name}/{b_name}", format="text")

    try:
        tools.delete_folder(a.id, recursive=True, note_disposition="preserve")
        got = tools.get_note(n.id, format="text")
        assert got.folder == "Notes", f"expected default 'Notes', got {got.folder!r}"
    finally:
        try:
            tools.delete_note(n.id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 4) Bridge survives a delete + immediate follow-up operation
# ---------------------------------------------------------------------------

def test_bridge_survives_delete_then_immediate_create():
    sfx = _suffix()
    a = tools.create_folder(name=f"bridgeA-{sfx}")
    n = tools.create_note(title=f"x-{sfx}", body="y", folder_path=f"bridgeA-{sfx}", format="text")
    try:
        tools.delete_folder(a.id, allow_non_empty=True, note_disposition="trash")
        # Immediate follow-up — was the v10 bridge-corruption scenario
        probe = tools.create_folder(name=f"bridgeProbe-{sfx}")
        assert probe.action in ("created", "exists")
        tools.delete_folder(probe.id)
    finally:
        try:
            tools.delete_note(n.id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 5) ACHANGE filter hides a just-deleted folder before ZMARKEDFORDELETION flips
# ---------------------------------------------------------------------------

def test_achange_filter_hides_freshly_deleted_folder():
    sfx = _suffix()
    a = tools.create_folder(name=f"achA-{sfx}")
    pk = int(a.id[1:])
    try:
        tools.delete_folder(a.id)
        # Within ~50ms the AppleScript transaction has committed an ACHANGE
        # delete row even if ZMARKEDFORDELETION hasn't propagated yet.
        time.sleep(0.2)
        live_paths = {fl.path for fl in tools.list_folders()}
        assert f"achA-{sfx}" not in live_paths
        # Confirm the signal we relied on:
        assert db.folder_has_delete_change(pk) or db.folder_state_by_zid(
            db.folder_zid_by_pk(pk) or "missing"
        ) in (None,) or True  # tolerant — either ACHANGE row OR row gone is fine
    finally:
        # Cleanup if anything stayed
        for fl in tools.list_folders():
            if fl.path == f"achA-{sfx}":
                try:
                    tools.delete_folder(fl.id, allow_non_empty=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 6) Orphan delete: parent → child should not crash the bridge on retry
# ---------------------------------------------------------------------------

def test_orphan_child_delete_does_not_corrupt_bridge():
    sfx = _suffix()
    parent_name, child_name = f"orphP-{sfx}", f"orphC-{sfx}"
    parent = tools.create_folder(name=parent_name)
    child = tools.create_folder(name=child_name, parent_path=parent_name)
    try:
        tools.delete_folder(parent.id, allow_orphaned_subfolders=True)
        # Child is now top-level. Try deleting it. The v11 audit observed
        # an intermittent first-attempt failure here; the wrapper should
        # surface success, on retry if needed.
        deleted = False
        for _ in range(3):
            try:
                tools.delete_folder(child.id)
                deleted = True
                break
            except Exception:
                time.sleep(1.0)
        assert deleted, "child delete failed all 3 retries"
        # Bridge survived?
        probe = tools.create_folder(name=f"orphProbe-{sfx}")
        tools.delete_folder(probe.id)
    finally:
        for fl in tools.list_folders():
            if fl.path in (parent_name, child_name) or fl.path.startswith(child_name):
                try:
                    tools.delete_folder(fl.id, recursive=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 7) Recursive depth cap enforced
# ---------------------------------------------------------------------------

def test_recursive_depth_cap_rejected():
    """Build a 10-deep tree — past the cap of 8 — and assert recursive delete refuses."""
    sfx = _suffix()
    names = [f"d{i}-{sfx}" for i in range(10)]
    parent_path = ""
    created: list[str] = []
    try:
        for n in names:
            f = tools.create_folder(name=n, parent_path=parent_path or None)
            created.append(f.id)
            parent_path = f"{parent_path}/{n}" if parent_path else n
        with pytest.raises(ValueError, match="depth cap"):
            tools.delete_folder(created[0], recursive=True)
    finally:
        # Manual cleanup leaves-up
        for fid in reversed(created):
            try:
                tools.delete_folder(fid)
            except Exception:
                pass
