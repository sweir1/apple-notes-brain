"""v1.1 Part 4 Phase 2 — delete_note verification behaviour.

When AppleScript reports success but SQLite hasn't yet committed the move
to Recently Deleted (Notes.app's MOC backlogs under load), we used to
raise. Now we:

  1. Fall back to an AppleScript object-graph probe (the same trick
     _attempt_folder_delete uses) — if AS reports the container IS the
     trash folder (or the note is un-addressable), promote to
     verified=True.
  2. Only if BOTH SQLite and the AS probe fail to confirm, return
     MutationResult(verified=False, warning=...) — no raise.

The pre-existing happy paths (SQLite confirms the move) keep returning
verified=True (default) with no warning.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain import applescript as _aps
from apple_notes_brain import tools
from apple_notes_brain.schemas import MutationResult


# Patch surface that's common to every test in this file. We bypass the
# heavy resolve_id / note_meta / share_role / cache.refresh path so the
# test focuses on the verification branch.
def _delete_note_patch_stack(
    *,
    note_state_responses,
    note_has_delete_change=False,
    aps_run=lambda script: "",
):
    """Return a list of patch context managers."""
    state_iter = iter(note_state_responses)
    return [
        patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)),
        patch("apple_notes_brain.sqlite_reader.note_meta",
              return_value={"locked": False, "folder_pk": 99, "title": "x"}),
        patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}),
        patch("apple_notes_brain.sqlite_reader.note_share_role", return_value=None),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              return_value="x-coredata://X/ICNote/p100"),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"),
        patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"),
        patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
              side_effect=lambda _z: next(state_iter)),
        patch("apple_notes_brain.sqlite_reader.note_has_delete_change",
              return_value=note_has_delete_change),
        patch("apple_notes_brain.applescript.run", side_effect=aps_run),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.tools._aps_run_with_recovery",
              side_effect=lambda script, **kw: aps_run(script)),
        patch("apple_notes_brain.cache.sync_after_write", return_value=None),
        patch("apple_notes_brain.cache.adjust_note_count", return_value=None),
        # The zid lookup uses db._open() context manager; we don't need a
        # real DB — mock it to return a fake row tuple.
        patch("apple_notes_brain.sqlite_reader._open"),
    ]


def _enter_all(stack):
    """Enter every patch in the stack as a list of mocks."""
    entered = []
    for cm in stack:
        entered.append(cm.__enter__())
    return entered, stack


def _exit_all(stack):
    for cm in reversed(stack):
        cm.__exit__(None, None, None)


def test_delete_note_happy_path_returns_verified_true():
    """SQLite confirms the move on the first poll. Returns
    MutationResult with verified=True (default) and no warning."""
    # Verifier sees the note in trash immediately.
    states = [
        {"pk": 100, "folder_pk": 2, "marked": 0},
    ]
    stack = _delete_note_patch_stack(note_state_responses=states)
    _, stack = _enter_all(stack)
    try:
        result = tools.delete_note("p100")
    finally:
        _exit_all(stack)
    assert isinstance(result, MutationResult)
    assert result.action == "deleted"
    assert result.verified is True
    assert result.warning is None


def test_delete_note_returns_verified_false_when_both_signals_fail(monkeypatch):
    """SQLite never moves the note + AS probe also fails to confirm →
    return verified=False with a warning. No raise."""
    # Shorten the timeout so the test doesn't actually wait 60s.
    monkeypatch.setattr(tools, "MOC_COMMIT_TIMEOUT_S", 0.5)

    # Note stays in source folder forever — SQLite never confirms.
    states_inf = iter([{"pk": 100, "folder_pk": 99, "marked": 0}] * 100)

    def aps_run_returns_source(script, **kw):
        # AS probe returns the source folder name → does NOT contain "delet"
        if "container of" in script:
            return "Some Random Folder"
        return ""

    stack = [
        patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)),
        patch("apple_notes_brain.sqlite_reader.note_meta",
              return_value={"locked": False, "folder_pk": 99, "title": "x"}),
        patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}),
        patch("apple_notes_brain.sqlite_reader.note_share_role", return_value=None),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              return_value="x-coredata://X/ICNote/p100"),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"),
        patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"),
        patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
              side_effect=lambda _z: next(states_inf)),
        patch("apple_notes_brain.sqlite_reader.note_has_delete_change", return_value=False),
        patch("apple_notes_brain.applescript.run", side_effect=aps_run_returns_source),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.tools._aps_run_with_recovery",
              side_effect=lambda script, **kw: aps_run_returns_source(script)),
        patch("apple_notes_brain.cache.sync_after_write", return_value=None),
        patch("apple_notes_brain.cache.adjust_note_count", return_value=None),
        patch("apple_notes_brain.sqlite_reader._open"),
        # Mock the inline ZID lookup (db._open in tools.py uses context manager)
    ]
    # Mock the inline ZID fetch — return a fake zid so the verifier branch fires.
    def fake_open():
        class _Conn:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                pass
            def execute(self_inner, *args, **kw):
                class _Cur:
                    def fetchone(self_c):
                        return ("zid-abc",)
                return _Cur()
        return _Conn()

    stack[-1] = patch("apple_notes_brain.sqlite_reader._open", side_effect=fake_open)

    _, stack = _enter_all(stack)
    try:
        result = tools.delete_note("p100")
    finally:
        _exit_all(stack)

    assert isinstance(result, MutationResult)
    assert result.action == "deleted"
    assert result.verified is False
    assert result.warning is not None
    # The warning must mention retry semantics so the model knows what to do.
    assert "retry" in result.warning.lower() or "shortly" in result.warning.lower()


def test_delete_note_as_probe_promotes_to_verified_true_on_trash_container(monkeypatch):
    """SQLite doesn't confirm in time, but the AS probe says the
    container IS 'Recently Deleted'. We trust the AS object graph and
    return verified=True (no warning) even though SQLite is lagging."""
    monkeypatch.setattr(tools, "MOC_COMMIT_TIMEOUT_S", 0.5)
    states_inf = iter([{"pk": 100, "folder_pk": 99, "marked": 0}] * 100)

    def aps_run_returns_trash(script, **kw):
        if "container of" in script:
            return "Recently Deleted"
        return ""

    def fake_open():
        class _Conn:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                pass
            def execute(self_inner, *args, **kw):
                class _Cur:
                    def fetchone(self_c):
                        return ("zid-abc",)
                return _Cur()
        return _Conn()

    stack = [
        patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)),
        patch("apple_notes_brain.sqlite_reader.note_meta",
              return_value={"locked": False, "folder_pk": 99, "title": "x"}),
        patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}),
        patch("apple_notes_brain.sqlite_reader.note_share_role", return_value=None),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              return_value="x-coredata://X/ICNote/p100"),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"),
        patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"),
        patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
              side_effect=lambda _z: next(states_inf)),
        patch("apple_notes_brain.sqlite_reader.note_has_delete_change", return_value=False),
        patch("apple_notes_brain.applescript.run", side_effect=aps_run_returns_trash),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.tools._aps_run_with_recovery",
              side_effect=lambda script, **kw: aps_run_returns_trash(script)),
        patch("apple_notes_brain.cache.sync_after_write", return_value=None),
        patch("apple_notes_brain.cache.adjust_note_count", return_value=None),
        patch("apple_notes_brain.sqlite_reader._open", side_effect=fake_open),
    ]
    _, stack = _enter_all(stack)
    try:
        result = tools.delete_note("p100")
    finally:
        _exit_all(stack)
    assert result.verified is True
    assert result.warning is None


def test_delete_note_as_probe_promotes_to_verified_true_on_unaddressable(monkeypatch):
    """SQLite doesn't confirm in time. AS raises 'Invalid index' on the
    probe — meaning AS can no longer address the note. That's a
    confirmed delete even without SQLite catching up."""
    monkeypatch.setattr(tools, "MOC_COMMIT_TIMEOUT_S", 0.5)
    states_inf = iter([{"pk": 100, "folder_pk": 99, "marked": 0}] * 100)

    def aps_run_raises_invalid(script, **kw):
        if "container of" in script:
            raise _aps.AppleScriptError(
                "osascript failed: Can't get note 1. Invalid index. (-1719)"
            )
        return ""

    def fake_open():
        class _Conn:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                pass
            def execute(self_inner, *args, **kw):
                class _Cur:
                    def fetchone(self_c):
                        return ("zid-abc",)
                return _Cur()
        return _Conn()

    stack = [
        patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)),
        patch("apple_notes_brain.sqlite_reader.note_meta",
              return_value={"locked": False, "folder_pk": 99, "title": "x"}),
        patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}),
        patch("apple_notes_brain.sqlite_reader.note_share_role", return_value=None),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              return_value="x-coredata://X/ICNote/p100"),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"),
        patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"),
        patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
              side_effect=lambda _z: next(states_inf)),
        patch("apple_notes_brain.sqlite_reader.note_has_delete_change", return_value=False),
        patch("apple_notes_brain.applescript.run", side_effect=aps_run_raises_invalid),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.tools._aps_run_with_recovery",
              side_effect=lambda script, **kw: aps_run_raises_invalid(script)),
        patch("apple_notes_brain.cache.sync_after_write", return_value=None),
        patch("apple_notes_brain.cache.adjust_note_count", return_value=None),
        patch("apple_notes_brain.sqlite_reader._open", side_effect=fake_open),
    ]
    _, stack = _enter_all(stack)
    try:
        result = tools.delete_note("p100")
    finally:
        _exit_all(stack)
    assert result.verified is True


def test_mutation_result_schema_has_verified_and_warning():
    """The schema must declare the new fields with sensible defaults."""
    # Default: verified=True, warning=None.
    r = MutationResult(id="p100", action="deleted")
    assert r.verified is True
    assert r.warning is None

    # Explicit verified=False + warning round-trips through Pydantic.
    r2 = MutationResult(
        id="p100", action="deleted", verified=False, warning="MOC backlog",
    )
    assert r2.verified is False
    assert r2.warning == "MOC backlog"


def test_delete_note_does_not_raise_on_moc_timeout(monkeypatch):
    """Regression: pre-1.1 raised ValueError on MOC timeout. 1.1 must
    NOT raise — it returns verified=False instead. This locks in the
    contract shift so a future revert is caught."""
    monkeypatch.setattr(tools, "MOC_COMMIT_TIMEOUT_S", 0.3)
    states_inf = iter([{"pk": 100, "folder_pk": 99, "marked": 0}] * 100)

    def aps_run_neither_confirms(script, **kw):
        if "container of" in script:
            return "Some Other Folder"
        return ""

    def fake_open():
        class _Conn:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                pass
            def execute(self_inner, *args, **kw):
                class _Cur:
                    def fetchone(self_c):
                        return ("zid-abc",)
                return _Cur()
        return _Conn()

    stack = [
        patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)),
        patch("apple_notes_brain.sqlite_reader.note_meta",
              return_value={"locked": False, "folder_pk": 99, "title": "x"}),
        patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}),
        patch("apple_notes_brain.sqlite_reader.note_share_role", return_value=None),
        patch("apple_notes_brain.sqlite_reader.to_uri",
              return_value="x-coredata://X/ICNote/p100"),
        patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"),
        patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100"),
        patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
              side_effect=lambda _z: next(states_inf)),
        patch("apple_notes_brain.sqlite_reader.note_has_delete_change", return_value=False),
        patch("apple_notes_brain.applescript.run", side_effect=aps_run_neither_confirms),
        patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'),
        patch("apple_notes_brain.tools._aps_run_with_recovery",
              side_effect=lambda script, **kw: aps_run_neither_confirms(script)),
        patch("apple_notes_brain.cache.sync_after_write", return_value=None),
        patch("apple_notes_brain.cache.adjust_note_count", return_value=None),
        patch("apple_notes_brain.sqlite_reader._open", side_effect=fake_open),
    ]
    _, stack = _enter_all(stack)
    try:
        # Must NOT raise — must return verified=False instead.
        result = tools.delete_note("p100")
    finally:
        _exit_all(stack)
    assert isinstance(result, MutationResult)
    assert result.verified is False
