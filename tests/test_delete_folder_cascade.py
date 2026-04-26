"""Regression tests for delete_folder cascade behaviour.

We test the cascade logic (move-each-note-then-delete-folder) without
actually touching the user's Apple Notes data — by mocking the AppleScript
runner and the SQLite state lookups.
"""
from unittest.mock import patch

import pytest

from apple_notes_brain import tools


def test_delete_folder_invalid_disposition_rejected():
    with pytest.raises(ValueError, match="invalid note_disposition"):
        with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 999)), \
             patch("apple_notes_brain.sqlite_reader.list_folders", return_value=[
                 {"id": "f999", "name": "Test", "path": "Test", "is_trash": False, "account": "iCloud"}
             ]), \
             patch("apple_notes_brain.sqlite_reader.is_default_folder", return_value=False), \
             patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f999"):
            tools.delete_folder("f999", note_disposition="bogus")  # type: ignore[arg-type]


def test_delete_folder_refuses_default_notes_folder():
    """Cannot delete the system default 'Notes' folder."""
    with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 3)), \
         patch("apple_notes_brain.sqlite_reader.list_folders", return_value=[
             {"id": "f3", "name": "Notes", "path": "Notes", "is_trash": False, "account": "iCloud"}
         ]), \
         patch("apple_notes_brain.sqlite_reader.is_default_folder", return_value=True), \
         patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f3"):
        with pytest.raises(ValueError, match="default 'Notes' folder"):
            tools.delete_folder("f3", allow_non_empty=True)


def test_delete_folder_refuses_non_empty_without_flag():
    """Without allow_non_empty, refuses with a message that mentions both dispositions."""
    with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("folder", 999)), \
         patch("apple_notes_brain.sqlite_reader.list_folders", return_value=[
             {"id": "f999", "name": "Test", "path": "Test", "is_trash": False, "account": "iCloud"}
         ]), \
         patch("apple_notes_brain.sqlite_reader.is_default_folder", return_value=False), \
         patch("apple_notes_brain.sqlite_reader.list_notes", return_value=([], None, False, 5)), \
         patch("apple_notes_brain.cache.get_count_delta", return_value=0), \
         patch("apple_notes_brain.sqlite_reader.short_folder_id", return_value="f999"):
        with pytest.raises(ValueError) as exc_info:
            tools.delete_folder("f999")
        msg = str(exc_info.value)
        assert "5 note" in msg
        assert "trash" in msg.lower() and "preserve" in msg.lower()


def test_cascade_to_trash_succeeds_when_note_actually_moves():
    """_cascade_note_to_trash returns success when SQLite shows the note left source folder."""
    states = iter([
        {"pk": 100, "folder_pk": 999, "marked": 0},  # initial
        {"pk": 100, "folder_pk": 2, "marked": 0},     # after AS (moved to trash folder pk=2)
    ])
    with patch("apple_notes_brain.sqlite_reader.note_state_by_zid", side_effect=lambda _: next(states)), \
         patch("apple_notes_brain.sqlite_reader.to_uri", return_value="x-coredata://X/ICNote/p100"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.applescript.run", return_value=""), \
         patch("apple_notes_brain.applescript.quote", return_value='"X"'):
        ok, err = tools._cascade_note_to_trash("zid-abc", 999, timeout_s=1.0)
    assert ok is True, err


def test_cascade_to_trash_detects_silent_lie():
    """When AppleScript returns success but SQLite shows note still in source, fail."""
    with patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
               return_value={"pk": 100, "folder_pk": 999, "marked": 0}), \
         patch("apple_notes_brain.sqlite_reader.to_uri", return_value="x-coredata://X/ICNote/p100"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.applescript.run", return_value="DELETED"), \
         patch("apple_notes_brain.applescript.quote", return_value='"X"'), \
         patch("apple_notes_brain.cache.refresh", return_value={"ok": True, "ms": 1}):
        ok, err = tools._cascade_note_to_trash("zid-abc", 999, timeout_s=0.5)
    assert ok is False
    assert "did not leave" in (err or "")


def test_cascade_to_folder_succeeds_when_note_arrives():
    states = iter([
        {"pk": 100, "folder_pk": 999, "marked": 0},  # initial — in source
        {"pk": 100, "folder_pk": 3, "marked": 0},     # after AS — at destination
    ])
    with patch("apple_notes_brain.sqlite_reader.note_state_by_zid", side_effect=lambda _: next(states)), \
         patch("apple_notes_brain.sqlite_reader.to_uri", side_effect=lambda *a, **k: "x-coredata://X/x"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.applescript.run", return_value=""), \
         patch("apple_notes_brain.applescript.quote", return_value='"X"'):
        ok, err = tools._cascade_note_to_folder("zid-abc", 999, 3, timeout_s=1.0)
    assert ok is True, err


def test_attempt_folder_delete_succeeds_when_row_disappears():
    """First call: alive. After AppleScript runs: gone (and stays gone)."""
    call_count = {"n": 0}
    def state_seq(_zid):
        call_count["n"] += 1
        return {"pk": 999, "marked": 0} if call_count["n"] == 1 else None

    with patch("apple_notes_brain.sqlite_reader.folder_state_by_zid", side_effect=state_seq), \
         patch("apple_notes_brain.sqlite_reader.to_uri", return_value="x-coredata://X/ICFolder/p999"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.applescript.run", return_value=""), \
         patch("apple_notes_brain.applescript.quote", return_value='"X"'):
        ok, err = tools._attempt_folder_delete("zid-abc", 999, "TestFolder", timeout_s=1.0)
    assert ok is True, err


def test_delete_note_refuses_trash_note_no_permanent_delete_path():
    """SAFETY GUARANTEE: this MCP cannot permanently delete notes.

    delete_note on a note already in Recently Deleted MUST refuse with a clear
    message, regardless of any parameters. The signature has no permanently_delete
    flag — there is no escape hatch. The only path to permanent deletion is the
    user manually emptying Recently Deleted in Notes.app."""
    import inspect

    # The signature must NOT include permanently_delete (param removed for safety)
    sig = inspect.signature(tools.delete_note)
    assert "permanently_delete" not in sig.parameters, (
        "delete_note must not have permanently_delete; safety guarantee violated"
    )

    # Calling delete_note on a trash note must refuse
    trash_pk = 2
    with patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100)), \
         patch("apple_notes_brain.sqlite_reader.note_meta", return_value={"locked": False, "folder_pk": trash_pk}), \
         patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={trash_pk}):
        with pytest.raises(ValueError) as exc_info:
            tools.delete_note("p100")
        msg = str(exc_info.value)
        assert "already in Recently Deleted" in msg
        assert "does not permanently delete" in msg.lower() or "manually empty" in msg.lower()


def test_delete_note_signature_does_not_accept_kwargs_for_permanent_delete():
    """Defense in depth: even if someone tries to pass permanently_delete=True,
    Python's signature rejects it. The _safe_tool wrapper converts the underlying
    TypeError into a clean ValueError so the model sees an actionable error
    rather than a raw stack-trace fragment — but the rejection still happens."""
    with pytest.raises(ValueError, match="unexpected keyword argument 'permanently_delete'"):
        tools.delete_note("p100", permanently_delete=True)  # type: ignore[call-arg]


def test_cascade_to_trash_uses_move_not_delete_to_avoid_permanent_destruction():
    """🔒 SAFETY-CRITICAL: cascade must use MOVE_NOTE pointing at trash, NOT DELETE_NOTE.

    AppleScript `delete note` on a note ALREADY in trash permanently destroys it.
    If our cascade retries (network glitch, sync lag), the second call would be
    on a now-trashed note and would permanently destroy it. v10 audit confirmed
    this happened in production. The fix is to use `move note to <trash folder>`
    which is idempotent (live → moves; in-trash → no-op; never permanent).

    This test asserts the AppleScript template used is MOVE_NOTE, never DELETE_NOTE."""
    from apple_notes_brain import scripts as _scripts
    captured_scripts: list[str] = []

    def capture_aps(script):
        captured_scripts.append(script)
        return ""

    states = iter([
        {"pk": 100, "folder_pk": 999, "marked": 0},  # initial
        {"pk": 100, "folder_pk": 2, "marked": 0},     # after AS — moved to trash pk=2
    ])
    with patch("apple_notes_brain.sqlite_reader.note_state_by_zid", side_effect=lambda _: next(states)), \
         patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
         patch("apple_notes_brain.sqlite_reader.to_uri", side_effect=lambda pk, *a, **k: f"x-coredata://X/{a[1] if len(a)>1 else 'ICNote'}/p{pk}"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.applescript.run", side_effect=capture_aps), \
         patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'):
        ok, err = tools._cascade_note_to_trash("zid-abc", 999, timeout_s=1.0)
    assert ok is True, err

    # The AppleScript MUST contain "move" semantics, NOT "delete n"
    assert len(captured_scripts) >= 1
    s = captured_scripts[0].lower()
    assert "move" in s, f"cascade did not use MOVE_NOTE template: {captured_scripts[0]!r}"
    assert "delete n\n" not in s and "delete n " not in s, (
        f"cascade used DELETE_NOTE template — UNSAFE on retry: {captured_scripts[0]!r}"
    )


def test_cascade_to_trash_refuses_when_note_already_in_trash():
    """Defense in depth: if a caller somehow asks us to trash a note already in trash,
    return success without issuing any AppleScript (prevents the destructive retry path)."""
    aps_calls = {"n": 0}
    def count_aps(script):
        aps_calls["n"] += 1
        return ""

    with patch("apple_notes_brain.sqlite_reader.note_state_by_zid",
               return_value={"pk": 100, "folder_pk": 2, "marked": 0}), \
         patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value={2}), \
         patch("apple_notes_brain.applescript.run", side_effect=count_aps), \
         patch("apple_notes_brain.applescript.quote", side_effect=lambda s: f'"{s}"'):
        ok, err = tools._cascade_note_to_trash("zid-abc", 2, timeout_s=1.0)
    assert ok is True
    assert aps_calls["n"] == 0, "cascade issued AppleScript on already-trashed note (UNSAFE)"


def test_translate_apple_error_handles_icnote_invalid_index():
    """v10 audit found delete_note leaks raw osascript on Invalid index for notes.
    The translator must produce a clean ValueError instead of the raw stacktrace."""
    from apple_notes_brain import applescript as _aps
    exc = _aps.AppleScriptError(
        "osascript failed (exit 1): Notes got an error: Can't get note 1 whose id "
        "= \"x-coredata://X/ICNote/p100\". Invalid index. (-1719)"
    )
    with pytest.raises(ValueError) as exc_info:
        tools._translate_apple_error(exc, note_id="p100")
    msg = str(exc_info.value)
    assert "unreachable" in msg.lower() or "bridge" in msg.lower()
    assert "restart" in msg.lower() or "quit" in msg.lower()


def test_aps_run_with_recovery_retries_after_bridge_restart():
    """Two-tier recovery: Tier 1 (MOC backpressure retry, 0.7s wait) handles most
    Invalid index errors WITHOUT restarting Notes.app. If Tier 1 still fails,
    Tier 2 escalates to a full bridge restart via recover_bridge()."""
    from apple_notes_brain import applescript as _aps
    calls = []

    def flaky_run(script):
        calls.append(script)
        # First two calls fail (initial + Tier 1 retry); third (after restart) succeeds
        if len(calls) <= 2:
            raise _aps.AppleScriptError(
                "Notes got an error: Can't get note ICNote. Invalid index. (-1719)"
            )
        return "OK"

    with patch("apple_notes_brain.applescript.run", side_effect=flaky_run), \
         patch("apple_notes_brain.cache.recover_bridge", return_value=True) as mock_recover, \
         patch("apple_notes_brain.tools.time.sleep"):  # skip the 0.7s Tier 1 wait
        out = tools._aps_run_with_recovery("test")
    assert out == "OK"
    assert len(calls) == 3, f"must do Tier 1 retry then Tier 2 retry (got {len(calls)})"
    assert mock_recover.call_count == 1


def test_aps_run_with_recovery_tier1_succeeds_without_bridge_restart():
    """Tier 1: when MOC backpressure clears within 0.7s, the retry succeeds and
    we never invoke the expensive Notes.app restart."""
    from apple_notes_brain import applescript as _aps
    calls = []

    def flaky_run(script):
        calls.append(script)
        if len(calls) == 1:
            raise _aps.AppleScriptError(
                "Notes got an error: Can't get note ICNote. Invalid index. (-1719)"
            )
        return "OK"

    with patch("apple_notes_brain.applescript.run", side_effect=flaky_run), \
         patch("apple_notes_brain.cache.recover_bridge") as mock_recover, \
         patch("apple_notes_brain.tools.time.sleep"):
        out = tools._aps_run_with_recovery("test")
    assert out == "OK"
    assert len(calls) == 2, "Tier 1 retry should succeed on second call"
    assert mock_recover.call_count == 0, "no bridge restart needed"


def test_aps_run_with_recovery_does_not_retry_on_unrelated_error():
    """Non-bridge errors must propagate immediately; no Notes.app restart."""
    from apple_notes_brain import applescript as _aps

    def boom(script):
        raise _aps.AppleScriptError("password protected")

    with patch("apple_notes_brain.applescript.run", side_effect=boom), \
         patch("apple_notes_brain.cache.recover_bridge") as mock_recover:
        with pytest.raises(_aps.AppleScriptError):
            tools._aps_run_with_recovery("test")
    assert mock_recover.call_count == 0


def test_aps_run_with_recovery_does_not_retry_when_recovery_fails():
    """Rate-limited cooldown returns False from recover_bridge; we still do
    Tier 1 retry, but skip Tier 2 restart and propagate the error."""
    from apple_notes_brain import applescript as _aps
    calls = []

    def always_fail(script):
        calls.append(script)
        raise _aps.AppleScriptError(
            "Can't get folder ICFolder. Invalid index. (-1719)"
        )

    with patch("apple_notes_brain.applescript.run", side_effect=always_fail), \
         patch("apple_notes_brain.cache.recover_bridge", return_value=False), \
         patch("apple_notes_brain.tools.time.sleep"):
        with pytest.raises(_aps.AppleScriptError):
            tools._aps_run_with_recovery("test")
    # Tier 1 retry runs (2 calls), then recover_bridge returns False, so we propagate.
    assert len(calls) == 2, f"Tier 1 retry then bail (got {len(calls)})"


def test_attempt_folder_delete_detects_silent_lie():
    """If both id-based and predicate AS calls leave the row alive, raise."""
    with patch("apple_notes_brain.sqlite_reader.folder_state_by_zid",
               return_value={"pk": 999, "marked": 0}), \
         patch("apple_notes_brain.sqlite_reader.to_uri", return_value="x-coredata://X/ICFolder/p999"), \
         patch("apple_notes_brain.sqlite_reader.store_uuid", return_value="X"), \
         patch("apple_notes_brain.sqlite_reader.folder_has_delete_change", return_value=False), \
         patch("apple_notes_brain.applescript.run", return_value="DELETED"), \
         patch("apple_notes_brain.applescript.quote", return_value='"X"'), \
         patch("apple_notes_brain.cache.refresh", return_value={"ok": True, "ms": 1}):
        ok, err = tools._attempt_folder_delete("zid-abc", 999, "TestFolder", timeout_s=0.5)
    assert ok is False
    # Error message wording can change; lock in only the actionable signals
    err_lower = (err or "").lower()
    assert "testfolder" in err_lower
    assert "could not be deleted" in err_lower or "could not delete" in err_lower
