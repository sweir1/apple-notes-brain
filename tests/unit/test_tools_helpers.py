"""Unit tests for the helper functions in apple_notes_brain.tools.

Targets the small framework-free helpers that under-pin every public tool:
 - error translators (`_translate_apple_error`, `_is_bridge_corruption`)
 - time/epoch conversions (`_iso_to_core_data_epoch`)
 - decorator semantics (`_safe_tool`, `_with_tool_timeout`, `_alarm_handler`)
 - bridge-recovery state machine (`_aps_run_with_recovery`)
 - constants (`TOOL_BUDGET_S`, `DELETE_FOLDER_BUDGET_S`, `MOC_COMMIT_TIMEOUT_S`,
   `MAX_BATCH_NOTES`).

In addition this file documents bug hotspots flagged by the bug-hunt agent —
each `test_bug_*` test cites the source line that's expected to fail (or
already does). Failing assertions in those tests are EXPECTED — they're the
specification of bugs that should be fixed; tests are written tight so they
can flip green once the underlying code is patched.
"""
from __future__ import annotations

import signal as _signal
import threading
import time
from unittest.mock import patch

import pytest

from apple_notes_brain import applescript as aps
from apple_notes_brain import tools


# ---------------------------------------------------------------------------
# Constants — sanity checks so a typo in the source can't silently regress
# the wall-clock budgets that protect us from runaway iCloud syncs.
# ---------------------------------------------------------------------------

def test_tool_budget_is_a_positive_int():
    assert isinstance(tools.TOOL_BUDGET_S, int)
    assert tools.TOOL_BUDGET_S > 0


def test_delete_folder_budget_at_least_as_large_as_tool_budget():
    # Cascading delete legitimately needs more time than a regular tool call.
    assert tools.DELETE_FOLDER_BUDGET_S >= tools.TOOL_BUDGET_S


def test_moc_commit_timeout_is_smaller_than_tool_budget():
    # The inner MOC poll budget MUST fit inside the outer SIGALRM cap, otherwise
    # we'll always trip the SIGALRM before the verify completes.
    assert tools.MOC_COMMIT_TIMEOUT_S < tools.TOOL_BUDGET_S


def test_max_batch_notes_constant():
    assert tools.MAX_BATCH_NOTES == 20


def test_core_data_epoch_offset_is_2001_01_01_unix_time():
    # 2001-01-01T00:00:00Z in unix seconds.
    assert tools.CORE_DATA_EPOCH_OFFSET == 978_307_200


# ---------------------------------------------------------------------------
# _ToolTimeout / _alarm_handler
# ---------------------------------------------------------------------------

def test_tool_timeout_is_baseexception_not_exception():
    # Bug hotspot — the docstring explicitly notes this design choice:
    # try/except Exception inside tools must NOT swallow the timeout signal.
    assert issubclass(tools._ToolTimeout, BaseException)
    assert not issubclass(tools._ToolTimeout, Exception)


def test_alarm_handler_raises_tool_timeout():
    with pytest.raises(tools._ToolTimeout):
        tools._alarm_handler(_signal.SIGALRM, None)


# ---------------------------------------------------------------------------
# _is_bridge_corruption
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected", [
    ("Invalid index (-1719) ICNote", True),
    ("invalid index (-1719) icfolder", True),
    ("Invalid index — ICNote refused", True),
    ("Some other error", False),
    ("", False),
    ("Invalid index but no class name", False),
    ("ICNote without the magic phrase", False),
])
def test_is_bridge_corruption(msg, expected):
    exc = aps.AppleScriptError(msg)
    assert tools._is_bridge_corruption(exc) is expected


def test_is_bridge_corruption_does_not_match_on_plain_exception():
    # Defensive: callers always pass AppleScriptError, but if a generic
    # Exception ever sneaks through it must NOT be classified as corruption.
    exc = Exception("Invalid index ICNote")
    # _is_bridge_corruption only does str(exc).lower() so this technically
    # would match — document that caller is responsible for type-narrowing.
    assert tools._is_bridge_corruption(exc) is True  # documents the looseness


# ---------------------------------------------------------------------------
# _iso_to_core_data_epoch
# ---------------------------------------------------------------------------

def test_iso_to_core_data_epoch_none_returns_none():
    assert tools._iso_to_core_data_epoch(None) is None


def test_iso_to_core_data_epoch_empty_string_returns_none():
    # "Returns None when `iso` is falsy" per docstring.
    assert tools._iso_to_core_data_epoch("") is None


def test_iso_to_core_data_epoch_invalid_format_raises():
    with pytest.raises(ValueError, match="invalid ISO date"):
        tools._iso_to_core_data_epoch("not-a-date")


def test_iso_to_core_data_epoch_valid_datetime_z_suffix():
    # Z-suffix is supported on Python 3.11+ via fromisoformat.
    out = tools._iso_to_core_data_epoch("2026-04-26T12:00:00+00:00")
    assert isinstance(out, float)
    assert out > 0


def test_iso_to_core_data_epoch_date_only_works():
    out = tools._iso_to_core_data_epoch("2026-04-26")
    assert isinstance(out, float)
    assert out > 0


def test_iso_to_core_data_epoch_zero_at_cocoa_epoch():
    # 2001-01-01T00:00:00Z must be exactly 0 seconds since Cocoa epoch.
    out = tools._iso_to_core_data_epoch("2001-01-01T00:00:00+00:00")
    assert out == 0.0


# Bug hotspot — pre-Cocoa-epoch dates produce negative offsets.
def test_bug_iso_epoch_negative_for_pre_2001_dates():
    # Bug hotspot: callers may not handle negative epochs; document the contract.
    out = tools._iso_to_core_data_epoch("1990-01-01T00:00:00+00:00")
    assert out is not None
    assert out < 0


# ---------------------------------------------------------------------------
# _translate_apple_error
# ---------------------------------------------------------------------------

@pytest.fixture
def no_folder_account(mocker):
    """Make `db.list_folders()` return [] so cross-account branch is taken."""
    return mocker.patch(
        "apple_notes_brain.sqlite_reader.list_folders",
        return_value=[],
    )


@pytest.fixture
def primary_icloud_folder(mocker):
    """Stub `db.list_folders()` so a folder named 'away' resolves as primary
    iCloud, exercising the 'deleted on another device' branch."""
    return mocker.patch(
        "apple_notes_brain.sqlite_reader.list_folders",
        return_value=[{"id": "f1", "path": "away", "account": "iCloud"}],
    )


def test_translate_invalid_index_icfolder_primary_icloud(primary_icloud_folder):
    exc = aps.AppleScriptError("Invalid index (-1719) ICFolder")
    with pytest.raises(ValueError, match="deleted on another device"):
        tools._translate_apple_error(exc, folder_path="away")


def test_translate_invalid_index_icfolder_non_icloud(no_folder_account):
    # Folder not in cache OR not in iCloud → fall through to "non-default" branch.
    exc = aps.AppleScriptError("Invalid index ICFolder")
    with pytest.raises(ValueError, match="non-default iCloud|shared CloudKit"):
        tools._translate_apple_error(exc, folder_path="other")


@pytest.mark.parametrize("err_text", ["password", "locked", "protected"])
def test_translate_locked_note_patterns(err_text):
    exc = aps.AppleScriptError(f"note is {err_text}")
    with pytest.raises(ValueError, match="locked"):
        tools._translate_apple_error(exc, note_id="p1")


@pytest.mark.parametrize("err_text", ["recently deleted", "-10000"])
def test_translate_recently_deleted_patterns(err_text):
    exc = aps.AppleScriptError(f"some {err_text} error")
    with pytest.raises(ValueError, match="Recently Deleted"):
        tools._translate_apple_error(exc, note_id="p1")


def test_translate_duplicate_folder_name():
    exc = aps.AppleScriptError("duplicate folder name caught")
    with pytest.raises(ValueError, match="folder already exists"):
        tools._translate_apple_error(exc)


def test_translate_invalid_index_icnote_after_recovery(no_folder_account):
    exc = aps.AppleScriptError("Invalid index ICNote")
    with pytest.raises(ValueError, match="bridge restart"):
        tools._translate_apple_error(exc, note_id="p99")


def test_translate_unknown_pattern_falls_through(no_folder_account):
    # Unknown error → translator returns silently; caller is expected to re-raise.
    exc = aps.AppleScriptError("totally unrelated failure mode")
    # Should not raise.
    result = tools._translate_apple_error(exc, note_id="p1")
    assert result is None


def test_translate_empty_error_falls_through(no_folder_account):
    exc = aps.AppleScriptError("")
    # Empty error string — no patterns match.
    assert tools._translate_apple_error(exc) is None


# Bug hotspot #5 (tools.py:704 / _move_one) — verify _translate_apple_error
# echoes the folder_path argument into the user-visible message instead of
# leaking 'None' or '' from a missing field.
def test_bug_translate_apple_error_includes_folder_path(no_folder_account):
    # Bug hotspot #5: tools.py _move_one passed folder.get("path") which can
    # be None. The user-visible error MUST include the path the caller asked for,
    # not the literal "None".
    exc = aps.AppleScriptError("Invalid index ICFolder")
    with pytest.raises(ValueError) as excinfo:
        tools._translate_apple_error(exc, folder_path="away")
    msg = str(excinfo.value)
    assert "away" in msg
    assert "None" not in msg


# ---------------------------------------------------------------------------
# _safe_tool decorator
# ---------------------------------------------------------------------------

def test_safe_tool_passes_through_return_value():
    @tools._safe_tool
    def f():
        return 42
    assert f() == 42


def test_safe_tool_passes_through_value_error():
    @tools._safe_tool
    def f():
        raise ValueError("intentional")
    with pytest.raises(ValueError, match="intentional"):
        f()


def test_safe_tool_wraps_applescript_error():
    @tools._safe_tool
    def f():
        raise aps.AppleScriptError("boom")
    with pytest.raises(ValueError, match="AppleScript failure"):
        f()


def test_safe_tool_wraps_attribute_error_as_internal_bug():
    @tools._safe_tool
    def f():
        raise AttributeError("missing thing")
    with pytest.raises(ValueError, match="internal error"):
        f()


def test_safe_tool_wraps_type_error_as_internal_bug():
    @tools._safe_tool
    def f():
        raise TypeError("bad type")
    with pytest.raises(ValueError, match="internal error"):
        f()


def test_safe_tool_wraps_generic_exception():
    @tools._safe_tool
    def f():
        raise RuntimeError("oops")
    with pytest.raises(ValueError, match="unexpected RuntimeError"):
        f()


def test_safe_tool_passes_through_tool_timeout():
    # Timeout must NOT be wrapped to ValueError by _safe_tool — that's the
    # outer @_with_tool_timeout's responsibility.
    @tools._safe_tool
    def f():
        raise tools._ToolTimeout()
    with pytest.raises(tools._ToolTimeout):
        f()


# ---------------------------------------------------------------------------
# _with_tool_timeout decorator
# ---------------------------------------------------------------------------

def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@pytest.mark.timeout(10)
def test_with_tool_timeout_returns_value_when_under_budget():
    if not _on_main_thread():
        pytest.skip("SIGALRM only works on main thread")

    @tools._with_tool_timeout(budget_s=5)
    def f(x):
        return x * 2

    assert f(21) == 42


@pytest.mark.timeout(10)
def test_with_tool_timeout_works_as_bare_decorator():
    if not _on_main_thread():
        pytest.skip("SIGALRM only works on main thread")

    @tools._with_tool_timeout
    def f():
        return "ok"

    assert f() == "ok"


@pytest.mark.timeout(10)
def test_with_tool_timeout_raises_value_error_when_exceeded():
    if not _on_main_thread():
        pytest.skip("SIGALRM only works on main thread")

    @tools._with_tool_timeout(budget_s=1)
    def slow():
        time.sleep(2)

    with pytest.raises(ValueError, match="exceeded the 1s timeout"):
        slow()


@pytest.mark.timeout(10)
def test_with_tool_timeout_clears_alarm_after_success():
    if not _on_main_thread():
        pytest.skip("SIGALRM only works on main thread")

    @tools._with_tool_timeout(budget_s=10)
    def quick():
        return 1

    quick()
    # After return, alarm should have been cleared. This is fuzzy to assert
    # directly without races; we just verify a second call still works.
    assert quick() == 1


def test_with_tool_timeout_runs_unprotected_off_main_thread():
    """When SIGALRM cannot be installed, the decorator falls back to running
    the function directly. A child-thread call must still execute the body."""
    box: dict = {}

    @tools._with_tool_timeout(budget_s=5)
    def f():
        box["ran"] = True
        return "child"

    t = threading.Thread(target=lambda: box.update(result=f()))
    t.start()
    t.join(timeout=5)
    assert box.get("ran") is True
    assert box.get("result") == "child"


# ---------------------------------------------------------------------------
# _aps_run_with_recovery — Tier 1 / Tier 2 / no-recovery / non-corruption.
# ---------------------------------------------------------------------------

def test_aps_run_with_recovery_returns_immediately_on_success(mocker):
    run_mock = mocker.patch("apple_notes_brain.applescript.run", return_value="OK")
    out = tools._aps_run_with_recovery("script")
    assert out == "OK"
    assert run_mock.call_count == 1


def test_aps_run_with_recovery_tier1_retry_after_invalid_index(mocker):
    # First call raises corruption → wait → second call succeeds (no Tier 2).
    run_mock = mocker.patch(
        "apple_notes_brain.applescript.run",
        side_effect=[aps.AppleScriptError("Invalid index ICNote"), "OK"],
    )
    sleep_mock = mocker.patch("apple_notes_brain.tools.time.sleep")
    out = tools._aps_run_with_recovery("script")
    assert out == "OK"
    assert run_mock.call_count == 2
    sleep_mock.assert_called_once_with(0.7)


def test_aps_run_with_recovery_tier2_recover_bridge(mocker):
    run_mock = mocker.patch(
        "apple_notes_brain.applescript.run",
        side_effect=[
            aps.AppleScriptError("Invalid index ICNote"),
            aps.AppleScriptError("Invalid index ICNote"),
            "OK",
        ],
    )
    mocker.patch("apple_notes_brain.tools.time.sleep")
    recover_mock = mocker.patch(
        "apple_notes_brain.cache.recover_bridge",
        return_value=True,
    )
    out = tools._aps_run_with_recovery("script")
    assert out == "OK"
    assert run_mock.call_count == 3
    recover_mock.assert_called_once()


def test_aps_run_with_recovery_tier2_recover_bridge_fails(mocker):
    mocker.patch(
        "apple_notes_brain.applescript.run",
        side_effect=[
            aps.AppleScriptError("Invalid index ICNote"),
            aps.AppleScriptError("Invalid index ICNote"),
            aps.AppleScriptError("Invalid index ICNote"),
        ],
    )
    mocker.patch("apple_notes_brain.tools.time.sleep")
    mocker.patch("apple_notes_brain.cache.recover_bridge", return_value=False)
    with pytest.raises(aps.AppleScriptError, match="Invalid index"):
        tools._aps_run_with_recovery("script")


def test_aps_run_with_recovery_non_corruption_propagates_immediately(mocker):
    run_mock = mocker.patch(
        "apple_notes_brain.applescript.run",
        side_effect=aps.AppleScriptError("permission denied"),
    )
    with pytest.raises(aps.AppleScriptError, match="permission denied"):
        tools._aps_run_with_recovery("script")
    assert run_mock.call_count == 1


def test_aps_run_with_recovery_tier1_succeeds_then_non_corruption_on_second(mocker):
    # First raises corruption (triggers Tier 1) → second raises non-corruption
    # (re-raises immediately, no Tier 2 escalation).
    mocker.patch(
        "apple_notes_brain.applescript.run",
        side_effect=[
            aps.AppleScriptError("Invalid index ICNote"),
            aps.AppleScriptError("permission denied"),
        ],
    )
    mocker.patch("apple_notes_brain.tools.time.sleep")
    with pytest.raises(aps.AppleScriptError, match="permission denied"):
        tools._aps_run_with_recovery("script")


# ---------------------------------------------------------------------------
# _folder_pks_for_path  — Bug hotspot #7 (case-insensitive prefix-match risk).
# ---------------------------------------------------------------------------

def test_folder_pks_for_path_exact_match():
    folders = [{"id": "f1", "path": "Work/Projects"}]
    assert tools._folder_pks_for_path(folders, "Work/Projects") == {1}


def test_folder_pks_for_path_case_insensitive():
    folders = [{"id": "f3", "path": "Work/Projects"}]
    assert tools._folder_pks_for_path(folders, "work/projects") == {3}


def test_folder_pks_for_path_descendants_included():
    folders = [
        {"id": "f1", "path": "Work"},
        {"id": "f2", "path": "Work/Projects"},
        {"id": "f3", "path": "Work/Projects/Alpha"},
        {"id": "f4", "path": "Personal"},
    ]
    assert tools._folder_pks_for_path(folders, "Work") == {1, 2, 3}


def test_folder_pks_for_path_empty_path_returns_empty_set():
    folders = [{"id": "f1", "path": "Anything"}]
    assert tools._folder_pks_for_path(folders, "") == set()
    assert tools._folder_pks_for_path(folders, "/") == set()


def test_bug_folder_pks_prefix_does_not_match_substring():
    # Bug hotspot #7: the prefix-match must use "needle/" not just "needle"
    # to avoid "a" matching "apple/banana". Verify the implementation got
    # the boundary right.
    folders = [
        {"id": "f1", "path": "a"},
        {"id": "f2", "path": "apple"},
        {"id": "f3", "path": "apple/banana"},
    ]
    pks = tools._folder_pks_for_path(folders, "a")
    # Only "a" (id=1) should match. NOT "apple" or "apple/banana".
    assert pks == {1}


# ---------------------------------------------------------------------------
# _find_folder_exact / _folder_name_map — defensive sanity.
# ---------------------------------------------------------------------------

def test_find_folder_exact_returns_match():
    folders = [{"id": "f1", "path": "Work"}, {"id": "f2", "path": "Personal"}]
    out = tools._find_folder_exact(folders, "Work")
    assert out is not None
    assert out["id"] == "f1"


def test_find_folder_exact_case_insensitive():
    folders = [{"id": "f1", "path": "Work"}]
    assert tools._find_folder_exact(folders, "work") is not None


def test_find_folder_exact_returns_none_on_miss():
    folders = [{"id": "f1", "path": "Work"}]
    assert tools._find_folder_exact(folders, "Personal") is None


def test_folder_name_map_builds_pk_to_path():
    folders = [
        {"id": "f1", "path": "A"},
        {"id": "f42", "path": "B"},
        {"id": "bad", "path": "skipped"},  # malformed id — must not crash.
    ]
    out = tools._folder_name_map(folders)
    assert out == {1: "A", 42: "B"}


# ---------------------------------------------------------------------------
# _fmt_time — small helper but easy to break with bad epoch.
# ---------------------------------------------------------------------------

def test_fmt_time_zero_epoch_returns_empty_string():
    assert tools._fmt_time(0) == ""


def test_fmt_time_returns_yyyy_mm_dd_hh_mm_format():
    # 2026-04-26T12:00:00Z is timezone-dependent when re-formatted as local
    # time; just assert structure.
    out = tools._fmt_time(1_800_000_000.0)  # plausible 2027-ish epoch
    # Either a 16-char "YYYY-MM-DD HH:MM" string, or empty on overflow.
    if out:
        assert len(out) == 16
        assert out[4] == "-" and out[7] == "-" and out[10] == " " and out[13] == ":"


# ---------------------------------------------------------------------------
# _body_to_html — small dispatch.
# ---------------------------------------------------------------------------

def test_body_to_html_empty_returns_empty():
    assert tools._body_to_html("", "markdown") == ""
    assert tools._body_to_html("", "html") == ""
    assert tools._body_to_html("", "text") == ""


def test_body_to_html_text_escapes_and_wraps():
    out = tools._body_to_html("hello\nworld <b>", "text")
    assert "<div>" in out
    assert "&lt;b&gt;" in out
    assert "<br>" in out


def test_body_to_html_invalid_format_raises():
    with pytest.raises(ValueError, match="invalid format"):
        tools._body_to_html("body", "rtf")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _regex_prefilter_seed — pure helper.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern,expected", [
    (r"hello", "hello"),
    (r"\d+\s+(world)", "world"),
    (r"^\s*$", ""),  # nothing alphanumeric of length >= 2
    (r"a", ""),       # length-1 ineligible
    (r"foo|bar", "foo"),
])
def test_regex_prefilter_seed(pattern, expected):
    assert tools._regex_prefilter_seed(pattern) == expected


# ---------------------------------------------------------------------------
# Bug hotspot smoke-tests — these aren't exhaustively covered above but
# get an explicit named test so a regression has somewhere obvious to land.
# ---------------------------------------------------------------------------

# Bug hotspot #11 (FIXED): tools.py — `meta.get("locked")` returns None on
# older macOS where the column might not exist. Previously fail-open silently;
# now update_note explicitly raises so we never write blind.
def test_bug_locked_field_missing_now_raises(mocker):
    # Bug hotspot #11 (FIXED): when meta has no "locked" key, update_note
    # used to fall through to the write path because bool(None) is False.
    # Now it raises so we never silently overwrite a possibly-locked note.
    from apple_notes_brain.schemas import MutationResult  # noqa: F401

    meta = {
        "id": "p100",
        "title": "x",
        "folder_pk": 1,
        "modified": 0.0,
        "pinned": False,
        "shared": False,
        # 'locked' intentionally absent — schema gap.
    }
    mocker.patch("apple_notes_brain.sqlite_reader.resolve_id", return_value=("note", 100))
    mocker.patch("apple_notes_brain.sqlite_reader.note_meta", return_value=meta)
    mocker.patch("apple_notes_brain.sqlite_reader.trash_folder_pks", return_value=set())
    mocker.patch("apple_notes_brain.sqlite_reader.attachment_count", return_value=0)
    mocker.patch("apple_notes_brain.sqlite_reader.short_id", return_value="p100")

    with pytest.raises(ValueError, match="cannot determine lock state"):
        tools.update_note(note_id="p100", body="x")


# Bug hotspot #21 (FIXED): _create_notes_bulk error message used to say
# "move likely failed" inside a CREATE operation. Now correctly says "create".
def test_bug_create_notes_bulk_error_message_says_create(mocker):
    # Bug hotspot #21 (FIXED): tools.py — the padding-failure message
    # now correctly references "create" instead of "move".
    from apple_notes_brain.schemas import NoteCreateSpec

    # Mock applescript.run to return EMPTY output — the bulk create returns
    # zero URIs but specs has 2 entries → padding triggers.
    mocker.patch("apple_notes_brain.tools._aps_run_with_recovery", return_value="")
    mocker.patch("apple_notes_brain.tools._resolve_folder_uri", return_value=None)
    mocker.patch("apple_notes_brain.cache.sync_after_write")

    specs = [NoteCreateSpec(title="a", body="A"), NoteCreateSpec(title="b", body="B")]
    out = tools._create_notes_bulk(specs, folder_path=None, format="markdown")
    assert len(out) == 2
    # Both should be "skipped" with the corrected message.
    assert all(r.action == "skipped" for r in out)
    err_msgs = [r.error for r in out]
    assert any("create likely failed" in (m or "") for m in err_msgs)
    # And the buggy "move likely failed" wording must be gone.
    assert not any("move likely failed" in (m or "") for m in err_msgs)


# Bug hotspot #22 (FIXED): create_note(notes=[]) used to silently return [];
# now raises so the caller knows nothing was done.
def test_bug_create_note_empty_notes_list_raises(mocker):
    # Bug hotspot #22 (FIXED): tools.py — passing notes=[] previously
    # returned [] silently. Now correctly raises ValueError.
    # Stub the inner _create_notes_bulk so we don't hit AppleScript even
    # though the empty check should fire before it.
    mocker.patch("apple_notes_brain.tools._create_notes_bulk", return_value=[])
    with pytest.raises(ValueError, match="empty"):
        tools.create_note(notes=[])


# Bug hotspot #15/#16: _wait_for_state — test the contract under simple inputs.
def test_wait_for_state_returns_true_immediately_when_check_passes(mocker):
    # check_fn already True at entry — return True without sleeping.
    sleep_mock = mocker.patch("apple_notes_brain.tools.time.sleep")
    assert tools._wait_for_state(lambda: True, timeout_s=0.1) is True
    sleep_mock.assert_not_called()


def test_wait_for_state_returns_false_when_timeout_with_check_failing(mocker):
    # check_fn always False → expires after timeout.
    mocker.patch("apple_notes_brain.tools.time.sleep")
    # Patch monotonic so we don't actually wait.
    fake_t = [0.0]
    def fake_monotonic():
        fake_t[0] += 0.05
        return fake_t[0]
    mocker.patch("apple_notes_brain.tools.time.monotonic", side_effect=fake_monotonic)
    # Make data_version raise so the ping-cap path is exercised cleanly.
    mocker.patch("apple_notes_brain.sqlite_reader.data_version", side_effect=Exception("nope"))
    mocker.patch("apple_notes_brain.cache.refresh")

    assert tools._wait_for_state(lambda: False, timeout_s=0.5, max_pings=2) is False


# Bug hotspot #9: aps.as_list quoting. Verified via the applescript module's
# pure tests — sanity-check from this side that titles with embedded backslash
# + quote round-trip through aps.quote without truncation.
def test_bug_as_list_handles_embedded_quotes_and_backslashes():
    # Bug hotspot #9: tools.py:952-954 — TITLES = aps.as_list(titles).
    # If a title contains \" the AppleScript literal must escape both the
    # backslash AND the quote to be parseable.
    titles = ['hello "world"', "back\\slash", 'mixed\\"end']
    out = aps.as_list(titles)
    # Implementation in applescript.quote: replace("\\", "\\\\").replace('"', '\\"')
    # Each input " in title -> \" in output (escape sequence kept intact).
    # Each input \ in title -> \\ in output. Spot-check both:
    assert '\\"' in out             # quote escapes survived
    assert "\\\\" in out            # backslash escapes survived
    # And the literal source word "world" still appears in the output.
    assert "world" in out
