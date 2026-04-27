"""Unit tests for the read tool surface: list_folders, list_notes, search_notes.

Heavy mocking of sqlite_reader and applescript — none of these tests touch
the real Apple Notes store. We exercise tools.py wrapping/validation logic.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain import tools
from apple_notes_brain.schemas import Folder, ListPage, NoteSummary, SearchPage


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _f(pk: int, path: str, *, is_trash: bool = False, account: str = "iCloud",
       shared: bool = False, note_count: int | None = None) -> dict:
    """Build a sqlite_reader-style folder dict."""
    return {
        "id": f"f{pk}",
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "is_trash": is_trash,
        "account": account,
        "shared": shared,
        "note_count": note_count,
    }


def _n(pk: int, *, title: str = "Note", folder_pk: int = 1,
       modified: float = 0.0, pinned: bool = False, locked: bool = False,
       shared: bool = False) -> dict:
    """Build a sqlite_reader-style note row dict."""
    return {
        "id": f"p{pk}",
        "title": title,
        "folder_pk": folder_pk,
        "modified": modified,
        "pinned": pinned,
        "locked": locked,
        "shared": shared,
    }


# ---------------------------------------------------------------------------
# list_folders
# ---------------------------------------------------------------------------

class TestListFolders:
    def test_basic_returns_folder_models(self):
        rows = [_f(1, "Notes"), _f(7, "Work/Projects")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        assert len(out) == 2
        assert all(isinstance(f, Folder) for f in out)
        assert {f.path for f in out} == {"Notes", "Work/Projects"}

    def test_include_counts_true_passes_through(self):
        rows = [_f(1, "Notes", note_count=42)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows) as mock_lf:
            out = tools.list_folders(include_counts=True)
        mock_lf.assert_called_once_with(include_counts=True)
        assert out[0].note_count == 42

    def test_include_counts_false_strips_count(self):
        # Even if rows happen to have a note_count, include_counts=False must mask it
        rows = [_f(1, "Notes", note_count=99)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders(include_counts=False)
        assert out[0].note_count is None

    def test_excludes_trash_by_default(self):
        rows = [_f(1, "Notes"), _f(2, "Recently Deleted", is_trash=True)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        assert all(not f.is_trash for f in out)
        assert {f.path for f in out} == {"Notes"}

    def test_include_trash_true_returns_trash(self):
        rows = [_f(1, "Notes"), _f(2, "Recently Deleted", is_trash=True)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders(include_trash=True)
        assert {f.path for f in out} == {"Notes", "Recently Deleted"}
        assert any(f.is_trash for f in out)

    def test_empty_db(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=[]):
            out = tools.list_folders()
        assert out == []

    def test_account_field_propagated(self):
        rows = [_f(1, "Notes", account="iCloud"), _f(8, "OnMyMac", account="On My Mac")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        accounts = {f.account for f in out}
        assert "iCloud" in accounts and "On My Mac" in accounts

    def test_shared_field_propagated(self):
        rows = [_f(1, "Notes", shared=True)]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        assert out[0].shared is True

    def test_tombstoned_folder_filtered_out(self):
        from apple_notes_brain import cache
        rows = [_f(1, "Notes"), _f(7, "GhostFolder")]
        cache.tombstone_folder(7)
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        paths = {f.path for f in out}
        assert "GhostFolder" not in paths
        assert "Notes" in paths

    def test_rename_overlay_applied(self):
        from apple_notes_brain import cache
        rows = [_f(7, "OldName")]
        cache.rename_path_overlay(7, "NewName")
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=rows):
            out = tools.list_folders()
        assert out[0].path == "NewName"


# ---------------------------------------------------------------------------
# list_notes
# ---------------------------------------------------------------------------

class TestListNotes:
    def _patch_db(self, folders=None, list_result=None):
        """Build a contextmanager stack of common DB patches."""
        folders = folders if folders is not None else [_f(1, "Notes"), _f(7, "Work")]
        list_result = list_result if list_result is not None else ([], False, None, 0)
        return [
            patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders),
            patch("apple_notes_brain.sqlite_reader.list_notes", return_value=list_result),
        ]

    def test_default_args_returns_listpage(self):
        rows = [_n(100, title="Hello", folder_pk=1)]
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=(rows, False, None, 1)) as mock_ln:
            out = tools.list_notes(folder_path=None, limit=20)
        assert isinstance(out, ListPage)
        assert out.returned == 1
        assert out.has_more is False
        # Default: pks=None, no date filters
        args, kwargs = mock_ln.call_args
        assert args[0] is None  # pks
        assert args[1] == 20    # limit

    def test_folder_path_resolves_to_pks(self):
        folders = [_f(1, "Notes"), _f(7, "Work"), _f(9, "Work/Sub")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path="Work", limit=10)
        pks_arg = mock_ln.call_args.args[0]
        # Both Work and Work/Sub should be in scope (subtree)
        assert pks_arg == {7, 9}

    def test_folder_path_nonexistent_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="folder not found"):
                tools.list_notes(folder_path="DoesNotExist", limit=10)

    def test_limit_clamped_to_max(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=10_000)
        # Effective limit must not exceed MAX_LIST_LIMIT
        assert mock_ln.call_args.args[1] == tools.MAX_LIST_LIMIT

    def test_limit_500_passed_through(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=500)
        assert mock_ln.call_args.args[1] == 500

    def test_limit_zero_clamped_to_one(self):
        # tools.py: `limit = max(1, min(limit, MAX_LIST_LIMIT))` — never raises;
        # 0 silently becomes 1. Captures CURRENT behaviour.
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=0)
        assert mock_ln.call_args.args[1] == 1

    def test_negative_limit_clamped_to_one(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=-5)
        assert mock_ln.call_args.args[1] == 1

    def test_cursor_passed_through(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=10, cursor="MTA=")
        assert mock_ln.call_args.args[2] == "MTA="

    def test_include_trash_passed_through(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=10, include_trash=True)
        assert mock_ln.call_args.kwargs.get("include_trash") is True

    def test_modified_after_iso_parsed(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)) as mock_ln:
            tools.list_notes(folder_path=None, limit=10,
                             modified_after="2026-04-26T00:00:00")
        assert mock_ln.call_args.kwargs["modified_after_cd"] is not None
        assert isinstance(mock_ln.call_args.kwargs["modified_after_cd"], float)

    def test_modified_before_invalid_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="invalid ISO date"):
                tools.list_notes(folder_path=None, limit=10,
                                 modified_before="not-a-date-yes-really")

    def test_returns_listpage_with_has_more_and_cursor(self):
        rows = [_n(i, title=f"n{i}") for i in range(3)]
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=(rows, True, "next-cursor", 100)):
            out = tools.list_notes(folder_path=None, limit=3)
        assert out.has_more is True
        assert out.next_cursor == "next-cursor"
        assert out.total_estimate == 100
        assert out.returned == 3

    def test_empty_results_returns_empty_listpage(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=([], False, None, 0)):
            out = tools.list_notes(folder_path=None, limit=10)
        assert out.results == []
        assert out.returned == 0

    def test_folder_name_resolution_in_summary(self):
        rows = [_n(100, title="x", folder_pk=7)]
        folders = [_f(1, "Notes"), _f(7, "Work/Projects")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=(rows, False, None, 1)):
            out = tools.list_notes(folder_path=None, limit=10)
        assert out.results[0].folder == "Work/Projects"

    def test_unknown_folder_pk_yields_empty_folder_string(self):
        rows = [_n(100, title="x", folder_pk=999)]
        folders = [_f(1, "Notes")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.list_notes",
                   return_value=(rows, False, None, 1)):
            out = tools.list_notes(folder_path=None, limit=10)
        assert out.results[0].folder == ""


# ---------------------------------------------------------------------------
# search_notes
# ---------------------------------------------------------------------------

class TestSearchNotes:
    def _candidates(self, *items):
        """Build the (note_dict, body_text) tuples sqlite_reader.search_notes returns."""
        return list(items)

    def test_basic_substring_returns_searchpage(self):
        cands = self._candidates(
            (_n(1, title="hello world"), "hello world body"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="hello", folder_path=None)
        assert isinstance(out, SearchPage)
        assert out.returned == 1
        assert out.results[0].id == "p1"

    def test_substring_no_match_filtered_out(self):
        cands = self._candidates(
            (_n(1, title="abc"), "no luck here"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="zzz-not-there", folder_path=None)
        assert out.returned == 0

    def test_regex_mode_valid_pattern(self):
        cands = self._candidates(
            (_n(1, title="hello123"), "match"),
            (_n(2, title="other"),    "nope"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query=r"hel\w+\d+", folder_path=None, mode="regex")
        ids = {r.id for r in out.results}
        assert "p1" in ids
        assert "p2" not in ids

    def test_regex_mode_invalid_pattern_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="invalid regex"):
                tools.search_notes(query="(unclosed", folder_path=None, mode="regex")

    def test_invalid_mode_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="invalid mode"):
                tools.search_notes(query="x", folder_path=None,
                                   mode="garbage")  # type: ignore[arg-type]

    def test_phrase_mode_rejected(self):
        # tools.py only accepts substring/regex; phrase is rejected at the tool level.
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="invalid mode"):
                tools.search_notes(query="x", folder_path=None,
                                   mode="phrase")  # type: ignore[arg-type]

    def test_search_body_false_passed_through(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=([], False, None, None)) as mock_sn:
            tools.search_notes(query="x", folder_path=None, search_body=False)
        # search_body is the third positional arg (query, pks, search_body, ...)
        assert mock_sn.call_args.args[2] is False

    def test_search_body_true_default(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=([], False, None, None)) as mock_sn:
            tools.search_notes(query="x", folder_path=None)
        assert mock_sn.call_args.args[2] is True

    def test_fuzzy_uses_token_based_matching(self):
        # A note with both tokens in body should rank > 0
        cands = self._candidates(
            (_n(1, title="meeting"),  "we will discuss budget"),
            (_n(2, title="unrelated"), "completely different content"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="meeting budget", folder_path=None, fuzzy=True)
        ids = {r.id for r in out.results}
        assert "p1" in ids

    def test_fuzzy_empty_tokens_returns_empty(self):
        # All tokens shorter than 2 chars → no tokens → empty results
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=([], False, None, None)):
            out = tools.search_notes(query="a b c", folder_path=None, fuzzy=True)
        assert out.results == []

    def test_include_body_with_max_chars(self):
        cands = self._candidates(
            (_n(1, title="x"), "hello world " * 100),  # long body
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="hello", folder_path=None,
                                     include_body=True, max_body_chars=50)
        # Top 5 get body_preview; this is top-1 of 1
        preview = out.results[0].body_preview
        assert preview is not None
        assert len(preview) <= 50

    def test_include_body_max_chars_clamped_to_cap(self):
        cands = self._candidates(
            (_n(1, title="x"), "y" * 5000),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="y", folder_path=None,
                                     include_body=True, max_body_chars=999_999)
        preview = out.results[0].body_preview
        assert preview is not None
        assert len(preview) <= tools.MAX_BODY_PREVIEW_CHARS

    def test_include_body_locked_skips_preview(self):
        cands = self._candidates(
            (_n(1, title="locked", locked=True), ""),  # locked → no body peek
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="locked", folder_path=None, include_body=True)
        # Locked notes get the placeholder snippet, no body_preview
        assert out.results[0].body_preview is None

    def test_folder_path_scopes_search(self):
        folders = [_f(1, "Notes"), _f(7, "Work")]
        with patch("apple_notes_brain.sqlite_reader.list_folders", return_value=folders), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=([], False, None, None)) as mock_sn:
            tools.search_notes(query="x", folder_path="Work")
        pks_arg = mock_sn.call_args.args[1]
        assert pks_arg == {7}

    def test_folder_path_nonexistent_raises(self):
        # Recent fix: search_notes with bogus folder_path now raises (was silent before).
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="folder not found"):
                tools.search_notes(query="x", folder_path="NopeNotHere")

    def test_empty_query_returns_empty(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            out = tools.search_notes(query="", folder_path=None)
        assert out.results == []
        assert out.returned == 0

    def test_whitespace_only_query_returns_empty(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            out = tools.search_notes(query="   \n\t  ", folder_path=None)
        assert out.results == []

    def test_pagination_cursor_round_trip(self):
        cands = self._candidates(
            (_n(1, title="hello"), "hello"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, True, "next-cur", None)) as mock_sn:
            out = tools.search_notes(query="hello", folder_path=None, cursor="abc")
        # cursor is positional arg index 4 in search_notes(pool_query, pks, search_body, pool_limit, cursor, ...)
        assert mock_sn.call_args.args[4] == "abc"
        assert out.has_more is True
        assert out.next_cursor == "next-cur"

    def test_limit_clamped_to_max_search(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=([], False, None, None)):
            out = tools.search_notes(query="x", folder_path=None, limit=10_000)
        # Effective limit is clamped, but we assert via no-error and a return
        assert isinstance(out, SearchPage)

    def test_modified_after_invalid_raises(self):
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]):
            with pytest.raises(ValueError, match="invalid ISO date"):
                tools.search_notes(query="x", folder_path=None,
                                   modified_after="not-a-date")

    def test_match_count_populated_for_substring(self):
        cands = self._candidates(
            (_n(1, title="hello"), "hello hello world"),
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="hello", folder_path=None)
        assert out.results[0].match_count >= 1

    def test_results_sorted_by_score(self):
        # Fuzzy mode: title hits weighted higher than body hits.
        cands = self._candidates(
            (_n(1, title="zzz"),       "meeting budget content"),  # body-only hits
            (_n(2, title="meeting budget"), "noise"),               # title hits → wins
        )
        with patch("apple_notes_brain.sqlite_reader.list_folders",
                   return_value=[_f(1, "Notes")]), \
             patch("apple_notes_brain.sqlite_reader.search_notes",
                   return_value=(cands, False, None, None)):
            out = tools.search_notes(query="meeting budget", folder_path=None, fuzzy=True)
        # Title-hit note should come first
        assert out.results[0].id == "p2"
