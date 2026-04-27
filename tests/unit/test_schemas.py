"""Unit tests for apple_notes_brain.schemas — Pydantic v2 model contracts.

Covers:
- Construction with required-only fields (defaults populate correctly)
- Missing-required-field ValidationErrors
- Type coercion / wrong-type rejection
- JSON and dict round-trips
- Hypothesis property tests for round-trip stability
- Boundary cases (empty strings, long strings, unicode)
- Equality semantics
- Extra-field handling
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from apple_notes_brain.schemas import (
    Folder,
    ListPage,
    MutationResult,
    NoteCreateSpec,
    NoteDetail,
    NoteSummary,
    SearchPage,
)


# ---------------------------------------------------------------------------
# 1. Construction with required fields only — defaults populate correctly
# ---------------------------------------------------------------------------


def test_folder_minimal_construction_defaults():
    f = Folder(id="f1", path="Notes")
    assert f.id == "f1"
    assert f.path == "Notes"
    assert f.note_count is None
    assert f.is_trash is False
    assert f.account is None
    assert f.shared is False


def test_note_summary_minimal_construction_defaults():
    n = NoteSummary(id="p1", title="x", folder="Notes", modified="2026-04-26 12:00")
    assert n.id == "p1"
    assert n.title == "x"
    assert n.folder == "Notes"
    assert n.modified == "2026-04-26 12:00"
    assert n.snippets == []
    assert n.match_count == 0
    assert n.body_preview is None
    assert n.pinned is False
    assert n.locked is False
    assert n.account is None
    assert n.attachments == 0
    assert n.shared is False


def test_note_detail_minimal_construction_defaults():
    n = NoteDetail(
        id="p1",
        title="t",
        folder="Notes",
        modified="2026-04-26 12:00",
        body="hello",
        format="markdown",
    )
    assert n.id == "p1"
    assert n.body == "hello"
    assert n.format == "markdown"
    assert n.pinned is False
    assert n.locked is False
    assert n.account is None
    assert n.attachments == 0
    assert n.shared is False


def test_search_page_minimal_construction():
    page = SearchPage(
        results=[],
        returned=0,
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )
    assert page.results == []
    assert page.returned == 0
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.total_estimate is None


def test_list_page_minimal_construction():
    page = ListPage(
        results=[],
        returned=0,
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )
    assert page.results == []
    assert page.returned == 0
    assert page.has_more is False


def test_note_create_spec_minimal_construction():
    spec = NoteCreateSpec(title="My Note")
    assert spec.title == "My Note"
    assert spec.body == ""


def test_mutation_result_minimal_construction():
    r = MutationResult(id="p1", action="created")
    assert r.id == "p1"
    assert r.action == "created"
    assert r.error is None


# ---------------------------------------------------------------------------
# 2. Missing required fields → ValidationError
# ---------------------------------------------------------------------------


def test_folder_missing_required_raises():
    with pytest.raises(ValidationError) as exc_info:
        Folder()  # type: ignore[call-arg]
    msg = str(exc_info.value)
    assert "id" in msg
    assert "path" in msg


def test_folder_missing_path_raises():
    with pytest.raises(ValidationError) as exc_info:
        Folder(id="f1")  # type: ignore[call-arg]
    assert "path" in str(exc_info.value)


def test_note_summary_missing_required_raises():
    with pytest.raises(ValidationError) as exc_info:
        NoteSummary(id="p1")  # type: ignore[call-arg]
    msg = str(exc_info.value)
    assert "title" in msg
    assert "folder" in msg
    assert "modified" in msg


def test_note_detail_missing_required_raises():
    with pytest.raises(ValidationError) as exc_info:
        NoteDetail(id="p1", title="t", folder="Notes", modified="2026-04-26")  # type: ignore[call-arg]
    msg = str(exc_info.value)
    assert "body" in msg
    assert "format" in msg


def test_search_page_missing_required_raises():
    with pytest.raises(ValidationError):
        SearchPage(results=[])  # type: ignore[call-arg]


def test_list_page_missing_required_raises():
    with pytest.raises(ValidationError):
        ListPage(results=[])  # type: ignore[call-arg]


def test_note_create_spec_missing_required_raises():
    with pytest.raises(ValidationError) as exc_info:
        NoteCreateSpec()  # type: ignore[call-arg]
    assert "title" in str(exc_info.value)


def test_mutation_result_missing_required_raises():
    with pytest.raises(ValidationError) as exc_info:
        MutationResult(id="p1")  # type: ignore[call-arg]
    assert "action" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Wrong types → ValidationError (or coercion, document the actual behaviour)
# ---------------------------------------------------------------------------


def test_folder_id_wrong_type_raises():
    # Pydantic v2 strict mode is off by default; ints to str is rejected
    # in v2 by default (no implicit int->str coercion).
    with pytest.raises(ValidationError):
        Folder(id=123, path="x")  # type: ignore[arg-type]


def test_note_summary_match_count_wrong_type_raises():
    with pytest.raises(ValidationError):
        NoteSummary(
            id="p1",
            title="x",
            folder="Notes",
            modified="m",
            match_count="three",  # type: ignore[arg-type]
        )


def test_folder_is_trash_string_yes_coerces_or_rejects():
    # Pydantic v2 default: "yes"/"no"/"true"/"false" are coerced for bool.
    # "yes" coerces to True. Document this behaviour.
    f = Folder(id="f1", path="x", is_trash="yes")  # type: ignore[arg-type]
    assert f.is_trash is True


def test_folder_is_trash_invalid_string_raises():
    with pytest.raises(ValidationError):
        Folder(id="f1", path="x", is_trash="not-a-bool")  # type: ignore[arg-type]


def test_note_summary_snippets_wrong_type_raises():
    with pytest.raises(ValidationError):
        NoteSummary(
            id="p1",
            title="x",
            folder="Notes",
            modified="m",
            snippets="not a list",  # type: ignore[arg-type]
        )


def test_search_page_results_wrong_element_type_raises():
    with pytest.raises(ValidationError):
        SearchPage(
            results=["not a NoteSummary"],  # type: ignore[list-item]
            returned=1,
            has_more=False,
            next_cursor=None,
            total_estimate=None,
        )


# ---------------------------------------------------------------------------
# 4. JSON round-trip: model_dump_json + model_validate_json
# ---------------------------------------------------------------------------


def test_folder_round_trip_json(sample_folder):
    payload = sample_folder.model_dump_json()
    restored = Folder.model_validate_json(payload)
    assert restored == sample_folder
    # And again, to ensure stability
    assert Folder.model_validate_json(restored.model_dump_json()) == sample_folder


def test_note_summary_round_trip_json_full(sample_note_summary):
    populated = sample_note_summary.model_copy(
        update={
            "snippets": ["a", "b", "…match…"],
            "match_count": 3,
            "body_preview": "Some preview text",
            "pinned": True,
            "locked": True,
            "attachments": 4,
            "shared": True,
        }
    )
    payload = populated.model_dump_json()
    restored = NoteSummary.model_validate_json(payload)
    assert restored == populated


def test_note_detail_round_trip_json_unicode(sample_note_detail):
    body = "Title 中文\n\n😀 emoji line\n\nمرحبا RTL text\n\nNoteBody"
    detail = sample_note_detail.model_copy(update={"body": body})
    payload = detail.model_dump_json()
    restored = NoteDetail.model_validate_json(payload)
    assert restored == detail
    assert restored.body == body


def test_search_page_round_trip_json_empty():
    page = SearchPage(
        results=[],
        returned=0,
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )
    restored = SearchPage.model_validate_json(page.model_dump_json())
    assert restored == page


def test_search_page_round_trip_json_populated(sample_note_summary):
    page = SearchPage(
        results=[sample_note_summary, sample_note_summary],
        returned=2,
        has_more=True,
        next_cursor="cursor-abc",
        total_estimate=42,
    )
    restored = SearchPage.model_validate_json(page.model_dump_json())
    assert restored == page
    assert len(restored.results) == 2


def test_list_page_round_trip_json_with_cursor(sample_note_summary):
    page = ListPage(
        results=[sample_note_summary],
        returned=1,
        has_more=True,
        next_cursor="next-page-token",
        total_estimate=100,
    )
    restored = ListPage.model_validate_json(page.model_dump_json())
    assert restored == page
    assert restored.next_cursor == "next-page-token"


def test_list_page_round_trip_json_no_cursor():
    page = ListPage(
        results=[],
        returned=0,
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )
    restored = ListPage.model_validate_json(page.model_dump_json())
    assert restored == page
    assert restored.next_cursor is None


def test_mutation_result_round_trip_json(sample_mutation_result):
    restored = MutationResult.model_validate_json(sample_mutation_result.model_dump_json())
    assert restored == sample_mutation_result


def test_note_create_spec_round_trip_json():
    spec = NoteCreateSpec(title="Title 中文 😀", body="Body line 1\nLine 2")
    restored = NoteCreateSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


# ---------------------------------------------------------------------------
# 5. Dict round-trip via model_dump() + model_validate()
# ---------------------------------------------------------------------------


def test_folder_round_trip_dict(sample_folder):
    d = sample_folder.model_dump()
    assert isinstance(d, dict)
    assert Folder.model_validate(d) == sample_folder


def test_note_summary_round_trip_dict(sample_note_summary):
    d = sample_note_summary.model_dump()
    assert NoteSummary.model_validate(d) == sample_note_summary


def test_note_detail_round_trip_dict(sample_note_detail):
    d = sample_note_detail.model_dump()
    assert NoteDetail.model_validate(d) == sample_note_detail


def test_search_page_round_trip_dict(sample_note_summary):
    page = SearchPage(
        results=[sample_note_summary],
        returned=1,
        has_more=False,
        next_cursor=None,
        total_estimate=1,
    )
    assert SearchPage.model_validate(page.model_dump()) == page


def test_list_page_round_trip_dict(sample_note_summary):
    page = ListPage(
        results=[sample_note_summary],
        returned=1,
        has_more=False,
        next_cursor=None,
        total_estimate=1,
    )
    assert ListPage.model_validate(page.model_dump()) == page


def test_mutation_result_round_trip_dict(sample_mutation_result):
    d = sample_mutation_result.model_dump()
    assert MutationResult.model_validate(d) == sample_mutation_result


# ---------------------------------------------------------------------------
# 6. Hypothesis property tests
# ---------------------------------------------------------------------------

# Reasonable text strategy — non-control unicode is fine, JSON-safe
_text = st.text(min_size=0, max_size=200)
_nonneg_int = st.integers(min_value=0, max_value=10_000)


@pytest.mark.property
@given(
    id_=_text,
    title=_text,
    folder=_text,
    modified=_text,
    snippets=st.lists(_text, max_size=10),
    match_count=_nonneg_int,
    body_preview=st.one_of(st.none(), _text),
    pinned=st.booleans(),
    locked=st.booleans(),
    account=st.one_of(st.none(), _text),
    attachments=_nonneg_int,
    shared=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_note_summary_dict_round_trip_property(
    id_, title, folder, modified, snippets, match_count,
    body_preview, pinned, locked, account, attachments, shared,
):
    n = NoteSummary(
        id=id_,
        title=title,
        folder=folder,
        modified=modified,
        snippets=snippets,
        match_count=match_count,
        body_preview=body_preview,
        pinned=pinned,
        locked=locked,
        account=account,
        attachments=attachments,
        shared=shared,
    )
    assert NoteSummary.model_validate(n.model_dump()) == n


@pytest.mark.property
@given(body=st.text(min_size=0, max_size=200))
@settings(max_examples=50, deadline=None)
def test_note_detail_json_round_trip_unicode_body_property(body):
    n = NoteDetail(
        id="p1",
        title="t",
        folder="Notes",
        modified="2026-04-26",
        body=body,
        format="markdown",
    )
    restored = NoteDetail.model_validate_json(n.model_dump_json())
    assert restored == n
    assert restored.body == body


@pytest.mark.property
@given(
    n_results=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=20, deadline=None)
def test_search_page_preserves_results_length_property(n_results):
    summary = NoteSummary(
        id="p1", title="t", folder="Notes", modified="m",
    )
    results = [summary] * n_results
    page = SearchPage(
        results=results,
        returned=n_results,
        has_more=False,
        next_cursor=None,
        total_estimate=None,
    )
    assert len(page.results) == n_results
    restored = SearchPage.model_validate(page.model_dump())
    assert len(restored.results) == n_results


@pytest.mark.property
@given(
    id_=_text,
    path=_text,
    note_count=st.one_of(st.none(), _nonneg_int),
    is_trash=st.booleans(),
    account=st.one_of(st.none(), _text),
    shared=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_folder_json_round_trip_property(id_, path, note_count, is_trash, account, shared):
    f = Folder(
        id=id_,
        path=path,
        note_count=note_count,
        is_trash=is_trash,
        account=account,
        shared=shared,
    )
    assert Folder.model_validate_json(f.model_dump_json()) == f


# ---------------------------------------------------------------------------
# 7. Boundary cases
# ---------------------------------------------------------------------------


def test_note_summary_empty_string_fields():
    n = NoteSummary(id="", title="", folder="", modified="")
    assert n.title == ""
    assert NoteSummary.model_validate_json(n.model_dump_json()) == n


def test_note_detail_empty_body():
    n = NoteDetail(
        id="p1", title="", folder="Notes", modified="m", body="", format="markdown",
    )
    assert n.body == ""


def test_note_detail_very_long_body():
    long_body = "x" * 10_000
    n = NoteDetail(
        id="p1", title="t", folder="Notes", modified="m",
        body=long_body, format="markdown",
    )
    assert len(n.body) == 10_000
    assert NoteDetail.model_validate_json(n.model_dump_json()).body == long_body


def test_note_summary_long_snippets():
    snippets = ["x" * 10_000 for _ in range(5)]
    n = NoteSummary(
        id="p1", title="t" * 10_000, folder="Notes", modified="m",
        snippets=snippets,
    )
    assert len(n.title) == 10_000
    assert all(len(s) == 10_000 for s in n.snippets)


@pytest.mark.parametrize(
    "text",
    [
        "中文标题",  # Chinese
        "😀🎉🚀",  # Emoji
        "مرحبا",  # Arabic / RTL
        "Здравствуйте",  # Cyrillic
        "한국어",  # Korean
        "🇺🇸🇬🇧",  # Flag emoji (multi-codepoint)
        "ZWJ‍char",  # Zero-width joiner
    ],
)
def test_unicode_text_in_all_string_fields_round_trips(text):
    n = NoteDetail(
        id=text, title=text, folder=text, modified=text,
        body=text, format=text, account=text,
    )
    restored = NoteDetail.model_validate_json(n.model_dump_json())
    assert restored == n


def test_folder_note_count_zero_distinct_from_none():
    f_none = Folder(id="f1", path="x", note_count=None)
    f_zero = Folder(id="f1", path="x", note_count=0)
    assert f_none != f_zero
    assert f_none.note_count is None
    assert f_zero.note_count == 0


def test_note_summary_body_preview_none_distinct_from_empty():
    n_none = NoteSummary(id="p1", title="t", folder="Notes", modified="m", body_preview=None)
    n_empty = NoteSummary(id="p1", title="t", folder="Notes", modified="m", body_preview="")
    assert n_none != n_empty
    assert n_none.body_preview is None
    assert n_empty.body_preview == ""


@pytest.mark.parametrize("cursor", [None, "abc", "0", "very-long-cursor-" + "x" * 1000])
def test_search_page_cursor_variants(cursor):
    page = SearchPage(
        results=[], returned=0, has_more=cursor is not None,
        next_cursor=cursor, total_estimate=None,
    )
    assert page.next_cursor == cursor
    assert SearchPage.model_validate_json(page.model_dump_json()) == page


@pytest.mark.parametrize("cursor", [None, "abc", "0", "very-long-cursor-" + "x" * 1000])
def test_list_page_cursor_variants(cursor):
    page = ListPage(
        results=[], returned=0, has_more=cursor is not None,
        next_cursor=cursor, total_estimate=None,
    )
    assert page.next_cursor == cursor


def test_mutation_result_success_no_error():
    r = MutationResult(id="p1", action="created", error=None)
    assert r.error is None


def test_mutation_result_failure_with_error():
    r = MutationResult(id="p1", action="skipped", error="Note is locked")
    assert r.error == "Note is locked"


@pytest.mark.parametrize(
    "action",
    ["created", "updated", "renamed", "moved", "deleted", "skipped", "anything-goes"],
)
def test_mutation_result_action_values(action):
    r = MutationResult(id="p1", action=action)
    assert r.action == action
    assert MutationResult.model_validate_json(r.model_dump_json()).action == action


# ---------------------------------------------------------------------------
# 8. Equality and hashing
# ---------------------------------------------------------------------------


def test_folder_equality_same_fields():
    a = Folder(id="f1", path="Notes", note_count=5)
    b = Folder(id="f1", path="Notes", note_count=5)
    assert a == b


def test_folder_inequality_differing_note_count():
    a = Folder(id="f1", path="Notes", note_count=5)
    b = Folder(id="f1", path="Notes", note_count=6)
    assert a != b


def test_folder_inequality_differing_id():
    a = Folder(id="f1", path="Notes")
    b = Folder(id="f2", path="Notes")
    assert a != b


def test_note_summary_equality_same_fields():
    kwargs = dict(id="p1", title="t", folder="Notes", modified="m")
    assert NoteSummary(**kwargs) == NoteSummary(**kwargs)


def test_pydantic_models_not_hashable_by_default():
    # Pydantic v2 models default to __hash__ = None (mutable, so unhashable).
    f = Folder(id="f1", path="x")
    with pytest.raises(TypeError):
        hash(f)
    with pytest.raises(TypeError):
        {f}  # noqa: B018 — exercising unhashability


# ---------------------------------------------------------------------------
# 9. Extra fields / aliases
# ---------------------------------------------------------------------------


def test_folder_extra_field_ignored_by_default():
    # Pydantic v2 default: extra="ignore" — unknown keys silently dropped.
    f = Folder(id="f1", path="x", unknown_field="ignored")  # type: ignore[call-arg]
    assert f.id == "f1"
    assert not hasattr(f, "unknown_field")
    # Dump shouldn't contain the extra field
    assert "unknown_field" not in f.model_dump()


def test_note_summary_extra_field_ignored_by_default():
    n = NoteSummary(  # type: ignore[call-arg]
        id="p1", title="t", folder="Notes", modified="m",
        bogus_extra=42,
    )
    assert "bogus_extra" not in n.model_dump()


def test_mutation_result_extra_field_ignored():
    r = MutationResult(id="p1", action="created", extra="x")  # type: ignore[call-arg]
    assert r.action == "created"
    assert "extra" not in r.model_dump()
