"""Unit tests for the pure-function helpers in apple_notes_brain.applescript.

Covers:
- quote()       — AppleScript string-literal escaping (\\ and " handling)
- as_list()     — AppleScript list-literal formatting from a Python list[str]
- parse_records() — RECORD_SEP / UNIT_SEP splitting of osascript stdout
- AppleScriptError — exception class shape (subclass, message, chaining)
- RECORD_SEP / UNIT_SEP module constants

The run() subprocess wrapper is intentionally NOT covered here — it is
exercised via mocking in tests/test_delete_folder_cascade.py and similar.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apple_notes_brain.applescript import (
    AppleScriptError,
    RECORD_SEP,
    UNIT_SEP,
    as_list,
    parse_records,
    quote,
)


# ---------------------------------------------------------------------------
# quote() — AppleScript string escaping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello", '"hello"'),
        ("", '""'),
        ("a b c", '"a b c"'),
        ("123", '"123"'),
    ],
)
def test_quote_plain_strings(raw: str, expected: str) -> None:
    assert quote(raw) == expected


def test_quote_escapes_double_quote() -> None:
    # `say "hi"` → "say \"hi\""
    assert quote('say "hi"') == '"say \\"hi\\""'


def test_quote_escapes_backslash() -> None:
    # `a\b` → "a\\b"
    assert quote("a\\b") == '"a\\\\b"'


def test_quote_escapes_backslash_then_quote_order_matters() -> None:
    # Input `a\"b` (one backslash + one quote, 3 chars)
    # The implementation MUST escape backslashes first, then quotes — otherwise
    # the backslash injected by quote-escaping would itself be doubled.
    # Expected output: "a\\\"b"  (literal: a, \\, \", b inside outer quotes)
    raw = 'a\\"b'
    out = quote(raw)
    assert out == '"a\\\\\\"b"'
    # Sanity: the output starts and ends with a real double-quote and the
    # inner content faithfully encodes the original.
    assert out.startswith('"') and out.endswith('"')


def test_quote_preserves_newline_as_is() -> None:
    # The implementation does NOT escape newlines — they pass through into
    # the AppleScript literal as a literal LF. AppleScript accepts this in
    # double-quoted strings (it's a multi-line string).
    out = quote("a\nb")
    assert out == '"a\nb"'
    assert "\n" in out


def test_quote_preserves_tab_and_cr() -> None:
    # Tabs / CRs likewise pass through.
    assert quote("a\tb") == '"a\tb"'
    assert quote("a\rb") == '"a\rb"'


@pytest.mark.parametrize(
    "raw",
    [
        "café",                      # latin-1 supplement
        "你好",                       # Chinese
        "🎉🚀",                       # emoji
        "مرحبا",                     # RTL Arabic
        "𝓗𝓮𝓵𝓵𝓸",                # mathematical script (BMP-outside)
        "👨‍👩‍👧‍👦",                 # ZWJ family sequence
    ],
)
def test_quote_unicode_characters(raw: str) -> None:
    out = quote(raw)
    assert out.startswith('"') and out.endswith('"')
    # No escaping happens for unicode — it round-trips byte-for-byte.
    assert out == f'"{raw}"'


def test_quote_very_long_string_does_not_crash() -> None:
    raw = "x" * 10_000
    out = quote(raw)
    assert len(out) == 10_002  # 10k chars + 2 surrounding quotes
    assert out.startswith('"') and out.endswith('"')


def test_quote_round_trip_idempotent_shape() -> None:
    # Quoting a quoted string yields a still-valid AS literal (the inner
    # quotes get escaped). This is a useful invariant for callers that
    # accidentally double-quote.
    once = quote("hi")
    twice = quote(once)
    assert twice.startswith('"') and twice.endswith('"')
    # The inner quotes from `once` must have been escaped in `twice`.
    assert '\\"' in twice


def test_quote_only_backslashes() -> None:
    assert quote("\\\\") == '"\\\\\\\\"'  # 2 backslashes → 4 in output


def test_quote_only_quotes() -> None:
    assert quote('""') == '"\\"\\""'  # 2 quotes → \" \" in output


# ---- hypothesis property test for quote() ---------------------------------

@pytest.mark.property
@given(s=st.text())
@settings(max_examples=200)
def test_quote_property_invariants(s: str) -> None:
    """For any string, quote() output:
      - starts and ends with a double-quote,
      - has every input backslash represented as a doubled backslash,
      - has every input double-quote represented as backslash-quote,
      - and contains no other transformations.
    """
    out = quote(s)
    assert out.startswith('"')
    assert out.endswith('"')
    # Reconstruct the inner payload by stripping the outer quotes.
    inner = out[1:-1]
    # The inner should equal the input with \ → \\ and " → \" applied.
    expected_inner = s.replace("\\", "\\\\").replace('"', '\\"')
    assert inner == expected_inner


# ---------------------------------------------------------------------------
# as_list() — AppleScript list literal
# ---------------------------------------------------------------------------

def test_as_list_empty() -> None:
    assert as_list([]) == "{}"


def test_as_list_single_element() -> None:
    assert as_list(["a"]) == '{"a"}'


def test_as_list_multiple_elements() -> None:
    assert as_list(["a", "b"]) == '{"a", "b"}'
    assert as_list(["a", "b", "c"]) == '{"a", "b", "c"}'


def test_as_list_elements_with_quotes() -> None:
    out = as_list(['say "hi"'])
    assert out == '{"say \\"hi\\""}'


def test_as_list_elements_with_backslashes() -> None:
    out = as_list(["a\\b", "c\\d"])
    assert out == '{"a\\\\b", "c\\\\d"}'


def test_as_list_long_list_does_not_crash() -> None:
    items = [f"item{i}" for i in range(1000)]
    out = as_list(items)
    assert out.startswith("{") and out.endswith("}")
    # 1000 quoted strings → 1000 quote-pairs → 2000 unescaped quote chars.
    assert out.count('"') == 2000


def test_as_list_unicode_elements() -> None:
    out = as_list(["café", "🎉", "你好"])
    assert out == '{"café", "🎉", "你好"}'


def test_as_list_preserves_empty_string_elements() -> None:
    assert as_list(["", "a", ""]) == '{"", "a", ""}'


# ---- hypothesis property test for as_list() -------------------------------

@pytest.mark.property
@given(items=st.lists(st.text(), max_size=20))
@settings(max_examples=200)
def test_as_list_property_invariants(items: list[str]) -> None:
    """For any list of strings, as_list() output:
      - starts with '{' and ends with '}',
      - is exactly '{}' iff the input is empty,
      - otherwise contains exactly len(items) outer (unescaped) quote pairs
        — i.e. 2*len(items) unescaped double-quote characters.
    """
    out = as_list(items)
    assert out.startswith("{")
    assert out.endswith("}")
    if not items:
        assert out == "{}"
    else:
        # Count UNESCAPED double-quote chars by removing escape sequences first.
        # In the output, every literal " from input is encoded as \" — strip those.
        # What remains: outer-quote pairs around each element + literal backslashes.
        sanitized = out.replace('\\\\', '').replace('\\"', '')
        assert sanitized.count('"') == 2 * len(items)


# ---------------------------------------------------------------------------
# parse_records() — RECORD_SEP / UNIT_SEP splitting
# ---------------------------------------------------------------------------

def test_parse_records_single_record_single_unit() -> None:
    assert parse_records("a") == [["a"]]


def test_parse_records_single_record_multiple_units() -> None:
    assert parse_records(f"a{UNIT_SEP}b{UNIT_SEP}c") == [["a", "b", "c"]]


def test_parse_records_multiple_records() -> None:
    raw = f"a{UNIT_SEP}b{RECORD_SEP}c{UNIT_SEP}d"
    assert parse_records(raw) == [["a", "b"], ["c", "d"]]


def test_parse_records_trailing_record_separator() -> None:
    # osascript often emits a trailing RS — must NOT produce an empty record.
    raw = f"a{UNIT_SEP}b{RECORD_SEP}c{UNIT_SEP}d{RECORD_SEP}"
    assert parse_records(raw) == [["a", "b"], ["c", "d"]]


def test_parse_records_leading_newline_is_stripped() -> None:
    # osascript sometimes prefixes output with a stray newline.
    raw = f"\na{UNIT_SEP}b"
    assert parse_records(raw) == [["a", "b"]]


def test_parse_records_leading_newline_in_each_record() -> None:
    raw = f"\na{UNIT_SEP}b{RECORD_SEP}\nc{UNIT_SEP}d"
    assert parse_records(raw) == [["a", "b"], ["c", "d"]]


def test_parse_records_empty_string() -> None:
    assert parse_records("") == []


def test_parse_records_only_separators() -> None:
    # Only RS chars — every "record" is empty/whitespace and should be skipped.
    assert parse_records(f"{RECORD_SEP}{RECORD_SEP}{RECORD_SEP}") == []


def test_parse_records_only_newlines() -> None:
    # A bare newline is whitespace-only and yields no records.
    assert parse_records("\n\n\n") == []


def test_parse_records_embedded_newlines_within_field() -> None:
    # Newlines inside a field (not at the very start of a record) survive.
    raw = f"line1\nline2{UNIT_SEP}b"
    assert parse_records(raw) == [["line1\nline2", "b"]]


def test_parse_records_empty_units_in_record() -> None:
    # `a||b` (using US for |) is a record with a deliberately empty middle field.
    raw = f"a{UNIT_SEP}{UNIT_SEP}b"
    assert parse_records(raw) == [["a", "", "b"]]


def test_parse_records_single_empty_field_record_is_filtered() -> None:
    # A record consisting of one empty field encodes to the empty string,
    # which is indistinguishable from a no-record blank by the impl's
    # `if not rec.strip("\n")` guard — so it is filtered out. Documenting
    # this so the round-trip property test's strategy is justified.
    assert parse_records("") == []


def test_parse_records_unicode_payload() -> None:
    raw = f"café{UNIT_SEP}🎉{RECORD_SEP}你好{UNIT_SEP}مرحبا"
    assert parse_records(raw) == [["café", "🎉"], ["你好", "مرحبا"]]


def test_parse_records_blank_record_in_middle_is_skipped() -> None:
    # Two records separated by an empty record in the middle: the blank
    # record (just \n or empty) must be filtered.
    raw = f"a{UNIT_SEP}b{RECORD_SEP}\n{RECORD_SEP}c{UNIT_SEP}d"
    assert parse_records(raw) == [["a", "b"], ["c", "d"]]


# ---- hypothesis property test for parse_records() -------------------------

# Strategy: text without RS, US, or leading/trailing whitespace-only artefacts
# that the implementation deliberately strips. We also disallow strings that
# are entirely \n (since the impl strips records that are only newlines).
_field_text = st.text(
    alphabet=st.characters(
        blacklist_characters=[RECORD_SEP, UNIT_SEP, "\n"],
        # Also exclude surrogates (Python str can't represent them losslessly
        # via subprocess pipes, and they're not relevant for AS output).
        blacklist_categories=("Cs",),
    ),
    max_size=20,
)


# A record must encode to a non-whitespace string for parse_records to keep
# it — a record consisting of a single empty field encodes to "" which the
# implementation filters out by design (see `if not rec.strip("\n")` guard).
# The property test therefore filters out records whose US-join is empty/blank.
def _record_strategy() -> st.SearchStrategy[list[str]]:
    return st.lists(_field_text, min_size=1, max_size=4).filter(
        lambda rec: bool(UNIT_SEP.join(rec).strip("\n"))
    )


@pytest.mark.property
@given(records=st.lists(_record_strategy(), min_size=1, max_size=6))
@settings(max_examples=200)
def test_parse_records_round_trip(records: list[list[str]]) -> None:
    """Encoding a list-of-lists via RS/US joins and parsing back recovers
    the original — provided fields contain no RS/US/leading-newline chars
    AND each record encodes to a non-blank string (the impl filters blanks).
    """
    encoded = RECORD_SEP.join(UNIT_SEP.join(rec) for rec in records)
    assert parse_records(encoded) == records


@pytest.mark.property
@given(records=st.lists(_record_strategy(), min_size=1, max_size=6))
@settings(max_examples=200)
def test_parse_records_round_trip_with_trailing_rs(records: list[list[str]]) -> None:
    """Adding a trailing RS (osascript artefact) must not change the result."""
    encoded = RECORD_SEP.join(UNIT_SEP.join(rec) for rec in records) + RECORD_SEP
    assert parse_records(encoded) == records


# ---------------------------------------------------------------------------
# AppleScriptError exception class
# ---------------------------------------------------------------------------

def test_applescript_error_is_runtime_error_subclass() -> None:
    assert issubclass(AppleScriptError, RuntimeError)
    # Instances should also be RuntimeError instances (so generic except clauses
    # for RuntimeError still catch them).
    assert isinstance(AppleScriptError("x"), RuntimeError)


def test_applescript_error_carries_message() -> None:
    with pytest.raises(AppleScriptError, match="boom"):
        raise AppleScriptError("boom")


def test_applescript_error_str_returns_message() -> None:
    err = AppleScriptError("osascript failed (exit 1): nope")
    assert str(err) == "osascript failed (exit 1): nope"


def test_applescript_error_chaining_preserves_cause() -> None:
    original = ValueError("underlying cause")
    try:
        try:
            raise original
        except ValueError as e:
            raise AppleScriptError("wrapped") from e
    except AppleScriptError as caught:
        assert caught.__cause__ is original
        assert str(caught) == "wrapped"
    else:  # pragma: no cover — defensive
        pytest.fail("expected AppleScriptError to be raised")


def test_applescript_error_can_be_constructed_with_no_args() -> None:
    # RuntimeError allows zero-arg construction; AppleScriptError must too.
    err = AppleScriptError()
    assert isinstance(err, AppleScriptError)
    assert str(err) == ""


# ---------------------------------------------------------------------------
# RECORD_SEP / UNIT_SEP module constants
# ---------------------------------------------------------------------------

def test_record_sep_is_ascii_rs() -> None:
    assert RECORD_SEP == "\x1e"
    assert ord(RECORD_SEP) == 30
    assert len(RECORD_SEP) == 1


def test_unit_sep_is_ascii_us() -> None:
    assert UNIT_SEP == "\x1f"
    assert ord(UNIT_SEP) == 31
    assert len(UNIT_SEP) == 1


def test_separators_are_distinct() -> None:
    assert RECORD_SEP != UNIT_SEP


@pytest.mark.parametrize(
    "sample",
    [
        "Plain note title",
        "A note with\nseveral lines\nof prose.",
        "Bullet:\n- one\n- two\n- three",
        "Unicode: café 🎉 你好 مرحبا",
        "Markdown: **bold** _italic_ `code` [link](https://x)",
        "Tab\tseparated\tvalues — still safe",
        "",  # empty
    ],
)
def test_separators_absent_from_normal_note_content(sample: str) -> None:
    # The whole point of using \x1e / \x1f as delimiters: they cannot collide
    # with anything a user would plausibly type into Notes.
    assert RECORD_SEP not in sample
    assert UNIT_SEP not in sample
