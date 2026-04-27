"""Unit tests for `apple_notes_brain.html_text`.

Covers the public API:
- ``html_to_text(html)`` — HTML → plaintext extraction.
- ``snippets(text, query, window, max_spans)`` — case-insensitive snippet
  windows around matches.
- ``count_matches(text, query)`` — non-overlapping case-insensitive count.

Behaviour invariants pinned by these tests (verified against the actual
implementation, not assumed):

* Block-level tags (p, div, br, li, ul, ol, h1-h6, blockquote, tr, hr) emit
  newlines on both open and close; the post-pass collapses runs of blank
  lines to at most one and strips leading/trailing whitespace.
* Inline tags (b, i, span, a, strong, em…) leave content flush — no extra
  whitespace inserted.
* On no match, ``snippets`` returns a single fallback snippet of the first
  ``2 * window`` characters (with a trailing ellipsis if truncated). Empty
  text or empty query returns ``[]``.
* ``snippets`` deduplicates near-by matches: a subsequent match must start
  at least ``window`` chars past the previous span's end.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apple_notes_brain.html_text import count_matches, html_to_text, snippets


# ---------------------------------------------------------------------------
# html_to_text — basics
# ---------------------------------------------------------------------------


def test_html_to_text_empty_returns_empty() -> None:
    assert html_to_text("") == ""


def test_html_to_text_none_safe_via_falsy() -> None:
    # The function is annotated `str` but guards on falsy; document that.
    assert html_to_text("") == ""


def test_html_to_text_plain_text_unchanged() -> None:
    assert html_to_text("hello") == "hello"


def test_html_to_text_single_paragraph_strips_outer_whitespace() -> None:
    # `<p>` emits newlines around content but the post-pass strips the
    # leading/trailing blank lines.
    assert html_to_text("<p>hello</p>") == "hello"


def test_html_to_text_two_paragraphs_separated_by_blank_line() -> None:
    # Block tags emit newlines on open AND close → between two <p>s you
    # get a blank line. The collapse pass keeps a single blank.
    assert html_to_text("<p>a</p><p>b</p>") == "a\n\nb"


def test_html_to_text_inline_tags_no_newlines() -> None:
    assert html_to_text("hello <b>world</b>") == "hello world"


@pytest.mark.parametrize(
    "tag",
    ["b", "i", "u", "strong", "em", "span", "a"],
)
def test_html_to_text_inline_tag_does_not_break_line(tag: str) -> None:
    assert html_to_text(f"foo<{tag}>bar</{tag}>baz") == "foobarbaz"


@pytest.mark.parametrize(
    "tag",
    ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr"],
)
def test_html_to_text_block_tag_introduces_newline(tag: str) -> None:
    out = html_to_text(f"a<{tag}>b</{tag}>c")
    # 'a' precedes the open, 'b' is the content, 'c' follows the close.
    # The exact spacing varies (collapsed blanks) but 'a', 'b', 'c' must
    # appear on different non-empty lines (or with at most one blank between).
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines == ["a", "b", "c"]


def test_html_to_text_self_closing_br_creates_break() -> None:
    out = html_to_text("a<br/>b")
    # <br> open+close (HTMLParser treats startendtag as start+end) → blank line.
    assert out == "a\n\nb"


def test_html_to_text_hr_creates_break() -> None:
    out = html_to_text("a<hr/>b")
    assert "a" in out and "b" in out
    # 'a' and 'b' must NOT be on the same non-blank line.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines == ["a", "b"]


# ---------------------------------------------------------------------------
# html_to_text — entities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html_in,expected",
    [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        # &nbsp; comes through as the actual nbsp char (\xa0). The post-pass
        # treats it as non-whitespace for line-trim purposes (rstrip strips
        # ASCII whitespace only — nbsp survives).
        ("a&nbsp;b", "a\xa0b"),
        ("&amp;amp;", "&amp;"),  # double-encoded
    ],
)
def test_html_to_text_entities_decoded(html_in: str, expected: str) -> None:
    assert html_to_text(html_in) == expected


def test_html_to_text_mixed_entities_and_text() -> None:
    assert html_to_text("Tom &amp; Jerry &lt;tag&gt;") == "Tom & Jerry <tag>"


# ---------------------------------------------------------------------------
# html_to_text — malformed / quirky input
# ---------------------------------------------------------------------------


def test_html_to_text_unclosed_tag_does_not_crash() -> None:
    # No assertion on exact output beyond "doesn't raise and returns text".
    assert html_to_text("<unclosed>x") == "x"


def test_html_to_text_broken_nesting_does_not_crash() -> None:
    out = html_to_text("<p><b></p></b>")
    # Empty content → empty plaintext after strip.
    assert out == ""


def test_html_to_text_script_content_kept_as_text() -> None:
    # html_text.py does NOT strip script content — that's html_validate's job.
    # Document the actual behaviour: text content of <script> is preserved.
    assert html_to_text("<script>alert(1)</script>after") == "alert(1)after"


def test_html_to_text_style_content_kept_as_text() -> None:
    assert html_to_text("<style>.x{color:red}</style>body") == ".x{color:red}body"


def test_html_to_text_object_tag_content_dropped() -> None:
    # <object> isn't a block tag; with no inner data text it leaves nothing.
    assert html_to_text('<object data="x"/>after') == "after"


def test_html_to_text_real_world_apple_html() -> None:
    apple = '<div><b><span style="font-size: 24px">Heading</span></b></div>'
    assert html_to_text(apple) == "Heading"


def test_html_to_text_collapses_multiple_blank_lines() -> None:
    # Many block tags in a row would produce many blank lines; the collapse
    # pass caps at one.
    out = html_to_text("<p>a</p><p></p><p></p><p>b</p>")
    # No more than one blank line between 'a' and 'b'.
    assert "\n\n\n" not in out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines == ["a", "b"]


def test_html_to_text_strips_leading_and_trailing_whitespace() -> None:
    assert html_to_text("   spaces   ") == "spaces"


def test_html_to_text_preserves_internal_inline_spaces() -> None:
    # Inline runs aren't whitespace-collapsed; the only collapsing is at
    # line-rstrip and blank-line dedup.
    assert html_to_text("<b>a   b</b>") == "a   b"


def test_html_to_text_nested_tags_no_extra_spaces() -> None:
    assert html_to_text("<p>hello <b>world</b></p>") == "hello world"


# ---------------------------------------------------------------------------
# html_to_text — property test
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    text=st.text(
        alphabet=st.characters(
            blacklist_characters="<>&\r",
            blacklist_categories=("Cs", "Cc"),
        ),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip() and "\n" not in s),
)
@settings(max_examples=50, deadline=None)
def test_html_to_text_plain_text_roundtrip(text: str) -> None:
    """Text with no tags/entities/newlines passes through unchanged after strip."""
    assert html_to_text(text) == text.strip()


# ---------------------------------------------------------------------------
# snippets — basic returns
# ---------------------------------------------------------------------------


def test_snippets_empty_query_returns_empty() -> None:
    assert snippets("any text", "") == []


def test_snippets_empty_text_returns_empty() -> None:
    assert snippets("", "x") == []


def test_snippets_max_spans_zero_returns_empty() -> None:
    assert snippets("hello hello", "hello", max_spans=0) == []


def test_snippets_no_match_returns_fallback_short() -> None:
    # No match in short text → fallback is first 2*window chars; if the whole
    # text fits, no trailing ellipsis.
    out = snippets("hello world", "xyz", window=100)
    assert out == ["hello world"]


def test_snippets_no_match_returns_fallback_truncated() -> None:
    # Long text with no match → fallback is first 2*window chars + "…".
    out = snippets("a" * 500, "zz", window=20)
    assert len(out) == 1
    assert out[0].endswith("…")
    # Body before the ellipsis is exactly 2*window 'a's.
    assert out[0] == "a" * 40 + "…"


def test_snippets_query_longer_than_text_falls_back() -> None:
    # No match possible → fallback path returns first 2*window of text.
    out = snippets("aaaaa", "aaaaaaaa", window=100)
    assert out == ["aaaaa"]


# ---------------------------------------------------------------------------
# snippets — single match
# ---------------------------------------------------------------------------


def test_snippets_single_match_middle_has_both_ellipses() -> None:
    text = "a" * 200 + "FIND" + "b" * 200
    out = snippets(text, "FIND", window=20)
    assert len(out) == 1
    span = out[0]
    assert span.startswith("…") and span.endswith("…")
    assert "FIND" in span
    # length = ellipsis + window 'a's + 'FIND' + window 'b's + ellipsis
    assert span == "…" + "a" * 20 + "FIND" + "b" * 20 + "…"


def test_snippets_match_at_start_no_leading_ellipsis() -> None:
    text = "FIND" + "x" * 50
    out = snippets(text, "FIND", window=5)
    assert len(out) == 1
    assert not out[0].startswith("…")
    assert out[0].endswith("…")


def test_snippets_match_at_end_no_trailing_ellipsis() -> None:
    text = "x" * 50 + "FIND"
    out = snippets(text, "FIND", window=5)
    assert len(out) == 1
    assert out[0].startswith("…")
    assert not out[0].endswith("…")


def test_snippets_case_insensitive() -> None:
    out = snippets("HELLO world", "hello")
    assert len(out) == 1
    # Returned text preserves original casing of the source.
    assert "HELLO" in out[0]


def test_snippets_window_zero_returns_match_only() -> None:
    out = snippets("hello", "h", window=0)
    # Match is at start so no leading ellipsis; 'ello' follows so trailing
    # ellipsis appears.
    assert out == ["h…"]


def test_snippets_collapses_internal_newlines_to_spaces() -> None:
    text = "intro\n\nFIND\n\noutro"
    out = snippets(text, "FIND", window=10)
    assert len(out) == 1
    # Newlines within the snippet are converted to spaces and runs collapsed.
    assert "\n" not in out[0]


def test_snippets_unicode_text_and_query() -> None:
    out = snippets("café au lait", "CAFÉ", window=20)
    assert len(out) == 1
    assert "café" in out[0]


# ---------------------------------------------------------------------------
# snippets — multiple matches / max_spans
# ---------------------------------------------------------------------------


def test_snippets_multiple_well_separated_matches() -> None:
    # Three matches each separated by enough text that they don't visually
    # overlap with window=2.
    text = "aaXaa" + "bbbbb" + "aaXaa" + "bbbbb" + "aaXaa"
    out = snippets(text, "X", window=2, max_spans=3)
    assert len(out) == 3


def test_snippets_max_spans_caps_results() -> None:
    text = ("aaXaa" + "bbbbb") * 5
    out = snippets(text, "X", window=2, max_spans=2)
    assert len(out) == 2


def test_snippets_overlap_filter_skips_close_matches() -> None:
    # Two X's only 2 chars apart with window=5 — second one overlaps the
    # first span and is filtered.
    text = "aaaaaXaaXaaaaa"
    out = snippets(text, "X", window=5, max_spans=3)
    # The two X's are too close to produce two separate spans.
    assert len(out) == 1


@pytest.mark.parametrize(
    "text,query,window,max_spans,expected_len",
    [
        # 1. single match, plenty of room
        ("abc FIND xyz", "FIND", 5, 3, 1),
        # 2. case-insensitive match
        ("ABC find XYZ", "FIND", 5, 3, 1),
        # 3. fallback for no match (text shorter than 2*window)
        ("short", "missing", 100, 3, 1),
        # 4. empty query
        ("anything", "", 10, 3, 0),
        # 5. empty text
        ("", "anything", 10, 3, 0),
        # 6. max_spans=0
        ("a FIND b", "FIND", 5, 0, 0),
        # 7. unicode
        ("hello café world", "café", 5, 3, 1),
        # 8. multiple non-overlapping
        ("aaXaaaaaaaaaaaaXaa", "X", 2, 3, 2),
        # 9. trailing match (no closing ellipsis)
        ("xxxxxxFIND", "FIND", 3, 1, 1),
        # 10. leading match (no opening ellipsis)
        ("FINDxxxxxx", "FIND", 3, 1, 1),
        # 11. window=0 — degenerate but valid
        ("hello", "ll", 0, 3, 1),
        # 12. max_spans=1 with many matches → only first
        ("XXXXX", "X", 2, 1, 1),
    ],
)
def test_snippets_parametrized(
    text: str, query: str, window: int, max_spans: int, expected_len: int
) -> None:
    out = snippets(text, query, window=window, max_spans=max_spans)
    assert len(out) == expected_len


# ---------------------------------------------------------------------------
# snippets — property test
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    text=st.text(min_size=0, max_size=200),
    query=st.text(min_size=0, max_size=10),
    max_spans=st.integers(min_value=0, max_value=10),
    window=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_snippets_never_exceeds_max_spans(
    text: str, query: str, max_spans: int, window: int
) -> None:
    out = snippets(text, query, window=window, max_spans=max_spans)
    # Fallback path returns up to 1 element regardless of max_spans, so the
    # bound is max(1, max_spans). When max_spans=0 we explicitly return [].
    if max_spans == 0:
        assert out == []
    else:
        assert len(out) <= max(1, max_spans)


@pytest.mark.property
@given(
    text=st.text(min_size=1, max_size=200),
    query=st.text(min_size=1, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_snippets_no_newlines_in_output(text: str, query: str) -> None:
    """Returned snippets should never contain raw newline chars."""
    for span in snippets(text, query, window=10, max_spans=3):
        assert "\n" not in span
        assert "\r" not in span


# ---------------------------------------------------------------------------
# count_matches
# ---------------------------------------------------------------------------


def test_count_matches_no_match_returns_zero() -> None:
    assert count_matches("hello", "xyz") == 0


def test_count_matches_single_match() -> None:
    assert count_matches("hello", "hello") == 1


def test_count_matches_multiple_non_overlapping() -> None:
    assert count_matches("aaaa", "a") == 4


def test_count_matches_overlapping_pattern_counts_non_overlapping() -> None:
    # "aaaa" / "aa" → indices 0 and 2 (non-overlapping).
    assert count_matches("aaaa", "aa") == 2


def test_count_matches_case_insensitive() -> None:
    assert count_matches("ABC abc AbC", "abc") == 3


def test_count_matches_empty_query_returns_zero() -> None:
    assert count_matches("text", "") == 0


def test_count_matches_empty_text_returns_zero() -> None:
    assert count_matches("", "x") == 0


def test_count_matches_unicode() -> None:
    assert count_matches("café CAFÉ café", "café") == 3


@pytest.mark.parametrize(
    "text,query,expected",
    [
        ("hello world", "hello", 1),
        ("hello hello", "hello", 2),
        ("aaa", "aa", 1),  # match at 0; next search starts at 2 — no more.
        ("aaaaa", "aa", 2),  # 0, 2; next at 4 — no more.
        ("HELLO", "hello", 1),
        ("xxxxx", "y", 0),
        ("", "", 0),
        ("abc", "", 0),
    ],
)
def test_count_matches_parametrized(text: str, query: str, expected: int) -> None:
    assert count_matches(text, query) == expected


@pytest.mark.property
@given(
    text=st.text(min_size=0, max_size=200),
    query=st.text(min_size=0, max_size=5),
)
@settings(max_examples=50, deadline=None)
def test_count_matches_never_negative(text: str, query: str) -> None:
    assert count_matches(text, query) >= 0
