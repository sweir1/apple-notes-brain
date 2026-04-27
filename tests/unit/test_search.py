"""Unit tests for the search-matching helpers in apple_notes_brain.search.

Covers:
- compile_matcher() — substring, phrase, regex modes; error paths.
- Matcher.test / .count / .first_match / .literal — public API surface.
- tokenize() — lowercase whitespace split, drops <2-char tokens.
- selective_token() — longest-token heuristic.
- score() — title-weighted ranking heuristic.
- Hypothesis property tests for invariants.
"""
from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from apple_notes_brain.search import (
    Matcher,
    compile_matcher,
    score,
    selective_token,
    tokenize,
)


# ---------------------------------------------------------------------------
# compile_matcher() — substring mode
# ---------------------------------------------------------------------------


def test_compile_matcher_substring_plain_query_matches() -> None:
    m = compile_matcher("hello", "substring")
    assert isinstance(m, Matcher)
    assert m.test("say hello") is True


def test_compile_matcher_substring_no_match() -> None:
    m = compile_matcher("hello", "substring")
    assert m.test("goodbye world") is False


def test_compile_matcher_substring_special_chars_are_literalised() -> None:
    """The `.` in the query is treated literally, not as regex any-char."""
    m = compile_matcher("a.b", "substring")
    assert m.test("a.b") is True
    assert m.test("axb") is False


@pytest.mark.parametrize(
    "query",
    ["a+b", "a*b", "a?b", "a|b", "(foo)", "[bar]", "^start", "end$", "a\\b"],
)
def test_compile_matcher_substring_regex_metachars_are_literal(query: str) -> None:
    """Every regex metachar in the query becomes a literal in substring mode."""
    m = compile_matcher(query, "substring")
    assert m.test(query) is True
    # The escaped form must be present in the compiled pattern.
    assert m.test(f"prefix {query} suffix") is True


def test_compile_matcher_substring_case_insensitive() -> None:
    m = compile_matcher("HELLO", "substring")
    assert m.test("hello world") is True
    assert m.test("Hello World") is True
    assert m.test("HeLLo") is True


def test_compile_matcher_substring_empty_query_always_matches() -> None:
    """Empty query compiles to the empty pattern, which matches anywhere
    (including empty string). Documenting actual behaviour."""
    m = compile_matcher("", "substring")
    assert m.test("") is True
    assert m.test("anything") is True


# ---------------------------------------------------------------------------
# compile_matcher() — phrase mode
# ---------------------------------------------------------------------------


def test_compile_matcher_phrase_behaves_like_substring() -> None:
    """Per the docstring, phrase mode is identical to substring mode."""
    m = compile_matcher("hello world", "phrase")
    assert m.test("say hello world today") is True
    assert m.test("hello there world") is False  # phrase requires literal sequence


def test_compile_matcher_phrase_special_chars_literal() -> None:
    m = compile_matcher("a.b", "phrase")
    assert m.test("a.b") is True
    assert m.test("axb") is False


def test_compile_matcher_phrase_case_insensitive() -> None:
    m = compile_matcher("Foo Bar", "phrase")
    assert m.test("FOO BAR") is True


# ---------------------------------------------------------------------------
# compile_matcher() — regex mode
# ---------------------------------------------------------------------------


def test_compile_matcher_regex_word_boundary() -> None:
    m = compile_matcher(r"\bfoo\b", "regex")
    assert m.test("foo bar") is True
    assert m.test("foobar") is False


def test_compile_matcher_regex_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid regex"):
        compile_matcher("(unclosed", "regex")


def test_compile_matcher_regex_invalid_chains_re_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        compile_matcher("[", "regex")
    assert isinstance(excinfo.value.__cause__, re.error)


def test_compile_matcher_regex_case_insensitive_by_default() -> None:
    m = compile_matcher("foo", "regex")
    assert m.test("FOO") is True
    assert m.test("Foo") is True


def test_compile_matcher_regex_anchors() -> None:
    start = compile_matcher(r"^foo", "regex")
    assert start.test("foo bar") is True
    assert start.test("bar foo") is False

    end = compile_matcher(r"bar$", "regex")
    assert end.test("foo bar") is True
    assert end.test("bar baz") is False


def test_compile_matcher_regex_unicode_cjk() -> None:
    m = compile_matcher(r"[一-鿿]+", "regex")
    assert m.test("hello 你好 world") is True
    assert m.test("ascii only") is False


# ---------------------------------------------------------------------------
# compile_matcher() — unknown mode
# ---------------------------------------------------------------------------


def test_compile_matcher_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown search mode"):
        compile_matcher("x", "garbage_mode")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["", "SUBSTRING", "Regex", "fuzzy", "exact"])
def test_compile_matcher_rejects_non_canonical_mode_strings(mode: str) -> None:
    with pytest.raises(ValueError, match="unknown search mode"):
        compile_matcher("x", mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Matcher.test()
# ---------------------------------------------------------------------------


def test_matcher_test_returns_bool() -> None:
    m = compile_matcher("foo", "substring")
    assert m.test("foo") is True
    assert m.test("bar") is False


def test_matcher_test_empty_text_no_match() -> None:
    m = compile_matcher("foo", "substring")
    assert m.test("") is False


def test_matcher_test_empty_text_with_empty_query() -> None:
    """Both empty: empty pattern matches the empty string."""
    m = compile_matcher("", "substring")
    assert m.test("") is True


def test_matcher_test_none_text_raises_type_error() -> None:
    """re.Pattern.search rejects None — document the propagated TypeError."""
    m = compile_matcher("foo", "substring")
    with pytest.raises(TypeError):
        m.test(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Matcher.count()
# ---------------------------------------------------------------------------


def test_matcher_count_zero_on_no_match() -> None:
    m = compile_matcher("foo", "substring")
    assert m.count("bar baz qux") == 0


def test_matcher_count_non_overlapping() -> None:
    """`aa` in `aaaa` yields 2 non-overlapping matches."""
    m = compile_matcher("aa", "substring")
    assert m.count("aaaa") == 2


def test_matcher_count_many_matches() -> None:
    m = compile_matcher("ab", "substring")
    assert m.count("ab ab ab cdab xyab") == 5


def test_matcher_count_case_insensitive() -> None:
    m = compile_matcher("foo", "substring")
    assert m.count("FOO foo Foo fOo") == 4


def test_matcher_count_regex() -> None:
    m = compile_matcher(r"\d+", "regex")
    assert m.count("a1 b22 c333 d") == 3


# ---------------------------------------------------------------------------
# Matcher.first_match()
# ---------------------------------------------------------------------------


def test_matcher_first_match_returns_substring_with_text_casing() -> None:
    m = compile_matcher("hello", "substring")
    # Original text casing is preserved, not the query casing.
    assert m.first_match("Say HELLO World") == "HELLO"


def test_matcher_first_match_none_on_no_match() -> None:
    m = compile_matcher("foo", "substring")
    assert m.first_match("bar baz") is None


def test_matcher_first_match_regex_returns_group_zero() -> None:
    """Even with capturing groups, group(0) — the whole match — is returned."""
    m = compile_matcher(r"(\d+)-(\d+)", "regex")
    assert m.first_match("range 12-34 here") == "12-34"


def test_matcher_first_match_returns_first_occurrence() -> None:
    m = compile_matcher("ab", "substring")
    assert m.first_match("xx ab yy ab zz") == "ab"


# ---------------------------------------------------------------------------
# Matcher.literal property
# ---------------------------------------------------------------------------


def test_matcher_literal_substring_returns_query() -> None:
    m = compile_matcher("hello world", "substring")
    assert m.literal == "hello world"


def test_matcher_literal_phrase_returns_query() -> None:
    m = compile_matcher("hello world", "phrase")
    assert m.literal == "hello world"


def test_matcher_literal_regex_returns_none() -> None:
    m = compile_matcher(r"\bfoo\b", "regex")
    assert m.literal is None


def test_matcher_literal_substring_preserves_metachars_unescaped() -> None:
    """The literal is the original query, not the escaped form used internally."""
    m = compile_matcher("a.b+c", "substring")
    assert m.literal == "a.b+c"


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("", []),
        ("hello", ["hello"]),
        ("hello world", ["hello", "world"]),
        ("HELLO", ["hello"]),
        ("HeLLo WoRLd", ["hello", "world"]),
        ("a hi", ["hi"]),  # 1-char tokens dropped
        ("a b c", []),  # all 1-char dropped
        ("ab cd", ["ab", "cd"]),  # 2-char retained
        ("   spaced   out   ", ["spaced", "out"]),
        ("tab\tseparated", ["tab", "separated"]),
        ("multi\n\nline\twords", ["multi", "line", "words"]),
    ],
)
def test_tokenize_cases(query: str, expected: list[str]) -> None:
    assert tokenize(query) == expected


def test_tokenize_keeps_punctuation_in_token() -> None:
    """tokenize uses str.split() with no separator, so punctuation stays attached."""
    assert tokenize("foo, bar.") == ["foo,", "bar."]


def test_tokenize_unicode() -> None:
    assert tokenize("你好 world") == ["你好", "world"]


def test_tokenize_unicode_lowercased() -> None:
    """Unicode characters are passed through .lower()."""
    assert tokenize("CAFÉ MAÑANA") == ["café", "mañana"]


# ---------------------------------------------------------------------------
# selective_token()
# ---------------------------------------------------------------------------


def test_selective_token_empty_returns_empty_string() -> None:
    assert selective_token([]) == ""


def test_selective_token_single() -> None:
    assert selective_token(["hello"]) == "hello"


def test_selective_token_picks_longest() -> None:
    assert selective_token(["a", "longest", "mid"]) == "longest"


def test_selective_token_tied_length_returns_first() -> None:
    """max() with key=len returns the first maximal element on ties."""
    assert selective_token(["foo", "bar", "baz"]) == "foo"


def test_selective_token_with_unicode() -> None:
    assert selective_token(["ab", "你好世界", "cd"]) == "你好世界"


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_returns_float() -> None:
    result = score(["foo"], "foo bar", "baz")
    assert isinstance(result, float)


def test_score_no_tokens_returns_zero() -> None:
    assert score([], "anything", "anywhere") == 0.0


def test_score_no_matches_returns_zero() -> None:
    assert score(["zzz"], "abc", "def") == 0.0


def test_score_title_match_outweighs_body_match() -> None:
    title_score = score(["foo"], "foo bar", "x")
    body_score = score(["foo"], "x", "foo bar")
    assert title_score > body_score


def test_score_token_coverage_bonus() -> None:
    """Matching more distinct tokens scores higher than matching fewer."""
    full = score(["a", "b"], "a b text", "")
    partial = score(["aa", "bb"], "aa text", "")
    assert full > partial


def test_score_title_only_match_positive() -> None:
    assert score(["foo"], "foo bar", "") > 0.0


def test_score_body_only_match_positive() -> None:
    assert score(["foo"], "title", "foo bar") > 0.0


def test_score_case_insensitive_for_lowercase_tokens() -> None:
    """Tokens are assumed lowercased (per tokenize). Title/body are .lower()-ed
    for matching, so casing doesn't affect the score."""
    upper = score(["foo"], "FOO BAR", "")
    lower = score(["foo"], "foo bar", "")
    assert upper == lower


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(query=st.text(min_size=1, max_size=20), text=st.text(max_size=200))
def test_property_substring_matches_python_in_operator(query: str, text: str) -> None:
    """Matcher.test in substring mode matches the case-insensitive `in` check."""
    m = compile_matcher(query, "substring")
    assert m.test(text) == (query.lower() in text.lower())


@pytest.mark.property
@settings(max_examples=100)
@given(
    tokens=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=0,
            max_size=8,
        ),
        max_size=10,
    )
)
def test_property_tokenize_round_trip_via_join(tokens: list[str]) -> None:
    """tokenize(' '.join(tokens)) == filtered LOWERCASED tokens of length >= 2.

    Implementation lowercases FIRST then filters by length, so we replicate
    that order. The Turkish capital İ (U+0130) lowercases to U+0069 U+0307
    (i + combining dot above) — length 2 — which would not survive a
    pre-lowercase length filter, but DOES survive the post-lowercase one.
    Hypothesis surfaced this with `tokens=['İ']`.
    """
    joined = " ".join(tokens)
    expected = [tl for t in tokens if t.strip() and " " not in t for tl in [t.lower()] if len(tl) >= 2]
    assert tokenize(joined) == expected


@pytest.mark.property
@settings(max_examples=200)
@given(text=st.text(max_size=200))
def test_property_count_non_negative(text: str) -> None:
    m = compile_matcher("x", "substring")
    assert m.count(text) >= 0


@pytest.mark.property
@settings(max_examples=200)
@given(text=st.text(max_size=200))
def test_property_count_consistent_with_test(text: str) -> None:
    """count() > 0  iff  test() is True."""
    m = compile_matcher("a", "substring")
    assert (m.count(text) > 0) == m.test(text)


@pytest.mark.property
@settings(max_examples=100)
@given(
    tokens=st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=6),
        min_size=1,
        max_size=8,
    )
)
def test_property_selective_token_is_member(tokens: list[str]) -> None:
    """selective_token always returns one of the input tokens (when non-empty)."""
    chosen = selective_token(tokens)
    assert chosen in tokens


@pytest.mark.property
@settings(max_examples=100)
@given(
    tokens=st.lists(
        st.text(alphabet="abcdefghij", min_size=1, max_size=6),
        min_size=1,
        max_size=8,
    )
)
def test_property_selective_token_is_longest(tokens: list[str]) -> None:
    """No input token is strictly longer than the chosen one."""
    chosen = selective_token(tokens)
    assert all(len(t) <= len(chosen) for t in tokens)


@pytest.mark.property
@settings(max_examples=100)
@given(
    tokens=st.lists(
        st.text(alphabet="abc", min_size=2, max_size=4), min_size=0, max_size=5
    ),
    title=st.text(max_size=50),
    body=st.text(max_size=100),
)
def test_property_score_non_negative(
    tokens: list[str], title: str, body: str
) -> None:
    assert score(tokens, title, body) >= 0.0
