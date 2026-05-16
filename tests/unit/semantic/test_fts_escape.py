"""Tests for FTS5 query escaping."""
from __future__ import annotations

import pytest

from apple_notes_brain.semantic.fts import escape_fts5_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("hello", "hello"),
        ("hello world", "hello world"),
        ("hello*", "hello*"),                       # FTS5 prefix wildcard ok
        ("foo-bar", '"foo-bar"'),                    # `-` treated as col operator → quote
        ("hello AND world", '"hello AND world"'),   # quote keyword operators
        ('foo "bar" baz', '"foo ""bar"" baz"'),     # double internal quotes
        ("foo. bar?", '"foo. bar?"'),               # punctuation → quoted
        ("special!chars()", '"special!chars()"'),
    ],
)
def test_escape_table(raw, expected):
    assert escape_fts5_query(raw) == expected


def test_escape_preserves_empty():
    assert escape_fts5_query("") == ""


def test_escape_handles_unicode():
    # Unicode word chars are kept; \w matches them under re.UNICODE which
    # is Python's default. Whether that's "safe" depends on FTS5's
    # tokenizer; we just verify the function doesn't crash.
    out = escape_fts5_query("東京 tokyo")
    assert "東京" in out or '"東京' in out
