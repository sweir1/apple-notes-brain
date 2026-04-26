"""Query tokenization and simple fuzzy ranking.

Fuzzy mode treats the query as whitespace-separated tokens. A note matches when
every token appears somewhere in its title or body (case-insensitive). Ranking
favours title hits and quantity of matched tokens. This handles word order and
extra words without needing edit-distance libraries.
"""
from __future__ import annotations

import re
from typing import Literal

SearchMode = Literal["substring", "phrase", "regex"]


class Matcher:
    """Opaque compiled query. Use compile_matcher() to build one.

    All three modes are reduced to a compiled re.Pattern:
    - substring: re.escape(query) with IGNORECASE — matches any occurrence of
      the query as a literal substring.
    - phrase: identical to substring (Apple Notes has no token boundaries we
      trust, so phrase and substring behave the same).
    - regex: the query is compiled as-is with IGNORECASE. An invalid pattern
      raises ValueError("invalid regex: ...").
    """

    def __init__(self, pattern: re.Pattern[str], query: str, is_literal: bool) -> None:
        self._pattern = pattern
        self._query = query
        self._is_literal = is_literal

    def test(self, text: str) -> bool:
        """Return True if the pattern matches anywhere in *text*."""
        return self._pattern.search(text) is not None

    def count(self, text: str) -> int:
        """Return the number of non-overlapping matches in *text*."""
        return sum(1 for _ in self._pattern.finditer(text))

    def first_match(self, text: str) -> str | None:
        """Return the first matched substring (group 0) or None."""
        m = self._pattern.search(text)
        return m.group(0) if m else None

    @property
    def literal(self) -> str | None:
        """The literal substring to highlight in snippets.

        For substring/phrase modes this is the original query string.
        For regex mode this is None (no safe literal to seed a snippet with).
        """
        return self._query if self._is_literal else None


def compile_matcher(query: str, mode: SearchMode) -> Matcher:
    """Compile a query+mode into a reusable Matcher with .test(text)->bool and .count(text)->int."""
    if mode in ("substring", "phrase"):
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return Matcher(pattern, query, is_literal=True)
    elif mode == "regex":
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return Matcher(pattern, query, is_literal=False)
    else:
        raise ValueError(f"unknown search mode: {mode!r}")


def tokenize(query: str) -> list[str]:
    """Lowercase whitespace-split, dropping tokens shorter than 2 chars."""
    return [t for t in query.lower().split() if len(t) >= 2]


def selective_token(tokens: list[str]) -> str:
    """Return the longest token — most likely to narrow the candidate pool."""
    return max(tokens, key=len) if tokens else ""


def _hit_count(tokens: list[str], haystack: str) -> int:
    lowered = haystack.lower()
    return sum(1 for t in tokens if t in lowered)


def score(tokens: list[str], title: str, body_text: str) -> float:
    """Rank a note. Title hits weigh more than body hits."""
    if not tokens:
        return 0.0
    title_hits = _hit_count(tokens, title)
    body_hits = _hit_count(tokens, body_text)
    total_hits = title_hits + body_hits
    if total_hits == 0:
        return 0.0
    # fraction of distinct tokens matched + title bonus
    distinct = len({t for t in tokens if t in title.lower() or t in body_text.lower()})
    coverage = distinct / len(tokens)
    title_bonus = 0.5 * (title_hits / len(tokens))
    return coverage + title_bonus
