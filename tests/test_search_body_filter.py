#!/usr/bin/env python3
"""Regression test for the search_body=False title-only contract."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from apple_notes_brain import tools
from apple_notes_brain.sqlite_reader import NOTE_STORE_PATH


def _search(query: str, search_body: bool):
    return tools.search_notes(
        query=query,
        folder_path=None,
        search_body=search_body,
        fuzzy=False,
        mode="substring",
        limit=20,
    )


def main() -> int:
    if not NOTE_STORE_PATH.exists():
        print(f"SKIP: NoteStore not found at {NOTE_STORE_PATH}")
        return 0

    query = "butter"

    # Test 1: search_body=False must not leak body-only matches
    false_page = _search(query, search_body=False)
    for r in false_page.results:
        assert query.lower() in r.title.lower(), (
            f"regression: search_body=False returned note {r.id!r} "
            f"with title {r.title!r} — query {query!r} not in title"
        )
    print(f"[PASS] search_body=False title-only ({false_page.returned} results)")

    # Test 2: search_body=True should find at least as many hits
    true_page = _search(query, search_body=True)
    assert true_page.returned >= false_page.returned, (
        f"regression: search_body=True returned fewer hits ({true_page.returned}) "
        f"than False ({false_page.returned})"
    )
    print(f"[PASS] search_body=True >= False ({true_page.returned} >= {false_page.returned})")

    # Test 3: title-only matches still come through when search_body=True
    title_matches_in_true = [r for r in true_page.results if query.lower() in r.title.lower()]
    assert len(title_matches_in_true) >= false_page.returned, (
        "regression: fix suppressed title-only matches when search_body=True"
    )
    print(f"[PASS] title-only matches survive when search_body=True ({len(title_matches_in_true)} titles)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
