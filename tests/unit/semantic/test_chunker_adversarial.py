"""Chunker adversarial inputs — the things that broke real chunkers in
the wild and the things a stack of clever regex tricks might not handle.
"""
from __future__ import annotations

import pytest

from apple_notes_brain.semantic.chunker import chunk_markdown
from apple_notes_brain.semantic.types import ChunkerConfig

TINY = ChunkerConfig(chunk_size=100, min_chunk_chars=10, heading_split_depth=4)


def test_huge_blob_completes_quickly():
    """10MB note body must chunk in reasonable time — no quadratic
    blow-up. Wall-clock is loose because pytest.timeout already enforces
    a global limit; this is just a sanity check that we don't loop."""
    blob = ("Lorem ipsum dolor sit amet. " * 50_000)  # ~1.4MB
    chunks = chunk_markdown(blob, config=ChunkerConfig(chunk_size=500, min_chunk_chars=20))
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.content) <= 700  # chunk_size + slack


def test_only_hash_chars_no_split():
    """A run of `#` characters with no space afterwards is NOT a heading
    (markdown spec). Must not crash, must not over-split."""
    doc = "#######\nLine with hashes.\n" + "Body content longer than min chars.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) >= 1
    joined = " ".join(c.content for c in chunks)
    assert "#######" in joined or "Line with hashes" in joined


def test_sentinel_char_in_user_text_survives():
    """A user note that contains the PUA sentinel chars must not break
    restoration. The sentinel includes a numeric index that won't match
    any real entry in the restore map, so the substitution falls through
    leaving the original chars intact."""
    user_text = (
        "Body with the literal char  in it and more body after, "
        "long enough to keep this chunk around for sure.\n"
    )
    chunks = chunk_markdown(user_text, config=TINY)
    assert len(chunks) >= 1
    # The exotic character survives somewhere.
    assert any("" in c.content for c in chunks)


def test_null_byte_in_input_does_not_crash():
    doc = "Body with a \x00 null in it and more text after.\n"
    chunks = chunk_markdown(doc, config=TINY)
    # Doesn't crash; chunk content keeps the null.
    assert len(chunks) >= 1
    assert any("\x00" in c.content for c in chunks)


def test_long_url_paragraph_hard_cuts():
    """A long URL has no whitespace — must hard-cut without infinite loop."""
    url = "https://example.com/" + "abc" * 200  # 620 chars
    cfg = ChunkerConfig(chunk_size=100, min_chunk_chars=10, heading_split_depth=4)
    chunks = chunk_markdown(url, config=cfg)
    assert len(chunks) >= 6
    for c in chunks:
        assert len(c.content) <= cfg.chunk_size


def test_mixed_code_and_latex_blocks():
    doc = (
        "# H\n\n"
        "Body around blocks.\n\n"
        "```python\nfor i in range(10):\n    print(i)\n```\n\n"
        "Middle paragraph here.\n\n"
        "$$\n\\int_0^1 x^2 \\, dx = \\frac{1}{3}\n$$\n\n"
        "Trailing body content longer than min.\n"
    )
    chunks = chunk_markdown(doc, config=ChunkerConfig(chunk_size=400, min_chunk_chars=10))
    joined = "\n".join(c.content for c in chunks)
    assert "for i in range(10):" in joined
    assert "\\int_0^1" in joined


def test_leading_blank_lines():
    doc = "\n\n\n# H\n\nBody content longer than min chars.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert chunks[0].heading == "H"


def test_trailing_whitespace_in_heading():
    doc = "# Heading with trailing   \n\nBody content longer than min.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert chunks[0].heading == "Heading with trailing"


def test_extreme_chunk_size_one_yields_per_char():
    """chunk_size=1 is silly but legal; output should be many tiny chunks
    (post min_chunk_chars filter most get dropped — we want no crash)."""
    cfg = ChunkerConfig(chunk_size=1, min_chunk_chars=1, heading_split_depth=4)
    chunks = chunk_markdown("abcdef", config=cfg)
    # Each char ≥ min_chunk_chars=1 so 6 chunks expected.
    assert len(chunks) == 6


def test_repeated_chunk_content_unique_indices_distinct_hashes():
    """Two paragraphs with identical content under different headings must
    get distinct hashes (heading included) and distinct indices."""
    doc = (
        "# Alpha\n\nIdentical paragraph content.\n\n"
        "# Beta\n\nIdentical paragraph content.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 2
    assert chunks[0].chunk_index != chunks[1].chunk_index
    assert chunks[0].content_hash != chunks[1].content_hash
