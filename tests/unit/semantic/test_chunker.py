"""Unit tests for `apple_notes_brain.semantic.chunker`.

Mirrors obsidian-brain's `test/embeddings/chunker.test.ts` and adds
adversarial / regression cases. Property-based tests live separately
in `test_chunker_property.py`.
"""
from __future__ import annotations

import pytest

from apple_notes_brain.semantic.chunker import (
    build_chunk_embedding_text,
    chunk_id,
    chunk_markdown,
)
from apple_notes_brain.semantic.types import (
    DEFAULT_CHUNKER_CONFIG,
    Chunk,
    ChunkerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY = ChunkerConfig(chunk_size=100, min_chunk_chars=10, heading_split_depth=4)


# ---------------------------------------------------------------------------
# Empty / trivial inputs
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    assert chunk_markdown("") == []


def test_only_whitespace_returns_empty_list():
    assert chunk_markdown("   \n\n   \t\n") == []


def test_below_min_chunk_chars_returns_empty_list():
    """A single paragraph shorter than min_chunk_chars must drop out —
    embedding it is noise."""
    assert chunk_markdown("hi", config=TINY) == []


def test_single_short_paragraph_yields_one_chunk():
    chunks = chunk_markdown("Hello world, here is a short note.", config=TINY)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.heading is None
    assert c.heading_level is None
    assert c.chunk_index == 0
    assert "Hello world" in c.content


# ---------------------------------------------------------------------------
# Heading splits
# ---------------------------------------------------------------------------

def test_splits_on_h1():
    doc = (
        "# First\n\n"
        "Body of first section here, long enough to clear min_chunk.\n\n"
        "# Second\n\n"
        "Body of second section here, also long enough.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 2
    assert chunks[0].heading == "First"
    assert chunks[0].heading_level == 1
    assert chunks[1].heading == "Second"
    assert chunks[1].heading_level == 1


def test_splits_on_h1_h2_h3_h4():
    doc = (
        "# H1\n\nAlpha body content here longer than min.\n\n"
        "## H2\n\nBeta body content here longer than min.\n\n"
        "### H3\n\nGamma body content here longer than min.\n\n"
        "#### H4\n\nDelta body content here longer than min.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert [(c.heading, c.heading_level) for c in chunks] == [
        ("H1", 1),
        ("H2", 2),
        ("H3", 3),
        ("H4", 4),
    ]


def test_h5_h6_do_not_split_by_default():
    """heading_split_depth=4 by default; H5/H6 must stay inside parent."""
    doc = (
        "# Parent\n\n"
        "Top-level body.\n\n"
        "##### Nested 5\n\n"
        "Nested content under five.\n\n"
        "###### Nested 6\n\n"
        "Nested content under six.\n"
    )
    chunks = chunk_markdown(doc)
    assert len(chunks) == 1
    assert chunks[0].heading == "Parent"
    # H5/H6 lines survive as body text.
    assert "Nested 5" in chunks[0].content
    assert "Nested 6" in chunks[0].content


def test_custom_depth_includes_h5():
    cfg = ChunkerConfig(chunk_size=1000, min_chunk_chars=10, heading_split_depth=5)
    doc = (
        "# Parent\n\nParent body content here longer than min.\n\n"
        "##### Five\n\nUnder five body content here longer than min.\n"
    )
    chunks = chunk_markdown(doc, config=cfg)
    assert {c.heading for c in chunks} == {"Parent", "Five"}


def test_heading_only_section_with_no_body_emits_no_chunk():
    """A heading with no body content and nothing else doesn't earn a chunk —
    the empty-note fallback in the indexer handles 'whole note is empty'."""
    doc = "# Empty\n"
    assert chunk_markdown(doc, config=TINY) == []


def test_preamble_before_first_heading_is_kept():
    doc = (
        "Preamble paragraph before any heading, long enough to keep.\n\n"
        "# First\n\n"
        "First-section body, long enough to keep.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 2
    assert chunks[0].heading is None
    assert "Preamble" in chunks[0].content
    assert chunks[1].heading == "First"


# ---------------------------------------------------------------------------
# Frontmatter handling
# ---------------------------------------------------------------------------

def test_frontmatter_stripped():
    doc = (
        "---\n"
        "title: Test\n"
        "tags: [a, b]\n"
        "---\n"
        "Body text here, long enough to keep.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert "title:" not in chunks[0].content
    assert "Body text here" in chunks[0].content


def test_malformed_frontmatter_treated_as_body():
    """Open fence with no close — treat the whole doc as body."""
    doc = "---\ntitle: Test\n\nBody content without close fence here.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    # The body retains the open fence and the title line.
    assert "title: Test" in chunks[0].content


def test_no_frontmatter_unchanged():
    doc = "Body text with no leading triple-dashes here.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("Body text")


def test_frontmatter_with_doc_after_yields_correct_lines():
    """Frontmatter is 4 lines (---\\nkey: v\\n---\\n); body lines should
    be reported relative to the original doc."""
    doc = "---\nx: 1\n---\n# H\n\nBody line content here.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    # start_line should reflect that the body sits after the frontmatter.
    assert chunks[0].start_line >= 4


# ---------------------------------------------------------------------------
# Code block preservation — headings inside fences must NOT trigger splits
# ---------------------------------------------------------------------------

def test_code_fence_preserves_inner_hash_as_not_a_heading():
    doc = (
        "# Real heading\n\n"
        "Some body before.\n\n"
        "```\n"
        "# this is a Python comment, not a markdown heading\n"
        "print('hi')\n"
        "```\n\n"
        "More body after.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    # Single section under "Real heading"
    assert len(chunks) >= 1
    headings = [c.heading for c in chunks]
    assert headings.count("Real heading") == len(chunks)
    # Code fence content survives in some chunk.
    joined = "\n".join(c.content for c in chunks)
    assert "print('hi')" in joined
    assert "# this is a Python comment" in joined


def test_latex_block_preserved():
    doc = (
        "Body content here longer than min chars.\n\n"
        "$$\n# inside latex\nx^2 + y^2 = z^2\n$$\n\n"
        "More body here longer than min chars.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    joined = "\n".join(c.content for c in chunks)
    assert "x^2 + y^2 = z^2" in joined
    # # inside latex must NOT have become a heading split.
    assert all(c.heading != "inside latex" for c in chunks)


# ---------------------------------------------------------------------------
# Oversize handling
# ---------------------------------------------------------------------------

def test_oversize_section_splits_on_paragraphs():
    cfg = ChunkerConfig(chunk_size=80, min_chunk_chars=5, heading_split_depth=4)
    paras = [f"Paragraph {i} content content content." for i in range(10)]
    doc = "# H\n\n" + "\n\n".join(paras) + "\n"
    chunks = chunk_markdown(doc, config=cfg)
    assert len(chunks) > 1
    # Every chunk respects chunk_size (with small slack for sentence merge).
    for c in chunks:
        assert len(c.content) <= cfg.chunk_size + 10


def test_oversize_paragraph_splits_on_sentences():
    cfg = ChunkerConfig(chunk_size=100, min_chunk_chars=5, heading_split_depth=4)
    sentences = [
        "Apple released a new device.",
        "Reviews were mixed at launch.",
        "Battery life was the biggest concern.",
        "Pricing was higher than expected.",
        "Many buyers waited for the second generation.",
    ]
    doc = "# H\n\n" + " ".join(sentences) + "\n"
    chunks = chunk_markdown(doc, config=cfg)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content) <= cfg.chunk_size + 40


def test_oversize_unsplittable_text_hard_cut():
    """A single contiguous blob with no paragraph/sentence breaks must
    still produce bounded chunks via the hard-cut fallback."""
    cfg = ChunkerConfig(chunk_size=50, min_chunk_chars=5, heading_split_depth=4)
    blob = "x" * 500
    chunks = chunk_markdown(blob, config=cfg)
    assert len(chunks) >= 5
    # Hard-cut chunks are exact slices ≤ chunk_size.
    for c in chunks:
        assert len(c.content) <= cfg.chunk_size


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def test_content_hash_is_deterministic():
    doc = "# H\n\nIdentical body across runs.\n"
    a = chunk_markdown(doc, config=TINY)[0].content_hash
    b = chunk_markdown(doc, config=TINY)[0].content_hash
    assert a == b


def test_content_hash_differs_for_different_text():
    a = chunk_markdown("# A\n\nFirst body content.\n", config=TINY)[0].content_hash
    b = chunk_markdown("# A\n\nSecond body content.\n", config=TINY)[0].content_hash
    assert a != b


def test_content_hash_includes_heading():
    """Two chunks with the same body but different headings must have
    different hashes — otherwise content-hash dedup would erroneously
    skip re-embedding when only the heading changed."""
    a = chunk_markdown("# A\n\nSame body content here.\n", config=TINY)[0].content_hash
    b = chunk_markdown("# B\n\nSame body content here.\n", config=TINY)[0].content_hash
    assert a != b


def test_content_hash_is_32_hex_chars():
    h = chunk_markdown("# H\n\nBody content longer than min.\n", config=TINY)[0].content_hash
    assert len(h) == 32
    int(h, 16)  # raises if not hex


# ---------------------------------------------------------------------------
# chunk_index contiguity + chunk_id
# ---------------------------------------------------------------------------

def test_chunk_indices_are_contiguous():
    doc = (
        "# A\n\nFirst body content here.\n\n"
        "# B\n\nSecond body content here.\n\n"
        "# C\n\nThird body content here.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_chunk_id_format():
    assert chunk_id("note-abc", 0) == "note-abc#0"
    assert chunk_id("note-abc", 42) == "note-abc#42"


# ---------------------------------------------------------------------------
# build_chunk_embedding_text
# ---------------------------------------------------------------------------

def test_build_embedding_text_with_heading_prepends_it():
    c = Chunk(
        chunk_index=0,
        heading="Setup",
        heading_level=2,
        content="Install dependencies first.",
        content_hash="x",
        start_line=1,
        end_line=1,
    )
    assert build_chunk_embedding_text(c) == "Setup\n\nInstall dependencies first."


def test_build_embedding_text_without_heading_returns_body():
    c = Chunk(
        chunk_index=0,
        heading=None,
        heading_level=None,
        content="Plain body content.",
        content_hash="x",
        start_line=1,
        end_line=1,
    )
    assert build_chunk_embedding_text(c) == "Plain body content."


# ---------------------------------------------------------------------------
# Line endings + unicode
# ---------------------------------------------------------------------------

def test_crlf_line_endings_normalised():
    doc = "# H\r\n\r\nBody content here longer than min.\r\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    # No \r survives in chunk content.
    assert "\r" not in chunks[0].content


def test_cr_only_line_endings_normalised():
    doc = "# H\r\rBody content here longer than min.\r"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert "\r" not in chunks[0].content


def test_unicode_heading_and_body():
    doc = "# 東京\n\n本文の内容、十分に長い文字列です。\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert chunks[0].heading == "東京"
    assert "本文の内容" in chunks[0].content


def test_emoji_in_content():
    doc = "# H\n\nNote with emojis 🎉 here and longer text.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) == 1
    assert "🎉" in chunks[0].content


# ---------------------------------------------------------------------------
# Line numbers
# ---------------------------------------------------------------------------

def test_line_numbers_one_based_and_reflect_body_position():
    doc = "Line one.\n\nLine three. " + "x" * 30 + "\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert len(chunks) >= 1
    assert chunks[0].start_line >= 1
    assert chunks[0].end_line >= chunks[0].start_line
