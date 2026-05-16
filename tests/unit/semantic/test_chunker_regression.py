"""Regression tests pinning behaviours that bit obsidian-brain in the past
or that we explicitly designed for.

Each test cites the *symptom* a future change would cause if it broke
the invariant, so future-us can decide whether to fix the test or fix
the regression.
"""
from __future__ import annotations

from apple_notes_brain.semantic.chunker import chunk_markdown
from apple_notes_brain.semantic.types import ChunkerConfig

TINY = ChunkerConfig(chunk_size=100, min_chunk_chars=10, heading_split_depth=4)


def test_empty_note_with_only_heading_produces_no_chunks():
    """Regression: obsidian-brain v1.7.2 silently dropped notes whose
    chunker output was empty. We want the chunker to confirm 'no chunks
    here'; the indexer's empty-note fallback then handles synthesising
    a title-only embedding.

    Symptom if this breaks: notes that consist of nothing but a single
    heading get a confusing 'partial chunk' instead of triggering the
    indexer's synthetic fallback. Search recall for those notes degrades.
    """
    assert chunk_markdown("# Just a heading", config=TINY) == []


def test_frontmatter_only_doc_produces_no_chunks():
    """Regression: a doc that's entirely frontmatter (no body) must yield
    no chunks — indexer's empty-note fallback takes over.

    Symptom if this breaks: indexer over-counts 'real' chunks for notes
    with no embeddable body and we'd waste an embed call on whitespace.
    """
    doc = "---\ntitle: x\ntags: [a]\n---\n"
    assert chunk_markdown(doc, config=TINY) == []


def test_code_fence_at_document_start():
    """Regression: a doc starting with a code fence shouldn't break
    frontmatter detection (`---` heuristic looks at the very first chars
    only; we used to over-aggressively try to strip).

    Symptom if this breaks: a Python file pasted into a note that begins
    with `---` separator-style headers gets the leading code mistakenly
    stripped as frontmatter.
    """
    doc = "```python\ndef hello():\n    pass\n```\n\nBody after fence here.\n"
    chunks = chunk_markdown(doc, config=TINY)
    joined = "\n".join(c.content for c in chunks)
    assert "def hello" in joined


def test_consecutive_headings_no_body_between_keeps_both():
    """Regression: when two headings sit back-to-back with no body, both
    must still appear in some sense — either as section markers or
    rolled into the parent.

    Symptom if this breaks: pure-outline notes (just nested headings
    with no body) become invisible to semantic search.
    """
    doc = (
        "# Top\n\n"
        "## Mid\n\n"
        "Body of mid section, long enough to keep this chunk.\n"
    )
    chunks = chunk_markdown(doc, config=TINY)
    headings = [c.heading for c in chunks]
    assert "Mid" in headings


def test_heading_with_special_chars_preserved():
    """Regression: heading text with `: ` or other punctuation must be
    preserved verbatim, NOT split into multiple headings.

    Symptom if this breaks: notes with headings like 'Q1: Sales Review'
    have the heading text mangled in chunk metadata.
    """
    doc = "# Q1: Sales Review (2026)\n\nBody content here longer than min.\n"
    chunks = chunk_markdown(doc, config=TINY)
    assert chunks[0].heading == "Q1: Sales Review (2026)"


def test_content_hash_includes_heading_so_renames_re_embed():
    """Regression: if the user renames a heading without touching the
    body, the chunk's content_hash MUST change so the indexer re-embeds.
    Heading text carries strong retrieval signal — embedding under the
    old heading would surface stale matches.

    Symptom if this breaks: heading renames don't trigger re-embed; user
    sees old-headed chunks ranked above new-headed ones until full reindex.
    """
    a = chunk_markdown("# Old name\n\nSame body content here.\n", config=TINY)[0]
    b = chunk_markdown("# New name\n\nSame body content here.\n", config=TINY)[0]
    assert a.content_hash != b.content_hash
