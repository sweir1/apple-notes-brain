"""Hypothesis property tests for the chunker.

Property tests are slower than example-based ones, so they're marked
`@pytest.mark.property` and use a shrunken `max_examples` profile — the
unit suite still runs them by default but they don't dominate wall time.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from apple_notes_brain.semantic.chunker import chunk_markdown
from apple_notes_brain.semantic.types import ChunkerConfig

pytestmark = pytest.mark.property

# A relaxed setting: 50 examples is enough to flush out structural bugs
# without making the suite drag.
_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# Restrict the alphabet to printable ASCII + basic newline so generated
# documents look vaguely like real markdown. We DO want backticks and `#`
# to appear so the algorithm gets exercised on realistic noise.
_MD_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
        whitelist_categories=("Ll", "Lu", "Nd", "Po", "Zs"),
        whitelist_characters="\n#`$ ",
    ),
    min_size=0,
    max_size=1500,
)


@_SETTINGS
@given(text=_MD_TEXT)
def test_chunk_indices_always_contiguous(text):
    chunks = chunk_markdown(
        text, config=ChunkerConfig(chunk_size=200, min_chunk_chars=10)
    )
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


@_SETTINGS
@given(text=_MD_TEXT)
def test_chunk_content_hash_is_stable_across_runs(text):
    cfg = ChunkerConfig(chunk_size=200, min_chunk_chars=10)
    a = [c.content_hash for c in chunk_markdown(text, config=cfg)]
    b = [c.content_hash for c in chunk_markdown(text, config=cfg)]
    assert a == b


@_SETTINGS
@given(text=_MD_TEXT)
def test_chunk_sizes_bounded(text):
    """No chunk exceeds chunk_size by more than the documented slack.

    Slack accounts for sentence-merge fudge inside `_split_oversized_paragraph`
    and for the heading-prepend in build_chunk_embedding_text. 60 chars is
    generous; if this trips, the chunker has a real bug, not a corner case.
    """
    cfg = ChunkerConfig(chunk_size=200, min_chunk_chars=10)
    chunks = chunk_markdown(text, config=cfg)
    for c in chunks:
        assert len(c.content) <= cfg.chunk_size + 60, (
            f"chunk len {len(c.content)} exceeded {cfg.chunk_size} by too much: "
            f"{c.content!r}"
        )


@_SETTINGS
@given(text=_MD_TEXT)
def test_chunks_satisfy_min_chunk_chars(text):
    cfg = ChunkerConfig(chunk_size=200, min_chunk_chars=10)
    chunks = chunk_markdown(text, config=cfg)
    for c in chunks:
        # `min_chunk_chars` is enforced post-strip, so equality is OK.
        assert len(c.content) >= cfg.min_chunk_chars


@_SETTINGS
@given(prefix=_MD_TEXT, mid=_MD_TEXT, suffix=_MD_TEXT)
def test_single_char_mutation_changes_at_least_one_hash(prefix, mid, suffix):
    """Editing any byte in the body changes at least one chunk's hash —
    otherwise content-hash dedup would skip re-embedding after real edits."""
    cfg = ChunkerConfig(chunk_size=200, min_chunk_chars=10)
    base = prefix + mid + suffix
    mutated = prefix + mid + "Z" + suffix
    base_hashes = {c.content_hash for c in chunk_markdown(base, config=cfg)}
    mut_hashes = {c.content_hash for c in chunk_markdown(mutated, config=cfg)}
    # Either the chunk set differs OR no chunks emerged from this input
    # (e.g. all-whitespace) — both are acceptable; trip only on overlap
    # being identical with at least one chunk on each side.
    if base_hashes and mut_hashes:
        assert base_hashes != mut_hashes
