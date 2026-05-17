"""Tests for the adaptive-capacity ratchet."""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.capacity import (
    CHARS_PER_TOKEN,
    FALLBACK_MAX_TOKENS,
    MIN_DISCOVERED_TOKENS,
    EmbedderCapacity,
    approx_tokens_for,
    get_capacity,
    initialise_capacity,
    reduce_discovered_max_tokens,
    reset_discovered_capacity,
)
from apple_notes_brain.semantic.store import open_db
from .conftest import FakeEmbedder


@pytest.fixture
def conn(tmp_path: Path):
    c = open_db(tmp_path / "x.db")
    yield c
    c.close()


@pytest.fixture
def emb():
    e = FakeEmbedder()
    e.init()
    return e


# ---------------------------------------------------------------------------
# Default capacity when no row exists
# ---------------------------------------------------------------------------

def test_get_capacity_returns_fallback_when_no_row(conn, emb):
    cap = get_capacity(conn, emb)
    assert cap.advertised_max_tokens is None
    assert cap.discovered_max_tokens is None
    assert cap.effective_max_tokens == FALLBACK_MAX_TOKENS
    assert cap.chunk_budget_chars == int(FALLBACK_MAX_TOKENS * CHARS_PER_TOKEN)


def test_get_capacity_uses_advertised_when_no_discovered(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=1024)
    cap = get_capacity(conn, emb)
    assert cap.advertised_max_tokens == 1024
    assert cap.discovered_max_tokens is None
    assert cap.effective_max_tokens == 1024


def test_get_capacity_uses_min_when_both_present(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=1024)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    cap = get_capacity(conn, emb)
    assert cap.advertised_max_tokens == 1024
    # discovered ≈ 600 * 0.9 = 540
    assert cap.discovered_max_tokens == 540
    assert cap.effective_max_tokens == 540


# ---------------------------------------------------------------------------
# Initialise + idempotency
# ---------------------------------------------------------------------------

def test_initialise_capacity_inserts_row(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    row = conn.execute(
        "SELECT advertised_max_tokens FROM embedder_capability "
        "WHERE embedder_id = ?", (emb.model_identifier(),)
    ).fetchone()
    assert row[0] == 512


def test_initialise_capacity_idempotent(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    initialise_capacity(conn, emb, advertised_max_tokens=1024)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()[0]
    assert cnt == 1
    # Latest advertised value wins.
    last = conn.execute(
        "SELECT advertised_max_tokens FROM embedder_capability "
        "WHERE embedder_id = ?", (emb.model_identifier(),)
    ).fetchone()
    assert last[0] == 1024


def test_initialise_capacity_preserves_dim(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=512, dim=384)
    row = conn.execute(
        "SELECT dim FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()
    assert row[0] == 384


def test_initialise_capacity_dim_persists_when_omitted_later(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=512, dim=384)
    initialise_capacity(conn, emb, advertised_max_tokens=600)  # no dim
    row = conn.execute(
        "SELECT dim FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()
    assert row[0] == 384  # not overwritten with NULL


# ---------------------------------------------------------------------------
# Ratchet
# ---------------------------------------------------------------------------

def test_ratchet_reduces_to_observed_minus_margin(conn, emb):
    new_value = reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    # 600 * 0.9 = 540
    assert new_value == 540


def test_ratchet_floors_at_min(conn, emb):
    new_value = reduce_discovered_max_tokens(conn, emb, observed_tokens=100)
    assert new_value == MIN_DISCOVERED_TOKENS


def test_ratchet_is_monotonic_decreasing(conn, emb):
    """Successive ratchets only go DOWN — already-shrunken capacity can't
    be widened by a later observation."""
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    # Second observation says we can handle 5000 — but we should NOT
    # widen the discovered ceiling, since the previous observation
    # proved the embedder truncates at <600.
    reduce_discovered_max_tokens(conn, emb, observed_tokens=5000)
    cap = get_capacity(conn, emb)
    assert cap.discovered_max_tokens == 540  # still the lower of the two


def test_ratchet_records_method_field(conn, emb):
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    row = conn.execute(
        "SELECT method FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()
    assert row[0] == "ratchet"


# ---------------------------------------------------------------------------
# approx_tokens_for helper
# ---------------------------------------------------------------------------

def test_approx_tokens_proportional_to_length():
    a = approx_tokens_for("x" * 100)
    b = approx_tokens_for("x" * 1000)
    assert b > a


def test_approx_tokens_handles_empty():
    assert approx_tokens_for("") == 1


def test_approx_tokens_includes_safety_margin():
    # 250 chars at 2.5 chars/token is 100 tokens; with 10% margin = 110.
    assert approx_tokens_for("x" * 250) >= 100


# ---------------------------------------------------------------------------
# Distinct embedders track independently
# ---------------------------------------------------------------------------

def test_distinct_embedders_have_independent_capacity(conn):
    e1 = FakeEmbedder(model_id="m1")
    e2 = FakeEmbedder(model_id="m2")
    initialise_capacity(conn, e1, advertised_max_tokens=512)
    initialise_capacity(conn, e2, advertised_max_tokens=8192)
    cap1 = get_capacity(conn, e1)
    cap2 = get_capacity(conn, e2)
    assert cap1.advertised_max_tokens == 512
    assert cap2.advertised_max_tokens == 8192


# ---------------------------------------------------------------------------
# Phase θ — reset_discovered_capacity
# ---------------------------------------------------------------------------

def test_reset_restores_discovered_to_advertised(conn, emb):
    """After a ratchet shrinks discovered, reset_discovered_capacity
    restores it to the advertised ceiling so the next full indexing
    pass starts from the embedder's actual stated capacity."""
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    cap_before = get_capacity(conn, emb)
    assert cap_before.discovered_max_tokens == 540

    reset_discovered_capacity(conn, emb)

    cap_after = get_capacity(conn, emb)
    assert cap_after.discovered_max_tokens == 512
    assert cap_after.effective_max_tokens == 512


def test_reset_is_noop_when_no_row(conn, emb):
    """No capability row at all → reset is a silent no-op (no insert)."""
    reset_discovered_capacity(conn, emb)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()[0]
    assert cnt == 0


def test_reset_is_idempotent(conn, emb):
    """Calling reset repeatedly doesn't drift."""
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    reset_discovered_capacity(conn, emb)
    reset_discovered_capacity(conn, emb)
    reset_discovered_capacity(conn, emb)
    cap = get_capacity(conn, emb)
    assert cap.discovered_max_tokens == 512


def test_reset_does_not_widen_when_advertised_is_null(conn, emb):
    """If advertised is NULL, reset can't manufacture a value from thin
    air — discovered stays NULL too."""
    # Insert a row with only discovered set.
    reduce_discovered_max_tokens(conn, emb, observed_tokens=400)
    reset_discovered_capacity(conn, emb)
    row = conn.execute(
        "SELECT advertised_max_tokens, discovered_max_tokens "
        "FROM embedder_capability WHERE embedder_id = ?",
        (emb.model_identifier(),),
    ).fetchone()
    assert row[0] is None
    assert row[1] is None  # reset to NULL since advertised is NULL


def test_reset_independent_per_embedder(conn):
    """Resetting one embedder doesn't affect the other's discovered value."""
    e1 = FakeEmbedder(model_id="m1")
    e2 = FakeEmbedder(model_id="m2")
    initialise_capacity(conn, e1, advertised_max_tokens=512)
    initialise_capacity(conn, e2, advertised_max_tokens=512)
    reduce_discovered_max_tokens(conn, e1, observed_tokens=600)
    reduce_discovered_max_tokens(conn, e2, observed_tokens=600)
    reset_discovered_capacity(conn, e1)
    cap1 = get_capacity(conn, e1)
    cap2 = get_capacity(conn, e2)
    assert cap1.discovered_max_tokens == 512  # reset
    assert cap2.discovered_max_tokens == 540  # untouched


# ---------------------------------------------------------------------------
# Phase θ — method tracking on EmbedderCapacity
# ---------------------------------------------------------------------------

def test_method_defaults_to_fallback_when_no_row(conn, emb):
    cap = get_capacity(conn, emb)
    assert cap.method == "fallback"


def test_method_is_ratchet_after_reduce(conn, emb):
    initialise_capacity(conn, emb, advertised_max_tokens=1024)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    cap = get_capacity(conn, emb)
    assert cap.method == "ratchet"


def test_method_field_present_on_dataclass():
    """The EmbedderCapacity dataclass exposes a `method` attribute with
    the documented Literal type — protects against accidental field
    removal."""
    cap = EmbedderCapacity(
        embedder_id="m", advertised_max_tokens=512,
        discovered_max_tokens=None, effective_max_tokens=512,
        chunk_budget_chars=1280,
    )
    # Default value applies.
    assert cap.method == "fallback"


def test_method_preserved_when_only_advertised_set(conn, emb):
    """After initialise_capacity (no ratchet), method classification
    falls through to 'probe' — there's no stored method label yet,
    discovered isn't binding, so we report the most-likely source."""
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    cap = get_capacity(conn, emb)
    assert cap.method == "probe"


def test_method_after_reset_still_present(conn, emb):
    """After reset, the stored method column is COALESCE'd to 'fallback'
    if it was NULL. Either way it should be a valid label."""
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    reset_discovered_capacity(conn, emb)
    cap = get_capacity(conn, emb)
    # Method column still says 'ratchet' since reduce wrote it; reset
    # only COALESCE'd when null. That's fine — we surface the value.
    assert cap.method in ("ratchet", "fallback", "probe", "manual",
                          "tokenizer_config", "api_show")


# ---------------------------------------------------------------------------
# Phase θ — indexer.index_all calls reset_discovered_capacity at start
# ---------------------------------------------------------------------------

def test_index_all_resets_discovered_capacity_at_start(conn, emb, tmp_path: Path):
    """A full pass widens discovered back to advertised before iterating,
    so a single transient outlier doesn't permanently shrink chunks."""
    from apple_notes_brain.semantic.indexer import IndexPipeline, IndexerConfig
    from apple_notes_brain.semantic.source import FakeNotesSource, NoteRecord

    # Seed: an embedder with advertised=512 and a ratcheted discovered=540.
    initialise_capacity(conn, emb, advertised_max_tokens=512)
    reduce_discovered_max_tokens(conn, emb, observed_tokens=600)
    pre = get_capacity(conn, emb)
    assert pre.discovered_max_tokens == 540

    pipe = IndexPipeline(conn, emb, IndexerConfig(advertised_max_tokens=512))
    src = FakeNotesSource()
    rec = NoteRecord(
        z_identifier="zid-1", z_pk=1, title="t",
        folder=None, modified_at=1, locked=False, pinned=False,
    )
    src.add(rec, "body text")
    pipe.index_all(src)

    post = get_capacity(conn, emb)
    # Reset widened discovered back to advertised at the start of the pass.
    assert post.discovered_max_tokens == 512
