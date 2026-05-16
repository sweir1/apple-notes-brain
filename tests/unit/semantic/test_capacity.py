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
