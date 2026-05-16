"""Tests for the FakeEmbedder used across the unit suite.

If these slip, every downstream test that builds on FakeEmbedder is also
suspect — so they live as their own file rather than piggybacking on a
fixture-only conftest.
"""
from __future__ import annotations

import numpy as np
import pytest

from apple_notes_brain.semantic.types import Embedder, TooLongError

from .conftest import FakeEmbedder


def test_fake_embedder_satisfies_protocol():
    """Drift here breaks every downstream test using the fake; pin it."""
    emb = FakeEmbedder()
    assert isinstance(emb, Embedder)


def test_fake_embedder_init_increments_counter():
    emb = FakeEmbedder()
    assert emb.init_count == 0
    emb.init()
    assert emb.init_count == 1


def test_fake_embedder_deterministic_for_same_input():
    emb = FakeEmbedder()
    v1 = emb.embed("hello world")
    v2 = emb.embed("hello world")
    np.testing.assert_array_equal(v1, v2)


def test_fake_embedder_different_inputs_differ():
    emb = FakeEmbedder()
    v1 = emb.embed("hello world")
    v2 = emb.embed("goodbye world")
    assert not np.array_equal(v1, v2)


def test_fake_embedder_task_type_differentiates():
    """Asymmetric-model path: query vs document produce different vectors."""
    emb = FakeEmbedder()
    v_doc = emb.embed("hello", task_type="document")
    v_qry = emb.embed("hello", task_type="query")
    assert not np.array_equal(v_doc, v_qry)


def test_fake_embedder_output_is_float32():
    emb = FakeEmbedder()
    v = emb.embed("anything")
    assert v.dtype == np.float32


def test_fake_embedder_output_is_unit_normalised():
    emb = FakeEmbedder()
    v = emb.embed("anything reasonable")
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_fake_embedder_output_dim_matches_declared():
    for dim in (64, 384, 768):
        emb = FakeEmbedder(dim=dim)
        v = emb.embed("x")
        assert v.shape == (dim,)
        assert emb.dimensions() == dim


def test_fake_embedder_too_long_raises():
    emb = FakeEmbedder(max_chars=10)
    with pytest.raises(TooLongError):
        emb.embed("x" * 11)


def test_fake_embedder_under_limit_works():
    emb = FakeEmbedder(max_chars=10)
    v = emb.embed("x" * 10)
    assert v.shape == (384,)


def test_fake_embedder_provider_name_and_model_id():
    emb = FakeEmbedder()
    assert emb.provider_name() == "fake"
    assert emb.model_identifier() == "fake/deterministic-v1"


def test_fake_embedder_dispose_idempotent():
    emb = FakeEmbedder()
    emb.dispose()
    emb.dispose()
    assert emb.dispose_count == 2


def test_fake_embedder_call_counters_increment():
    emb = FakeEmbedder()
    emb.embed("a")
    emb.embed("b")
    emb.embed("c")
    assert emb.embed_count == 3
