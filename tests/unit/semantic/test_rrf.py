"""Tests for the Reciprocal Rank Fusion implementation."""
from __future__ import annotations

import pytest

from apple_notes_brain.semantic.search import reciprocal_rank_fusion


def _key(x):
    return x  # x is a hashable string


def test_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([], _key) == []
    assert reciprocal_rank_fusion([[], []], _key) == []


def test_single_list_passthrough_ordering():
    out = reciprocal_rank_fusion([["a", "b", "c"]], _key, k=60)
    assert [r.item for r in out] == ["a", "b", "c"]


def test_single_list_top_score_matches_formula():
    out = reciprocal_rank_fusion([["a"]], _key, k=60)
    assert out[0].score == pytest.approx(1.0 / 61)


def test_two_identical_lists_double_score():
    out = reciprocal_rank_fusion([["a"], ["a"]], _key, k=60)
    assert out[0].score == pytest.approx(2.0 / 61)


def test_item_appears_in_two_lists_at_different_positions():
    """Item 'x' at rank 0 in list1 and rank 2 in list2:
    score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63"""
    out = reciprocal_rank_fusion([["x", "y"], ["a", "b", "x"]], _key, k=60)
    x_entry = next(r for r in out if r.item == "x")
    assert x_entry.score == pytest.approx(1.0 / 61 + 1.0 / 63)


def test_output_sorted_descending_by_score():
    out = reciprocal_rank_fusion(
        [["a", "b", "c", "d"], ["b", "c", "a", "d"]], _key, k=60
    )
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_lower_k_amplifies_top_rank():
    out_k1 = reciprocal_rank_fusion([["a", "b"]], _key, k=1)
    out_k1000 = reciprocal_rank_fusion([["a", "b"]], _key, k=1000)
    # k=1 → a's score = 1/2, b's = 1/3 (ratio ≈ 1.5)
    # k=1000 → a's = 1/1001, b's = 1/1002 (ratio ≈ 1.001)
    assert out_k1[0].score / out_k1[1].score > out_k1000[0].score / out_k1000[1].score


def test_custom_key_fn_dedups_across_lists():
    """When two lists hold different object instances representing the
    same logical item, they merge via key_fn."""
    out = reciprocal_rank_fusion(
        [
            [{"id": "n1", "rank": 0}, {"id": "n2", "rank": 1}],
            [{"id": "n1", "rank": 0}, {"id": "n3", "rank": 1}],
        ],
        key_fn=lambda o: o["id"],
        k=60,
    )
    ids = [r.item["id"] for r in out]
    assert ids.count("n1") == 1  # merged
    # n1 should have the highest score (appears in both at rank 0).
    assert ids[0] == "n1"


def test_empty_in_one_list_is_ignored():
    out = reciprocal_rank_fusion([["a", "b"], []], _key, k=60)
    assert [r.item for r in out] == ["a", "b"]


def test_score_strictly_positive_for_present_items():
    out = reciprocal_rank_fusion([["a"]], _key, k=60)
    assert out[0].score > 0


def test_stable_across_runs():
    args = ([["a", "b", "c"], ["b", "a", "c"]], _key)
    a = reciprocal_rank_fusion(*args, k=60)
    b = reciprocal_rank_fusion(*args, k=60)
    assert [(r.item, r.score) for r in a] == [(r.item, r.score) for r in b]
