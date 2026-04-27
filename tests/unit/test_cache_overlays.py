"""Unit tests for cache overlay logic — tombstones, renames, count deltas.

Covers:
- Tombstone API: tombstone_folder / is_tombstoned / reap_tombstones
- Rename overlay: rename_path_overlay / apply_rename_overlay (with cascade)
- Count delta: adjust_note_count / get_count_delta
- Reaper: reap_overlays
- TTL boundaries (bug hotspot #19)
- Concurrent / nested rename cascade behaviour (bug hotspot #20)
- Thread-safety of the shared `_tomb_lock`

All time-dependent tests use the `frozen_monotonic` fixture (patches
`apple_notes_brain.cache.time.monotonic`). Cache global state is reset
before every test by the autouse `_reset_cache_state` fixture in
conftest.py.
"""
from __future__ import annotations

import threading

import pytest

from apple_notes_brain import cache


# ---------------------------------------------------------------------------
# Tombstones
# ---------------------------------------------------------------------------

class TestTombstones:
    def test_tombstone_then_is_tombstoned_true(self, frozen_monotonic):
        cache.tombstone_folder(99)
        assert cache.is_tombstoned(99) is True

    def test_different_pk_not_tombstoned(self, frozen_monotonic):
        cache.tombstone_folder(99)
        assert cache.is_tombstoned(100) is False

    def test_no_tombstone_returns_false(self, frozen_monotonic):
        assert cache.is_tombstoned(1234) is False

    def test_tombstone_expires_after_ttl(self, frozen_monotonic):
        cache.tombstone_folder(99)
        frozen_monotonic.advance(30)
        assert cache.is_tombstoned(99) is True
        frozen_monotonic.advance(35)  # total 65s
        assert cache.is_tombstoned(99) is False

    def test_tombstone_ttl_boundary_at_exactly_60s(self, frozen_monotonic):
        """Bug hotspot #19: at exactly TTL, what does is_tombstoned return?

        Code path: deadline = now + 60. Later: if monotonic > deadline -> expired.
        At exactly t+60: monotonic == deadline, NOT > deadline, so still alive.
        """
        cache.tombstone_folder(99)
        frozen_monotonic.advance(60.0)  # exactly TTL
        # Documented behaviour: still tombstoned at exact boundary.
        assert cache.is_tombstoned(99) is True

    def test_tombstone_ttl_boundary_just_over(self, frozen_monotonic):
        cache.tombstone_folder(99)
        frozen_monotonic.advance(60.0001)
        assert cache.is_tombstoned(99) is False

    def test_is_tombstoned_lazy_pops_expired(self, frozen_monotonic):
        """is_tombstoned() lazily removes expired entries on read."""
        cache.tombstone_folder(99)
        frozen_monotonic.advance(70)
        assert cache.is_tombstoned(99) is False
        # Internal map should no longer hold key 99.
        with cache._tomb_lock:
            assert 99 not in cache._tombstones

    def test_reap_removes_expired(self, frozen_monotonic):
        cache.tombstone_folder(99)
        frozen_monotonic.advance(70)
        cache.reap_tombstones()
        with cache._tomb_lock:
            assert cache._tombstones == {}

    def test_reap_keeps_live_entries(self, frozen_monotonic):
        cache.tombstone_folder(1)
        frozen_monotonic.advance(30)
        cache.tombstone_folder(2)
        frozen_monotonic.advance(40)  # total 70s; PK 1 expired (deadline=60), PK 2 still live (deadline=90)
        cache.reap_tombstones()
        with cache._tomb_lock:
            assert 1 not in cache._tombstones
            assert 2 in cache._tombstones

    def test_tombstone_overwrite_extends_deadline(self, frozen_monotonic):
        """Re-tombstoning the same PK refreshes the TTL."""
        cache.tombstone_folder(99)
        frozen_monotonic.advance(50)
        cache.tombstone_folder(99)  # refresh
        frozen_monotonic.advance(40)  # total 90 from first call, but only 40 since refresh
        assert cache.is_tombstoned(99) is True

    def test_thread_safety_concurrent_tombstones(self, frozen_monotonic):
        """Spawn many threads adding distinct tombstones; lock must serialise."""
        n_threads = 10
        ids: list[int] = []
        lock = threading.Lock()

        def worker():
            tid = threading.get_ident()
            cache.tombstone_folder(tid)
            with lock:
                ids.append(tid)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with cache._tomb_lock:
            for tid in ids:
                assert tid in cache._tombstones


# ---------------------------------------------------------------------------
# Rename overlays
# ---------------------------------------------------------------------------

class TestRenameOverlay:
    def test_rename_applies_to_matching_row(self, frozen_monotonic):
        cache.rename_path_overlay(7, "New/Path")
        rows = [{"id": "f7", "path": "Old/Path"}]
        out = cache.apply_rename_overlay(rows)
        assert out[0]["path"] == "New/Path"

    def test_no_overlay_leaves_rows_unchanged(self, frozen_monotonic):
        rows = [{"id": "f7", "path": "Old"}]
        out = cache.apply_rename_overlay(rows)
        assert out[0]["path"] == "Old"

    def test_overlay_cascades_to_descendants(self, frozen_monotonic):
        """Bug hotspot: rename folder 7 path 'a' -> 'X'; descendant 'a/sub' becomes 'X/sub'."""
        cache.rename_path_overlay(7, "X")
        rows = [
            {"id": "f7", "path": "a"},          # the renamed folder
            {"id": "f8", "path": "a/sub"},      # descendant
            {"id": "f9", "path": "a/sub/leaf"}, # deeper descendant
            {"id": "f10", "path": "other"},     # unrelated
        ]
        out = cache.apply_rename_overlay(rows)
        # Renamed row gets its new path
        assert out[0]["path"] == "X"
        # Descendants get prefix-rewritten
        assert out[1]["path"] == "X/sub"
        assert out[2]["path"] == "X/sub/leaf"
        # Unrelated unchanged
        assert out[3]["path"] == "other"

    def test_multiple_renames_applied_to_mixed_rows(self, frozen_monotonic):
        cache.rename_path_overlay(7, "X")
        cache.rename_path_overlay(8, "Y")
        rows = [
            {"id": "f7", "path": "a"},
            {"id": "f8", "path": "b"},
            {"id": "f9", "path": "c"},
        ]
        out = cache.apply_rename_overlay(rows)
        assert out[0]["path"] == "X"
        assert out[1]["path"] == "Y"
        assert out[2]["path"] == "c"

    def test_overlay_expires_after_ttl(self, frozen_monotonic):
        cache.rename_path_overlay(7, "New")
        frozen_monotonic.advance(70)
        rows = [{"id": "f7", "path": "Old"}]
        out = cache.apply_rename_overlay(rows)
        # Expired -> should NOT mutate
        assert out[0]["path"] == "Old"

    def test_concurrent_nested_renames_bug_hotspot_20(self, frozen_monotonic):
        """Bug hotspot #20: rename child 'a/b' -> 'a/B', then parent 'a' -> 'X'.

        Pinning the ACTUAL current behaviour. Implementation has two passes:

        Pass 1 — for each row, if its PK has a rename overlay, set
                 row['path']=new_path AND record old_path+'/' -> new_path+'/'
                 in prefix_map.
        Pass 2 — for each row, walk prefix_map; if the (possibly already-
                 mutated) path starts with any old_prefix, replace the prefix
                 with the matching new_prefix and break.

        With rows ordered [pk=60('a'), pk=50('a/b'), pk=70('a/b/c')]:
          After pass 1: paths = ['X', 'a/B', 'a/b/c']
                        prefix_map = {'a/': 'X/', 'a/b/': 'a/B/'}  (insertion order)

          Pass 2:
          - row[0]='X': no prefix match (no 'a/' or 'a/b/' prefix). Stays 'X'.
          - row[1]='a/B': starts with 'a/' (first key) -> becomes 'X/B'.
            ** Pass 2 RE-MUTATES a row that pass 1 already rewrote. **
            This means renamed rows can be silently re-rewritten by an ancestor
            rename — likely a real bug for callers expecting pass 1 to be
            authoritative.
          - row[2]='a/b/c': starts with 'a/' (first key) -> becomes 'X/b/c'.
            The 'a/b/' prefix is never tried due to the `break`.

        This is the documented buggy behaviour. If implementation ever changes
        this test will flag the change (failing loudly so the new behaviour
        can be reviewed).
        """
        cache.rename_path_overlay(50, "a/B")
        cache.rename_path_overlay(60, "X")
        rows = [
            {"id": "f60", "path": "a"},      # parent
            {"id": "f50", "path": "a/b"},    # child
            {"id": "f70", "path": "a/b/c"},  # grandchild
        ]
        out = cache.apply_rename_overlay(rows)
        # Parent row: 'X' (no prefix matched in pass 2).
        assert out[0]["path"] == "X"
        # Child row: pass 1 wrote 'a/B'; pass 2 re-rewrote via 'a/'->'X/' to 'X/B'.
        # **This is a real bug** — child's explicit rename overlay is partly
        # overridden by parent's prefix cascade. Pin behaviour to flag changes.
        assert out[1]["path"] == "X/B", (
            f"got {out[1]['path']!r}; if implementation was fixed to skip "
            "pass-2 cascade for rows already renamed in pass 1, expected 'a/B'"
        )
        # Grandchild: 'a/' matches before 'a/b/' (insertion order) -> 'X/b/c'.
        # The deeper 'a/b/' prefix is never applied because of `break`.
        assert out[2]["path"] == "X/b/c", (
            f"got {out[2]['path']!r}; expected first-prefix-wins result 'X/b/c'"
        )

    def test_apply_skips_rows_with_invalid_id(self, frozen_monotonic):
        cache.rename_path_overlay(7, "New")
        rows = [
            {"id": "fNOT_AN_INT", "path": "Old"},
            {"path": "missing-id-key"},
            {"id": "f7", "path": "OldReal"},
        ]
        out = cache.apply_rename_overlay(rows)
        # Bad id rows untouched; good row gets renamed
        assert out[0]["path"] == "Old"
        assert out[1]["path"] == "missing-id-key"
        assert out[2]["path"] == "New"

    def test_reap_removes_expired_renames(self, frozen_monotonic):
        cache.rename_path_overlay(7, "X")
        frozen_monotonic.advance(70)
        cache.reap_overlays()
        with cache._tomb_lock:
            assert 7 not in cache._renames

    def test_reap_keeps_live_renames(self, frozen_monotonic):
        cache.rename_path_overlay(7, "X")
        frozen_monotonic.advance(30)
        cache.reap_overlays()
        with cache._tomb_lock:
            assert 7 in cache._renames


# ---------------------------------------------------------------------------
# Count deltas
# ---------------------------------------------------------------------------

class TestCountDelta:
    def test_single_negative_delta(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        assert cache.get_count_delta(5) == -1

    def test_single_positive_delta(self, frozen_monotonic):
        cache.adjust_note_count(5, 3)
        assert cache.get_count_delta(5) == 3

    def test_deltas_accumulate(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        cache.adjust_note_count(5, -1)
        assert cache.get_count_delta(5) == -2

    def test_mixed_signs_sum(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        cache.adjust_note_count(5, -1)
        cache.adjust_note_count(5, 3)
        assert cache.get_count_delta(5) == 1

    def test_untouched_pk_returns_zero(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        assert cache.get_count_delta(99) == 0

    def test_no_delta_returns_zero(self, frozen_monotonic):
        assert cache.get_count_delta(42) == 0

    def test_delta_expires_after_ttl(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        frozen_monotonic.advance(70)
        assert cache.get_count_delta(5) == 0

    def test_reap_overlays_clears_expired_count_deltas(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        frozen_monotonic.advance(70)
        cache.reap_overlays()
        with cache._tomb_lock:
            assert 5 not in cache._count_deltas

    def test_reap_overlays_keeps_live_count_deltas(self, frozen_monotonic):
        cache.adjust_note_count(5, -1)
        frozen_monotonic.advance(30)
        cache.reap_overlays()
        with cache._tomb_lock:
            assert 5 in cache._count_deltas


# ---------------------------------------------------------------------------
# reap_overlays — both renames AND count_deltas, but NOT tombstones
# ---------------------------------------------------------------------------

class TestReapOverlays:
    def test_reap_overlays_cleans_both(self, frozen_monotonic):
        cache.rename_path_overlay(7, "X")
        cache.adjust_note_count(8, -2)
        frozen_monotonic.advance(70)
        cache.reap_overlays()
        with cache._tomb_lock:
            assert cache._renames == {}
            assert cache._count_deltas == {}

    def test_reap_overlays_does_not_touch_tombstones(self, frozen_monotonic):
        cache.tombstone_folder(99)
        cache.rename_path_overlay(7, "X")
        frozen_monotonic.advance(70)
        cache.reap_overlays()
        # Tombstone still present (reap_overlays only handles renames + count_deltas)
        with cache._tomb_lock:
            assert 99 in cache._tombstones
            assert 7 not in cache._renames

    def test_reap_tombstones_does_not_touch_overlays(self, frozen_monotonic):
        cache.rename_path_overlay(7, "X")
        cache.adjust_note_count(8, -1)
        cache.tombstone_folder(99)
        frozen_monotonic.advance(70)
        cache.reap_tombstones()
        with cache._tomb_lock:
            assert 99 not in cache._tombstones
            # Overlays untouched by reap_tombstones
            assert 7 in cache._renames
            assert 8 in cache._count_deltas
