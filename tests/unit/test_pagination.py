"""Unit tests for cursor pagination in ``apple_notes_brain.sqlite_reader``.

Covers:
- ``encode_cursor(offset) -> str`` / ``decode_cursor(cursor) -> int`` round-trip.
- ``decode_cursor`` error surface (malformed, non-base64, non-integer payload).
- Bug-hotspot probes over the cursor + pagination logic in ``list_notes`` and
  ``search_notes`` (sqlite_reader.py:782-852, 877-963).

Bug hotspots being targeted (numbered to match the exploration notes):

* **#1** sqlite_reader.py:833-835 — ``has_more`` boundary: the function fetches
  ``limit + 1`` rows and slices to ``limit`` if over. Off-by-one risk on the
  exact boundary (``len == limit`` vs ``len == limit + 1``).
* **#2** sqlite_reader.py:837 — ``encode_cursor(offset + limit)`` cannot
  overflow in Python (arbitrary-precision ints), but the *next* request that
  uses the result still hits SQLite OFFSET. Verified to never produce a cursor
  on the final page (``has_more=False``).
* **#5** sqlite_reader.py:782 — ``decode_cursor`` now enforces bounds
  (0 <= offset <= MAX_CURSOR_OFFSET) and rejects out-of-range or negative
  offsets with ``ValueError("invalid cursor: ...")``.
* **#6** sqlite_reader.py:826 — LIMIT/OFFSET passed straight to SQLite. The
  defence-in-depth gap is now closed by the bounds check in ``decode_cursor``.

Most tests are direct ``encode_cursor`` / ``decode_cursor`` unit tests; only
the ``has_more`` boundary tests mock the underlying connection.
"""
from __future__ import annotations

import base64

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apple_notes_brain.sqlite_reader import (
    MAX_CURSOR_OFFSET,
    decode_cursor,
    encode_cursor,
)


# ---------------------------------------------------------------------------
# encode_cursor / decode_cursor round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [0, 1, 50, 100, 999, 1_000, 1_000_000, 10**8, MAX_CURSOR_OFFSET],
)
def test_cursor_roundtrip_preserves_offset(offset: int) -> None:
    """For every plausible non-negative offset within the cap,
    decode(encode(o)) == o."""
    assert decode_cursor(encode_cursor(offset)) == offset


def test_cursor_roundtrip_zero() -> None:
    assert decode_cursor(encode_cursor(0)) == 0


def test_cursor_roundtrip_one_million() -> None:
    assert decode_cursor(encode_cursor(1_000_000)) == 1_000_000


def test_encode_cursor_negative_offset_now_rejected() -> None:
    """Defence-in-depth: encode_cursor refuses negative offsets up-front so
    our own pagination code can't accidentally produce a malformed cursor."""
    with pytest.raises(ValueError, match="negative offset"):
        encode_cursor(-5)


def test_decode_cursor_negative_offset_now_rejected() -> None:
    """Even if a client crafts a base64-encoded negative integer directly
    (bypassing encode_cursor), decode_cursor must reject it."""
    cursor = base64.urlsafe_b64encode(b"-5").decode()
    with pytest.raises(ValueError, match="negative offset"):
        decode_cursor(cursor)


def test_cursor_decode_none_returns_zero() -> None:
    """``cursor=None`` is the documented 'first page' sentinel."""
    assert decode_cursor(None) == 0


# ---------------------------------------------------------------------------
# encode_cursor — direct properties
# ---------------------------------------------------------------------------


def test_encode_cursor_returns_str() -> None:
    assert isinstance(encode_cursor(42), str)


def test_encode_cursor_is_deterministic() -> None:
    """Same input must produce the same output (no salt / nonce)."""
    assert encode_cursor(123) == encode_cursor(123)


def test_encode_cursor_distinct_inputs_distinct_outputs() -> None:
    """Different offsets must produce different cursors (within reason —
    base64 of distinct decimal strings is injective)."""
    seen = {encode_cursor(o) for o in range(0, 1000, 37)}
    assert len(seen) == len(range(0, 1000, 37))


def test_encode_cursor_output_is_valid_urlsafe_base64() -> None:
    """The cursor must be decodable by stdlib urlsafe_b64decode without error."""
    cursor = encode_cursor(12345)
    base64.urlsafe_b64decode(cursor.encode())  # must not raise


# ---------------------------------------------------------------------------
# decode_cursor — invalid input
# ---------------------------------------------------------------------------


def test_decode_cursor_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor("")


def test_decode_cursor_non_base64_raises() -> None:
    """Strings outside the urlsafe_b64 alphabet must raise."""
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor("not base 64!@#$%")


def test_decode_cursor_base64_of_non_integer_raises() -> None:
    """Valid base64 but the payload isn't a parseable integer."""
    cursor = base64.urlsafe_b64encode(b"hello").decode()
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor(cursor)


def test_decode_cursor_base64_of_float_raises() -> None:
    """``int('3.14')`` raises — should bubble up as 'invalid cursor'."""
    cursor = base64.urlsafe_b64encode(b"3.14").decode()
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor(cursor)


def test_decode_cursor_whitespace_raises() -> None:
    """A whitespace-only cursor isn't valid base64 of an integer."""
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor("   ")


def test_decode_cursor_very_long_input_handled_gracefully() -> None:
    """A pathologically long base64 string must not hang / crash.

    Python 3.11+ caps ``int(s)`` parsing at 4300 digits by default
    (``sys.set_int_max_str_digits``) — so payloads longer than that raise
    ``ValueError`` from ``int()``, which the wrapper re-raises as
    'invalid cursor'. This is good defence-in-depth: long-payload DoS
    against ``int()`` is mitigated by stdlib, not our code.

    A short-but-large all-digits payload (still under the digit cap) parses
    successfully into an int but now exceeds MAX_CURSOR_OFFSET, so
    decode_cursor rejects it with the bounds-check message.
    """
    # Under the 4300-digit cap: ``int()`` succeeds, but the value is far
    # above MAX_CURSOR_OFFSET, so decode_cursor now raises 'exceeds maximum'.
    short_digits = ("1" * 4000).encode()
    cursor = base64.urlsafe_b64encode(short_digits).decode()
    with pytest.raises(ValueError, match="exceeds maximum"):
        decode_cursor(cursor)

    # 100k digits — over the int-parse cap. Must raise, not hang.
    payload = ("1" * 100_000).encode()
    cursor = base64.urlsafe_b64encode(payload).decode()
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor(cursor)

    # Base64 of pure garbage of similar size — must also raise, not hang.
    garbage = base64.urlsafe_b64encode(b"\x00abc!" * 20_000).decode()
    with pytest.raises(ValueError, match="invalid cursor"):
        decode_cursor(garbage)


# ---------------------------------------------------------------------------
# Bug hotspot #5 — decode_cursor now enforces upper bound on offsets
# ---------------------------------------------------------------------------


def test_huge_offset_now_rejected() -> None:
    """Hotspot #5 (Bug #6) fix: ``decode_cursor`` now rejects offsets above
    MAX_CURSOR_OFFSET (10^9). A client-supplied cursor of base64('100000000000')
    used to round-trip verbatim and force SQLite into an expensive full-index
    walk via the multi-billion OFFSET. After the fix it raises ValueError
    before the SQL ever runs.
    """
    cursor = base64.urlsafe_b64encode(b"100000000000").decode()
    with pytest.raises(ValueError, match="exceeds maximum"):
        decode_cursor(cursor)


def test_negative_offset_now_rejected() -> None:
    """Companion to #5: negative offsets are now rejected by decode_cursor."""
    cursor = base64.urlsafe_b64encode(b"-1").decode()
    with pytest.raises(ValueError, match="negative offset"):
        decode_cursor(cursor)


def test_offset_at_cap_still_works() -> None:
    """The cap is inclusive: MAX_CURSOR_OFFSET itself round-trips fine."""
    assert decode_cursor(encode_cursor(MAX_CURSOR_OFFSET)) == MAX_CURSOR_OFFSET


def test_offset_one_past_cap_rejected() -> None:
    """One past the cap raises — bounds check is strictly > MAX_CURSOR_OFFSET."""
    cursor = base64.urlsafe_b64encode(str(MAX_CURSOR_OFFSET + 1).encode()).decode()
    with pytest.raises(ValueError, match="exceeds maximum"):
        decode_cursor(cursor)


# ---------------------------------------------------------------------------
# Bug hotspot #1 — has_more boundary in list_notes
#
# These mock the inside of ``list_notes`` by patching ``sqlite3.connect`` via
# ``apple_notes_brain.sqlite_reader._open`` so we can drive the row count
# returned by the data SELECT. We don't try to validate field shapes — just
# the (rows, has_more, next_cursor) tuple shape relative to the row count.
# ---------------------------------------------------------------------------


def _fake_row(pk: int) -> tuple:
    """A 9-tuple matching the SELECT in list_notes (data_sql)."""
    # (pk, title, folder_pk, modified, pinned, lock_pp, lock_iv, lock_mode, shared)
    return (pk, f"Note {pk}", 1, 0.0, 0, 0, None, None, 0)


class _FakeConn:
    """A minimal sqlite3.Connection stand-in for list_notes.

    list_notes calls (in order):
      1. _select_optional(...)  — runs PRAGMA table_info; we need columns to
         exist so the optional column helper picks the real name. Easiest path:
         claim NO optional columns exist (returns the fallback alias).
      2. _live_folder_subquery(conn) — inspects schema; returns SQL string.
      3. trash_folder_pks() — opens its OWN connection, so we patch it.
      4. count_sql query — returns (total,).
      5. data_sql query — returns ``rows``.
    """

    def __init__(self, rows: list[tuple], total: int) -> None:
        self._rows = rows
        self._total = total
        self._call_count = 0

    def execute(self, sql: str, params: list | tuple = ()) -> "_FakeCursor":
        # PRAGMA table_info — return empty so optional columns fall back to
        # their literal alias (e.g. "0 AS pinned").
        if "PRAGMA" in sql or "table_info" in sql:
            return _FakeCursor([])
        # _live_folder_subquery uses an EXISTS check — return a row so it
        # picks the LIVE-folder branch.
        if "sqlite_master" in sql.lower():
            return _FakeCursor([(1,)])
        # COUNT(*) query
        if sql.strip().upper().startswith("SELECT COUNT"):
            return _FakeCursor([(self._total,)])
        # Data query — return the configured rows
        return _FakeCursor(self._rows)

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple]:
        return self._rows

    def __iter__(self):
        return iter(self._rows)


@pytest.fixture
def patch_list_notes_internals(mocker):
    """Patch the internals list_notes touches so it returns whatever rows
    the test configures, without hitting the real Notes SQLite."""

    def _setup(rows: list[tuple], total: int = 0) -> None:
        fake = _FakeConn(rows=rows, total=total)
        mocker.patch(
            "apple_notes_brain.sqlite_reader._open",
            return_value=fake,
        )
        # _select_optional inspects schema separately — short-circuit it.
        mocker.patch(
            "apple_notes_brain.sqlite_reader._select_optional",
            side_effect=lambda conn, table, col, alias, table_alias=None: f"0 AS {alias}",
        )
        mocker.patch(
            "apple_notes_brain.sqlite_reader._live_folder_subquery",
            return_value="SELECT 1",
        )
        mocker.patch(
            "apple_notes_brain.sqlite_reader.trash_folder_pks",
            return_value=set(),
        )

    return _setup


def test_bug_has_more_false_when_exactly_limit_rows(patch_list_notes_internals) -> None:
    """Hotspot #1 (sqlite_reader.py:833-835): when the DB has *exactly*
    ``limit`` rows starting at this offset, ``fetch_limit = limit + 1``
    returns ``limit`` rows (not ``limit+1``), so ``has_more`` must be False
    and ``next_cursor`` must be None.
    """
    from apple_notes_brain.sqlite_reader import list_notes

    limit = 50
    # Fake DB returns exactly `limit` rows for the data query.
    patch_list_notes_internals([_fake_row(i) for i in range(limit)], total=limit)

    rows, has_more, next_cursor, total = list_notes(
        folder_pks=None, limit=limit, cursor=None
    )

    assert len(rows) == limit
    assert has_more is False
    assert next_cursor is None
    assert total == limit


def test_bug_has_more_true_when_limit_plus_one_rows(patch_list_notes_internals) -> None:
    """Hotspot #1: when the DB returns ``limit + 1`` rows (the sentinel that
    says 'there's at least one more page'), ``has_more`` must be True and
    the returned rows must be sliced down to ``limit``.
    """
    from apple_notes_brain.sqlite_reader import list_notes

    limit = 10
    patch_list_notes_internals([_fake_row(i) for i in range(limit + 1)], total=100)

    rows, has_more, next_cursor, _ = list_notes(
        folder_pks=None, limit=limit, cursor=None
    )

    assert len(rows) == limit  # truncated
    assert has_more is True
    assert next_cursor is not None
    # next_cursor must decode to offset + limit = 0 + 10 = 10
    assert decode_cursor(next_cursor) == limit


def test_bug_has_more_false_when_fewer_than_limit_rows(patch_list_notes_internals) -> None:
    """Hotspot #1: under-full page (fewer than ``limit`` rows) → terminal
    page, no next cursor."""
    from apple_notes_brain.sqlite_reader import list_notes

    limit = 50
    patch_list_notes_internals([_fake_row(i) for i in range(7)], total=7)

    rows, has_more, next_cursor, _ = list_notes(
        folder_pks=None, limit=limit, cursor=None
    )

    assert len(rows) == 7
    assert has_more is False
    assert next_cursor is None


def test_bug_empty_result_set(patch_list_notes_internals) -> None:
    """Hotspot #1 corner: zero rows. has_more=False, next_cursor=None."""
    from apple_notes_brain.sqlite_reader import list_notes

    patch_list_notes_internals([], total=0)
    rows, has_more, next_cursor, total = list_notes(
        folder_pks=None, limit=25, cursor=None
    )
    assert rows == []
    assert has_more is False
    assert next_cursor is None
    assert total == 0


# ---------------------------------------------------------------------------
# Bug hotspot #2 — encode_cursor(offset + limit) on (near-)final page
# ---------------------------------------------------------------------------


def test_bug_no_overflow_on_final_page(patch_list_notes_internals) -> None:
    """Hotspot #2 (sqlite_reader.py:837): ``encode_cursor(offset + limit)``
    is only invoked when ``has_more=True``. Verify that on the exact-fit
    final page (offset=950, limit=50, exactly 50 rows returned), no cursor
    is produced — guarding against an overflow / phantom-page bug.
    """
    from apple_notes_brain.sqlite_reader import list_notes

    offset = 950
    limit = 50
    cursor = encode_cursor(offset)
    # DB returns exactly 50 rows for offset=950, limit=51
    patch_list_notes_internals([_fake_row(i) for i in range(limit)], total=1000)

    _, has_more, next_cursor, _ = list_notes(
        folder_pks=None, limit=limit, cursor=cursor
    )

    assert has_more is False
    assert next_cursor is None  # critical: no phantom 1000-offset cursor


def test_bug_next_cursor_is_offset_plus_limit(patch_list_notes_internals) -> None:
    """Hotspot #2: when has_more=True, next_cursor must encode exactly
    ``offset + limit``. Anything else is a pagination skip / dup bug."""
    from apple_notes_brain.sqlite_reader import list_notes

    offset = 100
    limit = 20
    patch_list_notes_internals([_fake_row(i) for i in range(limit + 1)], total=10_000)

    _, has_more, next_cursor, _ = list_notes(
        folder_pks=None, limit=limit, cursor=encode_cursor(offset),
    )

    assert has_more is True
    assert next_cursor is not None
    assert decode_cursor(next_cursor) == offset + limit


# ---------------------------------------------------------------------------
# Bug hotspot #6 — note: LIMIT/OFFSET without max-offset validation
# ---------------------------------------------------------------------------


def test_max_offset_validation_now_enforced() -> None:
    """Hotspot #6 fix: decode_cursor now enforces an upper bound on offsets
    so a malicious cursor cannot force SQLite into a full-index walk.

    encode_cursor(10**18) succeeds (the producer-side check only rejects
    negatives), but decode_cursor refuses to return that value.
    """
    huge = 10**18
    cursor = encode_cursor(huge)
    with pytest.raises(ValueError, match="exceeds maximum"):
        decode_cursor(cursor)


# ---------------------------------------------------------------------------
# Hypothesis property — round-trip preservation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(offset=st.integers(min_value=0, max_value=10**9))
@settings(max_examples=200, deadline=None)
def test_property_roundtrip_preserves_nonneg_offset(offset: int) -> None:
    """For any non-negative integer ≤ 1e9, ``decode(encode(o)) == o``."""
    assert decode_cursor(encode_cursor(offset)) == offset


@pytest.mark.property
@given(offset=st.integers(min_value=0, max_value=10**6))
@settings(max_examples=100, deadline=None)
def test_property_encoded_cursor_is_valid_base64(offset: int) -> None:
    """The encoded cursor must always decode under stdlib urlsafe_b64decode."""
    encoded = encode_cursor(offset)
    base64.urlsafe_b64decode(encoded.encode())  # must not raise


@pytest.mark.property
@given(offset=st.integers(min_value=MAX_CURSOR_OFFSET + 1, max_value=10**18))
@settings(max_examples=100, deadline=None)
def test_property_above_cap_offset_rejected(offset: int) -> None:
    """For any int > MAX_CURSOR_OFFSET, decode_cursor must raise ValueError.

    Build the cursor manually (bypassing encode_cursor, which only rejects
    negatives) so we exercise decode_cursor's upper-bound check directly.
    """
    cursor = base64.urlsafe_b64encode(str(offset).encode()).decode()
    with pytest.raises(ValueError, match="exceeds maximum"):
        decode_cursor(cursor)


@pytest.mark.property
@given(offset=st.integers(min_value=-(10**9), max_value=-1))
@settings(max_examples=50, deadline=None)
def test_property_negative_offset_rejected(offset: int) -> None:
    """For any negative int, decode_cursor must raise ValueError."""
    cursor = base64.urlsafe_b64encode(str(offset).encode()).decode()
    with pytest.raises(ValueError, match="negative offset"):
        decode_cursor(cursor)
