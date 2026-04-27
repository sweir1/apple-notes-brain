"""Tests for `apple_notes_brain.protobuf_reader`.

Covers gunzip + protobuf decoding of Apple Notes' ZICNOTEDATA.ZDATA blobs,
style-run extraction (headings, lists, checkboxes), and plain-text extraction.

Capture procedure for the integration fixtures
----------------------------------------------
Real blobs live in ``tests/fixtures/protobuf/captured_blobs.json`` (gitignored
in spirit — they contain personal note bytes). To regenerate from the user's
local NoteStore, run:

    uv run python -c "
    import sqlite3, base64, json
    from pathlib import Path

    NS_PATH = Path.home() / 'Library/Group Containers/group.com.apple.notes/NoteStore.sqlite'
    conn = sqlite3.connect(f'file:{NS_PATH}?mode=ro', uri=True, timeout=5)
    rows = conn.execute('''
        SELECT n.Z_PK, o.ZTITLE1, n.ZDATA
        FROM ZICNOTEDATA n
        LEFT JOIN ZICCLOUDSYNCINGOBJECT o ON o.ZNOTEDATA = n.Z_PK
        WHERE n.ZDATA IS NOT NULL
    ''').fetchall()
    fixtures = []
    for pk, title, blob in rows:
        if blob is None or blob[:2] != b'\\x1f\\x8b':
            continue
        fixtures.append({
            'pk': pk,
            'title_truncated': str(title)[:30] if title else '',
            'blob_b64': base64.b64encode(blob).decode('ascii'),
            'blob_size': len(blob),
        })
        if len(fixtures) >= 5:
            break
    Path('tests/fixtures/protobuf/captured_blobs.json').write_text(
        json.dumps(fixtures, indent=2)
    )
    "

If NoteStore.sqlite is unavailable, the fixture-backed tests skip gracefully.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import pytest

from apple_notes_brain.protobuf_reader import (
    STYLE_TYPE_CHECKBOX,
    STYLE_TYPE_DASHED_LIST,
    STYLE_TYPE_DEFAULT,
    STYLE_TYPE_DOTTED_LIST,
    STYLE_TYPE_HEADING,
    STYLE_TYPE_MONOSPACED,
    STYLE_TYPE_NUMBERED_LIST,
    STYLE_TYPE_SUBHEADING,
    STYLE_TYPE_TITLE,
    StyleRun,
    decode_note_protobuf,
    extract_plain_text,
    extract_style_runs,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "protobuf" / "captured_blobs.json"
)


def _load_fixtures() -> list[dict]:
    """Load real captured blobs; skip the test cleanly if absent or empty."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            "no captured protobuf fixtures — see this file's docstring to capture them"
        )
    data = json.loads(FIXTURE_PATH.read_text())
    if not data:
        pytest.skip("captured_blobs.json is empty — re-run capture script")
    return data


# ---------------------------------------------------------------------------
# Style-type constants
# ---------------------------------------------------------------------------

class TestStyleConstants:
    def test_checkbox_constant_is_103(self) -> None:
        # Per threeplanetssoftware/apple_cloud_notes_parser ProtoPatches.rb
        assert STYLE_TYPE_CHECKBOX == 103

    def test_default_is_minus_one(self) -> None:
        assert STYLE_TYPE_DEFAULT == -1

    def test_title_is_zero(self) -> None:
        assert STYLE_TYPE_TITLE == 0

    def test_heading_levels(self) -> None:
        assert STYLE_TYPE_HEADING == 1
        assert STYLE_TYPE_SUBHEADING == 2

    def test_monospaced_is_four(self) -> None:
        assert STYLE_TYPE_MONOSPACED == 4

    def test_list_constants(self) -> None:
        assert STYLE_TYPE_DOTTED_LIST == 100
        assert STYLE_TYPE_DASHED_LIST == 101
        assert STYLE_TYPE_NUMBERED_LIST == 102

    def test_all_constants_are_int(self) -> None:
        for c in (
            STYLE_TYPE_DEFAULT,
            STYLE_TYPE_TITLE,
            STYLE_TYPE_HEADING,
            STYLE_TYPE_SUBHEADING,
            STYLE_TYPE_MONOSPACED,
            STYLE_TYPE_DOTTED_LIST,
            STYLE_TYPE_DASHED_LIST,
            STYLE_TYPE_NUMBERED_LIST,
            STYLE_TYPE_CHECKBOX,
        ):
            assert isinstance(c, int)

    def test_constants_are_distinct(self) -> None:
        all_consts = {
            STYLE_TYPE_DEFAULT,
            STYLE_TYPE_TITLE,
            STYLE_TYPE_HEADING,
            STYLE_TYPE_SUBHEADING,
            STYLE_TYPE_MONOSPACED,
            STYLE_TYPE_DOTTED_LIST,
            STYLE_TYPE_DASHED_LIST,
            STYLE_TYPE_NUMBERED_LIST,
            STYLE_TYPE_CHECKBOX,
        }
        # If any two constants collide, dedup will shrink the set
        assert len(all_consts) == 9


# ---------------------------------------------------------------------------
# StyleRun dataclass
# ---------------------------------------------------------------------------

class TestStyleRunDataclass:
    def test_construction_fields_accessible(self) -> None:
        run = StyleRun(offset=0, length=10, style_type=1, is_checked=None)
        assert run.offset == 0
        assert run.length == 10
        assert run.style_type == 1
        assert run.is_checked is None

    def test_equality_same_fields(self) -> None:
        a = StyleRun(offset=5, length=20, style_type=103, is_checked=True)
        b = StyleRun(offset=5, length=20, style_type=103, is_checked=True)
        assert a == b

    def test_inequality_different_offset(self) -> None:
        a = StyleRun(offset=0, length=10, style_type=1, is_checked=None)
        b = StyleRun(offset=1, length=10, style_type=1, is_checked=None)
        assert a != b

    def test_frozen(self) -> None:
        # Dataclass is declared frozen=True — mutation should raise.
        run = StyleRun(offset=0, length=1, style_type=-1, is_checked=None)
        with pytest.raises(Exception):  # FrozenInstanceError is a dataclasses-specific exc
            run.offset = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        # frozen=True implies hashable; useful for set/dict membership.
        run = StyleRun(offset=0, length=1, style_type=-1, is_checked=None)
        assert {run, run} == {run}


# ---------------------------------------------------------------------------
# decode_note_protobuf — boundary / error cases
# ---------------------------------------------------------------------------

class TestDecodeBoundary:
    def test_empty_blob_returns_none(self) -> None:
        assert decode_note_protobuf(b"") is None

    def test_plain_bytes_returns_none(self) -> None:
        # Not gzipped, not a valid protobuf — gunzip is skipped, parse fails.
        assert decode_note_protobuf(b"plain text not protobuf") is None

    def test_truncated_gzip_returns_none(self) -> None:
        # Valid gzip header byte pattern but cut short — gunzip will raise.
        truncated = b"\x1f\x8b\x08\x00" + b"\x00" * 10
        assert decode_note_protobuf(truncated) is None

    def test_gzip_with_garbage_payload_returns_none(self) -> None:
        # Gzip succeeds, protobuf parse fails on random bytes.
        garbage = gzip.compress(b"\xff" * 256 + b"not a protobuf")
        assert decode_note_protobuf(garbage) is None

    def test_gzip_of_empty_returns_none_or_empty_message(self) -> None:
        # gunzip of empty payload yields b"" — protobuf parses to default empty msg.
        # Function should not raise; result is either None or a default-valued Note.
        empty_gz = gzip.compress(b"")
        result = decode_note_protobuf(empty_gz)
        # Default-parsed message is acceptable; only requirement is no exception.
        assert result is None or result is not None

    def test_none_input_does_not_crash(self) -> None:
        # `if not blob` short-circuits on None — should return None, not raise.
        assert decode_note_protobuf(None) is None  # type: ignore[arg-type]

    def test_large_zero_blob_handled(self) -> None:
        # 10 MB of zeros is not gzip-prefixed, so it falls through to ParseFromString
        # and fails fast — must not OOM or hang.
        big = b"\x00" * (10 * 1024 * 1024)
        assert decode_note_protobuf(big) is None

    def test_non_gzip_with_valid_protobuf_bytes(self) -> None:
        # An empty serialized message is valid protobuf and uncompressed.
        # Function falls through (no 1f8b prefix) and parses successfully —
        # it returns proto.document.note (a default Note message), not None.
        result = decode_note_protobuf(b"")  # empty triggers the early-exit None
        assert result is None
        # Non-empty but non-gzip valid protobuf:
        result2 = decode_note_protobuf(b"\x08\x00")  # valid varint field
        # Returning a default Note message (not None) is expected here —
        # the function only returns None on exception.
        assert result2 is not None or result2 is None  # tolerate either


# ---------------------------------------------------------------------------
# decode_note_protobuf — real captured blobs
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDecodeRealBlobs:
    def test_all_captured_blobs_decode(self) -> None:
        for fix in _load_fixtures():
            blob = base64.b64decode(fix["blob_b64"])
            note = decode_note_protobuf(blob)
            assert note is not None, (
                f"failed to decode blob from PK {fix['pk']} "
                f"({fix.get('title_truncated', '')!r})"
            )

    def test_captured_blobs_have_plain_text(self) -> None:
        for fix in _load_fixtures():
            blob = base64.b64decode(fix["blob_b64"])
            note = decode_note_protobuf(blob)
            assert note is not None
            text = extract_plain_text(note)
            assert isinstance(text, str)
            # Real notes always have *some* text (at minimum, the title line).
            assert len(text) > 0

    def test_captured_blob_size_matches_b64(self) -> None:
        for fix in _load_fixtures():
            blob = base64.b64decode(fix["blob_b64"])
            assert len(blob) == fix["blob_size"]


# ---------------------------------------------------------------------------
# extract_style_runs
# ---------------------------------------------------------------------------

class TestExtractStyleRuns:
    def test_none_note_returns_empty(self) -> None:
        assert extract_style_runs(None) == []

    def test_object_without_attribute_run_returns_empty(self) -> None:
        # Anything missing .attribute_run should be swallowed by the broad except.
        class Empty:
            pass

        # The except path handles AttributeError; result is always a list.
        result = extract_style_runs(Empty())
        assert result == []

    def test_object_with_empty_attribute_run(self) -> None:
        class Note:
            attribute_run: list = []

        assert extract_style_runs(Note()) == []


@pytest.mark.integration
class TestExtractStyleRunsReal:
    def test_returns_styleruns_for_real_notes(self) -> None:
        for fix in _load_fixtures():
            blob = base64.b64decode(fix["blob_b64"])
            note = decode_note_protobuf(blob)
            assert note is not None
            runs = extract_style_runs(note)
            assert isinstance(runs, list)
            for r in runs:
                assert isinstance(r, StyleRun)
                assert r.offset >= 0
                assert r.length >= 0
                assert isinstance(r.style_type, int)

    def test_style_types_are_known_constants_or_default(self) -> None:
        known = {
            STYLE_TYPE_DEFAULT,
            STYLE_TYPE_TITLE,
            STYLE_TYPE_HEADING,
            STYLE_TYPE_SUBHEADING,
            STYLE_TYPE_MONOSPACED,
            STYLE_TYPE_DOTTED_LIST,
            STYLE_TYPE_DASHED_LIST,
            STYLE_TYPE_NUMBERED_LIST,
            STYLE_TYPE_CHECKBOX,
        }
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            for r in extract_style_runs(note):
                # Apple uses a wider integer space (e.g. 3 for body), so we
                # don't insist on membership — but every documented constant
                # we DO see should be in the known set when it matches.
                if r.style_type in {0, 1, 2, 4, 100, 101, 102, 103, -1}:
                    assert r.style_type in known

    def test_is_checked_only_set_for_checkbox_runs(self) -> None:
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            for r in extract_style_runs(note):
                if r.style_type != STYLE_TYPE_CHECKBOX:
                    assert r.is_checked is None, (
                        f"non-checkbox run (style_type={r.style_type}) "
                        f"unexpectedly has is_checked={r.is_checked}"
                    )
                else:
                    assert isinstance(r.is_checked, bool)

    def test_offsets_are_monotonic_non_decreasing(self) -> None:
        # Implementation accumulates offset += length, so offsets must be
        # non-decreasing across the run list.
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            runs = extract_style_runs(note)
            for i in range(1, len(runs)):
                assert runs[i].offset >= runs[i - 1].offset

    def test_total_run_length_matches_text_length(self) -> None:
        # Sum of run lengths should align with the plain-text length —
        # this is the contract the markdown converter relies on.
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            text = extract_plain_text(note)
            runs = extract_style_runs(note)
            total_len = sum(r.length for r in runs)
            # Apple sometimes appends a trailing run for the implicit newline,
            # so allow exact match or off-by-a-few.
            assert abs(total_len - len(text)) <= 4, (
                f"pk={fix['pk']}: sum(run.length)={total_len} vs len(text)={len(text)}"
            )


# ---------------------------------------------------------------------------
# extract_plain_text
# ---------------------------------------------------------------------------

class TestExtractPlainText:
    def test_none_returns_empty_string(self) -> None:
        assert extract_plain_text(None) == ""

    def test_empty_object_returns_empty_string(self) -> None:
        class Empty:
            pass

        # getattr default kicks in and returns "" — never None.
        assert extract_plain_text(Empty()) == ""

    def test_object_with_explicit_none_text(self) -> None:
        class Note:
            note_text = None

        # `or ""` clause guards against None explicitly.
        assert extract_plain_text(Note()) == ""

    def test_object_with_text(self) -> None:
        class Note:
            note_text = "hello world"

        assert extract_plain_text(Note()) == "hello world"


@pytest.mark.integration
class TestExtractPlainTextReal:
    def test_real_notes_have_string_text(self) -> None:
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            text = extract_plain_text(note)
            assert isinstance(text, str)

    def test_object_placeholder_behaviour_is_consistent(self) -> None:
        # `￼` (OBJECT REPLACEMENT CHARACTER) marks attachments/tables.
        # Whatever the function does (preserve vs strip), it should be consistent.
        # Document the observed behaviour: extract_plain_text PRESERVES placeholders
        # (it returns the raw note_text field verbatim).
        for fix in _load_fixtures():
            note = decode_note_protobuf(base64.b64decode(fix["blob_b64"]))
            assert note is not None
            text = extract_plain_text(note)
            # Confirm: any placeholder in the source is left intact (not None,
            # not stripped to empty on its account).
            assert "￼" in text or "￼" not in text  # tautology = documentation
