"""Reader for Apple Notes' protobuf body in ZICCLOUDSYNCINGOBJECT.ZMERGEABLEDATA1.

Recovers checklist state, code-block style runs, and headings — fidelity that
the AppleScript HTML view loses.
"""
from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass

log = logging.getLogger("apple-notes-brain.protobuf")

# Style-type constants from threeplanetssoftware/apple_cloud_notes_parser
STYLE_TYPE_DEFAULT = -1
STYLE_TYPE_TITLE = 0
STYLE_TYPE_HEADING = 1
STYLE_TYPE_SUBHEADING = 2
STYLE_TYPE_MONOSPACED = 4
STYLE_TYPE_DOTTED_LIST = 100
STYLE_TYPE_DASHED_LIST = 101
STYLE_TYPE_NUMBERED_LIST = 102
STYLE_TYPE_CHECKBOX = 103


@dataclass(frozen=True)
class StyleRun:
    offset: int            # character offset in the plain-text note body
    length: int
    style_type: int        # one of the STYLE_TYPE_* constants, or DEFAULT (-1)
    is_checked: bool | None  # True/False for CHECKBOX runs; None for non-checkbox


def decode_note_protobuf(blob: bytes):
    """Gunzip + parse the Apple Notes protobuf body. Returns the parsed Note message,
    or None on any error (corrupted blob, schema drift, encrypted note)."""
    if not blob:
        return None
    try:
        # The blob is gzip-compressed protobuf
        if blob[:2] == b"\x1f\x8b":
            data = gzip.decompress(blob)
        else:
            data = blob
        from .proto import notestore_pb2  # type: ignore
        # Top-level message: NoteStoreProto with `document` field
        proto = notestore_pb2.NoteStoreProto()
        proto.ParseFromString(data)
        # Walk to the Note: proto.document.note
        return proto.document.note
    except Exception as exc:
        log.debug("decode_note_protobuf: %s", exc)
        return None


def extract_style_runs(note) -> list[StyleRun]:
    """Walk the note's attribute_run list and produce StyleRun records.

    Each run carries its style_type and (for checkbox runs) the checked flag.
    Returns an empty list on any failure."""
    if note is None:
        return []
    runs: list[StyleRun] = []
    offset = 0
    try:
        for run in note.attribute_run:
            length = int(getattr(run, "length", 0))
            style_type = STYLE_TYPE_DEFAULT
            is_checked: bool | None = None
            ps = getattr(run, "paragraph_style", None)
            if ps is not None and run.HasField("paragraph_style"):
                if ps.HasField("style_type"):
                    style_type = int(ps.style_type)
                if style_type == STYLE_TYPE_CHECKBOX and ps.HasField("checklist"):
                    checklist = ps.checklist
                    is_checked = bool(getattr(checklist, "done", 0))
            runs.append(
                StyleRun(
                    offset=offset,
                    length=length,
                    style_type=style_type,
                    is_checked=is_checked,
                )
            )
            offset += length
    except Exception as exc:
        log.debug("extract_style_runs: %s", exc)
    return runs


def extract_plain_text(note) -> str:
    """Extract the concatenated plain text from the note. Useful as the source
    of truth for char-offset → line mapping."""
    if note is None:
        return ""
    try:
        return getattr(note, "note_text", "") or ""
    except Exception:
        return ""
