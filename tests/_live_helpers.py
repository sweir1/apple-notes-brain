"""Helpers for ad-hoc live testing against real Notes.app.

These exist because I keep leaving orphan folders when inline test scripts
crash between create and cleanup. Use these from any throwaway probe / test
script to guarantee cleanup runs.

Example:
    from tests._live_helpers import managed_folder, managed_note

    with managed_folder(name='probe-folder') as f:
        with managed_note(title='probe-note', folder_path=f.path) as n:
            # any operation; if it raises, both note and folder are deleted
            ...
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from apple_notes_brain import server


@contextmanager
def managed_folder(name: str, parent_folder_path: str | None = None) -> Iterator:
    """Create a folder; delete it (recursive) on exit, even on exception.

    Yields the MutationResult of create_folder. On exit, attempts a recursive
    delete to clean up the folder AND any descendants/notes added during the
    block. Cleanup errors are swallowed so the original exception (if any)
    isn't masked.
    """
    f = server.create_folder(name=name, parent_folder_path=parent_folder_path)
    try:
        yield f
    finally:
        try:
            server.delete_folder(f.id, recursive=True, note_disposition="trash")
        except Exception:  # noqa: BLE001 — cleanup is best-effort
            pass


@contextmanager
def managed_note(
    title: str,
    body: str = "",
    folder_path: str | None = None,
    format: str = "text",
) -> Iterator:
    """Create a note; trash it on exit, even on exception."""
    n = server.create_note(title=title, body=body, folder_path=folder_path, format=format)
    try:
        yield n
    finally:
        try:
            server.delete_note(n.id)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
