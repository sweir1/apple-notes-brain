"""Snapshot-based regression tests for markdown.py.

Each fixture is a golden HTML or markdown input, and the converter's output
is captured as a syrupy snapshot. Subsequent runs compare against the
recorded snapshot — any change in converter output fails the test.

To regenerate snapshots after intentional changes:
    uv run pytest tests/unit/test_markdown_roundtrip.py --snapshot-update
"""
from pathlib import Path

import pytest

from apple_notes_brain.markdown import html_to_markdown, markdown_to_html

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "markdown"
HTML_INPUTS = sorted((FIXTURES_DIR / "html_inputs").glob("*.html"))
MD_INPUTS = sorted((FIXTURES_DIR / "md_inputs").glob("*.md"))


@pytest.mark.parametrize("html_path", HTML_INPUTS, ids=lambda p: p.stem)
def test_html_to_markdown_snapshot(html_path, snapshot):
    html = html_path.read_text(encoding="utf-8")
    result = html_to_markdown(html)
    assert result == snapshot


@pytest.mark.parametrize("md_path", MD_INPUTS, ids=lambda p: p.stem)
def test_markdown_to_html_snapshot(md_path, snapshot):
    md = md_path.read_text(encoding="utf-8")
    result = markdown_to_html(md)
    assert result == snapshot
