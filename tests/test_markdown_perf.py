"""Regression tests for markdown converter performance + safety.

Catches the kind of bug the v7 audit hit: a regex sub loop that never converges
and pins CPU at 100%. Each test asserts conversion completes within a tight
wall-clock budget. The 60s decorator backstop is in markdown.py; these tests
catch pathologies long before that fires.
"""
import time

import pytest

from apple_notes_brain.markdown import html_to_markdown, markdown_to_html


WALL_CLOCK_BUDGET_S = 2.0  # Generous; healthy conversions are <100ms.


@pytest.mark.parametrize(
    "label,markdown_input",
    [
        # The exact v7 audit case that hung indefinitely.
        ("v7-audit-ordered", "Intro paragraph.\n\n1. First item\n2. Second item\n3. Third item\n\nEnd paragraph."),
        # v6 audit case: paragraph immediately followed by bullets (no blank).
        ("v6-audit-unordered", "Para text.\n- item 1\n- item 2"),
        # Same shape with ordered list (would have hung if regex still buggy).
        ("ordered-no-blank", "Para text.\n1. one\n2. two"),
        # Mixed paragraph -> ordered -> paragraph -> unordered.
        ("mixed-blocks", "# Heading\n\nIntro.\n- a\n- b\n\nMore.\n1. c\n2. d\n\nEnd."),
        # Checklist markdown.
        ("checklist", "Tasks:\n- [ ] todo one\n- [x] done two\n- [ ] todo three"),
        # Long ordered list — adversarial input shape.
        ("long-ordered", "\n".join(f"{i}. Item {i}" for i in range(1, 200))),
        # Long unordered list.
        ("long-unordered", "\n".join(f"- Item {i}" for i in range(200))),
        # Paragraph followed by long list (forces preprocess).
        ("para-then-long", "Intro.\n" + "\n".join(f"{i}. Item {i}" for i in range(1, 100))),
        # Huge plain text.
        ("huge-text", "This is a paragraph. " * 10_000),
        # Deeply nested-looking inputs.
        ("many-headings", "\n".join(f"{'#' * (i % 6 + 1)} Header {i}" for i in range(200))),
        # Empty / whitespace.
        ("empty", ""),
        ("whitespace-only", "   \n\n\t\n"),
    ],
)
def test_markdown_to_html_does_not_hang(label, markdown_input):
    t0 = time.monotonic()
    out = markdown_to_html(markdown_input)
    elapsed = time.monotonic() - t0
    assert elapsed < WALL_CLOCK_BUDGET_S, f"{label}: took {elapsed:.2f}s (budget {WALL_CLOCK_BUDGET_S}s)"
    assert isinstance(out, str)


@pytest.mark.parametrize(
    "label,html_input",
    [
        # Apple's heading-as-bold-span shape (the v6 round-trip case).
        ("apple-h1", '<div><b><span style="font-size: 24px">Title</span></b></div>'),
        ("apple-h2", '<div><b><span style="font-size: 18px">Subtitle</span></b></div>'),
        # Apple's Courier code shape.
        ("apple-inline-code", '<font face="Courier"><span style="font-size: 12px">code</span></font>'),
        ("apple-code-block", '<div><font face="Courier"><tt>line one</tt></font></div><div><font face="Courier"><tt>line two</tt></font></div>'),
        # Apple's table-as-object shape.
        ("apple-table", '<table><tr><td><b>H1</b></td><td><b>H2</b></td></tr><tr><td>a</td><td>b</td></tr></table>'),
        # Strike.
        ("strike", "<strike>deleted</strike> and <s>also</s>"),
        # Link with stripped href.
        ("link", '<u>visible text</u>'),
        # Pathological: deeply nested divs.
        ("deep-divs", "<div>" * 500 + "text" + "</div>" * 500),
        # Long flat list.
        ("long-ul", "<ul>" + "".join(f"<li>item {i}</li>" for i in range(500)) + "</ul>"),
        # Mixed real-world shape.
        ("realistic-mix", (
            '<div><b><span style="font-size: 24px">Title</span></b></div>'
            '<div>Para with <b>bold</b> and <i>italic</i> and <strike>strike</strike>.</div>'
            '<ul><li>a</li><li>b</li></ul>'
            '<ol><li>1</li><li>2</li></ol>'
            '<font face="Courier"><span>code</span></font>'
        )),
        # Empty.
        ("empty", ""),
    ],
)
def test_html_to_markdown_does_not_hang(label, html_input):
    t0 = time.monotonic()
    out = html_to_markdown(html_input)
    elapsed = time.monotonic() - t0
    assert elapsed < WALL_CLOCK_BUDGET_S, f"{label}: took {elapsed:.2f}s (budget {WALL_CLOCK_BUDGET_S}s)"
    assert isinstance(out, str)


def test_v7_ordered_list_specific_regression():
    """The exact input from the v7 audit that hung the MCP server."""
    src = "Intro paragraph.\n\n1. First item\n2. Second item\n3. Third item\n\nEnd paragraph."
    out = markdown_to_html(src)
    assert "<ol>" in out
    assert "<li>First item</li>" in out
    assert "<li>Second item</li>" in out
    assert "<li>Third item</li>" in out


def test_v6_unordered_no_blank_regression():
    """v6 audit: paragraph immediately followed by bullets without blank line."""
    out = markdown_to_html("Para text.\n- item 1\n- item 2")
    assert "<ul>" in out
    assert "<li>item 1</li>" in out
    assert "<li>item 2</li>" in out
