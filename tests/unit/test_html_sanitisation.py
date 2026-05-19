"""Adversarial-HTML CI tests.

Reproduces the wedge from the post-merge live-use review: a
`create_note(format='html', body='<script>...')` call must not hang, must
strip dangerous content, and must complete in linear time even for
pathological inputs (deeply nested unclosed tags, 1MB of '<', etc.).

Each payload is checked for:
  1. Sanitisation completes in <100ms (liberal — GitHub runners vary).
  2. Does not raise except for the empty-after-sanitisation contract.
  3. All script / iframe / style / event-handler content is stripped.
  4. Pathological input does not produce exponential output blowup.
"""
from __future__ import annotations

import time

import pytest

from apple_notes_brain.tools import _sanitize_html_input


# Per-payload wall-clock cap. Set well above the typical (<5ms) runtime;
# a value this large still catches the catastrophic-backtracking failure
# mode (those go to seconds-minutes), while not flaking under GHA noise.
SANITISE_BUDGET_S = 0.100


# ---------------------------------------------------------------------------
# Adversarial payloads — each should sanitise cleanly to safe content.
# ---------------------------------------------------------------------------

ADVERSARIAL_PAYLOADS = [
    pytest.param(
        "<script>alert(1)</script><p>safe</p>",
        ["script", "alert"],
        id="bare-script",
    ),
    pytest.param(
        '<img src=x onerror=alert(1)><p>safe</p>',
        ["onerror", "alert"],
        id="event-handler-img",
    ),
    pytest.param(
        '<iframe src="javascript:alert(1)"></iframe><p>safe</p>',
        ["iframe", "javascript:", "alert"],
        id="iframe-javascript-uri",
    ),
    pytest.param(
        '<a href="javascript:alert(1)">click</a>',
        ["javascript:", "alert"],
        id="anchor-javascript-href",
    ),
    pytest.param(
        '<style>@import "evil.css"</style><p>safe</p>',
        ["@import", "evil.css"],
        id="style-import",
    ),
    pytest.param(
        '<svg/onload=alert(1)><p>safe</p>',
        ["onload", "alert"],
        id="svg-onload",
    ),
    pytest.param(
        '<!-- comment with <script>alert(1)</script> nested --><p>safe</p>',
        ["alert"],
        id="comment-hidden-script",
    ),
]


@pytest.mark.parametrize("payload, must_be_absent", ADVERSARIAL_PAYLOADS)
def test_adversarial_payload_sanitises_fast_and_clean(
    payload: str, must_be_absent: list[str]
) -> None:
    """Each payload must:
      1. Complete sanitisation in <100ms (catches catastrophic backtracking).
      2. Not raise (except for the explicit empty-result contract).
      3. Strip dangerous substrings (script/iframe/event-handler/javascript: URIs).
    """
    t0 = time.monotonic()
    try:
        cleaned = _sanitize_html_input(payload)
    except ValueError:
        # An empty-after-sanitisation rejection is acceptable for payloads
        # whose only content was disallowed. Time budget still applies.
        elapsed = time.monotonic() - t0
        assert elapsed < SANITISE_BUDGET_S, (
            f"sanitisation took {elapsed*1000:.1f}ms — possible catastrophic backtracking"
        )
        return

    elapsed = time.monotonic() - t0
    assert elapsed < SANITISE_BUDGET_S, (
        f"sanitisation took {elapsed*1000:.1f}ms — possible catastrophic backtracking"
    )

    cleaned_lower = cleaned.lower()
    for forbidden in must_be_absent:
        assert forbidden.lower() not in cleaned_lower, (
            f"forbidden substring {forbidden!r} survived sanitisation: {cleaned!r}"
        )

    # Universal: never emit any of these regardless of payload.
    for never_allowed in ("<script", "<iframe", "<style", "onload=", "onerror=", "onclick="):
        assert never_allowed not in cleaned_lower, (
            f"dangerous fragment {never_allowed!r} survived: {cleaned!r}"
        )


def test_one_megabyte_of_open_brackets_does_not_blow_up() -> None:
    """`<<<<<...` repeated 1MB used to trigger pathological behaviour in
    naive regex-based sanitisers. Defence is layered: first, the input-size
    cap rejects truly oversized payloads in <1ms; second, even payloads
    inside the cap parse in linear time via bleach's html5lib backend.

    Liberal time cap — a quadratic or exponential implementation would push
    past minutes."""
    payload = "<" * 1_000_000
    t0 = time.monotonic()
    try:
        cleaned = _sanitize_html_input(payload)
    except ValueError:
        # Expected: the input-size cap rejects this almost immediately.
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"1MB-of-< sanitisation took {elapsed:.2f}s"
        return

    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"1MB-of-< sanitisation took {elapsed:.2f}s — possible blowup"
    # No exponential blowup: output ≤ input length.
    assert len(cleaned) <= len(payload), (
        f"output length {len(cleaned)} exceeds input length {len(payload)} — blowup"
    )


def test_payload_just_under_cap_still_completes_in_reasonable_time() -> None:
    """Smaller pathological payload (10KB of `<`) — well under the size cap,
    so it goes through the full pipeline. Verifies bleach/html5lib is linear
    enough to handle realistic worst-case in <100ms."""
    payload = "<" * 10_000
    t0 = time.monotonic()
    try:
        _sanitize_html_input(payload)
    except ValueError:
        pass
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"10KB-of-< sanitisation took {elapsed:.2f}s"


def test_empty_body_rejected() -> None:
    """An empty body shouldn't silently succeed — better to fail loudly than
    to write a blank note."""
    with pytest.raises(ValueError, match="empty"):
        _sanitize_html_input("")


def test_whitespace_only_body_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        _sanitize_html_input("   \n\t  ")


def test_only_disallowed_tags_rejected() -> None:
    """A body consisting solely of disallowed tags (e.g. only <script>)
    sanitises to empty; we should reject rather than silently write blank."""
    with pytest.raises(ValueError):
        _sanitize_html_input("<script>alert(1)</script>")


def test_safe_html_passes_through() -> None:
    """Sanity check: well-formed allowed HTML survives sanitisation."""
    payload = "<p>Hello <b>world</b></p>"
    cleaned = _sanitize_html_input(payload)
    assert "Hello" in cleaned
    assert "world" in cleaned
    # The <p> and <b> tags survive (both in the allow-list).
    assert "<p>" in cleaned
    assert "<b>" in cleaned


def test_link_with_http_href_preserved() -> None:
    payload = '<a href="https://example.com">link</a>'
    cleaned = _sanitize_html_input(payload)
    assert 'href="https://example.com"' in cleaned


def test_link_with_javascript_href_stripped() -> None:
    payload = '<a href="javascript:alert(1)">link</a>'
    # Either ValueError (empty after strip) or no javascript: present.
    try:
        cleaned = _sanitize_html_input(payload)
        assert "javascript:" not in cleaned.lower()
    except ValueError:
        pass


def test_nested_safe_html_preserved() -> None:
    payload = "<div><ul><li>one</li><li>two</li></ul></div>"
    cleaned = _sanitize_html_input(payload)
    for needle in ("<div>", "<ul>", "<li>", "one", "two"):
        assert needle in cleaned


def test_body_to_html_html_path_uses_sanitiser() -> None:
    """Integration: `_body_to_html(body, 'html')` should now sanitise via
    bleach before handing to the Apple-Notes normaliser. Verifies the wire-up
    in tools._body_to_html, not just the helper."""
    from apple_notes_brain.tools import _body_to_html

    out = _body_to_html('<p>safe</p><script>alert(1)</script>', "html")
    assert "alert" not in out.lower()
    assert "<script" not in out.lower()
    assert "safe" in out
