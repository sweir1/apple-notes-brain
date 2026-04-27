"""Unit tests for `apple_notes_brain.html_validate`.

Covers ``normalize_html(html) -> str`` and the module's behaviour around its
allow-list / strip-list / unwrap-list / tag-normalize map / event-attr regex.

Behaviour pinned by these tests (verified against the implementation):

* Empty / whitespace-only input → ``""``.
* ``_STRIP_COMPLETE`` tags (``script style iframe embed form input button
  link``) are removed along with their text content.
* ``_UNWRAP`` tags (``span font``) are removed but their text content is
  preserved.
* ``_TAG_NORMALIZE`` rewrites: ``strong→b``, ``em→i``, ``del→strike``,
  ``s→strike``, ``p→div``.
* ``h4`` through ``h9`` are downgraded to ``div`` (``h1 h2 h3`` keep their
  level).
* Any remaining tag not in ``_ALLOWED`` is unwrapped (content kept).
* All ``on*`` event-handler attributes are stripped from every tag, but
  ``href`` is preserved as-is — including ``javascript:`` / ``data:``
  schemes (sanitization here is not URL-aware; that's a known scope).
* ``normalize_html`` never raises on malformed input; broken nesting and
  unclosed tags are best-effort cleaned by lxml.
* The function is idempotent (running it twice on its own output is a
  no-op).
"""
from __future__ import annotations

import pytest

from apple_notes_brain.html_validate import (
    _ALLOWED,
    _EVENT_ATTR,
    _STRIP_COMPLETE,
    _TAG_NORMALIZE,
    _UNWRAP,
    normalize_html,
)


# ---------------------------------------------------------------------------
# Empty / trivial input
# ---------------------------------------------------------------------------


def test_normalize_html_empty_string_returns_empty() -> None:
    assert normalize_html("") == ""


def test_normalize_html_whitespace_only_returns_empty() -> None:
    assert normalize_html("   \n\t  ") == ""


def test_normalize_html_plain_text_unchanged() -> None:
    # No tags → returns the text wrapped by the parser. lxml may add a
    # wrapper but the text content must round-trip.
    out = normalize_html("hello")
    assert "hello" in out


# ---------------------------------------------------------------------------
# Allowed tags pass through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html_in",
    [
        "<b>bold</b>",
        "<i>italic</i>",
        "<u>underline</u>",
        "<h1>h1</h1>",
        "<h2>h2</h2>",
        "<h3>h3</h3>",
        "<ul><li>item</li></ul>",
        "<ol><li>item</li></ol>",
        "<div>block</div>",
        "<blockquote>quoted</blockquote>",
        "<pre><code>code</code></pre>",
        "<table><tr><td>cell</td></tr></table>",
    ],
)
def test_normalize_html_allowed_tags_preserved(html_in: str) -> None:
    out = normalize_html(html_in)
    # The first-named tag in the input is in _ALLOWED → it should appear
    # somewhere in the output.
    first_tag = html_in.split(">", 1)[0].lstrip("<")
    assert f"<{first_tag}" in out


def test_normalize_html_anchor_with_href_preserved() -> None:
    out = normalize_html('<a href="https://example.com">link</a>')
    assert '<a href="https://example.com">link</a>' in out


# ---------------------------------------------------------------------------
# _STRIP_COMPLETE — remove tag + content
# ---------------------------------------------------------------------------


# `<input>` and `<link>` are HTML void elements — lxml does not allow them
# to wrap content. For those, exercise the "tag is removed" path with
# self-closing markup; the others get the full content-removal path.
_VOID_STRIP_TAGS = {"input", "link"}


@pytest.mark.parametrize(
    "tag", sorted(_STRIP_COMPLETE - _VOID_STRIP_TAGS),
)
def test_normalize_html_strip_complete_removes_tag_and_content(tag: str) -> None:
    html_in = f"<{tag}>danger</{tag}>after"
    out = normalize_html(html_in)
    assert "danger" not in out
    assert f"<{tag}" not in out
    assert "after" in out


@pytest.mark.parametrize("tag", sorted(_VOID_STRIP_TAGS))
def test_normalize_html_strip_complete_void_tag_removed(tag: str) -> None:
    # Void tags carry no content of their own; just verify the tag itself
    # disappears and surrounding text survives.
    out = normalize_html(f"before<{tag}/>after")
    assert f"<{tag}" not in out
    assert "before" in out
    assert "after" in out


def test_normalize_html_script_alert_removed() -> None:
    out = normalize_html("<script>alert(1)</script>after")
    assert "alert" not in out
    assert "after" in out


def test_normalize_html_style_block_removed() -> None:
    out = normalize_html("<style>.x{color:red}</style>visible")
    assert "color:red" not in out
    assert "visible" in out


# ---------------------------------------------------------------------------
# _UNWRAP — remove tag, keep content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag", sorted(_UNWRAP))
def test_normalize_html_unwrap_keeps_content(tag: str) -> None:
    html_in = f"<{tag}>kept</{tag}>"
    out = normalize_html(html_in)
    assert "kept" in out
    assert f"<{tag}" not in out


def test_normalize_html_span_unwrapped() -> None:
    out = normalize_html('<span style="color:red">text</span>')
    assert "<span" not in out
    assert "text" in out


def test_normalize_html_font_unwrapped() -> None:
    assert normalize_html("<font>hi</font>") == "hi"


# ---------------------------------------------------------------------------
# _TAG_NORMALIZE — rewrite mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new",
    sorted(_TAG_NORMALIZE.items()),
)
def test_normalize_html_tag_normalize(old: str, new: str) -> None:
    html_in = f"<{old}>x</{old}>"
    out = normalize_html(html_in)
    assert f"<{new}>" in out
    assert f"<{old}>" not in out


def test_normalize_html_strong_becomes_b() -> None:
    assert normalize_html("<strong>bold</strong>") == "<b>bold</b>"


def test_normalize_html_em_becomes_i() -> None:
    assert normalize_html("<em>i</em>") == "<i>i</i>"


def test_normalize_html_p_becomes_div() -> None:
    assert normalize_html("<p>para</p>") == "<div>para</div>"


def test_normalize_html_del_and_s_become_strike() -> None:
    assert normalize_html("<del>d</del>") == "<strike>d</strike>"
    assert normalize_html("<s>s</s>") == "<strike>s</strike>"


# ---------------------------------------------------------------------------
# h4-h9 → div downgrade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [4, 5, 6, 7, 8, 9])
def test_normalize_html_h4_through_h9_downgraded_to_div(level: int) -> None:
    out = normalize_html(f"<h{level}>x</h{level}>")
    assert "<div>x</div>" in out
    assert f"<h{level}" not in out


@pytest.mark.parametrize("level", [1, 2, 3])
def test_normalize_html_h1_h2_h3_preserved(level: int) -> None:
    html_in = f"<h{level}>x</h{level}>"
    out = normalize_html(html_in)
    assert f"<h{level}>" in out


# ---------------------------------------------------------------------------
# Event-handler attribute stripping
# ---------------------------------------------------------------------------


def test_normalize_html_strips_onclick() -> None:
    out = normalize_html('<a href="x" onclick="evil()">link</a>')
    assert "onclick" not in out
    assert 'href="x"' in out


def test_normalize_html_strips_multiple_event_handlers() -> None:
    out = normalize_html('<div onmouseover="x" onclick="y" onload="z">a</div>')
    assert "onmouseover" not in out
    assert "onclick" not in out
    assert "onload" not in out
    assert "a" in out


@pytest.mark.parametrize(
    "attr",
    ["onclick", "onmouseover", "onload", "onerror", "onfocus", "onblur"],
)
def test_normalize_html_event_attr_regex_matches(attr: str) -> None:
    assert _EVENT_ATTR.match(attr) is not None


@pytest.mark.parametrize(
    "attr",
    ["href", "src", "class", "id", "style", "alt", "title"],
)
def test_normalize_html_event_attr_regex_does_not_match_normal_attrs(attr: str) -> None:
    assert _EVENT_ATTR.match(attr) is None


def test_normalize_html_uppercase_event_attr_also_stripped() -> None:
    # _EVENT_ATTR uses re.IGNORECASE.
    out = normalize_html('<div ONCLICK="x">y</div>')
    assert "onclick" not in out.lower()


# ---------------------------------------------------------------------------
# URL scheme handling — documents current (intentionally permissive) behaviour
# ---------------------------------------------------------------------------


def test_normalize_html_javascript_href_preserved_as_is() -> None:
    # Documents current scope: normalize_html does NOT URL-sanitize.
    # If that ever changes, update this test.
    out = normalize_html('<a href="javascript:alert(1)">x</a>')
    assert 'href="javascript:alert(1)"' in out


def test_normalize_html_data_url_href_preserved() -> None:
    out = normalize_html('<a href="data:image/png;base64,xxx">x</a>')
    assert "data:image/png" in out


# ---------------------------------------------------------------------------
# Unknown tags get unwrapped
# ---------------------------------------------------------------------------


def test_normalize_html_unknown_tag_unwrapped() -> None:
    out = normalize_html("<custom>kept</custom>")
    assert "<custom" not in out
    assert "kept" in out


def test_normalize_html_marquee_unwrapped() -> None:
    out = normalize_html("<marquee>scroll</marquee>")
    assert "<marquee" not in out
    assert "scroll" in out


# ---------------------------------------------------------------------------
# Malformed input doesn't crash
# ---------------------------------------------------------------------------


def test_normalize_html_unclosed_tag_does_not_raise() -> None:
    out = normalize_html("<unclosed>x")
    assert "x" in out


def test_normalize_html_broken_nesting_does_not_raise() -> None:
    # lxml's parser repairs as best it can; we just need no exception and
    # the surviving content to be reachable.
    out = normalize_html("<p><b></p></b>")
    assert isinstance(out, str)


def test_normalize_html_invalid_attribute_syntax() -> None:
    # No exception on weird attribute markup.
    out = normalize_html("<a href=>x</a>")
    assert "x" in out


# ---------------------------------------------------------------------------
# Edge cases — comments, CDATA, self-closing
# ---------------------------------------------------------------------------


def test_normalize_html_self_closing_br_preserved() -> None:
    out = normalize_html("a<br/>b")
    assert "<br" in out
    assert "a" in out and "b" in out


def test_normalize_html_html_comment_preserved() -> None:
    # bs4/lxml retains comment nodes by default; this documents the behaviour.
    out = normalize_html("<!--evil-->visible")
    assert "visible" in out


# ---------------------------------------------------------------------------
# Nested combinations
# ---------------------------------------------------------------------------


def test_normalize_html_nested_strip_inside_unwrap() -> None:
    # <span>(unwrap) > <script>(strip) — content of script gone, span unwrapped.
    out = normalize_html("<span><script>evil</script>safe</span>")
    assert "evil" not in out
    assert "safe" in out
    assert "<span" not in out
    assert "<script" not in out


def test_normalize_html_nested_normalize_inside_allowed() -> None:
    out = normalize_html("<div><strong>bold</strong></div>")
    assert "<div>" in out
    assert "<b>bold</b>" in out
    assert "<strong" not in out


def test_normalize_html_complex_apple_input() -> None:
    apple = (
        '<div><font face="Helvetica"><strong>Title</strong></font></div>'
        "<p>body <em>emph</em></p>"
        '<script>tracker()</script>'
    )
    out = normalize_html(apple)
    assert "<font" not in out
    assert "<strong" not in out
    assert "<em" not in out
    assert "<script" not in out
    assert "tracker" not in out
    # Content survives.
    assert "Title" in out
    assert "body" in out
    assert "emph" in out


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html_in",
    [
        "<b>bold</b>",
        "<p><strong>x</strong></p>",
        "<div><font>y</font></div>",
        "<ul><li>a</li><li>b</li></ul>",
        '<a href="x" onclick="y">z</a>',
        "<h5>downgrade me</h5>",
        "<custom>unknown</custom>",
        "",
        "plain text",
    ],
)
def test_normalize_html_idempotent(html_in: str) -> None:
    once = normalize_html(html_in)
    twice = normalize_html(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Module-level constant sanity
# ---------------------------------------------------------------------------


def test_strip_complete_and_unwrap_disjoint() -> None:
    # A tag should not be both stripped and unwrapped.
    assert _STRIP_COMPLETE.isdisjoint(_UNWRAP)


def test_strip_complete_and_allowed_disjoint() -> None:
    # A dangerous tag must not also be allowed.
    assert _STRIP_COMPLETE.isdisjoint(_ALLOWED)


def test_tag_normalize_targets_are_allowed() -> None:
    # Every rewrite target must be in the allow-list, otherwise the
    # subsequent pass would unwrap it.
    for target in _TAG_NORMALIZE.values():
        assert target in _ALLOWED, f"normalize target {target!r} not in _ALLOWED"
