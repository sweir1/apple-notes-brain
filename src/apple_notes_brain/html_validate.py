"""HTML sanitization for Apple Notes body inputs.

Accepts user-supplied HTML and returns a cleaned, Apple-preferred variant
suitable for AppleScript `set body of note`. Never raises — malformed input
is cleaned up best-effort and a DEBUG log line records any transformations.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("apple-notes-brain.html_validate")

# Tags stripped entirely (element + contents)
_STRIP_COMPLETE = {"script", "style", "iframe", "embed", "form", "input", "button", "link"}

# Tags that get unwrapped (content kept, tag removed)
_UNWRAP = {"span", "font"}

# Tag normalization: Apple Notes prefers legacy tags
_TAG_NORMALIZE = {
    "strong": "b",
    "em": "i",
    "del": "strike",
    "s": "strike",
    "p": "div",
}

# Allowed tags after normalization — anything not here will be unwrapped
_ALLOWED = {
    "b", "i", "strike", "u",
    "h1", "h2", "h3",
    "ul", "ol", "li",
    "a", "img",
    "pre", "code",
    "table", "tr", "td", "th", "thead", "tbody",
    "br", "div", "blockquote",
}

# Regex for event handler attributes
_EVENT_ATTR = re.compile(r"^on[a-z]+$", re.IGNORECASE)


def normalize_html(html: str) -> str:
    """Return a sanitized, Apple-preferred HTML variant of `html`.
    Never raises on malformed input — returns best-effort cleanup."""
    if not html or not html.strip():
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        log.warning("bs4 unavailable; returning raw HTML: %s", exc)
        return html
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        log.warning("bs4 parse failed; returning raw HTML: %s", exc)
        return html

    warnings: list[str] = []

    # Strip event-handler attributes on every tag
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if _EVENT_ATTR.match(attr):
                del tag.attrs[attr]
                warnings.append(f"stripped event-handler: {attr}")

    # Strip dangerous tags entirely
    for name in _STRIP_COMPLETE:
        for el in soup.find_all(name):
            el.decompose()
            warnings.append(f"stripped dangerous tag: <{name}>")

    # h4-h9 → div (Apple renders these as plain; keeps content visible)
    for level in range(4, 10):
        for el in soup.find_all(f"h{level}"):
            el.name = "div"
            warnings.append(f"downgraded <h{level}> to <div>")

    # Normalize tags (strong→b etc)
    for old, new in _TAG_NORMALIZE.items():
        for el in soup.find_all(old):
            el.name = new

    # Unwrap tags that should keep content but lose the wrapper
    for name in _UNWRAP:
        for el in soup.find_all(name):
            el.unwrap()

    # Unwrap any remaining tag not in the allow-list
    for tag in list(soup.find_all(True)):
        if tag.name not in _ALLOWED:
            tag.unwrap()
            warnings.append(f"unwrapped unknown tag: <{tag.name}>")

    if warnings:
        log.debug("normalized HTML: %s", "; ".join(warnings[:10]))

    return str(soup)
