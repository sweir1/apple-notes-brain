"""HTML → plaintext and snippet helpers."""
from __future__ import annotations

from html.parser import HTMLParser


_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "tr", "hr",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        text = "".join(self._chunks)
        lines = [ln.rstrip() for ln in text.splitlines()]
        cleaned: list[str] = []
        blank = 0
        for ln in lines:
            if not ln.strip():
                blank += 1
                if blank <= 1:
                    cleaned.append("")
            else:
                blank = 0
                cleaned.append(ln)
        return "\n".join(cleaned).strip()


def html_to_text(html: str) -> str:
    """Return a readable plaintext rendering of `html`."""
    if not html:
        return ""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def snippets(text: str, query: str, window: int = 100, max_spans: int = 3) -> list[str]:
    """Return up to max_spans non-overlapping windows (±window chars) around case-insensitive matches of query in text.

    Rules:
    - Case-insensitive substring match on the full query string.
    - First span always included; subsequent spans must start at a position at least `window` chars after the previous span's end (so they don't visually overlap).
    - Each span is text[max(0, idx-window) : idx+len(query)+window], with newlines → spaces and leading/trailing "…" when truncated.
    - If no match found: return a single-element list with text[:2*window] (or empty list if text is empty).
    - If query is empty: return [] so callers can skip snippet rendering.
    """
    if not query:
        return []
    if not text:
        return []
    window = max(0, window)
    if max_spans < 1:
        return []
    lower_text = text.lower()
    lower_query = query.lower()
    result: list[str] = []
    search_start = 0
    last_end = -1
    while len(result) < max_spans:
        idx = lower_text.find(lower_query, search_start)
        if idx < 0:
            break
        start = max(0, idx - window)
        end = min(len(text), idx + len(query) + window)
        if last_end >= 0 and start < last_end + window:
            search_start = idx + 1
            continue
        raw = text[start:end].replace("\n", " ").replace("\r", " ")
        raw = " ".join(raw.split())
        if start > 0:
            raw = "…" + raw
        if end < len(text):
            raw = raw + "…"
        result.append(raw)
        last_end = end
        search_start = idx + 1
    if not result:
        fallback = text[: 2 * window]
        if not fallback:
            return []
        raw = fallback.replace("\n", " ").replace("\r", " ")
        raw = " ".join(raw.split())
        if len(text) > 2 * window:
            raw = raw + "…"
        return [raw]
    return result


def count_matches(text: str, query: str) -> int:
    """Count case-insensitive, non-overlapping occurrences of query in text. Returns 0 if query empty."""
    if not query or not text:
        return 0
    lower_text = text.lower()
    lower_query = query.lower()
    count = 0
    start = 0
    while True:
        idx = lower_text.find(lower_query, start)
        if idx < 0:
            break
        count += 1
        start = idx + len(lower_query)
    return count
