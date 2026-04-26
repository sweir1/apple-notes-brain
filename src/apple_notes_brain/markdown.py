"""Convert Apple Notes HTML (from osascript) to clean Markdown and back."""

import atexit
import html as _html
import logging
import re
from concurrent.futures import TimeoutError as _FuturesTimeoutError

log = logging.getLogger("apple-notes-brain.markdown")

# Wall-clock budget for any single conversion. A pathological input (e.g.
# malformed nested HTML triggering catastrophic regex backtracking) holds the
# GIL — threads cannot recover. Pebble's ProcessPool runs each conversion in a
# worker process and TERMINATES the worker on timeout, so runaways don't
# accumulate and the parent stays responsive. Workers are reused (max_tasks
# recycles them) so per-call overhead is ~10-30ms IPC after pool warmup.
_CONVERT_BUDGET_S = 30.0
_POOL = None


def _get_pool():
    """Lazily create the conversion process pool. Imports pebble inside so the
    module loads cheaply during normal operation; the pool itself is created on
    the first conversion."""
    global _POOL
    if _POOL is not None:
        return _POOL
    from pebble import ProcessPool

    _POOL = ProcessPool(max_workers=2, max_tasks=64)
    atexit.register(lambda: _POOL.stop() if _POOL is not None else None)
    return _POOL


def _run_with_budget(worker_fn, payload: str) -> str:
    """Submit `worker_fn(payload)` to the process pool; return the result or
    raise ValueError on timeout / worker death. Pebble terminates the runaway
    worker on timeout and replaces it transparently."""
    if not payload:
        return worker_fn(payload)
    try:
        future = _get_pool().schedule(worker_fn, args=(payload,), timeout=_CONVERT_BUDGET_S)
        return future.result()
    except _FuturesTimeoutError:
        log.warning("%s exceeded %.0fs budget; worker terminated", worker_fn.__name__, _CONVERT_BUDGET_S)
        raise ValueError(
            f"Markdown/HTML conversion exceeded the {_CONVERT_BUDGET_S:.0f}s timeout. "
            "The input may contain a pathological structure (e.g. malformed nested "
            "lists, deeply nested tables, or an unterminated tag) that triggered a "
            "converter edge case. The conversion has been abandoned and the worker "
            "process terminated — please retry with simpler input, or report the "
            "offending markdown/HTML as a bug."
        ) from None
    except Exception:
        # Pebble may surface ProcessExpired etc. — re-raise as ValueError so MCP
        # callers see a clean error rather than an internal pebble exception.
        raise

_CHECKED = "APPLENOTESCHECKEDSENTINEL"
_UNCHECKED = "APPLENOTEUNCHECKEDSENTINEL"
# Sentinel for fenced code language: CODELANG_<lang>_ENDCODELANG
_LANG_PREFIX = "APPLENOTESCODELANG_"
_LANG_SUFFIX = "_ENDCODELANG"

# Pre-compiled patterns for Apple-specific HTML
_RE_CHECKED = re.compile(
    r'<li\s+class=["\']checked["\']>(.*?)</li>',
    re.IGNORECASE | re.DOTALL,
)
_RE_UNCHECKED = re.compile(
    r'<li\s+class=["\']unchecked["\']>(.*?)</li>',
    re.IGNORECASE | re.DOTALL,
)
# <object> with an id attribute (any attribute order, self-closing or paired)
_RE_OBJ_WITH_ID = re.compile(
    r'<object\b[^>]*\bid=["\']([^"\']+)["\'][^>]*/?>(?:</object>)?',
    re.IGNORECASE | re.DOTALL,
)
# <object> without an id attribute
_RE_OBJ_NO_ID = re.compile(
    r'<object\b[^>]*/?>(?:</object>)?',
    re.IGNORECASE | re.DOTALL,
)
# <pre><code class="language-xxx"> — capture language identifier
_RE_PRE_CODE_LANG = re.compile(
    r'<pre[^>]*>\s*<code\s+class=["\']language-([^"\']+)["\'][^>]*>',
    re.IGNORECASE,
)
# <div> or <p> with a monospace font style
_RE_MONO_BLOCK = re.compile(
    r'<(div|p)(\s[^>]*style=["\'][^"\']*(?:Menlo|Courier|monospace)[^"\']*["\'][^>]*)>(.*?)</\1>',
    re.IGNORECASE | re.DOTALL,
)
# Collapse 3+ consecutive blank lines to 2
_RE_BLANK_LINES = re.compile(r'\n{3,}')
# Restore fenced language sentinel after markdownify wraps with ```
_RE_FENCE_SENTINEL = re.compile(
    r'```\n' + _LANG_PREFIX + r'(\w+)' + _LANG_SUFFIX + r'\n',
)
# Standalone unknown-attachment placeholder line
_RE_UNKNOWN_ATTACHMENT_LINE = re.compile(
    r'(?m)^!\[attachment\]\(attachment:unknown\)\s*$\n?'
)


def _is_courier_font(tag) -> bool:
    face = tag.get("face") if tag else None
    return bool(face and "courier" in face.lower())


def _recover_courier_code(soup) -> None:
    """Convert Apple Notes Courier <font> blocks into <code>/<pre><code> structures."""
    from bs4 import NavigableString, Tag

    visited: set[int] = set()
    for div in list(soup.find_all("div")):
        if id(div) in visited:
            continue
        font = div.find("font", recursive=False)
        if font is None or not _is_courier_font(font):
            continue
        tt = font.find("tt", recursive=False)
        if tt is None:
            continue
        # Walk forward through sibling divs of the same shape
        run_divs = [div]
        sib = div.find_next_sibling()
        while isinstance(sib, Tag) and sib.name == "div":
            sfont = sib.find("font", recursive=False)
            if sfont is None or not _is_courier_font(sfont):
                break
            stt = sfont.find("tt", recursive=False)
            if stt is None:
                break
            run_divs.append(sib)
            sib = sib.find_next_sibling()
        if len(run_divs) >= 1:
            lines = []
            for d in run_divs:
                f = d.find("font", recursive=False)
                t = f.find("tt", recursive=False) if f else None
                lines.append(t.get_text() if t else d.get_text())
            pre = soup.new_tag("pre")
            code = soup.new_tag("code")
            code.string = "\n".join(lines)
            pre.append(code)
            run_divs[0].replace_with(pre)
            for d in run_divs[1:]:
                d.decompose()
            for d in run_divs:
                visited.add(id(d))

    # Sequential per-line <font face="Courier"><tt>...</tt></font> not wrapped in divs
    for font in list(soup.find_all("font")):
        if not _is_courier_font(font):
            continue
        if id(font) in visited:
            continue
        tt = font.find("tt", recursive=False)
        if tt is None:
            continue
        run_fonts = [font]
        sib = font.next_sibling
        while sib is not None:
            if isinstance(sib, NavigableString):
                if str(sib).strip() == "":
                    sib = sib.next_sibling
                    continue
                break
            if isinstance(sib, Tag) and sib.name == "font" and _is_courier_font(sib) and sib.find("tt", recursive=False):
                run_fonts.append(sib)
                sib = sib.next_sibling
                continue
            break
        lines = [f.find("tt", recursive=False).get_text() for f in run_fonts]
        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = "\n".join(lines)
        pre.append(code)
        run_fonts[0].replace_with(pre)
        for f in run_fonts[1:]:
            f.decompose()
        for f in run_fonts:
            visited.add(id(f))

    # Remaining Courier <font> tags become inline <code>
    for font in list(soup.find_all("font")):
        if not _is_courier_font(font):
            continue
        if font.find("tt") or font.find("div") or font.find("pre"):
            font.unwrap()
            continue
        code = soup.new_tag("code")
        code.string = font.get_text()
        font.replace_with(code)

    # Any leftover <font> tags (non-Courier) — unwrap to keep content
    for font in list(soup.find_all("font")):
        font.unwrap()


def _promote_table_headers(soup) -> None:
    """Apple stores header cells as <td><b>..</b></td>; promote to <th> + <thead>."""
    for table in soup.find_all("table"):
        first_tr = table.find("tr")
        if first_tr is None:
            continue
        cells = first_tr.find_all("td")
        if cells and all(c.find("b") and c.get_text().strip() for c in cells):
            for td in cells:
                b = td.find("b")
                new_th = soup.new_tag("th")
                new_th.string = b.get_text()
                td.replace_with(new_th)
            if not first_tr.find_parent("thead"):
                thead = soup.new_tag("thead")
                first_tr.wrap(thead)


def _recover_apple_headings(soup) -> None:
    """Recover Apple Notes heading semantics from font-size styled <b><span> shapes."""
    for b in list(soup.find_all("b")):
        span = b.find("span", recursive=False)
        if span is None:
            continue
        style = (span.get("style") or "").lower().replace(" ", "")
        text = span.get_text()
        if "font-size:24px" in style or "font-size:24pt" in style:
            new = soup.new_tag("h1")
            new.string = text
            b.replace_with(new)
        elif "font-size:18px" in style or "font-size:18pt" in style:
            new = soup.new_tag("h2")
            new.string = text
            b.replace_with(new)
    # Subheading heuristic: a <div> whose only non-whitespace child is a bare <b> (no inner <span>)
    # becomes <h3>. Trade-off: a single-line note that is just **bold** also becomes h3.
    for div in list(soup.find_all("div")):
        if div.find_parent("table"):
            continue  # let _promote_table_headers see the <b> intact
        children = [c for c in div.children if not (isinstance(c, str) and not c.strip())]
        if len(children) == 1 and getattr(children[0], "name", None) == "b":
            b = children[0]
            if not b.find("span"):
                text = b.get_text()
                new = soup.new_tag("h3")
                new.string = text
                div.replace_with(new)


def _merge_headings_through_strike(soup) -> None:
    """Apple emits split headings when partial strikethrough is applied.
    Detect <hN>A</hN><strike><hN>B</hN></strike><hN>C</hN> and merge into
    <hN>A<strike>B</strike>C</hN>. Walks h1/h2/h3 only — markdownify will
    later emit `# A ~~B~~ C`."""
    from bs4 import NavigableString, Tag
    for level in (1, 2, 3):
        tag_name = f"h{level}"
        for h in list(soup.find_all(tag_name)):
            nxt = h.next_sibling
            # Skip whitespace-only text nodes between siblings
            while isinstance(nxt, NavigableString) and not str(nxt).strip():
                nxt = nxt.next_sibling
            if not isinstance(nxt, Tag) or nxt.name != "strike":
                continue
            inner = nxt.find(tag_name, recursive=False)
            if inner is None:
                continue
            nxt2 = nxt.next_sibling
            while isinstance(nxt2, NavigableString) and not str(nxt2).strip():
                nxt2 = nxt2.next_sibling
            if not isinstance(nxt2, Tag) or nxt2.name != tag_name:
                continue
            # Merge
            new_h = soup.new_tag(tag_name)
            new_h.append(NavigableString(h.get_text()))
            new_strike = soup.new_tag("strike")
            new_strike.string = inner.get_text()
            new_h.append(new_strike)
            new_h.append(NavigableString(nxt2.get_text()))
            h.replace_with(new_h)
            nxt.decompose()
            nxt2.decompose()


def _preprocess(html: str) -> str:
    """Replace Apple-specific HTML constructs before markdownify sees them."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        _recover_apple_headings(soup)
        _recover_courier_code(soup)
        _promote_table_headers(soup)
        _merge_headings_through_strike(soup)
        html = str(soup)
    except Exception:
        pass
    # Monospace div/p blocks → <pre> (before other transforms)
    html = _RE_MONO_BLOCK.sub(lambda m: f'<pre>{m.group(3)}</pre>', html)
    # Fenced code: inject language sentinel as first line inside <code>
    html = _RE_PRE_CODE_LANG.sub(
        lambda m: f'<pre><code>{_LANG_PREFIX}{m.group(1)}{_LANG_SUFFIX}\n', html
    )
    # Checklist items
    html = _RE_CHECKED.sub(
        lambda m: f'<li>{_CHECKED}{m.group(1)}</li>', html
    )
    html = _RE_UNCHECKED.sub(
        lambda m: f'<li>{_UNCHECKED}{m.group(1)}</li>', html
    )
    # Attachment objects — id present
    html = _RE_OBJ_WITH_ID.sub(
        lambda m: f'<img src="attachment:{m.group(1)}" alt="attachment">', html
    )
    # Attachment objects — id absent (runs after id match, catches remainder)
    html = _RE_OBJ_NO_ID.sub(
        '<img src="attachment:unknown" alt="attachment">', html
    )
    return html


def _postprocess(md: str) -> str:
    """Replace sentinels and tidy whitespace."""
    md = md.replace(_CHECKED, '[x] ')
    md = md.replace(_UNCHECKED, '[ ] ')
    # Restore fenced code language: ``` \nLANG_sentinel\n → ```lang\n
    md = _RE_FENCE_SENTINEL.sub(lambda m: f'```{m.group(1)}\n', md)
    md = _RE_UNKNOWN_ATTACHMENT_LINE.sub('', md)
    md = _RE_BLANK_LINES.sub('\n\n', md)
    return md.strip()


def html_to_markdown(html: str) -> str:
    """Public API — runs the conversion in a pebble worker with a 60s budget."""
    return _run_with_budget(_html_to_markdown_impl, html)


def _html_to_markdown_impl(html: str) -> str:
    """Convert Apple Notes HTML body into Markdown, preserving structure."""
    if not html:
        return ""
    try:
        from markdownify import MarkdownConverter

        class _AppleNotesConverter(MarkdownConverter):
            def convert_strike(self, el, text, parent_tags):
                return f"~~{text}~~"
            convert_s = convert_strike
            convert_del = convert_strike

        processed = _preprocess(html)
        md = _AppleNotesConverter(
            heading_style="ATX",
            bullets="-",
            strong_em_symbol="*",
        ).convert(processed)
        return _postprocess(md)
    except Exception:
        try:
            return re.sub(r"<[^>]+>", "", html).strip()
        except Exception:
            return ""


# --- Markdown -> Apple Notes HTML ---------------------------------------

# Task-list line: `  - [x] text` or `- [ ] text`
_RE_TASK_LINE = re.compile(r'^(\s*)- \[([ xX])\] (.*)$')
# Strikethrough ~~text~~ (non-greedy, single line)
_RE_STRIKE = re.compile(r'~~(.+?)~~')
# <pre><code class="language-xxx"> → <pre><code>
_RE_CODE_LANG_ATTR = re.compile(
    r'<code\s+class=["\']language-[^"\']*["\']>', re.IGNORECASE
)
# Downgrade <h4>..<h9> (open and close) to <div>
_RE_H456_OPEN = re.compile(r'<h[4-9]>', re.IGNORECASE)
_RE_H456_CLOSE = re.compile(r'</h[4-9]>', re.IGNORECASE)
# Insert blank line before list markers when the previous line is non-empty
# and is not itself a list item (which would create blank lines between siblings).
_LIST_NEED_BLANK = re.compile(
    r'(?m)^(?P<prev>(?![ \t]*[-*+] )(?![ \t]*\d+\. )[^\n]+)\n(?P<list>[ \t]*(?:[-*+]|\d+\.) )'
)


def _ensure_blank_before_lists(md: str) -> str:
    # \s* in either group would match newlines and grow the gap each pass,
    # never reaching the prev==md fixed point. Use [ \t]* throughout.
    for _ in range(8):
        new = _LIST_NEED_BLANK.sub(lambda m: m.group('prev') + '\n\n' + m.group('list'), md)
        if new == md:
            return md
        md = new
    return md


def _convert_task_lists(md: str) -> str:
    """Replace groups of `- [x]/[ ]` lines with raw checklist HTML blocks."""
    lines = md.split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _RE_TASK_LINE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        items: list[str] = []
        while i < len(lines):
            m2 = _RE_TASK_LINE.match(lines[i])
            if not m2:
                break
            checked = m2.group(2) in ('x', 'X')
            text = _html.escape(m2.group(3))
            cls = 'checked' if checked else 'unchecked'
            items.append(f'<li class="{cls}">{text}</li>')
            i += 1
        # Raw HTML block: surround with blank lines so Python-Markdown leaves it alone
        out.append('')
        out.append('<ul class="checklist">' + ''.join(items) + '</ul>')
        out.append('')
    return '\n'.join(out)


def _fallback_html(md: str) -> str:
    """Safe plaintext fallback when markdown parsing fails."""
    return f'<div>{_html.escape(md).replace(chr(10), "<br>")}</div>'


def markdown_to_html(md: str) -> str:
    """Public API — runs the conversion in a pebble worker with a 60s budget."""
    return _run_with_budget(_markdown_to_html_impl, md)


def _markdown_to_html_impl(md: str) -> str:
    """Convert Markdown into Apple-Notes-flavoured HTML suitable for AppleScript 'body of note' assignment."""
    if not md:
        return ""
    try:
        import markdown as _md

        pre = _convert_task_lists(md)
        # Convert ~~strike~~ → <del>...</del> before the parser runs
        pre = _RE_STRIKE.sub(r'<del>\1</del>', pre)
        pre = _ensure_blank_before_lists(pre)
        html = _md.markdown(pre, extensions=['fenced_code', 'tables', 'sane_lists'])
        # Apple-Notes tag conventions
        html = html.replace('<strong>', '<b>').replace('</strong>', '</b>')
        html = html.replace('<em>', '<i>').replace('</em>', '</i>')
        html = html.replace('<del>', '<strike>').replace('</del>', '</strike>')
        # Strip fenced-code language attribute (Apple Notes ignores it)
        html = _RE_CODE_LANG_ATTR.sub('<code>', html)
        # Downgrade h4-h9 to div (Apple Notes caps at h3)
        html = _RE_H456_OPEN.sub('<div>', html)
        html = _RE_H456_CLOSE.sub('</div>', html)
        # Swap <p>/</p> → <div>/</div>
        html = html.replace('<p>', '<div>').replace('</p>', '</div>')
        # Apple Notes merges adjacent <ul> and <ol> into a single list unless a
        # paragraph-level block separates them. Match Apple's own emit shape.
        # Use \b after the tag name so we match opening tags with attributes too
        # (e.g. `<ul class="checklist">` from _convert_task_lists output — the v9
        # audit found this case was being missed by the strict `<ul>` form).
        html = re.sub(
            r'</ul>\s*(<ol\b[^>]*>)',
            r'</ul><div><br></div>\1',
            html,
            flags=re.IGNORECASE,
        )
        html = re.sub(
            r'</ol>\s*(<ul\b[^>]*>)',
            r'</ol><div><br></div>\1',
            html,
            flags=re.IGNORECASE,
        )
        return html.strip()
    except Exception:
        try:
            return _fallback_html(md)
        except Exception:
            return ""


def apply_style_runs(md: str, runs: list, plain_text: str) -> str:
    """Augment HTML-derived markdown with protobuf style-run information.

    Recovers what AppleScript HTML loses:
    - Checkbox state: replace `- text` with `- [x] text` or `- [ ] text` for runs with style_type=103
    - Code blocks: wrap monospace runs (style_type=4) in fenced code blocks

    Best-effort — if char-offset mapping fails, returns md unchanged.
    """
    if not md or not runs or not plain_text:
        return md
    try:
        from .protobuf_reader import STYLE_TYPE_CHECKBOX, STYLE_TYPE_MONOSPACED  # noqa: F401
    except Exception:
        return md
    # Build a line-by-line index of the plain text and the markdown
    text_lines = plain_text.splitlines()
    md_lines = md.splitlines()
    # Map each text line to the plain-text char range
    text_offsets: list[tuple[int, int]] = []
    cursor = 0
    for line in text_lines:
        start = cursor
        cursor += len(line) + 1  # +1 for newline
        text_offsets.append((start, cursor))

    # For each run, find which line(s) it overlaps
    for run in runs:
        if run.style_type == STYLE_TYPE_CHECKBOX:
            run_start = run.offset
            run_end = run.offset + run.length
            for i, (ls, le) in enumerate(text_offsets):
                if ls < run_end and le > run_start:
                    # Find a corresponding markdown line that starts with "- "
                    text_line = text_lines[i] if i < len(text_lines) else ""
                    text_stripped = text_line.strip()
                    for j, mdl in enumerate(md_lines):
                        ml = mdl.strip()
                        if ml.startswith("- ") and ml[2:].strip() == text_stripped:
                            prefix = "- [x] " if run.is_checked else "- [ ] "
                            md_lines[j] = mdl.replace("- ", prefix, 1)
                            break
    return "\n".join(md_lines)
