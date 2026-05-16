"""Heading-aware markdown chunker.

A faithful Python port of obsidian-brain's `src/embeddings/chunker.ts`.
The goal is *behavioural* parity, not bytewise identical output — chunks
carry the same headings, the same paragraph boundaries, the same content
hashes (modulo how the sha256 string is rendered, which is the same
hex(digest) on both sides).

Algorithm in five passes:
  1. Frontmatter strip — only when the document leads with `---\n` AND a
     closing `---` line is found. Malformed (open-only) frontmatter falls
     through to step 2 with the doc unchanged.
  2. Region protection — fenced code blocks (```…```) and LaTeX display
     blocks ($$…$$) are swapped out for PUA sentinel tokens so heading
     and paragraph regexes don't fire inside them.
  3. Heading split — break on lines like `^#{1,depth} `. Sections keep
     their heading line. Deeper headings stay inside their parent section.
  4. Oversize handling — sections > chunk_size are recursively split on
     paragraph boundaries → sentence boundaries → hard cut.
  5. Region restoration — sentinel tokens replaced with their original
     blocks.

The `start_line` / `end_line` per chunk reference the ORIGINAL document
(before frontmatter strip and region protection) so callers can render
"this chunk came from lines 42–88" annotations.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .types import Chunk, ChunkerConfig, DEFAULT_CHUNKER_CONFIG


# ---------------------------------------------------------------------------
# Regexes — compiled once at module load
# ---------------------------------------------------------------------------

# Frontmatter: `---\n` opening, then content, then `\n---\n` or `\n---$`.
_FRONTMATTER_OPEN = "---\n"
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*\n?", re.DOTALL)

# Fenced code blocks: ```…``` (greedy across newlines). Allows ```lang etc.
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# LaTeX display block: $$…$$ (greedy across newlines). Excludes single-$
# inline math, which doesn't trip heading regexes anyway.
_LATEX_BLOCK_RE = re.compile(r"\$\$[\s\S]*?\$\$", re.MULTILINE)

# Heading line: 1+ `#` followed by a space and then text. Captures level
# (via len of the first group) and the heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Paragraph break: one-or-more blank lines.
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")

# Sentence boundary: end-of-sentence punctuation followed by whitespace
# and an uppercase ASCII letter (or unicode upper). Imperfect for
# abbreviations ("Mr. Smith") but standard.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ſ])")

# Sentinel chars — PUA U+E000/U+E001 frame a numeric counter. Chosen
# because no normal markdown contains these.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""


def _sentinel_for(index: int) -> str:
    return f"{_SENTINEL_OPEN}{index}{_SENTINEL_CLOSE}"


# Capture every emitted sentinel, regardless of index, for restoration.
_SENTINEL_RE = re.compile(f"{_SENTINEL_OPEN}(\\d+){_SENTINEL_CLOSE}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_id(node_id: str, chunk_index: int) -> str:
    """Stable composite identifier for a chunk. Used as primary key in
    the `chunks` table and the rowid-anchored `chunks_vec` virtual table."""
    return f"{node_id}#{chunk_index}"


def build_chunk_embedding_text(chunk: Chunk) -> str:
    """Compose the string fed to `embedder.embed()`.

    Headings carry strong signal for retrieval ("methodology section"
    queries should rank methodology-headed chunks high) so we prepend
    the heading separated by a blank line. Bodies without a heading are
    embedded as-is — adding noise hurts more than it helps.
    """
    if chunk.heading:
        return f"{chunk.heading}\n\n{chunk.content}"
    return chunk.content


def chunk_markdown(
    content: str,
    config: ChunkerConfig = DEFAULT_CHUNKER_CONFIG,
) -> list[Chunk]:
    """Slice `content` into a list of Chunks per ChunkerConfig.

    Returns `[]` for empty input. Chunks below `min_chunk_chars` are
    dropped — the indexer's empty-note fallback handles "doc had content
    but nothing made the cut" so we don't have to here.
    """
    if not content:
        return []

    # Normalise line endings BEFORE any line-counting so start_line /
    # end_line in the output reference the post-normalisation document.
    # (Apple Notes bodies are LF already but we keep this for safety.)
    normalised = content.replace("\r\n", "\n").replace("\r", "\n")

    # Pass 1: frontmatter strip (track offset so line numbers stay aligned)
    body, line_offset = _strip_frontmatter(normalised)
    if not body.strip():
        return []

    # Pass 2: protect code + LaTeX blocks
    protected, restore_map = _protect_regions(body, config)

    # Pass 3: heading split
    sections = _split_by_headings(protected, config.heading_split_depth)

    # Pass 4 + 5: oversize handling + region restoration + line tracking
    chunks: list[Chunk] = []
    chunk_counter = 0
    # Cumulative offset across sections so start/end lines accumulate.
    for section in sections:
        for raw_text in _split_oversized(section.body, config):
            restored = _restore_regions(raw_text, restore_map)
            content_stripped = restored.strip()
            if len(content_stripped) < config.min_chunk_chars:
                continue
            start_line, end_line = _locate_chunk_lines(
                normalised, content_stripped, line_offset
            )
            chunk_content_hash = _hash_chunk(section.heading, content_stripped)
            chunks.append(
                Chunk(
                    chunk_index=chunk_counter,
                    heading=section.heading,
                    heading_level=section.heading_level,
                    content=content_stripped,
                    content_hash=chunk_content_hash,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            chunk_counter += 1
    return chunks


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _Section:
    __slots__ = ("heading", "heading_level", "body")

    def __init__(self, heading: str | None, heading_level: int | None, body: str):
        self.heading = heading
        self.heading_level = heading_level
        self.body = body


def _strip_frontmatter(text: str) -> tuple[str, int]:
    """Remove YAML frontmatter if present. Returns (body, line_offset).

    A malformed frontmatter (opening `---\\n` but no matching close)
    leaves the document untouched and line_offset=0.
    """
    if not text.startswith(_FRONTMATTER_OPEN):
        return text, 0
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        # Open-fence without a close — treat as plain body.
        return text, 0
    stripped = text[match.end():]
    line_offset = text[: match.end()].count("\n")
    return stripped, line_offset


def _protect_regions(
    text: str, config: ChunkerConfig
) -> tuple[str, dict[int, str]]:
    """Replace fenced code blocks + LaTeX blocks with sentinel tokens.

    Returns (text_with_sentinels, {index: original_block}). Restoration
    is order-independent because each sentinel embeds its own index.
    """
    restore: dict[int, str] = {}
    counter = 0
    out = text

    if config.preserve_code_blocks:

        def _sub_code(match: re.Match[str]) -> str:
            nonlocal counter
            restore[counter] = match.group(0)
            sentinel = _sentinel_for(counter)
            counter += 1
            return sentinel

        out = _CODE_FENCE_RE.sub(_sub_code, out)

    if config.preserve_latex_blocks:

        def _sub_latex(match: re.Match[str]) -> str:
            nonlocal counter
            restore[counter] = match.group(0)
            sentinel = _sentinel_for(counter)
            counter += 1
            return sentinel

        out = _LATEX_BLOCK_RE.sub(_sub_latex, out)

    return out, restore


def _restore_regions(text: str, restore_map: dict[int, str]) -> str:
    """Reverse `_protect_regions`: swap sentinel tokens back for their
    original block content. Sentinels that have no entry in the map
    survive verbatim — defensive against sentinel-collisions in user
    text (vanishingly unlikely with PUA chars, but still)."""
    if not restore_map:
        return text

    def _expand(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return restore_map.get(idx, match.group(0))

    return _SENTINEL_RE.sub(_expand, text)


def _split_by_headings(text: str, depth: int) -> list[_Section]:
    """Break `text` into Sections at headings of level <= depth.

    The first section's heading is None unless the document starts with
    a qualifying heading. Body whitespace at the ends is trimmed.
    """
    # Collect heading match positions that qualify under the depth limit.
    heads: list[tuple[int, int, int, str]] = []  # (start, end, level, title)
    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        if level <= depth:
            heads.append((match.start(), match.end(), level, match.group(2).strip()))

    if not heads:
        body = text.strip()
        return [_Section(None, None, body)] if body else []

    sections: list[_Section] = []
    # Preamble before first qualifying heading.
    if heads[0][0] > 0:
        pre = text[: heads[0][0]].strip()
        if pre:
            sections.append(_Section(None, None, pre))

    # Each qualifying heading owns the slice up to the next qualifying
    # heading. The heading line itself is NOT included in body — we
    # carry it on the Section so build_chunk_embedding_text can prepend.
    for idx, (h_start, h_end, level, title) in enumerate(heads):
        next_start = heads[idx + 1][0] if idx + 1 < len(heads) else len(text)
        body = text[h_end:next_start].strip()
        if body:
            sections.append(_Section(title, level, body))
        else:
            # Heading with empty body — still emit so embedding-text path
            # can carry it. Empty-body sections get filtered later by
            # min_chunk_chars unless heading itself is long enough.
            sections.append(_Section(title, level, ""))

    return sections


def _split_oversized(body: str, config: ChunkerConfig) -> Iterable[str]:
    """Yield body slices each ≤ chunk_size where possible.

    Splits recursively: paragraph → sentence → hard cut. Returns the
    body unchanged when it fits.
    """
    if len(body) <= config.chunk_size:
        yield body
        return

    # Paragraph split first.
    paragraphs = _PARAGRAPH_BREAK_RE.split(body)
    buffer: list[str] = []
    buf_len = 0
    sep_len = 2  # `\n\n`

    def flush() -> Iterable[str]:
        if buffer:
            yield "\n\n".join(buffer)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > config.chunk_size:
            # Flush whatever's buffered, then split this oversized paragraph.
            yield from flush()
            buffer, buf_len = [], 0
            yield from _split_oversized_paragraph(para, config)
            continue
        if buf_len + len(para) + sep_len > config.chunk_size and buffer:
            yield from flush()
            buffer, buf_len = [], 0
        buffer.append(para)
        buf_len += len(para) + sep_len

    yield from flush()


def _split_oversized_paragraph(para: str, config: ChunkerConfig) -> Iterable[str]:
    """Sentence-split a paragraph that itself exceeds chunk_size.

    Falls through to a hard-cut at the chunk_size boundary if even a
    single sentence is too long (long URLs, base64 blobs, etc).
    """
    sentences = _SENTENCE_BREAK_RE.split(para)
    if len(sentences) <= 1:
        yield from _hard_cut(para, config.chunk_size)
        return

    buffer: list[str] = []
    buf_len = 0

    def flush() -> Iterable[str]:
        if buffer:
            yield " ".join(buffer)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > config.chunk_size:
            yield from flush()
            buffer, buf_len = [], 0
            yield from _hard_cut(sent, config.chunk_size)
            continue
        if buf_len + len(sent) + 1 > config.chunk_size and buffer:
            yield from flush()
            buffer, buf_len = [], 0
        buffer.append(sent)
        buf_len += len(sent) + 1

    yield from flush()


def _hard_cut(text: str, max_len: int) -> Iterable[str]:
    """Final fallback: slice into max_len pieces with no boundary awareness.

    Used when the higher splitters can't make progress (a 5000-char
    word, a base64 blob, a contiguous URL). The chunks are still useful
    for embedding even if the boundary is ugly.
    """
    if not text:
        return
    if max_len <= 0:
        yield text
        return
    for i in range(0, len(text), max_len):
        piece = text[i : i + max_len]
        if piece:
            yield piece


def _hash_chunk(heading: str | None, content: str) -> str:
    """Sha256 over (heading or '') + '\\n\\n' + content. First 32 hex chars."""
    payload = f"{heading or ''}\n\n{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _locate_chunk_lines(
    full_text: str, chunk_content: str, line_offset: int
) -> tuple[int, int]:
    """Approximate (start_line, end_line) by searching for chunk_content
    in the post-normalisation full document.

    "Approximate" because:
      - chunk_content has been stripped + had region-sentinels restored,
        but the inner whitespace may differ from the original by trailing
        whitespace trims;
      - if the same chunk text appears twice, we take the first match.
    For accurate-but-uglier alternatives we'd track offsets through every
    regex pass — not worth the complexity at our use case.

    Lines are 1-indexed; line_offset accounts for the stripped frontmatter.
    """
    if not chunk_content:
        return line_offset, line_offset

    idx = full_text.find(chunk_content)
    if idx < 0:
        # Couldn't locate — fall back to (line_offset, line_offset). Rare
        # (only when region sentinels caused a textual transform) but
        # acceptable: line numbers are a UI nicety, not a correctness gate.
        return line_offset, line_offset

    start_line = full_text[:idx].count("\n") + 1
    end_line = start_line + chunk_content.count("\n")
    return start_line, end_line
