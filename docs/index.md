---
hide:
  - navigation
  - toc
---

# apple-notes-brain

A local MCP server that gives Claude and other MCP clients full read/write access to Apple Notes on macOS — Markdown round-trip, semantic + hybrid search, no cloud, notes never leave your machine.

[Quick start](getting-started.md){ .md-button .md-button--primary } [Browse tools](tools.md){ .md-button }

---

## Why apple-notes-brain

<div class="grid cards" markdown>

-   :material-apple: &nbsp; **Native Apple Notes**

    Reads the on-disk SQLite store and writes via AppleScript. No iCloud round-trip, no scraping, no plugin.

-   :material-language-markdown: &nbsp; **Markdown round-trip**

    Bidirectional HTML ↔ Markdown that preserves headings, lists, checklists, tables, and inline formatting.

-   :material-magnify: &nbsp; **Semantic + hybrid search**

    Optional ONNX/Ollama embeddings + RRF fusion with FTS5 for natural-language queries over your notes.

-   :material-shield-lock-outline: &nbsp; **Stays on your Mac**

    Stdio MCP transport. No telemetry, no cloud calls, vault data never leaves the machine.

-   :material-speedometer: &nbsp; **Cursor pagination**

    Bounded, deterministic listing so an LLM can walk a 10k-note vault without blowing context.

-   :material-test-tube: &nbsp; **Hardened**

    ~900 tests, hypothesis property tests, snapshot tests, MCP smoke tests, and a strict pre-release preflight.

</div>

---

## Install in 60 seconds

```bash
# macOS one-liner — installs Python via uv if needed and wires up Claude Desktop
curl -fsSL https://raw.githubusercontent.com/sweir1/apple-notes-brain/main/scripts/install.sh | bash
```

Or pick your client manually on the [install page](install-clients.md).

---

## Recent releases

<!-- The macros plugin reads docs/CHANGELOG.md and renders the latest entries here. -->
{{ recent_releases(5) }}

See the full [changelog](CHANGELOG.md) for everything else.
