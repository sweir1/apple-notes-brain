# apple-notes-brain

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A [Model Context Protocol](https://modelcontextprotocol.io/) server for Apple
Notes on macOS. Gives Claude (or any MCP client) read, write, and search access
to your notes — including searching **inside** note bodies, nested-folder
scoping, full Markdown fidelity on reads, Markdown input on writes, and a set
of ergonomics aimed at minimising token use and avoiding follow-up tool calls.

> Part of the `-brain` family of MCP servers. Sibling project:
> [`obsidian-brain`](https://github.com/sweir1/obsidian-brain) — same idea for
> Obsidian vaults.

## Why this exists

The default "just wrap AppleScript" approach either strips every piece of
formatting (so headings, bullets, links, checklists, tables all vanish) or
returns the raw HTML (which eats tokens). This server:

- Reads via SQLite (sub-100ms, concurrent with Notes.app) and writes via
  AppleScript (the only supported Apple API that preserves iCloud sync).
- Returns structured bodies as **Markdown by default** so LLMs get real
  headings, `**bold**`, `*italic*`, `- [x]` checklists, `[text](url)` links,
  fenced code blocks, tables, and `![](attachment:…)` placeholders for
  embedded images.
- Accepts **Markdown input** on `create_note` / `update_note` and converts to
  Apple's HTML dialect (`<b>` not `<strong>`, `<ul class="checklist">`, etc.)
  before assignment.
- Uses short 4–6-char IDs (`p160`) on the wire, with a resolver that still
  accepts the full `x-coredata://…/ICNote/pNNN` URI.
- Paginates with a `has_more`/`next_cursor` envelope so the LLM can tell when
  there's more to fetch.
- Returns up to three non-overlapping snippet spans and a `match_count` per
  search hit so the LLM can judge relevance without fetching the full body.
- Respects locked notes: body never read, `locked: true` surfaced on every
  affected row, writes refused with a clear error.

## Tools

| Tool | Kind | What it does |
|---|---|---|
| `list_folders(include_counts?)` | read | Every folder as a nested path with `is_trash` flag. Set `include_counts: true` for per-folder note counts. |
| `list_notes(folder_path?, limit?, cursor?, include_trash?, modified_after?, modified_before?)` | read | Paginated list, most-recent first. Default 20 rows, max 500 per page. Excludes Recently Deleted by default. Accepts ISO-8601 date or datetime bounds. |
| `search_notes(query, folder_path?, search_body?, fuzzy?, mode?, limit?, cursor?, include_body?, max_body_chars?, include_trash?, modified_after?, modified_before?)` | read | Search across titles and body plaintext. Returns up to 3 snippet spans + `match_count` per hit. `mode` is `substring` (default) or `regex`. `include_body=true` bundles up to `max_body_chars` (≤2000) of text on the TOP 5 results. |
| `get_note(note_id, format?, fast?)` | read | Full note. `format` is `markdown` (default), `text`, or `html`. `fast=true` uses a lossy SQLite reader (text only, sub-100ms, no formatting). |
| `get_notes(note_ids, format?, fast?)` | read | Batch fetch up to 20 notes. Parallel AppleScript fan-out by default. Locked notes come back with a sentinel body; the batch does not fail. |
| `create_note(title, body, folder_path?, format?)` | write | Create a note. `format` is `markdown` (default), `html`, or `text`. |
| `update_note(note_id, body, append?, format?, allow_attachment_loss?)` | write | Replace or append to the end of the body. **Refuses notes with attachments** unless `allow_attachment_loss=true` (Apple bug — `set body` destroys attachments). Refuses locked notes. Does not rename — use `rename_note`. Does not move — use `move_note`. |
| `rename_note(note_id, new_title)` | write | Rename one or many notes. `note_id` + `new_title` can both be strings (single) OR both lists of equal length (batch, up to 20). Body untouched. |
| `move_note(note_id, folder_path)` | write | Move one or many notes. `note_id` can be a string (single) or a list (batch, up to 20); `folder_path` is always a single destination. Body untouched; refuses moves into Recently Deleted. |
| `create_folder(name, parent_folder_path?)` | write | Create a folder, optionally nested under a parent. Name cannot contain `/`. |
| `delete_note(note_id)` | destructive | Move to Recently Deleted, or **permanently delete** if already there. Refuses locked notes. |
| `semantic_search(query, limit?, unique?)` | read | Embedding-based chunk-level search. Same envelope as `search_notes`, plus `semantic_score` on each hit. `unique='notes'` (default) dedups multi-chunk hits per note; `unique='chunks'` returns chunk-grained hits. Requires `[semantic]` extra. |
| `hybrid_search(query, limit?, unique?)` | read | Reciprocal-rank-fused semantic + lexical (RRF k=60). Higher recall than either alone. Each hit carries both `semantic_score` and `lexical_score`. Requires `[semantic]` extra. |
| `reindex_semantic(force?)` | write | Trigger a full pass of the semantic indexer over NoteStore.sqlite. Content-hash dedup so re-running is cheap when nothing changed. Returns counters + a capped list of any per-chunk failures. |
| `semantic_index_status()` | read | Snapshot: indexed nodes/chunks, failed-chunk count, last-indexed timestamp, active embedder + dim + ONNX execution providers, on-disk paths. Useful for "is the index actually built / is CoreML EP active?" sanity checks. |

All write tools return `{id, action, error?}` (action ∈ `created`/`updated`/`renamed`/`moved`/`deleted`/`skipped`). All read tools return typed Pydantic models so FastMCP emits a proper `outputSchema`.

### Batch semantics (`rename_note`, `move_note`)

- **Single-note call:** raises on any failure (locked note, missing note, missing folder, etc.). Returns one `MutationResult`.
- **Batch call:** validation errors (missing folder, too many notes, shape mismatch) raise up-front before any AppleScript runs. Once the batch starts, per-note failures come back as `MutationResult(id, action="skipped", error="…")` instead of killing the batch — mirrors how `get_notes` handles locked notes. The batch fans out over 5 concurrent AppleScript workers.
- Max 20 notes per batch. Empty list → empty list (no-op).

### macOS permission prompt (writes only)

The first write-path tool call per session may trigger an OS-level Automation permission dialog for the process running the server (typically Claude Desktop). If the user doesn't see or doesn't approve the prompt, `osascript` hangs until the 60-second timeout. Approve once and it's sticky for that process.

## Resources

On clients that support MCP resources (Claude Desktop, Claude Code, Cursor,
Continue):

- `notes://note/{id}` → note body as Markdown (default).
- `notes://note/{id}/html` → raw HTML.
- `notes://note/{id}/text` → plaintext.

On server startup the 50 most-recently-modified notes are also registered as
individual `notes://recent/{id}` resources so they appear in `@`-mention
autocomplete. Locked notes are omitted from the autocomplete list. Clients
that don't speak resources ignore all of this and use the `get_note` tool
instead — capability negotiation handles it automatically.

## Prompt

`notes_server_overview` — a single built-in prompt that returns a short
architecture summary (SQLite for reads, AppleScript for writes, ID format,
locked-note semantics, pagination envelope, FTS availability on this
machine). Useful as a one-shot context-primer.

## How it actually works

- **SQLite path** (`sqlite_reader.py`) opens `~/Library/Group
  Containers/group.com.apple.notes/NoteStore.sqlite` read-only via
  `file:…?mode=ro`. Apple Notes uses WAL mode, so our reads never block
  Notes.app's writes and vice versa. Requires **Full Disk Access** for the
  process running the server.
- **AppleScript path** (`applescript.py` + `scripts.py`) shells out to
  `osascript` with `character id 30/31` record/unit separators to parse
  output unambiguously. Values are escaped via `applescript.quote()` before
  substitution — injection-safe. Requires **Automation permission** for the
  process running the server.
- **Folder nesting** is resolved by walking `ZPARENT` in `ZICCLOUDSYNCINGOBJECT`.
  A scoped call like `folder_path="Work"` matches that folder and every
  descendant.
- **Note bodies** are stored as gzip-compressed protobuf in
  `ZICNOTEDATA.ZDATA`. Apple doesn't publish the schema. For search, the
  server decompresses and walks the protobuf extracting UTF-8 string fields
  (lossy but fine for substring matching). For rendered output, AppleScript's
  `body of note` property returns proper HTML which the server converts to
  Markdown via `markdownify` + Apple-specific pre-processing (checklists,
  attachment stubs, monospaced blocks).
- **FTS** — the code probes for a SQLite FTS shadow table at startup. If
  Apple ships one on your macOS version it will be used for search.
  Otherwise the server falls back to the decompress-and-scan loop.

## Markdown round-trip

Read direction (HTML → Markdown):

- Headings `<h1>`/`<h2>`/`<h3>` → `#`/`##`/`###`
- Bold/italic/strike (`<b>`, `<i>`, `<strike>`) → `**`/`*`/`~~`
- Lists, numbered lists, nested lists
- **Checklists** `<ul class="checklist"><li class="checked">` → `- [x]` / `- [ ]`
- Links `<a href>` → `[text](url)`
- Fenced code `<pre><code class="language-py">` → ```` ```py ... ``` ````
- Attachments `<object id="...">` → `![attachment](attachment:ID)` placeholder
  (binary lives separately at `~/Library/Group Containers/group.com.apple.notes/Accounts/.../Media/` — not returned inline)
- Tables → standard Markdown tables

Write direction (Markdown → Apple HTML):

- Inverse of the above, using Apple's preferred tags (`<b>` not `<strong>`,
  `<div>` not `<p>`).
- `h4`+ are downgraded to plain `<div>` (Apple Notes only renders h1–h3).
- `- [x]` / `- [ ]` → `<ul class="checklist"><li class="checked">`.

## ⚠️ Attachment-destructive writes

Apple's AppleScript `set body of note` has a known, silent bug: **it deletes
every attachment on the target note** (images, sketches, scans, file
attachments) with no warning from Apple. `update_note` guards against this:

- Before writing, the server queries the attachment count for the target
  note via SQLite.
- If the count is > 0 and `allow_attachment_loss=False` (the default), the
  write is refused with a clear error naming how many attachments would be
  lost.
- An LLM / client can only override by passing `allow_attachment_loss=True`
  — which should only happen after explicit user confirmation.

`create_note` is unaffected (new notes have no attachments). `delete_note`
is a deliberate deletion — no guard needed.

The `attachments` count is also returned on every `get_note` / `get_notes`
response so the LLM can surface it without a separate query.

## Locked notes

Apple Notes' password-protected notes encrypt the body blob. This server never
decrypts and never peeks:

- `search_notes` matches locked notes by **title only** — the body is never
  decompressed. Matched rows come back with `locked: true` and a synthetic
  snippet of `[locked — title matched; body encrypted]`.
- `list_notes` surfaces locked notes with `locked: true` and an empty body
  preview.
- `get_note` on a locked note short-circuits before AppleScript and returns
  a body of `[locked — unlock this note in Notes.app to read its contents]`
  plus `locked: true`.
- `update_note` and `delete_note` on a locked note raise an error rather than
  failing mid-AppleScript.
- Locked notes are omitted from the `notes://recent/…` autocomplete list.

## Semantic + hybrid search (optional)

The four tools `semantic_search`, `hybrid_search`, `reindex_semantic`,
and `semantic_index_status` use embedding-based retrieval. They're
gated behind the `[semantic]` install extra so the lexical CRUD tools
stay zero-extra-dependency. Without the extra installed, those four
tools return a structured `{"error": "...", "code": "missing-extras"}`
envelope; the rest of the server is unaffected.

```bash
# Install with the semantic extras
pip install 'apple-notes-brain[semantic]'

# or for `uv` users
uv add 'apple-notes-brain[semantic]'

# After first call, the embedder downloads BAAI/bge-small-en-v1.5
# (ONNX-quantised, ~30MB) into ~/.local/share/apple-notes-brain/models/
# and runs in-process via onnxruntime. On macOS Apple Silicon, the
# CoreMLExecutionProvider is preferred — check via semantic_index_status().
```

**Why direct ONNX (not fastembed)**: fastembed has open Apple Silicon
performance issues ([qdrant/fastembed#535](https://github.com/qdrant/fastembed/issues/535)
and [#97](https://github.com/qdrant/fastembed/issues/97)) — its CPU
fallback would be a regression for the macOS-only target. We use
`onnxruntime` + `tokenizers` + `huggingface-hub` directly so the
CoreML execution provider lights up on M-series machines.

**Why not sentence-transformers**: even with the `[onnx]` extra,
sentence-transformers still installs PyTorch (~600MB). The
direct-ONNX path stays at ~80MB total install.

**Use Ollama instead** by setting `EMBEDDING_PROVIDER=ollama` and
optionally `OLLAMA_BASE_URL` / `EMBEDDING_MODEL`. Models are
auto-pulled at first call unless `APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL=0`.

**Environment variables** (full list in `server.json`):

| Var | Default | Notes |
|---|---|---|
| `APPLE_NOTES_BRAIN_DATA_DIR` | `~/.local/share/apple-notes-brain` | Where the semantic index DB + model cache live. |
| `APPLE_NOTES_BRAIN_NO_WATCH` | `0` | `1` disables the background reindex watcher. |
| `APPLE_NOTES_BRAIN_INDEX_INTERVAL` | `30` | Seconds between watcher ticks. Cheap when idle. |
| `EMBEDDING_PROVIDER` | `onnx` | `onnx` or `ollama`. |
| `EMBEDDING_MODEL` | `bge-small-en-v1.5` | Preset short-name or full HF/Ollama identifier. |
| `EMBEDDING_ONNX_PROVIDERS` | (auto) | Override the onnxruntime EP list (e.g. `CPUExecutionProvider`). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | When `EMBEDDING_PROVIDER=ollama`. |

## Install

Requires **macOS**, **Python 3.11+**, and **Apple Notes**.

### One-line macOS installer (recommended)

Handles uv install, the `/usr/local/bin` PATH symlinks Claude Desktop needs,
the Claude Desktop config merge (preserves your other MCP servers), the
Full Disk Access walkthrough for **both** Claude.app **and** the uv-managed
Python (the cached-Python TCC quirk that breaks SQLite reads — see [Known
issue](#known-issue-full-disk-access--uvx-managed-python-on-macos) below),
and the Automation prompt heads-up.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sweir1/apple-notes-brain/main/scripts/install.sh)"
```

Idempotent — safe to re-run if you change your mind, hit a TCC mismatch,
or want to update.

### Quickest — `uvx` (no install)

```bash
uvx apple-notes-brain
```

Runs the server in an ephemeral environment. Best for trying it out or for
client config that should always pull the latest.

### Tool install — `uv`

```bash
uv tool install apple-notes-brain
```

Installs the `apple-notes-brain` command globally (managed by uv). Update
later with `uv tool upgrade apple-notes-brain`.

### Classic — `pip`

```bash
pip install apple-notes-brain
```

### From source

```bash
git clone https://github.com/sweir1/apple-notes-brain.git
cd apple-notes-brain
uv sync                         # or: python3.11 -m venv .venv && .venv/bin/pip install -e .
```

## Hook up to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "apple-notes-brain": {
      "command": "uvx",
      "args": [
        "apple-notes-brain"
      ]
    }
  }
}
```

Or, if you `uv tool install`-ed it:

```json
{
  "mcpServers": {
    "apple-notes-brain": {
      "command": "apple-notes-brain"
    }
  }
}
```

Restart Claude Desktop. On first use macOS will prompt for:

- **Full Disk Access** for the parent process (needed to read NoteStore.sqlite).
- **Automation permission** to control Notes (needed for full body reads and
  all writes).

Grant both. The Automation prompt is sticky once approved; Full Disk Access
is granted in System Settings → Privacy & Security → Full Disk Access.

### Known issue: Full Disk Access on macOS

If Claude Desktop's logs show:

```
could not pre-populate recent notes: cannot open NoteStore (unable to open database file).
apple-notes-brain registered 0 recent notes as resources
```

…and SQLite-backed tools (`list_folders`, `search_notes`, `list_notes`) return
empty results, then `uvx` doesn't have Full Disk Access. The process chain is
**Claude.app → `uvx` → Python**, and Full Disk Access on `uvx` propagates
down the chain.

**Fix:**

1. System Settings → Privacy & Security → Full Disk Access → `+` →
   `Cmd+Shift+G` → paste `/usr/local/bin/uvx` → Open → toggle on.
2. Quit and relaunch Claude Desktop.

(Claude.app should also have FDA toggled on — that's the standard MCP
client requirement.)

For new users on macOS, the [install script](./scripts/install.sh) walks
through this automatically.

## Other MCP clients

Same idea — point the client's MCP server config at `uvx apple-notes-brain`
or the installed `apple-notes-brain` binary. Tested with Claude Desktop,
Claude Code, Cursor, and Continue.

## Example: search → fetch

```
search_notes(query="butter chicken", limit=3)
→ {
    "results": [
      {"id": "p160", "title": "For butter chicken", "folder": "Notes",
       "modified": "2026-04-09 11:12", "match_count": 6,
       "snippets": ["…big ass pot melt butter 2) add…", "…150-200g organic butter (to cook…", "…peanut butter 🔴 Rewe?…"],
       "pinned": false, "locked": false},
      ...
    ],
    "returned": 3, "has_more": false, "next_cursor": null, "total_estimate": null
  }

get_note("p160")   # note the short id
→ NoteDetail(id="p160", title="For butter chicken", format="markdown",
              body="## For butter chicken\n\n2-4 people …",
              pinned=false, locked=false, …)
```

## Cache coherence — when counts look stale

This server reads from SQLite directly and writes through Notes.app via AppleScript. The two are **eventually consistent**, not instantaneously consistent, and most counts / folder FKs you see at the start of a new session may be a few hundred milliseconds to a few minutes behind reality.

Three distinct classes of staleness:

- **Zombie trash rows** — notes auto-purged by Apple's 30-day rule remain as rows in SQLite until the app next compacts. They carry `ZFOLDERTYPE=1` but are already gone in the live app. Harmless for reads (their folder is `Recently Deleted`); confusing when counting.
- **Stale `ZFOLDER` FKs** — a note moved in the app may still show its old folder in SQLite until the app persists the move. This is the "why is p269 in BuildProtect?" case.
- **Post-write read lag** — after an AppleScript write returns, the next SQLite read may still show the old state for a brief window (usually < 1 second).

**Mitigations built in (fully automatic — nothing to call manually):**

- **Startup pre-warm**: at server startup, `cache.prewarm()` runs a no-op AppleScript ping. This (a) triggers the macOS Automation permission prompt BEFORE any user-invoked tool call, and (b) wakes Notes.app so it flushes its in-memory state. Takes < 200ms when permission is already granted.
- **Post-write sync**: every write call (`create_note`, `update_note`, `rename_note`, `move_note`, `delete_note`, folder ops) auto-flushes via `cache.sync_after_write()` — the next SQLite read sees the change.
- **Background auto-refresh**: a daemon thread pings Notes.app every **4 seconds** while you're actively using MCP tools. Catches changes made *outside* the MCP (user edits in Notes.app directly, iCloud sync from another device). Layered cost controls:
  - **Idle pause**: if no MCP tool has been called for 5 minutes, ticks stop until activity resumes. The next tool call instantly wakes the refresher.
  - **Notes.app closed skip**: when Notes.app isn't running, ticks short-circuit (~5ms `pgrep` check, no AppleScript invocation, no auto-launch). CloudKit daemons still flow iCloud changes into SQLite independently, so reads stay fresh without needing our ping.
  - **System sleep freeze**: lid close / true sleep freezes the thread (`CLOCK_UPTIME_RAW` doesn't advance during sleep). Resumes within one interval after wake. Zero cost during sleep.
  - **Client-gated lifecycle**: thread only exists while the MCP client (Claude Desktop, etc.) is connected. Quit the client → process dies → thread dies.
- **Cost in the busiest state** (Claude active + Notes.app open + chatting): ~15 ticks/min × ~100ms = 2.5% of one core ≈ 0.25% system-wide. Everything else is cheaper.

Environment knobs:
  - `NOTES_MCP_AUTO_REFRESH=0` — disable the background thread entirely.
  - `NOTES_MCP_REFRESH_INTERVAL=10` — cadence in seconds (default 4, min 1).
  - `NOTES_MCP_IDLE_THRESHOLD=600` — idle pause threshold in seconds (default 300 = 5 min; set to 0 to disable idle pausing).

Apple provides no formal flush API, so worst-case staleness is one background tick interval (~4s by default) for changes made outside the MCP during an active session. MCP-initiated changes are always immediately visible.

## Known limitations

- **FTS**: not enabled unless macOS ships a compatible shadow table on your
  install. The code probes at startup and logs availability.
- **Attachments** (image binaries, sketches, scans) are surfaced as
  placeholder Markdown — binary fetching is not implemented yet.
- **Account info** (iCloud vs On My Mac vs Gmail) is not exposed per-note yet.
- **Tags** (`#tag` syntax) are not separately surfaced — they appear as plain
  text in the body.
- **`pin_note`** is not implemented. Apple's AppleScript surface for Notes
  doesn't expose the `pinned` property as settable. Reading the current state
  (`pinned` field on `NoteSummary` / `NoteDetail`) works via SQLite; flipping
  it requires writing to `ZISPINNED` directly, which would desync iCloud —
  deliberately out of scope.
- **`bulk_delete`** is not a separate tool; loop `delete_note`. Each call is
  a cheap AppleScript dispatch.
- **`get_attachments`** (fetching image/sketch binaries from the Media
  directory) is deferred.
- **Checklist tick state on writes**: reliable only when Apple Notes' HTML
  includes `class="checked"` on the `<li>`. On some macOS versions the class
  is stripped on read — your `- [x]` input is always correctly encoded on
  write, but round-tripping may lose the tick.
- **Cache staleness at session start**: ZFOLDER FKs and zombie trash rows may show wrong state briefly until the pre-warm AppleScript ping flushes Notes.app. The background auto-refresh (10s default) catches this automatically.

## Development

```bash
git clone https://github.com/sweir1/apple-notes-brain.git
cd apple-notes-brain
uv sync --extra dev
```

### Test suite

The project ships with **898 tests** (~65% line coverage). Default test run
excludes opt-in live tests (those need Notes.app + iCloud + Automation
permission):

```bash
uv run pytest                                  # full suite + coverage
uv run pytest -m integration                   # integration-tier only
uv run pytest -m property                      # hypothesis property tests
uv run pytest -m live                          # OPT-IN: real Notes.app
```

Coverage report is generated in `htmlcov/` after each run; open
`htmlcov/index.html` to browse.

### Smoke-test the server boots

```bash
uv run python -m apple_notes_brain < /dev/null   # blocks waiting for MCP traffic; Ctrl-C to exit
```

Or run the MCP smoke tests (subprocess-spawns the server, exercises the
JSON-RPC handshake):

```bash
uv run pytest --no-cov -m integration tests/integration/test_mcp_smoke.py
```

### Pre-release validation (preflight)

`scripts/preflight.sh` runs locally what CI runs in GitHub Actions, plus
extras you'd want before tagging a release (`uv build`, `server.json` ↔
`pyproject.toml` version sync, console-script presence). Idempotent.

```bash
./scripts/preflight.sh
# → uv sync, pytest --cov, MCP smoke, uv build, version sync, binary check
```

If everything's green, you're ready to bump version + tag + push.

### Inspecting tools interactively

```bash
uv run --with 'mcp[cli]' mcp dev src/apple_notes_brain/server.py
```

## Layout

```
src/notes_mcp/
  server.py         # FastMCP registration: tools, resources, prompt, annotations
  tools.py          # Tool implementations (framework-free, Pydantic returns)
  schemas.py        # Pydantic output models
  sqlite_reader.py  # Read-only SQLite access, short IDs, cursor pagination, FTS probe
  applescript.py    # osascript subprocess runner + quoting + record parsing
  scripts.py        # AppleScript templates
  html_text.py      # HTML → plaintext, multi-span snippets, match counter
  markdown.py       # HTML ↔ Markdown (both directions, Apple-flavoured)
  search.py         # Token ranking + substring/phrase/regex matchers
  protobuf_reader.py # ZMERGEABLEDATA1 decoder for checklist state recovery
  proto/            # Apple Notes protobuf schema (vendored, MIT)
```

## License

[Apache License 2.0](./LICENSE) — Copyright 2026 sweir1.

## Related projects

- [`obsidian-brain`](https://github.com/sweir1/obsidian-brain) — sibling MCP
  server for Obsidian vaults: semantic search, knowledge graph, vault editing.
- [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk) —
  the official MCP Python SDK this server is built on (provides the `FastMCP`
  class at `mcp.server.fastmcp`).
