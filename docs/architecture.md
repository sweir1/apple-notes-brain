# Architecture

A short tour of what the server is actually doing under the hood. The goal is that if you ever need to debug a "why is this note stale" or "why does my count look wrong" question, you know exactly where to look.

## Read / write split

```mermaid
flowchart LR
  Client[MCP client<br/>(Claude Desktop, Cursor, ...)]
  Server[apple-notes-brain]
  SQLite[(NoteStore.sqlite<br/>read-only WAL)]
  AS[osascript subprocess]
  NotesApp[Notes.app]

  Client -- stdio JSON-RPC --> Server
  Server -- file:?mode=ro --> SQLite
  Server -- AppleScript --> AS
  AS --> NotesApp
  NotesApp <-->|iCloud sync| Cloud[(iCloud)]
  Cloud --> SQLite
```

The split is deliberate:

- **Reads** go through SQLite directly. Sub-100ms p99, concurrent with Notes.app's own writes thanks to WAL mode. Pagination is cursor-based and bounded so listing a 10k-note vault never blows context.
- **Writes** go through AppleScript because it is the **only** Apple API that preserves iCloud sync. Writing to `NoteStore.sqlite` directly would orphan the change at the next sync.

## The reader (`sqlite_reader.py`)

Opens `~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite` read-only via `file:…?mode=ro`. Apple Notes uses WAL mode, so our reads never block Notes.app's writes and vice versa.

Requires **Full Disk Access** for the process running the server — see [Troubleshooting → Full Disk Access](troubleshooting.md#full-disk-access).

Key responsibilities:

- Short 4–6 character ID resolution (`p160` ↔ full `x-coredata://…/ICNote/pNNN`).
- Cursor pagination with a `has_more` / `next_cursor` envelope.
- Folder-path resolution via a recursive `ZPARENT` walk on `ZICCLOUDSYNCINGOBJECT` (a scoped call like `folder_path="Work"` matches that folder and every descendant).
- FTS probe at startup — uses the SQLite FTS shadow table when Apple ships one on this macOS version, falls back to a decompress-and-scan loop otherwise.

## The writer (`applescript.py` + `scripts.py`)

Shells out to `osascript` with `character id 30/31` record/unit separators to parse output unambiguously. Values are escaped via `applescript.quote()` before substitution — injection-safe.

Requires **Automation permission** for the process running the server (sticky once approved per process; see [Troubleshooting → Automation permission](troubleshooting.md#automation-permission)).

Concurrency: write batches (`rename_note`, `move_note`, `get_notes`) fan out over 5 concurrent osascript workers via the `pebble` thread pool.

## Note bodies (`protobuf_reader.py`)

Note bodies are stored as gzip-compressed protobuf in `ZICNOTEDATA.ZDATA`. Apple doesn't publish the schema; we vendor a community one at [`src/apple_notes_brain/proto/notestore_pb2.py`](https://github.com/sweir1/apple-notes-brain/blob/main/src/apple_notes_brain/proto/notestore_pb2.py).

- For **search**, the server decompresses and walks the protobuf extracting UTF-8 string fields (lossy but fine for substring matching).
- For **rendered output**, AppleScript's `body of note` property returns proper HTML which the server converts to Markdown via `markdownify` + Apple-specific pre-processing (checklists, attachment stubs, monospaced blocks). See [Markdown round-trip](#markdown-round-trip).
- For **checklist state** that AppleScript strips on read, `protobuf_reader.py` can recover the tick state directly from the protobuf.

## Markdown round-trip

Read direction (HTML → Markdown):

- Headings `<h1>`/`<h2>`/`<h3>` → `#`/`##`/`###`
- Bold/italic/strike (`<b>`, `<i>`, `<strike>`) → `**` / `*` / `~~`
- Lists, numbered lists, nested lists
- **Checklists** `<ul class="checklist"><li class="checked">` → `- [x]` / `- [ ]`
- Links `<a href>` → `[text](url)`
- Fenced code `<pre><code class="language-py">` → ```` ```py ... ``` ````
- Attachments `<object id="...">` → `![attachment](attachment:ID)` placeholder (binary lives separately at `~/Library/Group Containers/group.com.apple.notes/Accounts/.../Media/` — not returned inline)
- Tables → standard Markdown tables

Write direction (Markdown → Apple HTML):

- Inverse of the above, using Apple's preferred tags (`<b>` not `<strong>`, `<div>` not `<p>`).
- `h4`+ are downgraded to plain `<div>` (Apple Notes only renders h1–h3).
- `- [x]` / `- [ ]` → `<ul class="checklist"><li class="checked">`.

## Attachment-destructive writes

Apple's AppleScript `set body of note` has a known, silent bug: **it deletes every attachment on the target note** (images, sketches, scans, file attachments) with no warning from Apple. `update_note` guards against this:

- Before writing, the server queries the attachment count for the target note via SQLite.
- If the count is > 0 and `allow_attachment_loss=False` (the default), the write is refused with a clear error naming how many attachments would be lost.
- An LLM / client can only override by passing `allow_attachment_loss=True` — which should only happen after explicit user confirmation.

`create_note` is unaffected (new notes have no attachments). `delete_note` is a deliberate deletion — no guard needed. The `attachments` count is returned on every `get_note` / `get_notes` response so the LLM can surface it without a separate query.

## Locked notes

Apple Notes' password-protected notes encrypt the body blob. This server never decrypts and never peeks:

- `search_notes` matches locked notes by **title only**. Matched rows come back with `locked: true` and a synthetic snippet of `[locked — title matched; body encrypted]`.
- `list_notes` surfaces locked notes with `locked: true` and an empty body preview.
- `get_note` on a locked note short-circuits before AppleScript and returns `[locked — unlock this note in Notes.app to read its contents]` plus `locked: true`.
- `update_note` and `delete_note` on a locked note raise an error rather than failing mid-AppleScript.
- Locked notes are omitted from the `notes://recent/…` resource autocomplete list.

## Cache coherence

This server reads from SQLite directly and writes through Notes.app via AppleScript. The two are **eventually consistent**, not instantaneously consistent. Most counts and folder FKs you see at the start of a new session may be a few hundred milliseconds to a few minutes behind reality.

Three classes of staleness:

- **Zombie trash rows** — notes auto-purged by Apple's 30-day rule remain as rows in SQLite until the app next compacts. They carry `ZFOLDERTYPE=1` but are already gone in the live app. Harmless for reads; confusing when counting.
- **Stale `ZFOLDER` FKs** — a note moved in the app may still show its old folder in SQLite until the app persists the move. The "why is p269 in BuildProtect?" case.
- **Post-write read lag** — after an AppleScript write returns, the next SQLite read may still show the old state for a brief window (usually < 1s).

Built-in mitigations (fully automatic):

- **Startup pre-warm** (`cache.prewarm()`) — runs a no-op AppleScript ping at server startup. Triggers the Automation permission prompt before any user-invoked tool call, and wakes Notes.app so it flushes its in-memory state. < 200ms when permission is already granted.
- **Post-write sync** (`cache.sync_after_write()`) — every write call auto-flushes; the next SQLite read sees the change.
- **Background auto-refresh** — daemon thread pings Notes.app every 4 seconds (default) while you're actively using MCP tools. Catches changes made *outside* the MCP (user edits in Notes.app directly, iCloud sync from another device).
    - **Idle pause:** if no MCP tool has been called for 5 minutes, ticks stop until activity resumes.
    - **Notes.app closed skip:** when Notes.app isn't running, ticks short-circuit (~5ms `pgrep` check, no AppleScript invocation, no auto-launch).
    - **System sleep freeze:** lid close / true sleep freezes the thread (`CLOCK_UPTIME_RAW` doesn't advance during sleep). Resumes within one interval after wake.
    - **Client-gated lifecycle:** the thread only exists while the MCP client is connected. Quit the client → process dies → thread dies.
- **Busiest-state cost:** Claude active + Notes.app open + chatting ≈ 15 ticks/min × ~100ms = 2.5% of one core ≈ 0.25% system-wide.

Knobs: `NOTES_MCP_AUTO_REFRESH`, `NOTES_MCP_REFRESH_INTERVAL`, `NOTES_MCP_IDLE_THRESHOLD` — see [Configuration](configuration.md).

Apple provides no formal flush API, so worst-case staleness is one background-tick interval (~4s by default) for changes made outside the MCP during an active session. MCP-initiated changes are always immediately visible.

## Layout

```
src/apple_notes_brain/
  server.py           # FastMCP registration: tools, resources, prompt, annotations
  tools.py            # Lexical tool implementations
  tools_semantic.py   # Semantic/hybrid tool implementations
  schemas.py          # Pydantic output models
  sqlite_reader.py    # Read-only SQLite, short IDs, cursor pagination, FTS probe
  applescript.py      # osascript runner, escaping, record parsing
  scripts.py          # AppleScript templates
  html_text.py        # HTML → plaintext, multi-span snippets, match counter
  markdown.py         # HTML ↔ Markdown (Apple-flavoured)
  search.py           # Token ranking, substring/phrase/regex matchers
  html_validate.py    # HTML sanitation/validation
  protobuf_reader.py  # ZMERGEABLEDATA1 decoder for checklist state recovery
  cache.py            # Background refresh, pre-warm, post-write sync
  semantic/           # ONNX/Ollama embedders, chunker, sqlite-vec store
  data/               # Bundled seed-models.json (offline embedding metadata)
  proto/              # Apple Notes protobuf schema (vendored, MIT)
```

For internal-only references (raw feature catalogue, cache-research notes), see [`docs/apple-notes-features.md`](apple-notes-features.md) and [`docs/cache-sync-research.md`](cache-sync-research.md).
