# Troubleshooting

## Full Disk Access

**Symptoms:**

```
could not pre-populate recent notes: cannot open NoteStore (unable to open database file).
apple-notes-brain registered 0 recent notes as resources
```

…and SQLite-backed tools (`list_folders`, `search_notes`, `list_notes`) return empty results.

**Cause:** `uvx` doesn't have Full Disk Access. The process chain is **Claude.app → `uvx` → Python**, and Full Disk Access on `uvx` propagates down the chain.

**Fix:**

1. **System Settings → Privacy & Security → Full Disk Access**.
2. Click **+**, press **⌘ Shift G**, paste `/usr/local/bin/uvx`, click **Open**, toggle the entry on.
3. Make sure **Claude** (or whichever MCP client you're using) is also in the list and toggled on.
4. Quit and relaunch the client.

For first-time setup, the [install script](https://github.com/sweir1/apple-notes-brain/blob/main/scripts/install.sh) walks you through this automatically. See also the [non-technical macOS walkthrough](install-mac-nontechnical.md).

## Automation permission

**Symptoms:** the first write-path tool call (or any tool that needs `body of note` from AppleScript) hangs for 60 seconds, then times out.

**Cause:** macOS popped an Automation permission dialog that was dismissed (or never appeared because Notes.app wasn't running), so `osascript` is waiting for permission that will never come.

**Fix:**

1. **System Settings → Privacy & Security → Automation**.
2. Expand **Claude** (or your MCP client).
3. Toggle **Notes** on.
4. Re-invoke the tool — it should succeed instantly.

Once granted, this is sticky for that copy of the client.

## Attachment-destructive writes

**Symptoms:** `update_note` returns an error like `refused: note has 3 attachments that would be destroyed; pass allow_attachment_loss=True to override`.

**Cause:** Apple's AppleScript `set body of note` has a known silent bug — **it deletes every attachment on the target note** with no warning. `update_note` guards against this by checking the attachment count before writing.

**Fix:**

- If you didn't intend to lose attachments: split the work. Use `create_note` for the new content, or write somewhere that doesn't have attachments.
- If the user explicitly confirms the loss is OK: pass `allow_attachment_loss=True` on the next call.

See [Architecture → Attachment-destructive writes](architecture.md#attachment-destructive-writes) for the full guard logic.

## Locked notes

**Symptoms:** `get_note` on a password-protected note returns `body: "[locked — unlock this note in Notes.app to read its contents]"` and `locked: true`. Writes (`update_note`, `delete_note`) refuse with an error.

**Cause:** Apple Notes encrypts the body blob for locked notes. apple-notes-brain never decrypts and never peeks.

**Fix:** Unlock the note in Notes.app first. apple-notes-brain matches locked notes by **title only** during search, so `search_notes` still discovers them; it just can't read or modify the body.

## FTS not available

**Symptoms:** `search_notes` works but feels slow on large vaults; `semantic_index_status()` (or server startup logs) report "FTS unavailable".

**Cause:** Apple ships a SQLite FTS shadow table only on certain macOS versions. When it's not there, apple-notes-brain falls back to a decompress-and-scan loop over `ZICNOTEDATA.ZDATA`. Correct results, slower.

**Fix:** None on apple-notes-brain's side — this is a macOS version thing. If you have the `[semantic]` extra installed, use `semantic_search` or `hybrid_search` instead for faster results on large vaults.

## Cache staleness

**Symptoms:** counts look wrong; a note you just moved still shows in the old folder; a note you deleted still appears in `list_notes`.

**Cause:** apple-notes-brain reads SQLite and writes via AppleScript — they're eventually consistent. There are three classes of staleness: zombie trash rows, stale `ZFOLDER` FKs, and post-write read lag.

**Fix:** Usually nothing — the background auto-refresh thread pings Notes.app every `NOTES_MCP_REFRESH_INTERVAL` seconds (default 4) and flushes its state. Worst-case staleness is one tick interval. If you need to force the issue:

- Make any MCP write call — the post-write sync is immediate.
- Or restart Notes.app — flushes everything.

See [Architecture → cache coherence](architecture.md#cache-coherence) for the full model.

## Semantic tools return `missing-extras`

**Symptoms:** `semantic_search` / `hybrid_search` / `reindex_semantic` / `semantic_index_status` all return `{"error": "...", "code": "missing-extras"}`.

**Cause:** the `[semantic]` install extra isn't installed.

**Fix:**

```bash
pip install 'apple-notes-brain[semantic]'
# or
uv add 'apple-notes-brain[semantic]'
```

Then restart the MCP client.

## Known limitations (won't-fix or deferred)

- **FTS** is not enabled unless macOS ships a compatible shadow table on your install.
- **Attachments** (image binaries, sketches, scans) are surfaced as placeholder Markdown — binary fetching is deferred.
- **Account info** (iCloud vs On My Mac vs Gmail) is not exposed per-note yet.
- **Tags** (`#tag` syntax) appear as plain text in the body — no separate `tags` field yet.
- **`pin_note`** is not implemented. Apple's AppleScript surface doesn't expose `pinned` as settable, and writing `ZISPINNED` directly would desync iCloud.
- **`bulk_delete`** is not a separate tool; loop `delete_note`. Each call is a cheap AppleScript dispatch.
- **`get_attachments`** (fetching image/sketch binaries from the Media directory) is deferred.
- **Checklist tick state on writes** is reliable only when Apple Notes' HTML includes `class="checked"` on the `<li>`. On some macOS versions the class is stripped on read; your `- [x]` input is always encoded correctly on write, but round-tripping may lose the tick.
