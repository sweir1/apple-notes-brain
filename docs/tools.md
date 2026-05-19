# Tools

apple-notes-brain exposes 15 MCP tools. The lexical CRUD tools (11) have zero extra dependencies; the four semantic tools require the [`semantic` install extra](getting-started.md#optional-semantic-hybrid-search).

The table below is generated from the tool registrations in [`src/apple_notes_brain/server.py`](https://github.com/sweir1/apple-notes-brain/blob/main/src/apple_notes_brain/server.py). To regenerate after a tool change, run `python scripts/gen_tools_docs.py`.

<!-- GENERATED:tools — DO NOT EDIT. Update the source, then run `python scripts/gen_tools_docs.py`. -->
## Lexical CRUD

| Tool | Kind | What it does |
|---|---|---|
| `list_folders(include_counts?, include_trash?)` | read | List Notes folders. Returns: id, path (slash-joined for nested), is_trash, account, shared. |
| `list_notes(folder_path?, limit?, cursor?, include_trash?, modified_after?, modified_before?)` | read | List notes, most-recently-modified first. Bodies NOT returned (call get_note). |
| `search_notes(query, folder_path?, search_body?, fuzzy?, mode?, limit?, cursor?, include_body?, max_body_chars?, include_trash?, modified_after?, modified_before?)` | read | Search notes by title and/or body. Matches on extracted plaintext (formatting stripped). |
| `get_note(note_id, format?, fast?)` | read | Read one or more notes' full content. |
| `create_note(title?, body?, folder_path?, format?, notes?)` | write | Create one note OR many notes in a single call. |
| `update_note(note_id, body, append?, format?, allow_attachment_loss?)` | write | Replace or append to a note's body. Returns {id, action: 'updated'}. |
| `rename_note(note_id, new_title)` | write | Rename one or many notes (title only — body untouched). |
| `move_note(note_id, folder_path)` | write | Move one or many notes to a folder (body + attachments untouched). |
| `create_folder(name, parent_folder_path?)` | write | Create a folder. Returns {id, action: 'created'}. |
| `rename_folder(folder_id, new_name)` | write | Rename a folder. Returns {id, action: 'renamed'}. |
| `delete_folder(folder_id, allow_non_empty?, note_disposition?, allow_orphaned_subfolders?, recursive?)` | destructive | Delete a folder. Returns {id, action: 'deleted'}. |
| `delete_note(note_id, confirm_shared_delete?)` | destructive | Move a note to Recently Deleted. Returns {id, action: 'deleted'}. |

## Semantic (requires `[semantic]` extra)

| Tool | Kind | What it does |
|---|---|---|
| `semantic_search(query, limit?, unique?, include_trash?)` | read | Embedding-based semantic search over chunked note bodies. |
| `hybrid_search(query, limit?, unique?, include_trash?)` | read | Reciprocal-rank-fused semantic + lexical search. Higher recall than either alone for most queries. |
| `reindex_semantic(force?)` | write | Trigger a full pass of the semantic indexer. |
| `semantic_index_status()` | read | Snapshot of the semantic index + embedder configuration. |
<!-- /GENERATED:tools -->

## Return shapes

All read tools return typed Pydantic models so FastMCP emits a proper `outputSchema`. All write tools return `MutationResult(id, action, error?)` where `action ∈ {created, updated, renamed, moved, deleted, skipped}`.

## Batch semantics (`rename_note`, `move_note`)

- **Single-note call:** raises on any failure (locked note, missing note, missing folder, etc.). Returns one `MutationResult`.
- **Batch call:** validation errors (missing folder, too many notes, shape mismatch) raise up-front before any AppleScript runs. Once the batch starts, per-note failures come back as `MutationResult(id, action="skipped", error="…")` instead of killing the batch — mirrors how `get_notes` handles locked notes. The batch fans out over 5 concurrent AppleScript workers.
- Max 20 notes per batch. Empty list → empty list (no-op).

## Permission prompt (writes only)

The first write-path tool call per session may trigger an OS-level Automation permission dialog for the process running the server (typically Claude Desktop). If the user doesn't see or doesn't approve the prompt, `osascript` hangs until the 60-second timeout. Approve once and it's sticky for that process. See [Troubleshooting → Automation permission](troubleshooting.md#automation-permission).

## Resources

On clients that support MCP resources (Claude Desktop, Claude Code, Cursor, Continue):

- `notes://note/{id}` → note body as Markdown (default)
- `notes://note/{id}/html` → raw HTML
- `notes://note/{id}/text` → plaintext

On server startup the 50 most-recently-modified notes are also registered as individual `notes://recent/{id}` resources so they appear in `@`-mention autocomplete. Locked notes are omitted. Clients that don't speak resources ignore all of this and use the `get_note` tool — capability negotiation handles it.

## Prompt

`notes_server_overview` — a single built-in prompt that returns a short architecture summary (SQLite reads, AppleScript writes, ID format, locked-note semantics, pagination envelope, FTS availability on this machine). Useful as a one-shot context-primer at the start of a session.

## Example

```text
search_notes(query="butter chicken", limit=3)
→ {
    "results": [
      {"id": "p160", "title": "For butter chicken", "folder": "Notes",
       "modified": "2026-04-09 11:12", "match_count": 6,
       "snippets": ["…big ass pot melt butter 2) add…", "…150-200g organic butter (to cook…"],
       "pinned": false, "locked": false},
      …
    ],
    "returned": 3, "has_more": false, "next_cursor": null, "total_estimate": null
  }

get_note("p160")
→ NoteDetail(id="p160", title="For butter chicken", format="markdown",
              body="## For butter chicken\n\n2-4 people …",
              pinned=false, locked=false, …)
```
