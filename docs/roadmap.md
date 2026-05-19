# Roadmap

Forward-looking plan for apple-notes-brain. Items are roughly ordered by intent, not commitment — anything can shift if a higher-priority bug or feature lands.

## Near-term (next 1–2 releases)

- **Tag surfacing.** Detect `#tag` syntax in note bodies and expose as a separate `tags` field on `NoteSummary` / `NoteDetail`. Today they're plain text.
- **Account info on notes.** Surface `account: "iCloud" | "On My Mac" | "Gmail"` per note. The data is in `ZICCLOUDSYNCINGOBJECT`, just not exposed yet.
- **Attachment fetching.** A new `get_attachment(note_id, attachment_id)` tool that returns the raw binary (with size guard) from the Media directory.
- **FTS-via-our-own-table fallback.** When macOS doesn't ship a compatible FTS shadow, optionally maintain our own FTS5 index in `<data_dir>/fts.db` so search stays fast on older macOS.

## Medium-term

- **Account switching.** First-class support for multi-account vaults (filter all tools by `account`).
- **Cross-device staleness signal.** Optional last-iCloud-sync-timestamp returned alongside `cache_coherence` info so the LLM can flag "this might be stale from another device".
- **Tighter pinning model.** Read-only pinned indication today; explore whether `ZISPINNED` can be flipped safely without desyncing iCloud.

## Long-term / exploratory

- **Linux client support.** Currently macOS-only because Apple Notes is macOS-only. If Apple ever ships a cross-platform API, follow.
- **Drawings.** Apple Notes has a drawingsRelationships table that today we ignore. Surfacing the SVG (where available) is plausible.

## Recently shipped

<!-- GENERATED:recently-shipped — DO NOT EDIT. Update docs/CHANGELOG.md, then run `python scripts/gen_readme_recent.py` (which also refreshes this block). -->
- See [Changelog](CHANGELOG.md) for the full history.
<!-- /GENERATED:recently-shipped -->

## Explicitly out of scope

- **Decrypting locked notes.** apple-notes-brain never decrypts. If you want locked-note bodies, unlock them in Notes.app.
- **Direct SQLite writes.** Would desync iCloud — AppleScript is the only supported write path.
- **Replacing Apple Notes UI.** This is a server that lets LLMs read/write your existing notes. Not a Notes client.
- **Cloud features.** No telemetry, no analytics, no remote-API integration. Notes never leave the machine.
