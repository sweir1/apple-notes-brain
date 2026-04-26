# Cache Sync Research: NoteStore.sqlite Staleness

_Research date: April 2026_

---

## (a) Executive Summary

No public API or AppleScript call is documented to reliably force a flush of Apple Notes' in-memory state to `NoteStore.sqlite`. The app holds an open WAL and checkpoints on its own schedule, likely at app-quit or under memory pressure. All surveyed open-source tools either ignore the staleness problem entirely (read-only forensic tools) or side-step it by using AppleScript exclusively for reads. FSEvents does **not** reliably fire on WAL writes; the `poll_monitor` fallback works but at polling cost. The most actionable zombie-row filter is `WHERE ZMARKEDFORDELETION = 0` combined with a folder-type check. TCC automation approval is stored in a system SQLite DB and survives process restarts — a single successful `osascript` call per source/target pair caches it permanently until `tccutil reset`.

---

## (b) Per-Question Findings

### Q1 — Forcing a flush from outside Notes.app

No documented or community-verified method exists to trigger an explicit WAL checkpoint in `NoteStore.sqlite` from outside the process.

- **AppleScript `count notes`**: No evidence this triggers persistence. AppleScript calls dispatch Apple Events into the Notes process but do not invoke `sqlite3_wal_checkpoint`. Community reports confirm that reads immediately after an AppleScript write still show stale SQLite state.
- **CFDistributedNotification / NSUbiquitousKeyValueStore**: No public notification name has been found that instructs Notes.app to flush. Apple's CloudKit sync for Notes is entirely internal.
- **`killall -USR1 Notes`**: No documented effect. POSIX signals are not part of the Notes IPC surface. `-TERM` quits the app (triggering WAL checkpoint as a side effect), but that is destructive.
- **WAL checkpoint from the reader side**: An external process can issue `PRAGMA wal_checkpoint(PASSIVE)` against a copy of the database, but this has no effect on the live file held by Notes.app — SQLite will refuse a full checkpoint while the owner holds a lock.

**Best known approximation of a flush trigger**: quit and relaunch Notes.app (forces WAL compaction), or wait for the app's internal auto-checkpoint (fires at 1000 WAL frames by SQLite default, or at app quit/backgrounding). Neither is suitable for a production MCP server.

Sources: [Notes large NoteStore.sqlite-wal (Apple Community)](https://discussions.apple.com/thread/251187431), [SQLite WAL Checkpoint docs](https://sqlite.org/c3ref/wal_checkpoint_v2.html), [Getting notes out of Apple Notes — Clutterstack](https://clutterstack.com/posts/2024-09-27-applenotes)

---

### Q2 — How open-source tools handle staleness

| Project | Language | Read method | Staleness strategy |
|---|---|---|---|
| `threeplanetssoftware/apple_cloud_notes_parser` | Ruby | SQLite direct (copy first) | **Copies DB to output dir before opening**; no WAL handling, no ZFOLDERTYPE filter; forensic tool — intentionally retains deleted rows |
| `HamburgChimps/apple-notes-liberator` | Java | SQLite direct (copy first) | Creates a copy of the DB, reads from copy; no staleness mitigations documented |
| `sweetrb/apple-notes-mcp` | Node/JS | AppleScript for all ops; SQLite only for checklist state | Has `get-sync-status` tool to detect active iCloud sync; no cache TTL or WAL watch |
| `sirmews/apple-notes-mcp` | Python | SQLite direct (live file) | **No staleness handling at all**; read-only server |
| `storizzi/notes-exporter` | Node | AppleScript exclusively | Incremental export tracked via JSON sidecar files; avoids SQLite entirely |
| `RafalWilinski/mcp-apple-notes` | Python | JXA (JavaScript for Automation) | LanceDB vector cache; no WAL/flush strategy |

**Key pattern**: forensic tools copy the DB first (safe but always stale). MCP servers that need live data do not address staleness at all.

Sources: [apple_cloud_notes_parser README](https://github.com/threeplanetssoftware/apple_cloud_notes_parser/blob/master/README.md), [sweetrb/apple-notes-mcp](https://github.com/sweetrb/apple-notes-mcp), [sirmews/apple-notes-mcp](https://github.com/sirmews/apple-notes-mcp), [storizzi/notes-exporter](https://github.com/storizzi/notes-exporter), [Show HN: Apple Notes Liberator](https://news.ycombinator.com/item?id=35316679)

---

### Q3 — How Apple Notes persists writes

- **iCloud sync cadence**: Not publicly documented. Community reports suggest near-immediate sync attempts on edit, with observable multi-second delays on slow connections. No timer value is documented by Apple.
- **SQLite WAL compaction**: The WAL is checkpointed when Notes.app quits normally, and likely also when the app backgrounds on macOS or under memory pressure. The `-wal` file can grow large if the app stays open for a long time without quitting (confirmed by multiple Apple Community threads with GB-sized WAL files).
- **"Last persisted" timestamp**: No documented column. `ZMODIFICATIONDATE1` reflects the last in-app edit time, not the last SQLite flush time.
- **Auto-checkpoint threshold**: SQLite default is 1000 pages; Notes.app may override this, but no source confirms the actual value.

Sources: [Notes large .sqlite-wal (Apple Community)](https://discussions.apple.com/thread/251187431), [Notes App iCloud Sync/Stability (Apple Developer Forums)](https://developer.apple.com/forums/thread/86677)

---

### Q4 — FSEvents / file watching `NoteStore.sqlite-wal`

FSEvents **does not reliably fire** on SQLite WAL file modifications. This is a documented, long-standing macOS limitation: the WAL write path bypasses the FSEvents kernel notification.

- **Workaround**: `fswatch -m poll_monitor NoteStore.sqlite-wal` reliably detects changes using stat-based polling, but introduces polling overhead (default 1-second interval).
- **Practical recommendation**: Watching `NoteStore.sqlite` (the main file, not `-wal`) with FSEvents does fire on WAL checkpoints (i.e., when data actually lands in the main file). This gives a coarser "data is definitely flushed" signal rather than a "write just happened" signal.

Source: [fswatch issue #150 — not detecting changes to SQLite in WAL mode](https://github.com/emcrisostomo/fswatch/issues/150)

---

### Q5 — Zombie-row detection columns

No single definitive public source documents all deletion markers, but forensic research and community SQL samples point to these reliable candidates:

- **`ZMARKEDFORDELETION`** on `ZICCLOUDSYNCINGOBJECT`: set to `1` when the note/object is logically deleted. The Velociraptor artifact SQL does **not** filter this out (intentional for forensics). For a live MCP server, add `WHERE ZMARKEDFORDELETION = 0` (or `IS NULL`) to exclude soft-deleted rows.
- **`ZFOLDER` / folder `ZFOLDERTYPE`**: The Recently Deleted folder has a distinct `ZFOLDERTYPE` value (commonly observed as `1` in schema dumps, versus `0` for normal folders). Filter notes whose parent folder has `ZFOLDERTYPE = 1` to exclude the Recently Deleted bucket. **Caution**: exact integer values have not been confirmed in an authoritative public source — verify against a live DB.
- **`ZISPASSWORDPROTECTED`**, **`ZISRECENTLYDELETED`**: Some schema diagrams list `ZISRECENTLYDELETED` as a column, but this has not been independently confirmed in macOS 14/15 schema dumps. Treat as unverified.
- **Zombie rows that survive 30-day purge**: These appear as rows where `ZMARKEDFORDELETION = 1` and `ZFOLDER` references the Recently Deleted folder, but the 30-day auto-purge has not yet triggered SQLite row removal. No timestamp column reliably encodes "purge-by" date — the 30-day window is managed by the app, not stamped in the DB.

Sources: [Velociraptor macOS.Applications.Notes artifact](https://docs.velociraptor.app/exchange/artifacts/pages/macos.applications.notes/), [Ciofeca Forensics — Revisiting Apple Notes (1)](https://www.ciofecaforensics.com/2020/01/10/apple-notes-revisited/), [Ciofeca Forensics — iOS 18 Notes](https://www.ciofecaforensics.com/2024/12/10/ios18-notes/)

---

### Q6 — macOS Automation permission (TCC) pre-warming

- **Approval is permanent per source/target pair**: TCC stores the grant in a system SQLite DB (`/Library/Application Support/com.apple.TCC/TCC.db`). Once approved, all subsequent `osascript` calls from the same bundle/binary to the same target app are allowed without re-prompting, **across process restarts**.
- **Pre-warming strategy**: Issue a cheap, idempotent AppleScript call at server startup (e.g., `tell application "Notes" to get name`). If the permission has never been granted, this triggers the one-time prompt. All subsequent tool calls in that session (and future sessions) proceed silently.
- **CLI / unsigned binary caveat**: For unsigned CLI binaries, TCC attributes the automation permission to the **parent process** (e.g., Terminal, Claude Desktop), not the binary itself. To make a CLI tool "own" its own TCC entry, it must be signed with a Developer ID and embed an `Info.plist` with `NSAppleEventsUsageDescription`. See Steipete's 2025 guide for the `responsibility_spawnattrs_setdisclaim` undocumented API approach.
- **PPPC profiles**: MDM-managed machines can pre-approve via a Privacy Preferences Policy Control profile — not applicable for consumer deployments.

Sources: [scriptingosx.com — Avoiding AppleScript Security and Privacy Requests](https://scriptingosx.com/2020/09/avoiding-applescript-security-and-privacy-requests/), [steipete.me — Making AppleScript Work in macOS CLI Tools (2025)](https://steipete.me/posts/2025/applescript-cli-macos-complete-guide), [macOS TCC — HackTricks](https://angelica.gitbook.io/hacktricks/macos-hardening/macos-security-and-privilege-escalation/macos-security-protections/macos-tcc)

---

## (c) Concrete Recommendations for notes-mcp

1. **Post-write read lag**: After any AppleScript write, enforce a minimum 500 ms–1 s delay before querying SQLite. No flush mechanism exists; this is the only reliable mitigation. Alternatively, re-read via AppleScript (not SQLite) immediately after a write.

2. **Zombie rows**: Add `WHERE n.ZMARKEDFORDELETION = 0` (or `IS NULL`) to all note-listing queries. Additionally join to the folder table and exclude notes where the parent folder has `ZFOLDERTYPE = 1` (Recently Deleted). Confirm the exact integer value against a live DB before shipping.

3. **Wrong ZFOLDER FK (moved notes)**: For folder membership, prefer querying via AppleScript (`folder of note x`) over SQLite when freshness matters. SQLite folder FK is the stalest field.

4. **Cache freshness signal**: Watch `NoteStore.sqlite` (main file) with FSEvents for checkpoint events as a coarse "cache definitely stale" invalidation. Do **not** rely on FSEvents for `-wal` changes — use `poll_monitor` if finer granularity is needed, accepting the polling overhead.

5. **TCC pre-warming**: At server startup, issue `osascript -e 'tell application "Notes" to get name'`. This is the standard pre-warm pattern; it triggers the one-time prompt if needed and caches approval for all subsequent calls.

6. **DB access pattern**: Never read the live `NoteStore.sqlite` file directly for bulk operations while Notes.app is open. Either (a) use WAL-mode read (SQLite's default reader isolation is sufficient for SELECT queries, but rows written after WAL may not be visible) or (b) make a snapshot copy for expensive scans. For lightweight single-note lookups, reading the live file with `PRAGMA journal_mode=WAL` is safe.

---

## (d) Uncertainties / Unverified Items

- **Exact `ZFOLDERTYPE` integer values**: Not confirmed in a public authoritative source for macOS 14/15. Must be verified empirically against a live NoteStore.sqlite.
- **`ZISRECENTLYDELETED` column existence on macOS**: Referenced in some schema diagrams but not confirmed in macOS 14/15 forensic writeups (Ciofeca's iOS 18 article does not mention it).
- **Auto-checkpoint page threshold**: Notes.app may override SQLite's default 1000-page threshold — unverified.
- **`count notes` as flush trigger**: No one has definitively tested and published whether this AppleScript idiom triggers WAL checkpointing. Empirical testing needed.
- **WAL checkpoint on Notes.app backgrounding**: Likely (standard macOS app lifecycle), but not confirmed in any public source.
- **TCC behaviour for unsigned `node` binary**: If the MCP server runs as a plain `node` process, TCC may attach the automation permission to Terminal/Claude Desktop rather than `node` itself, causing re-prompts after parent process change. Needs empirical testing in the actual deployment context.

---

## Storage format verification (follow-up)

_Verification date: April 2026_

### Verdict: **Confirmed** — with one important nuance on encrypted notes.

The claim that `ZICNOTEDATA.ZDATA` holds gzip-compressed protobuf (not HTML) is correct for all **unlocked** notes. Encrypted (locked) notes store a **binary plist (NSKeyedArchiver)** wrapper, not bare gzip, around the AES-GCM ciphertext — the "encrypted bodies" claim is also correct but the exact container format is bplist, not raw bytes.

### Per-note hex probe table

| Z_PK | Title (truncated) | First 3 bytes | Format | Encrypted | Decompressed size | HTML tag count |
|------|-------------------|--------------|--------|-----------|-------------------|---------------|
| 1 | You are late for work… | `1f 8b 08` | gzip+protobuf | No | 62,986 B | 0 |
| 6 | Curry | `1f 8b 08` | gzip+protobuf | No | 5,489 B | 0 |
| 7 | (locked) | `62 70 6c` | bplist (encrypted) | Yes | 2,937 B | 0 |
| 8 | (locked) | `62 70 6c` | bplist (encrypted) | Yes | 2,457 B | 0 |
| 11 | Places in scotland | `1f 8b 08` | gzip+protobuf | No | 4,692 B | 0 |

### Distinct HTML tags found across 5 gzip notes

None. `re.findall(rb"<[a-z][a-z0-9]*\b", raw)` returned an empty set across all decompressed blobs. The decompressed binary is pure protobuf (confirmed: first byte `0x08`, field-1/wire-type-0 varint; note text appears as raw UTF-8 strings inside the protobuf framing, e.g. `\x0a\nYou are late for work…`).

### Cross-reference table

| Source | URL | What it says |
|--------|-----|-------------|
| apple_cloud_notes_parser README | https://github.com/threeplanetssoftware/apple_cloud_notes_parser | "takes the gzipped blob in the ZDATA field, gunzips it, and parses the protobuf that is inside" — adds `ZPLAINTEXTDATA` column back |
| Ciofeca Forensics – Revisiting Apple Notes (1) | https://ciofecaforensics.com/2020/01/10/apple-notes-revisited/ | "note was stored as a protocol buffer (protobuf) that was gzipped and put into the database as a blob" since iOS 9 |
| Ciofeca Forensics – Encrypted Notes | https://www.ciofecaforensics.com/2020/07/31/apple-notes-revisited-encrypted-notes/ | ZDATA for locked notes begins `0x1f 0x8b` after decryption; confirmed AES-GCM encryption of note + attachments |
| macosxautomation AppleScript Notes docs | https://www.macosxautomation.com/applescript/notes/04.html | `body (text) : the HTML content of the note` — AppleScript property defined as HTML; no storage mechanism described |
| Ciofeca Forensics – CloudKit Data | https://ciofecaforensics.com/2020/10/20/apple-notes-cloudkit-data/ | iCloud sync uses CloudKit `CKRecord` + NSKeyedArchiver; `ZMERGEABLEDATA` is gzip-compressed; no HTML in sync wire format |

### Corrections

One nuance: locked/encrypted notes in `ZICNOTEDATA.ZDATA` are wrapped in a **binary plist (NSKeyedArchiver / `bplist00` magic)**, not stored as raw encrypted bytes. The IV and tag live separately in `ZCRYPTOINITIALIZATIONVECTOR` / `ZCRYPTOTAG` on `ZICCLOUDSYNCINGOBJECT`. This is consistent with the original claim but more precise.

No HTML files were found anywhere under `~/Library/Group Containers/group.com.apple.notes/`. The `NotesIndexerState-HTML` file is a small XML plist (Spotlight indexer state token), not cached HTML content.

### Why the user sees HTML

1. **AppleScript render path**: Notes.app exposes `body of note` as `text (HTML)` per its scripting dictionary. When an Apple Event requests the body, Notes.app deserializes the protobuf in-process and renders an HTML string on demand. The HTML is returned transiently in the Apple Event reply buffer — never written to disk.
2. **iCloud sync**: Uses CloudKit CKRecords with NSKeyedArchiver-encoded protobuf payloads (`ZMERGEABLEDATA`). HTML is never transmitted.
3. **Markdown export (macOS Tahoe / macOS 26)**: Native `File > Export as > Markdown` is implemented inside Notes.app. The most likely pipeline is protobuf → internal attributed-string model → Markdown text; there is no evidence of an intermediate HTML file on disk. Third-party exporters (e.g. `apple-notes-exporter`) that produce HTML do so by calling AppleScript `body of note` and post-processing the returned HTML string.
4. **No HTML cache**: The `Thumbnails/` and `Previews/` directories contain image/PDF previews of attachments, not HTML.
