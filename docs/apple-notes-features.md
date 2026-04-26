# Apple Notes Feature Reference for MCP Server Development

> macOS Sequoia / iOS 18 era. Verified against live NoteStore.sqlite (April 2026).
> Schema verified on: `~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite`

---

## Section 1 — Feature Inventory

All note content lives in `ZICNOTEDATA.ZDATA` as a **gzip-compressed NoteStoreProto protobuf**. The proto has `note_text` (plain string; U+FFFC marks embedded objects) and `attribute_run[]` arrays encoding styling per character range. `ZMERGEABLEDATA`/`ZMERGEABLEDATA1` on `ZICCLOUDSYNCINGOBJECT` are CRDTs for tables and drawings.

### Paragraph styles, inline formatting, lists

| Feature | Proto | AppleScript `body` HTML | Notes |
|---|---|---|---|
| Title | `style_type=0` | `<div><h1>` | `ZTITLE1` mirrors it |
| Heading | `style_type=1` | `<div><h2>` | — |
| Subheading | `style_type=2` | `<div><h3>` | — |
| Monostyled / Code Block | `style_type=4` | `<div><tt>` | Added Sonoma/iOS 17; no syntax highlighting |
| Body | — | `<div>` | — |
| Bold | `font_weight` (field 5) | `<b>` | — |
| Italic | font name | `<i>` | — |
| Underline | `underlined` (field 6) | `<u>` | — |
| Strikethrough | `strikethrough` (field 7) | `<s>` | — |
| Superscript/Subscript | `superscript` (field 8, ±1) | `<sup>`/`<sub>` | — |
| Link (http/notes://) | `link` (field 9, string) | `<a href="…">` | `notes://` internal links preserved |
| Highlight (5 colours) | AttributeRun field 14 | **not in body HTML** | Added iOS 18 / Sequoia |
| Bulleted list | `style_type=100` | `<ul><li>` | — |
| Dashed list | `style_type=101` | `<ul><li>` | dash distinction lost in HTML |
| Numbered list | `style_type=102` | `<ol><li>` | — |
| Checklist | `style_type=103` | `<ul><li>` | **tick state stripped by AppleScript** |

Checklist tick state: `ParagraphStyle.checklist.done` (Checklist sub-message, field 2). SQL pre-filters: `ZHASCHECKLIST`, `ZHASCHECKLISTINPROGRESS`. Nesting via `ParagraphStyle.indent_amount` (field 4).

### Complex embedded objects

| Feature | UTI / column | AppleScript body | Notes |
|---|---|---|---|
| Table | `com.apple.notes.table`; `ZMERGEABLEDATA1` | U+FFFC only | CRDT proto; complex nested UUID structure |
| Image/Photo | `public.jpeg/png/heic`; disk `Media/<UUID>/` | `<img src="data:…base64…">` | Confirmed in live output |
| Sketch/Drawing | `com.apple.drawing.2`; `ZMERGEABLEDATA1` | U+FFFC | Preview JPEG at `FallbackImages/<id>.jpg`; OCR text in `ZADDITIONALINDEXABLETEXT` |
| Scanned doc | `ZFALLBACKPDFGENERATION`; `FallbackPDFs/` | U+FFFC | Multi-page PDF |
| File/audio attachment | UTI varies; disk `Media/<UUID>/` | U+FFFC | `ZTYPEUTI`, `ZFILENAME` columns |

### Metadata features

| Feature | SQL column / location | AppleScript | Notes |
|---|---|---|---|
| Tags (`#tag`) | `ZTOKENCONTENTIDENTIFIER` on Z_ENT=9 rows (ICInlineAttachment); UTI `com.apple.notes.inlinetextattachment.hashtag` | Raw `#tag` text in body | Added iOS 15; uppercased, no `#` in column |
| Smart folders | `ZSMARTFOLDERQUERYJSON` (JSON) on ICFolder | Not visible | `ZFOLDERTYPE=0`; distinguish by non-NULL JSON |
| Pinned | `ZISPINNED` (0/1) | **Absent from dictionary** | Must use SQLite |
| Locked | `ZISPASSWORDPROTECTED=1`; `ZDATA` AES-GCM encrypted | `password protected` (bool, confirmed) | Title/dates unencrypted; body returns empty HTML |
| Shared | `ZINVITATION` FK → ZICINVITATION | `shared` (bool, confirmed) | `ZSHAREURL` in ZICINVITATION |
| Quick Notes | Folder `ZIDENTIFIER=QuickNotesFolder-CloudKit` | Folder visible | `ZFOLDERTYPE=0`; no special int flag |
| Recently Deleted | `ZMARKEDFORDELETION=1` + `ZFOLDER→TrashFolder-CloudKit` (`ZFOLDERTYPE=1`) | Invisible | Purged after ~30 days |
| Accounts | Z_ENT=14; `ZACCOUNTTYPE`: 1=iCloud | `folders of account` | On My Mac = separate SQLite at `~/Library/Containers/com.apple.Notes/…` |
| Version history | `ZREPLICAIDTONOTESVERSIONDATA` BLOB | None | No public schema; not readable |
| Sequoia extras | `ZOUTLINESTATEDATA` (collapsible headings), `ZNEEDSTRANSCRIPTION`, math inline attachments | None | iOS 18 / macOS 15 |

Source (iOS 18 schema): [Ciofeca Forensics iOS 18](https://www.ciofecaforensics.com/2024/12/10/ios18-notes/)

---

## Section 2 — AppleScript API Surface

`tell application "Notes"` — all confirmed on macOS Sequoia.

**Readable:** `name`, `id` (`x-coredata://…/ICNote/p<n>`), `body` (HTML with base64 images), `modification date`, `creation date`, `container`, `password protected` (bool), `shared` (bool), `attachments` (list; each has `id`, `name`, `creation date`, `modification date`, `url`, `container`).

**NOT exposed:** `pinned`, checklist tick state, highlight colour, collapsed-heading state, tag objects.

**Write:** `make new note` with `body` HTML; `set body of note` — but this **silently destroys all existing attachments**. Cannot set `pinned`, tick state, lock/unlock, or table cells. No direct file-attach command.

| Feature | Minimum macOS |
|---|---|
| Tags `#` / Quick Notes | Monterey 12 |
| Monostyled / Code Block | Sonoma 14 |
| Math, Audio transcription, Highlights, Collapsible headings | Sequoia 15 |

---

## Section 3 — SQLite Schema Reference

**Database:** `~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite`

One mega-table `ZICCLOUDSYNCINGOBJECT` holds all objects. `Z_ENT` (from `Z_PRIMARYKEY`) distinguishes types: 5=ICAttachment, 9=ICInlineAttachment (tags), 11=ICMedia, 12=ICNote, 14=ICAccount, 15=ICFolder.

### Key columns on `ZICCLOUDSYNCINGOBJECT`

| Column | Entity | Meaning |
|---|---|---|
| `ZTITLE1` | ICNote | Note title |
| `ZTITLE2` | ICFolder | Folder display name |
| `ZFOLDER` | ICNote | FK → ICFolder Z_PK |
| `ZNOTEDATA` | ICNote | FK → ZICNOTEDATA Z_PK |
| `ZMODIFICATIONDATE1` | ICNote | Modified timestamp (Core Data epoch: 2001-01-01) |
| `ZCREATIONDATE1` | ICNote | Created timestamp |
| `ZMARKEDFORDELETION` | ICNote | 1 = Recently Deleted |
| `ZISPINNED` | ICNote | 1 = pinned |
| `ZISPASSWORDPROTECTED` | ICNote | 1 = AES-GCM encrypted |
| `ZIDENTIFIER` | ICNote/ICFolder | Stable UUID; special: `TrashFolder-CloudKit`, `DefaultFolder-CloudKit`, `QuickNotesFolder-CloudKit` |
| `ZACCOUNT3` | ICNote | FK → ICAccount Z_PK |
| `ZFOLDERTYPE` | ICFolder | 0 = user/smart folder, 1 = Recently Deleted |
| `ZSMARTFOLDERQUERYJSON` | ICFolder | Non-NULL = Smart Folder JSON predicate |
| `ZHASCHECKLIST` / `ZHASCHECKLISTINPROGRESS` | ICNote | Cheap checklist flags |
| `ZINVITATION` | ICNote | Non-NULL = shared (FK → ZICINVITATION) |
| `ZTOKENCONTENTIDENTIFIER` | ICInlineAttachment | Uppercase tag string (no `#`) |
| `ZTYPEUTI` / `ZFILENAME` | ICAttachment | Attachment type UTI and filename |
| `ZACCOUNTTYPE` | ICAccount | 1=iCloud; On My Mac/Exchange/Gmail use other values |

### `ZICNOTEDATA` table

`ZNOTE` (FK → ICNote) + `ZDATA` (gzip NoteStoreProto; AES-GCM ciphertext if locked).

### FTS

**No FTS shadow table exists in NoteStore.sqlite.** The `NotesIndexerState-Modern` and `NotesIndexerState-HTML` files in the group container are opaque blobs. For text search: use AppleScript `search` command, or scan `ZTITLE1`, `ZSUMMARY`, `ZADDITIONALINDEXABLETEXT` columns.

### Attachment files on disk (confirmed structure)

```
Accounts/<ACCOUNT-ZIDENTIFIER>/
  Media/<ATTACHMENT-ZIDENTIFIER>/<counter>_<UUID>/<filename>  ← images, files, audio
  FallbackImages/<ATTACHMENT-ZIDENTIFIER>.jpg                 ← sketch previews
  FallbackPDFs/                                               ← scan PDFs
```

Sources: [Ciofeca Forensics Tables](https://www.ciofecaforensics.com/2020/01/14/apple-notes-revisited-embedded-tables/) · [notestore.proto](https://github.com/HamburgChimps/apple-notes-liberator/blob/main/src/main/proto/notestore.proto)

---

## Section 4 — MCP Server Recommendations

**Cheap wins (no protobuf needed):**
- `ZISPINNED` — SQLite only (not in AppleScript); trivial to expose
- Tag list — `SELECT ZTOKENCONTENTIDENTIFIER FROM … WHERE Z_ENT=9` joined to note
- Lock/shared/checklist-presence flags — SQL columns or AppleScript `password protected`/`shared`
- Attachment listing (`ZTYPEUTI`, `ZFILENAME`) — SQL join, no parsing
- Recently-deleted / folder filtering — `ZMARKEDFORDELETION`, `ZFOLDER`, `ZFOLDERTYPE`

**Traps:**
- **Protobuf parsing** — `ZDATA` field numbers shift across OS versions; use [apple_cloud_notes_parser](https://github.com/threeplanetssoftware/apple_cloud_notes_parser) as reference schema; never assume field positions
- **Checklist tick state** — `body of note` strips it; must parse `ZDATA` protobuf (`Checklist.done` field); there is no SQL shortcut for per-item state
- **Tables** — body returns U+FFFC; content needs CRDT parse from `ZMERGEABLEDATA1`
- **Locked notes** — `ZDATA` is AES-GCM ciphertext; `body` returns empty HTML; unreadable without passphrase
- **`set body` destroys attachments** — any `set body of note` call silently removes all embedded content
- **Multiple accounts** — On My Mac notes are in a *different* SQLite (`~/Library/Containers/com.apple.Notes/…`); Group Container DB only covers iCloud/Exchange/Gmail

**Permissions:**
- **Automation only** — `body`, `name`, dates, `shared`, `password protected`, attachment metadata
- **Full Disk Access required** — SQLite reads, `ZISPINNED`, tags, checklist state, resolving `attachment url`, reading binary files from `Media/`

**All sources:**
[apple_cloud_notes_parser](https://github.com/threeplanetssoftware/apple_cloud_notes_parser) · [Ciofeca iOS 18](https://www.ciofecaforensics.com/2024/12/10/ios18-notes/) · [Ciofeca iOS 15/Tags](https://www.ciofecaforensics.com/2021/11/08/ios-15-changes/) · [Ciofeca Encrypted](https://www.ciofecaforensics.com/2020/07/31/apple-notes-revisited-encrypted-notes/) · [Ciofeca Attachments](https://ciofecaforensics.com/2020/01/13/apple-notes-revisited-easy-embedded-objects/) · [Ciofeca Tables](https://www.ciofecaforensics.com/2020/01/14/apple-notes-revisited-embedded-tables/) · [notestore.proto](https://github.com/HamburgChimps/apple-notes-liberator/blob/main/src/main/proto/notestore.proto) · [Apple Security/encryption](https://support.apple.com/guide/security/secure-features-in-the-notes-app-sec1782bcab1/web) · [apple-notes-mcp](https://github.com/sweetrb/apple-notes-mcp)
