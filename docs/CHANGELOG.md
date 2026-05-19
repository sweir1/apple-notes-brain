# Changelog

All notable changes to apple-notes-brain are recorded here. The format follows the convention used by sibling project [obsidian-brain](https://github.com/sweir1/obsidian-brain) — one `## vX.Y.Z — YYYY-MM-DD — title` heading per release (em dash, not hyphen). Headings in that exact shape are parsed by:

- `scripts/gen_readme_recent.py` — refreshes the **Recent releases** block in `README.md`.
- `release.yml` — extracts release notes for the GitHub Release.
- `website/main.py` macros — feeds the home-page recent-releases block on the docs site.

Don't free-form these headings. Edits below the heading line are unconstrained.

## v1.1.0 — 2026-05-19 — Semantic search + HF metadata resolver chain

- New optional `[semantic]` install extra: `semantic_search`, `hybrid_search`, `reindex_semantic`, `semantic_index_status` tools.
- Six named embedding presets mirroring obsidian-brain: `english`, `english-fast`, `english-quality`, `multilingual`, `multilingual-quality`, `multilingual-ollama`.
- HF metadata resolver chain — bundled `seed-models.json`, on-disk cache, live HF fallback, user overrides.
- Ollama provider with auto-pull (gated by `APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL`).
- Background semantic-reindex watcher with idle-pause + Notes.app-closed-skip + sleep-freeze.
- HTML wedge fix: adversarial HTML + stuck `osascript` no longer deadlocks the server.
- Trash exclusion in semantic search; score provenance on hybrid results.
- ~90 new tests covering the resolver chain, capacity tracking, and preset resolution.

## v1.0.3 — 2026-04-21 — First PyPI publish

- First public release on PyPI as `apple-notes-brain`.
- 11 lexical CRUD tools (`list_folders`, `list_notes`, `search_notes`, `get_note`, `get_notes`, `create_note`, `update_note`, `rename_note`, `move_note`, `create_folder`, `delete_note`).
- Markdown ↔ Apple HTML round-trip with checklist + table support.
- Cursor pagination, short 4–6 char IDs, locked-note handling.
- Attachment-destructive-write guard on `update_note`.
- Background cache-coherence refresh thread.
- macOS one-line installer (`scripts/install.sh`) with TCC + Full Disk Access walkthrough.
- OIDC trusted publishing to PyPI.
