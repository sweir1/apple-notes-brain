"""User-config layer — survives `pip install --upgrade apple-notes-brain`.

Mirrors obsidian-brain's `src/embeddings/user-config.ts`. Two files live in
a per-user config directory outside the Python package:

  - `seed-models.json` — a user-fetched seed (refreshed via a future
    `apple-notes-brain models fetch-seed` CLI). When present, takes
    priority over the bundled seed. Lets users pull in upstream MTEB
    fixes without waiting for a PyPI release.
  - `model-overrides.json` — hand-edited per-model overrides for any of
    `max_tokens`, `query_prefix`, `document_prefix`. Layered on top of
    the resolved metadata in `metadata_resolver` (Phase δ wires it in).
    Used to correct upstream errors locally (e.g. "MTEB says
    max_tokens=1024 but the real model only supports 512" — set the
    override and ship it via dotfiles).

Resolution order including these layers (Phase δ chain):
  override (this layer) → cache → seed (user-fetched > bundled) → HF →
  embedder probe → safe defaults.

Override changes are picked up on next process boot. The prefix-strategy
hash in `bootstrap` (Phase η) detects `query_prefix` / `document_prefix`
changes and forces a reindex. `max_tokens` overrides take effect on the
next reindex (they don't auto-trigger one — chunker behaviour changes,
but existing vectors stay valid).

Path resolution (XDG-compliant):
  - `$APPLE_NOTES_BRAIN_CONFIG_DIR` (explicit override; takes precedence)
  - `$XDG_CONFIG_HOME/apple-notes-brain/` if set
  - `~/.config/apple-notes-brain/` (most common path on macOS, where
    `XDG_CONFIG_HOME` is rarely set)

The directory is created on first access by `get_user_config_dir()`.
The individual config files are NOT created here — that's the caller's
job (load_overrides() returns empty on missing file; save_override()
mkdir+writes lazily).
"""
from __future__ import annotations

import os
from pathlib import Path


ENV_CONFIG_DIR = "APPLE_NOTES_BRAIN_CONFIG_DIR"
_PACKAGE_DIRNAME = "apple-notes-brain"


def get_user_config_dir() -> Path:
    """Return the absolute path to the per-user apple-notes-brain config dir.

    Resolution order:
      1. `$APPLE_NOTES_BRAIN_CONFIG_DIR` (used verbatim if non-empty after
         stripping whitespace).
      2. `$XDG_CONFIG_HOME/apple-notes-brain` (if `XDG_CONFIG_HOME` set
         and non-empty after stripping).
      3. `~/.config/apple-notes-brain` (the default — most macOS users
         don't have `XDG_CONFIG_HOME` set so this is the common path).

    The directory is created on first access via `mkdir(parents=True,
    exist_ok=True)`. Individual files (seed-models.json,
    model-overrides.json) are NOT created — callers handle those.
    """
    explicit = os.environ.get(ENV_CONFIG_DIR)
    if explicit and explicit.strip():
        path = Path(explicit.strip()).expanduser()
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg and xdg.strip():
            path = Path(xdg.strip()).expanduser() / _PACKAGE_DIRNAME
        else:
            path = Path.home() / ".config" / _PACKAGE_DIRNAME

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_seed_path() -> Path:
    """Absolute path to the user-fetched seed JSON.

    The file may not exist; callers (seed loader) must tolerate its
    absence. Created by the future `models fetch-seed` CLI.
    """
    return get_user_config_dir() / "seed-models.json"


def get_overrides_path() -> Path:
    """Absolute path to the user model-overrides JSON.

    The file may not exist; `overrides.load_overrides()` returns an empty
    map in that case. Created lazily by `overrides.save_override()`.
    """
    return get_user_config_dir() / "model-overrides.json"
