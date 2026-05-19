"""Env-var configuration for the semantic subsystem.

All env vars consumed by `apple_notes_brain.semantic.*` resolve through
this module. Centralising keeps the surface auditable (a single grep for
`os.environ` covers every config knob) and makes it trivial to mock in
tests via monkeypatch.

Defaults are conservative and prefer "works out of the box on macOS with
no setup" over "minimum install footprint":
- data dir lives under `~/.local/share/apple-notes-brain` (XDG-style)
- ONNX embedder ships as the default provider (no Ollama dependency)
- the watcher polls every 30s but only re-indexes when `PRAGMA data_version`
  actually changed (so an idle Notes.app is free)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Env-var names — single source of truth (referenced from server.json and
# the README; rename here and the rest follows).
# ---------------------------------------------------------------------------

ENV_DATA_DIR = "APPLE_NOTES_BRAIN_DATA_DIR"
ENV_MODEL_CACHE = "APPLE_NOTES_BRAIN_MODEL_CACHE"
ENV_NO_WATCH = "APPLE_NOTES_BRAIN_NO_WATCH"
ENV_INDEX_INTERVAL = "APPLE_NOTES_BRAIN_INDEX_INTERVAL"
ENV_MAX_CHUNK_TOKENS = "APPLE_NOTES_BRAIN_MAX_CHUNK_TOKENS"
ENV_DEBUG = "APPLE_NOTES_BRAIN_DEBUG"

ENV_PROVIDER = "EMBEDDING_PROVIDER"
ENV_MODEL = "EMBEDDING_MODEL"
ENV_PRESET = "EMBEDDING_PRESET"
ENV_DIM = "EMBEDDING_DIM"
ENV_ONNX_PROVIDERS = "EMBEDDING_ONNX_PROVIDERS"

ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
ENV_OLLAMA_NUM_CTX = "OLLAMA_NUM_CTX"
ENV_OLLAMA_AUTO_PULL = "APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL"

# ---------------------------------------------------------------------------
# Defaults — chosen to match obsidian-brain where they overlap
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER: Literal["onnx", "ollama"] = "onnx"
# Concrete HF model id of the default preset (`english`). Resolved via
# `resolve_preset_config()`; kept as a module constant for back-compat
# imports (tests assert on it).
DEFAULT_MODEL_PRESET = "Xenova/bge-small-en-v1.5"  # 384-dim, ~30MB ONNX-quantised
DEFAULT_INDEX_INTERVAL_SECONDS = 30.0
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_NUM_CTX_FALLBACK = 8192


# ---------------------------------------------------------------------------
# Parsing helpers — small, picky, and aggressively defensive about garbage
# input. Bad env values raise ValueError with a clear message naming the
# offending variable.
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var. Accepts 1/0, true/false, yes/no, on/off."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a valid boolean. "
        f"Use one of: 1/0, true/false, yes/no, on/off."
    )


def _int_env(name: str, default: int, *, min_value: int | None = None) -> int:
    """Parse an integer env var. Optionally enforce a minimum."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} is not a valid integer."
        ) from exc
    if min_value is not None and value < min_value:
        raise ValueError(
            f"{name}={value} is below the minimum allowed value {min_value}."
        )
    return value


def _float_env(name: str, default: float, *, min_value: float | None = None) -> float:
    """Parse a float env var. Optionally enforce a minimum."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} is not a valid number."
        ) from exc
    if min_value is not None and value < min_value:
        raise ValueError(
            f"{name}={value} is below the minimum allowed value {min_value}."
        )
    return value


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_data_dir() -> Path:
    """Return the directory where semantic state lives (DB + model cache).

    Resolution order:
      1. $APPLE_NOTES_BRAIN_DATA_DIR (if set, used verbatim)
      2. $XDG_DATA_HOME/apple-notes-brain (if XDG_DATA_HOME set)
      3. ~/.local/share/apple-notes-brain
    The directory is created on first access; parents must exist.
    """
    raw = os.environ.get(ENV_DATA_DIR)
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            path = Path(xdg).expanduser().resolve() / "apple-notes-brain"
        else:
            path = Path.home() / ".local" / "share" / "apple-notes-brain"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_cache() -> Path:
    """Return the directory where embedder models are cached.

    Defaults to `<data_dir>/models`. Overridable via env so users with an
    existing `~/.cache/huggingface` setup can point at it.
    """
    raw = os.environ.get(ENV_MODEL_CACHE)
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        path = resolve_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_db_path() -> Path:
    """Return the absolute path to the semantic-index sqlite file."""
    return resolve_data_dir() / "semantic_index.db"


# ---------------------------------------------------------------------------
# Typed config snapshot — read once at boot, immutable thereafter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticConfig:
    """Snapshot of every env-derived setting the semantic subsystem reads.

    Built via `load_config()`. Tests can construct one directly for
    determinism instead of monkeypatching env vars.
    """
    data_dir: Path
    db_path: Path
    model_cache: Path
    no_watch: bool
    index_interval_s: float
    max_chunk_tokens_override: int | None
    debug: bool

    provider: Literal["onnx", "ollama"]
    model: str
    # Canonical short name of the preset that produced (provider, model),
    # or None when the user set EMBEDDING_MODEL explicitly (the
    # "power-user" path) or `EMBEDDING_PROVIDER=ollama` alone (no preset
    # implied — falls back to DEFAULT_OLLAMA_MODEL). Resolved atomically
    # alongside provider/model via `resolve_preset_config()`.
    preset_short_name: str | None
    dim_override: int | None
    onnx_providers_override: tuple[str, ...] | None

    ollama_base_url: str
    ollama_num_ctx_override: int | None
    ollama_auto_pull: bool


def load_config() -> SemanticConfig:
    """Materialise the full env-derived config. Safe to call many times;
    each call re-reads os.environ so tests can monkeypatch between calls.

    Provider + model + preset_short_name are resolved atomically via
    `resolve_preset_config()` in `embedder/presets.py` so they cannot
    drift from each other. Calling that resolver also performs the
    EMBEDDING_PROVIDER validity check, so we drop the separate
    `provider_raw not in {…}` guard.
    """
    # Imported lazily so config.py stays import-cheap (tests that build
    # SemanticConfig directly via the dataclass don't pay the cost).
    from .embedder.presets import resolve_preset_config

    resolved = resolve_preset_config(os.environ)

    ep_raw = os.environ.get(ENV_ONNX_PROVIDERS, "").strip()
    onnx_providers: tuple[str, ...] | None = None
    if ep_raw:
        onnx_providers = tuple(p.strip() for p in ep_raw.split(",") if p.strip())
        if not onnx_providers:
            onnx_providers = None

    return SemanticConfig(
        data_dir=resolve_data_dir(),
        db_path=resolve_db_path(),
        model_cache=resolve_model_cache(),
        no_watch=_bool_env(ENV_NO_WATCH, default=False),
        index_interval_s=_float_env(
            ENV_INDEX_INTERVAL,
            default=DEFAULT_INDEX_INTERVAL_SECONDS,
            min_value=1.0,
        ),
        max_chunk_tokens_override=(
            _int_env(ENV_MAX_CHUNK_TOKENS, default=0, min_value=1) or None
            if os.environ.get(ENV_MAX_CHUNK_TOKENS)
            else None
        ),
        debug=_bool_env(ENV_DEBUG, default=False),
        provider=resolved.provider,
        model=resolved.model,
        preset_short_name=resolved.preset_short_name,
        dim_override=(
            _int_env(ENV_DIM, default=0, min_value=1) or None
            if os.environ.get(ENV_DIM)
            else None
        ),
        onnx_providers_override=onnx_providers,
        ollama_base_url=os.environ.get(ENV_OLLAMA_BASE_URL, DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        ollama_num_ctx_override=(
            _int_env(ENV_OLLAMA_NUM_CTX, default=0, min_value=1) or None
            if os.environ.get(ENV_OLLAMA_NUM_CTX)
            else None
        ),
        ollama_auto_pull=_bool_env(ENV_OLLAMA_AUTO_PULL, default=True),
    )
