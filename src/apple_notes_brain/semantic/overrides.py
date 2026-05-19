"""Model-metadata overrides — user-controlled, survives `pip install --upgrade`.

Mirrors obsidian-brain's `src/embeddings/overrides.ts`.

Reads / writes `<user_config_dir>/model-overrides.json` (XDG-compliant
via `user_config.get_overrides_path()`). Each override is a partial
patch on top of the resolved metadata for a model id — set just
`max_tokens`, just `query_prefix`, or any combination. Omitted fields
keep the underlying (cache → seed → HF → ...) value.

File shape (v1):

  {
    "$version": 1,
    "models": {
      "BAAI/bge-small-en-v1.5": { "maxTokens": 512 },
      "intfloat/e5-mistral-7b-instruct": {
        "queryPrefix": "Custom: ",
        "documentPrefix": ""
      }
    }
  }

The JSON field names mirror obsidian-brain (camelCase) so the same file
can be shared across both ecosystems via dotfiles. Internally we use
the Pythonic dataclass field names `max_tokens` / `query_prefix` /
`document_prefix`.

Bad shape / missing file / wrong $version → empty map + a warning log;
the resolver falls through to the next layer. We never crash on a
malformed override file. Validation is permissive PER FIELD: an entry
with one bad field plus one good field still keeps the good field —
drops only the bad one.

Sentinel semantics for prefix fields:
  - field absent in JSON → `query_prefix=None` (dataclass default
    `None`) meaning "not overridden, fall through to next layer".
  - field present with explicit `""` (empty string) → `query_prefix=""`
    meaning "override; clear the prefix to empty".
  - field present with `null` → `query_prefix=None` meaning "explicitly
    clear / not set" — treated identically to absent for the resolver.

Override CHANGES are picked up on the next process boot. `save_override`
and `remove_override` reset the in-process cache so a long-running
server picks them up on the next `load_overrides()` call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .user_config import get_overrides_path


_log = logging.getLogger("apple-notes-brain")

SUPPORTED_VERSION = 1


@dataclass(frozen=True)
class ModelOverride:
    """A partial patch on top of resolved metadata for a single model.

    `None` means "not overridden — fall through to the next resolver
    layer". An explicit empty string for a prefix field means "clear
    the prefix to empty" (still an override). `max_tokens=None` means
    "not overridden"; any positive int is a real override.
    """

    max_tokens: int | None = None
    query_prefix: str | None = None
    document_prefix: str | None = None

    def has_any(self) -> bool:
        """True if this override actually sets at least one field.

        Used by load/save to drop entries that contributed nothing
        after validation.
        """
        return (
            self.max_tokens is not None
            or self.query_prefix is not None
            or self.document_prefix is not None
        )


# JSON key <-> dataclass field name. obsidian-brain ships camelCase on
# disk; we keep snake_case in Python.
_FIELD_JSON_TO_PY: dict[str, str] = {
    "maxTokens": "max_tokens",
    "queryPrefix": "query_prefix",
    "documentPrefix": "document_prefix",
}
_FIELD_PY_TO_JSON: dict[str, str] = {v: k for k, v in _FIELD_JSON_TO_PY.items()}


# Process-local cache. None means "not yet loaded"; an empty dict is a
# valid loaded value (e.g. missing file or bad JSON).
_cache: dict[str, ModelOverride] | None = None


def _reset_overrides_cache() -> None:
    """Test hook + CLI write-then-read flow. Clears the in-process cache
    so the next `load_overrides()` call re-reads from disk."""
    global _cache
    _cache = None


def _validate_entry(model_id: str, raw: Any) -> ModelOverride | None:
    """Validate a single JSON entry, returning a `ModelOverride` or None.

    Permissive: an entry with one bad field plus one good field still
    keeps the good field — drops only the bad one. Returns None only
    when no valid fields remain (so callers can skip the entry entirely).
    """
    if not isinstance(raw, dict):
        _log.warning(
            "model-overrides: entry for %r is not an object — dropping", model_id
        )
        return None

    max_tokens: int | None = None
    query_prefix: str | None = None
    document_prefix: str | None = None
    has_field = False

    if "maxTokens" in raw:
        val = raw["maxTokens"]
        # bool is a subclass of int in Python; reject it explicitly so
        # `{"maxTokens": true}` doesn't silently round-trip as 1.
        if (
            isinstance(val, int)
            and not isinstance(val, bool)
            and val > 0
        ):
            max_tokens = val
            has_field = True
        else:
            _log.warning(
                "model-overrides: %s.maxTokens must be a positive int — dropping (got %r)",
                model_id,
                val,
            )

    for json_key, py_attr in (
        ("queryPrefix", "query_prefix"),
        ("documentPrefix", "document_prefix"),
    ):
        if json_key not in raw:
            continue
        val = raw[json_key]
        if val is None or isinstance(val, str):
            # Both None and explicit string (incl. "") are accepted.
            # Empty string is a meaningful override ("clear the prefix").
            # JSON `null` collapses to Python None which we treat as
            # "not overridden" — same as field absent — but we mark
            # has_field so the entry isn't dropped if the user's intent
            # was to declare it explicitly.
            if py_attr == "query_prefix":
                query_prefix = val
            else:
                document_prefix = val
            if val is not None:
                has_field = True
        else:
            _log.warning(
                "model-overrides: %s.%s must be string or null — dropping (got %r)",
                model_id,
                json_key,
                val,
            )

    if not has_field:
        return None

    return ModelOverride(
        max_tokens=max_tokens,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


def load_overrides() -> dict[str, ModelOverride]:
    """Return the model-id → ModelOverride map. Process-cached.

    Empty dict on:
      - missing file
      - unreadable file / OSError
      - malformed JSON
      - top-level value not an object
      - unsupported `$version`
      - `models` key missing or not an object

    A warning is logged for each failure mode (except missing file,
    which is the common "no overrides configured" path). Per-entry
    validation failures log a warning and skip that entry; valid
    entries from the same file are still loaded.
    """
    global _cache
    if _cache is not None:
        return _cache

    result: dict[str, ModelOverride] = {}
    path = get_overrides_path()

    if not path.exists():
        _cache = result
        return result

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning(
            "model-overrides: cannot read %s (%s) — ignoring", path, exc
        )
        _cache = result
        return result

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _log.warning(
            "model-overrides: invalid JSON at %s (%s) — ignoring", path, exc
        )
        _cache = result
        return result

    if not isinstance(parsed, dict):
        _log.warning(
            "model-overrides: top-level value at %s is not an object — ignoring",
            path,
        )
        _cache = result
        return result

    version = parsed.get("$version")
    if version != SUPPORTED_VERSION:
        _log.warning(
            "model-overrides: unsupported $version %r (expected %d) — ignoring",
            version,
            SUPPORTED_VERSION,
        )
        _cache = result
        return result

    models = parsed.get("models")
    if not isinstance(models, dict):
        # Missing / wrong-typed `models` is benign — treat as "no
        # overrides". No warning: the file is structurally valid, just
        # empty.
        _cache = result
        return result

    for model_id, raw_entry in models.items():
        if not isinstance(model_id, str) or not model_id:
            _log.warning(
                "model-overrides: skipping entry with non-string key %r", model_id
            )
            continue
        validated = _validate_entry(model_id, raw_entry)
        if validated is not None:
            result[model_id] = validated

    _cache = result
    return result


def get_override(model_id: str) -> ModelOverride | None:
    """Convenience accessor for the resolver chain (Phase δ).

    Returns the override for `model_id` if one exists, else None. Uses
    the same process-cached `load_overrides()` map.
    """
    return load_overrides().get(model_id)


def _safe_read_or_empty(path: Path) -> dict[str, Any]:
    """Read+parse the overrides file, returning a known-good shape.

    Used by save/remove to avoid clobbering valid neighbour entries on
    a partial-read failure. Falls back to a fresh empty $version=1
    document on any error.
    """
    empty: dict[str, Any] = {"$version": SUPPORTED_VERSION, "models": {}}
    if not path.exists():
        return empty
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if (
        isinstance(parsed, dict)
        and parsed.get("$version") == SUPPORTED_VERSION
        and isinstance(parsed.get("models"), dict)
    ):
        return parsed
    return empty


def _override_to_json(override: ModelOverride) -> dict[str, Any]:
    """Serialise a ModelOverride to its JSON dict shape.

    Fields with value `None` are OMITTED (None means "not overridden").
    Empty-string prefix fields ARE included (they mean "clear to empty").
    """
    out: dict[str, Any] = {}
    if override.max_tokens is not None:
        out["maxTokens"] = override.max_tokens
    if override.query_prefix is not None:
        out["queryPrefix"] = override.query_prefix
    if override.document_prefix is not None:
        out["documentPrefix"] = override.document_prefix
    return out


def save_override(model_id: str, override: ModelOverride) -> None:
    """Persist (or update) the override for `model_id`.

    Merges with any existing fields for that model id — calling with
    `ModelOverride(max_tokens=1024)` won't wipe a previously-set
    `query_prefix`. Resets the in-process cache so subsequent
    `load_overrides()` calls in the same process pick up the change.

    Creates the config dir + file on first write.
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError(f"save_override: model_id must be a non-empty string, got {model_id!r}")

    path = get_overrides_path()
    # get_overrides_path() already mkdir's the parent via get_user_config_dir().

    existing = _safe_read_or_empty(path)
    models = existing["models"]

    merged_dict: dict[str, Any] = dict(models.get(model_id, {}))
    merged_dict.update(_override_to_json(override))
    models[model_id] = merged_dict

    # Pretty-print so users can hand-edit. Trailing newline matches POSIX
    # convention (and obsidian-brain).
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    _reset_overrides_cache()


def remove_override(model_id: str) -> bool:
    """Delete the override entry for `model_id`.

    Returns True if an entry was removed, False if no-op (file missing
    or model_id not present). Resets the in-process cache on a
    successful delete so the next load reflects the change.
    """
    path = get_overrides_path()
    if not path.exists():
        return False

    existing = _safe_read_or_empty(path)
    models = existing["models"]
    if model_id not in models:
        return False

    del models[model_id]
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    _reset_overrides_cache()
    return True
