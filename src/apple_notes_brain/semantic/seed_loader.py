"""Bundled-seed JSON loader for embedder metadata.

Reads `apple_notes_brain/data/seed-models.json` (committed anchor, refreshed
from obsidian-brain's matching file) and exposes typed `SeedEntry` lookups
keyed on the HuggingFace repo id. Pure read; no DB; no fetch.

Mirrors obsidian-brain's `src/embeddings/seed-loader.ts` exactly so the
two -brain servers stay metadata-compatible. The schema is documented
there in full; the short version:

  Schema v2 (current): three load-bearing fields per entry —
  `maxTokens`, `queryPrefix`, `documentPrefix`. Everything else (dim,
  sizeBytes, prefixSource, modelType, baseModel, hasDenseLayer,
  hasNormalize) was dropped because runtime probes `dim` from the loaded
  ONNX and the rest is informational.

  Schema v1 (older anchors / pre-Python build script): superset shape
  with all the cosmetic fields. Loaded transparently — the v1→v2 adapter
  pulls the three fields we still care about and discards the rest.

Two-tier load:
  Priority 1: user-fetched seed at `~/.config/apple-notes-brain/seed-models.json`
              (written by a future `apple-notes-brain models fetch-seed` CLI).
              Takes precedence when present and parseable — lets users pull
              in upstream MTEB fixes without a release.
  Priority 2: bundled package copy at `apple_notes_brain.data.seed-models.json`.

Bad shape / missing file → returns an empty dict + writes a single stderr
warning. Resolver chain falls through to the live HF fetcher (Phase δ);
we never crash.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from importlib.resources import files as _resource_files
from pathlib import Path

_log = logging.getLogger("apple-notes-brain")


SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})


@dataclass(frozen=True)
class SeedEntry:
    """Per-model metadata triple — the v2 schema's load-bearing fields.

    `max_tokens` is the effective input-length cap. `query_prefix` and
    `document_prefix` are `None` for symmetric models (BGE-family, etc.)
    and non-None strings for asymmetric models (e5-family,
    qwen3-embedding, etc.). For asymmetric models that prepend only to
    queries, `document_prefix == ''`.
    """

    max_tokens: int
    query_prefix: str | None
    document_prefix: str | None


# ─── In-process cache (module-level) ─────────────────────────────────────
# Single dict shared across the process. `_reset_seed_cache()` is the test
# hook; production code never invalidates the cache.
_SEED_CACHE: dict[str, SeedEntry] | None = None
_SEED_META: dict[str, object] | None = None


def _reset_seed_cache() -> None:
    """Test hook — clears the process-wide seed cache."""
    global _SEED_CACHE, _SEED_META
    _SEED_CACHE = None
    _SEED_META = None


def _warn_stderr(msg: str) -> None:
    """Sync stderr write — race-safe on crash."""
    sys.stderr.write(f"apple-notes-brain WARN: {msg}\n")
    sys.stderr.flush()
    _log.warning(msg)


def get_user_seed_path() -> Path:
    """`~/.config/apple-notes-brain/seed-models.json`.

    Resolution intentionally simple — no XDG_CONFIG_HOME yet (that comes
    in Phase ε's user-config module, owned by a different agent). When
    that lands, this function may be replaced by a call to
    `user_config.get_user_seed_path()`.
    """
    return Path.home() / ".config" / "apple-notes-brain" / "seed-models.json"


# ─── Parsing helpers ─────────────────────────────────────────────────────

def _is_valid_entry_shape(raw: object) -> bool:
    """v2-or-v1 entry validity: `maxTokens` positive int; prefixes str|None."""
    if not isinstance(raw, dict):
        return False
    mt = raw.get("maxTokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0:
        return False
    qp = raw.get("queryPrefix")
    if qp is not None and not isinstance(qp, str):
        return False
    dp = raw.get("documentPrefix")
    if dp is not None and not isinstance(dp, str):
        return False
    return True


def _adapt_entry(raw: dict[str, object]) -> SeedEntry:
    """Project a raw JSON entry (v1 or v2) down to the SeedEntry trio.

    Drops every other field — v1 carries `dim`, `sizeBytes`, `prefixSource`,
    `modelType`, `baseModel`, `hasDenseLayer`, `hasNormalize`,
    `runnableViaTransformersJs`; all dropped.
    """
    return SeedEntry(
        max_tokens=int(raw["maxTokens"]),  # type: ignore[arg-type]
        query_prefix=raw.get("queryPrefix"),  # type: ignore[arg-type]
        document_prefix=raw.get("documentPrefix"),  # type: ignore[arg-type]
    )


def _parse_seed_payload(parsed: object) -> dict[str, SeedEntry]:
    """Validate and adapt a parsed seed JSON to a model_id → SeedEntry dict.

    Bad shape at any level returns an empty dict + warning. Individual bad
    entries are skipped (warning aggregated).
    """
    if not isinstance(parsed, dict):
        _warn_stderr("seed-loader: seed JSON has invalid shape — ignoring")
        return {}

    version = parsed.get("$schemaVersion")
    if not isinstance(version, int) or version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))
        _warn_stderr(
            f"seed-loader: seed JSON schema version {version!r} is not supported "
            f"(expected one of {supported}) — ignoring"
        )
        return {}

    models = parsed.get("models")
    if not isinstance(models, dict):
        _warn_stderr("seed-loader: seed JSON has no `models` object — ignoring")
        return {}

    out: dict[str, SeedEntry] = {}
    dropped = 0
    for model_id, raw in models.items():
        if not isinstance(model_id, str):
            dropped += 1
            continue
        if _is_valid_entry_shape(raw):
            out[model_id] = _adapt_entry(raw)  # type: ignore[arg-type]
        else:
            dropped += 1
    if dropped > 0:
        _warn_stderr(f"seed-loader: {dropped} seed entries skipped due to invalid shape")

    # Stash diagnostic metadata for callers that want it.
    global _SEED_META
    _SEED_META = {
        "generated_at": parsed.get("$generatedAt"),
        "source": parsed.get("$source") or parsed.get("$mtebRevision"),
        "entries": len(out),
        "schema_version": version,
    }
    return out


def _load_user_seed() -> dict[str, SeedEntry] | None:
    """Try to load the user-fetched seed. Returns None when absent or bad."""
    path = get_user_seed_path()
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        _warn_stderr(
            f"seed-loader: user-fetched seed at {path} is invalid "
            f"({exc.__class__.__name__}: {exc}) — falling back to bundled"
        )
        return None
    return _parse_seed_payload(parsed)


def _load_bundled_seed() -> dict[str, SeedEntry]:
    """Load the bundled package seed. Returns empty dict on any failure."""
    try:
        resource = _resource_files("apple_notes_brain.data").joinpath("seed-models.json")
        text = resource.read_text(encoding="utf-8")
        parsed = json.loads(text)
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError, OSError) as exc:
        _warn_stderr(
            f"seed-loader: bundled seed unavailable ({exc.__class__.__name__}: {exc}) — "
            "proceeding without seed (HF live fetch will populate cache)"
        )
        return {}
    return _parse_seed_payload(parsed)


def load_seed() -> dict[str, SeedEntry]:
    """Load the seed JSON once per process. Subsequent calls hit the cache.

    Empty dict on any load failure; never raises.
    """
    global _SEED_CACHE
    if _SEED_CACHE is not None:
        return _SEED_CACHE

    user = _load_user_seed()
    if user is not None:
        _SEED_CACHE = user
        return _SEED_CACHE

    _SEED_CACHE = _load_bundled_seed()
    return _SEED_CACHE


def get_seed_meta() -> dict[str, object] | None:
    """Diagnostic — exposed for `index_status` / future CLI to surface freshness."""
    if _SEED_CACHE is None:
        load_seed()
    return _SEED_META


# ─── Key normalisation ───────────────────────────────────────────────────
# Embedder.model_identifier() returns provider-prefixed identifiers:
#   - OnnxEmbedder:   "onnx::<hf_repo>::<file>"   e.g. "onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx"
#   - OllamaEmbedder: "ollama::<base_url>::<tag>" e.g. "ollama::http://localhost:11434::qwen3-embedding:0.6b"
#
# The seed JSON keys on the bare HF repo for ONNX models (e.g.
# "Xenova/bge-small-en-v1.5") and the bare Ollama tag for Ollama models
# (e.g. "qwen3-embedding:0.6b"). We normalise the embedder identifier to
# that bare form before lookup.

def _normalise_key(model_identifier: str) -> str:
    """Strip provider prefix + middle fragment from `model_identifier`.

    Examples:
      "onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx"
        → "Xenova/bge-small-en-v1.5"
      "ollama::http://localhost:11434::qwen3-embedding:0.6b"
        → "qwen3-embedding:0.6b"
      "Xenova/bge-small-en-v1.5"   (already-bare)
        → "Xenova/bge-small-en-v1.5"
      "ollama:qwen3-embedding:0.6b"  (obsidian-brain-style single-colon)
        → "qwen3-embedding:0.6b"
    """
    if model_identifier.startswith("onnx::"):
        rest = model_identifier[len("onnx::"):]
        # split on next "::" to drop the file-path fragment
        idx = rest.find("::")
        return rest[:idx] if idx != -1 else rest
    if model_identifier.startswith("ollama::"):
        rest = model_identifier[len("ollama::"):]
        # "<base_url>::<tag>" — model tag is the part after the last "::"
        idx = rest.find("::")
        if idx != -1:
            return rest[idx + 2 :]
        # Fallback: no second "::" → treat the whole tail as the tag
        return rest
    # Obsidian-brain back-compat: `ollama:` single-colon prefix.
    if model_identifier.startswith("ollama:"):
        return model_identifier[len("ollama:"):]
    return model_identifier


def lookup(model_identifier: str) -> SeedEntry | None:
    """Look up a model's seed entry by its embedder.model_identifier().

    Returns `None` when the model isn't in the seed (the resolver chain
    then falls through to HF live fetch / probe / fallback).
    """
    seed = load_seed()
    key = _normalise_key(model_identifier)
    if key in seed:
        return seed[key]
    # Last-ditch: a few seed entries may key on the bare model name
    # without the org prefix. Try stripping any leading "<org>/" and
    # re-looking-up. This matches obsidian-brain's chain (metadata-resolver
    # falls back to base-model lookups when the prefixed key misses).
    if "/" in key:
        bare = key.split("/", 1)[1]
        if bare in seed:
            return seed[bare]
    return None
