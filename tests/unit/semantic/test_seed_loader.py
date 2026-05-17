"""Tests for `apple_notes_brain.semantic.seed_loader`.

Covers:
  * Bundled seed loads, has expected schema version + non-empty models.
  * User-fetched seed at `~/.config/apple-notes-brain/seed-models.json`
    takes precedence over the bundled copy.
  * Bad shapes (malformed JSON, wrong schema version, missing models
    object) fall back gracefully without raising.
  * Missing files (no user seed AND bundled unavailable) → empty dict.
  * Individual entries with bad shape are skipped (warning logged); good
    entries on the same file are kept.
  * v1 schema entries get adapted down to the v2 trio.
  * Key normalisation strips `onnx::<hf_repo>::<file>` and
    `ollama::<base_url>::<tag>` prefixes.
  * In-process cache: load_seed() returns the same dict across calls;
    `_reset_seed_cache()` clears it.
  * Every preset model from Phase β has an entry in the bundled seed
    (sanity check — catches a missing seed entry on dep bump).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apple_notes_brain.semantic.embedder.presets import EMBEDDING_PRESETS
from apple_notes_brain.semantic.seed_loader import (
    SeedEntry,
    _normalise_key,
    _reset_seed_cache,
    get_seed_meta,
    get_user_seed_path,
    load_seed,
    lookup,
)


# ---------------------------------------------------------------------------
# Cache reset between tests so user-seed-vs-bundled tests don't bleed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_seed_cache(monkeypatch, tmp_path):
    """Reset cache + point HOME at tmp so no real user seed leaks in."""
    _reset_seed_cache()
    # Point HOME at a fresh tmp dir so get_user_seed_path() resolves to a
    # location that does NOT contain a real seed file. Tests that want a
    # user seed write it explicitly under tmp_path/.config/...
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: Path(str(tmp_path))))
    yield
    _reset_seed_cache()


# ---------------------------------------------------------------------------
# Bundled seed — happy path
# ---------------------------------------------------------------------------

def test_bundled_seed_loads_non_empty():
    seed = load_seed()
    assert isinstance(seed, dict)
    assert len(seed) > 0


def test_bundled_seed_contains_default_preset_model():
    """Sanity check: the bundled seed has the default preset's HF id."""
    seed = load_seed()
    assert "Xenova/bge-small-en-v1.5" in seed


def test_bundled_seed_entry_shape():
    """Each loaded entry is a SeedEntry with valid types."""
    seed = load_seed()
    entry = seed["Xenova/bge-small-en-v1.5"]
    assert isinstance(entry, SeedEntry)
    assert isinstance(entry.max_tokens, int)
    assert entry.max_tokens > 0
    assert entry.query_prefix is None or isinstance(entry.query_prefix, str)
    assert entry.document_prefix is None or isinstance(entry.document_prefix, str)


@pytest.mark.parametrize("preset_name,preset", list(EMBEDDING_PRESETS.items()))
def test_every_preset_has_bundled_seed_entry(preset_name, preset):
    """Sanity guard: each Phase β preset model is keyed in the bundled seed
    under the on-provider identifier the embedder will resolve to. This
    catches missed seed entries when bumping the obsidian-brain anchor."""
    seed = load_seed()
    # For onnx presets the key is the HF repo; for ollama presets it's
    # the bare Ollama tag.
    if preset.provider == "onnx":
        key = preset.onnx_repo
    else:
        key = preset.ollama_model
    assert key in seed, f"preset {preset_name!r}: seed missing key {key!r}"


def test_bge_small_query_prefix_matches_obsidian_brain():
    """Asymmetric model: BGE-family uses the standard `Represent this …` prefix."""
    seed = load_seed()
    entry = seed["Xenova/bge-small-en-v1.5"]
    assert entry.query_prefix == "Represent this sentence for searching relevant passages: "
    assert entry.document_prefix == ""


def test_multilingual_e5_uses_query_passage_prefixes():
    seed = load_seed()
    entry = seed["Xenova/multilingual-e5-small"]
    assert entry.query_prefix == "query: "
    assert entry.document_prefix == "passage: "


def test_qwen_seed_entry_has_long_context():
    """qwen3-embedding:0.6b advertises 32k context in the bundled seed."""
    seed = load_seed()
    entry = seed["qwen3-embedding:0.6b"]
    assert entry.max_tokens >= 32768


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

def test_cache_returns_same_instance_on_second_call():
    first = load_seed()
    second = load_seed()
    assert first is second


def test_reset_seed_cache_clears_cache():
    first = load_seed()
    _reset_seed_cache()
    second = load_seed()
    # Different dict instances, same content shape.
    assert first is not second
    assert set(first.keys()) == set(second.keys())


def test_get_seed_meta_returns_metadata():
    load_seed()
    meta = get_seed_meta()
    assert meta is not None
    assert meta["entries"] > 0
    assert meta["schema_version"] in {1, 2}


def test_get_seed_meta_loads_on_demand():
    """Calling get_seed_meta before load_seed still works."""
    _reset_seed_cache()
    meta = get_seed_meta()
    assert meta is not None


# ---------------------------------------------------------------------------
# User-fetched seed precedence
# ---------------------------------------------------------------------------

def _write_user_seed(tmp_path: Path, payload: dict) -> Path:
    """Write a JSON file at the user-seed path and return that path."""
    target = tmp_path / ".config" / "apple-notes-brain" / "seed-models.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_user_seed_takes_precedence_over_bundled(tmp_path):
    """User-fetched seed completely replaces the bundled seed (no merging)."""
    _write_user_seed(
        tmp_path,
        {
            "$schemaVersion": 2,
            "models": {
                "my-org/my-model": {
                    "maxTokens": 4096,
                    "queryPrefix": "Q: ",
                    "documentPrefix": "D: ",
                }
            },
        },
    )
    seed = load_seed()
    assert "my-org/my-model" in seed
    assert seed["my-org/my-model"].max_tokens == 4096
    # The bundled seed has Xenova/bge-small-en-v1.5; the user seed doesn't.
    # Since the user seed takes full precedence, it shouldn't be present.
    assert "Xenova/bge-small-en-v1.5" not in seed


def test_user_seed_path_resolves_under_home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: Path(str(tmp_path))))
    path = get_user_seed_path()
    assert path == tmp_path / ".config" / "apple-notes-brain" / "seed-models.json"


def test_user_seed_malformed_json_falls_back_to_bundled(tmp_path, capsys):
    target = tmp_path / ".config" / "apple-notes-brain" / "seed-models.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{this is not valid JSON", encoding="utf-8")
    seed = load_seed()
    # Bundled has Xenova/bge-small-en-v1.5; we should see that come back.
    assert "Xenova/bge-small-en-v1.5" in seed
    captured = capsys.readouterr()
    assert "invalid" in captured.err.lower() or "WARN" in captured.err


def test_user_seed_bad_shape_returns_empty(tmp_path, capsys):
    """A user seed with schema 99 is rejected → empty dict (no bundled fallback
    once user-seed is parseable; matches obsidian-brain semantics)."""
    _write_user_seed(tmp_path, {"$schemaVersion": 99, "models": {}})
    seed = load_seed()
    assert seed == {}
    captured = capsys.readouterr()
    assert "schema version" in captured.err.lower() or "WARN" in captured.err


def test_user_seed_missing_models_returns_empty(tmp_path, capsys):
    _write_user_seed(tmp_path, {"$schemaVersion": 2})
    seed = load_seed()
    assert seed == {}
    captured = capsys.readouterr()
    assert "models" in captured.err.lower() or "WARN" in captured.err


def test_user_seed_entries_with_bad_shape_skipped(tmp_path, capsys):
    """Good entries land in the cache, bad entries are dropped with a warning."""
    _write_user_seed(
        tmp_path,
        {
            "$schemaVersion": 2,
            "models": {
                "good/one": {"maxTokens": 512, "queryPrefix": None, "documentPrefix": None},
                "bad/no-max-tokens": {"queryPrefix": "x"},
                "bad/negative-tokens": {"maxTokens": -5, "queryPrefix": None, "documentPrefix": None},
                "bad/non-string-prefix": {"maxTokens": 512, "queryPrefix": 5, "documentPrefix": None},
                "good/two": {"maxTokens": 1024, "queryPrefix": "q", "documentPrefix": "d"},
            },
        },
    )
    seed = load_seed()
    assert set(seed.keys()) == {"good/one", "good/two"}
    captured = capsys.readouterr()
    assert "skipped" in captured.err.lower() or "WARN" in captured.err


def test_v1_schema_entries_adapted(tmp_path):
    """v1 entries carry extra fields; loader projects them down to v2 trio."""
    _write_user_seed(
        tmp_path,
        {
            "$schemaVersion": 1,
            "models": {
                "v1-model/example": {
                    "dim": 384,
                    "maxTokens": 512,
                    "queryPrefix": None,
                    "documentPrefix": None,
                    "modelType": "sentence-transformer",
                    "baseModel": None,
                    "hasDenseLayer": False,
                    "hasNormalize": True,
                    "sizeBytes": 30000000,
                    "runnableViaTransformersJs": True,
                }
            },
        },
    )
    seed = load_seed()
    entry = seed["v1-model/example"]
    assert entry.max_tokens == 512
    assert entry.query_prefix is None
    assert entry.document_prefix is None


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        # ONNX prefix scheme
        (
            "onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx",
            "Xenova/bge-small-en-v1.5",
        ),
        (
            "onnx::MongoDB/mdbr-leaf-ir::onnx/model_quantized.onnx",
            "MongoDB/mdbr-leaf-ir",
        ),
        # ONNX prefix scheme, no file fragment (defensive)
        ("onnx::Xenova/bge-small-en-v1.5", "Xenova/bge-small-en-v1.5"),
        # Ollama prefix scheme — model tag follows the base URL
        (
            "ollama::http://localhost:11434::qwen3-embedding:0.6b",
            "qwen3-embedding:0.6b",
        ),
        (
            "ollama::http://192.168.1.10:11434::nomic-embed-text",
            "nomic-embed-text",
        ),
        # Obsidian-brain back-compat single-colon prefix
        ("ollama:qwen3-embedding:0.6b", "qwen3-embedding:0.6b"),
        # Already-bare bare HF id passes through
        ("Xenova/bge-small-en-v1.5", "Xenova/bge-small-en-v1.5"),
        # Already-bare Ollama tag
        ("qwen3-embedding:0.6b", "qwen3-embedding:0.6b"),
    ],
)
def test_normalise_key(raw, expected):
    assert _normalise_key(raw) == expected


# ---------------------------------------------------------------------------
# lookup() — the public API
# ---------------------------------------------------------------------------

def test_lookup_onnx_full_identifier():
    """OnnxEmbedder.model_identifier() resolves to the bundled seed entry."""
    entry = lookup("onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx")
    assert entry is not None
    assert entry.max_tokens > 0
    assert entry.query_prefix is not None  # BGE asymmetric


def test_lookup_ollama_full_identifier():
    """OllamaEmbedder.model_identifier() resolves to the bundled seed entry."""
    entry = lookup("ollama::http://localhost:11434::qwen3-embedding:0.6b")
    assert entry is not None
    assert entry.max_tokens >= 32768


def test_lookup_bare_hf_id():
    """A bare HF repo id (e.g. as cfg.model holds) looks up cleanly."""
    entry = lookup("Xenova/multilingual-e5-small")
    assert entry is not None
    assert entry.query_prefix == "query: "


def test_lookup_unknown_model_returns_none():
    assert lookup("onnx::no/such-model::onnx/model.onnx") is None


def test_lookup_unknown_bare_returns_none():
    assert lookup("definitely-not-real:model") is None


def test_lookup_does_not_crash_on_empty_identifier():
    """Defensive — pathological input shouldn't blow up the resolver chain."""
    assert lookup("") is None
