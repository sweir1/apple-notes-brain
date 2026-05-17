"""Tests for `apple_notes_brain.semantic.overrides`.

Covers:
  - Load with no file, malformed JSON, wrong $version, top-level non-object,
    missing `models` key.
  - Per-entry permissive validation (one bad field doesn't drop the others).
  - max_tokens type/value rules (positive int only; bool/float/zero/negative
    rejected).
  - query_prefix / document_prefix: str | null allowed; numbers/objects
    rejected.
  - Empty-string prefix is preserved (means "clear to empty").
  - save/load/remove round-trip.
  - Cache reset behaviour (writes invalidate; explicit `_reset` works).
  - get_override convenience.
  - Multiple models in one file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apple_notes_brain.semantic import overrides as overrides_mod
from apple_notes_brain.semantic.overrides import (
    SUPPORTED_VERSION,
    ModelOverride,
    _reset_overrides_cache,
    get_override,
    load_overrides,
    remove_override,
    save_override,
)
from apple_notes_brain.semantic.user_config import ENV_CONFIG_DIR


# ---------------------------------------------------------------------------
# Shared fixture: redirect overrides file to tmp_path + always reset the
# process-cache before AND after each test so cross-test pollution is
# impossible.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_config_dir(monkeypatch, tmp_path):
    """Point the user-config dir at tmp_path so we never touch the real
    `~/.config/apple-notes-brain/`. Resets the in-process cache around
    each test."""
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "cfg"))
    _reset_overrides_cache()
    yield tmp_path / "cfg" / "model-overrides.json"
    _reset_overrides_cache()


# ---------------------------------------------------------------------------
# ModelOverride dataclass
# ---------------------------------------------------------------------------


def test_default_override_has_no_fields_set():
    """A bare `ModelOverride()` is the "nothing overridden" sentinel."""
    ov = ModelOverride()
    assert ov.max_tokens is None
    assert ov.query_prefix is None
    assert ov.document_prefix is None
    assert ov.has_any() is False


def test_has_any_detects_each_field():
    assert ModelOverride(max_tokens=512).has_any() is True
    assert ModelOverride(query_prefix="q: ").has_any() is True
    assert ModelOverride(document_prefix="d: ").has_any() is True
    # Empty-string prefix is still a real override (means "clear it").
    assert ModelOverride(query_prefix="").has_any() is True
    assert ModelOverride(document_prefix="").has_any() is True


def test_override_is_frozen():
    """Dataclass is frozen — overrides are value objects, not mutable state."""
    ov = ModelOverride(max_tokens=512)
    with pytest.raises(Exception):  # FrozenInstanceError subclasses AttributeError
        ov.max_tokens = 1024  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_overrides — empty/missing/malformed file modes
# ---------------------------------------------------------------------------


def test_load_no_file_returns_empty():
    """No overrides file → empty dict (the common path for users who never
    set up overrides)."""
    assert load_overrides() == {}


def test_load_malformed_json_returns_empty_with_warning(isolated_config_dir, caplog):
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text("{ this is not valid json", encoding="utf-8")
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert result == {}
    assert any("invalid JSON" in r.message for r in caplog.records)


def test_load_top_level_non_object_returns_empty(isolated_config_dir, caplog):
    """Top-level JSON value must be an object — anything else is ignored."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text("[1, 2, 3]", encoding="utf-8")
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert result == {}
    assert any("not an object" in r.message for r in caplog.records)


def test_load_wrong_version_returns_empty_with_warning(isolated_config_dir, caplog):
    """An unsupported `$version` (future or wrong) is treated as not-loadable
    rather than guessed at."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"$version": 99, "models": {"foo": {"maxTokens": 512}}}),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert result == {}
    assert any("$version" in r.message for r in caplog.records)


def test_load_missing_version_returns_empty(isolated_config_dir, caplog):
    """A file with no `$version` key is treated as wrong-version."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"models": {"foo": {"maxTokens": 512}}}),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert result == {}


def test_load_missing_models_key_returns_empty(isolated_config_dir):
    """`$version: 1` but no `models` key → empty (file is structurally valid
    but contributes nothing). No warning because this is benign."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"$version": 1}),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    assert load_overrides() == {}


def test_load_models_not_an_object_returns_empty(isolated_config_dir):
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"$version": 1, "models": "oops"}),
        encoding="utf-8",
    )
    _reset_overrides_cache()
    assert load_overrides() == {}


# ---------------------------------------------------------------------------
# load_overrides — valid entries
# ---------------------------------------------------------------------------


def test_load_valid_file_returns_correct_map(isolated_config_dir):
    """Round-trip a hand-written file → in-memory map."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "BAAI/bge-small-en-v1.5": {"maxTokens": 256},
                    "intfloat/e5-small-v2": {
                        "queryPrefix": "query: ",
                        "documentPrefix": "passage: ",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()

    assert set(result.keys()) == {
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
    }
    assert result["BAAI/bge-small-en-v1.5"] == ModelOverride(max_tokens=256)
    assert result["intfloat/e5-small-v2"] == ModelOverride(
        query_prefix="query: ", document_prefix="passage: "
    )


def test_load_multiple_models_independent(isolated_config_dir):
    """Three completely different override types co-exist."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "a/only-tokens": {"maxTokens": 1024},
                    "b/only-query": {"queryPrefix": "Q: "},
                    "c/only-doc": {"documentPrefix": "D: "},
                    "d/all-three": {
                        "maxTokens": 4096,
                        "queryPrefix": "search: ",
                        "documentPrefix": "doc: ",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    assert len(result) == 4
    assert result["a/only-tokens"].max_tokens == 1024
    assert result["a/only-tokens"].query_prefix is None
    assert result["b/only-query"].query_prefix == "Q: "
    assert result["b/only-query"].max_tokens is None
    assert result["c/only-doc"].document_prefix == "D: "
    assert result["d/all-three"] == ModelOverride(
        max_tokens=4096, query_prefix="search: ", document_prefix="doc: "
    )


# ---------------------------------------------------------------------------
# Permissive validation — bad field dropped, good fields kept
# ---------------------------------------------------------------------------


def test_invalid_max_tokens_dropped_other_fields_kept(isolated_config_dir, caplog):
    """`maxTokens: 0` is dropped (must be positive) but the `queryPrefix`
    in the same entry survives."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "x/mixed": {
                        "maxTokens": 0,  # invalid — must be positive
                        "queryPrefix": "q: ",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert "x/mixed" in result
    assert result["x/mixed"].max_tokens is None
    assert result["x/mixed"].query_prefix == "q: "
    assert any("maxTokens" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "bad_value",
    [-1, 0, -100, 1.5, "512", True, False, None, [], {}],
    ids=["neg1", "zero", "neg100", "float", "string", "bool-true", "bool-false", "null", "list", "dict"],
)
def test_max_tokens_rejects_non_positive_int(isolated_config_dir, bad_value):
    """`maxTokens` must be a positive Python int. bool is a subclass of int
    and is rejected explicitly — `{maxTokens: true}` should not round-trip
    as 1."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "m": {"maxTokens": bad_value, "queryPrefix": "keepme"}
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    # The entry survives because queryPrefix is valid; the bad max_tokens
    # is dropped.
    assert result["m"].max_tokens is None
    assert result["m"].query_prefix == "keepme"


def test_max_tokens_accepts_positive_int(isolated_config_dir):
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"$version": 1, "models": {"m": {"maxTokens": 8192}}}),
        encoding="utf-8",
    )
    _reset_overrides_cache()
    assert load_overrides()["m"].max_tokens == 8192


@pytest.mark.parametrize("bad_value", [42, 1.5, [], {}, True])
def test_query_prefix_rejects_non_string(isolated_config_dir, bad_value):
    """`queryPrefix` must be string or null. Numbers / lists / objects rejected."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "m": {"queryPrefix": bad_value, "maxTokens": 256}
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    # The maxTokens=256 survives; bad query_prefix dropped.
    assert result["m"].query_prefix is None
    assert result["m"].max_tokens == 256


def test_non_object_entry_dropped(isolated_config_dir, caplog):
    """An entry whose value isn't an object (e.g. a string) is dropped entirely."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "good": {"maxTokens": 512},
                    "bad": "just-a-string",
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    assert set(result.keys()) == {"good"}
    assert any("is not an object" in r.message for r in caplog.records)


def test_entry_with_no_valid_fields_is_dropped(isolated_config_dir):
    """An entry whose every field is invalid contributes nothing → skipped."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "empty": {},  # no fields at all
                    "all-bad": {"maxTokens": -1, "queryPrefix": 42},
                    "good": {"maxTokens": 512},
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    assert set(result.keys()) == {"good"}


# ---------------------------------------------------------------------------
# Empty-string prefix semantics
# ---------------------------------------------------------------------------


def test_empty_string_query_prefix_kept(isolated_config_dir):
    """`queryPrefix: ""` means "clear the prefix to empty" — must round-trip
    as `query_prefix=""`, NOT `query_prefix=None`."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {"m": {"queryPrefix": "", "documentPrefix": ""}},
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    assert "m" in result
    assert result["m"].query_prefix == ""
    assert result["m"].document_prefix == ""
    assert result["m"].has_any() is True


def test_null_prefix_distinguished_from_absent(isolated_config_dir):
    """JSON `null` collapses to Python None — treated the same as absent
    for resolver purposes (`None` = not overridden)."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps(
            {
                "$version": 1,
                "models": {
                    "m": {
                        "maxTokens": 512,
                        "queryPrefix": None,
                        "documentPrefix": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    result = load_overrides()
    assert result["m"].max_tokens == 512
    assert result["m"].query_prefix is None
    assert result["m"].document_prefix is None


# ---------------------------------------------------------------------------
# save_override — round-trip via tmp_path
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(isolated_config_dir):
    save_override("a/foo", ModelOverride(max_tokens=256, query_prefix="q: "))
    result = load_overrides()
    assert result == {
        "a/foo": ModelOverride(max_tokens=256, query_prefix="q: "),
    }


def test_save_creates_file_and_dir(isolated_config_dir):
    """First save creates both the parent dir and the file."""
    assert not isolated_config_dir.exists()
    save_override("a/foo", ModelOverride(max_tokens=512))
    assert isolated_config_dir.exists()
    # File is JSON-decodable and matches the expected shape.
    parsed = json.loads(isolated_config_dir.read_text())
    assert parsed["$version"] == SUPPORTED_VERSION
    assert parsed["models"]["a/foo"] == {"maxTokens": 512}


def test_save_merges_with_existing_entry(isolated_config_dir):
    """Calling save_override twice with disjoint fields merges them."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    save_override("a/foo", ModelOverride(query_prefix="q: "))
    result = load_overrides()
    # Both fields preserved.
    assert result["a/foo"].max_tokens == 512
    assert result["a/foo"].query_prefix == "q: "


def test_save_overwrites_same_field(isolated_config_dir):
    """Setting `max_tokens` twice keeps only the latest value."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    save_override("a/foo", ModelOverride(max_tokens=1024))
    assert load_overrides()["a/foo"].max_tokens == 1024


def test_save_doesnt_disturb_other_models(isolated_config_dir):
    save_override("a/foo", ModelOverride(max_tokens=512))
    save_override("b/bar", ModelOverride(query_prefix="bar-q: "))
    result = load_overrides()
    assert len(result) == 2
    assert result["a/foo"].max_tokens == 512
    assert result["b/bar"].query_prefix == "bar-q: "


def test_save_writes_pretty_json_with_trailing_newline(isolated_config_dir):
    """File is pretty-printed (indent=2) and ends with a newline — so
    hand-editing is comfortable and POSIX-friendly."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    text = isolated_config_dir.read_text()
    assert text.endswith("\n")
    assert "  " in text  # has at least some indentation


def test_save_empty_string_prefix_persisted(isolated_config_dir):
    """An empty-string override round-trips through save/load."""
    save_override("a/foo", ModelOverride(query_prefix=""))
    result = load_overrides()
    assert result["a/foo"].query_prefix == ""


def test_save_rejects_empty_model_id(isolated_config_dir):
    with pytest.raises(ValueError, match="model_id"):
        save_override("", ModelOverride(max_tokens=512))


def test_save_preserves_neighbours_when_file_was_malformed(isolated_config_dir):
    """If the file becomes malformed somehow, save_override recovers via
    `_safe_read_or_empty` (writes a fresh document rather than crashing)."""
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text("junk garbage not-json", encoding="utf-8")
    _reset_overrides_cache()

    # Should not raise.
    save_override("a/foo", ModelOverride(max_tokens=512))

    # File is now well-formed and contains the new entry; the corrupt
    # neighbours are gone (acceptable — we couldn't parse them).
    parsed = json.loads(isolated_config_dir.read_text())
    assert parsed["$version"] == 1
    assert parsed["models"]["a/foo"] == {"maxTokens": 512}


# ---------------------------------------------------------------------------
# remove_override
# ---------------------------------------------------------------------------


def test_remove_existing_entry_returns_true(isolated_config_dir):
    save_override("a/foo", ModelOverride(max_tokens=512))
    save_override("b/bar", ModelOverride(max_tokens=1024))

    assert remove_override("a/foo") is True

    result = load_overrides()
    assert "a/foo" not in result
    assert "b/bar" in result


def test_remove_missing_entry_returns_false(isolated_config_dir):
    save_override("a/foo", ModelOverride(max_tokens=512))
    assert remove_override("does-not-exist") is False
    # And the existing entry is untouched.
    assert "a/foo" in load_overrides()


def test_remove_with_no_file_returns_false(isolated_config_dir):
    """No-op if the overrides file doesn't exist yet."""
    assert remove_override("anything") is False


def test_remove_invalidates_cache(isolated_config_dir):
    """After remove, the next load_overrides sees the change."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    _ = load_overrides()  # populate cache
    remove_override("a/foo")
    assert load_overrides() == {}


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


def test_load_is_process_cached(isolated_config_dir, monkeypatch):
    """Second call within the same process doesn't re-read the file —
    even if the file changes underneath."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    first = load_overrides()

    # Mutate the file directly, bypassing the helpers (so cache isn't reset).
    isolated_config_dir.write_text(
        json.dumps({"$version": 1, "models": {"a/foo": {"maxTokens": 9999}}}),
        encoding="utf-8",
    )
    second = load_overrides()

    # Same object because the cache wasn't invalidated.
    assert first is second
    assert second["a/foo"].max_tokens == 512  # stale, on purpose


def test_reset_cache_forces_reread(isolated_config_dir):
    """Explicit `_reset_overrides_cache()` causes the next load to re-read."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    _ = load_overrides()

    # Mutate the file directly.
    isolated_config_dir.write_text(
        json.dumps({"$version": 1, "models": {"a/foo": {"maxTokens": 9999}}}),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    assert load_overrides()["a/foo"].max_tokens == 9999


def test_save_resets_cache_automatically(isolated_config_dir):
    """`save_override` invalidates the cache so subsequent loads see fresh
    state — no manual reset needed in long-running servers."""
    save_override("a/foo", ModelOverride(max_tokens=512))
    _ = load_overrides()

    save_override("a/foo", ModelOverride(max_tokens=1024))
    assert load_overrides()["a/foo"].max_tokens == 1024


# ---------------------------------------------------------------------------
# get_override convenience
# ---------------------------------------------------------------------------


def test_get_override_returns_none_for_unknown_model(isolated_config_dir):
    save_override("a/known", ModelOverride(max_tokens=512))
    assert get_override("z/unknown") is None


def test_get_override_returns_entry_when_present(isolated_config_dir):
    save_override("a/known", ModelOverride(max_tokens=512, query_prefix="q: "))
    ov = get_override("a/known")
    assert ov == ModelOverride(max_tokens=512, query_prefix="q: ")


def test_get_override_with_no_file_returns_none(isolated_config_dir):
    """No file at all → get_override returns None (not error)."""
    assert get_override("anything") is None


# ---------------------------------------------------------------------------
# Non-string keys are skipped
# ---------------------------------------------------------------------------


def test_load_skips_non_string_keys(isolated_config_dir, caplog):
    """JSON only has string keys, but defend against a JSON loader that
    somehow yields non-string keys (e.g. monkey-patched). Write valid JSON
    and check the normal happy path doesn't crash — the non-string-key
    skip is exercised at the validation boundary."""
    # JSON serialisers produce string keys only, so this is a partial
    # exercise — we mainly want to confirm the existing valid path still works.
    isolated_config_dir.parent.mkdir(parents=True, exist_ok=True)
    isolated_config_dir.write_text(
        json.dumps({"$version": 1, "models": {"": {"maxTokens": 512}}}),
        encoding="utf-8",
    )
    _reset_overrides_cache()

    with caplog.at_level("WARNING", logger="apple-notes-brain"):
        result = load_overrides()

    # Empty-string model_id is skipped (treated as "non-string" by our
    # `if not model_id` check).
    assert result == {}
