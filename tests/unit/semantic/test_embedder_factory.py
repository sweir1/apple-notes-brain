"""Tests for the embedder factory + preset registry + atomic env resolver.

Covers:
  * The six named presets (mirror obsidian-brain) with expected
    (provider, model, dim).
  * Back-compat `resolve_preset(model, provider)` for the embedder
    constructors that pass `config.model` in.
  * Deprecation aliases (`bge-small-en-v1.5`, `bge-base-en-v1.5`,
    `all-MiniLM-L6-v2`, `fastest`, `balanced`) resolve to their canonical
    preset with a one-shot stderr warning.
  * The factory dispatches on `cfg.provider`.
  * `resolve_preset_config()` precedence ladder (every branch):
      1. `EMBEDDING_MODEL` set → raw model, provider = `EMBEDDING_PROVIDER`
         or default `onnx`.
      2. `EMBEDDING_PROVIDER` + `EMBEDDING_PRESET` mismatch → provider
         wins, preset's model carried, mismatch warning emitted.
      3. `EMBEDDING_PRESET` alone → preset's declared (provider, model).
      4. `EMBEDDING_PROVIDER` alone → provider-default model.
      5. Nothing set → DEFAULT_PRESET.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.embedder import create_embedder
from apple_notes_brain.semantic.embedder.presets import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PRESET,
    DEPRECATED_PRESET_ALIASES,
    EMBEDDING_PRESETS,
    ResolvedPresetConfig,
    _reset_warning_state,
    resolve_preset,
    resolve_preset_config,
)


# ---------------------------------------------------------------------------
# Autouse: reset the module-level one-shot warning trackers between tests
# so test order doesn't suppress warnings we want to assert on.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_preset_warnings():
    _reset_warning_state()
    yield
    _reset_warning_state()


# ---------------------------------------------------------------------------
# Preset registry — the six obsidian-brain presets
# ---------------------------------------------------------------------------

def test_default_preset_is_english():
    assert DEFAULT_PRESET.short_name == "english"
    assert DEFAULT_PRESET.dim == 384
    assert DEFAULT_PRESET.provider == "onnx"
    assert DEFAULT_PRESET.model == "Xenova/bge-small-en-v1.5"


def test_default_ollama_model_matches_obsidian_brain():
    assert DEFAULT_OLLAMA_MODEL == "nomic-embed-text"


def test_registry_has_exactly_six_presets():
    """If this breaks, mirror obsidian-brain/src/embeddings/presets.ts."""
    assert set(EMBEDDING_PRESETS.keys()) == {
        "english",
        "english-fast",
        "english-quality",
        "multilingual",
        "multilingual-quality",
        "multilingual-ollama",
    }


# Per-preset shape parity with obsidian-brain. If any of these (provider,
# model, dim) triples drifts from obsidian-brain, fix here first — the
# bundled seed (Phase γ) keys on these model ids.
_EXPECTED_PRESETS = {
    "english": ("onnx", "Xenova/bge-small-en-v1.5", 384),
    "english-fast": ("onnx", "MongoDB/mdbr-leaf-ir", 384),
    "english-quality": ("onnx", "Xenova/bge-base-en-v1.5", 768),
    "multilingual": ("onnx", "Xenova/multilingual-e5-small", 384),
    "multilingual-quality": ("onnx", "Xenova/multilingual-e5-base", 768),
    "multilingual-ollama": ("ollama", "qwen3-embedding:0.6b", 1024),
}


@pytest.mark.parametrize("name,expected", list(_EXPECTED_PRESETS.items()))
def test_preset_provider_model_dim_matches_obsidian_brain(name, expected):
    preset = EMBEDDING_PRESETS[name]
    assert (preset.provider, preset.model, preset.dim) == expected


@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_preset_has_non_empty_description(name):
    """Every preset surfaces a one-line description (powers `models list` UI)."""
    p = EMBEDDING_PRESETS[name]
    assert isinstance(p.description, str)
    assert len(p.description.strip()) > 0


@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_preset_short_name_matches_registry_key(name):
    """Defends against typos in the registry."""
    assert EMBEDDING_PRESETS[name].short_name == name


# ---------------------------------------------------------------------------
# resolve_preset() — the back-compat (model, provider) lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_resolve_preset_by_short_name_onnx(name):
    preset, identifier = resolve_preset(name, "onnx")
    assert preset is not None
    assert preset.short_name == name
    assert identifier == preset.onnx_repo


@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_resolve_preset_by_short_name_ollama(name):
    preset, identifier = resolve_preset(name, "ollama")
    assert preset is not None
    assert preset.short_name == name
    assert identifier == preset.ollama_model


def test_resolve_preset_reverse_lookup_by_hf_repo():
    """A resolved HF repo id (what cfg.model holds after env resolution)
    should map back to the preset that declares it. Otherwise OnnxEmbedder
    can't recover the onnx_file path for the preset."""
    preset, identifier = resolve_preset("Xenova/bge-small-en-v1.5", "onnx")
    assert preset is not None
    assert preset.short_name == "english"
    assert identifier == "Xenova/bge-small-en-v1.5"


def test_resolve_preset_reverse_lookup_by_ollama_tag():
    preset, identifier = resolve_preset("qwen3-embedding:0.6b", "ollama")
    assert preset is not None
    assert preset.short_name == "multilingual-ollama"
    assert identifier == "qwen3-embedding:0.6b"


def test_resolve_preset_unknown_returns_literal():
    preset, identifier = resolve_preset("my/custom-model", "onnx")
    assert preset is None
    assert identifier == "my/custom-model"


@pytest.mark.parametrize("alias,canonical", list(DEPRECATED_PRESET_ALIASES.items()))
def test_resolve_preset_deprecation_alias(capsys, alias, canonical):
    """Each alias resolves to the canonical preset and emits a stderr warning."""
    preset, identifier = resolve_preset(alias, "onnx")
    assert preset is not None
    assert preset.short_name == canonical
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower() or "WARN" in captured.err


def test_resolve_preset_alias_warning_only_emitted_once(capsys):
    """One-shot warning: the second call for the same alias is silent."""
    resolve_preset("bge-small-en-v1.5", "onnx")
    first = capsys.readouterr().err
    resolve_preset("bge-small-en-v1.5", "onnx")
    second = capsys.readouterr().err
    assert "deprecated" in first.lower()
    assert second == "" or "deprecated" not in second.lower()


# ---------------------------------------------------------------------------
# resolve_preset_config() — the atomic env-var resolver
# ---------------------------------------------------------------------------

def test_resolver_no_env_returns_default_preset():
    """(5) Nothing set → DEFAULT_PRESET."""
    cfg = resolve_preset_config({})
    assert cfg == ResolvedPresetConfig(
        provider="onnx",
        model="Xenova/bge-small-en-v1.5",
        preset_short_name="english",
        source="default",
    )


def test_resolver_embedding_model_alone_uses_onnx_default():
    """(1) EMBEDDING_MODEL set → raw model, provider defaults to onnx."""
    cfg = resolve_preset_config({"EMBEDDING_MODEL": "my/custom-model"})
    assert cfg.provider == "onnx"
    assert cfg.model == "my/custom-model"
    assert cfg.preset_short_name is None
    assert cfg.source == "env-model"


def test_resolver_embedding_model_with_explicit_provider():
    """(1) EMBEDDING_MODEL set + EMBEDDING_PROVIDER → both honoured."""
    cfg = resolve_preset_config(
        {"EMBEDDING_MODEL": "qwen3-embedding:0.6b", "EMBEDDING_PROVIDER": "ollama"}
    )
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen3-embedding:0.6b"
    assert cfg.preset_short_name is None
    assert cfg.source == "env-model"


def test_resolver_embedding_model_blank_treated_as_unset():
    """Whitespace-only EMBEDDING_MODEL doesn't trigger the power-user path."""
    cfg = resolve_preset_config({"EMBEDDING_MODEL": "   "})
    assert cfg.source == "default"
    assert cfg.preset_short_name == "english"


def test_resolver_embedding_model_legacy_alias_expanded(capsys):
    """EMBEDDING_MODEL=bge-small-en-v1.5 expands to Xenova/bge-small-en-v1.5
    with a deprecation warning so existing user configs keep working."""
    cfg = resolve_preset_config({"EMBEDDING_MODEL": "bge-small-en-v1.5"})
    assert cfg.model == "Xenova/bge-small-en-v1.5"
    assert cfg.preset_short_name == "english"
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()


def test_resolver_embedding_preset_alone_english():
    """(3) EMBEDDING_PRESET set → preset's pair, no warnings."""
    cfg = resolve_preset_config({"EMBEDDING_PRESET": "english"})
    assert cfg == ResolvedPresetConfig(
        provider="onnx",
        model="Xenova/bge-small-en-v1.5",
        preset_short_name="english",
        source="env-preset",
    )


@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_resolver_each_preset_resolves_atomically(name):
    """Every preset name produces (provider, model) matching the registry."""
    cfg = resolve_preset_config({"EMBEDDING_PRESET": name})
    preset = EMBEDDING_PRESETS[name]
    assert cfg.provider == preset.provider
    assert cfg.model == preset.model
    assert cfg.preset_short_name == name
    assert cfg.source == "env-preset"


def test_resolver_embedding_preset_lowercased():
    """Mixed-case preset names normalise (matches obsidian-brain)."""
    cfg = resolve_preset_config({"EMBEDDING_PRESET": "MultiLingual"})
    assert cfg.preset_short_name == "multilingual"
    assert cfg.provider == "onnx"


def test_resolver_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PRESET"):
        resolve_preset_config({"EMBEDDING_PRESET": "ultra-mega-preset"})


def test_resolver_embedding_preset_alias_with_warning(capsys):
    """Deprecated alias inside EMBEDDING_PRESET resolves + warns."""
    cfg = resolve_preset_config({"EMBEDDING_PRESET": "fastest"})
    assert cfg.preset_short_name == "english-fast"
    assert cfg.provider == "onnx"
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()


def test_resolver_provider_matching_preset_no_warning(capsys):
    """When EMBEDDING_PROVIDER == preset.provider, no mismatch warning fires."""
    cfg = resolve_preset_config(
        {"EMBEDDING_PRESET": "english", "EMBEDDING_PROVIDER": "onnx"}
    )
    assert cfg.provider == "onnx"
    captured = capsys.readouterr()
    assert "overrides EMBEDDING_PRESET" not in captured.err


def test_resolver_provider_mismatch_emits_warning(capsys):
    """(2) EMBEDDING_PROVIDER=ollama + EMBEDDING_PRESET=english (onnx preset)
    → provider wins, preset's model carried, mismatch warning."""
    cfg = resolve_preset_config(
        {"EMBEDDING_PRESET": "english", "EMBEDDING_PROVIDER": "ollama"}
    )
    assert cfg.provider == "ollama"
    assert cfg.model == "Xenova/bge-small-en-v1.5"  # carried from preset
    assert cfg.preset_short_name == "english"
    captured = capsys.readouterr()
    assert "EMBEDDING_PROVIDER" in captured.err
    assert "overrides EMBEDDING_PRESET" in captured.err


def test_resolver_provider_mismatch_warning_only_emitted_once(capsys):
    resolve_preset_config(
        {"EMBEDDING_PRESET": "english", "EMBEDDING_PROVIDER": "ollama"}
    )
    capsys.readouterr()  # drain
    resolve_preset_config(
        {"EMBEDDING_PRESET": "multilingual-ollama", "EMBEDDING_PROVIDER": "onnx"}
    )
    second = capsys.readouterr().err
    assert "overrides EMBEDDING_PRESET" not in second


def test_resolver_provider_ollama_alone():
    """(4) EMBEDDING_PROVIDER=ollama alone → DEFAULT_OLLAMA_MODEL."""
    cfg = resolve_preset_config({"EMBEDDING_PROVIDER": "ollama"})
    assert cfg.provider == "ollama"
    assert cfg.model == "nomic-embed-text"
    assert cfg.preset_short_name is None
    assert cfg.source == "env-provider"


def test_resolver_provider_onnx_alone():
    """(4) EMBEDDING_PROVIDER=onnx alone → DEFAULT_PRESET's model."""
    cfg = resolve_preset_config({"EMBEDDING_PROVIDER": "onnx"})
    assert cfg.provider == "onnx"
    assert cfg.model == "Xenova/bge-small-en-v1.5"
    assert cfg.preset_short_name == "english"
    assert cfg.source == "env-provider"


def test_resolver_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        resolve_preset_config({"EMBEDDING_PROVIDER": "garbage"})


def test_resolver_blank_provider_treated_as_unset():
    cfg = resolve_preset_config({"EMBEDDING_PROVIDER": "  "})
    assert cfg.source == "default"
    assert cfg.preset_short_name == "english"


def test_resolver_provider_uppercased():
    """`EMBEDDING_PROVIDER=ONNX` normalises to lowercase."""
    cfg = resolve_preset_config({"EMBEDDING_PROVIDER": "OLLAMA"})
    assert cfg.provider == "ollama"


def test_resolver_precedence_model_beats_preset():
    """(1) > (3): EMBEDDING_MODEL wins over EMBEDDING_PRESET."""
    cfg = resolve_preset_config(
        {"EMBEDDING_MODEL": "my/raw-model", "EMBEDDING_PRESET": "english"}
    )
    assert cfg.model == "my/raw-model"
    assert cfg.source == "env-model"
    assert cfg.preset_short_name is None


def test_resolver_precedence_preset_beats_provider():
    """(3) > (4): preset wins over provider-only when both present."""
    cfg = resolve_preset_config(
        {"EMBEDDING_PRESET": "english-quality", "EMBEDDING_PROVIDER": "onnx"}
    )
    assert cfg.preset_short_name == "english-quality"
    assert cfg.source == "env-preset"


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_default_provider_is_onnx(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    cfg = load_config()
    assert cfg.provider == "onnx"
    # Patch OnnxEmbedder to avoid downloading a model.
    with patch("apple_notes_brain.semantic.embedder.onnx.OnnxEmbedder") as cls:
        cls.return_value = object()
        out = create_embedder(cfg)
        cls.assert_called_once_with(config=cfg)
        assert out is cls.return_value


def test_factory_dispatches_to_ollama(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    cfg = load_config()
    assert cfg.provider == "ollama"
    with patch("apple_notes_brain.semantic.embedder.ollama.OllamaEmbedder") as cls:
        cls.return_value = object()
        out = create_embedder(cfg)
        cls.assert_called_once_with(config=cfg)
        assert out is cls.return_value


def test_factory_rejects_unknown_provider_via_config(monkeypatch, tmp_path):
    """Error surfaces at load_config — not create_embedder."""
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "made-up")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        load_config()


def test_factory_falls_back_to_load_config_when_none_passed(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    with patch("apple_notes_brain.semantic.embedder.onnx.OnnxEmbedder") as cls:
        cls.return_value = object()
        create_embedder()
        assert cls.called


def test_factory_preset_env_routes_to_ollama(monkeypatch, tmp_path):
    """EMBEDDING_PRESET=multilingual-ollama → factory picks OllamaEmbedder
    even though EMBEDDING_PROVIDER is unset."""
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PRESET", "multilingual-ollama")
    cfg = load_config()
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen3-embedding:0.6b"
    assert cfg.preset_short_name == "multilingual-ollama"
    with patch("apple_notes_brain.semantic.embedder.ollama.OllamaEmbedder") as cls:
        cls.return_value = object()
        create_embedder(cfg)
        cls.assert_called_once_with(config=cfg)
