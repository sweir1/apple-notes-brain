"""Tests for the embedder factory + preset resolver.

The factory dispatches on `EMBEDDING_PROVIDER`. We don't actually
construct an OnnxEmbedder/OllamaEmbedder here (those have their own
test files); we verify the dispatch and that the preset resolver
returns sane (preset, identifier) pairs.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.embedder import create_embedder
from apple_notes_brain.semantic.embedder.presets import (
    DEFAULT_PRESET,
    EMBEDDING_PRESETS,
    resolve_preset,
)


# ---------------------------------------------------------------------------
# Preset resolver
# ---------------------------------------------------------------------------

def test_default_preset_is_bge_small():
    assert DEFAULT_PRESET.short_name == "bge-small-en-v1.5"
    assert DEFAULT_PRESET.dim == 384


def test_resolve_preset_known_short_name_onnx():
    preset, identifier = resolve_preset("bge-small-en-v1.5", "onnx")
    assert preset is not None
    assert preset.short_name == "bge-small-en-v1.5"
    assert identifier == "Xenova/bge-small-en-v1.5"


def test_resolve_preset_known_short_name_ollama():
    preset, identifier = resolve_preset("bge-small-en-v1.5", "ollama")
    assert preset is not None
    assert identifier == "bge-small-en-v1.5"


def test_resolve_preset_unknown_returns_literal():
    preset, identifier = resolve_preset("my/custom-model", "onnx")
    assert preset is None
    assert identifier == "my/custom-model"


@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_all_registered_presets_resolve(name):
    preset, identifier = resolve_preset(name, "onnx")
    assert preset is not None
    assert preset.short_name == name


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_default_provider_is_onnx(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    cfg = load_config()
    assert cfg.provider == "onnx"
    # Patch the OnnxEmbedder to avoid actually downloading a model; we
    # only verify the factory wires the right class.
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
    """The error surfaces at load_config — not create_embedder."""
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
