"""Tests that ``set_metadata`` + per-task prefix application work end-to-end.

These tests pin the wiring between the metadata resolver chain and the
concrete embedders. They mock the ONNX session / Ollama HTTP client so
no model files are downloaded and no Ollama server is required.

The core observable: ``embed(text, task_type='query')`` and
``embed(text, task_type='document')`` produce DIFFERENT vectors for
an asymmetric model (non-empty distinct prefixes) and IDENTICAL
vectors for a symmetric model (empty prefixes). The vector difference
is sufficient evidence the prefix was prepended before tokenisation.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.embedder.onnx import OnnxEmbedder
from apple_notes_brain.semantic.embedder.ollama import OllamaEmbedder
from apple_notes_brain.semantic.types import EmbedderMetadata


# ---------------------------------------------------------------------------
# OnnxEmbedder — manual surgery to avoid the real model download.
# ---------------------------------------------------------------------------


def _make_onnx_embedder_with_stubs(tmp_path, monkeypatch) -> tuple[OnnxEmbedder, MagicMock]:
    """Return an OnnxEmbedder whose tokenize→run path is mocked.

    The mock records every text passed to embed() so tests can assert
    the prefix was prepended. The mock returns a vector derived
    deterministically from the input text so different inputs → different
    outputs.
    """
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    cfg = load_config()
    emb = OnnxEmbedder(cfg)

    # Mark as "initialised" by setting the post-init() attributes manually.
    emb._dim = 4
    emb._disposed = False

    seen_texts: list[str] = []

    class _FakeTokenizer:
        def encode(_self, text: str):
            seen_texts.append(text)
            # 4 ids, attention mask 1111. Use the text length to vary
            # output slightly so the output vector changes per text.
            length = max(len(text) % 4 + 1, 2)
            ids = [1] * length
            mask = [1] * length
            enc = MagicMock()
            enc.ids = ids
            enc.attention_mask = mask
            return enc

    class _FakeSession:
        def get_inputs(_self):
            class _Inp:
                name = "input_ids"

            class _Inp2:
                name = "attention_mask"

            return [_Inp(), _Inp2()]

        def run(_self, _names, inputs):
            # token-embeddings shape (1, seq_len, hidden=4) — values
            # derived from the actual input ids so different prefixes
            # produce different output vectors.
            ids = inputs["input_ids"]
            seq_len = ids.shape[1]
            text = seen_texts[-1]
            base = float((hash(text) % 7919) / 7919.0) + 0.1
            token_emb = (
                np.arange(seq_len * 4, dtype=np.float32).reshape(1, seq_len, 4) + base
            )
            return [token_emb]

    emb._tokenizer = _FakeTokenizer()
    emb._session = _FakeSession()

    recorder = MagicMock()
    recorder.seen_texts = seen_texts
    return emb, recorder


def test_onnx_set_metadata_is_idempotent(tmp_path, monkeypatch):
    emb, _ = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    meta1 = EmbedderMetadata(
        model_id="m", dim=4, max_tokens=512,
        query_prefix="q1: ", document_prefix="d1: ",
    )
    meta2 = EmbedderMetadata(
        model_id="m", dim=4, max_tokens=512,
        query_prefix="q2: ", document_prefix="d2: ",
    )
    emb.set_metadata(meta1)
    emb.set_metadata(meta2)
    assert emb._metadata == meta2  # last call wins


def test_onnx_embed_without_metadata_uses_text_verbatim(tmp_path, monkeypatch):
    """Pre-Phase-δ behaviour: no metadata attached → no prefix prepended."""
    emb, rec = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.embed("hello world", task_type="query")
    emb.embed("hello world", task_type="document")
    assert rec.seen_texts == ["hello world", "hello world"]


def test_onnx_asymmetric_metadata_prepends_correct_prefix(tmp_path, monkeypatch):
    emb, rec = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="query: ", document_prefix="passage: ",
        )
    )
    emb.embed("hello world", task_type="query")
    emb.embed("hello world", task_type="document")
    assert rec.seen_texts == ["query: hello world", "passage: hello world"]


def test_onnx_asymmetric_metadata_produces_different_vectors(tmp_path, monkeypatch):
    """The whole point of the prefix mechanism — different prefix
    → different embedding for the same source text."""
    emb, _ = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="query: ", document_prefix="passage: ",
        )
    )
    v_query = emb.embed("hello world", task_type="query")
    v_doc = emb.embed("hello world", task_type="document")
    assert v_query.shape == v_doc.shape
    # Different prefixes → different vectors.
    assert not np.array_equal(v_query, v_doc)


def test_onnx_symmetric_metadata_produces_identical_vectors(tmp_path, monkeypatch):
    """Empty prefixes → query and document embeddings identical."""
    emb, _ = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="", document_prefix="",
        )
    )
    v_query = emb.embed("hello world", task_type="query")
    v_doc = emb.embed("hello world", task_type="document")
    np.testing.assert_array_equal(v_query, v_doc)


def test_onnx_task_type_none_treated_as_document(tmp_path, monkeypatch):
    emb, rec = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="q: ", document_prefix="d: ",
        )
    )
    emb.embed("hello", task_type=None)
    assert rec.seen_texts[-1] == "d: hello"


def test_onnx_empty_query_prefix_is_noop(tmp_path, monkeypatch):
    """Asymmetric model with empty query_prefix (instruction-only doc prefix)."""
    emb, rec = _make_onnx_embedder_with_stubs(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="", document_prefix="passage: ",
        )
    )
    emb.embed("hello", task_type="query")
    emb.embed("hello", task_type="document")
    assert rec.seen_texts == ["hello", "passage: hello"]


# ---------------------------------------------------------------------------
# OllamaEmbedder — mock the HTTP client.
# ---------------------------------------------------------------------------


def _make_ollama_embedder_with_mock(tmp_path, monkeypatch) -> tuple[OllamaEmbedder, list[str]]:
    """Return an OllamaEmbedder whose embed-via-http is mocked.

    Records every input text in ``seen_texts``. The returned vector
    embeds the text length so different inputs → different outputs.
    """
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("APPLE_NOTES_BRAIN_OLLAMA_NUM_CTX", "2048")
    cfg = load_config()
    emb = OllamaEmbedder(cfg)
    emb._dim = 4
    emb._disposed = False
    emb._cached_num_ctx = 2048

    seen_texts: list[str] = []

    class _FakeClient:
        def post(self, _path, json=None, **_):
            seen_texts.append(json["input"] if "input" in json else json.get("prompt"))
            resp = MagicMock()
            resp.status_code = 200
            # Hash-derived 4-dim vector so different inputs → different out.
            t = seen_texts[-1]
            base = float((hash(t) % 7919) / 7919.0) + 0.1
            resp.json = lambda: {"embeddings": [[base, base + 0.1, base + 0.2, base + 0.3]]}
            return resp

        def close(self):
            pass

    emb._client = _FakeClient()
    return emb, seen_texts


def test_ollama_set_metadata_is_idempotent(tmp_path, monkeypatch):
    emb, _ = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    meta1 = EmbedderMetadata(
        model_id="m", dim=4, max_tokens=512,
        query_prefix="q1: ", document_prefix="d1: ",
    )
    meta2 = EmbedderMetadata(
        model_id="m", dim=4, max_tokens=512,
        query_prefix="q2: ", document_prefix="d2: ",
    )
    emb.set_metadata(meta1)
    emb.set_metadata(meta2)
    assert emb._metadata == meta2


def test_ollama_embed_without_metadata_uses_text_verbatim(tmp_path, monkeypatch):
    emb, seen = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    emb.embed("hello", task_type="query")
    emb.embed("hello", task_type="document")
    assert seen == ["hello", "hello"]


def test_ollama_asymmetric_metadata_prepends_correct_prefix(tmp_path, monkeypatch):
    emb, seen = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="Instruct: ", document_prefix="",
        )
    )
    emb.embed("hello", task_type="query")
    emb.embed("hello", task_type="document")
    # Document side has empty prefix → no change. Query side is prefixed.
    assert seen == ["Instruct: hello", "hello"]


def test_ollama_asymmetric_metadata_produces_different_vectors(tmp_path, monkeypatch):
    emb, _ = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="query: ", document_prefix="passage: ",
        )
    )
    v_query = emb.embed("hello world", task_type="query")
    v_doc = emb.embed("hello world", task_type="document")
    assert v_query.shape == v_doc.shape
    assert not np.array_equal(v_query, v_doc)


def test_ollama_symmetric_metadata_produces_identical_vectors(tmp_path, monkeypatch):
    emb, _ = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="", document_prefix="",
        )
    )
    v_query = emb.embed("hello world", task_type="query")
    v_doc = emb.embed("hello world", task_type="document")
    np.testing.assert_array_equal(v_query, v_doc)


def test_ollama_task_type_none_treated_as_document(tmp_path, monkeypatch):
    emb, seen = _make_ollama_embedder_with_mock(tmp_path, monkeypatch)
    emb.set_metadata(
        EmbedderMetadata(
            model_id="m", dim=4, max_tokens=512,
            query_prefix="q: ", document_prefix="d: ",
        )
    )
    emb.embed("hi", task_type=None)
    assert seen[-1] == "d: hi"
