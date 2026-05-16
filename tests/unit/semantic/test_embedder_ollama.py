"""Tests for the Ollama HTTP embedder.

The `httpx.Client` is mocked so no network goes out. We verify request
shape (URLs, payloads), response parsing (both `/api/embed` and the
legacy `/api/embeddings`), retry on 5xx, auto-pull paths, and the
num_ctx resolution precedence.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from apple_notes_brain.semantic.config import (
    DEFAULT_OLLAMA_NUM_CTX_FALLBACK,
    load_config,
)
from apple_notes_brain.semantic.embedder.ollama import OllamaEmbedder
from apple_notes_brain.semantic.types import EmbedderDeadError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    return load_config()


class _FakeResponse:
    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeStreamResponse:
    """Used by /api/pull. Provides .read() and iter_lines() and acts as a
    context manager."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines
        self._read_bytes = b"".join(lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def read(self):
        return self._read_bytes

    def iter_lines(self):
        for line in self._lines:
            yield line


def _build_client(*, tags=None, embed_response=None, show_params="num_ctx 4096"):
    """Construct a MagicMock httpx.Client wired up with sensible defaults."""
    client = MagicMock(spec=httpx.Client)

    tags_payload = {"models": [{"name": n} for n in (tags or ["bge-small-en-v1.5"])]}

    def get(url, *a, **kw):
        if url == "/api/tags":
            return _FakeResponse(payload=tags_payload)
        return _FakeResponse(status_code=404, text="not found")

    def post(url, *a, **kw):
        if url == "/api/embed":
            return _FakeResponse(payload=embed_response)
        if url == "/api/embeddings":
            return _FakeResponse(
                payload={"embedding": embed_response["embeddings"][0]}
                if embed_response and "embeddings" in embed_response
                else {"embedding": [0.5, 0.5, 0.5, 0.5]}
            )
        if url == "/api/show":
            return _FakeResponse(payload={"parameters": show_params})
        return _FakeResponse(status_code=404, text="not found")

    client.get.side_effect = get
    client.post.side_effect = post
    return client


def _patch_httpx_client(client):
    return patch("httpx.Client", return_value=client)


# ---------------------------------------------------------------------------
# Init / model availability
# ---------------------------------------------------------------------------

def test_init_succeeds_when_model_present(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(
        tags=["bge-small-en-v1.5"],
        embed_response={"embeddings": [[0.1, 0.2, 0.3, 0.4]]},
    )
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        assert emb.dimensions() == 4
        assert emb.provider_name() == "ollama"


def test_init_matches_tag_stripped_name(monkeypatch, tmp_path):
    """Local copy `bge-small-en-v1.5:latest` is still considered present."""
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(
        tags=["bge-small-en-v1.5:latest"],
        embed_response={"embeddings": [[1.0, 0.0]]},
    )
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()


def test_init_auto_pulls_missing_model(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(
        tags=["different-model"],
        embed_response={"embeddings": [[1.0, 0.0]]},
    )
    pulled = []

    def stream(method, url, json=None, **kw):
        pulled.append((method, url, json))
        return _FakeStreamResponse(
            lines=[b'{"status": "pulling"}', b'{"status": "success"}']
        )

    client.stream.side_effect = stream
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        assert pulled and pulled[0][1] == "/api/pull"


def test_init_refuses_when_auto_pull_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL", "0")
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(tags=["different-model"])
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        with pytest.raises(EmbedderDeadError, match="AUTO_PULL=0"):
            emb.init()


def test_init_surfaces_connection_failure(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = MagicMock(spec=httpx.Client)
    client.get.side_effect = httpx.ConnectError("connection refused")
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        with pytest.raises(EmbedderDeadError, match="Cannot reach Ollama"):
            emb.init()


# ---------------------------------------------------------------------------
# Embedding requests + response parsing
# ---------------------------------------------------------------------------

def test_embed_calls_modern_endpoint(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0, 0.0, 0.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        v = emb.embed("hello")
        assert v.shape == (3,)
        # Sent to /api/embed with model + input.
        relevant = [c for c in client.post.mock_calls if c.args[0] == "/api/embed"]
        assert relevant


def test_embed_parses_legacy_endpoint_when_modern_404s(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = MagicMock(spec=httpx.Client)
    # /api/tags + /api/show usual; /api/embed → 404; /api/embeddings → 200.
    def post(url, *a, **kw):
        if url == "/api/embed":
            return _FakeResponse(status_code=404, text="endpoint not found")
        if url == "/api/embeddings":
            return _FakeResponse(payload={"embedding": [0.6, 0.8]})
        if url == "/api/show":
            return _FakeResponse(payload={"parameters": "num_ctx 4096"})
        return _FakeResponse(status_code=404)

    client.get.side_effect = lambda url, *a, **kw: (
        _FakeResponse(payload={"models": [{"name": "bge-small-en-v1.5"}]})
        if url == "/api/tags"
        else _FakeResponse(status_code=404)
    )
    client.post.side_effect = post
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        v = emb.embed("x")
        assert v.shape == (2,)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)


def test_embed_output_is_l2_normalised(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[3.0, 4.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        v = emb.embed("x")
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)


def test_embed_retries_on_5xx_then_raises(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0]]})

    seq = iter([
        _FakeResponse(status_code=500, text="boom"),
        _FakeResponse(status_code=503, text="still boom"),
    ])

    def post(url, *a, **kw):
        if url in ("/api/embed", "/api/embeddings"):
            return next(seq)
        if url == "/api/show":
            return _FakeResponse(payload={"parameters": "num_ctx 4096"})
        return _FakeResponse(status_code=404)

    client.post.side_effect = post
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        # init() does its own embed-probe; with all-5xx that probe fails.
        with pytest.raises(EmbedderDeadError):
            emb.init()


def test_embed_raises_for_missing_embeddings_key(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"unexpected": "shape"})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        with pytest.raises(EmbedderDeadError, match="response missing"):
            emb.init()


# ---------------------------------------------------------------------------
# num_ctx resolution precedence
# ---------------------------------------------------------------------------

def test_num_ctx_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "1234")
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        assert emb._effective_num_ctx() == 1234


def test_num_ctx_from_api_show_when_no_override(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(
        embed_response={"embeddings": [[1.0]]}, show_params="num_ctx 8192"
    )
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        assert emb._effective_num_ctx() == 8192


def test_num_ctx_falls_back_to_default(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(
        embed_response={"embeddings": [[1.0]]}, show_params="(no num_ctx here)"
    )
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        assert emb._effective_num_ctx() == DEFAULT_OLLAMA_NUM_CTX_FALLBACK


# ---------------------------------------------------------------------------
# Metadata / disposal
# ---------------------------------------------------------------------------

def test_model_identifier_includes_base_url_and_model(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        ident = emb.model_identifier()
        assert "ollama::" in ident
        assert "localhost:11434" in ident
        assert "bge-small-en-v1.5" in ident


def test_dispose_closes_client(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        emb.dispose()
        client.close.assert_called_once()


def test_embed_after_dispose_raises(monkeypatch, tmp_path):
    cfg = _ollama_cfg(monkeypatch, tmp_path)
    client = _build_client(embed_response={"embeddings": [[1.0]]})
    with _patch_httpx_client(client):
        emb = OllamaEmbedder(config=cfg)
        emb.init()
        emb.dispose()
        with pytest.raises(EmbedderDeadError):
            emb.embed("x")
