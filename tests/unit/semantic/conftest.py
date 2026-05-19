"""Shared fixtures + a FakeEmbedder for deterministic semantic-search tests.

Why a FakeEmbedder: the real OnnxEmbedder downloads a 30MB model on first
run and runs inference; that's not appropriate for the unit suite. The
FakeEmbedder hashes the input text into a deterministic float32 vector,
L2-normalises, and returns. Two inputs that hash to nearby keys produce
nearby vectors — close enough to test kNN behaviour, fully reproducible.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from apple_notes_brain.semantic.types import TooLongError


# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic, no I/O, no model file
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """A deterministic stand-in for OnnxEmbedder used across the unit suite.

    The embed function is: take sha256 of (text, task_type), seed a numpy
    random generator with it, draw `dim` standard-normal floats, L2-normalise.
    Same input → same vector. Different inputs → far-apart vectors (high
    probability), which is what kNN tests want.

    Optional `max_chars` simulates a token limit — exceeding it raises
    TooLongError so `test_indexer_failures.py` can exercise the ratchet.
    """

    def __init__(
        self,
        dim: int = 384,
        max_chars: int | None = None,
        provider: str = "fake",
        model_id: str = "fake/deterministic-v1",
    ):
        self._dim = dim
        self._max_chars = max_chars
        self._provider = provider
        self._model_id = model_id
        self.init_count = 0
        self.embed_count = 0
        self.dispose_count = 0

    def init(self) -> None:
        self.init_count += 1

    def embed(self, text: str, task_type: str | None = None) -> np.ndarray:
        if self._max_chars is not None and len(text) > self._max_chars:
            raise TooLongError(
                f"FakeEmbedder: input length {len(text)} exceeds "
                f"max_chars={self._max_chars}"
            )
        self.embed_count += 1
        seed_bytes = hashlib.sha256(
            f"{task_type or ''}\x00{text}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros(self._dim, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def dimensions(self) -> int:
        return self._dim

    def model_identifier(self) -> str:
        return self._model_id

    def provider_name(self) -> str:
        return self._provider

    def dispose(self) -> None:
        self.dispose_count += 1

    def set_metadata(self, meta) -> None:
        """Phase δ Protocol member. Stored for inspection in tests that care."""
        self.metadata = meta


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A fresh init()'d FakeEmbedder per test."""
    emb = FakeEmbedder()
    emb.init()
    return emb


@pytest.fixture
def fake_embedder_factory():
    """Factory for parameterised FakeEmbedders (custom dim / max_chars)."""

    def _make(dim: int = 384, max_chars: int | None = None) -> FakeEmbedder:
        emb = FakeEmbedder(dim=dim, max_chars=max_chars)
        emb.init()
        return emb

    return _make


# ---------------------------------------------------------------------------
# Tmp data-dir fixture — points APPLE_NOTES_BRAIN_DATA_DIR at tmp_path so
# tests don't touch real user state and don't bleed into each other.
# ---------------------------------------------------------------------------

@pytest.fixture
def semantic_data_dir(tmp_path: Path, monkeypatch) -> Iterator[Path]:
    """Set the semantic data dir to a fresh per-test tmp_path."""
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("APPLE_NOTES_BRAIN_MODEL_CACHE", raising=False)
    yield tmp_path


# ---------------------------------------------------------------------------
# Reset env-var-derived config between tests so test order doesn't bite
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_semantic_env(monkeypatch):
    """Remove every APPLE_NOTES_BRAIN_* / EMBEDDING_* / OLLAMA_* env var
    so each test starts from a clean defaults baseline."""
    for key in list(os.environ.keys()):
        if (
            key.startswith("APPLE_NOTES_BRAIN_")
            or key.startswith("EMBEDDING_")
            or key.startswith("OLLAMA_")
        ):
            monkeypatch.delenv(key, raising=False)
    yield
