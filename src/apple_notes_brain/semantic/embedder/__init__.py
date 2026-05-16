"""Embedder factory + concrete provider implementations.

Provider selection is `EMBEDDING_PROVIDER` env var. Default 'onnx'.
"""
from __future__ import annotations

from ..config import SemanticConfig, load_config
from ..types import Embedder
from .presets import resolve_preset

__all__ = ["create_embedder", "resolve_preset"]


def create_embedder(config: SemanticConfig | None = None) -> Embedder:
    """Instantiate an Embedder per the current env config.

    Imports are deferred so missing extras (no onnxruntime, no httpx)
    produce a targeted ImportError naming only the relevant provider
    rather than a confusing transitive failure.
    """
    cfg = config or load_config()

    if cfg.provider == "onnx":
        from .onnx import OnnxEmbedder

        return OnnxEmbedder(config=cfg)
    if cfg.provider == "ollama":
        from .ollama import OllamaEmbedder

        return OllamaEmbedder(config=cfg)
    # config.py already validates the provider; this is belt-and-braces.
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER={cfg.provider!r}")
