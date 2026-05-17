"""Embedder factory + concrete provider implementations.

Provider/model selection runs through `resolve_preset_config(os.environ)`
in `presets.py` — never re-implement env-var precedence locally.
"""
from __future__ import annotations

from ..config import SemanticConfig, load_config
from ..types import Embedder
from .presets import resolve_preset, resolve_preset_config

__all__ = ["create_embedder", "resolve_preset", "resolve_preset_config"]


def create_embedder(config: SemanticConfig | None = None) -> Embedder:
    """Instantiate an Embedder per the current env config.

    Imports are deferred so missing extras (no onnxruntime, no httpx)
    produce a targeted ImportError naming only the relevant provider
    rather than a confusing transitive failure.

    The provider/model pair on `config` is the source of truth — by the
    time we get here, `load_config()` has already routed env vars
    through `resolve_preset_config()`, so the factory just dispatches
    on `config.provider`.
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
