"""Semantic + hybrid search for apple-notes-brain.

Activated when the `[semantic]` install extra is present (sqlite-vec,
onnxruntime, tokenizers, huggingface-hub, httpx, numpy). Without those,
the four semantic MCP tools degrade to a structured `missing-extras`
error and the lexical `search_notes` tool keeps working unchanged.

Public API surface (re-exported below) intentionally mirrors the names
and shapes of obsidian-brain's TypeScript stack so behaviour stays
comparable across the two -brain servers. Cross-language code sharing
is rejected by design; parallel implementations live and die together
on this seam.

This module does NOT eagerly import heavy dependencies — each submodule
imports them as it needs them, so a bare `import apple_notes_brain.semantic`
succeeds even without the extra installed. The shim in `tools_semantic`
is what catches ImportError and returns the structured tool error.
"""
from __future__ import annotations

from .types import (
    Chunk,
    ChunkAwareResult,
    ChunkerConfig,
    DEFAULT_CHUNKER_CONFIG,
    Embedder,
    EmbedderDeadError,
    EmbedderMetadata,
    MissingExtrasError,
    ModelDownloadError,
    ModelLoadError,
    SearchResult,
    SearchUnique,
    TooLongError,
)

__all__ = [
    "Chunk",
    "ChunkAwareResult",
    "ChunkerConfig",
    "DEFAULT_CHUNKER_CONFIG",
    "Embedder",
    "EmbedderDeadError",
    "EmbedderMetadata",
    "MissingExtrasError",
    "ModelDownloadError",
    "ModelLoadError",
    "SearchResult",
    "SearchUnique",
    "TooLongError",
]
