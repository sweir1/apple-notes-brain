"""Small preset registry mapping short names → (provider, repo_id, dim).

Keeps the env-var ergonomics tight (`EMBEDDING_MODEL=bge-small-en-v1.5`)
while allowing power users to pass a fully-qualified repo id like
`BAAI/bge-base-en-v1.5` and have it just work.

Mirrors obsidian-brain's `src/embeddings/presets.ts` — same short names,
same default. When obsidian-brain adds a preset, mirror it here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingPreset:
    """A named (provider-agnostic) embedding model spec.

    `onnx_repo` / `ollama_model` differ because providers use different
    identifier conventions: HuggingFace uses `<org>/<model>`, Ollama uses
    `<model>:<tag>` (often a short alias).
    """
    short_name: str
    onnx_repo: str          # HuggingFace repo for ONNX-quantised weights
    onnx_file: str          # path inside repo for the quantised model
    onnx_tokenizer: str     # path inside repo for tokenizer.json
    ollama_model: str       # ollama-side identifier
    dim: int
    description: str


# ONNX files for BGE come from HuggingFace's official conversions. The
# Xenova/bge-* repos provide pre-quantised ONNX files matched to
# transformers.js's convention (which is what obsidian-brain uses), so
# we point at the same files — bit-for-bit identical model output.
EMBEDDING_PRESETS: dict[str, EmbeddingPreset] = {
    "bge-small-en-v1.5": EmbeddingPreset(
        short_name="bge-small-en-v1.5",
        onnx_repo="Xenova/bge-small-en-v1.5",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="bge-small-en-v1.5",
        dim=384,
        description="Default. 384-dim, ~30MB, fast. MTEB ~91 on retrieval tasks.",
    ),
    "bge-base-en-v1.5": EmbeddingPreset(
        short_name="bge-base-en-v1.5",
        onnx_repo="Xenova/bge-base-en-v1.5",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="bge-base-en-v1.5",
        dim=768,
        description="Higher quality, larger. 768-dim, ~100MB.",
    ),
    "all-MiniLM-L6-v2": EmbeddingPreset(
        short_name="all-MiniLM-L6-v2",
        onnx_repo="Xenova/all-MiniLM-L6-v2",
        onnx_file="onnx/model_quantized.onnx",
        onnx_tokenizer="tokenizer.json",
        ollama_model="all-minilm",
        dim=384,
        description="Classic. 384-dim, ~22MB. Weaker than BGE but lighter.",
    ),
}

DEFAULT_PRESET = EMBEDDING_PRESETS["bge-small-en-v1.5"]


def resolve_preset(
    model: str, provider: Literal["onnx", "ollama"]
) -> tuple[EmbeddingPreset | None, str]:
    """Resolve `EMBEDDING_MODEL` to (preset, concrete_identifier).

    - If `model` matches a preset short_name, return (preset, preset.onnx_repo
      or preset.ollama_model depending on provider).
    - Otherwise treat `model` as a literal — return (None, model) and let
      the concrete embedder use it as-is.
    """
    preset = EMBEDDING_PRESETS.get(model)
    if preset is None:
        return None, model
    identifier = preset.onnx_repo if provider == "onnx" else preset.ollama_model
    return preset, identifier
