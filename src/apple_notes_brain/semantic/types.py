"""Core types for the semantic search subsystem.

Mirrors `obsidian-brain/src/embeddings/types.ts` shape-for-shape. Where
the TS version exposes Promise<Float32Array>, the Python port returns a
sync `numpy.ndarray` — async was unnecessary indirection for a per-chunk
indexing loop and complicated the test surface.

Importing this module does NOT pull in numpy. The `embed()` return type
is annotated via `TYPE_CHECKING` so a bare `import` works even when the
`[semantic]` extra is missing — useful for the tools_semantic shim that
needs to detect missing deps without crashing the whole server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Exceptions — structured so callers can pattern-match on type
# ---------------------------------------------------------------------------

class SemanticError(Exception):
    """Base for all semantic-search errors."""


class MissingExtrasError(SemanticError):
    """Raised when the `[semantic]` install extra is required but absent."""


class EmbedderDeadError(SemanticError):
    """Raised when the embedder is unreachable / unrecoverable.

    The indexer treats this as a *fatal* signal: abort the current pass
    rather than soldiering on and producing a partially-populated index.
    """


class TooLongError(SemanticError):
    """Raised when a chunk exceeds the embedder's token capacity.

    The indexer catches this, records the chunk in `failed_chunks` with
    reason='too-long', ratchets `discovered_max_tokens` downward, and
    keeps going. Re-runs use the smaller capacity to keep chunks safe.
    """


class ModelDownloadError(SemanticError):
    """Raised when fetching the embedding model from Hugging Face fails."""


class ModelLoadError(SemanticError):
    """Raised when the downloaded model file can't be parsed by onnxruntime
    (or the tokenizer.json can't be parsed). The OnnxEmbedder catches this
    once, clears the cache, and retries — a second failure propagates."""


# ---------------------------------------------------------------------------
# Embedder protocol — every provider implements this
# ---------------------------------------------------------------------------

# TaskType discriminates query vs document embedding. Symmetric models
# (BGE-small, MiniLM) ignore it. Asymmetric models (BGE-large w/ prefixes,
# E5) prepend a query/document prefix that affects retrieval quality.
TaskType = Literal["document", "query"]


@runtime_checkable
class Embedder(Protocol):
    """Stable contract every concrete embedder satisfies.

    Sync by design: ONNX in-process is sync, and the indexer's per-chunk
    loop has no need to fan out across an event loop. If an Ollama-backed
    embedder wants to overlap network I/O across chunks, it can do that
    internally with a thread pool without changing the protocol.
    """

    def init(self) -> None:
        """Download model / probe dimensions / cache tokenizer. Called once
        before the first embed()."""

    def embed(self, text: str, task_type: TaskType | None = None) -> "np.ndarray":
        """Return an L2-normalised float32 vector of `self.dimensions()`.

        For symmetric models, `task_type` is ignored. For asymmetric models,
        'query' and 'document' produce different vectors.

        Raises:
            TooLongError: chunk exceeds the embedder's token capacity.
            EmbedderDeadError: provider is unreachable (Ollama down,
                ONNX session invalid, etc.).
        """

    def dimensions(self) -> int:
        """Output vector dimensionality. Stable after init()."""

    def model_identifier(self) -> str:
        """Stable string identifying the (provider, model, quantisation)
        triple. Used as part of the cache key so a model swap forces a
        full reindex without manual intervention."""

    def provider_name(self) -> str:
        """Short stable name: 'onnx', 'ollama', 'fake'."""

    def dispose(self) -> None:
        """Release resources. Idempotent."""

    def set_metadata(self, meta: "EmbedderMetadata") -> None:
        """Attach resolved metadata so embed() can apply per-task prefixes.

        Idempotent — safe to call multiple times. Replaces any previously
        attached metadata. The asymmetric prefixes are read from the
        passed-in record on every embed() call (no caching), so future
        re-resolution after a prefix-strategy change just needs another
        set_metadata() — no embedder restart required.
        """


# ---------------------------------------------------------------------------
# Embedder metadata — resolved once at init; cached in embedder_capability
# ---------------------------------------------------------------------------

PrefixSource = Literal[
    "override", "seed", "metadata", "metadata-base", "readme", "fallback", "none"
]


@dataclass(frozen=True)
class EmbedderMetadata:
    """Resolved per-model metadata. Set on the embedder after init().

    Mirrors obsidian-brain's EmbedderMetadata at `src/embeddings/types.ts`.
    """
    model_id: str
    dim: int | None
    max_tokens: int
    query_prefix: str = ""
    document_prefix: str = ""
    prefix_source: PrefixSource = "none"
    base_model: str | None = None
    size_bytes: int | None = None


# ---------------------------------------------------------------------------
# Chunker config + Chunk record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkerConfig:
    """Tuning knobs for `chunk_markdown()`. Defaults match obsidian-brain."""
    chunk_size: int = 1000
    heading_split_depth: int = 4
    preserve_code_blocks: bool = True
    preserve_latex_blocks: bool = True
    min_chunk_chars: int = 50


DEFAULT_CHUNKER_CONFIG = ChunkerConfig()


@dataclass(frozen=True)
class Chunk:
    """One indexable unit. `content_hash` is sha256(heading + content) and
    drives content-addressed dedup across reindex passes."""
    chunk_index: int
    heading: str | None
    heading_level: int | None
    content: str
    content_hash: str
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Search result records
# ---------------------------------------------------------------------------

SearchUnique = Literal["notes", "chunks"]


@dataclass
class SearchResult:
    """Note-level hit. Returned by `Search.semantic()` and `Search.fulltext()`."""
    note_id: str
    title: str
    score: float           # higher = better (cosine similarity for semantic,
                           # negated BM25 for fulltext, RRF sum for hybrid)
    excerpt: str = ""
    folder: str | None = None


@dataclass
class ChunkAwareResult(SearchResult):
    """Note-level hit with chunk metadata attached. The chunk_* fields are
    populated when `unique='chunks'` — they reveal which span of the note
    matched, useful for UI excerpts.

    Score-field semantics (post v1.1 fix):
      * `semantic_score`: raw cosine similarity from the kNN ranker, or
        None if this hit only came in via fulltext.
      * `lexical_score`: negated-BM25 from the fulltext ranker, or None
        if this hit only came in via semantic.
      * `fused_score`: the RRF (reciprocal-rank-fusion) sum across the
        two ranker outputs, populated by `Search.hybrid`. None for pure
        single-ranker calls (`Search.semantic_chunks`, `Search.fulltext`).
      * `score`: the field used for sorting. For pure semantic calls it
        equals `semantic_score`; for hybrid it equals `fused_score`.
    """
    chunk_id: str | None = None
    chunk_heading: str | None = None
    chunk_start_line: int | None = None
    chunk_end_line: int | None = None
    chunk_excerpt: str = ""
    semantic_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None


# ---------------------------------------------------------------------------
# Stats records — returned by the indexer
# ---------------------------------------------------------------------------

@dataclass
class IndexStats:
    """Returned by `IndexPipeline.index_all()`."""
    notes_seen: int = 0
    notes_indexed: int = 0
    notes_skipped: int = 0       # unchanged since last pass
    notes_deleted: int = 0       # removed from source between passes
    chunks_embedded: int = 0
    chunks_skipped: int = 0      # content-hash dedup
    chunks_failed: int = 0
    took_ms: int = 0
    failures: list[dict] = field(default_factory=list)


@dataclass
class SingleNoteResult:
    """Returned by `IndexPipeline.index_single()` (watcher path)."""
    note_id: str
    event: Literal["add", "change", "unlink"]
    chunks_embedded: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    error: str | None = None
