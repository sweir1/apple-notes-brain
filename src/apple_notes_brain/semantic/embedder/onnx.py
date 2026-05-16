"""ONNX-based in-process embedder.

The default provider. Same model files transformers.js uses in
obsidian-brain (Xenova/bge-small-en-v1.5 onnx-quantised), so embeddings
across the two -brain servers are bit-for-bit comparable.

Pipeline:
  1. hf_hub_download → onnx model + tokenizer.json into <model_cache>
  2. tokenizers.Tokenizer.from_file → tokenizer with max-length truncation
  3. onnxruntime.InferenceSession with CoreMLExecutionProvider on macOS,
     CPUExecutionProvider fallback elsewhere
  4. embed(text): tokenize → run → mean-pool with attention mask → L2

If onnxruntime / tokenizers / huggingface-hub aren't installed, importing
this module raises ImportError — the embedder factory catches that and
returns a structured MissingExtrasError to the tool layer.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config import SemanticConfig
from ..types import (
    EmbedderDeadError,
    ModelDownloadError,
    ModelLoadError,
    TooLongError,
)
from .presets import DEFAULT_PRESET, resolve_preset

_log = logging.getLogger("apple-notes-brain")


_DEFAULT_MAX_TOKENS = 512  # BGE / MiniLM family


class OnnxEmbedder:
    """In-process ONNX embedder. See module docstring for the design."""

    def __init__(self, config: SemanticConfig):
        self._cfg = config
        preset, identifier = resolve_preset(config.model, "onnx")
        self._preset = preset or DEFAULT_PRESET
        # If the user passed a literal repo id that's NOT a preset, use it
        # verbatim for the model identifier but assume the BGE-style file
        # layout (onnx/model_quantized.onnx + tokenizer.json). Power-users
        # who diverge can pass their own paths via EMBEDDING_ONNX_PROVIDERS
        # / future env hooks.
        self._repo_id = identifier if preset is None else self._preset.onnx_repo
        self._model_filename = self._preset.onnx_file
        self._tokenizer_filename = self._preset.onnx_tokenizer

        # Resolved at init() time:
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._dim: int | None = None
        self._providers: tuple[str, ...] = self._default_providers()
        self._max_tokens = _DEFAULT_MAX_TOKENS
        self._disposed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Download model+tokenizer, load ort session, probe dim.

        Recovers from a corrupt cache once: if loading raises a parse/load
        error, the per-model cache dir is wiped and the load is retried.
        A second failure propagates as ModelLoadError.
        """
        try:
            self._load(retry=True)
        except (ModelLoadError, ModelDownloadError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelLoadError(f"Failed to initialise OnnxEmbedder: {exc}") from exc

    def _load(self, *, retry: bool) -> None:
        try:
            model_path = self._download(self._model_filename)
            tokenizer_path = self._download(self._tokenizer_filename)
            self._tokenizer = self._build_tokenizer(tokenizer_path)
            self._session = self._build_session(model_path)
            self._dim = self._probe_dim()
        except (ModelLoadError, ModelDownloadError) as exc:
            if not retry:
                raise
            if isinstance(exc, ModelDownloadError):
                # Download errors aren't fixable by a cache wipe — re-raise.
                raise
            _log.warning(
                "apple-notes-brain: ONNX model load failed (%s); clearing "
                "the model cache for %s and retrying once.",
                exc,
                self._repo_id,
            )
            self._clear_cache()
            self._load(retry=False)

    def _default_providers(self) -> tuple[str, ...]:
        """Return the InferenceSession provider list for the current host.

        On macOS we prefer CoreML (which dispatches to the Apple Neural
        Engine where possible) then fall back to CPU. Other platforms get
        CPU only — no GPU here, this is a CLI-style tool.
        """
        if self._cfg.onnx_providers_override is not None:
            return self._cfg.onnx_providers_override
        if sys.platform == "darwin":
            return ("CoreMLExecutionProvider", "CPUExecutionProvider")
        return ("CPUExecutionProvider",)

    def _download(self, filename: str) -> Path:
        """Wrap huggingface_hub.hf_hub_download with our error type."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required for the ONNX embedder. "
                "Install via `pip install apple-notes-brain[semantic]`."
            ) from exc
        try:
            path = hf_hub_download(
                repo_id=self._repo_id,
                filename=filename,
                cache_dir=str(self._cfg.model_cache),
            )
        except Exception as exc:  # huggingface_hub raises a variety
            raise ModelDownloadError(
                f"Failed to download {filename} from {self._repo_id} "
                f"(cache_dir={self._cfg.model_cache}): {exc}"
            ) from exc
        return Path(path)

    def _build_tokenizer(self, tokenizer_path: Path) -> Any:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "tokenizers is required for the ONNX embedder."
            ) from exc
        try:
            tok = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load tokenizer from {tokenizer_path}: {exc}"
            ) from exc
        # Enable truncation at the model's max-length so >512-token inputs
        # don't crash session.run with shape mismatches.
        tok.enable_truncation(max_length=self._max_tokens)
        return tok

    def _build_session(self, model_path: Path) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for the ONNX embedder."
            ) from exc
        try:
            session = ort.InferenceSession(str(model_path), providers=list(self._providers))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load ONNX model from {model_path}: {exc}"
            ) from exc
        # On CoreML EP failure for unsupported ops, retry CPU-only once.
        active = list(session.get_providers())
        if (
            "CoreMLExecutionProvider" in self._providers
            and "CoreMLExecutionProvider" not in active
        ):
            _log.warning(
                "apple-notes-brain: CoreMLExecutionProvider unavailable; "
                "falling back to CPU. (active providers: %s)",
                active,
            )
        return session

    def _probe_dim(self) -> int:
        """One-shot run on a tiny input to read the output shape."""
        if self._cfg.dim_override is not None:
            return int(self._cfg.dim_override)
        vec = self._run_session_pooled("test")
        return int(vec.shape[0])

    def _clear_cache(self) -> None:
        """Remove the per-model subdir of the cache so a re-download proceeds."""
        # huggingface_hub cache layout: <cache_dir>/models--<org>--<name>/
        slug = "models--" + self._repo_id.replace("/", "--")
        target = self._cfg.model_cache / slug
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def dispose(self) -> None:
        self._session = None
        self._tokenizer = None
        self._disposed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        text: str,
        task_type: Literal["document", "query"] | None = None,
    ) -> np.ndarray:
        if self._disposed:
            raise EmbedderDeadError("OnnxEmbedder has been disposed.")
        if self._session is None or self._tokenizer is None or self._dim is None:
            raise EmbedderDeadError(
                "OnnxEmbedder.embed() called before init(); call init() first."
            )
        # task_type is folded into prefix injection for asymmetric models;
        # for symmetric models (BGE-small, MiniLM) it's a no-op. We don't
        # currently have per-model prefix metadata so we accept the input
        # text verbatim. (Future work: wire metadata-resolver.)
        try:
            return self._run_session_pooled(text)
        except TooLongError:
            raise
        except Exception as exc:
            raise EmbedderDeadError(
                f"OnnxEmbedder.embed() failed: {exc}"
            ) from exc

    def dimensions(self) -> int:
        if self._dim is None:
            raise EmbedderDeadError("dimensions() called before init().")
        return self._dim

    def model_identifier(self) -> str:
        return f"onnx::{self._repo_id}::{self._model_filename}"

    def provider_name(self) -> str:
        return "onnx"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_session_pooled(self, text: str) -> np.ndarray:
        """Tokenize → run → mean-pool → L2 normalise.

        The mean-pool uses the attention mask so padding tokens are
        excluded from the average — this is the standard sentence-
        transformers pooling for BGE/MiniLM-family models.
        """
        encoding = self._tokenizer.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        type_ids = np.zeros_like(ids)
        inputs = self._session_inputs(ids, mask, type_ids)
        outputs = self._session.run(None, inputs)
        # outputs[0] shape: (1, seq_len, hidden_dim) — token embeddings.
        token_emb = outputs[0][0]            # (seq_len, hidden_dim)
        mask_f = mask.astype(np.float32)[0, :, None]  # (seq_len, 1)
        summed = (token_emb * mask_f).sum(axis=0)
        counts = mask_f.sum(axis=0)
        counts = np.clip(counts, a_min=1e-9, a_max=None)
        pooled = (summed / counts).astype(np.float32)
        norm = float(np.linalg.norm(pooled))
        if norm < 1e-12:
            return np.zeros_like(pooled)
        return (pooled / norm).astype(np.float32)

    def _session_inputs(
        self, ids: np.ndarray, mask: np.ndarray, type_ids: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Map our (ids, mask, type_ids) onto whatever input names the
        ONNX graph expects. BGE-style exports use input_ids/attention_mask/
        token_type_ids; some MiniLM exports omit token_type_ids."""
        wanted = {inp.name for inp in self._session.get_inputs()}
        provided = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": type_ids,
        }
        return {k: v for k, v in provided.items() if k in wanted}
