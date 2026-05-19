"""Ollama HTTP embedder — the secondary provider.

`obsidian-brain/src/embeddings/ollama.ts` is the reference. The same
provider semantics carry across:
  * `/api/embed` (modern, preferred) → falls back to `/api/embeddings`
  * `/api/show` to probe num_ctx
  * `/api/pull` to auto-download missing models (gated by config)

Errors map to the structured EmbedderDeadError so the indexer can abort
the current pass cleanly when Ollama is unreachable.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

import numpy as np

from .._logging import debug_log
from ..config import SemanticConfig, DEFAULT_OLLAMA_NUM_CTX_FALLBACK
from ..types import EmbedderDeadError, EmbedderMetadata, TooLongError
from .presets import DEFAULT_PRESET, resolve_preset

_log = logging.getLogger("apple-notes-brain")


class OllamaEmbedder:
    """HTTP embedder talking to a local Ollama server."""

    def __init__(self, config: SemanticConfig):
        self._cfg = config
        preset, identifier = resolve_preset(config.model, "ollama")
        self._preset = preset or DEFAULT_PRESET
        self._model = identifier if preset is None else preset.ollama_model
        self._base_url = config.ollama_base_url.rstrip("/")
        self._dim: int | None = None
        self._cached_num_ctx: int | None = None
        self._client: Any | None = None
        self._disposed = False
        # Resolved metadata — attached post-init() by the metadata resolver.
        # None means "no prefixes; assume symmetric model".
        self._metadata: EmbedderMetadata | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        debug_log(f"ollama: probing {self._base_url} for model {self._model}")
        self._client = self._build_client()
        # Ensure the model is available locally. If it isn't, auto-pull
        # when allowed; otherwise raise — embedding against a missing
        # model would 404 every call.
        present = self._model_available()
        debug_log(f"ollama: model present={present}")
        if not present:
            if self._cfg.ollama_auto_pull:
                debug_log(f"ollama: pulling {self._model}...")
                self._pull()
            else:
                raise EmbedderDeadError(
                    f"Ollama model {self._model!r} not present on {self._base_url} "
                    f"and APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL=0. "
                    "Either run `ollama pull <model>` or enable auto-pull."
                )
        self._cached_num_ctx = self._probe_num_ctx()
        # Probe dim by embedding a one-token input.
        vec = self._embed_via_http("test")
        self._dim = int(vec.shape[0])
        _log.info(
            "ollama embedder ready: model=%s dim=%d", self._model, self._dim
        )

    def dispose(self) -> None:
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._client = None
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
            raise EmbedderDeadError("OllamaEmbedder has been disposed.")
        if self._client is None or self._dim is None:
            raise EmbedderDeadError("OllamaEmbedder.embed() called before init().")
        # Soft input-length guard. Ollama itself silently truncates beyond
        # num_ctx; we surface a TooLongError so the indexer's ratchet kicks
        # in for chunks visibly oversized.
        approx_tokens = len(text) // 4 + 1  # cheap upper bound
        if approx_tokens > self._effective_num_ctx() * 2:
            raise TooLongError(
                f"OllamaEmbedder: input ~{approx_tokens} tokens exceeds 2x "
                f"effective num_ctx {self._effective_num_ctx()}"
            )
        prefixed = self._apply_prefix(text, task_type)
        try:
            return self._embed_via_http(prefixed)
        except TooLongError:
            raise
        except Exception as exc:
            raise EmbedderDeadError(
                f"OllamaEmbedder.embed() failed: {exc}"
            ) from exc

    def set_metadata(self, meta: EmbedderMetadata) -> None:
        """Attach resolved metadata. Idempotent — last call wins."""
        self._metadata = meta

    def _apply_prefix(
        self, text: str, task_type: Literal["document", "query"] | None
    ) -> str:
        """Prepend the query / document prefix from resolved metadata.

        Symmetric models (empty prefixes) and pre-resolver state (no
        metadata attached) are no-ops; the input is returned unchanged.
        """
        meta = self._metadata
        if meta is None:
            return text
        prefix = meta.query_prefix if task_type == "query" else meta.document_prefix
        if not prefix:
            return text
        return prefix + text

    def dimensions(self) -> int:
        if self._dim is None:
            raise EmbedderDeadError("dimensions() called before init().")
        return self._dim

    def model_identifier(self) -> str:
        return f"ollama::{self._base_url}::{self._model}"

    def provider_name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _build_client(self) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the Ollama embedder. "
                "Install via `pip install apple-notes-brain[semantic]`."
            ) from exc
        return httpx.Client(base_url=self._base_url, timeout=httpx.Timeout(60.0))

    def _model_available(self) -> bool:
        """Check /api/tags for our model name. False on any connection error."""
        try:
            resp = self._client.get("/api/tags")
        except Exception as exc:
            raise EmbedderDeadError(
                f"Cannot reach Ollama at {self._base_url} ({exc}). "
                "Start Ollama or override OLLAMA_BASE_URL."
            ) from exc
        if resp.status_code != 200:
            raise EmbedderDeadError(
                f"Ollama /api/tags returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        names = [m.get("name", "") for m in data.get("models", [])]
        # Tag-stripped name match (e.g. our model is 'bge-small-en-v1.5',
        # local copy might be 'bge-small-en-v1.5:latest').
        wanted = self._model
        wanted_stem = wanted.split(":")[0]
        for name in names:
            if name == wanted:
                return True
            if name.split(":")[0] == wanted_stem:
                return True
        return False

    def _pull(self) -> None:
        """Pull the model via /api/pull. Streams progress lines; we don't
        surface them here (the MCP transport is stdio so any console
        printing would corrupt the protocol)."""
        try:
            with self._client.stream(
                "POST",
                "/api/pull",
                json={"name": self._model, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    raise EmbedderDeadError(
                        f"Ollama /api/pull returned {resp.status_code} for "
                        f"{self._model}: {resp.read().decode('utf-8', 'replace')[:200]}"
                    )
                # Consume the stream so the server completes the pull
                # before we return.
                for chunk in resp.iter_lines():
                    if not chunk:
                        continue
                    # `chunk` is already bytes; httpx decodes if str=True.
                    try:
                        payload = json.loads(chunk) if isinstance(chunk, (str, bytes)) else {}
                    except Exception:
                        payload = {}
                    status = payload.get("status", "")
                    if status == "success":
                        return
                    if "error" in payload:
                        raise EmbedderDeadError(
                            f"Ollama pull failed: {payload['error']}"
                        )
        except EmbedderDeadError:
            raise
        except Exception as exc:
            raise EmbedderDeadError(
                f"Ollama pull of {self._model} failed: {exc}"
            ) from exc

    def _probe_num_ctx(self) -> int | None:
        """Read /api/show parameters.num_ctx if present. None on failure."""
        if self._cfg.ollama_num_ctx_override is not None:
            return self._cfg.ollama_num_ctx_override
        try:
            resp = self._client.post("/api/show", json={"name": self._model})
            if resp.status_code != 200:
                return None
            params = resp.json().get("parameters", "")
            if isinstance(params, str):
                # /api/show returns parameters as a multi-line string with
                # KEY VALUE lines. Look for `num_ctx N`.
                for line in params.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == "num_ctx":
                        try:
                            return int(parts[1])
                        except ValueError:
                            return None
            return None
        except Exception:
            return None

    def _effective_num_ctx(self) -> int:
        """Mirrors `OllamaEmbedder.effectiveNumCtx` in obsidian-brain."""
        if self._cfg.ollama_num_ctx_override is not None:
            return self._cfg.ollama_num_ctx_override
        if self._cached_num_ctx is not None:
            return self._cached_num_ctx
        return DEFAULT_OLLAMA_NUM_CTX_FALLBACK

    def _embed_via_http(self, text: str) -> np.ndarray:
        """POST /api/embed with retry on 5xx. Returns float32 unit vector."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._client.post(
                    "/api/embed",
                    json={"model": self._model, "input": text},
                )
                if resp.status_code == 404:
                    # Older Ollama: /api/embed didn't exist; try legacy.
                    resp = self._client.post(
                        "/api/embeddings",
                        json={"model": self._model, "prompt": text},
                    )
                if 500 <= resp.status_code < 600:
                    last_exc = EmbedderDeadError(
                        f"Ollama {resp.status_code}: {resp.text[:200]}"
                    )
                    time.sleep(0.2 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    raise EmbedderDeadError(
                        f"Ollama returned {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                # Modern /api/embed returns {"embeddings": [[...]]}; legacy
                # /api/embeddings returns {"embedding": [...]}. Accept both.
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    raw = data["embeddings"][0]
                elif "embedding" in data:
                    raw = data["embedding"]
                else:
                    raise EmbedderDeadError(
                        f"Ollama response missing embeddings: {data!r}"
                    )
                vec = np.asarray(raw, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vec))
                if norm < 1e-12:
                    return vec
                return (vec / norm).astype(np.float32)
            except Exception as exc:
                last_exc = exc
                if attempt == 1:
                    break
        raise EmbedderDeadError(
            f"Ollama embed failed after retries: {last_exc}"
        ) from last_exc
