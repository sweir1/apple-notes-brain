"""Resolver chain orchestration for embedder metadata.

Mirrors obsidian-brain's ``src/embeddings/metadata-resolver.ts``. Pure
orchestration — every dependency is injectable so the chain is fully
testable in isolation.

Resolution order (each layer can short-circuit; cache lives forever
until explicit invalidation):

  1. **User override (complete)** — if all three load-bearing fields
     (max_tokens, query_prefix, document_prefix) are present in the
     user's overrides file, we don't need cache / seed / HF for
     anything. Persist the synthetic record and return.
  2. **Cache** — ``embedder_capability`` row with ``fetched_at`` set.
     Stale-cache promotion runs here: pre-resolver rows with both
     prefixes NULL get fixed up from the seed when available.
  3. **Seed** — bundled or user-fetched ``seed-models.json``.
  4. **HF live fetch** — full live read of HF config files +
     README. ONNX only (Ollama tags don't map cleanly to HF repos).
  5. **Embedder probe** — ``embedder.dimensions()`` for dim, 512 +
     empty prefixes for the rest. Always succeeds for an initialised
     embedder.
  6. **Safe defaults** — last-resort 384-dim, 512 max_tokens, empty
     prefixes. Never raises.

After whichever layer matched, a **partial override** (one or two
fields set) is applied on top. Partial overrides DO go through the
full chain so HF / seed / cache can supply the missing fields.

Sync variant (cache + seed only) is used by the bootstrap fast-path,
which can't block on network I/O.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from ._logging import debug_log
from . import seed_loader
from .metadata_cache import (
    CachedMetadata,
    invalidate_cache,
    load_cached,
    upsert_cache,
)
from .overrides import ModelOverride, get_override

if TYPE_CHECKING:  # pragma: no cover - typing only
    import sqlite3
    from .types import Embedder
    from .hf_metadata import HfMetadata


_log = logging.getLogger("apple-notes-brain")

# Safe-default constants. Used as the final fallback when every other
# layer fails (e.g. embedder failed to init AND HF is unreachable).
FALLBACK_MAX_TOKENS = 512
FALLBACK_DIM = 384

ResolvedFrom = Literal["cache", "seed", "hf", "embedder-probe", "fallback"]


# Sentinel for "argument not supplied" — distinguishes from None which is
# a valid value (e.g. an explicit "no override loaded").
_MISSING: object = object()


@dataclass(frozen=True)
class ResolvedMetadata:
    """The materialised result of the resolver chain. Always non-null.

    Extends :class:`CachedMetadata` with two diagnostic fields:
      * ``resolved_from`` — which layer produced this result
      * ``override_applied`` — True iff the user-overrides layer patched
        any field on top
    """

    model_id: str
    dim: int | None
    max_tokens: int
    query_prefix: str
    document_prefix: str
    prefix_source: str
    base_model: str | None
    size_bytes: int | None
    resolved_from: ResolvedFrom
    override_applied: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_model_metadata(
    db: "sqlite3.Connection",
    embedder: "Embedder",
    *,
    allow_hf: bool = True,
    fetch_hf: Callable[[str], "HfMetadata | None"] | None = None,
    override: ModelOverride | None | object = _MISSING,
) -> ResolvedMetadata:
    """Async-but-sync resolver chain. Always returns a ResolvedMetadata.

    The cache writes happen as the chain finds an answer, so subsequent
    boots short-circuit at step 2.

    ``allow_hf=False`` skips the HF live fetch (used by future fast-path
    code that can't block on network). ``fetch_hf`` is the injectable
    HF entry point (tests override; default lazy-imports
    :func:`semantic.hf_metadata.get_embedding_metadata`).

    ``override`` is the injectable override hook (tests pass a sentinel
    to skip the user-overrides file entirely; default reads via
    :func:`semantic.overrides.get_override`).
    """
    model_id = embedder.model_identifier()

    if override is _MISSING:
        override = get_override(model_id)
    assert override is None or isinstance(override, ModelOverride)

    # Step 1: complete-override short-circuit.
    if _is_complete_override(override):
        debug_log(
            "metadata-resolver: complete-override short-circuit", model=model_id
        )
        synthetic = _override_to_cached(model_id, override)
        upsert_cache(db, embedder, synthetic)
        return _materialise(synthetic, "cache", override)

    # Step 2: cache.
    cached = load_cached(db, embedder)
    if cached is not None:
        promoted = _promote_from_seed_if_stale(db, embedder, cached)
        if promoted is not None:
            return _materialise(promoted, "seed", override)
        return _materialise(cached, "cache", override)

    # Step 3: seed.
    seed_entry = seed_loader.lookup(model_id)
    if seed_entry is not None:
        from_seed = _seed_entry_to_cached(model_id, seed_entry)
        upsert_cache(db, embedder, from_seed)
        return _materialise(from_seed, "seed", override)

    # Step 4: HF live fetch (ONNX only — Ollama tags don't map cleanly
    # to HF repos and obsidian-brain handles Ollama via /api/show
    # elsewhere).
    if allow_hf and embedder.provider_name() == "onnx":
        try:
            if fetch_hf is None:
                from .hf_metadata import get_embedding_metadata as _hf

                fetch_hf = _hf
            repo = _onnx_repo_id(model_id)
            live = fetch_hf(repo) if repo else None
            if live is not None:
                from_hf = _hf_metadata_to_cached(model_id, live)
                upsert_cache(db, embedder, from_hf)
                return _materialise(from_hf, "hf", override)
        except Exception as exc:
            _log.warning(
                "metadata-resolver: HF fetch raised %s for %s — falling through",
                exc.__class__.__name__,
                model_id,
            )

    # Step 5: embedder probe.
    try:
        probed = _embedder_probe_to_cached(model_id, embedder)
        upsert_cache(db, embedder, probed)
        return _materialise(probed, "embedder-probe", override)
    except Exception as exc:
        _log.warning(
            "metadata-resolver: embedder probe failed for %s (%s); "
            "using safe defaults",
            model_id,
            exc,
        )

    # Step 6: safe defaults — never raises.
    fallback = _safe_defaults(model_id)
    upsert_cache(db, embedder, fallback)
    return _materialise(fallback, "fallback", override)


def resolve_model_metadata_sync(
    db: "sqlite3.Connection",
    embedder: "Embedder",
    *,
    override: ModelOverride | None | object = _MISSING,
) -> ResolvedMetadata | None:
    """Cache + seed only — never blocks on I/O. Returns ``None`` on miss.

    Used by future bootstrap fast paths that need the prefix-strategy
    hash before the network is reachable. Callers treat ``None`` as
    "skip the optimisation, no reindex triggered".
    """
    model_id = embedder.model_identifier()
    if override is _MISSING:
        override = get_override(model_id)
    assert override is None or isinstance(override, ModelOverride)

    if _is_complete_override(override):
        synthetic = _override_to_cached(model_id, override)
        upsert_cache(db, embedder, synthetic)
        return _materialise(synthetic, "cache", override)

    cached = load_cached(db, embedder)
    if cached is not None:
        promoted = _promote_from_seed_if_stale(db, embedder, cached)
        if promoted is not None:
            return _materialise(promoted, "seed", override)
        return _materialise(cached, "cache", override)

    seed_entry = seed_loader.lookup(model_id)
    if seed_entry is not None:
        from_seed = _seed_entry_to_cached(model_id, seed_entry)
        upsert_cache(db, embedder, from_seed)
        return _materialise(from_seed, "seed", override)

    return None


def promote_from_seed_if_stale(
    db: "sqlite3.Connection", embedder: "Embedder"
) -> ResolvedMetadata | None:
    """Public helper — promote a pre-resolver cache row from the seed.

    Returns the new :class:`ResolvedMetadata` if promotion happened,
    ``None`` if the cache row was either absent, already populated, or
    the model isn't in the seed.
    """
    cached = load_cached(db, embedder)
    if cached is None:
        return None
    promoted = _promote_from_seed_if_stale(db, embedder, cached)
    if promoted is None:
        return None
    override = get_override(embedder.model_identifier())
    return _materialise(promoted, "seed", override)


def refresh_cache(
    db: "sqlite3.Connection", embedder: "Embedder | None" = None
) -> int:
    """Public alias of :func:`metadata_cache.invalidate_cache`.

    Mirrors obsidian-brain's `models refresh-cache` CLI behaviour —
    clears the v7 metadata columns so the next resolver pass re-fills
    from the seed → HF chain.
    """
    return invalidate_cache(db, embedder)


# ---------------------------------------------------------------------------
# Internals — adapters between layer types
# ---------------------------------------------------------------------------


def _is_complete_override(override: ModelOverride | None) -> bool:
    """True iff override fully specifies all three load-bearing fields.

    For prefixes, the obsidian-brain spec treats explicit ``None`` /
    ``""`` as a real override. Since our :class:`ModelOverride` doesn't
    distinguish "field absent" from "explicit None", we treat
    ``query_prefix is None`` and ``document_prefix is None`` as "not
    set" — only an explicit empty string or a non-empty string counts
    as a complete override field. This matches the obsidian-brain
    schema-v1 file shape where omitted fields stay omitted in JSON.
    """
    if override is None:
        return False
    if override.max_tokens is None:
        return False
    if override.query_prefix is None:
        return False
    if override.document_prefix is None:
        return False
    return True


def _override_to_cached(model_id: str, override: ModelOverride) -> CachedMetadata:
    """Project a (complete) override into a CachedMetadata record."""
    assert override.max_tokens is not None
    return CachedMetadata(
        model_id=model_id,
        dim=None,
        max_tokens=override.max_tokens,
        query_prefix=override.query_prefix,
        document_prefix=override.document_prefix,
        prefix_source="override",
        base_model=None,
        size_bytes=None,
        fetched_at=None,  # upsert_cache stamps int(time.time())
    )


def _seed_entry_to_cached(
    model_id: str, entry: "seed_loader.SeedEntry"
) -> CachedMetadata:
    """Project a SeedEntry into a CachedMetadata record."""
    return CachedMetadata(
        model_id=model_id,
        # v2 seed dropped `dim` — runtime probes from the loaded ONNX.
        dim=None,
        max_tokens=entry.max_tokens,
        query_prefix=entry.query_prefix,
        document_prefix=entry.document_prefix,
        prefix_source="seed",
        base_model=None,
        size_bytes=None,
        fetched_at=None,
    )


def _hf_metadata_to_cached(
    model_id: str, live: "HfMetadata"
) -> CachedMetadata:
    """Project an HfMetadata record into a CachedMetadata record.

    Keeps the original embedder model_identifier (so subsequent cache
    reads match) even though the HF fetch used the bare HF repo id.
    """
    return CachedMetadata(
        model_id=model_id,
        dim=live.dim,
        max_tokens=live.max_tokens,
        query_prefix=live.query_prefix,
        document_prefix=live.document_prefix,
        prefix_source=live.prefix_source,
        base_model=live.base_model,
        size_bytes=live.size_bytes,
        fetched_at=None,
    )


def _embedder_probe_to_cached(
    model_id: str, embedder: "Embedder"
) -> CachedMetadata:
    """Build a CachedMetadata from a live, initialised embedder.

    Probe dim from the embedder; max_tokens defaults to 512; prefixes
    empty (symmetric model assumed — when it's not, the user can patch
    via overrides).
    """
    return CachedMetadata(
        model_id=model_id,
        dim=int(embedder.dimensions()),
        max_tokens=FALLBACK_MAX_TOKENS,
        query_prefix=None,
        document_prefix=None,
        prefix_source="fallback",
        base_model=None,
        size_bytes=None,
        fetched_at=None,
    )


def _safe_defaults(model_id: str) -> CachedMetadata:
    """Last-resort metadata — used when even the embedder probe fails."""
    return CachedMetadata(
        model_id=model_id,
        dim=FALLBACK_DIM,
        max_tokens=FALLBACK_MAX_TOKENS,
        query_prefix=None,
        document_prefix=None,
        prefix_source="fallback",
        base_model=None,
        size_bytes=None,
        fetched_at=None,
    )


def _promote_from_seed_if_stale(
    db: "sqlite3.Connection",
    embedder: "Embedder",
    cached: CachedMetadata,
) -> CachedMetadata | None:
    """Promote a pre-resolver row from the bundled seed when appropriate.

    Pre-resolver rows have ``query_prefix IS NULL`` and
    ``document_prefix IS NULL``; if the seed has the model, write the
    seed values over them so asymmetric models start embedding queries
    with the correct prefix on the next pass.
    """
    if cached.query_prefix is not None or cached.document_prefix is not None:
        return None
    seed_entry = seed_loader.lookup(cached.model_id)
    if seed_entry is None:
        return None
    # Only promote if the seed actually has at least one non-None prefix —
    # otherwise we'd write back the same NULLs.
    if seed_entry.query_prefix is None and seed_entry.document_prefix is None:
        return None
    promoted = _seed_entry_to_cached(cached.model_id, seed_entry)
    upsert_cache(db, embedder, promoted)
    debug_log(
        "metadata-resolver: stale-cache promotion",
        model=cached.model_id,
        from_prefix_source=cached.prefix_source,
    )
    return promoted


def _materialise(
    meta: CachedMetadata,
    resolved_from: ResolvedFrom,
    override: ModelOverride | None,
) -> ResolvedMetadata:
    """Apply override (partial merge) and project to ResolvedMetadata.

    The override layer treats ``None`` as "not overridden" (consistent
    with the JSON-omitted-field semantics in
    :class:`overrides.ModelOverride`). Empty-string prefixes ARE
    meaningful overrides (clear the prefix to empty).
    """
    base_max_tokens = meta.max_tokens if meta.max_tokens is not None else FALLBACK_MAX_TOKENS
    base_query = meta.query_prefix if meta.query_prefix is not None else ""
    base_document = meta.document_prefix if meta.document_prefix is not None else ""

    override_applied = False
    eff_max_tokens = base_max_tokens
    eff_query = base_query
    eff_document = base_document
    if override is not None:
        if override.max_tokens is not None:
            eff_max_tokens = override.max_tokens
            override_applied = True
        if override.query_prefix is not None:
            eff_query = override.query_prefix
            override_applied = True
        if override.document_prefix is not None:
            eff_document = override.document_prefix
            override_applied = True

    prefix_source = "override" if override_applied else meta.prefix_source

    return ResolvedMetadata(
        model_id=meta.model_id,
        dim=meta.dim,
        max_tokens=eff_max_tokens,
        query_prefix=eff_query,
        document_prefix=eff_document,
        prefix_source=prefix_source,
        base_model=meta.base_model,
        size_bytes=meta.size_bytes,
        resolved_from=resolved_from,
        override_applied=override_applied,
    )


def _onnx_repo_id(model_id: str) -> str | None:
    """Extract the bare HF repo id from an ONNX embedder model identifier.

    OnnxEmbedder identifiers look like
    ``onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx`` — the
    repo id sits between the first and second ``::`` separators.
    """
    if not model_id.startswith("onnx::"):
        return None
    rest = model_id[len("onnx::"):]
    idx = rest.find("::")
    return rest[:idx] if idx != -1 else rest
