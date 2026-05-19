"""Tests for ``semantic.metadata_resolver``.

The resolver is the orchestrator that ties cache + seed + HF + override
together. These tests inject every dependency (override, HF fetcher,
embedder) so the chain order is exhaustively pinned without touching
the network or filesystem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from apple_notes_brain.semantic import metadata_resolver, seed_loader, store
from apple_notes_brain.semantic.hf_metadata import HfMetadata, HfSources
from apple_notes_brain.semantic.metadata_cache import (
    CachedMetadata,
    load_cached,
    upsert_cache,
)
from apple_notes_brain.semantic.metadata_resolver import (
    ResolvedMetadata,
    resolve_model_metadata,
    resolve_model_metadata_sync,
    promote_from_seed_if_stale,
)
from apple_notes_brain.semantic.overrides import ModelOverride

from .conftest import FakeEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path):
    conn = store.open_db(tmp_path / "resolver.db")
    yield conn
    conn.close()


@pytest.fixture
def onnx_embedder() -> FakeEmbedder:
    """A FakeEmbedder that pretends to be an ONNX embedder.

    Its model_identifier uses the obsidian-brain ONNX format so the
    resolver's _onnx_repo_id helper can extract a bare HF repo from it.
    """
    emb = FakeEmbedder(
        provider="onnx",
        model_id="onnx::Xenova/bge-small-en-v1.5::onnx/model_quantized.onnx",
    )
    emb.init()
    return emb


@pytest.fixture
def fake_emb() -> FakeEmbedder:
    """A FakeEmbedder NOT pretending to be ONNX (HF step skipped)."""
    emb = FakeEmbedder(provider="fake", model_id="fake/non-onnx")
    emb.init()
    return emb


@pytest.fixture(autouse=True)
def _reset_seed_cache():
    seed_loader._reset_seed_cache()
    yield
    seed_loader._reset_seed_cache()


def _empty_seed(monkeypatch):
    """Force the seed loader to return an empty dict."""
    monkeypatch.setattr(seed_loader, "load_seed", lambda: {})


def _seed_with(monkeypatch, entries: dict[str, seed_loader.SeedEntry]):
    monkeypatch.setattr(seed_loader, "load_seed", lambda: entries)


def _hf_returning(meta: HfMetadata | None) -> Callable:
    def _fetch(_repo_id: str):
        return meta
    return _fetch


def _hf_raising(exc: Exception) -> Callable:
    def _fetch(_repo_id: str):
        raise exc
    return _fetch


def _hf_meta(**kw) -> HfMetadata:
    """Build a default-shape HfMetadata for tests."""
    base = dict(
        model_id="Xenova/bge-small-en-v1.5",
        dim=384,
        max_tokens=512,
        query_prefix=None,
        document_prefix=None,
        prefix_source="none",
        base_model=None,
        size_bytes=None,
        sources=HfSources(),
    )
    base.update(kw)
    return HfMetadata(**base)


# ---------------------------------------------------------------------------
# Cache hit short-circuits
# ---------------------------------------------------------------------------


def test_cache_hit_short_circuits_chain(db, onnx_embedder, monkeypatch):
    """Pre-populated cache → resolver returns 'cache' without calling seed/HF."""
    _empty_seed(monkeypatch)
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix="cached_q: ",
            document_prefix="cached_d: ",
            prefix_source="metadata",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )

    def _hf_should_not_be_called(_):
        pytest.fail("HF fetch should not be called on cache hit")

    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_should_not_be_called, override=None
    )
    assert result.resolved_from == "cache"
    assert result.query_prefix == "cached_q: "
    assert result.document_prefix == "cached_d: "
    assert result.override_applied is False


def test_cache_hit_with_partial_override_layers_on_top(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix="cached_q: ",
            document_prefix="cached_d: ",
            prefix_source="seed",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    override = ModelOverride(max_tokens=256)
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=override
    )
    assert result.resolved_from == "cache"
    assert result.max_tokens == 256  # overridden
    assert result.query_prefix == "cached_q: "  # not overridden
    assert result.override_applied is True
    assert result.prefix_source == "override"  # attribution flips when override applied


# ---------------------------------------------------------------------------
# Seed hit (cache miss → seed hit)
# ---------------------------------------------------------------------------


def test_cache_miss_then_seed_hit_writes_cache(db, onnx_embedder, monkeypatch):
    """Cache miss + seed hit → seed becomes cache row + return seed metadata."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix="seed_d: ",
            ),
        },
    )

    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "seed"
    assert result.query_prefix == "seed_q: "
    assert result.document_prefix == "seed_d: "

    # Cache populated.
    cached = load_cached(db, onnx_embedder)
    assert cached is not None
    assert cached.prefix_source == "seed"
    assert cached.query_prefix == "seed_q: "


def test_seed_hit_symmetric_model(db, onnx_embedder, monkeypatch):
    """Seed entry with None prefixes (symmetric model) → empty strings out."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix=None,
                document_prefix=None,
            ),
        },
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "seed"
    assert result.query_prefix == ""
    assert result.document_prefix == ""


# ---------------------------------------------------------------------------
# HF live fetch (cache miss + seed miss → HF)
# ---------------------------------------------------------------------------


def test_cache_miss_seed_miss_hf_hit_writes_cache(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    live = _hf_meta(
        dim=384,
        max_tokens=512,
        query_prefix="hf_q: ",
        document_prefix="hf_d: ",
        prefix_source="metadata",
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(live), override=None
    )
    assert result.resolved_from == "hf"
    assert result.query_prefix == "hf_q: "
    assert result.document_prefix == "hf_d: "
    assert result.prefix_source == "metadata"
    # Cache populated.
    cached = load_cached(db, onnx_embedder)
    assert cached is not None
    assert cached.prefix_source == "metadata"


def test_hf_step_skipped_when_allow_hf_false(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)

    def _hf_must_not_be_called(_):
        pytest.fail("HF must not be called when allow_hf=False")

    result = resolve_model_metadata(
        db,
        onnx_embedder,
        allow_hf=False,
        fetch_hf=_hf_must_not_be_called,
        override=None,
    )
    # Falls through to embedder probe.
    assert result.resolved_from == "embedder-probe"


def test_hf_step_skipped_for_non_onnx_embedder(db, fake_emb, monkeypatch):
    """Ollama / fake embedders skip the HF step."""
    _empty_seed(monkeypatch)

    def _hf_must_not_be_called(_):
        pytest.fail("HF skipped for non-onnx providers")

    result = resolve_model_metadata(
        db, fake_emb, fetch_hf=_hf_must_not_be_called, override=None
    )
    assert result.resolved_from == "embedder-probe"


def test_hf_unreachable_falls_through_to_embedder_probe(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    import httpx

    result = resolve_model_metadata(
        db,
        onnx_embedder,
        fetch_hf=_hf_raising(httpx.ConnectError("network down")),
        override=None,
    )
    assert result.resolved_from == "embedder-probe"
    assert result.dim == 384  # FakeEmbedder default


def test_hf_returns_none_falls_through(db, onnx_embedder, monkeypatch):
    """HF fetcher returning None (model not on HF / 404 on config) → probe."""
    _empty_seed(monkeypatch)
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "embedder-probe"


# ---------------------------------------------------------------------------
# Embedder probe fallback
# ---------------------------------------------------------------------------


def test_embedder_probe_uses_embedder_dimensions(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "embedder-probe"
    assert result.dim == onnx_embedder.dimensions()
    assert result.max_tokens == 512
    assert result.query_prefix == ""
    assert result.document_prefix == ""


def test_embedder_probe_writes_cache(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    cached = load_cached(db, onnx_embedder)
    assert cached is not None
    assert cached.prefix_source == "fallback"


# ---------------------------------------------------------------------------
# Safe defaults
# ---------------------------------------------------------------------------


def test_safe_defaults_when_embedder_probe_raises(db, monkeypatch):
    """Every layer fails → safe defaults; never raises."""
    _empty_seed(monkeypatch)

    class BrokenEmbedder(FakeEmbedder):
        def dimensions(self) -> int:
            raise RuntimeError("probe failed")

        def provider_name(self) -> str:
            return "onnx"

        def model_identifier(self) -> str:
            return "onnx::broken/model::onnx/model.onnx"

    broken = BrokenEmbedder()
    broken.init()
    result = resolve_model_metadata(
        db, broken, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "fallback"
    assert result.dim == 384
    assert result.max_tokens == 512


# ---------------------------------------------------------------------------
# Override short-circuit
# ---------------------------------------------------------------------------


def test_complete_override_short_circuits_before_cache(db, onnx_embedder, monkeypatch):
    """A complete override skips cache / seed / HF entirely."""
    _empty_seed(monkeypatch)

    def _hf_must_not_be_called(_):
        pytest.fail("HF must not be called on complete override")

    override = ModelOverride(
        max_tokens=1024, query_prefix="my_q: ", document_prefix="my_d: "
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_must_not_be_called, override=override
    )
    assert result.resolved_from == "cache"  # synthetic write counts as cache
    assert result.max_tokens == 1024
    assert result.query_prefix == "my_q: "
    assert result.document_prefix == "my_d: "
    assert result.override_applied is True
    assert result.prefix_source == "override"


def test_complete_override_persists_to_cache(db, onnx_embedder, monkeypatch):
    """Persisted override → next boot short-circuits at cache step too."""
    _empty_seed(monkeypatch)
    override = ModelOverride(
        max_tokens=1024, query_prefix="my_q: ", document_prefix="my_d: "
    )
    resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=override
    )
    cached = load_cached(db, onnx_embedder)
    assert cached is not None
    assert cached.prefix_source == "override"
    assert cached.max_tokens == 1024


def test_partial_override_with_only_max_tokens_still_goes_through_chain(
    db, onnx_embedder, monkeypatch
):
    """max_tokens-only override → cache misses, seed hit, override layers on."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix="seed_d: ",
            ),
        },
    )
    override = ModelOverride(max_tokens=2048)
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=override
    )
    assert result.resolved_from == "seed"
    assert result.max_tokens == 2048  # overridden
    assert result.query_prefix == "seed_q: "  # from seed
    assert result.override_applied is True


def test_partial_override_with_empty_string_prefix(db, onnx_embedder, monkeypatch):
    """Override with query_prefix='' (clear to empty) — meaningful override."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix="seed_d: ",
            ),
        },
    )
    override = ModelOverride(query_prefix="")
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=override
    )
    assert result.query_prefix == ""  # cleared by override
    assert result.document_prefix == "seed_d: "  # untouched
    assert result.override_applied is True


# ---------------------------------------------------------------------------
# Sync variant
# ---------------------------------------------------------------------------


def test_sync_variant_returns_none_when_cache_and_seed_miss(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    result = resolve_model_metadata_sync(db, onnx_embedder, override=None)
    assert result is None


def test_sync_variant_hits_cache(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix="cached: ",
            document_prefix="",
            prefix_source="metadata",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata_sync(db, onnx_embedder, override=None)
    assert result is not None
    assert result.resolved_from == "cache"
    assert result.query_prefix == "cached: "


def test_sync_variant_hits_seed(db, onnx_embedder, monkeypatch):
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix=None,
            ),
        },
    )
    result = resolve_model_metadata_sync(db, onnx_embedder, override=None)
    assert result is not None
    assert result.resolved_from == "seed"
    assert result.query_prefix == "seed_q: "


def test_sync_variant_complete_override_short_circuits(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    override = ModelOverride(
        max_tokens=2048, query_prefix="a: ", document_prefix="b: "
    )
    result = resolve_model_metadata_sync(db, onnx_embedder, override=override)
    assert result is not None
    assert result.max_tokens == 2048
    assert result.query_prefix == "a: "
    assert result.override_applied is True


# ---------------------------------------------------------------------------
# Stale-cache promotion (pre-v1.7.5 row → seed)
# ---------------------------------------------------------------------------


def test_promote_from_seed_when_cache_has_null_prefixes(db, onnx_embedder, monkeypatch):
    """Embedder-probe-written row (NULL prefixes) + seed has them → promote."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix="seed_d: ",
            ),
        },
    )
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix=None,
            document_prefix=None,
            prefix_source="fallback",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "seed"
    assert result.query_prefix == "seed_q: "
    assert result.document_prefix == "seed_d: "


def test_no_promotion_when_cache_has_prefixes(db, onnx_embedder, monkeypatch):
    """Cache row has prefixes already → no promotion even if seed differs."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix="seed_q: ",
                document_prefix="seed_d: ",
            ),
        },
    )
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix="cached_q: ",
            document_prefix="cached_d: ",
            prefix_source="metadata",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "cache"
    assert result.query_prefix == "cached_q: "


def test_no_promotion_when_seed_also_null_prefixes(db, onnx_embedder, monkeypatch):
    """Symmetric-in-seed model → no promotion (would write back same NULLs)."""
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512,
                query_prefix=None,
                document_prefix=None,
            ),
        },
    )
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix=None,
            document_prefix=None,
            prefix_source="fallback",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.resolved_from == "cache"


def test_promote_from_seed_if_stale_helper_returns_none_on_missing_cache(
    db, onnx_embedder, monkeypatch
):
    _empty_seed(monkeypatch)
    assert promote_from_seed_if_stale(db, onnx_embedder) is None


def test_promote_from_seed_if_stale_helper_returns_resolved_on_success(
    db, onnx_embedder, monkeypatch
):
    _seed_with(
        monkeypatch,
        {
            "Xenova/bge-small-en-v1.5": seed_loader.SeedEntry(
                max_tokens=512, query_prefix="seed_q: ", document_prefix=None
            ),
        },
    )
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=512,
            query_prefix=None,
            document_prefix=None,
            prefix_source="fallback",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = promote_from_seed_if_stale(db, onnx_embedder)
    assert result is not None
    assert result.query_prefix == "seed_q: "
    assert isinstance(result, ResolvedMetadata)


# ---------------------------------------------------------------------------
# Materialise / dim / base_model / size_bytes pass-through
# ---------------------------------------------------------------------------


def test_dim_pass_through_from_cache(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=768,
            max_tokens=512,
            query_prefix=None,
            document_prefix=None,
            prefix_source="seed",
            base_model="upstream/base",
            size_bytes=12345,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.dim == 768
    assert result.base_model == "upstream/base"
    assert result.size_bytes == 12345


def test_max_tokens_fallback_when_cache_has_null(db, onnx_embedder, monkeypatch):
    _empty_seed(monkeypatch)
    upsert_cache(
        db,
        onnx_embedder,
        CachedMetadata(
            model_id=onnx_embedder.model_identifier(),
            dim=384,
            max_tokens=None,
            query_prefix=None,
            document_prefix=None,
            prefix_source="fallback",
            base_model=None,
            size_bytes=None,
            fetched_at=1700000000,
        ),
    )
    result = resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_hf_returning(None), override=None
    )
    assert result.max_tokens == 512  # FALLBACK_MAX_TOKENS


# ---------------------------------------------------------------------------
# Onnx repo id extraction
# ---------------------------------------------------------------------------


def test_onnx_repo_id_extraction_passes_bare_repo_to_hf(db, onnx_embedder, monkeypatch):
    """The HF fetcher receives the bare repo id, not the full identifier."""
    _empty_seed(monkeypatch)
    received_repo: list[str] = []

    def _record(repo_id):
        received_repo.append(repo_id)
        return None

    resolve_model_metadata(
        db, onnx_embedder, fetch_hf=_record, override=None
    )
    assert received_repo == ["Xenova/bge-small-en-v1.5"]
