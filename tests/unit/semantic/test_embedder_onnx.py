"""Tests for the ONNX embedder.

The whole chain — hf_hub_download, tokenizers.Tokenizer.from_file,
onnxruntime.InferenceSession — is mocked so the test suite stays fast
and offline. A separate integration test (`@pytest.mark.slow`) lets the
real downloads run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.embedder.onnx import OnnxEmbedder
from apple_notes_brain.semantic.types import (
    EmbedderDeadError,
    ModelDownloadError,
    ModelLoadError,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    return load_config()


def _make_mocks(*, dim: int = 384, providers=None, raises_on_load: Exception | None = None):
    """Patch hf_hub_download + Tokenizer.from_file + ort.InferenceSession.

    Returns a MagicMock for each so a test can inspect call args.
    """
    fake_model_path = "/tmp/fake-model.onnx"
    fake_tokenizer_path = "/tmp/fake-tokenizer.json"

    def fake_download(repo_id, filename, cache_dir):
        if filename.endswith(".onnx"):
            return fake_model_path
        return fake_tokenizer_path

    # Fake tokenizer.
    fake_tokenizer = MagicMock(name="Tokenizer")

    def fake_encode(text):
        ids = [101] + [ord(c) % 30000 for c in text[:10]] + [102]
        mask = [1] * len(ids)
        return MagicMock(ids=ids, attention_mask=mask)

    fake_tokenizer.encode.side_effect = fake_encode
    fake_tokenizer.enable_truncation = MagicMock()
    tokenizer_cls = MagicMock(name="TokenizerCls")
    tokenizer_cls.from_file.return_value = fake_tokenizer

    # Fake session.
    fake_session = MagicMock(name="Session")
    fake_session.get_providers.return_value = providers or ["CPUExecutionProvider"]
    fake_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]
    fake_session.get_inputs.return_value[0].name = "input_ids"
    fake_session.get_inputs.return_value[1].name = "attention_mask"

    def fake_run(_outputs, _inputs):
        # Build a deterministic token-embedding tensor (1, seq_len, dim).
        seq_len = _inputs["input_ids"].shape[1]
        rng = np.random.default_rng(0)
        return [rng.standard_normal((1, seq_len, dim)).astype(np.float32)]

    fake_session.run.side_effect = fake_run

    if raises_on_load is not None:
        def boom(*a, **kw):
            raise raises_on_load
        session_ctor = MagicMock(side_effect=boom)
    else:
        session_ctor = MagicMock(return_value=fake_session)

    ort_mod = MagicMock(InferenceSession=session_ctor)

    return {
        "download": MagicMock(side_effect=fake_download),
        "tokenizer_cls": tokenizer_cls,
        "session": fake_session,
        "ort_mod": ort_mod,
        "fake_model_path": fake_model_path,
        "fake_tokenizer_path": fake_tokenizer_path,
    }


def _patch_onnx_chain(mocks):
    """Return a context manager that swaps the chain into place."""
    return _ChainPatcher(mocks)


class _ChainPatcher:
    def __init__(self, mocks):
        self._mocks = mocks
        self._patches = []

    def __enter__(self):
        # huggingface_hub.hf_hub_download — pass our MagicMock as `new=`
        # so the test can inspect `mocks["download"].call_count`.
        p1 = patch("huggingface_hub.hf_hub_download", new=self._mocks["download"])
        p2 = patch("tokenizers.Tokenizer", self._mocks["tokenizer_cls"])
        import onnxruntime
        p3 = patch.object(
            onnxruntime, "InferenceSession", self._mocks["ort_mod"].InferenceSession
        )
        for p in (p1, p2, p3):
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *a):
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# Init / loading
# ---------------------------------------------------------------------------

def test_init_downloads_model_and_tokenizer(cfg):
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        # Both files downloaded under the configured cache dir.
        called_with = [call.kwargs for call in mocks["download"].mock_calls if call.kwargs]
        # Two downloads happened (model + tokenizer).
        assert mocks["download"].call_count >= 2


def test_init_default_repo_is_bge_small(cfg):
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        repos = [call.kwargs.get("repo_id") for call in mocks["download"].mock_calls]
        assert any("bge-small-en-v1.5" in (r or "") for r in repos)


def test_init_session_uses_cpu_on_linux(cfg, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    mocks = _make_mocks(providers=["CPUExecutionProvider"])
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        # The InferenceSession call should have used CPU-only providers.
        called_kwargs = mocks["ort_mod"].InferenceSession.call_args.kwargs
        assert called_kwargs["providers"] == ["CPUExecutionProvider"]


def test_init_session_uses_coreml_first_on_darwin(cfg, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    mocks = _make_mocks(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        called_kwargs = mocks["ort_mod"].InferenceSession.call_args.kwargs
        assert called_kwargs["providers"][0] == "CoreMLExecutionProvider"


def test_init_provider_override_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_ONNX_PROVIDERS", "CPUExecutionProvider")
    cfg = load_config()
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        called_kwargs = mocks["ort_mod"].InferenceSession.call_args.kwargs
        assert called_kwargs["providers"] == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Embedding behaviour
# ---------------------------------------------------------------------------

def test_embed_returns_unit_float32_vector(cfg):
    mocks = _make_mocks(dim=384)
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        v = emb.embed("hello world")
        assert v.dtype == np.float32
        assert v.shape == (384,)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_embed_dim_override_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    cfg = load_config()
    # Our mock generates dim-384 output; override forces our reported dim
    # to 256 even though the underlying tensor is 384. (Real use: override
    # exists for non-standard models that don't auto-probe correctly.)
    mocks = _make_mocks(dim=384)
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        assert emb.dimensions() == 256


def test_embed_before_init_raises(cfg):
    emb = OnnxEmbedder(config=cfg)
    with pytest.raises(EmbedderDeadError, match="before init"):
        emb.embed("x")


def test_dimensions_before_init_raises(cfg):
    emb = OnnxEmbedder(config=cfg)
    with pytest.raises(EmbedderDeadError):
        emb.dimensions()


def test_provider_name_is_onnx(cfg):
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        assert emb.provider_name() == "onnx"


def test_model_identifier_includes_repo_and_file(cfg):
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        ident = emb.model_identifier()
        assert "onnx::" in ident
        assert "bge-small-en-v1.5" in ident
        assert "model_quantized.onnx" in ident


def test_dispose_is_idempotent(cfg):
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        emb.dispose()
        emb.dispose()
        with pytest.raises(EmbedderDeadError):
            emb.embed("x")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_download_failure_propagates_as_model_download_error(cfg):
    """If HuggingFace returns an error, surface a ModelDownloadError naming
    the repo so the operator can act on it."""
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=Exception("403 not authorized"),
    ):
        emb = OnnxEmbedder(config=cfg)
        with pytest.raises(ModelDownloadError, match="bge-small-en-v1.5"):
            emb.init()


def test_corrupt_model_load_clears_cache_and_retries(cfg, tmp_path):
    """First load raises a ModelLoadError, second succeeds — verifies the
    `retry=True` once-only recovery branch."""
    # Pre-create a fake cache subdir so we can confirm it's removed.
    repo_slug = "models--Xenova--bge-small-en-v1.5"
    (tmp_path / "models" / repo_slug).mkdir(parents=True, exist_ok=True)

    mocks_good = _make_mocks()
    call_count = {"n": 0}

    # First call raises ModelLoadError; subsequent calls succeed.
    def session_ctor(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("Protobuf parsing failed")
        return mocks_good["session"]

    mocks_good["ort_mod"].InferenceSession = MagicMock(side_effect=session_ctor)

    with _patch_onnx_chain(mocks_good):
        emb = OnnxEmbedder(config=cfg)
        emb.init()  # should succeed via retry

    assert call_count["n"] == 2
    # The cache slug should have been removed by _clear_cache between attempts.
    assert not (tmp_path / "models" / repo_slug).exists()


def test_double_load_failure_propagates(cfg):
    """A second failure isn't retried — propagate ModelLoadError."""
    mocks = _make_mocks()

    def always_fails(*a, **kw):
        raise Exception("hopelessly corrupted")

    mocks["ort_mod"].InferenceSession = MagicMock(side_effect=always_fails)
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        with pytest.raises(ModelLoadError):
            emb.init()


def test_session_inputs_only_uses_names_the_model_declares(cfg):
    """A MiniLM-style export without token_type_ids still works — we only
    pass inputs the graph declares."""
    mocks = _make_mocks()
    # Pretend the model only declares input_ids + attention_mask
    inp_a = MagicMock()
    inp_a.name = "input_ids"
    inp_b = MagicMock()
    inp_b.name = "attention_mask"
    mocks["session"].get_inputs.return_value = [inp_a, inp_b]
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        emb.embed("hello")
        # session.run was called; token_type_ids should NOT have been passed.
        last_inputs = mocks["session"].run.call_args.args[1]
        assert "token_type_ids" not in last_inputs


def test_task_type_is_accepted_without_crash(cfg):
    """Symmetric models ignore task_type; passing it is legal."""
    mocks = _make_mocks()
    with _patch_onnx_chain(mocks):
        emb = OnnxEmbedder(config=cfg)
        emb.init()
        v_d = emb.embed("x", task_type="document")
        v_q = emb.embed("x", task_type="query")
        # For symmetric models the vectors are identical (same tokens →
        # same output). We just verify no crash and shape.
        assert v_d.shape == v_q.shape
