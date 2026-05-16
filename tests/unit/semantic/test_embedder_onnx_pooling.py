"""Golden-vector pooling tests for the ONNX embedder.

These verify that `_run_session_pooled` does the right math: mean-pool
over the attention mask, then L2 normalise. Failures here propagate
straight into bad retrieval, so they get their own file.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apple_notes_brain.semantic.config import load_config
from apple_notes_brain.semantic.embedder.onnx import OnnxEmbedder


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_NOTES_BRAIN_DATA_DIR", str(tmp_path))
    return load_config()


def _build_embedder_with_fixed_outputs(
    cfg, token_emb: np.ndarray, attention_mask: list[int]
) -> OnnxEmbedder:
    """Construct an OnnxEmbedder with everything mocked so the next
    embed() call returns the pooled+normalised version of `token_emb`."""
    # Tokenizer returns an Encoding whose ids/mask we control.
    fake_tokenizer = MagicMock()
    fake_tokenizer.enable_truncation = MagicMock()
    fake_tokenizer.encode.return_value = MagicMock(
        ids=[101] * len(attention_mask),
        attention_mask=attention_mask,
    )
    tokenizer_cls = MagicMock()
    tokenizer_cls.from_file.return_value = fake_tokenizer

    # Session returns the prepared token-embedding tensor (1, seq_len, dim).
    fake_session = MagicMock()
    fake_session.get_providers.return_value = ["CPUExecutionProvider"]
    inp_a = MagicMock()
    inp_a.name = "input_ids"
    inp_b = MagicMock()
    inp_b.name = "attention_mask"
    fake_session.get_inputs.return_value = [inp_a, inp_b]
    fake_session.run.return_value = [token_emb[None, :, :]]  # add batch dim

    with patch("huggingface_hub.hf_hub_download", return_value="/tmp/fake"):
        with patch("tokenizers.Tokenizer", tokenizer_cls):
            import onnxruntime

            with patch.object(
                onnxruntime, "InferenceSession", return_value=fake_session
            ):
                emb = OnnxEmbedder(config=cfg)
                emb.init()
                return emb


def test_pooling_all_ones_mask_averages_evenly(cfg):
    """Hand-built tensor with mask [1,1,1] — pooled vector is the mean."""
    token_emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    emb = _build_embedder_with_fixed_outputs(
        cfg, token_emb=token_emb, attention_mask=[1, 1, 1]
    )
    v = emb.embed("dummy")
    expected = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(v, expected, atol=1e-6)


def test_pooling_padding_token_excluded_from_average(cfg):
    """Tokens with mask=0 are NOT counted in the mean — that's the whole
    point of attention-masked mean-pooling.

    With mask [1, 1, 0], averaging tokens 0+1 should produce
    (0.5, 0.5, 0) before normalisation, NOT (0.33, 0.33, 0.33).
    """
    token_emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],  # would dominate if not masked
        ],
        dtype=np.float32,
    )
    emb = _build_embedder_with_fixed_outputs(
        cfg, token_emb=token_emb, attention_mask=[1, 1, 0]
    )
    v = emb.embed("dummy")
    expected = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(v, expected, atol=1e-6)


def test_pooling_all_zeros_mask_returns_zero_vector(cfg):
    """Defensive: a fully-padded input (all-zero mask) must not produce
    NaN/Inf. The store layer would happily accept those and search
    results would mysteriously become 'NaN ranked first' until someone
    spent an afternoon debugging it. So we pin to zero-vector."""
    token_emb = np.zeros((3, 4), dtype=np.float32)
    emb = _build_embedder_with_fixed_outputs(
        cfg, token_emb=token_emb, attention_mask=[0, 0, 0]
    )
    v = emb.embed("dummy")
    # Pooled vector is zero (sum/count=0 hits the epsilon floor; then
    # norm < 1e-12 short-circuits the divide and we return zeros).
    np.testing.assert_array_equal(v, np.zeros(4, dtype=np.float32))


def test_pooling_l2_normalises(cfg):
    """The final output should have unit norm for any non-zero input."""
    token_emb = np.array(
        [
            [3.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    emb = _build_embedder_with_fixed_outputs(
        cfg, token_emb=token_emb, attention_mask=[1, 1]
    )
    v = emb.embed("dummy")
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-6)


def test_pooling_dim_matches_hidden_size(cfg):
    """Output dim equals the hidden_dim of the token-embedding tensor."""
    rng = np.random.default_rng(42)
    token_emb = rng.standard_normal((5, 7)).astype(np.float32)
    emb = _build_embedder_with_fixed_outputs(
        cfg, token_emb=token_emb, attention_mask=[1] * 5
    )
    v = emb.embed("dummy")
    assert v.shape == (7,)
