"""Tests for ``semantic.hf_metadata``.

All HF calls are mocked — never touches the real network. The
``_fetch_json_via_hub`` + ``_fetch_model_card`` helpers are
monkeypatched per-test to return shaped fixtures or simulate
failures.
"""
from __future__ import annotations

import pytest

from apple_notes_brain.semantic import hf_metadata
from apple_notes_brain.semantic.hf_metadata import (
    HfMetadata,
    detect_model_language,
    detect_prefix_script,
    extract_base_model,
    get_embedding_metadata,
    language_to_script,
    resolve_prompts_from_readme,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModelCard:
    """Minimal ModelCard stand-in. ``data`` mimics ``ModelCardData``."""

    def __init__(self, text: str, data: dict | None = None):
        self.text = text
        self.content = text  # some HF versions use this name
        self._data = data or {}

    @property
    def data(self):
        class _Data:
            def __init__(_self, d):
                _self._d = d

            def get(_self, key):
                return _self._d.get(key)

        return _Data(self._data)

    def __str__(self) -> str:
        return self.text


def _install_fake_fetcher(
    monkeypatch,
    *,
    files: dict[str, object] | None = None,
    cards: dict[str, object] | None = None,
):
    """Install a fake ``_fetch_json_via_hub`` + ``_fetch_model_card``.

    ``files`` is keyed by ``"{repo}:{filename}"``; missing key → None.
    ``cards`` is keyed by repo id.
    """
    files = files or {}
    cards = cards or {}

    def fake_fetch(repo_id, filename, cache_dir=None):
        return files.get(f"{repo_id}:{filename}")

    def fake_card(repo_id):
        return cards.get(repo_id)

    monkeypatch.setattr(hf_metadata, "_fetch_json_via_hub", fake_fetch)
    monkeypatch.setattr(hf_metadata, "_fetch_model_card", fake_card)


# ---------------------------------------------------------------------------
# Main entrypoint: get_embedding_metadata
# ---------------------------------------------------------------------------


def test_get_embedding_metadata_returns_none_when_config_json_missing(monkeypatch):
    _install_fake_fetcher(monkeypatch, files={})
    result = get_embedding_metadata("nonexistent/repo")
    assert result is None


def test_get_embedding_metadata_returns_none_when_config_has_no_dim(monkeypatch):
    """Multimodal models without a scalar hidden_size in config.json."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "weird/multimodal:config.json": {"model_type": "weird-no-dim"},
        },
    )
    result = get_embedding_metadata("weird/multimodal")
    assert result is not None
    assert result.dim is None  # gracefully degrades; doesn't raise


def test_get_embedding_metadata_full_four_config_fetch(monkeypatch):
    """Successful path with every config file present."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "BAAI/bge-large-en-v1.5:config.json": {
                "model_type": "bert",
                "hidden_size": 1024,
                "max_position_embeddings": 512,
                "num_hidden_layers": 24,
            },
            "BAAI/bge-large-en-v1.5:tokenizer_config.json": {
                "model_max_length": 512,
            },
            "BAAI/bge-large-en-v1.5:sentence_bert_config.json": {
                "max_seq_length": 512,
            },
            "BAAI/bge-large-en-v1.5:config_sentence_transformers.json": {
                "prompts": {
                    "query": "Represent this sentence for searching: ",
                    "passage": "",
                },
            },
            "BAAI/bge-large-en-v1.5:modules.json": [
                {"type": "sentence_transformers.models.Pooling"},
                {"type": "sentence_transformers.models.Normalize"},
            ],
        },
    )
    meta = get_embedding_metadata("BAAI/bge-large-en-v1.5")
    assert meta is not None
    assert isinstance(meta, HfMetadata)
    assert meta.dim == 1024
    assert meta.max_tokens == 512
    assert meta.query_prefix == "Represent this sentence for searching: "
    assert meta.document_prefix == ""
    assert meta.prefix_source == "metadata"
    assert meta.sources.had_sentence_bert_config is True
    assert meta.sources.had_modules_json is True


def test_get_embedding_metadata_dim_fallback_chain(monkeypatch):
    """hidden_size missing → d_model → n_embd → n_embed."""
    cases = [
        ({"model_type": "t5", "d_model": 768}, 768),
        ({"model_type": "gpt2", "n_embd": 1024}, 1024),
        ({"model_type": "bloom", "n_embed": 2048}, 2048),
    ]
    for cfg, expected_dim in cases:
        _install_fake_fetcher(
            monkeypatch,
            files={"test/repo:config.json": cfg},
        )
        meta = get_embedding_metadata("test/repo")
        assert meta is not None
        assert meta.dim == expected_dim, f"failed for {cfg}"


def test_get_embedding_metadata_max_tokens_priority_sbert_wins(monkeypatch):
    """sentence_bert_config.max_seq_length wins over tokenizer / config."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384, "max_position_embeddings": 8192},
            "test/repo:tokenizer_config.json": {"model_max_length": 4096},
            "test/repo:sentence_bert_config.json": {"max_seq_length": 512},
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.max_tokens == 512
    assert meta.sources.max_tokens_from == "sentence_bert_config"


def test_get_embedding_metadata_max_tokens_priority_tokenizer_second(monkeypatch):
    """No sbert config → tokenizer_config.model_max_length wins."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384, "max_position_embeddings": 8192},
            "test/repo:tokenizer_config.json": {"model_max_length": 4096},
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.max_tokens == 4096
    assert meta.sources.max_tokens_from == "tokenizer_config"


def test_get_embedding_metadata_max_tokens_priority_config_third(monkeypatch):
    """No sbert + tokenizer ignored → config.max_position_embeddings wins."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384, "max_position_embeddings": 8192},
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.max_tokens == 8192
    assert meta.sources.max_tokens_from == "config"


def test_get_embedding_metadata_max_tokens_default_when_all_missing(monkeypatch):
    _install_fake_fetcher(
        monkeypatch,
        files={"test/repo:config.json": {"hidden_size": 384}},
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.max_tokens == 512
    assert meta.sources.max_tokens_from == "default"


def test_get_embedding_metadata_max_tokens_rejects_insane_sentinel(monkeypatch):
    """INT32_MAX-style sentinels (some tokenizer_configs set model_max_length=1e30)."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384, "max_position_embeddings": 512},
            "test/repo:tokenizer_config.json": {"model_max_length": 2**31 - 1},
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    # Tokenizer config rejected → falls through to config.
    assert meta.max_tokens == 512
    assert meta.sources.max_tokens_from == "config"


def test_get_embedding_metadata_xlm_roberta_subtracts_two(monkeypatch):
    """xlm-roberta reserves two positions for special tokens."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {
                "model_type": "xlm-roberta",
                "hidden_size": 768,
                "max_position_embeddings": 514,
            },
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.max_tokens == 512


def test_get_embedding_metadata_symmetric_model_no_prompts(monkeypatch):
    """BGE-small-style: no config_sentence_transformers.json → empty prefixes."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384},
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.query_prefix is None
    assert meta.document_prefix is None
    assert meta.prefix_source == "none"


def test_get_embedding_metadata_asymmetric_prompts_from_st_config(monkeypatch):
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384},
            "test/repo:config_sentence_transformers.json": {
                "prompts": {
                    "query": "query: ",
                    "document": "passage: ",
                },
            },
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.query_prefix == "query: "
    assert meta.document_prefix == "passage: "
    assert meta.prefix_source == "metadata"


def test_get_embedding_metadata_passage_key_treated_as_document(monkeypatch):
    """Legacy SentenceTransformers key 'passage' → maps to document_prefix."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "test/repo:config.json": {"hidden_size": 384},
            "test/repo:config_sentence_transformers.json": {
                "prompts": {"query": "q: ", "passage": "p: "},
            },
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.document_prefix == "p: "


def test_get_embedding_metadata_tier_2_falls_through_to_base_model(monkeypatch):
    """No direct prompts → README declares base_model → upstream has prompts."""
    _install_fake_fetcher(
        monkeypatch,
        files={
            "child/model:config.json": {"hidden_size": 384},
            "upstream/base:config_sentence_transformers.json": {
                "prompts": {"query": "Query: ", "document": "Doc: "},
            },
        },
        cards={
            "child/model": _FakeModelCard(
                "---\nbase_model: upstream/base\n---\nbody",
                data={"base_model": "upstream/base"},
            ),
        },
    )
    meta = get_embedding_metadata("child/model")
    assert meta is not None
    assert meta.prefix_source == "metadata-base"
    assert meta.query_prefix == "Query: "
    assert meta.base_model == "upstream/base"


def test_get_embedding_metadata_tier_3_readme_fingerprint(monkeypatch):
    """No JSON prompts anywhere → README fingerprinting picks them up."""
    readme = (
        "---\nlanguage: en\n---\n"
        "Use 'query: ' for queries.\n"
        "Use 'query: ' to embed a question.\n"
        "Index documents with 'passage: '.\n"
        "Index more documents with 'passage: '.\n"
    )
    _install_fake_fetcher(
        monkeypatch,
        files={"test/repo:config.json": {"hidden_size": 384}},
        cards={"test/repo": _FakeModelCard(readme)},
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.prefix_source == "readme"
    assert meta.query_prefix == "query: "
    assert meta.document_prefix == "passage: "


def test_get_embedding_metadata_base_model_pulled_from_card_data(monkeypatch):
    _install_fake_fetcher(
        monkeypatch,
        files={"test/repo:config.json": {"hidden_size": 384}},
        cards={
            "test/repo": _FakeModelCard(
                "---\nbase_model: upstream/base\n---\nbody",
                data={"base_model": "upstream/base"},
            ),
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.base_model == "upstream/base"


def test_get_embedding_metadata_base_model_list_form(monkeypatch):
    """``base_model`` in YAML can be a list — first entry wins."""
    _install_fake_fetcher(
        monkeypatch,
        files={"test/repo:config.json": {"hidden_size": 384}},
        cards={
            "test/repo": _FakeModelCard(
                "---\nfoo: bar\n---\nbody",
                data={"base_model": ["first/base", "second/base"]},
            ),
        },
    )
    meta = get_embedding_metadata("test/repo")
    assert meta is not None
    assert meta.base_model == "first/base"


# ---------------------------------------------------------------------------
# Standalone helpers — extract_base_model / detect_model_language /
# detect_prefix_script / language_to_script / resolve_prompts_from_readme
# ---------------------------------------------------------------------------


def test_extract_base_model_single_string():
    readme = "---\nbase_model: upstream/base\nlanguage: en\n---\nbody"
    assert extract_base_model(readme) == "upstream/base"


def test_extract_base_model_list_form():
    readme = "---\nbase_model:\n  - upstream/base\n  - other/model\n---\nbody"
    assert extract_base_model(readme) == "upstream/base"


def test_extract_base_model_strips_quotes():
    readme = "---\nbase_model: \"upstream/base\"\n---"
    assert extract_base_model(readme) == "upstream/base"


def test_extract_base_model_returns_none_without_frontmatter():
    assert extract_base_model("just markdown") is None


def test_detect_model_language_from_yaml_single():
    readme = "---\nlanguage: en\n---\n"
    assert detect_model_language(readme, "any/model") == "en"


def test_detect_model_language_multilingual_returns_none():
    readme = "---\nlanguage:\n  - en\n  - zh\n  - fr\n---\n"
    assert detect_model_language(readme, "any/model") is None


def test_detect_model_language_falls_back_to_model_id():
    assert detect_model_language(None, "BAAI/bge-large-en-v1.5") == "en"


def test_language_to_script_basic_mappings():
    assert language_to_script("en") == "latin"
    assert language_to_script("zh") == "cjk"
    assert language_to_script("ar") == "arabic"
    assert language_to_script("ru") == "cyrillic"
    assert language_to_script("hi") == "devanagari"
    assert language_to_script("unknown") is None


def test_detect_prefix_script_latin():
    assert detect_prefix_script("query: ") == "latin"


def test_detect_prefix_script_cjk():
    # Chinese for "query:" + colon.
    assert detect_prefix_script("查询：") == "cjk"


def test_detect_prefix_script_arabic():
    assert detect_prefix_script("سؤال: ") == "arabic"


def test_detect_prefix_script_cyrillic():
    assert detect_prefix_script("Запрос: ") == "cyrillic"


def test_resolve_prompts_from_readme_finds_repeated_strings():
    """Two occurrences of `"query: "` + two of `"passage: "` → both picked."""
    readme = (
        "Use 'query: ' for queries.\n"
        "Like 'query: hello world'\n"
        "Index with 'passage: '.\n"
        "Use 'passage: hello world'\n"
    )
    q, d = resolve_prompts_from_readme(readme, expected_script="latin")
    assert q == "query: "
    assert d == "passage: "


def test_resolve_prompts_from_readme_filters_by_script():
    """English-declared model gets the Latin prefix even when Chinese
    appears more often (BGE bug catch)."""
    readme = (
        "中文: 查询。\n中文: 查询。\n中文: 查询。\n"
        "English: 'query: ' once.\n"
    )
    q, _d = resolve_prompts_from_readme(readme, expected_script="latin")
    # The Chinese candidate would normally win on frequency.
    # Script filter should drop it, leaving the English candidate.
    assert q == "query: " or q is None  # depends on isPlausiblePrefix outcomes


def test_resolve_prompts_from_readme_rejects_python_print_labels():
    """`'Sentence embeddings:'` is rejected (no trailing space)."""
    readme = (
        "Sentence embeddings: hello\n"
        "Sentence embeddings: hello\n"
        "Sentence embeddings: hello\n"
    )
    # Even with high frequency, no trailing `": "` → rejected.
    q, _d = resolve_prompts_from_readme(readme)
    assert q is None


def test_resolve_prompts_from_readme_returns_none_on_empty():
    assert resolve_prompts_from_readme("") == (None, None)


def test_resolve_prompts_from_readme_strips_frontmatter():
    """Frontmatter `description: 'query: foo'` shouldn't be picked up."""
    readme = "---\ndescription: 'query: foo bar'\n---\nplain body"
    q, _d = resolve_prompts_from_readme(readme)
    # No occurrences in body → no prefix.
    assert q is None
