"""Tests for `apple_notes_brain.semantic.config`.

Covers env-var parsing happy paths AND failure modes (the "test fake
things and things that should cause errors" lane), plus the path
resolution defaults / overrides.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.config import (
    DEFAULT_INDEX_INTERVAL_SECONDS,
    DEFAULT_MODEL_PRESET,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_PROVIDER,
    ENV_DATA_DIR,
    ENV_DEBUG,
    ENV_DIM,
    ENV_INDEX_INTERVAL,
    ENV_MODEL,
    ENV_MODEL_CACHE,
    ENV_NO_WATCH,
    ENV_OLLAMA_AUTO_PULL,
    ENV_OLLAMA_BASE_URL,
    ENV_OLLAMA_NUM_CTX,
    ENV_ONNX_PROVIDERS,
    ENV_PRESET,
    ENV_PROVIDER,
    SemanticConfig,
    _bool_env,
    _float_env,
    _int_env,
    load_config,
    resolve_data_dir,
    resolve_db_path,
    resolve_model_cache,
)
from apple_notes_brain.semantic.embedder.presets import (
    EMBEDDING_PRESETS,
    _reset_warning_state,
)


# ---------------------------------------------------------------------------
# Reset the preset module's one-shot warning trackers between tests so
# deprecation-alias coverage doesn't get suppressed by test ordering.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_preset_warnings():
    _reset_warning_state()
    yield
    _reset_warning_state()


# ---------------------------------------------------------------------------
# _bool_env
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on", " yes "])
def test_bool_env_truthy(monkeypatch, value):
    monkeypatch.setenv("ANB_TEST_FLAG", value)
    assert _bool_env("ANB_TEST_FLAG", default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "False", "no", "off", "  "])
def test_bool_env_falsy(monkeypatch, value):
    monkeypatch.setenv("ANB_TEST_FLAG", value)
    assert _bool_env("ANB_TEST_FLAG", default=True) is False


def test_bool_env_unset_uses_default(monkeypatch):
    monkeypatch.delenv("ANB_TEST_FLAG", raising=False)
    assert _bool_env("ANB_TEST_FLAG", default=True) is True
    assert _bool_env("ANB_TEST_FLAG", default=False) is False


def test_bool_env_rejects_garbage(monkeypatch):
    """Garbage env values surface as a clear ValueError naming the offending
    variable — beats silently picking a default and leaving the user wondering."""
    monkeypatch.setenv("ANB_TEST_FLAG", "maybe")
    with pytest.raises(ValueError, match="ANB_TEST_FLAG"):
        _bool_env("ANB_TEST_FLAG", default=False)


# ---------------------------------------------------------------------------
# _int_env / _float_env
# ---------------------------------------------------------------------------

def test_int_env_happy(monkeypatch):
    monkeypatch.setenv("ANB_N", "42")
    assert _int_env("ANB_N", default=0) == 42


def test_int_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("ANB_N", raising=False)
    assert _int_env("ANB_N", default=99) == 99


def test_int_env_blank_returns_default(monkeypatch):
    monkeypatch.setenv("ANB_N", "  ")
    assert _int_env("ANB_N", default=99) == 99


def test_int_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("ANB_N", "twelve")
    with pytest.raises(ValueError, match="ANB_N"):
        _int_env("ANB_N", default=0)


def test_int_env_enforces_min(monkeypatch):
    monkeypatch.setenv("ANB_N", "0")
    with pytest.raises(ValueError, match="below the minimum"):
        _int_env("ANB_N", default=10, min_value=1)


def test_float_env_happy(monkeypatch):
    monkeypatch.setenv("ANB_F", "3.14")
    assert _float_env("ANB_F", default=0.0) == pytest.approx(3.14)


def test_float_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("ANB_F", "pi")
    with pytest.raises(ValueError, match="ANB_F"):
        _float_env("ANB_F", default=0.0)


def test_float_env_enforces_min(monkeypatch):
    monkeypatch.setenv("ANB_F", "0.5")
    with pytest.raises(ValueError, match="below the minimum"):
        _float_env("ANB_F", default=1.0, min_value=1.0)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_data_dir_uses_explicit_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "custom"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    path = resolve_data_dir()
    assert path == (tmp_path / "custom").resolve()
    assert path.exists() and path.is_dir()


def test_resolve_data_dir_uses_xdg_when_no_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    path = resolve_data_dir()
    assert path == (tmp_path / "xdg" / "apple-notes-brain").resolve()
    assert path.exists()


def test_resolve_data_dir_falls_back_to_dot_local(monkeypatch):
    """When neither env var is set, default is `~/.local/share/apple-notes-brain`.

    We don't actually create that on the user's real machine — we patch
    Path.home() to a tmp dir for the test.
    """
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # Patch Path.home() so we don't touch the real home.
    import tempfile

    with tempfile.TemporaryDirectory() as fake_home:
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: Path(fake_home)))
        path = resolve_data_dir()
        assert path == Path(fake_home) / ".local" / "share" / "apple-notes-brain"
        assert path.exists()


def test_resolve_data_dir_expands_user(monkeypatch, tmp_path):
    """`~/foo` form is expanded correctly."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV_DATA_DIR, "~/my-anb-dir")
    path = resolve_data_dir()
    assert path == (tmp_path / "my-anb-dir").resolve()


def test_resolve_model_cache_default(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.delenv(ENV_MODEL_CACHE, raising=False)
    assert resolve_model_cache() == (tmp_path / "models").resolve()


def test_resolve_model_cache_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_MODEL_CACHE, str(tmp_path / "elsewhere"))
    path = resolve_model_cache()
    assert path == (tmp_path / "elsewhere").resolve()
    assert path.exists()


def test_resolve_db_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    assert resolve_db_path() == (tmp_path / "semantic_index.db").resolve()


# ---------------------------------------------------------------------------
# load_config — assembles the full snapshot
# ---------------------------------------------------------------------------

def test_load_config_all_defaults(monkeypatch, tmp_path):
    """With nothing set (modulo data dir for cleanliness), every field
    has its documented default."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    cfg = load_config()
    assert isinstance(cfg, SemanticConfig)
    assert cfg.provider == DEFAULT_PROVIDER == "onnx"
    # Post-uplift: load_config routes through resolve_preset_config which
    # expands the default preset (`english`) to its concrete HF id.
    assert cfg.model == DEFAULT_MODEL_PRESET == "Xenova/bge-small-en-v1.5"
    assert cfg.preset_short_name == "english"
    assert cfg.dim_override is None
    assert cfg.no_watch is False
    assert cfg.index_interval_s == DEFAULT_INDEX_INTERVAL_SECONDS
    assert cfg.max_chunk_tokens_override is None
    assert cfg.debug is False
    assert cfg.onnx_providers_override is None
    assert cfg.ollama_base_url == DEFAULT_OLLAMA_BASE_URL
    assert cfg.ollama_num_ctx_override is None
    assert cfg.ollama_auto_pull is True


def test_load_config_provider_ollama(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PROVIDER, "ollama")
    cfg = load_config()
    assert cfg.provider == "ollama"


def test_load_config_rejects_unknown_provider(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PROVIDER, "garbage")
    with pytest.raises(ValueError, match=r"EMBEDDING_PROVIDER=.*garbage"):
        load_config()


def test_load_config_onnx_providers_csv(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_ONNX_PROVIDERS, "CoreMLExecutionProvider, CPUExecutionProvider")
    cfg = load_config()
    assert cfg.onnx_providers_override == (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )


def test_load_config_onnx_providers_blank_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_ONNX_PROVIDERS, "  ,  ")
    cfg = load_config()
    assert cfg.onnx_providers_override is None


def test_load_config_dim_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_DIM, "768")
    cfg = load_config()
    assert cfg.dim_override == 768


def test_load_config_dim_zero_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_DIM, "0")
    with pytest.raises(ValueError, match="below the minimum"):
        load_config()


def test_load_config_no_watch(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_NO_WATCH, "1")
    cfg = load_config()
    assert cfg.no_watch is True


def test_load_config_interval_below_min_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_INDEX_INTERVAL, "0.5")
    with pytest.raises(ValueError, match="below the minimum"):
        load_config()


def test_load_config_ollama_base_url_trailing_slash_stripped(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_OLLAMA_BASE_URL, "http://localhost:11434/")
    cfg = load_config()
    assert cfg.ollama_base_url == "http://localhost:11434"


def test_load_config_ollama_auto_pull_disable(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_OLLAMA_AUTO_PULL, "0")
    cfg = load_config()
    assert cfg.ollama_auto_pull is False


def test_load_config_ollama_num_ctx_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_OLLAMA_NUM_CTX, "4096")
    cfg = load_config()
    assert cfg.ollama_num_ctx_override == 4096


def test_load_config_model_override_raw(monkeypatch, tmp_path):
    """Power-user path: EMBEDDING_MODEL=<raw HF repo> is taken verbatim."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_MODEL, "Xenova/bge-base-en-v1.5")
    cfg = load_config()
    assert cfg.model == "Xenova/bge-base-en-v1.5"
    assert cfg.preset_short_name is None


def test_load_config_model_legacy_alias_expanded(monkeypatch, tmp_path, capsys):
    """Legacy short-name preserved with deprecation warning + expansion."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_MODEL, "bge-base-en-v1.5")
    cfg = load_config()
    assert cfg.model == "Xenova/bge-base-en-v1.5"
    assert cfg.preset_short_name == "english-quality"
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()


# ---------------------------------------------------------------------------
# EMBEDDING_PRESET wiring through load_config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(EMBEDDING_PRESETS.keys()))
def test_load_config_preset_env_wires_atomically(monkeypatch, tmp_path, name):
    """`EMBEDDING_PRESET=<name>` populates provider/model/preset_short_name
    atomically — the resolver runs inside load_config and the dataclass
    carries all three."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PRESET, name)
    cfg = load_config()
    preset = EMBEDDING_PRESETS[name]
    assert cfg.provider == preset.provider
    assert cfg.model == preset.model
    assert cfg.preset_short_name == name


def test_load_config_preset_alias_emits_warning(monkeypatch, tmp_path, capsys):
    """A deprecated alias in EMBEDDING_PRESET still surfaces the warning
    through load_config."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PRESET, "fastest")
    cfg = load_config()
    assert cfg.preset_short_name == "english-fast"
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()


def test_load_config_preset_unknown_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PRESET, "not-a-real-preset")
    with pytest.raises(ValueError, match="EMBEDDING_PRESET"):
        load_config()


def test_load_config_preset_provider_mismatch_warning(monkeypatch, tmp_path, capsys):
    """EMBEDDING_PROVIDER=ollama + EMBEDDING_PRESET=english → provider wins,
    preset's model carried, warning emitted."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PRESET, "english")
    monkeypatch.setenv(ENV_PROVIDER, "ollama")
    cfg = load_config()
    assert cfg.provider == "ollama"
    assert cfg.model == "Xenova/bge-small-en-v1.5"
    assert cfg.preset_short_name == "english"
    captured = capsys.readouterr()
    assert "overrides EMBEDDING_PRESET" in captured.err


def test_load_config_provider_alone_no_preset(monkeypatch, tmp_path):
    """`EMBEDDING_PROVIDER=ollama` alone falls into branch (4): provider-default
    model (`nomic-embed-text`), preset_short_name is None."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_PROVIDER, "ollama")
    cfg = load_config()
    assert cfg.provider == "ollama"
    assert cfg.model == "nomic-embed-text"
    assert cfg.preset_short_name is None


def test_load_config_debug_flag(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_DEBUG, "yes")
    cfg = load_config()
    assert cfg.debug is True


def test_load_config_is_re_readable(monkeypatch, tmp_path):
    """Repeated calls to load_config re-read os.environ — important so
    tests can mutate env between calls and see updated values."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    cfg1 = load_config()
    monkeypatch.setenv(ENV_PROVIDER, "ollama")
    cfg2 = load_config()
    assert cfg1.provider == "onnx"
    assert cfg2.provider == "ollama"
