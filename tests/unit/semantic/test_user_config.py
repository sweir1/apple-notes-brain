"""Tests for `apple_notes_brain.semantic.user_config`.

Covers the three-way precedence (explicit env > XDG > default ~/.config),
side-effect-on-first-access (dir is created), tilde expansion, whitespace
handling, and the seed/overrides path derivations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apple_notes_brain.semantic.user_config import (
    ENV_CONFIG_DIR,
    get_overrides_path,
    get_user_config_dir,
    get_user_seed_path,
)


# ---------------------------------------------------------------------------
# get_user_config_dir — precedence ladder
# ---------------------------------------------------------------------------


def test_explicit_env_takes_precedence(monkeypatch, tmp_path):
    """`APPLE_NOTES_BRAIN_CONFIG_DIR` beats everything else, used verbatim."""
    custom = tmp_path / "my-custom-config"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(custom))
    # Make sure XDG and HOME would otherwise be picked — confirms precedence.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "should-not-be-used"))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    path = get_user_config_dir()
    assert path == custom
    assert path.exists()
    assert path.is_dir()


def test_xdg_used_when_no_explicit(monkeypatch, tmp_path):
    """When the explicit env is unset, `$XDG_CONFIG_HOME/apple-notes-brain` wins."""
    monkeypatch.delenv(ENV_CONFIG_DIR, raising=False)
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    path = get_user_config_dir()
    assert path == xdg / "apple-notes-brain"
    assert path.exists()


def test_default_falls_back_to_dot_config(monkeypatch, tmp_path):
    """No env at all → `~/.config/apple-notes-brain` (macOS default path)."""
    monkeypatch.delenv(ENV_CONFIG_DIR, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # Patch Path.home() so we don't touch the real home directory.
    monkeypatch.setattr(
        "pathlib.Path.home", staticmethod(lambda: tmp_path / "fake-home")
    )

    path = get_user_config_dir()
    assert path == tmp_path / "fake-home" / ".config" / "apple-notes-brain"
    assert path.exists()
    assert path.is_dir()


def test_first_access_creates_parent_dir(monkeypatch, tmp_path):
    """The function creates the dir on first access, including parents."""
    nested = tmp_path / "a" / "b" / "c" / "config"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(nested))
    assert not nested.exists()  # precondition

    path = get_user_config_dir()
    assert path.exists()
    assert path.is_dir()


def test_repeated_calls_are_idempotent(monkeypatch, tmp_path):
    """Calling twice doesn't raise (mkdir exist_ok=True)."""
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "cfg"))
    p1 = get_user_config_dir()
    p2 = get_user_config_dir()
    assert p1 == p2
    assert p1.exists()


def test_empty_env_var_treated_as_unset(monkeypatch, tmp_path):
    """`APPLE_NOTES_BRAIN_CONFIG_DIR=""` is treated as not set — falls through."""
    monkeypatch.setenv(ENV_CONFIG_DIR, "")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(
        "pathlib.Path.home", staticmethod(lambda: tmp_path / "fake-home")
    )

    path = get_user_config_dir()
    assert path == tmp_path / "fake-home" / ".config" / "apple-notes-brain"


def test_whitespace_only_env_var_treated_as_unset(monkeypatch, tmp_path):
    """`APPLE_NOTES_BRAIN_CONFIG_DIR="   "` falls through to XDG / default."""
    monkeypatch.setenv(ENV_CONFIG_DIR, "   ")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(
        "pathlib.Path.home", staticmethod(lambda: tmp_path / "fake-home")
    )

    path = get_user_config_dir()
    assert path == tmp_path / "fake-home" / ".config" / "apple-notes-brain"


def test_explicit_env_strips_whitespace(monkeypatch, tmp_path):
    """`APPLE_NOTES_BRAIN_CONFIG_DIR="  /tmp/x  "` is stripped."""
    custom = tmp_path / "cfg"
    monkeypatch.setenv(ENV_CONFIG_DIR, f"  {custom}  ")
    path = get_user_config_dir()
    assert path == custom
    assert path.exists()


def test_empty_xdg_falls_back_to_dot_config(monkeypatch, tmp_path):
    """`XDG_CONFIG_HOME=""` is treated as unset — fall through to ~/.config."""
    monkeypatch.delenv(ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setattr(
        "pathlib.Path.home", staticmethod(lambda: tmp_path / "fake-home")
    )

    path = get_user_config_dir()
    assert path == tmp_path / "fake-home" / ".config" / "apple-notes-brain"


def test_whitespace_xdg_falls_back(monkeypatch, tmp_path):
    """`XDG_CONFIG_HOME="   "` is treated as unset."""
    monkeypatch.delenv(ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "   ")
    monkeypatch.setattr(
        "pathlib.Path.home", staticmethod(lambda: tmp_path / "fake-home")
    )

    path = get_user_config_dir()
    assert path == tmp_path / "fake-home" / ".config" / "apple-notes-brain"


def test_explicit_env_expands_user(monkeypatch, tmp_path):
    """`APPLE_NOTES_BRAIN_CONFIG_DIR=~/foo` resolves tildes against HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV_CONFIG_DIR, "~/my-cfg")
    path = get_user_config_dir()
    assert path == tmp_path / "my-cfg"
    assert path.exists()


def test_xdg_expands_user(monkeypatch, tmp_path):
    """`$XDG_CONFIG_HOME=~/xdg` resolves tildes too."""
    monkeypatch.delenv(ENV_CONFIG_DIR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", "~/xdg")
    path = get_user_config_dir()
    assert path == tmp_path / "xdg" / "apple-notes-brain"
    assert path.exists()


def test_returns_pathlib_path(monkeypatch, tmp_path):
    """Return value is a pathlib.Path (not a string)."""
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "cfg"))
    assert isinstance(get_user_config_dir(), Path)


# ---------------------------------------------------------------------------
# get_user_seed_path / get_overrides_path
# ---------------------------------------------------------------------------


def test_seed_path_under_config_dir(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(cfg))
    assert get_user_seed_path() == cfg / "seed-models.json"


def test_overrides_path_under_config_dir(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(cfg))
    assert get_overrides_path() == cfg / "model-overrides.json"


def test_seed_and_overrides_paths_dont_create_file(monkeypatch, tmp_path):
    """Path accessors don't create the files themselves — only the dir."""
    cfg = tmp_path / "cfg"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(cfg))

    seed = get_user_seed_path()
    overrides = get_overrides_path()

    # Dir exists (created by get_user_config_dir during path lookup).
    assert cfg.exists()
    # Files do NOT exist.
    assert not seed.exists()
    assert not overrides.exists()


def test_seed_and_overrides_share_resolved_dir(monkeypatch, tmp_path):
    """Both helpers route through the same resolution chain."""
    monkeypatch.setenv(ENV_CONFIG_DIR, str(tmp_path / "shared"))
    assert get_user_seed_path().parent == get_overrides_path().parent
    assert get_user_seed_path().parent == get_user_config_dir()
