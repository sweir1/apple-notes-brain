"""MkDocs macros plugin entry point.

Exposes two things to docs templates:
  - {{ version }}            — current apple-notes-brain version (from pyproject.toml)
  - {{ recent_releases(n) }} — top N changelog entries as a markdown bulleted list

Reads pyproject.toml directly so we don't pull in extra runtime deps; the
docs venv is intentionally minimal (just mkdocs-material + macros + git-date).
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).parent.parent


def _read_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except (OSError, KeyError) as exc:
        print(f"[macros] warning: could not read version from pyproject.toml: {exc}", file=sys.stderr)
        return "unknown"


def define_env(env):
    env.variables["version"] = _read_version()

    @env.macro
    def recent_releases(n: int = 5) -> str:
        """Return a markdown bulleted list of the N most recent CHANGELOG entries."""
        changelog_path = REPO_ROOT / "docs" / "CHANGELOG.md"
        try:
            text = changelog_path.read_text(encoding="utf-8")
        except OSError:
            return "- (changelog unavailable)"

        pattern = re.compile(
            r"^## v(?P<ver>\d+\.\d+\.\d+)"
            r"(?: — (?P<date>\d{4}-\d{2}-\d{2}))?"
            r"(?: — (?P<title>.+))?$",
            re.MULTILINE,
        )

        bullets = []
        for m in pattern.finditer(text):
            ver = m.group("ver")
            date = m.group("date")
            title = m.group("title")

            parts = [f"**v{ver}**"]
            if date:
                parts.append(f"({date})")
            if title:
                parts.append(f"— {title.strip()}")
            bullets.append("- " + " ".join(parts))

            if len(bullets) >= n:
                break

        return "\n".join(bullets) if bullets else "- (no releases found)"
