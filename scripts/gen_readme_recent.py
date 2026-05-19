#!/usr/bin/env python3
"""Regenerate the Recent releases block in README.md from docs/CHANGELOG.md.

Mirrors obsidian-brain's gen-readme-recent.mjs:
  - parses `## vX.Y.Z — YYYY-MM-DD — Title` headers (em dash required)
  - filters to versions that have a `vX.Y.Z` git tag OR equal the current
    pyproject.toml version (so unreleased-but-in-flight versions appear)
  - renders the top N as markdown bullets inside the `recent-releases`
    marker in README.md
  - also keeps docs/roadmap.md's `recently-shipped` marker pointing at the
    changelog (no per-entry expansion there; a single bullet that links out
    is enough)

Usage:
    python scripts/gen_readme_recent.py
    python scripts/gen_readme_recent.py --check
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _markers import apply_or_check  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "docs" / "CHANGELOG.md"
README = REPO / "README.md"
ROADMAP = REPO / "docs" / "roadmap.md"
PYPROJECT = REPO / "pyproject.toml"

DEFAULT_N = 5

HEADER_RE = re.compile(
    r"^## v(?P<ver>\d+\.\d+\.\d+)"
    r"(?: — (?P<date>\d{4}-\d{2}-\d{2}))?"
    r"(?: — (?P<title>.+))?$",
    re.MULTILINE,
)


def _git_tags() -> set[str]:
    try:
        out = subprocess.check_output(
            ["git", "tag", "--list", "v*"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _current_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def parse_entries(text: str) -> list[dict[str, str | None]]:
    return [
        {
            "ver": m.group("ver"),
            "date": m.group("date"),
            "title": m.group("title").strip() if m.group("title") else None,
        }
        for m in HEADER_RE.finditer(text)
    ]


def filter_entries(
    entries: list[dict[str, str | None]],
    tags: set[str],
    current_version: str,
) -> list[dict[str, str | None]]:
    """Keep only entries with a matching `vX.Y.Z` tag or = current version."""
    out = []
    for e in entries:
        tag = f"v{e['ver']}"
        if tag in tags or e["ver"] == current_version:
            out.append(e)
    return out


def render_bullets(entries: list[dict[str, str | None]], n: int) -> str:
    bullets = []
    for e in entries[:n]:
        parts = [f"**v{e['ver']}**"]
        if e["date"]:
            parts.append(f"({e['date']})")
        if e["title"]:
            parts.append(f"— {e['title']}")
        bullets.append("- " + " ".join(parts))
    return "\n".join(bullets) if bullets else "- (no releases yet)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="exit 1 on drift; do not write")
    p.add_argument("--n", type=int, default=DEFAULT_N, help=f"how many to render (default {DEFAULT_N})")
    args = p.parse_args()

    text = CHANGELOG.read_text(encoding="utf-8")
    all_entries = parse_entries(text)
    tags = _git_tags()
    current = _current_version()
    filtered = filter_entries(all_entries, tags, current)
    bullets = render_bullets(filtered, args.n)

    rc = apply_or_check(README, "recent-releases", bullets, args.check)

    # Roadmap "recently shipped" — single static line pointing at the
    # changelog, so the roadmap doesn't fork from CHANGELOG over time.
    roadmap_block = "- See [Changelog](CHANGELOG.md) for the full history."
    rc2 = apply_or_check(ROADMAP, "recently-shipped", roadmap_block, args.check)

    return rc or rc2


if __name__ == "__main__":
    raise SystemExit(main())
