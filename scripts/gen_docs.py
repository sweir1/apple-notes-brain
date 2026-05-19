#!/usr/bin/env python3
"""Regenerate docs/configuration.md's env-vars table from server.json.

server.json's `packages[0].environmentVariables[]` array is the source of
truth (the MCP Registry reads it directly). This script renders it as a
markdown table inside the `env-vars` marker in docs/configuration.md.

Usage:
    python scripts/gen_docs.py             # regenerate the file in-place
    python scripts/gen_docs.py --check     # CI mode: exit 1 on drift, no write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _markers import apply_or_check  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SERVER_JSON = REPO / "server.json"
CONFIG_DOC = REPO / "docs" / "configuration.md"


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def render_table(server_json: dict) -> str:
    env_vars = server_json["packages"][0]["environmentVariables"]
    rows = ["| Variable | Required | Default | Description |", "|---|---|---|---|"]
    for ev in env_vars:
        name = f"`{ev['name']}`"
        required = "required" if ev.get("isRequired") else "optional"
        default = ev.get("default")
        default_md = f"`{default}`" if default not in (None, "") else "_(unset)_"
        desc = _md_escape((ev.get("description") or "").strip().rstrip("."))
        if desc and not desc.endswith("."):
            desc = desc + "."
        rows.append(f"| {name} | {required} | {default_md} | {desc} |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="exit 1 on drift; do not write")
    args = p.parse_args()

    server = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    table = render_table(server)
    return apply_or_check(CONFIG_DOC, "env-vars", table, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
