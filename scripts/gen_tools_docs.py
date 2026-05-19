#!/usr/bin/env python3
"""Regenerate docs/tools.md's tool tables from src/apple_notes_brain/server.py.

Approach: static AST parse (not runtime import — that would bring up the
whole MCP server, the cache prewarm, the AppleScript ping, etc.).

A "tool" is any module-level `def` whose decorator list contains a call to
`_mcp_tool(annotations=...)`. The decorator's `annotations=` argument tells
us the tool's kind (READ_ONLY / WRITE / DESTRUCTIVE). The function's
signature gives us the param list; the first paragraph of its docstring
gives us the description.

Tools are partitioned into two tables:
  - Lexical CRUD: everything except the four semantic ones
  - Semantic: `semantic_search`, `hybrid_search`, `reindex_semantic`,
    `semantic_index_status` — needs the `[semantic]` install extra

Usage:
    python scripts/gen_tools_docs.py
    python scripts/gen_tools_docs.py --check
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _markers import apply_or_check  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SERVER_PY = REPO / "src" / "apple_notes_brain" / "server.py"
TOOLS_DOC = REPO / "docs" / "tools.md"

SEMANTIC_TOOLS = {
    "semantic_search",
    "hybrid_search",
    "reindex_semantic",
    "semantic_index_status",
}

KIND_LABEL = {
    "READ_ONLY": "read",
    "WRITE": "write",
    "DESTRUCTIVE": "destructive",
}


def _format_arg(arg: ast.arg, default: ast.expr | None) -> str:
    name = arg.arg
    return f"{name}?" if default is not None else name


def _format_signature(func: ast.FunctionDef) -> str:
    args = func.args
    parts: list[str] = []
    pos_only = list(args.posonlyargs)
    positional = list(args.args)

    # Map defaults to the trailing positional args (Python semantics).
    all_positional = pos_only + positional
    defaults = list(args.defaults)
    n_defaults = len(defaults)
    default_offset = len(all_positional) - n_defaults
    for i, a in enumerate(all_positional):
        d = defaults[i - default_offset] if i >= default_offset else None
        parts.append(_format_arg(a, d))

    # Keyword-only — every kw-only with a non-None default is optional.
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_format_arg(a, d))

    return f"`{func.name}({', '.join(parts)})`"


def _annotation_kind(deco: ast.expr) -> str | None:
    """Return READ_ONLY / WRITE / DESTRUCTIVE for `@_mcp_tool(annotations=X)`."""
    if not isinstance(deco, ast.Call):
        return None
    fn = deco.func
    if isinstance(fn, ast.Name) and fn.id == "_mcp_tool":
        for kw in deco.keywords:
            if kw.arg == "annotations" and isinstance(kw.value, ast.Name):
                return kw.value.id
    return None


def _first_paragraph(docstring: str | None) -> str:
    if not docstring:
        return ""
    para = docstring.strip().split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())


def collect_tools(source: str) -> list[tuple[str, str, str, str]]:
    """Return list of (name, kind_label, signature, description)."""
    tree = ast.parse(source)
    out: list[tuple[str, str, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        kind = None
        for deco in node.decorator_list:
            kind = _annotation_kind(deco)
            if kind:
                break
        if kind is None:
            continue
        label = KIND_LABEL.get(kind, kind.lower())
        sig = _format_signature(node)
        desc = _first_paragraph(ast.get_docstring(node)) or "_(no docstring)_"
        # Escape any pipe chars in the description so they don't break the table.
        desc = desc.replace("|", "\\|")
        out.append((node.name, label, sig, desc))
    return out


def render(tools: list[tuple[str, str, str, str]]) -> str:
    lexical = [t for t in tools if t[0] not in SEMANTIC_TOOLS]
    semantic = [t for t in tools if t[0] in SEMANTIC_TOOLS]

    def fmt_table(rows: list[tuple[str, str, str, str]]) -> str:
        lines = ["| Tool | Kind | What it does |", "|---|---|---|"]
        for _name, kind, sig, desc in rows:
            lines.append(f"| {sig} | {kind} | {desc} |")
        return "\n".join(lines)

    parts = ["## Lexical CRUD", "", fmt_table(lexical), ""]
    if semantic:
        parts += [
            "## Semantic (requires `[semantic]` extra)",
            "",
            fmt_table(semantic),
        ]
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="exit 1 on drift; do not write")
    args = p.parse_args()

    source = SERVER_PY.read_text(encoding="utf-8")
    tools = collect_tools(source)
    block = render(tools)
    return apply_or_check(TOOLS_DOC, "tools", block, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
