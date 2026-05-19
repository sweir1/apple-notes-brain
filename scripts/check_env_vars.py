#!/usr/bin/env python3
"""Verify every env-var read in src/ has a matching entry in server.json.

The MCP Registry treats `server.json.packages[0].environmentVariables[]` as
the canonical list — clients use it to ask for configuration up front. If
the source reads `os.environ.get('NEW_VAR')` but server.json doesn't list
`NEW_VAR`, a user installing through the registry won't know they need to
set it.

This script enforces the contract:
  - Every os.environ.get / os.getenv key string literal in src/ must appear
    in server.json's environmentVariables[] list.
  - Allowlisted exceptions can be put in ALLOWLIST below (e.g. PYTHONPATH).

Usage:
    python scripts/check_env_vars.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "apple_notes_brain"
SERVER_JSON = REPO / "server.json"

# Env vars read in source but NOT user-configurable / NOT part of the
# server's documented surface. Adding to the allowlist is a deliberate
# decision; keep this list short and commented.
ALLOWLIST = {
    # Standard interpreter / OS env, not the server's contract.
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    # CI / test plumbing that's not part of the user-facing surface.
    "CI",
    "GITHUB_ACTIONS",
    "PYTEST_CURRENT_TEST",
}


def _is_os_environ(node: ast.expr) -> bool:
    """True iff `node` is the `os.environ` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_os_getenv(node: ast.expr) -> bool:
    """True iff `node` is the `os.getenv` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "getenv"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _resolve(node: ast.expr | None, aliases: dict[str, str]) -> str | None:
    """Resolve a string constant or a Name pointing to one in the alias table."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def _env_keys_from_call(call: ast.Call, aliases: dict[str, str]) -> list[str]:
    """Return literal env-var names read by os.environ.get / os.getenv / os.environ.pop."""
    fn = call.func

    is_env_method = (
        isinstance(fn, ast.Attribute)
        and fn.attr in ("get", "pop", "setdefault")
        and _is_os_environ(fn.value)
    )
    is_getenv = isinstance(fn, ast.Attribute) and _is_os_getenv(fn)

    if not (is_env_method or is_getenv):
        return []

    if not call.args:
        return []
    resolved = _resolve(call.args[0], aliases)
    return [resolved] if resolved else []


def _env_keys_from_subscript(sub: ast.Subscript, aliases: dict[str, str]) -> list[str]:
    """Return env var name from os.environ['KEY']."""
    if not _is_os_environ(sub.value):
        return []
    resolved = _resolve(sub.slice, aliases)
    return [resolved] if resolved else []


def _module_string_aliases(tree: ast.Module) -> dict[str, str]:
    """Map module-level `NAME = "literal"` assignments → literal value.

    Captures patterns like:
        ENV_PROVIDER = "EMBEDDING_PROVIDER"
        ENV_PRESET: Final[str] = "EMBEDDING_PRESET"
    so a later os.environ.get(ENV_PROVIDER) resolves to "EMBEDDING_PROVIDER".
    """
    aliases: dict[str, str] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                aliases[t.id] = value.value
    return aliases


def collect_source_keys() -> set[str]:
    keys: set[str] = set()
    for py in SRC.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        aliases = _module_string_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                keys.update(_env_keys_from_call(node, aliases))
            elif isinstance(node, ast.Subscript):
                keys.update(_env_keys_from_subscript(node, aliases))
    return keys


def collect_server_keys() -> set[str]:
    data = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    return {ev["name"] for ev in data["packages"][0].get("environmentVariables", [])}


def main() -> int:
    src_keys = collect_source_keys()
    server_keys = collect_server_keys()

    missing = (src_keys - server_keys) - ALLOWLIST
    extra = server_keys - src_keys - ALLOWLIST

    if not missing and not extra:
        print("✓ env-var contract OK")
        return 0

    if missing:
        print("\n✗ env vars read in src/ but NOT listed in server.json:", file=sys.stderr)
        for k in sorted(missing):
            print(f"  - {k}", file=sys.stderr)
        print(
            "\nAdd them to server.json's packages[0].environmentVariables[], "
            "then run `python scripts/gen_docs.py` to refresh docs/configuration.md.",
            file=sys.stderr,
        )

    if extra:
        # Treat extras as a soft warning — they may be docs-only, future,
        # or read in subprocess code we couldn't see statically.
        print("\n⚠ env vars listed in server.json but not detected in src/:", file=sys.stderr)
        for k in sorted(extra):
            print(f"  - {k}", file=sys.stderr)
        print(
            "(soft warning — may be docs-only or read indirectly. "
            "Add to ALLOWLIST in scripts/check_env_vars.py if intentional.)",
            file=sys.stderr,
        )

    # Only `missing` is fatal — extras are informational.
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
