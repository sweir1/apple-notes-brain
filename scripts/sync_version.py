#!/usr/bin/env python3
"""Sync the in-repo version into pyproject.toml + server.json.

apple-notes-brain stores its version in three places that MUST stay in
sync:
  - `pyproject.toml`        [project] version = "X.Y.Z"
  - `server.json`           top-level "version": "X.Y.Z"
  - `server.json`           packages[0].version = "X.Y.Z"

This script writes all three from a single argument. It's idempotent —
running it twice with the same version is a no-op.

Usage:
    python scripts/sync_version.py 1.2.0          # writes 1.2.0 to all three
    python scripts/sync_version.py 1.2.0 --check  # exit 1 on drift, no write
    python scripts/sync_version.py --check        # check that all three agree

If `--check` is passed without a version, the script just verifies that all
three current values match (good as a CI guard against accidental drift).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
SERVER_JSON = REPO / "server.json"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
PYPROJECT_LINE_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)


def _read_pyproject_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _read_server_versions() -> tuple[str, str]:
    data = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    return data["version"], data["packages"][0]["version"]


def _write_pyproject_version(new: str) -> bool:
    """Rewrite the `version = "..."` line under [project]. Returns True if changed."""
    text = PYPROJECT.read_text(encoding="utf-8")
    # Only the FIRST top-level version=... in pyproject.toml is the project
    # version (Hatchling allows nested table versions but [project].version
    # is the load-bearing one).
    def _sub(m: re.Match[str]) -> str:
        return f'{m.group(1)}{new}{m.group(3)}'

    new_text, n = PYPROJECT_LINE_RE.subn(_sub, text, count=1)
    if n == 0:
        raise SystemExit("error: could not find `version = \"...\"` in pyproject.toml")
    if new_text == text:
        return False
    PYPROJECT.write_text(new_text, encoding="utf-8")
    return True


def _write_server_versions(new: str) -> bool:
    text = SERVER_JSON.read_text(encoding="utf-8")
    data = json.loads(text)
    changed = False
    if data.get("version") != new:
        data["version"] = new
        changed = True
    if data["packages"][0].get("version") != new:
        data["packages"][0]["version"] = new
        changed = True
    if changed:
        # Preserve trailing newline and the existing 2-space indent style.
        SERVER_JSON.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("version", nargs="?", help="X.Y.Z to write to all three places")
    p.add_argument("--check", action="store_true", help="exit 1 on drift; do not write")
    args = p.parse_args()

    py_v = _read_pyproject_version()
    srv_v, srv_pkg_v = _read_server_versions()

    if args.version:
        if not VERSION_RE.match(args.version):
            raise SystemExit(f"error: not a valid version: {args.version!r}")
        target = args.version
    else:
        # No version given → check-only mode against the current pyproject version.
        if not args.check:
            raise SystemExit(
                "usage: python scripts/sync_version.py <X.Y.Z> [--check]\n"
                "       python scripts/sync_version.py --check"
            )
        target = py_v

    drift = (py_v != target) or (srv_v != target) or (srv_pkg_v != target)
    if args.check:
        if drift:
            print(
                f"✗ version drift detected:\n"
                f"  pyproject.toml         = {py_v}\n"
                f"  server.json (top)      = {srv_v}\n"
                f"  server.json (packages) = {srv_pkg_v}\n"
                f"  target                 = {target}\n"
                f"\nRun `python scripts/sync_version.py {target}` to fix.",
                file=sys.stderr,
            )
            return 1
        print(f"✓ version {target} is consistent")
        return 0

    py_changed = _write_pyproject_version(target)
    srv_changed = _write_server_versions(target)
    if not py_changed and not srv_changed:
        print(f"✓ version already {target} everywhere")
    else:
        print(f"wrote version {target} (pyproject.toml: {py_changed}, server.json: {srv_changed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
