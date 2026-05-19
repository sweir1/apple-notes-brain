#!/usr/bin/env python3
"""Pre-release validation orchestrator. Mirrors what CI runs.

Sequence:
    1. version sync   (sync_version.py --check)
    2. env-var drift  (check_env_vars.py)
    3. config drift   (gen_docs.py --check)
    4. tools drift    (gen_tools_docs.py --check)
    5. README drift   (gen_readme_recent.py --check)
    6. docs build     (mkdocs build --strict, if --with-docs)
    7. tests          (pytest, if --with-tests)
    8. smoke          (MCP smoke subset, if --with-tests)
    9. package build  (uv build, if --with-build)

Exit codes:
    0 — every step passed
    1 — at least one step failed (which one is logged to stderr)

Usage:
    python scripts/preflight.py                       # generators + version + env vars (fast)
    python scripts/preflight.py --with-docs           # also mkdocs build --strict
    python scripts/preflight.py --with-tests          # also pytest + smoke
    python scripts/preflight.py --with-build          # also uv build
    python scripts/preflight.py --all                 # everything (matches CI)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
WEBSITE = REPO / "website"


def _run(label: str, cmd: list[str], cwd: Path = REPO) -> bool:
    print(f"\n=== {label} ===")
    try:
        subprocess.check_call(cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"✗ {label} failed (exit {exc.returncode})", file=sys.stderr)
        return False
    except FileNotFoundError as exc:
        print(f"✗ {label} cannot start: {exc}", file=sys.stderr)
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--with-docs", action="store_true")
    p.add_argument("--with-tests", action="store_true")
    p.add_argument("--with-build", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all:
        args.with_docs = args.with_tests = args.with_build = True

    py = sys.executable
    steps: list[tuple[str, list[str], Path]] = [
        ("version sync (--check)", [py, str(SCRIPTS / "sync_version.py"), "--check"], REPO),
        ("env-var drift", [py, str(SCRIPTS / "check_env_vars.py")], REPO),
        ("docs/configuration.md drift", [py, str(SCRIPTS / "gen_docs.py"), "--check"], REPO),
        ("docs/tools.md drift", [py, str(SCRIPTS / "gen_tools_docs.py"), "--check"], REPO),
        ("README recent-releases drift", [py, str(SCRIPTS / "gen_readme_recent.py"), "--check"], REPO),
    ]

    if args.with_docs:
        steps.append(("mkdocs build --strict", ["mkdocs", "build", "--strict"], WEBSITE))

    if args.with_tests:
        steps.append(("pytest", ["uv", "run", "pytest", "--no-cov", "-q"], REPO))
        steps.append((
            "MCP smoke",
            ["uv", "run", "pytest", "--no-cov", "-m", "integration", "tests/integration/test_mcp_smoke.py"],
            REPO,
        ))

    if args.with_build:
        steps.append(("uv build", ["uv", "build"], REPO))

    failures: list[str] = []
    for label, cmd, cwd in steps:
        if not _run(label, cmd, cwd=cwd):
            failures.append(label)

    print()
    if failures:
        print(f"✗ preflight failed: {len(failures)} step(s) failed", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("✓ preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
