#!/usr/bin/env python3
"""One-time setup of GitHub branch-protection rulesets for sweir1/apple-notes-brain.

Applies two rulesets:

  - `apple-notes-brain/main` — hard rules nobody can bypass:
      • no force-push
      • no deletion
      • PR required to merge (linear history not enforced; merges allowed)

  - `apple-notes-brain/main-workflow` — workflow rules that admin can bypass
    for emergency release commits:
      • required status check: CI workflow on the head SHA must be green

  - `apple-notes-brain/dev` — light protection:
      • no deletion

Requires `gh` CLI installed and authenticated as a repo admin. Idempotent —
re-running with the same rulesets is a no-op.

Usage:
    python scripts/setup_branch_protection.py
    python scripts/setup_branch_protection.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

REPO = "sweir1/apple-notes-brain"

RULESETS = [
    {
        "name": "apple-notes-brain/main",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                    "required_review_thread_resolution": False,
                },
            },
        ],
        "bypass_actors": [],
    },
    {
        "name": "apple-notes-brain/main-workflow",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": False,
                    "required_status_checks": [
                        {"context": "validate"},
                    ],
                },
            },
        ],
        # 5 == OrganizationAdmin / RepositoryAdmin; lets admin bypass for
        # emergency release commits (matches obsidian-brain pattern).
        "bypass_actors": [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
    },
    {
        "name": "apple-notes-brain/dev",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/dev"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
        ],
        "bypass_actors": [],
    },
]


def _existing_rulesets() -> list[dict]:
    out = subprocess.check_output(
        ["gh", "api", f"/repos/{REPO}/rulesets"],
        text=True,
    )
    return json.loads(out)


def _apply(payload: dict, dry_run: bool) -> None:
    existing = _existing_rulesets()
    match = next((r for r in existing if r["name"] == payload["name"]), None)
    body = json.dumps(payload)
    if match:
        url = f"/repos/{REPO}/rulesets/{match['id']}"
        method = "PUT"
    else:
        url = f"/repos/{REPO}/rulesets"
        method = "POST"
    if dry_run:
        print(f"would {method} {url}\n  body: {body}")
        return
    subprocess.check_call(
        ["gh", "api", "--method", method, "-H", "Accept: application/vnd.github+json", url, "--input", "-"],
        input=body.encode("utf-8"),
    )
    print(f"✓ {payload['name']} ({method})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        subprocess.check_output(["gh", "auth", "status"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"error: gh not authenticated or not installed: {exc}", file=sys.stderr)
        return 1
    for r in RULESETS:
        _apply(r, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
