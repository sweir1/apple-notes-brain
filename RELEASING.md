# Releasing apple-notes-brain

Release flow is `dev → PR → main → tag`. `main` is never red; tag must be on `main` for release.yml to publish. PyPI + MCP Registry both publish via GitHub OIDC — no API tokens stored anywhere.

## TL;DR

```bash
# 1. Land your changes on dev (small PRs from feature branches)
gh pr create --base dev
# CI green → merge

# 2. Promote dev → main
git checkout dev && git pull
gh pr create --head dev --base main
# CI green twice (the PR's CI, and the post-merge CI) → merge

# 3. Cut the release (explicit, signed-off, one tag at a time)
git checkout main && git pull
python scripts/sync_version.py X.Y.Z
# verify the diff looks right
git diff
git commit -am "chore: vX.Y.Z"
git tag vX.Y.Z
git push --follow-tags
```

That's it. Pushing the `vX.Y.Z` tag triggers `release.yml`, which:

1. **verify** — tag must be on main; pyproject.toml + server.json versions must match the tag; `check_env_vars.py` must pass.
2. **test** — full pytest suite on macos-latest with semantic extras + ONNX model prefetched.
3. **build** — `uv build` → exactly 1 wheel + 1 sdist.
4. **publish** — to PyPI via OIDC trusted publishing (no API token). Gated by the `pypi` GitHub Environment so you get an explicit approval click in the GitHub UI.
5. **publish-mcp-registry** — to <https://registry.modelcontextprotocol.io/> via `mcp-publisher login github-oidc`. PyPI verification: the registry checks the published README for `<!-- mcp-name: io.github.sweir1/apple-notes-brain -->`. Don't remove that line from `README.md`.

## Pre-release checklist

Before tagging:

- [ ] `python scripts/preflight.py` exits 0 locally
- [ ] `docs/CHANGELOG.md` has a fresh `## vX.Y.Z — YYYY-MM-DD — Title` entry (em dash, not hyphen)
- [ ] `python scripts/gen_readme_recent.py` ran cleanly (README "Recent releases" block reflects the new entry)
- [ ] `python scripts/sync_version.py X.Y.Z` ran cleanly (pyproject.toml + both server.json version fields match)
- [ ] You've actually decided to ship — explicit sign-off, not "the plan says version X.Y.Z"

## Branch protection

One-time setup (admin only):

```bash
python scripts/setup_branch_protection.py
```

Applies three rulesets:

- **`apple-notes-brain/main`** — no force-push, no deletion, PR required (hard rules, no bypass).
- **`apple-notes-brain/main-workflow`** — required status check on the CI workflow (admin can bypass for emergency release commits).
- **`apple-notes-brain/dev`** — deletion blocked.

## What's automated, what's manual

| Step | Automated? |
|------|------------|
| pyproject.toml + server.json version sync | `scripts/sync_version.py` (manual invocation) |
| README "Recent releases" block | `scripts/gen_readme_recent.py` (manual invocation; `--check` in CI) |
| `docs/configuration.md` env-var table | `scripts/gen_docs.py` (manual invocation; `--check` in CI) |
| `docs/tools.md` tool table | `scripts/gen_tools_docs.py` (manual invocation; `--check` in CI) |
| Git tag | Manual (`git tag vX.Y.Z`) |
| PyPI publish | Automatic on tag push (release.yml job `publish`) |
| MCP Registry publish | Automatic on tag push (release.yml job `publish-mcp-registry`) |
| GitHub Release | Auto-created (GitHub Environment "pypi" approval gate before PyPI step) |
| server.json `environmentVariables[]` | **Manual.** Hand-edit when adding env vars to source. `check_env_vars.py` enforces no drift. |

## Rollback

If a release publishes a broken artifact:

1. **Yank from PyPI** — `pypi.org/manage/project/apple-notes-brain/release/X.Y.Z/` → yank.
2. **Delete the MCP Registry entry** — the registry doesn't support deletion in preview; bump the version and republish a fixed build.
3. **Revert on main** — `git revert <bad-commit>` on dev, PR to main, merge, tag a fresh `vX.Y.Z+1`. Don't `git reset --hard` main (it's protected and would break linear history).

## Why this is the flow

- **`dev → PR → main → tag`** keeps main always green. Branch protection (see `scripts/setup_branch_protection.py`) enforces this on the server side.
- **No force-pushes** to shared branches. Add follow-up commits or `git revert` + new commit.
- **No auto-ship.** Every release needs an explicit "ship it", even if a plan lists multiple versions to cut.
- **`server.json` env vars are hand-edited.** `check_env_vars.py` enforces no drift between source and the MCP Registry contract.
