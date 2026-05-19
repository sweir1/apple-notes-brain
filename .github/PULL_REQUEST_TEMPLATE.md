## Summary

<!-- Describe what this PR changes and why. -->

## Checklist

- [ ] CHANGELOG entry added for any user-visible change (`## vX.Y.Z — YYYY-MM-DD — Title`, em dash)
- [ ] If env vars in source changed, `server.json.packages[0].environmentVariables[]` updated and `python scripts/gen_docs.py` rerun
- [ ] If a `@_mcp_tool` was added/removed/renamed, `python scripts/gen_tools_docs.py` rerun
- [ ] `python scripts/preflight.py` passes locally
- [ ] `python scripts/check_env_vars.py` exits 0
- [ ] `cd website && uv run mkdocs build --strict` passes (only if docs changed)

## Test plan

<!-- How did you verify this change? Include marker-specific runs if relevant (e.g. `-m live` for Notes.app integration). -->
