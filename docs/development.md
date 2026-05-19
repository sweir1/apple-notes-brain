# Build from source

```bash
git clone https://github.com/sweir1/apple-notes-brain.git
cd apple-notes-brain
uv sync --extra dev --extra semantic --extra docs
```

That installs the runtime deps plus dev tooling (pytest, hypothesis, syrupy), the semantic extras (sqlite-vec, onnxruntime, tokenizers, huggingface-hub), and the docs site dependencies (MkDocs Material).

## Test suite

The project ships **~900 tests** at roughly 65% line coverage. The default run excludes opt-in live tests (those need Notes.app + iCloud + Automation permission):

```bash
uv run pytest                                  # full suite + coverage
uv run pytest -m integration                   # integration-tier only
uv run pytest -m property                      # hypothesis property tests
uv run pytest -m live                          # OPT-IN: real Notes.app
```

Markers:

| Marker | Meaning |
|---|---|
| `live` | Hits the real Notes.app + iCloud. Slow. Requires Notes.app installed + Automation permission. Skipped by default. |
| `integration` | Multiple modules together, mocked subsystems, no live system. |
| `property` | Hypothesis-based property tests. May take longer than unit tests. |
| `slow` | >1s wall-clock; not in the fast feedback loop. |

Coverage report is generated in `htmlcov/` after each run; open `htmlcov/index.html` to browse.

## Smoke-test the server boots

```bash
# Blocks waiting for MCP traffic. Ctrl-C to exit.
uv run python -m apple_notes_brain < /dev/null
```

Or the MCP smoke tests (subprocess-spawns the server, exercises the JSON-RPC handshake):

```bash
uv run pytest --no-cov -m integration tests/integration/test_mcp_smoke.py
```

## Preflight

`python scripts/preflight.py` runs what CI runs (plus a couple of extras you'd want before tagging a release): pytest, MCP smoke, `uv build`, server.json ↔ pyproject.toml version sync, drift checks for the generated docs (`gen_docs.py --check`, `gen_tools_docs.py --check`, `gen_readme_recent.py --check`), and `check_env_vars.py`.

```bash
python scripts/preflight.py
# → pytest, MCP smoke, uv build, version sync, generator drift checks
```

Green preflight = ready to open a PR / cut a release.

## Inspecting tools interactively

```bash
uv run --with 'mcp[cli]' mcp dev src/apple_notes_brain/server.py
```

Opens the MCP Inspector against a fresh apple-notes-brain process. Click through each tool, see the live request/response.

## Building the docs site locally

```bash
cd website
uv run mkdocs serve --dev-addr 127.0.0.1:8000
```

Then open <http://127.0.0.1:8000>. Edits to `docs/*.md` reload live.

For a strict build (mirrors CI, fails on broken anchors / missing nav entries):

```bash
cd website && uv run mkdocs build --strict
```

## Release process

See [`RELEASING.md`](https://github.com/sweir1/apple-notes-brain/blob/main/RELEASING.md) for the full dev → PR → main → tag flow.

## Where things live

```
.
├── src/apple_notes_brain/   # source code (see Architecture for full breakdown)
├── tests/                   # pytest suite
├── docs/                    # this docs site's markdown source
├── website/                 # MkDocs configuration + macros
├── scripts/                 # automation: gen_*, check_env_vars, preflight, sync_version
└── .github/workflows/       # ci.yml, release.yml, docs.yml
```
