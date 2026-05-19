# Quick start

Requires **macOS**, **Python 3.11+**, and **Apple Notes**.

## One-line installer (recommended)

Handles the [`uv`](https://docs.astral.sh/uv/) install, the `/usr/local/bin` PATH symlinks Claude Desktop needs, the Claude Desktop config merge (preserves your other MCP servers), the Full Disk Access walkthrough for **both** Claude.app **and** the uv-managed Python (the cached-Python TCC quirk that breaks SQLite reads — see [Troubleshooting](troubleshooting.md)), and the Automation-permission heads-up.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sweir1/apple-notes-brain/main/scripts/install.sh)"
```

Idempotent — safe to re-run if you change your mind, hit a TCC mismatch, or want to update.

If you'd rather drive the install yourself, scroll down.

## Manual install

=== "`uvx` (no install)"

    ```bash
    uvx apple-notes-brain
    ```

    Ephemeral environment. Best for trying it out or for client configs that should always pull the latest published version.

=== "`uv tool install`"

    ```bash
    uv tool install apple-notes-brain
    ```

    Installs the `apple-notes-brain` command globally (managed by uv). Update later with `uv tool upgrade apple-notes-brain`.

=== "`pip`"

    ```bash
    pip install apple-notes-brain
    ```

=== "from source"

    ```bash
    git clone https://github.com/sweir1/apple-notes-brain.git
    cd apple-notes-brain
    uv sync                 # or: python3.11 -m venv .venv && .venv/bin/pip install -e .
    ```

## Wire up Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "apple-notes-brain": {
      "command": "uvx",
      "args": ["apple-notes-brain"]
    }
  }
}
```

Or, if you `uv tool install`-ed it:

```json
{
  "mcpServers": {
    "apple-notes-brain": {
      "command": "apple-notes-brain"
    }
  }
}
```

Restart Claude Desktop. macOS will prompt for **Full Disk Access** (to read `NoteStore.sqlite`) and **Automation permission** (to drive Notes.app via AppleScript). Grant both. Full Disk Access is granted in System Settings → Privacy & Security → Full Disk Access.

For other MCP clients, see [Install in your MCP client](install-clients.md).

## Optional: semantic + hybrid search

The four semantic tools (`semantic_search`, `hybrid_search`, `reindex_semantic`, `semantic_index_status`) are gated behind the `[semantic]` install extra so the lexical CRUD tools stay zero-extra-dependency.

```bash
pip install 'apple-notes-brain[semantic]'
# or, with uv
uv add 'apple-notes-brain[semantic]'
```

After the first call, the embedder downloads `BAAI/bge-small-en-v1.5` (ONNX-quantised, ~30MB) into `~/.local/share/apple-notes-brain/models/`. On Apple Silicon the CoreML execution provider lights up automatically — verify via `semantic_index_status()`.

See [How embeddings work](embeddings.md) and [Models](models.md) for presets and Ollama.
