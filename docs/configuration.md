# Configuration

Every knob is a process-level environment variable. Set it in your MCP client's config under `env:`, e.g. for Claude Desktop:

```json
{
  "mcpServers": {
    "apple-notes-brain": {
      "command": "uvx",
      "args": ["apple-notes-brain"],
      "env": {
        "EMBEDDING_PRESET": "english-quality",
        "NOTES_MCP_REFRESH_INTERVAL": "10"
      }
    }
  }
}
```

Restart the client to pick up changes.

The table below is generated from [`server.json`](https://github.com/sweir1/apple-notes-brain/blob/main/server.json) — the source of truth for the MCP Registry. To add or remove a variable, edit `server.json`, then run `python scripts/gen_docs.py`.

<!-- GENERATED:env-vars — DO NOT EDIT. Update server.json, then run `python scripts/gen_docs.py`. -->
| Variable | Required | Default | Description |
|---|---|---|---|
| `NOTES_MCP_AUTO_REFRESH` | optional | `1` | Set to '0' to disable the background auto-refresh thread that pings Notes.app to keep the SQLite cache fresh while the MCP client is active. Default '1' (enabled). |
| `NOTES_MCP_REFRESH_INTERVAL` | optional | `4` | Background auto-refresh cadence in seconds (default 4, minimum 1). Lower = fresher reads of edits made outside the MCP, higher CPU. Has no effect on writes (those auto-flush regardless). |
| `NOTES_MCP_IDLE_THRESHOLD` | optional | `300` | Seconds of MCP-tool inactivity before the background auto-refresh pauses (default 300 = 5 minutes). Set to 0 to disable idle pausing entirely. Next tool call instantly wakes the refresher. |
| `APPLE_NOTES_BRAIN_DATA_DIR` | optional | _(unset)_ | Directory for the semantic-index SQLite DB and embedding model cache. Defaults to $XDG_DATA_HOME/apple-notes-brain or ~/.local/share/apple-notes-brain. Used only by the semantic+hybrid search tools (v1.1); has no effect without the [semantic] install extra. |
| `APPLE_NOTES_BRAIN_MODEL_CACHE` | optional | _(unset)_ | Directory for embedding model downloads. Defaults to <data_dir>/models. Maps onto huggingface_hub's cache layout so an existing HF cache can be reused. |
| `APPLE_NOTES_BRAIN_NO_WATCH` | optional | `0` | Set to '1' to disable the background semantic-index watcher that polls PRAGMA data_version. Default '0' (watcher enabled when [semantic] is installed). |
| `APPLE_NOTES_BRAIN_INDEX_INTERVAL` | optional | `30` | Seconds between semantic-index watcher ticks (default 30, minimum 1). Ticks are cheap when nothing changed in NoteStore.sqlite — only a single data_version probe. |
| `APPLE_NOTES_BRAIN_MAX_CHUNK_TOKENS` | optional | _(unset)_ | Override the chunker's per-chunk token budget (computed at 2.5 chars/token). Defaults to the embedder's discovered/advertised max. Lower it if the embedder OOMs on long chunks. |
| `APPLE_NOTES_BRAIN_DEBUG` | optional | `0` | Set to '1' for verbose semantic-subsystem logging on stderr. Default '0'. |
| `EMBEDDING_PROVIDER` | optional | `onnx` | Which embedder to use. 'onnx' (default, in-process ONNX with CoreML execution provider on macOS Apple Silicon) or 'ollama' (HTTP, expects an Ollama server at OLLAMA_BASE_URL). When paired with EMBEDDING_PRESET and the two disagree on provider, EMBEDDING_PROVIDER wins and a one-shot warning is emitted on stderr. |
| `EMBEDDING_PRESET` | optional | `english` | Named embedding preset (mirrors obsidian-brain). 'english' (default, bge-small-en-v1.5, 384d), 'english-fast' (mdbr-leaf-ir, 384d), 'english-quality' (bge-base-en-v1.5, 768d), 'multilingual' (e5-small, 384d), 'multilingual-quality' (e5-base, 768d), or 'multilingual-ollama' (qwen3-embedding:0.6b via Ollama, 1024d). Sets provider + model atomically. EMBEDDING_MODEL overrides this when both are set. |
| `EMBEDDING_MODEL` | optional | _(unset)_ | Literal HuggingFace repo (e.g. 'Xenova/bge-small-en-v1.5') or Ollama model identifier. Overrides EMBEDDING_PRESET when both are set. Switching models forces a full re-embed on next indexing pass. Legacy short-names (bge-small-en-v1.5, bge-base-en-v1.5, all-MiniLM-L6-v2) still resolve to their canonical preset with a deprecation warning. |
| `EMBEDDING_DIM` | optional | _(unset)_ | Override the embedder's output dimensionality. Auto-probed at init; only set this for non-standard models that don't auto-probe correctly. |
| `EMBEDDING_ONNX_PROVIDERS` | optional | _(unset)_ | Comma-separated list of onnxruntime execution providers to try in order. Default on macOS: 'CoreMLExecutionProvider,CPUExecutionProvider'. Default elsewhere: 'CPUExecutionProvider'. Set 'CPUExecutionProvider' alone to force CPU on Apple Silicon (e.g. if CoreML EP misbehaves for a particular model). |
| `OLLAMA_BASE_URL` | optional | `http://localhost:11434` | Base URL of the local Ollama server (only used when EMBEDDING_PROVIDER=ollama). Default http://localhost:11434. Trailing slash is stripped. |
| `OLLAMA_NUM_CTX` | optional | _(unset)_ | Override the context-window probe for the Ollama embedder. When unset, /api/show is queried; if num_ctx is unavailable, falls back to 8192. |
| `APPLE_NOTES_BRAIN_OLLAMA_AUTO_PULL` | optional | `1` | When EMBEDDING_PROVIDER=ollama and the configured model isn't present locally, '1' (default) auto-pulls it via /api/pull at boot; '0' refuses to start until the model exists. |
| `APPLE_NOTES_BRAIN_NO_PREWARM` | optional | `0` | Set to '1' to skip the AppleScript pre-warm ping at server startup. Pre-warm normally triggers the Automation permission prompt before any user-invoked tool call and wakes Notes.app so it flushes in-memory state. Useful in headless / CI contexts where there's no user to grant the prompt. |
| `APPLE_NOTES_BRAIN_NO_CATCHUP` | optional | `0` | Set to '1' to skip the catch-up semantic reindex on startup. Catch-up is normally run once on boot to bring the semantic index in sync with NoteStore.sqlite after a server restart. Has no effect without the [semantic] install extra. |
| `APPLE_NOTES_BRAIN_CONFIG_DIR` | optional | _(unset)_ | Directory for user-config overrides (model-overrides.json, etc.). Defaults to $XDG_CONFIG_HOME/apple-notes-brain or ~/.config/apple-notes-brain. Read at startup to layer over bundled / fetched model metadata. |
<!-- /GENERATED:env-vars -->

## Picking values

- **`NOTES_MCP_REFRESH_INTERVAL`** controls how stale your "outside the MCP" view can get. `4` (default) catches edits made directly in Notes.app within ~4 seconds. Raise to `10`+ if you don't need that and want a quieter background. See [Architecture → cache coherence](architecture.md#cache-coherence) for the cost model.
- **`EMBEDDING_PRESET`** controls semantic-search quality and disk footprint. See [Models](models.md).
- **`APPLE_NOTES_BRAIN_DATA_DIR`** is where the SQLite-vec index and ONNX models cache. Default lives under `~/.local/share` and survives across reinstalls.

## What's not configurable on purpose

- **Read source** is always macOS's `NoteStore.sqlite`. There is no remote-API alternative.
- **Write path** is always AppleScript. Direct SQLite writes would desync iCloud — deliberately out of scope.
- **Locked-note decryption.** apple-notes-brain never decrypts.
