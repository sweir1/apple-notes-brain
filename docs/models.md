# Models

apple-notes-brain ships with six named presets mirroring [obsidian-brain](https://github.com/sweir1/obsidian-brain)'s set so you can swap between the two without learning a second vocabulary. Pick one with `EMBEDDING_PRESET`, or override with a literal `EMBEDDING_MODEL`.

## Preset table

| Preset | Provider | Model | Dim | Notes |
|---|---|---|---|---|
| `english` (default) | onnx | `Xenova/bge-small-en-v1.5` | 384 | Fast English-only; ~30 MB; asymmetric (query prefix). |
| `english-fast` | onnx | `MongoDB/mdbr-leaf-ir` | 384 | Fastest English; Apache-2.0; asymmetric. |
| `english-quality` | onnx | `Xenova/bge-base-en-v1.5` | 768 | Higher-quality English; ~100 MB. |
| `multilingual` | onnx | `Xenova/multilingual-e5-small` | 384 | Multilingual; `query:` / `passage:` prefixes. |
| `multilingual-quality` | onnx | `Xenova/multilingual-e5-base` | 768 | Higher-quality multilingual; known long-input `token_type_ids` quirk in transformers.js. |
| `multilingual-ollama` | ollama | `qwen3-embedding:0.6b` | 1024 | Lossless multilingual via Ollama (32k ctx). Requires a local Ollama server. |

Legacy short-names (`bge-small-en-v1.5`, `bge-base-en-v1.5`, `all-MiniLM-L6-v2`) still resolve to their canonical preset with a one-shot deprecation warning. Switching presets or models forces a full re-embed on the next indexing pass — the dim or tokeniser is different, the old vectors are stale.

## Picking a preset

| You want… | Use |
|-----------|-----|
| Sane default for English notes | `english` |
| The smallest possible install + fastest queries (English) | `english-fast` |
| Higher accuracy on long technical notes (English) | `english-quality` |
| Multilingual support (Spanish, French, German, etc.) | `multilingual` |
| Multilingual + accuracy | `multilingual-quality` |
| Ollama backend (local server, lossless quality, larger model) | `multilingual-ollama` |

Switch the preset by setting it in your client's env block (see [Configuration](configuration.md)) and restarting the client.

## Bundled metadata

`src/apple_notes_brain/data/seed-models.json` bundles canonical model metadata (output dimension, tokeniser, prefixes, license) so a first-launch resolution doesn't need network access. The file is built from the MTEB upstream registry on release and shipped inside the wheel.

If a preset's model is missing from the bundle (you set `EMBEDDING_MODEL` to a model the seed doesn't know about), the metadata resolver falls back to the live HuggingFace API and caches the answer under `<data_dir>/models/metadata-cache.json`.

## Model location on disk

| What | Where |
|---|---|
| ONNX weights + tokenizer.json | `<data_dir>/models/<hf-org>/<hf-repo>/` |
| Metadata cache | `<data_dir>/models/metadata-cache.json` |
| User overrides | `<data_dir>/models/model-overrides.json` |
| Vector index DB | `<data_dir>/index.db` |

`<data_dir>` defaults to `~/.local/share/apple-notes-brain` and can be overridden via `APPLE_NOTES_BRAIN_DATA_DIR`.

## User overrides

`model-overrides.json` is a structured JSON file that lets you override individual fields of a model's metadata without forking the codebase. Useful when an upstream model ships incorrect metadata (wrong reported dimension, missing prefix). The file is read once at startup and merges over the bundled / fetched metadata.

See `src/apple_notes_brain/semantic/user_config.py` for the schema.
