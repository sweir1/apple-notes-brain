# Install in your MCP client

apple-notes-brain speaks stdio-only MCP — every MCP client supports it. The shape is always the same: tell the client to run `uvx apple-notes-brain` (or the installed `apple-notes-brain` binary). The Full Disk Access + Automation permission requirements are macOS-level and apply identically across clients.

## Claude Desktop

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

Restart Claude Desktop. Grant Full Disk Access for both Claude and `/usr/local/bin/uvx` (see [Troubleshooting](troubleshooting.md#full-disk-access) for the why).

## Claude Code

Add to `~/.claude.json` (or in-project `.mcp.json`):

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

Or use the CLI:

```bash
claude mcp add apple-notes-brain --command uvx --args apple-notes-brain
```

## Cursor

Cursor settings → MCP → Add new server:

- **Name:** `apple-notes-brain`
- **Command:** `uvx`
- **Arguments:** `apple-notes-brain`

## Continue

In `~/.continue/config.json`, add to `experimental.modelContextProtocolServers`:

```json
{
  "transport": {
    "type": "stdio",
    "command": "uvx",
    "args": ["apple-notes-brain"]
  }
}
```

## Cline / Roo Code / Zed / other stdio MCP clients

The shape is the same — point the client at `uvx apple-notes-brain`. Refer to the client's MCP docs for the exact config file location.

## Choosing `uvx` vs installed binary

| Form | Command | Pros | Cons |
|------|---------|------|------|
| `uvx` (ephemeral) | `uvx apple-notes-brain` | Always latest, zero install | First-launch download (~1s on warm cache, ~5s cold) |
| `uv tool install`-ed | `apple-notes-brain` | Instant launch, pinned version | Manual upgrades (`uv tool upgrade`) |
| `pip install`-ed | `apple-notes-brain` | Works on systems without uv | Slower than uv-managed |

`uvx` is the default everywhere in these docs because it's the most ergonomic on first install. The Full Disk Access requirement is the same either way — see [Troubleshooting](troubleshooting.md#full-disk-access).
