# macOS walkthrough (non-technical)

This page is for users who don't normally use Terminal. It walks you through the exact same install the [one-line installer](getting-started.md#one-line-installer-recommended) does, but step by step so you know what each prompt means.

You will need:

- A Mac running **macOS 12 (Monterey) or newer**
- The **Notes** app (built in to macOS)
- The **Claude Desktop** app installed and signed in
- About 5 minutes

## 1. Open Terminal

Press **⌘ Space**, type **Terminal**, hit **Return**. A white-on-black window opens. That's where you'll paste one command.

## 2. Paste the installer

Copy this whole line, paste it into Terminal, and press **Return**:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sweir1/apple-notes-brain/main/scripts/install.sh)"
```

The installer talks you through everything else. It will:

1. **Install `uv`** if you don't already have it (Python package manager from Astral).
2. **Install `apple-notes-brain`** itself.
3. **Create symlinks in `/usr/local/bin`** so Claude Desktop can find both `uv` and the server (Claude Desktop has a minimal `PATH` and otherwise wouldn't see them — you'll be prompted for your Mac password here, that's macOS asking permission to write to a system folder).
4. **Edit your Claude Desktop config** to add `apple-notes-brain` (preserves any MCP servers you already have).
5. **Walk you through Full Disk Access** — opens System Settings to the right pane, tells you which two entries to enable.
6. **Tell you what to expect on first use** — the Automation permission dialog.

## 3. Grant Full Disk Access

The installer opens **System Settings → Privacy & Security → Full Disk Access**. You need to enable **two** entries:

1. **Claude** — the Claude Desktop app itself
2. **uvx** — the launcher that runs `apple-notes-brain` (full path `/usr/local/bin/uvx`)

If either is missing from the list, click the **+** button, press **⌘ Shift G**, paste the path the installer shows you, and click **Open** → toggle the entry on.

Why two: Apple Notes' database is in a protected location. macOS only lets a process read it if **the whole process chain** has Full Disk Access. The chain is **Claude → uvx → Python**, so both Claude and uvx need it.

## 4. Restart Claude Desktop

Quit Claude Desktop (⌘ Q) and re-open it. The installer prints the exact command if you'd rather do it from Terminal.

## 5. First use → Automation prompt

The first time you ask Claude to do something that touches Notes (e.g. "what notes did I edit yesterday"), macOS pops up an **Automation** permission prompt: "Claude wants permission to control Notes". Click **OK**.

You only see this once. It's sticky for that copy of Claude Desktop.

That's it — Claude can now read, write, and search your Apple Notes.

## What if something goes wrong

- **"could not open NoteStore" in Claude logs, empty results from any read tool** → Full Disk Access didn't take. Re-open the FDA pane and check both Claude and uvx are toggled on. Then quit and re-open Claude Desktop.
- **Tools hang for 60 seconds, then time out** → the Automation prompt was dismissed or never appeared. Open System Settings → Privacy & Security → Automation → expand **Claude** → toggle **Notes** on.
- **Locked note** → the body never decrypts. apple-notes-brain matches locked notes by title only and returns a sentinel body. Unlock the note in Notes.app if you want to read it.

For more, see [Troubleshooting](troubleshooting.md).
