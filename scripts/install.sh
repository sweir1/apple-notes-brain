#!/bin/bash
# apple-notes-brain one-line macOS installer
# version: 2026-04-26
#
# Usage:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/sweir1/apple-notes-brain/main/scripts/install.sh)"
#
# Does:
#   - Verifies macOS + Apple Notes
#   - Installs uv if missing (via Astral's curl installer)
#   - Symlinks uv/uvx into /usr/local/bin so Claude Desktop's GUI PATH sees them
#   - Pre-warms apple-notes-brain via uvx (downloads from PyPI, validates package)
#   - Merges into Claude Desktop config (preserves other MCP servers)
#   - Walks the user through Full Disk Access for BOTH Claude.app AND the
#     uv-managed Python (the cached-Python TCC quirk that breaks SQLite reads)
#   - Heads-up about the Automation permission prompt
#   - Restarts Claude Desktop

set -euo pipefail
umask 022

# ---------------------------- pretty output ---------------------------- #

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN=''
fi

CURRENT_STEP="preflight"

info()  { printf '%s==>%s %s%s%s\n' "$C_BLUE" "$C_RESET" "$C_BOLD" "$1" "$C_RESET"; }
note()  { printf '    %s%s%s\n' "$C_DIM" "$1" "$C_RESET"; }
ok()    { printf '%s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
warn()  { printf '%s⚠%s %s\n' "$C_YELLOW" "$C_RESET" "$1"; }
die()   { printf '\n%sError:%s %s\n' "$C_RED" "$C_RESET" "$1" >&2; exit 1; }

on_error() {
  local exit_code=$?
  printf '\n%s✗ Step "%s" failed (exit %d).%s\n' "$C_RED" "$CURRENT_STEP" "$exit_code" "$C_RESET" >&2
  printf '  Repo + issues: https://github.com/sweir1/apple-notes-brain\n' >&2
}
trap on_error ERR

# ---------------------------- /dev/tty helpers ---------------------------- #
# When run via `/bin/bash -c "$(curl ...)"` stdin is the curl output, not the
# user's terminal. Route prompts/pauses to /dev/tty.

have_tty() { [[ -r /dev/tty && -w /dev/tty ]]; }

pause_enter() {
  local msg=$1
  if ! have_tty; then
    warn "Non-interactive run — skipping pause."
    return 0
  fi
  printf '\n%s%s%s ' "$C_YELLOW" "$msg" "$C_RESET" > /dev/tty
  IFS= read -r _ < /dev/tty || true
}

# ---------------------------- Step 0: preflight ---------------------------- #

CURRENT_STEP="preflight"
info "Preflight checks"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  die "Do not run this installer as root. Run it as your regular user — sudo will be invoked only when needed (one symlink step)."
fi

OS="$(uname -s)"
if [[ "$OS" != "Darwin" ]]; then
  cat >&2 <<EOF

This installer is for macOS only. apple-notes-brain depends on Apple Notes,
which only exists on macOS.

Detected OS: $OS
EOF
  exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
  arm64|x86_64) ;;
  *) die "Unsupported CPU architecture: $ARCH" ;;
esac

if [[ ! -d "/System/Applications/Notes.app" && ! -d "/Applications/Notes.app" ]]; then
  warn "Apple Notes.app not found in /System/Applications or /Applications."
  note "The server will install but won't have anything to talk to until Notes.app is restored."
fi

ok "macOS detected ($ARCH), Notes.app present"

# ---------------------------- Step 1: install uv ---------------------------- #

CURRENT_STEP="uv-install"
info "Checking for uv"

# Common uv locations to add to PATH if missing — covers the Astral installer
# (~/.local/bin), Homebrew Apple Silicon (/opt/homebrew/bin), and Homebrew Intel
# (/usr/local/bin).
for d in "$HOME/.local/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
  case ":$PATH:" in
    *":$d:"*) ;;
    *) PATH="$d:$PATH" ;;
  esac
done
export PATH

if command -v uv >/dev/null 2>&1; then
  ok "uv already installed: $(uv --version)"
else
  note "uv not found — installing via Astral's official installer."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The Astral installer writes to ~/.local/bin and prints a hint about adding
  # to PATH. Source the env file it drops, if present, to make uv available now.
  for env_file in "$HOME/.local/bin/env" "$HOME/.cargo/env"; do
    [[ -r "$env_file" ]] && . "$env_file"
  done
  command -v uv >/dev/null 2>&1 || die "uv install completed but 'uv' is still not on PATH. Open a new Terminal and rerun."
  ok "uv installed: $(uv --version)"
fi

UV_BIN="$(command -v uv)"
UVX_BIN="$(command -v uvx)"
[[ -x "$UVX_BIN" ]] || die "uvx not found alongside uv at $UV_BIN — uv install may be incomplete."

# ---------------------------- Step 2: GUI PATH symlinks ---------------------- #

CURRENT_STEP="symlinks"
info "Linking uv and uvx into /usr/local/bin so GUI apps can find them"
note "Some Claude Desktop builds spawn child processes with a minimal PATH that may not include ~/.local/bin."
note "Symlinking into /usr/local/bin (which IS on the GUI PATH) guarantees discovery. Requires sudo once."

sudo mkdir -p /usr/local/bin
sudo ln -sf "$UV_BIN"  /usr/local/bin/uv
sudo ln -sf "$UVX_BIN" /usr/local/bin/uvx

if /usr/local/bin/uvx --version >/dev/null 2>&1; then
  ok "/usr/local/bin/uvx -> $UVX_BIN ($(/usr/local/bin/uvx --version))"
else
  die "Symlink created but /usr/local/bin/uvx is not executable."
fi

# ---------------------------- Step 3: pre-warm package ---------------------- #

CURRENT_STEP="prewarm"
info "Pre-warming apple-notes-brain (downloads ~12 MiB of deps from PyPI on first run)"

# The MCP server runs a long-lived stdio loop — there's no --version flag we
# can check. Instead we launch it backgrounded with stdin closed, give it a
# few seconds to import + initialise, then kill and grep stderr for the
# success signal.
PREWARM_LOG="/tmp/apple-notes-brain-prewarm.$$"
/usr/local/bin/uvx apple-notes-brain </dev/null >"$PREWARM_LOG" 2>&1 &
PREWARM_PID=$!

# Up to ~25s for cold cache (lxml + pydantic-core wheels + protobuf)
for _ in $(seq 1 25); do
  if grep -q "apple-notes-brain starting" "$PREWARM_LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$PREWARM_PID" 2>/dev/null; then
    # process exited before we saw the signal
    break
  fi
  sleep 1
done

kill "$PREWARM_PID" 2>/dev/null || true
wait "$PREWARM_PID" 2>/dev/null || true

if grep -q "apple-notes-brain starting" "$PREWARM_LOG" 2>/dev/null; then
  ok "apple-notes-brain boots successfully via uvx"
  rm -f "$PREWARM_LOG"
else
  printf '\n%s---- pre-warm output ----%s\n' "$C_DIM" "$C_RESET" >&2
  cat "$PREWARM_LOG" >&2 || true
  printf '%s-------------------------%s\n\n' "$C_DIM" "$C_RESET" >&2
  rm -f "$PREWARM_LOG"
  die "apple-notes-brain did not boot via uvx. See output above."
fi

# ---------------------------- Step 4: merge Claude Desktop config ----------- #

CURRENT_STEP="config-merge"
info "Merging into Claude Desktop config"

CLAUDE_CFG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CFG="$CLAUDE_CFG_DIR/claude_desktop_config.json"

mkdir -p "$CLAUDE_CFG_DIR"

if [[ -f "$CLAUDE_CFG" ]]; then
  backup="$CLAUDE_CFG.bak.$(date +%s)"
  cp "$CLAUDE_CFG" "$backup"
  note "Existing config backed up to $backup"
fi

# python3 ships with macOS Command Line Tools / Xcode — virtually always
# present on a Mac that's installed Claude Desktop. Use it for safe JSON merge
# rather than depending on jq (not always installed).
CFG_PATH="$CLAUDE_CFG" python3 - <<'PY'
import json, os, sys
p = os.environ["CFG_PATH"]
cfg = {}
if os.path.exists(p):
    try:
        with open(p) as f:
            cfg = json.load(f) or {}
    except Exception as e:
        print(f"Existing config is not valid JSON — starting fresh (old file preserved in .bak): {e}", file=sys.stderr)
        cfg = {}
if not isinstance(cfg, dict):
    cfg = {}
cfg.setdefault("mcpServers", {})
prev = cfg["mcpServers"].get("apple-notes-brain")
cfg["mcpServers"]["apple-notes-brain"] = {
    "command": "uvx",
    "args": [
        "apple-notes-brain"
    ],
}
with open(p, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print("replaced" if prev else "added")
PY

ok "Claude Desktop config: apple-notes-brain entry written ($CLAUDE_CFG)"

# ---------------------------- Step 5: detect uv-managed Python -------------- #

CURRENT_STEP="detect-python"
info "Detecting the uv-managed Python that uvx will spawn"
note "We do NOT use 'uv python find' here because it returns whatever Python uv prefers"
note "in the current directory, which may be a project .venv if you happen to be in one."
note "We scan the uv cache directly — that's what uvx actually executes from."

# Scan ~/.local/share/uv/python/ for installed managed Pythons matching 3.11+.
# Pre-warm in step 3 will have populated this directory with at least one
# matching Python (uv downloads on demand if missing).
UV_PY=""
UV_PY_VER=""
for path in "$HOME"/.local/share/uv/python/cpython-*/bin/python; do
  [[ -e "$path" ]] || continue
  base_dir="${path%/bin/python}"
  base_name="${base_dir##*/}"          # cpython-3.12.13-macos-aarch64-none
  ver="${base_name#cpython-}"
  ver="${ver%%-*}"                     # "3.12.13" or "3.12"
  major="${ver%%.*}"
  rest="${ver#*.}"
  minor="${rest%%.*}"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || continue
  # apple-notes-brain requires Python >= 3.11
  if (( major > 3 )) || { (( major == 3 )) && (( minor >= 11 )); }; then
    if [[ -z "$UV_PY" ]]; then
      UV_PY="$path"
      UV_PY_VER="$ver"
    else
      # Pick the highest-versioned interpreter — that's what uvx will resolve to.
      higher="$(printf '%s\n%s' "$UV_PY_VER" "$ver" | sort -V | tail -1)"
      if [[ "$higher" == "$ver" ]]; then
        UV_PY="$path"
        UV_PY_VER="$ver"
      fi
    fi
  fi
done

if [[ -n "$UV_PY" && -e "$UV_PY" ]]; then
  ok "uv-managed Python: $UV_PY (Python $UV_PY_VER)"
else
  warn "No uv-managed Python found at ~/.local/share/uv/python/cpython-*/bin/python."
  warn "Pre-warm should have downloaded one — this is unexpected."
  UV_PY="$HOME/.local/share/uv/python/  (find your installed Python under here — version >= 3.11)"
fi

# ---------------------------- Step 6: Full Disk Access ---------------------- #

CURRENT_STEP="full-disk-access"
info "Granting Full Disk Access (TWO entries needed)"

cat <<EOF

${C_BOLD}macOS requires you to toggle Full Disk Access by hand — it's a kernel-enforced
permission (TCC) that no script can grant, not even with sudo.${C_RESET}

apple-notes-brain reads NoteStore.sqlite directly for fast search. macOS gates
that behind Full Disk Access, and FDA on Claude Desktop alone is NOT enough —
it needs to be granted to the uv-managed Python that ${C_BOLD}uvx${C_RESET} spawns too.

${C_BOLD}Opening System Settings → Privacy & Security → Full Disk Access now.${C_RESET}

EOF

FDA_URL_MODERN="x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles"
FDA_URL_LEGACY="x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"
open "$FDA_URL_MODERN" 2>/dev/null || open "$FDA_URL_LEGACY" 2>/dev/null || open -a "System Settings" || warn "Couldn't open System Settings automatically — open it manually."

UV_PY_DIR="$(dirname "$UV_PY")"
cat <<EOF
You need to toggle ${C_BOLD}three${C_RESET} entries ${C_BOLD}ON${C_RESET} in the Full Disk Access list. Try
them in this order — if SQLite reads work after the first two, you can skip
the third.

  ${C_BOLD}1. Claude${C_RESET} (the app)
     - If listed → flip its toggle ON.
     - If not listed → click ${C_BOLD}+${C_RESET}, choose Applications → Claude.app, toggle ON.

  ${C_BOLD}2. uvx${C_RESET} (the wrapper that spawns Python)
     - Click ${C_BOLD}+${C_RESET}, press ${C_BOLD}Cmd+Shift+G${C_RESET}, paste:

         ${C_CYAN}/usr/local/bin/uvx${C_RESET}

     - Press Enter, click ${C_BOLD}Open${C_RESET}, then toggle ON. While you're there, do
       the same for ${C_CYAN}/usr/local/bin/uv${C_RESET} too — same procedure.
     - These are signed Astral binaries — they appear normally in the picker,
       no gray-out. macOS TCC ${C_BOLD}should${C_RESET} propagate this FDA grant down to the
       python child uvx spawns.

  After steps 1 and 2: ${C_BOLD}quit and relaunch Claude Desktop${C_RESET}, then ask it to
  list your notes. If you see notes returned, you're done — skip step 3.

  ${C_BOLD}3. The uv-managed Python${C_RESET} (only if step 2 wasn't enough)

     ${C_YELLOW}Heads-up:${C_RESET} this Python is grayed out in the picker. uv's bundled
     Python is ad-hoc signed, and macOS's FDA file picker rejects
     ad-hoc-signed Mach-O binaries. Use ${C_BOLD}drag-and-drop${C_RESET} instead.

     Steps (keep the FDA pane open):
       a. Switch to Finder (or open one with Cmd+N).
       b. In Finder, press ${C_BOLD}Cmd+Shift+G${C_RESET} (Go to Folder).
       c. Paste this directory path:

            ${C_CYAN}$UV_PY_DIR${C_RESET}

       d. Press Enter. Finder shows the bin/ contents.
       e. ${C_BOLD}Drag${C_RESET} the file ${C_BOLD}python3${C_RESET} (the real binary near the bottom — NOT
          the symlinks above it) from Finder ${C_BOLD}directly onto the Full Disk
          Access list pane${C_RESET} in System Settings.
       f. The new entry appears with a toggle — flip it ON.

     If drag-and-drop fails too: in the FDA ${C_BOLD}+${C_RESET} picker, paste the path with
     Cmd+Shift+G, then ${C_BOLD}type${C_RESET} ${C_BOLD}python3${C_RESET} into the filename bar and press Enter.
     Some macOS builds let typed names bypass the gray-out filter.

  If after all three you still see "cannot open NoteStore" — check
  https://github.com/sweir1/apple-notes-brain/issues for the latest workaround.

The Python path may change when uv updates its bundled Python (e.g. after
${C_DIM}uv self update${C_RESET}). If reads stop working after that, redo step 3 with
the new path:
   ${C_DIM}ls -d ~/.local/share/uv/python/cpython-*/bin/python3${C_RESET}

EOF

pause_enter "Press Enter once Claude + uvx are toggled ON in Full Disk Access (and python too if step 2 wasn't enough)..."

ok "Continuing — Full Disk Access checked off"

# ---------------------------- Step 7: Automation heads-up ------------------- #

CURRENT_STEP="automation"
info "About the Automation permission prompt"

cat <<EOF

The first time apple-notes-brain calls a write tool (create_note, update_note,
delete_note, etc.) macOS will pop up a dialog:

  ${C_BOLD}"Claude" wants to control "Notes.app"${C_RESET}

Click ${C_BOLD}OK${C_RESET}. This is the Automation permission — required for AppleScript-based
writes. It's separate from Full Disk Access, sticky once approved, and only
ever asked once per (parent process, target app) pair.

If you accidentally click "Don't Allow", reset with:
  ${C_DIM}tccutil reset AppleEvents com.anthropic.claudefordesktop${C_RESET}
…and the prompt will fire again on the next write tool use.

EOF

# ---------------------------- Step 8: relaunch Claude ----------------------- #

CURRENT_STEP="relaunch"
info "Restarting Claude Desktop"

if [[ -d "/Applications/Claude.app" ]]; then
  osascript -e 'tell application "Claude" to quit' >/dev/null 2>&1 || true
  sleep 2
  open -a "Claude"
  ok "Claude Desktop relaunched — your new config + permissions are live"
else
  warn "Claude Desktop is not installed at /Applications/Claude.app"
  note "Download it from https://claude.ai/download — your config is already in place and will work when you install it."
fi

# ---------------------------- Step 9: summary ------------------------------- #

CURRENT_STEP="summary"
trap - ERR

cat <<EOF

${C_GREEN}${C_BOLD}✓ apple-notes-brain is now wired into Claude Desktop.${C_RESET}

  Package:  ${C_BOLD}apple-notes-brain${C_RESET} on PyPI (https://pypi.org/project/apple-notes-brain/)
  Config:   ${C_DIM}$CLAUDE_CFG${C_RESET}
  Python:   ${C_DIM}$UV_PY${C_RESET}

${C_BOLD}Try it out:${C_RESET}
  Open Claude Desktop, then say:
  "List my Apple Notes folders."
  "Search my notes for 'TODO'."
  "Create a note titled 'Test' with body 'Hello from Claude'."

${C_BOLD}Update later:${C_RESET}
  uvx caches packages by version — run this to pull a newer release:
    ${C_DIM}uvx --refresh apple-notes-brain${C_RESET}
  Or wipe the cache entry and let the next launch re-fetch:
    ${C_DIM}uv cache clean apple-notes-brain${C_RESET}

${C_BOLD}Repo + issues:${C_RESET}
  https://github.com/sweir1/apple-notes-brain

${C_BOLD}Sibling project (Obsidian):${C_RESET}
  https://github.com/sweir1/obsidian-brain

EOF
