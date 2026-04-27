#!/bin/bash
# preflight.sh — local pre-release validation for apple-notes-brain.
#
# Mirrors what CI runs on PR/push, plus a few extras you'd want before
# tagging a release. Idempotent; safe to re-run. All steps stream output
# live; final summary at the end with timings.
#
# Usage:
#   ./scripts/preflight.sh              # run all steps
#   ./scripts/preflight.sh --skip-build # skip uv build
#
# Exit code is non-zero if any required step fails.

set -uo pipefail
umask 022

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE=''
fi

SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *)
      printf '%sUnknown flag:%s %s\n' "$C_RED" "$C_RESET" "$arg" >&2
      exit 2
      ;;
  esac
done

# Always run from repo root regardless of where the user invoked us.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Step record: (label, status, elapsed_seconds)
declare -a STEP_RESULTS=()

run_step() {
  local label="$1"; shift
  local optional="${OPTIONAL:-0}"
  printf '\n%s==>%s %s%s%s\n' "$C_BLUE" "$C_RESET" "$C_BOLD" "$label" "$C_RESET"
  local t0 t1
  t0=$(date +%s)
  if "$@"; then
    t1=$(date +%s)
    local elapsed=$((t1 - t0))
    printf '%s✓ %s%s in %ss\n' "$C_GREEN" "$label" "$C_RESET" "$elapsed"
    STEP_RESULTS+=("OK|$label|${elapsed}s")
    return 0
  else
    local rc=$?
    t1=$(date +%s)
    local elapsed=$((t1 - t0))
    if [[ "$optional" == "1" ]]; then
      printf '%s⚠ %s skipped/failed (optional)%s in %ss\n' "$C_YELLOW" "$label" "$C_RESET" "$elapsed"
      STEP_RESULTS+=("SKIP|$label|${elapsed}s")
      return 0
    fi
    printf '%s✗ %s failed (rc=%d)%s in %ss\n' "$C_RED" "$label" "$rc" "$C_RESET" "$elapsed"
    STEP_RESULTS+=("FAIL|$label|${elapsed}s")
    return $rc
  fi
}

# ---------- Step 1: dependency sync ----------
run_step "uv sync --extra dev" \
  uv sync --extra dev || exit 1

# ---------- Step 2: full pytest with coverage ----------
run_step "pytest --cov (full suite, fail_under=60)" \
  uv run pytest -q || exit 1

# ---------- Step 3: integration smoke (no-cov: covered by step 2) ----------
run_step "MCP smoke (subprocess-spawn the server)" \
  uv run pytest --no-cov -m integration -q tests/integration/test_mcp_smoke.py || exit 1

# ---------- Step 4: build wheel + sdist ----------
if [[ "$SKIP_BUILD" == "0" ]]; then
  run_step "uv build (wheel + sdist)" \
    uv build || exit 1
else
  printf '\n%s==>%s %s(skipped)%s uv build\n' "$C_DIM" "$C_RESET" "$C_DIM" "$C_RESET"
  STEP_RESULTS+=("SKIP|uv build|0s")
fi

# ---------- Step 5: server.json shape sanity ----------
run_step "server.json validates as JSON + version matches pyproject" \
  uv run python -c "
import json, tomllib
with open('pyproject.toml', 'rb') as f:
    proj_v = tomllib.load(f)['project']['version']
with open('server.json') as f:
    sj = json.load(f)
assert sj['version'] == proj_v, f'top-level version mismatch: {sj[\"version\"]!r} vs pyproject {proj_v!r}'
assert sj['packages'][0]['version'] == proj_v, f'packages[0].version mismatch: {sj[\"packages\"][0][\"version\"]!r} vs pyproject {proj_v!r}'
print(f'✓ server.json + pyproject both at {proj_v}')
print('  reminder: hand-edit server.json environmentVariables if you added new env vars in code')
" || exit 1

# ---------- Step 6: console script registers ----------
run_step "apple-notes-brain console script imports cleanly" \
  uv run python -c "
import subprocess, sys
from pathlib import Path
binary = Path(sys.executable).parent / 'apple-notes-brain'
assert binary.exists(), f'console script missing: {binary}'
print(f'✓ binary at {binary}')
" || exit 1

# ---------- Final summary ----------

printf '\n%s%s========================================%s\n' "$C_BOLD" "$C_BLUE" "$C_RESET"
printf '%sPreflight summary%s\n' "$C_BOLD" "$C_RESET"
printf '%s%s========================================%s\n' "$C_BOLD" "$C_BLUE" "$C_RESET"

printed_failure=0
for entry in "${STEP_RESULTS[@]}"; do
  IFS='|' read -r status label elapsed <<<"$entry"
  case "$status" in
    OK)   printf '  %s✓%s %-50s %s\n' "$C_GREEN" "$C_RESET" "$label" "$elapsed" ;;
    SKIP) printf '  %s⚠%s %-50s %s\n' "$C_YELLOW" "$C_RESET" "$label" "$elapsed" ;;
    FAIL) printf '  %s✗%s %-50s %s\n' "$C_RED" "$C_RESET" "$label" "$elapsed"
          printed_failure=1
          ;;
  esac
done

git_state="$(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null)"
printf '\n  %sgit:%s %s\n' "$C_DIM" "$C_RESET" "$git_state"
status_change="$(git status --short 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$status_change" != "0" ]]; then
  printf '  %sworking tree:%s %s changed file(s)\n' "$C_DIM" "$C_RESET" "$status_change"
fi

if [[ "$printed_failure" == "1" ]]; then
  printf '\n%s✗ Preflight FAILED — fix issues before tagging.%s\n' "$C_RED" "$C_RESET"
  exit 1
fi

printf '\n%s✓ Preflight passed.%s Ready to bump version + tag.\n' "$C_GREEN" "$C_RESET"
