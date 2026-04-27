"""Cache coherence helpers — force NoteStore.sqlite to reflect live Notes.app state."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

from . import sqlite_reader as _db

log = logging.getLogger("apple-notes-brain.cache")

# A trivial AppleScript that forces Notes.app to process any queued mutations
# and flush its Core Data store. `count of notes` is cheap and read-only, but
# the very act of entering a `tell application "Notes"` block makes the app
# page in its state if it was suspended.
_PING_SCRIPT = 'tell application "Notes" to get (count of notes)'


def prewarm(timeout_s: float = 30.0) -> bool:
    """Run a no-op AppleScript call at server start.

    Purpose: force macOS to show the Automation-permission prompt NOW (before
    any user-invoked tool call) and force Notes.app to flush its cache so our
    first SQLite read sees fresh data. Idempotent and safe to call multiple times.

    Returns True on success, False on failure (prompt denied, Notes.app not
    installed, osascript missing). Never raises.

    Env var ``APPLE_NOTES_BRAIN_NO_PREWARM=1`` skips the AppleScript ping and
    returns False immediately. Useful in CI runners and headless environments
    where Notes.app is unavailable and osascript would block on a permission
    dialog that nothing can grant.
    """
    if os.environ.get("APPLE_NOTES_BRAIN_NO_PREWARM") == "1":
        log.info("AppleScript pre-warm skipped (APPLE_NOTES_BRAIN_NO_PREWARM=1)")
        return False
    try:
        result = subprocess.run(
            ["osascript", "-e", _PING_SCRIPT],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.info("AppleScript pre-warmed")
            return True
        stderr_excerpt = (result.stderr or "").strip()[:200]
        log.warning("AppleScript pre-warm failed (rc=%d): %s", result.returncode, stderr_excerpt)
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("AppleScript pre-warm errored: %s", exc)
        return False


def sync_after_write(timeout_s: float = 10.0) -> None:
    """Block briefly after a write so the next SQLite read sees the effect.

    Makes a cheap AppleScript call that round-trips through Notes.app, forcing
    it to persist any pending changes before returning. Swallows errors."""
    try:
        subprocess.run(
            ["osascript", "-e", _PING_SCRIPT],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("sync_after_write: %s", exc)


def refresh() -> dict:
    """Internal cache refresh. Calls the ping and returns a small dict
    describing what happened (e.g. {'ok': True, 'ms': 342}). Used by the
    background auto-refresher and sync_after_write."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["osascript", "-e", _PING_SCRIPT],
            timeout=30.0,
            capture_output=True,
            text=True,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        ok = result.returncode == 0
        error = None if ok else ((result.stderr or "").strip() or f"osascript exited {result.returncode}")
        return {"ok": ok, "ms": elapsed_ms, "error": error}
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {"ok": False, "ms": elapsed_ms, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tombstone storage — short-lived "just deleted" set for cache-layer hiding
# ---------------------------------------------------------------------------

_tombstones: dict[int, float] = {}
_tomb_lock = threading.Lock()
_TOMBSTONE_TTL_S = 60.0


def tombstone_folder(pk: int) -> None:
    """Mark a folder PK as 'just deleted' in the in-process cache layer.

    Used after a successful AppleScript folder delete to immediately hide the
    folder from list_folders(), even though the SQLite cache won't reflect the
    deletion for many seconds (CloudKit ack + Core Data save lag).
    Auto-expires after _TOMBSTONE_TTL_S so a sync-conflict resurrection
    (per Apple Discussions #250246292) eventually re-surfaces the folder.
    """
    with _tomb_lock:
        _tombstones[pk] = time.monotonic() + _TOMBSTONE_TTL_S


def is_tombstoned(pk: int) -> bool:
    with _tomb_lock:
        deadline = _tombstones.get(pk)
        if deadline is None:
            return False
        if time.monotonic() > deadline:
            _tombstones.pop(pk, None)
            return False
        return True


def reap_tombstones() -> None:
    """Drop expired tombstones. Cheap; safe to call from the bg loop."""
    now = time.monotonic()
    with _tomb_lock:
        for pk in [k for k, d in _tombstones.items() if d < now]:
            _tombstones.pop(pk, None)


# ---------------------------------------------------------------------------
# Rename overlay — short-lived "just renamed" map for cache-layer path rewrite
# ---------------------------------------------------------------------------

_renames: dict[int, tuple[float, str]] = {}


def rename_path_overlay(folder_pk: int, new_path: str) -> None:
    """Record a folder rename so list_folders() shows the new path immediately."""
    with _tomb_lock:
        _renames[folder_pk] = (time.monotonic() + _TOMBSTONE_TTL_S, new_path)


def apply_rename_overlay(rows: list[dict]) -> list[dict]:
    """Mutate path field for rows with live overlay; cascade to descendants
    whose old path was prefixed by the renamed parent."""
    with _tomb_lock:
        prefix_map: dict[str, str] = {}
        now = time.monotonic()
        for row in rows:
            try:
                pk = int(row["id"][1:])
            except (KeyError, ValueError, TypeError):
                continue
            entry = _renames.get(pk)
            if entry and entry[0] > now:
                old_path = row.get("path") or ""
                new_path = entry[1]
                row["path"] = new_path
                if old_path:
                    prefix_map[old_path + "/"] = new_path + "/"
        # Cascade
        for row in rows:
            path = row.get("path") or ""
            for old_prefix, new_prefix in prefix_map.items():
                if path.startswith(old_prefix):
                    row["path"] = new_prefix + path[len(old_prefix):]
                    break
    return rows


# ---------------------------------------------------------------------------
# Note-count delta overlay — short-lived adjustments while SQLite catches up
# ---------------------------------------------------------------------------

_count_deltas: dict[int, tuple[float, int]] = {}


def adjust_note_count(folder_pk: int, delta: int) -> None:
    """Record a count delta for a folder (negative for deletes/moves-out, positive for moves-in)."""
    with _tomb_lock:
        deadline = time.monotonic() + _TOMBSTONE_TTL_S
        if folder_pk in _count_deltas:
            existing_deadline, existing = _count_deltas[folder_pk]
            _count_deltas[folder_pk] = (deadline, existing + delta)
        else:
            _count_deltas[folder_pk] = (deadline, delta)


def get_count_delta(folder_pk: int) -> int:
    """Live delta for a folder, or 0 if expired/absent."""
    with _tomb_lock:
        entry = _count_deltas.get(folder_pk)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        return 0


def reap_overlays() -> None:
    """Drop expired rename and count-delta entries. Cheap; safe per bg tick."""
    now = time.monotonic()
    with _tomb_lock:
        for pk in [k for k, (d, _) in _renames.items() if d < now]:
            _renames.pop(pk, None)
        for pk in [k for k, (d, _) in _count_deltas.items() if d < now]:
            _count_deltas.pop(pk, None)


# ---------------------------------------------------------------------------
# AppleScript bridge recovery — quit + relaunch Notes.app on bridge corruption
# ---------------------------------------------------------------------------

_recover_lock = threading.Lock()
_recover_last_attempt: float = 0.0
_RECOVER_COOLDOWN_S = 60.0


def recover_bridge(timeout_s: float = 15.0) -> bool:
    """Force-restart Notes.app to recover its AppleScript bridge.

    Rapid-fire osascript invocations can corrupt the bridge's NSXPCConnection
    state, causing every subsequent `tell application "Notes"` to fail with
    `Invalid index` even for valid Z_PKs. The only known recovery is to quit
    and relaunch Notes.app.

    Rate-limited to once per _RECOVER_COOLDOWN_S to prevent retry storms.
    Returns True if Notes.app is responsive after recovery, False otherwise.
    """
    global _recover_last_attempt
    with _recover_lock:
        now = time.monotonic()
        if now - _recover_last_attempt < _RECOVER_COOLDOWN_S:
            return False
        _recover_last_attempt = now

        log.warning("AppleScript bridge corrupted — restarting Notes.app")
        try:
            subprocess.run(
                ["osascript", "-e", 'tell application "Notes" to quit'],
                timeout=5.0,
                capture_output=True,
            )
        except Exception:
            pass

        deadline = time.monotonic() + min(timeout_s, 8.0)
        while time.monotonic() < deadline:
            try:
                r = subprocess.run(["pgrep", "-x", "Notes"], capture_output=True, timeout=1.0)
                if r.returncode != 0:
                    break
            except Exception:
                break
            time.sleep(0.2)
        else:
            try:
                subprocess.run(["pkill", "-x", "Notes"], timeout=2.0, capture_output=True)
            except Exception:
                pass

        try:
            subprocess.run(["open", "-a", "Notes", "-g"], timeout=5.0, capture_output=True)
        except Exception as exc:
            log.warning("recover_bridge: failed to relaunch Notes.app: %s", exc)
            return False

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["osascript", "-e", 'tell application "Notes" to count of accounts'],
                    timeout=3.0,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    log.info("AppleScript bridge recovered after Notes.app restart")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        log.warning("recover_bridge: Notes.app did not become responsive in %.1fs", timeout_s)
        return False


# ---------------------------------------------------------------------------
# Background auto-refresh — periodic non-blocking cache sync, idle-aware
# ---------------------------------------------------------------------------

_bg_thread: threading.Thread | None = None
_bg_stop = threading.Event()
_bg_wake = threading.Event()  # set by mark_activity() to wake the loop early
_bg_last_refresh_ms: int = 0
_bg_tick_count: int = 0
_bg_skip_count: int = 0  # ticks skipped due to idle or Notes.app not running
_last_data_version: int = 0  # last seen PRAGMA data_version; gates AppleScript pings

# Monotonic seconds of last MCP tool invocation. Initialized at loop start.
_last_activity_monotonic: float = 0.0

# How long with no tool activity before we pause ticking. 0 = never pause.
_idle_threshold_s: float = 300.0


def mark_activity() -> None:
    """Record that an MCP tool was invoked.

    Call at the top of every user-facing tool function. Resets the idle
    timer. If the refresher was paused due to idle, wakes it immediately so
    the next tool call (already in flight or imminent) sees fresh cache.
    """
    global _last_activity_monotonic
    now = time.monotonic()
    was_idle = (
        _idle_threshold_s > 0
        and _last_activity_monotonic > 0
        and (now - _last_activity_monotonic) > _idle_threshold_s
    )
    _last_activity_monotonic = now
    if was_idle:
        _bg_wake.set()


def _notes_running() -> bool:
    """Return True if Notes.app is currently running, without launching it.

    Uses pgrep (universally available on macOS). This lets the background
    refresher skip ticks when Notes.app is closed — otherwise we'd auto-launch
    the app every tick, which is terrible UX.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", "Notes"],
            capture_output=True,
            timeout=1.0,
        )
        return result.returncode == 0
    except Exception:
        return True


def _bg_loop(interval_s: float) -> None:
    """Daemon thread body: wake every interval_s seconds and refresh the cache.

    Idle-aware: if no MCP tool has been called for `_idle_threshold_s` seconds,
    skip ticks until activity resumes (signalled by `_bg_wake`). Always exits
    cleanly when `_bg_stop` is set.
    """
    global _bg_last_refresh_ms, _bg_tick_count, _bg_skip_count, _last_activity_monotonic, _last_data_version
    _last_activity_monotonic = time.monotonic()
    log.info(
        "apple-notes-brain background refresh loop started (interval=%.1fs, idle_threshold=%.0fs)",
        interval_s,
        _idle_threshold_s,
    )
    while not _bg_stop.is_set():
        # Wait for either the tick interval OR an early wake-up from mark_activity().
        woken_by_activity = _bg_wake.wait(interval_s)
        _bg_wake.clear()
        if _bg_stop.is_set():
            break

        # Cheap maintenance every tick regardless of whether we ping.
        reap_tombstones()
        reap_overlays()

        idle_for = time.monotonic() - _last_activity_monotonic
        # Skip if we're idle and weren't explicitly woken by activity resumption.
        if _idle_threshold_s > 0 and idle_for > _idle_threshold_s and not woken_by_activity:
            _bg_skip_count += 1
            continue

        if not _notes_running():
            _bg_skip_count += 1
            continue

        # data_version gate: skip the AppleScript ping when SQLite is quiet.
        # PRAGMA data_version increments cross-process on every committed write,
        # so an unchanged value means there's nothing for us to flush.
        try:
            cur_dv = _db.data_version()
        except Exception:
            cur_dv = -1  # forces a ping on read failure

        if (
            cur_dv != -1
            and _last_data_version != 0
            and cur_dv == _last_data_version
            and not woken_by_activity
        ):
            _bg_skip_count += 1
            continue

        if cur_dv != -1:
            _last_data_version = cur_dv

        _bg_tick_count += 1
        try:
            r = refresh()
            _bg_last_refresh_ms = r.get("ms", 0)
            if not r.get("ok"):
                log.debug("bg refresh failed: %s", r.get("error"))
        except Exception as exc:  # noqa: BLE001
            log.debug("bg refresh error: %s", exc)
    log.info("apple-notes-brain background refresh loop exiting")


def start_background_refresh(interval_s: float | None = None) -> bool:
    """Start the periodic background cache refresher.

    Env vars (read at startup):
      - NOTES_MCP_AUTO_REFRESH=0   disable the thread entirely.
      - NOTES_MCP_REFRESH_INTERVAL=<seconds>  cadence (default 10).
      - NOTES_MCP_IDLE_THRESHOLD=<seconds>    pause after N idle seconds
        (default 300, i.e. 5 min; set to 0 to disable idle pausing).

    Returns True if started, False if already running or disabled.
    """
    global _bg_thread, _idle_threshold_s

    if os.environ.get("NOTES_MCP_AUTO_REFRESH", "1") == "0":
        log.info("background refresh disabled (NOTES_MCP_AUTO_REFRESH=0)")
        return False

    if _bg_thread is not None and _bg_thread.is_alive():
        return False

    if interval_s is None:
        try:
            interval_s = float(os.environ.get("NOTES_MCP_REFRESH_INTERVAL", "4"))
        except (TypeError, ValueError):
            interval_s = 4.0

    if interval_s < 1.0:
        log.warning("background refresh interval %.1fs too low; clamping to 1s", interval_s)
        interval_s = 1.0

    try:
        _idle_threshold_s = float(os.environ.get("NOTES_MCP_IDLE_THRESHOLD", "300"))
    except (TypeError, ValueError):
        _idle_threshold_s = 300.0
    if _idle_threshold_s < 0:
        _idle_threshold_s = 0.0

    _bg_stop.clear()
    _bg_wake.clear()
    _bg_thread = threading.Thread(
        target=_bg_loop,
        args=(interval_s,),
        daemon=True,
        name="apple-notes-brain-bg-refresh",
    )
    _bg_thread.start()
    return True


def stop_background_refresh(join_timeout_s: float = 2.0) -> None:
    """Stop the background refresher. Safe to call multiple times."""
    global _bg_thread
    _bg_stop.set()
    _bg_wake.set()  # wake the loop so it exits promptly rather than after the next tick
    if _bg_thread is not None and _bg_thread.is_alive():
        _bg_thread.join(timeout=join_timeout_s)
    _bg_thread = None


def background_refresh_status() -> dict:
    """Inspect the background refresher state. Useful for debugging / introspection."""
    running = _bg_thread is not None and _bg_thread.is_alive()
    now = time.monotonic()
    idle_for_s = (
        max(0.0, now - _last_activity_monotonic)
        if _last_activity_monotonic > 0
        else 0.0
    )
    is_idle = _idle_threshold_s > 0 and idle_for_s > _idle_threshold_s
    return {
        "running": running,
        "tick_count": _bg_tick_count,
        "skip_count": _bg_skip_count,
        "last_refresh_ms": _bg_last_refresh_ms,
        "idle_for_s": round(idle_for_s, 1),
        "idle_threshold_s": _idle_threshold_s,
        "is_idle_paused": is_idle,
        "last_data_version": _last_data_version,
    }
