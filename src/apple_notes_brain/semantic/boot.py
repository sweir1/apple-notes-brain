"""Semantic-subsystem boot block — fire-and-forget background init.

Mirrors obsidian-brain `src/server.ts:128–239`. Called once per process
from `apple_notes_brain.server` startup. Decouples the MCP handshake
(instant) from the embedder model download + first-time index + watcher
start (slow). Tools that DON'T need the embedder (the 12 lexical/CRUD
tools) work immediately; semantic tools either wait on the cached state
or return the empty-index hint until the background work finishes.

Lifecycle phases (recorded on `SemanticState.boot_phase`):

  pending → embedder-init → bootstrap → indexing → ready

If anything raises, `boot_phase = 'failed'` and `state.init_error` carries
the exception. Tools degrade gracefully: searches return an embedder-
not-ready hint; the rest of the server is unaffected.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from . import _logging as anb_logging
from .bootstrap import run_bootstrap

if TYPE_CHECKING:
    from ..tools_semantic import SemanticState

_log = logging.getLogger("apple-notes-brain")


def start_semantic_subsystem_background() -> threading.Thread | None:
    """Spawn the boot thread. Returns the thread (or None if [semantic]
    isn't installed so the whole subsystem is inert).

    Safe to call repeatedly — only the first call spawns; subsequent
    calls return the existing thread.

    The thread is a daemon so an interpreter exit doesn't wait for it.
    """
    try:
        from .. import tools_semantic
    except ImportError:
        return None
    if not getattr(tools_semantic, "HAVE_SEMANTIC", False):
        anb_logging.debug_log("boot: [semantic] extras not installed — skipping")
        return None

    # Idempotent: if a state already exists OR a boot thread is alive,
    # don't re-spawn.
    existing = getattr(tools_semantic, "_boot_thread", None)
    if existing is not None and existing.is_alive():
        anb_logging.debug_log("boot: thread already running — no-op")
        return existing
    if getattr(tools_semantic, "_state", None) is not None:
        anb_logging.debug_log("boot: state already constructed — no-op")
        return None

    thread = threading.Thread(
        target=_boot_loop,
        name="anb-semantic-boot",
        daemon=True,
    )
    tools_semantic._boot_thread = thread  # type: ignore[attr-defined]
    thread.start()
    return thread


def _boot_loop() -> None:
    """Build state singleton, reconcile, index, start watcher. All in
    background; exceptions go onto state.init_error rather than killing
    the thread (which would silently make the subsystem un-recoverable
    until process restart)."""
    from .. import tools_semantic

    anb_logging.setup_logging()
    anb_logging.debug_log("boot: thread entry")

    state: SemanticState | None = None
    try:
        # 1. Construct state singleton (this does embedder download + init).
        state = tools_semantic.get_state()
        _set_phase(state, "embedder-init")

        # 2. Reconcile against drift.
        _set_phase(state, "bootstrap")
        result = run_bootstrap(state.conn, state.embedder)
        for reason in result.reasons:
            _log.info("bootstrap: %s", reason)

        # 3. Decide between first-time / drift-forced / catch-up index.
        _set_phase(state, "indexing")
        if result.db_is_empty:
            _log.info(
                "semantic: index is empty, running first-time index. "
                "Time depends on vault size — typically under a minute for "
                "small vaults, a few minutes for thousands of notes."
            )
            stats = state.indexer.index_all(state.source)
            _log.info(
                "semantic: first-time index complete — %d notes indexed, "
                "%d chunks embedded in %dms",
                stats.notes_indexed, stats.chunks_embedded, stats.took_ms,
            )
        elif result.needs_reindex:
            _log.info(
                "semantic: drift detected — re-indexing %d existing note(s)",
                _node_count(state),
            )
            stats = state.indexer.index_all(state.source)
            _log.info(
                "semantic: drift reindex complete — %d notes, %d chunks, %dms",
                stats.notes_indexed, stats.chunks_embedded, stats.took_ms,
            )
        elif _no_catchup():
            anb_logging.debug_log(
                "boot: APPLE_NOTES_BRAIN_NO_CATCHUP=1 — skipping catch-up reindex"
            )
        else:
            stats = state.indexer.index_all(state.source)
            if stats.notes_indexed > 0:
                _log.info(
                    "semantic: startup catch-up — reindexed %d note(s) "
                    "(modified while the server was down)",
                    stats.notes_indexed,
                )
            else:
                anb_logging.debug_log("boot: catch-up no-op (nothing changed)")

        # 4. Start the watcher (independent of index pass success).
        try:
            state.watcher.start()  # type: ignore[union-attr]
            anb_logging.debug_log("boot: watcher started")
        except AttributeError:
            # state has no `watcher` attribute pre-Phase ζ; defer wiring
            # to a follow-up commit that constructs it on the state.
            anb_logging.debug_log("boot: state has no watcher attribute (skipping)")
        except Exception as exc:
            _log.warning("semantic: watcher failed to start: %s", exc)

        _set_phase(state, "ready")
        _log.info("semantic: subsystem ready")
    except Exception as exc:  # noqa: BLE001 — broad on purpose
        _log.error("semantic: boot failed: %s", exc, exc_info=True)
        if state is not None:
            _set_phase(state, "failed")
            state.init_error = exc  # type: ignore[attr-defined]
        else:
            # State wasn't even built — stash error on the module so
            # the next get_state() call can surface it.
            tools_semantic._boot_init_error = exc  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_phase(state, phase: str) -> None:
    """Set state.boot_phase if the attribute exists; otherwise set it."""
    setattr(state, "boot_phase", phase)
    anb_logging.debug_log("boot: phase → %s" % phase)


def _no_catchup() -> bool:
    return os.environ.get("APPLE_NOTES_BRAIN_NO_CATCHUP", "").strip() == "1"


def _node_count(state) -> int:
    try:
        return state.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    except Exception:
        return 0
