"""Unit tests for cache background-refresh thread + bridge recovery.

Covers:
- mark_activity() updates _last_activity_monotonic and wakes idle thread.
- start_background_refresh() honours NOTES_MCP_AUTO_REFRESH=0, idempotent.
- stop_background_refresh() joins the thread cleanly.
- background_refresh_status() shape + values pre/post tick.
- Idle pausing via NOTES_MCP_IDLE_THRESHOLD.
- prewarm() / refresh() return contracts on success / timeout / non-zero rc.
- recover_bridge() cooldown (Bug hotspot #14) + concurrent-call race.

All tests are time-bounded with @pytest.mark.timeout(5) to surface hangs.
Subprocess invocations (osascript / pgrep / pkill / open) are stubbed via
the `mock_subprocess_run` fixture (or per-test mocker).
"""
from __future__ import annotations

import subprocess
import threading
import time

import pytest

from apple_notes_brain import cache


# Shared timeout marker — applied per-test to prevent CI hangs.
pytestmark = pytest.mark.timeout(5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.02) -> bool:
    """Poll predicate() until True or timeout. Returns whether it became True."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


# ---------------------------------------------------------------------------
# mark_activity
# ---------------------------------------------------------------------------

class TestMarkActivity:
    def test_updates_last_activity(self, frozen_monotonic):
        # Pretend time started at 1000; advance to 1500 then mark.
        frozen_monotonic.advance(500)
        cache.mark_activity()
        assert cache._last_activity_monotonic == frozen_monotonic.now()

    def test_first_call_does_not_set_wake_event(self, frozen_monotonic):
        """When _last_activity_monotonic is 0 (initial), was_idle is False."""
        cache._bg_wake.clear()
        cache.mark_activity()
        assert not cache._bg_wake.is_set()

    def test_wake_set_when_resuming_after_idle(self, frozen_monotonic):
        # Set up: had activity, then went idle past threshold.
        cache._idle_threshold_s = 10.0
        cache.mark_activity()  # establishes baseline
        cache._bg_wake.clear()
        frozen_monotonic.advance(20)  # now idle (> 10s)
        cache.mark_activity()
        assert cache._bg_wake.is_set()

    def test_wake_not_set_when_still_active(self, frozen_monotonic):
        cache._idle_threshold_s = 10.0
        cache.mark_activity()
        cache._bg_wake.clear()
        frozen_monotonic.advance(2)  # well within threshold
        cache.mark_activity()
        assert not cache._bg_wake.is_set()


# ---------------------------------------------------------------------------
# start_background_refresh / stop_background_refresh
# ---------------------------------------------------------------------------

class TestStartStopBackgroundRefresh:
    def test_disabled_via_env_returns_false(
        self, monkeypatch, mock_subprocess_run
    ):
        monkeypatch.setenv("NOTES_MCP_AUTO_REFRESH", "0")
        assert cache.start_background_refresh() is False
        assert cache._bg_thread is None

    @pytest.mark.slow
    def test_default_env_starts_thread(self, monkeypatch, mock_subprocess_run):
        monkeypatch.setenv("NOTES_MCP_AUTO_REFRESH", "1")
        try:
            assert cache.start_background_refresh(interval_s=1.0) is True
            assert cache._bg_thread is not None
            assert cache._bg_thread.is_alive()
        finally:
            cache.stop_background_refresh(join_timeout_s=2.0)

    @pytest.mark.slow
    def test_double_start_returns_false(self, monkeypatch, mock_subprocess_run):
        monkeypatch.setenv("NOTES_MCP_AUTO_REFRESH", "1")
        try:
            assert cache.start_background_refresh(interval_s=1.0) is True
            assert cache.start_background_refresh(interval_s=1.0) is False
        finally:
            cache.stop_background_refresh(join_timeout_s=2.0)

    @pytest.mark.slow
    def test_stop_joins_thread(self, monkeypatch, mock_subprocess_run):
        monkeypatch.setenv("NOTES_MCP_AUTO_REFRESH", "1")
        cache.start_background_refresh(interval_s=1.0)
        assert cache._bg_thread is not None
        cache.stop_background_refresh(join_timeout_s=2.0)
        assert cache._bg_thread is None

    def test_stop_when_not_running_is_safe(self):
        # No thread; should be a graceful no-op.
        cache.stop_background_refresh(join_timeout_s=0.5)
        assert cache._bg_thread is None

    def test_interval_clamped_to_min_1s(self, monkeypatch, mock_subprocess_run):
        monkeypatch.setenv("NOTES_MCP_AUTO_REFRESH", "1")
        try:
            assert cache.start_background_refresh(interval_s=0.1) is True
            # Hard to inspect internal interval, but starting must succeed.
            assert cache._bg_thread is not None
        finally:
            cache.stop_background_refresh(join_timeout_s=2.0)


# ---------------------------------------------------------------------------
# background_refresh_status
# ---------------------------------------------------------------------------

class TestBackgroundRefreshStatus:
    EXPECTED_KEYS = {
        "running",
        "tick_count",
        "skip_count",
        "last_refresh_ms",
        "idle_for_s",
        "idle_threshold_s",
        "is_idle_paused",
        "last_data_version",
    }

    def test_status_shape_when_not_running(self):
        status = cache.background_refresh_status()
        assert set(status.keys()) == self.EXPECTED_KEYS
        assert status["running"] is False
        assert status["tick_count"] == 0
        assert status["skip_count"] == 0

    def test_idle_paused_reflects_threshold(self, frozen_monotonic):
        cache._idle_threshold_s = 10.0
        cache.mark_activity()
        frozen_monotonic.advance(50)
        status = cache.background_refresh_status()
        assert status["is_idle_paused"] is True
        assert status["idle_for_s"] >= 50.0

    def test_idle_disabled_when_threshold_zero(self, frozen_monotonic):
        cache._idle_threshold_s = 0.0
        cache.mark_activity()
        frozen_monotonic.advance(1000)
        status = cache.background_refresh_status()
        assert status["is_idle_paused"] is False


# ---------------------------------------------------------------------------
# prewarm() / refresh()
# ---------------------------------------------------------------------------

class TestPrewarm:
    def test_returns_true_on_success(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 0
        assert cache.prewarm() is True

    def test_returns_false_on_timeout(self, mocker):
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=30.0),
        )
        assert cache.prewarm() is False

    def test_returns_false_on_nonzero_rc(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 1
        mock_subprocess_run.return_value.stderr = "permission denied"
        assert cache.prewarm() is False

    def test_returns_false_on_arbitrary_exception(self, mocker):
        mocker.patch("subprocess.run", side_effect=OSError("bang"))
        assert cache.prewarm() is False


class TestSyncAfterWrite:
    def test_swallows_subprocess_errors(self, mocker):
        mocker.patch("subprocess.run", side_effect=RuntimeError("boom"))
        # Must not raise
        cache.sync_after_write(timeout_s=1.0)

    def test_calls_subprocess_run(self, mock_subprocess_run):
        cache.sync_after_write(timeout_s=1.0)
        assert mock_subprocess_run.called


class TestNotesRunning:
    def test_pgrep_zero_means_running(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 0
        assert cache._notes_running() is True

    def test_pgrep_nonzero_means_not_running(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 1
        assert cache._notes_running() is False

    def test_pgrep_exception_assumes_running(self, mocker):
        """Defensive default: if pgrep fails, return True so we don't silently
        skip ticks forever."""
        mocker.patch("subprocess.run", side_effect=OSError("no pgrep"))
        assert cache._notes_running() is True


class TestRefresh:
    def test_success_returns_ok_true(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 0
        result = cache.refresh()
        assert result["ok"] is True
        assert isinstance(result["ms"], int)
        assert result["ms"] >= 0

    def test_failure_returns_ok_false_with_error(self, mock_subprocess_run):
        mock_subprocess_run.return_value.returncode = 2
        mock_subprocess_run.return_value.stderr = "boom"
        result = cache.refresh()
        assert result["ok"] is False
        assert result.get("error")

    def test_exception_returns_ok_false(self, mocker):
        mocker.patch("subprocess.run", side_effect=RuntimeError("kaboom"))
        result = cache.refresh()
        assert result["ok"] is False
        assert "kaboom" in (result.get("error") or "")
        assert isinstance(result["ms"], int)


# ---------------------------------------------------------------------------
# recover_bridge — Bug hotspot #14 (cooldown + race)
# ---------------------------------------------------------------------------

class TestRecoverBridge:
    def _stub_subprocess_for_recovery_success(self, mocker):
        """Stub subprocess.run so recover_bridge sees:
        - quit succeeds
        - pgrep returns non-zero (Notes not running) -> exits the wait loop quickly
        - open succeeds
        - osascript count of accounts returns "1\n" (responsive)
        """
        def _fake_run(args, **_kwargs):
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            p = P()
            if args[0] == "pgrep":
                p.returncode = 1  # Notes NOT running -> break out of wait
            elif args[0] == "osascript" and "accounts" in (args[-1] if args else ""):
                p.returncode = 0
                p.stdout = "1\n"
            elif args[0] == "open":
                p.returncode = 0
            else:
                p.returncode = 0
            return p

        return mocker.patch("subprocess.run", side_effect=_fake_run)

    def test_first_call_succeeds(self, mocker, frozen_monotonic):
        self._stub_subprocess_for_recovery_success(mocker)
        # Avoid the real time.sleep stalls in the loop
        mocker.patch("apple_notes_brain.cache.time.sleep")
        assert cache.recover_bridge(timeout_s=2.0) is True

    def test_second_call_within_cooldown_returns_false(
        self, mocker, frozen_monotonic
    ):
        self._stub_subprocess_for_recovery_success(mocker)
        mocker.patch("apple_notes_brain.cache.time.sleep")
        assert cache.recover_bridge(timeout_s=2.0) is True
        # Still within cooldown -> rate-limited.
        assert cache.recover_bridge(timeout_s=2.0) is False

    def test_call_after_cooldown_works_again(self, mocker, frozen_monotonic):
        self._stub_subprocess_for_recovery_success(mocker)
        mocker.patch("apple_notes_brain.cache.time.sleep")
        assert cache.recover_bridge(timeout_s=2.0) is True
        # Advance past the 60s cooldown.
        frozen_monotonic.advance(61)
        assert cache.recover_bridge(timeout_s=2.0) is True

    def test_open_failure_returns_false(self, mocker, frozen_monotonic):
        """If `open -a Notes` fails, recover_bridge bails out early with False."""
        def _fake_run(args, **_kwargs):
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            p = P()
            if args[0] == "pgrep":
                p.returncode = 1  # not running
            elif args[0] == "open":
                raise OSError("open failed")
            return p

        mocker.patch("subprocess.run", side_effect=_fake_run)
        mocker.patch("apple_notes_brain.cache.time.sleep")
        assert cache.recover_bridge(timeout_s=2.0) is False

    def test_unresponsive_after_relaunch_returns_false(
        self, mocker, frozen_monotonic
    ):
        """If osascript count-accounts never returns a digit, recover_bridge
        eventually times out and returns False."""
        def _fake_run(args, **_kwargs):
            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            p = P()
            if args[0] == "pgrep":
                p.returncode = 1
            elif args[0] == "open":
                p.returncode = 0
            elif args[0] == "osascript":
                # Never become responsive: rc=0 but non-numeric stdout.
                p.returncode = 0
                p.stdout = "not-a-number\n"
            return p

        mocker.patch("subprocess.run", side_effect=_fake_run)
        # Patch sleep to advance the frozen clock so the loop bails fast.
        def _advance_sleep(s):
            frozen_monotonic.advance(s)
        mocker.patch("apple_notes_brain.cache.time.sleep", side_effect=_advance_sleep)
        assert cache.recover_bridge(timeout_s=2.0) is False

    def test_concurrent_calls_only_one_runs(self, mocker, frozen_monotonic):
        """Bug hotspot #14: spawn two threads calling recover_bridge.

        One should run; the other (entering after the first sets
        _recover_last_attempt while still holding the lock) should be skipped
        by the cooldown. _recover_lock serialises, then cooldown gates.
        """
        self._stub_subprocess_for_recovery_success(mocker)
        mocker.patch("apple_notes_brain.cache.time.sleep")

        results: list[bool] = []
        results_lock = threading.Lock()

        def worker():
            r = cache.recover_bridge(timeout_s=2.0)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=4.0)

        assert sorted(results) == [False, True], (
            f"expected exactly one True and one False, got {results!r}. "
            "If both are True, the cooldown gate inside _recover_lock failed "
            "to suppress the second call (real race condition)."
        )
