"""
Tests for FeatureManager - lifecycle and delegation for KZA features.

Tests ensure that:
1. Initialization works with all deps and with no deps
2. start() calls start on all non-None subsystems
3. stop() calls stop on all non-None subsystems
4. get_status() returns proper dict
5. Passthrough methods delegate correctly
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest
import yaml

from src.alerts import AlertManager, AlertPriority, AlertScheduler, AlertType, CheckType
from src.pipeline.feature_manager import ALERTS_NONE, ALERTS_UNAVAILABLE, FeatureManager, build_feature_manager

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


# ============================================================
# Helpers
# ============================================================

def _make_timer_manager():
    """Create a mock NamedTimerManager."""
    m = MagicMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.get_status.return_value = {"active_timers": 0}
    m.get_active_timers.return_value = []
    # create_timer returns a mock NamedTimer
    mock_timer = MagicMock()
    mock_timer.name = "pasta"
    mock_timer.timer_id = "t1"
    m.create_timer.return_value = mock_timer
    m._find_timer_by_name.return_value = mock_timer
    m.cancel_timer.return_value = True
    return m


def _make_intercom():
    """Create a mock IntercomSystem."""
    m = MagicMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.get_status.return_value = {"zones": 3}
    mock_announcement = MagicMock()
    mock_announcement.announcement_id = "ann-1"
    m.announce = AsyncMock(return_value=mock_announcement)
    m.announce_emergency = AsyncMock(return_value=mock_announcement)
    m.get_zones.return_value = {"kitchen": "Cocina", "living_room": "Sala"}
    m.get_history.return_value = []
    return m


def _make_notifications():
    """Create a mock SmartNotificationManager."""
    m = MagicMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.get_status.return_value = {"pending": 0}
    mock_notification = MagicMock()
    mock_notification.notification_id = "notif-1"
    m.notify = AsyncMock(return_value=mock_notification)
    m.set_dnd = MagicMock()
    m.is_dnd_active.return_value = False
    m.get_history.return_value = []
    return m


def _make_alert_manager():
    """Create a mock AlertManager restricted to its real interface (no phantom calls)."""
    m = MagicMock(spec=AlertManager)
    m.get_active_alerts.return_value = []
    return m


def _make_alert_scheduler(handlers: int = 1, running: bool = True):
    """Create a mock AlertScheduler restricted to its real interface."""
    m = MagicMock(spec=AlertScheduler)
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.get_status.return_value = {
        "running": running, "checks": {},
        "registered_handlers": {"security": handlers, "pattern": 0, "device": 0},
    }
    return m


def _make_ha_integration():
    """Create a mock KZAHomeAssistantIntegration."""
    m = MagicMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    return m


def _make_briefing():
    """Create a mock MorningBriefing."""
    m = MagicMock()
    m.get_status.return_value = {"users": 1}
    return m


def _make_all_deps():
    """Return a dict of all mocked dependencies."""
    return {
        "timer_manager": _make_timer_manager(),
        "intercom": _make_intercom(),
        "notifications": _make_notifications(),
        "alert_manager": _make_alert_manager(),
        "alert_scheduler": _make_alert_scheduler(),
        "ha_integration": _make_ha_integration(),
        "briefing": _make_briefing(),
    }


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def all_deps():
    return _make_all_deps()


@pytest.fixture
def fm_full(all_deps):
    """FeatureManager with all dependencies."""
    return FeatureManager(**all_deps)


@pytest.fixture
def fm_empty():
    """FeatureManager with no dependencies."""
    return FeatureManager()


# ============================================================
# Init tests
# ============================================================

class TestFeatureManagerInit:
    """Initialization tests."""

    def test_init_with_all_deps(self, fm_full, all_deps):
        """All subsystems should be stored."""
        assert fm_full.timer_manager is all_deps["timer_manager"]
        assert fm_full.intercom is all_deps["intercom"]
        assert fm_full.notifications is all_deps["notifications"]
        assert fm_full.alert_manager is all_deps["alert_manager"]
        assert fm_full.alert_scheduler is all_deps["alert_scheduler"]
        assert fm_full.ha_integration is all_deps["ha_integration"]
        assert fm_full.briefing is all_deps["briefing"]
        assert fm_full._running is False

    def test_init_with_no_deps(self, fm_empty):
        """All subsystems should be None."""
        assert fm_empty.timer_manager is None
        assert fm_empty.intercom is None
        assert fm_empty.notifications is None
        assert fm_empty.alert_manager is None
        assert fm_empty.alert_scheduler is None
        assert fm_empty.ha_integration is None
        assert fm_empty.briefing is None
        assert fm_empty._running is False

    def test_init_with_partial_deps(self):
        """Only specified subsystems should be set."""
        tm = _make_timer_manager()
        fm = FeatureManager(timer_manager=tm)
        assert fm.timer_manager is tm
        assert fm.intercom is None


# ============================================================
# Lifecycle tests
# ============================================================

class TestFeatureManagerLifecycle:
    """start() and stop() tests."""

    @pytest.mark.asyncio
    async def test_start_calls_all_subsystems(self, fm_full, all_deps):
        """start() should call start on every non-None startable subsystem.

        Note: alert_scheduler.start() is dispatched as a background task
        (AlertScheduler.start() contains an awaited asyncio.gather over
        infinite check loops, so awaiting it directly would block the
        pipeline forever). The task is created but we don't assert it
        was awaited — the create_task is enough proof of invocation.
        """
        import asyncio
        await fm_full.start()

        all_deps["timer_manager"].start.assert_awaited_once()
        all_deps["intercom"].start.assert_awaited_once()
        all_deps["notifications"].start.assert_awaited_once()
        all_deps["ha_integration"].start.assert_awaited_once()
        assert fm_full._running is True

        # alert_scheduler.start() was scheduled as a background task
        assert fm_full._alert_scheduler_task is not None
        assert isinstance(fm_full._alert_scheduler_task, asyncio.Task)
        # Cancel it to keep the test tidy (AsyncMock yields immediately,
        # but being explicit avoids warnings if the mock changes).
        fm_full._alert_scheduler_task.cancel()

    @pytest.mark.asyncio
    async def test_start_with_no_deps(self, fm_empty):
        """start() should not raise when all deps are None."""
        await fm_empty.start()
        assert fm_empty._running is True

    @pytest.mark.asyncio
    async def test_start_idempotent(self, fm_full, all_deps):
        """Calling start() twice should only start subsystems once."""
        await fm_full.start()
        await fm_full.start()
        all_deps["timer_manager"].start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_calls_all_subsystems(self, fm_full, all_deps):
        """stop() should call stop on every non-None stoppable subsystem."""
        await fm_full.start()
        await fm_full.stop()

        all_deps["timer_manager"].stop.assert_awaited_once()
        all_deps["intercom"].stop.assert_awaited_once()
        all_deps["notifications"].stop.assert_awaited_once()
        all_deps["alert_scheduler"].stop.assert_awaited_once()
        all_deps["ha_integration"].stop.assert_awaited_once()
        assert fm_full._running is False

    @pytest.mark.asyncio
    async def test_stop_with_no_deps(self, fm_empty):
        """stop() should not raise when all deps are None."""
        await fm_empty.start()
        await fm_empty.stop()
        assert fm_empty._running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self, fm_full, all_deps):
        """stop() should be a no-op if not running."""
        await fm_full.stop()
        all_deps["timer_manager"].stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_partial_deps(self):
        """start() should only call start on provided deps."""
        tm = _make_timer_manager()
        ic = _make_intercom()
        fm = FeatureManager(timer_manager=tm, intercom=ic)
        await fm.start()

        tm.start.assert_awaited_once()
        ic.start.assert_awaited_once()
        assert fm._running is True


# ============================================================
# Status tests
# ============================================================

class TestFeatureManagerStatus:
    """get_status() tests."""

    def test_get_status_with_all_deps(self, fm_full):
        """Status should include all subsystem sections."""
        status = fm_full.get_status()

        assert "timers" in status
        assert "intercom" in status
        assert "notifications" in status
        assert "alerts" in status
        assert "briefings" in status
        assert "ha_integration" in status

        assert status["timers"]["status"] == {"active_timers": 0}
        assert status["intercom"]["status"] == {"zones": 3}
        assert status["notifications"]["status"] == {"pending": 0}
        assert status["alerts"] == {"available": True, "active": 0}
        assert status["briefings"]["enabled"] is True
        assert status["ha_integration"]["enabled"] is True

    def test_get_status_with_no_deps(self, fm_empty):
        """Status should return safe defaults when no deps."""
        status = fm_empty.get_status()

        assert status["timers"]["status"] is None
        assert status["intercom"]["status"] is None
        assert status["notifications"]["status"] is None
        assert status["alerts"] == {"available": False, "active": None}
        assert status["briefings"]["enabled"] is False
        assert status["ha_integration"]["enabled"] is False


# ============================================================
# Timer passthrough tests
# ============================================================

class TestFeatureManagerTimers:
    """Timer passthrough tests."""

    def test_create_timer(self, fm_full, all_deps):
        """create_timer should delegate to timer_manager."""
        result = fm_full.create_timer("pasta", 480, user_id="u1", zone_id="kitchen")
        all_deps["timer_manager"].create_timer.assert_called_once_with(
            "pasta", 480, "u1", "kitchen"
        )
        assert result.name == "pasta"

    def test_create_timer_no_manager(self, fm_empty):
        """create_timer should return None when no timer_manager."""
        result = fm_empty.create_timer("test", 60)
        assert result is None

    def test_get_active_timers(self, fm_full, all_deps):
        """get_active_timers should delegate."""
        result = fm_full.get_active_timers()
        all_deps["timer_manager"].get_active_timers.assert_called_once()
        assert result == []

    def test_get_active_timers_no_manager(self, fm_empty):
        """get_active_timers should return empty list when no manager."""
        assert fm_empty.get_active_timers() == []

    def test_cancel_timer(self, fm_full, all_deps):
        """cancel_timer should find by name then cancel by id."""
        result = fm_full.cancel_timer("pasta")
        all_deps["timer_manager"]._find_timer_by_name.assert_called_once_with("pasta")
        all_deps["timer_manager"].cancel_timer.assert_called_once_with("t1")
        assert result is True

    def test_cancel_timer_not_found(self, fm_full, all_deps):
        """cancel_timer should return False if name not found."""
        all_deps["timer_manager"]._find_timer_by_name.return_value = None
        result = fm_full.cancel_timer("nonexistent")
        assert result is False

    def test_cancel_timer_no_manager(self, fm_empty):
        """cancel_timer should return False when no manager."""
        assert fm_empty.cancel_timer("test") is False

    def test_get_timer_status(self, fm_full, all_deps):
        """get_timer_status should delegate."""
        result = fm_full.get_timer_status()
        assert result == {"active_timers": 0}

    def test_get_timer_status_no_manager(self, fm_empty):
        """get_timer_status should return empty dict when no manager."""
        assert fm_empty.get_timer_status() == {}


# ============================================================
# Intercom passthrough tests
# ============================================================

class TestFeatureManagerIntercom:
    """Intercom passthrough tests."""

    @pytest.mark.asyncio
    async def test_announce(self, fm_full, all_deps):
        """announce should delegate to intercom."""
        result = await fm_full.announce("Cena lista", zones=["kitchen"], priority="high")
        all_deps["intercom"].announce.assert_awaited_once()
        assert result["success"] is True
        assert result["announcement_id"] == "ann-1"

    @pytest.mark.asyncio
    async def test_announce_no_intercom(self, fm_empty):
        """announce should return error when no intercom."""
        result = await fm_empty.announce("test")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_announce_emergency(self, fm_full, all_deps):
        """announce_emergency should delegate."""
        result = await fm_full.announce_emergency("Fire!")
        all_deps["intercom"].announce_emergency.assert_awaited_once_with("Fire!")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_announce_emergency_no_intercom(self, fm_empty):
        """announce_emergency should return error when no intercom."""
        result = await fm_empty.announce_emergency("test")
        assert result["success"] is False

    def test_get_intercom_zones(self, fm_full, all_deps):
        """get_intercom_zones should delegate."""
        result = fm_full.get_intercom_zones()
        assert "kitchen" in result

    def test_get_intercom_zones_no_intercom(self, fm_empty):
        """get_intercom_zones should return empty dict."""
        assert fm_empty.get_intercom_zones() == {}


# ============================================================
# Notification passthrough tests
# ============================================================

class TestFeatureManagerNotifications:
    """Notification passthrough tests."""

    @pytest.mark.asyncio
    async def test_send_notification(self, fm_full, all_deps):
        """send_notification should delegate to notifications."""
        result = await fm_full.send_notification(
            "Hello", user_id="u1", title="Greet", priority="high"
        )
        all_deps["notifications"].notify.assert_awaited_once()
        assert result["success"] is True
        assert result["notification_id"] == "notif-1"

    @pytest.mark.asyncio
    async def test_send_notification_no_manager(self, fm_empty):
        """send_notification should return error when not configured."""
        result = await fm_empty.send_notification("test")
        assert result["success"] is False

    def test_set_do_not_disturb(self, fm_full, all_deps):
        """set_do_not_disturb should delegate."""
        fm_full.set_do_not_disturb("u1", True)
        all_deps["notifications"].set_dnd.assert_called_once_with("u1", True)

    def test_set_do_not_disturb_no_manager(self, fm_empty):
        """set_do_not_disturb should not raise when no manager."""
        fm_empty.set_do_not_disturb("u1", True)  # should not raise

    def test_is_do_not_disturb(self, fm_full, all_deps):
        """is_do_not_disturb should delegate."""
        result = fm_full.is_do_not_disturb("u1")
        all_deps["notifications"].is_dnd_active.assert_called_once_with("u1")
        assert result is False

    def test_is_do_not_disturb_no_manager(self, fm_empty):
        """is_do_not_disturb should return False when no manager."""
        assert fm_empty.is_do_not_disturb("u1") is False


# ============================================================
# Alert passthrough tests
# ============================================================

class TestFeatureManagerAlerts:
    """Alert passthrough tests."""

    def test_phantom_alert_methods_removed(self, fm_full):
        """No FeatureManager method may call an API the alert classes do not have."""
        assert not hasattr(fm_full, "add_alert_condition")
        assert not hasattr(fm_full, "acknowledge_alert")
        for name in ("get_pending_alerts", "acknowledge", "get_summary"):
            assert not hasattr(AlertManager, name)
        assert not hasattr(AlertScheduler, "add_check_simple")

    def test_get_active_alerts(self, fm_full, all_deps):
        """get_active_alerts should delegate to the manager's real method."""
        all_deps["alert_manager"].get_active_alerts.return_value = ["a"]
        assert fm_full.get_active_alerts() == ["a"]
        all_deps["alert_manager"].get_active_alerts.assert_called_once_with()

    def test_get_active_alerts_no_manager(self, fm_empty):
        """get_active_alerts should return empty list."""
        assert fm_empty.get_active_alerts() == []

    def test_get_alert_summary_no_manager_is_unavailable(self, fm_empty):
        """Without a manager the summary never says there are no alerts."""
        text = fm_empty.get_alert_summary()
        assert text == ALERTS_UNAVAILABLE
        assert "no está disponible" in text
        assert "no hay alertas" not in text.lower()

    def test_get_alert_summary_without_scheduler_is_unavailable(self):
        """A manager without producers cannot know there are no alerts."""
        fm = FeatureManager(alert_manager=_make_alert_manager())
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
        assert fm.alerts_available() is False
        assert fm.get_status()["alerts"] == {"available": False, "active": None}

    def test_get_alert_summary_without_producers_is_unavailable(self):
        """A scheduler with no registered producer is not a source of truth."""
        fm = FeatureManager(alert_manager=_make_alert_manager(), alert_scheduler=_make_alert_scheduler(0))
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
        assert fm.alerts_available() is False

    def test_get_alert_summary_no_alerts_with_producers(self, fm_full):
        """Only a connected producer allows 'no hay alertas activas'."""
        assert fm_full.alerts_available() is True
        assert fm_full.get_alert_summary() == ALERTS_NONE == "No hay alertas activas."

    def test_get_alert_summary_one_alert(self, fm_full, all_deps):
        all_deps["alert_manager"].get_active_alerts.return_value = [MagicMock(message="Puerta de entrada abierta")]
        assert fm_full.get_alert_summary() == "Hay una alerta activa: Puerta de entrada abierta."

    def test_get_alert_summary_lists_three_and_counts_rest(self, fm_full, all_deps):
        all_deps["alert_manager"].get_active_alerts.return_value = [
            MagicMock(message=f"m{i}") for i in range(5)]
        assert fm_full.get_alert_summary() == "Hay 5 alertas activas: m0; m1; m2; y 2 más."

    def test_get_alert_summary_three_without_rest(self, fm_full, all_deps):
        all_deps["alert_manager"].get_active_alerts.return_value = [
            MagicMock(message=f"m{i}") for i in range(3)]
        assert fm_full.get_alert_summary() == "Hay 3 alertas activas: m0; m1; m2."

    def test_status_counts_active_alerts(self, fm_full, all_deps):
        all_deps["alert_manager"].get_active_alerts.return_value = ["a", "b"]
        assert fm_full.get_status()["alerts"] == {"available": True, "active": 2}

    @pytest.mark.asyncio
    async def test_summary_with_real_manager_skips_resolved_and_acknowledged(self):
        manager = AlertManager()
        fm = FeatureManager(alert_manager=manager, alert_scheduler=_make_alert_scheduler())
        door = await manager.create_alert(AlertType.SECURITY, AlertPriority.LOW, "Puerta abierta")
        assert fm.get_alert_summary() == "Hay una alerta activa: Puerta abierta."
        await manager.mark_processed(door.alert_id)
        assert fm.get_alert_summary() == ALERTS_NONE

    @pytest.mark.parametrize("running", [False, None, "yes"])
    def test_producers_without_running_scheduler_are_unavailable(self, running):
        """Review R2: registered producers do not count while the scheduler is not running."""
        fm = FeatureManager(alert_manager=_make_alert_manager(),
                            alert_scheduler=_make_alert_scheduler(handlers=2, running=running))
        assert fm.alerts_available() is False
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
        assert fm.get_status()["alerts"] == {"available": False, "active": None}

    @pytest.mark.asyncio
    async def test_summary_with_real_scheduler_needs_registered_producer(self):
        manager = AlertManager()
        scheduler = AlertScheduler(alert_manager=manager)
        fm = FeatureManager(alert_manager=manager, alert_scheduler=scheduler)
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0)
        try:
            assert scheduler.get_status()["running"] is True
            assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
            scheduler.register_device_handler(AsyncMock())
            assert fm.get_alert_summary() == ALERTS_NONE
        finally:
            await scheduler.stop()
            await task

    @pytest.mark.asyncio
    async def test_real_scheduler_with_producer_never_started_or_stopped_is_unavailable(self):
        """Review R2: before start() and after stop() the summary never says 'no hay alertas'."""
        manager = AlertManager()
        scheduler = AlertScheduler(alert_manager=manager)
        scheduler.register_device_handler(AsyncMock())
        fm = FeatureManager(alert_manager=manager, alert_scheduler=scheduler)
        assert scheduler.get_status()["running"] is False
        assert fm.alerts_available() is False
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
        assert fm.get_status()["alerts"] == {"available": False, "active": None}
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0)
        assert fm.get_alert_summary() == ALERTS_NONE
        await scheduler.stop()
        await task
        assert scheduler.get_status()["running"] is False
        assert fm.alerts_available() is False
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE
        assert fm.get_status()["alerts"] == {"available": False, "active": None}

    @pytest.mark.asyncio
    async def test_scheduler_loop_that_exits_on_error_is_unavailable(self, monkeypatch):
        """Review R2: start() ending on an exception leaves the alerts unavailable."""
        manager = AlertManager()
        scheduler = AlertScheduler(alert_manager=manager)
        scheduler.register_device_handler(AsyncMock())
        fm = FeatureManager(alert_manager=manager, alert_scheduler=scheduler)

        async def boom(check_type, _interval):
            if check_type is CheckType.DEVICE:
                raise RuntimeError("loop died")

        monkeypatch.setattr(scheduler, "_run_periodic_check", boom)
        await scheduler.start()
        assert scheduler.get_status()["running"] is False
        assert fm.get_alert_summary() == ALERTS_UNAVAILABLE


class TestBuildFeatureManagerAlerts:
    """Alert wiring from the existing alerts section of settings."""

    def _build(self, alerts: dict):
        handler = MagicMock()
        handler.aspeak_to_all_zones = AsyncMock()
        return build_feature_manager({"alerts": alerts}, response_handler=handler), handler

    def test_reannounce_interval_default_is_one_notice_per_episode(self):
        features, _ = self._build({})
        assert features.alert_manager.reannounce_interval_seconds is None

    def test_real_settings_keep_one_notice_per_episode(self):
        """Review R3: the shipped config/settings.yaml honours Q20 (one notice per episode)."""
        settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
        general = settings["alerts"]["general"]
        assert "reannounce_interval_seconds" in general
        assert general["reannounce_interval_seconds"] is None
        features, _ = self._build(settings["alerts"])
        assert features.alert_manager is not None
        assert features.alert_manager.reannounce_interval_seconds is None

    def test_reannounce_interval_from_config(self):
        features, handler = self._build({"general": {"reannounce_interval_seconds": 900}})
        assert features.alert_manager.reannounce_interval_seconds == 900
        assert features.alert_manager.tts_callback == handler.aspeak_to_all_zones

    def test_voice_notifications_off_mutes(self):
        features, _ = self._build({"general": {"voice_notifications": False}})
        assert features.alert_manager.tts_callback is None

    def test_production_wiring_without_producers_is_unavailable(self):
        """Today's graph registers no producer: the summary must say unavailable (CORE-05)."""
        features, _ = self._build({})
        assert features.alert_scheduler is not None
        assert features.get_alert_summary() == ALERTS_UNAVAILABLE
