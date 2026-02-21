"""
Tests for FeatureManager - lifecycle and delegation for KZA features.

Tests ensure that:
1. Initialization works with all deps and with no deps
2. start() calls start on all non-None subsystems
3. stop() calls stop on all non-None subsystems
4. get_status() returns proper dict
5. Passthrough methods delegate correctly
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules.setdefault('sounddevice', MagicMock())
sys.modules.setdefault('soundfile', MagicMock())
sys.modules.setdefault('pyaudio', MagicMock())
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch.cuda', MagicMock())

import pytest

from src.pipeline.feature_manager import FeatureManager


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
    """Create a mock AlertManager."""
    m = MagicMock()
    m.get_pending_alerts.return_value = []
    m.acknowledge.return_value = True
    m.get_summary.return_value = "No hay alertas activas"
    return m


def _make_alert_scheduler():
    """Create a mock AlertScheduler."""
    m = MagicMock()
    m.start = AsyncMock()
    m.stop = AsyncMock()
    m.add_check_simple = MagicMock()
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
        """start() should call start on every non-None startable subsystem."""
        await fm_full.start()

        all_deps["timer_manager"].start.assert_awaited_once()
        all_deps["intercom"].start.assert_awaited_once()
        all_deps["notifications"].start.assert_awaited_once()
        all_deps["alert_scheduler"].start.assert_awaited_once()
        all_deps["ha_integration"].start.assert_awaited_once()
        assert fm_full._running is True

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
        assert status["alerts"]["active"] == 0
        assert status["briefings"]["enabled"] is True
        assert status["ha_integration"]["enabled"] is True

    def test_get_status_with_no_deps(self, fm_empty):
        """Status should return safe defaults when no deps."""
        status = fm_empty.get_status()

        assert status["timers"]["status"] is None
        assert status["intercom"]["status"] is None
        assert status["notifications"]["status"] is None
        assert status["alerts"]["active"] == 0
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

    def test_add_alert_condition(self, fm_full, all_deps):
        """add_alert_condition should delegate to alert_scheduler."""
        fm_full.add_alert_condition(
            "binary_sensor.garage_door",
            state_equals="on",
            duration_minutes=30,
            message="Garage open for {duration}",
        )
        all_deps["alert_scheduler"].add_check_simple.assert_called_once_with(
            entity_id="binary_sensor.garage_door",
            state="on",
            duration_minutes=30,
            message="Garage open for {duration}",
        )

    def test_add_alert_condition_no_scheduler(self, fm_empty):
        """add_alert_condition should not raise when no scheduler."""
        fm_empty.add_alert_condition("sensor.x", state_equals="on")

    def test_get_active_alerts(self, fm_full, all_deps):
        """get_active_alerts should delegate."""
        result = fm_full.get_active_alerts()
        all_deps["alert_manager"].get_pending_alerts.assert_called_once()
        assert result == []

    def test_get_active_alerts_no_manager(self, fm_empty):
        """get_active_alerts should return empty list."""
        assert fm_empty.get_active_alerts() == []

    def test_acknowledge_alert(self, fm_full, all_deps):
        """acknowledge_alert should delegate."""
        result = fm_full.acknowledge_alert("a1", user_id="u1")
        all_deps["alert_manager"].acknowledge.assert_called_once_with("a1")
        assert result is True

    def test_acknowledge_alert_no_manager(self, fm_empty):
        """acknowledge_alert should return False when no manager."""
        assert fm_empty.acknowledge_alert("a1") is False

    def test_get_alert_summary(self, fm_full, all_deps):
        """get_alert_summary should delegate."""
        result = fm_full.get_alert_summary()
        assert result == "No hay alertas activas"

    def test_get_alert_summary_no_manager(self, fm_empty):
        """get_alert_summary should return default message."""
        assert fm_empty.get_alert_summary() == "No hay alertas activas"
