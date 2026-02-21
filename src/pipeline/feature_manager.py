"""
Feature Manager Module
Manages lifecycle and delegation for KZA's differentiating features:
timers, intercom, notifications, alerts, briefings, HA integration.

Extracted from VoicePipeline to reduce its responsibility.
"""

import logging
from typing import Optional

from src.timers import NamedTimerManager, NamedTimer
from src.intercom import IntercomSystem, AnnouncementPriority
from src.notifications import SmartNotificationManager, NotificationPriority
from src.alerts import AlertManager, AlertScheduler

logger = logging.getLogger(__name__)


class FeatureManager:
    """Owns lifecycle and delegation for KZA differentiating features.

    Manages timer, intercom, notification, alert, briefing, and
    Home Assistant integration subsystems.  Each dependency is optional
    so the system degrades gracefully when a subsystem is not configured.

    Args:
        timer_manager: Named timer subsystem.
        intercom: Intercom / announcement subsystem.
        notifications: Smart notification subsystem.
        alert_manager: Alert state holder.
        alert_scheduler: Alert check scheduler.
        ha_integration: KZA Home Assistant integration.
        briefing: Morning briefing subsystem.
    """

    def __init__(
        self,
        timer_manager: Optional[NamedTimerManager] = None,
        intercom: Optional[IntercomSystem] = None,
        notifications: Optional[SmartNotificationManager] = None,
        alert_manager: Optional[AlertManager] = None,
        alert_scheduler: Optional[AlertScheduler] = None,
        ha_integration=None,
        briefing=None,
    ):
        self.timer_manager = timer_manager
        self.intercom = intercom
        self.notifications = notifications
        self.alert_manager = alert_manager
        self.alert_scheduler = alert_scheduler
        self.ha_integration = ha_integration
        self.briefing = briefing
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all non-None subsystems."""
        if self._running:
            logger.warning("FeatureManager already running")
            return

        if self.timer_manager:
            await self.timer_manager.start()
            logger.info("Timer manager started")

        if self.intercom:
            await self.intercom.start()
            logger.info("Intercom started")

        if self.notifications:
            await self.notifications.start()
            logger.info("Notifications started")

        if self.alert_scheduler:
            await self.alert_scheduler.start()
            logger.info("Alert scheduler started")

        if self.ha_integration:
            await self.ha_integration.start()
            logger.info("HA integration started")

        self._running = True
        logger.info("FeatureManager started")

    async def stop(self) -> None:
        """Stop all non-None subsystems."""
        if not self._running:
            return

        if self.timer_manager:
            await self.timer_manager.stop()
            logger.info("Timer manager stopped")

        if self.intercom:
            await self.intercom.stop()
            logger.info("Intercom stopped")

        if self.notifications:
            await self.notifications.stop()
            logger.info("Notifications stopped")

        if self.alert_scheduler:
            await self.alert_scheduler.stop()
            logger.info("Alert scheduler stopped")

        if self.ha_integration:
            await self.ha_integration.stop()
            logger.info("HA integration stopped")

        self._running = False
        logger.info("FeatureManager stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return aggregated status dict for all subsystems.

        Returns:
            Dictionary with keys for each subsystem and their status.
        """
        status: dict = {}

        if self.timer_manager:
            status["timers"] = {"status": self.timer_manager.get_status()}
        else:
            status["timers"] = {"status": None}

        if self.intercom:
            status["intercom"] = {"status": self.intercom.get_status()}
        else:
            status["intercom"] = {"status": None}

        if self.notifications:
            status["notifications"] = {"status": self.notifications.get_status()}
        else:
            status["notifications"] = {"status": None}

        if self.alert_manager and hasattr(self.alert_manager, "get_pending_alerts"):
            status["alerts"] = {
                "active": len(self.alert_manager.get_pending_alerts())
            }
        else:
            status["alerts"] = {"active": 0}

        if self.briefing:
            status["briefings"] = {
                "enabled": True,
                "status": self.briefing.get_status() if hasattr(self.briefing, "get_status") else None,
            }
        else:
            status["briefings"] = {"enabled": False, "status": None}

        if self.ha_integration:
            status["ha_integration"] = {"enabled": True}
        else:
            status["ha_integration"] = {"enabled": False}

        return status

    # ------------------------------------------------------------------
    # Timer passthroughs
    # ------------------------------------------------------------------

    def create_timer(
        self,
        name: str,
        duration_seconds: int,
        user_id: str = None,
        zone_id: str = "default",
    ) -> Optional[NamedTimer]:
        """Create a named timer.

        Args:
            name: Descriptive name for the timer (e.g. "pasta").
            duration_seconds: Timer length in seconds.
            user_id: Owner user id.
            zone_id: Zone where the timer was created.

        Returns:
            The created NamedTimer, or None if timer_manager is unavailable.
        """
        if not self.timer_manager:
            logger.warning("create_timer called but timer_manager is None")
            return None
        return self.timer_manager.create_timer(name, duration_seconds, user_id, zone_id)

    def get_active_timers(self) -> list:
        """Return list of active NamedTimer instances."""
        if not self.timer_manager:
            return []
        return self.timer_manager.get_active_timers()

    def cancel_timer(self, name: str) -> bool:
        """Cancel a timer by its display name.

        Args:
            name: Timer name to cancel.

        Returns:
            True if the timer was found and cancelled.
        """
        if not self.timer_manager:
            return False
        timer = self.timer_manager._find_timer_by_name(name)
        if timer:
            return self.timer_manager.cancel_timer(timer.timer_id)
        return False

    def get_timer_status(self) -> dict:
        """Return timer subsystem status dict."""
        if not self.timer_manager:
            return {}
        return self.timer_manager.get_status()

    # ------------------------------------------------------------------
    # Intercom / announcement passthroughs
    # ------------------------------------------------------------------

    async def announce(
        self,
        message: str,
        zones: list = None,
        priority: str = "normal",
    ) -> dict:
        """Make an announcement through the intercom system.

        Args:
            message: Text to announce.
            zones: Target zones (None = all).
            priority: One of "low", "normal", "high", "emergency".

        Returns:
            Dict with success flag and announcement_id.
        """
        if not self.intercom:
            return {"success": False, "error": "Intercom not configured"}

        priority_map = {
            "low": AnnouncementPriority.LOW,
            "normal": AnnouncementPriority.NORMAL,
            "high": AnnouncementPriority.HIGH,
            "emergency": AnnouncementPriority.EMERGENCY,
        }
        announcement = await self.intercom.announce(
            message=message,
            zones=zones,
            priority=priority_map.get(priority, AnnouncementPriority.NORMAL),
        )
        return {"success": True, "announcement_id": announcement.announcement_id}

    async def announce_emergency(self, message: str) -> dict:
        """Emergency announcement across the entire house.

        Args:
            message: Emergency text.

        Returns:
            Dict with success flag and announcement_id.
        """
        if not self.intercom:
            return {"success": False, "error": "Intercom not configured"}

        announcement = await self.intercom.announce_emergency(message)
        return {"success": True, "announcement_id": announcement.announcement_id}

    def get_intercom_zones(self) -> dict:
        """Return available intercom zones."""
        if not self.intercom:
            return {}
        return self.intercom.get_zones()

    def get_announcement_history(self, limit: int = 10) -> list:
        """Return announcement history.

        Args:
            limit: Max entries to return.
        """
        if not self.intercom:
            return []
        return self.intercom.get_history(limit)

    # ------------------------------------------------------------------
    # Notification passthroughs
    # ------------------------------------------------------------------

    async def send_notification(
        self,
        message: str,
        user_id: str = None,
        title: str = None,
        priority: str = "normal",
    ) -> dict:
        """Send a smart notification.

        Args:
            message: Notification body.
            user_id: Target user (None = broadcast).
            title: Optional title.
            priority: One of "low", "normal", "high", "urgent", "emergency".

        Returns:
            Dict with success flag and notification_id.
        """
        if not self.notifications:
            return {"success": False, "error": "Notifications not configured"}

        priority_map = {
            "low": NotificationPriority.LOW,
            "normal": NotificationPriority.NORMAL,
            "high": NotificationPriority.HIGH,
            "urgent": NotificationPriority.URGENT,
            "emergency": NotificationPriority.EMERGENCY,
        }
        notification = await self.notifications.notify(
            message=message,
            title=title,
            user_id=user_id,
            priority=priority_map.get(priority, NotificationPriority.NORMAL),
        )
        return {"success": True, "notification_id": notification.notification_id}

    def set_do_not_disturb(self, user_id: str, enabled: bool = True) -> None:
        """Toggle Do Not Disturb for a user.

        Args:
            user_id: User to configure.
            enabled: True to activate DND.
        """
        if not self.notifications:
            logger.warning("set_do_not_disturb called but notifications is None")
            return
        self.notifications.set_dnd(user_id, enabled)

    def is_do_not_disturb(self, user_id: str) -> bool:
        """Check whether DND is active for a user."""
        if not self.notifications:
            return False
        return self.notifications.is_dnd_active(user_id)

    def get_notification_history(self, user_id: str = None, limit: int = 20) -> list:
        """Return notification history.

        Args:
            user_id: Filter by user (None = all).
            limit: Max entries to return.
        """
        if not self.notifications:
            return []
        return self.notifications.get_history(user_id, limit)

    # ------------------------------------------------------------------
    # Alert passthroughs
    # ------------------------------------------------------------------

    def add_alert_condition(
        self,
        entity_id: str,
        state_equals: str = None,
        duration_minutes: int = 0,
        message: str = None,
    ) -> None:
        """Add a simple alert condition.

        Args:
            entity_id: HA entity to watch.
            state_equals: Trigger when entity reaches this state.
            duration_minutes: How long the state must persist before alerting.
            message: Custom alert message (supports {duration} placeholder).
        """
        if not self.alert_scheduler:
            logger.warning("add_alert_condition called but alert_scheduler is None")
            return
        self.alert_scheduler.add_check_simple(
            entity_id=entity_id,
            state=state_equals,
            duration_minutes=duration_minutes,
            message=message or f"{entity_id} en estado {state_equals}",
        )

    def get_active_alerts(self) -> list:
        """Return currently active (pending) alerts."""
        if not self.alert_manager or not hasattr(self.alert_manager, "get_pending_alerts"):
            return []
        return self.alert_manager.get_pending_alerts()

    def acknowledge_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier.
            user_id: User acknowledging.

        Returns:
            True if the alert was found and acknowledged.
        """
        if not self.alert_manager or not hasattr(self.alert_manager, "acknowledge"):
            return False
        return self.alert_manager.acknowledge(alert_id)

    def get_alert_summary(self) -> str:
        """Return voice-friendly alert summary."""
        if not self.alert_manager or not hasattr(self.alert_manager, "get_summary"):
            return "No hay alertas activas"
        return self.alert_manager.get_summary()
