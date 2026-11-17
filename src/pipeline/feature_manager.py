"""
Feature Manager Module
Manages lifecycle and delegation for KZA's differentiating features:
timers, intercom, notifications, alerts, briefings, HA integration.

Extracted from VoicePipeline to reduce its responsibility.
"""

import asyncio
import logging

from src.timers import NamedTimerManager, NamedTimer
from src.intercom import IntercomSystem, AnnouncementPriority
from src.notifications import SmartNotificationManager, NotificationPriority
from src.alerts import AlertManager, AlertScheduler

logger = logging.getLogger(__name__)

# Voice texts of the alert summary (Spanish, rioplatense). An absent manager or
# producer is reported as unavailable, never as "no alerts".
ALERTS_UNAVAILABLE = "El sistema de alertas no está disponible ahora."
ALERTS_NONE = "No hay alertas activas."
_SUMMARY_LISTED = 3


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
        timer_manager: NamedTimerManager | None = None,
        intercom: IntercomSystem | None = None,
        notifications: SmartNotificationManager | None = None,
        alert_manager: AlertManager | None = None,
        alert_scheduler: AlertScheduler | None = None,
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
        self._started: list = []
        self._alert_scheduler_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all non-None subsystems."""
        if self._running:
            logger.warning("FeatureManager already running")
            return

        try:
            for service in (self.timer_manager, self.intercom, self.notifications):
                if service is not None:
                    self._started.append(service)
                    await service.start()
            if self.alert_scheduler is not None:
                self._started.append(self.alert_scheduler)
                self._alert_scheduler_task = asyncio.create_task(self.alert_scheduler.start())
            if self.ha_integration is not None:
                self._started.append(self.ha_integration)
                await self.ha_integration.start()
            self._running = True
            logger.info("FeatureManager started")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop attempted producers even after a partial startup failure."""
        self._running = False
        errors = []
        started, self._started = self._started, []
        for service in reversed(started):
            try:
                await service.stop()
            except Exception as error:
                errors.append(error)
        task, self._alert_scheduler_task = self._alert_scheduler_task, None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if errors:
            raise ExceptionGroup("Feature producer shutdown failed", errors)
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

        if self.alerts_available():
            status["alerts"] = {"available": True, "active": len(self.alert_manager.get_active_alerts())}
        else:
            status["alerts"] = {"available": False, "active": None}

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
    ) -> NamedTimer | None:
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

    def alerts_available(self) -> bool:
        """Whether alerts reflect the house: a manager plus a running scheduler with producers.

        Without producers (no handler registered in the scheduler) or while the
        scheduler is not running (never started, stopped, or its loop exited on
        an error) an empty alert list means "unknown", not "no alerts".
        """
        if self.alert_manager is None or self.alert_scheduler is None:
            return False
        status = self.alert_scheduler.get_status()
        if status.get("running") is not True:
            return False
        handlers = status.get("registered_handlers", {})
        return sum(handlers.values()) > 0

    def get_active_alerts(self) -> list:
        """Return active alerts (unacknowledged, condition not resolved), newest first.

        An empty list is only meaningful when alerts_available() is True.
        """
        if self.alert_manager is None:
            return []
        return self.alert_manager.get_active_alerts()

    def get_alert_summary(self) -> str:
        """Return a voice-friendly alert summary in Spanish.

        Returns:
            ALERTS_UNAVAILABLE without a manager or a running scheduler with producers,
            ALERTS_NONE when available and nothing is active, else the count
            and up to three alert messages.
        """
        if not self.alerts_available():
            return ALERTS_UNAVAILABLE
        active = self.alert_manager.get_active_alerts()
        if not active:
            return ALERTS_NONE
        listed = "; ".join(alert.message for alert in active[:_SUMMARY_LISTED])
        rest = len(active) - _SUMMARY_LISTED
        more = f"; y {rest} más" if rest > 0 else ""
        if len(active) == 1:
            return f"Hay una alerta activa: {listed}."
        return f"Hay {len(active)} alertas activas: {listed}{more}."


def build_feature_manager(config: dict, *, response_handler, zone_manager=None,
                          user_manager=None, ha_client=None) -> FeatureManager:
    """Build the application's feature producers with one local voice owner.

    Args:
        config: Existing central configuration, including fixed output defaults.
        response_handler: Application-owned synthesis/output boundary.
        zone_manager: Configured local zone routes, if any.
        user_manager: Notification profile lookup.
        ha_client: Existing feature data source, never a TTS fallback here.

    Returns:
        Feature manager whose voice-producing callbacks await shared delivery.
    """
    from src.core.voice_delivery import LocalVoiceDelivery

    streaming = config.get("tts", {}).get("streaming", {})
    voice = LocalVoiceDelivery(response_handler, streaming.get("default_output_zone"), zone_manager is not None)
    alerts = config.get("alerts", {})
    manager = scheduler = None
    if alerts.get("enabled", True):
        general = alerts.get("general", {})
        manager = AlertManager(
            tts_callback=response_handler.aspeak_to_all_zones if general.get("voice_notifications", True) else None,
            cooldown_seconds=general.get("cooldown_seconds", 300.0),
            max_history=general.get("max_history", 1000),
            reannounce_interval_seconds=general.get("reannounce_interval_seconds"),
        )
        schedule = alerts.get("scheduler", {})
        if schedule.get("enabled", True):
            scheduler = AlertScheduler(
                alert_manager=manager,
                security_interval=schedule.get("security_interval_seconds", 60.0),
                pattern_interval=schedule.get("pattern_interval_seconds", 300.0),
                device_interval=schedule.get("device_interval_seconds", 600.0),
            )
    features = FeatureManager(
        timer_manager=NamedTimerManager(tts_callback=voice.aspeak),
        intercom=IntercomSystem(tts_callback=voice.aspeak, zone_manager=zone_manager, ha_client=ha_client),
        notifications=SmartNotificationManager(tts_callback=voice.aspeak, user_manager=user_manager, ha_client=ha_client),
        alert_manager=manager, alert_scheduler=scheduler,
    )
    features.voice_delivery = voice
    return features
