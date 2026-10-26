"""Application-owned local voice output and feature producer construction."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from src.pipeline.feature_manager import FeatureManager, build_feature_manager
from src.pipeline.response_handler import ResponseHandler
from src.reminders.reminder_scheduler import ReminderScheduler


@dataclass
class VoiceServices:
    """Keep the graph's shared output and the reminder task it actually starts."""

    responses: ResponseHandler
    features: FeatureManager
    reminders: ReminderScheduler | None = None
    reminder_task: asyncio.Task | None = field(default=None, repr=False)

    def start_reminders(self) -> None:
        """Start exactly one tracked reminder task, including before its first turn."""
        if self.reminders is not None and (self.reminder_task is None or self.reminder_task.done()):
            self.reminder_task = asyncio.create_task(self.reminders.start())

    async def stop_reminders(self) -> None:
        """Stop reminder admission and await the exact task before stores close."""
        if self.reminders is not None:
            await self.reminders.stop()
        if self.reminder_task is not None:
            if not self.reminder_task.done():
                self.reminder_task.cancel()
            await asyncio.gather(self.reminder_task, return_exceptions=True)
            self.reminder_task = None


def build_voice_services(config: dict, *, tts, shutdown: AsyncExitStack, zone_manager=None,
                         llm=None, hooks=None, before_handler_warn_ms: float = 5.0,
                         user_manager=None, ha_client=None, home_actions=None,
                         reminder_store=None, presence_detector=None) -> VoiceServices:
    """Build the same local delivery graph used by main and register cleanup.

    Producers close before the shared output owner, including construction or
    warmup failures before the application starts listening.
    """
    streaming = config.get("tts", {}).get("streaming", {})
    response = ResponseHandler(
        tts, zone_manager, llm,
        streaming_enabled=streaming.get("enabled", True),
        streaming_buffer_ms=streaming.get("buffer_ms", 150),
        streaming_prebuffer_ms=streaming.get("prebuffer_ms", 80),
        sentence_queue_capacity=streaming.get("sentence_queue_capacity", 2),
        sentence_max_chars=streaming.get("sentence_max_chars", 400),
        max_pending_responses=streaming.get("max_pending_responses", 16),
        playback_stall_timeout_s=streaming.get("stall_timeout_s", 1.0),
        playback_drain_timeout_s=streaming.get("drain_timeout_s"),
        shutdown_timeout_s=streaming.get("shutdown_timeout_s", 1.0),
        hooks=hooks, before_handler_warn_ms=before_handler_warn_ms,
    )
    shutdown.push_async_callback(response.aclose)
    features = build_feature_manager(
        config, response_handler=response, zone_manager=zone_manager,
        user_manager=user_manager, ha_client=ha_client,
    )
    shutdown.push_async_callback(features.stop)
    services = VoiceServices(response, features)
    if config.get("rooms", {}).get("enabled", False) and reminder_store is not None:
        reminder_config = dict(config.get("reminders", {}))
        if reminder_config.get("output_zone") is None:
            reminder_config["output_zone"] = streaming.get("default_output_zone")
        services.reminders = ReminderScheduler(
            reminder_store, features.voice_delivery, presence_detector, ha_client,
            reminder_config, home_actions=home_actions,
        )
        shutdown.push_async_callback(services.stop_reminders)
    return services
