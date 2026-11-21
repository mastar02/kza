"""Application-owned local voice output and feature producer construction."""

import asyncio
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace

from src.core.household_clock import HouseholdClock
from src.core.task_outcome import EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.core.voice_delivery import ResponseBudgets
from src.nlu.household_intents import HouseholdOperation
from src.pipeline.feature_manager import FeatureManager, build_feature_manager
from src.pipeline.request_router import RequestRouter
from src.pipeline.response_handler import ResponseHandler
from src.reminders.reminder_scheduler import ReminderScheduler
from src.timers.named_timers import TimerWriteError
from src.timers.timer_store import TimerStore
from src.timers.voice_commands import NO_RING_ROUTE, TIMERS_UNAVAILABLE, TimerVoiceCommands, TimerVoiceReply


class DurableTimerVoiceCommands(TimerVoiceCommands):
    """Let the existing timer command logic prepare replies before SQLite commits."""

    async def execute(self, operation: HouseholdOperation, complement: str, *, text: str,
                      zone_id: str | None) -> TimerVoiceReply:
        """Return a mutation confirmation only after the timer transaction commits."""
        if self.manager is None:
            return await super().execute(operation, complement, text=text, zone_id=zone_id)
        if not self.manager.running:
            if self.manager.persistence is PersistenceState.UNCERTAIN:
                return self._write_failure(PersistenceState.UNCERTAIN)
            return await super().execute(operation, complement, text=text, zone_id=zone_id)
        try:
            reply = await self.manager.commit_command(
                lambda view: TimerVoiceCommands(view).execute(operation, complement, text=text, zone_id=zone_id)
            )
        except TimerWriteError as error:
            return self._write_failure(error.state)
        return replace(reply, outcome=self._committed(reply.outcome))

    async def execute_legacy(self, text: str, zone_id: str | None) -> TimerVoiceReply | None:
        """Use the legacy timer grammar at an async delivery boundary."""
        manager = self.manager
        if manager is None or not manager.claims_legacy_command(text):
            return None
        if zone_id is None and manager._is_create_command(text.lower().strip()):
            return TimerVoiceReply(NO_RING_ROUTE, TaskOutcome(TaskStatus.UNSUPPORTED))
        try:
            result = await manager.commit_command(lambda view: view.handle_voice_command(text, zone_id=zone_id))
        except TimerWriteError as error:
            return self._write_failure(error.state)
        return TimerVoiceReply(result["response"], self._committed(result["outcome"]))

    @staticmethod
    def _committed(outcome: TaskOutcome) -> TaskOutcome:
        if outcome.persistence is PersistenceState.VOLATILE:
            return replace(outcome, persistence=PersistenceState.COMMITTED)
        return outcome

    @staticmethod
    def _write_failure(state: PersistenceState) -> TimerVoiceReply:
        uncertain = state is PersistenceState.UNCERTAIN
        outcome = TaskOutcome(TaskStatus.UNCERTAIN if uncertain else TaskStatus.FAILED,
                              persistence=state,
                              effect=EffectState.UNCERTAIN if uncertain else EffectState.NOT_OBSERVED)
        response = ("No puedo confirmar que se haya guardado el cambio del temporizador."
                    if uncertain else TIMERS_UNAVAILABLE)
        return TimerVoiceReply(response, outcome)


class LegacyTimerRequestRouter(RequestRouter):
    """Route admitted legacy timer turns through the same durable voice service."""

    def __init__(self, *, timer_commands: DurableTimerVoiceCommands, timer_zone: str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timer_commands = timer_commands
        self._timer_zone = timer_zone

    async def _try_home_command(self, text: str, result: dict, *, room_id: str | None,
                                pipeline_start: float, extra_gate: bool) -> bool:
        # Preserve the legacy order: extended home controls keep first claim.
        if await super()._try_home_command(
            text, result, room_id=room_id, pipeline_start=pipeline_start, extra_gate=extra_gate,
        ):
            return True
        zone_id = self._response_zone.get() or self._timer_zone
        reply = await self._timer_commands.execute_legacy(text, zone_id)
        if reply is None:
            return False
        result.update(intent="timer", response=reply.response, success=reply.outcome.task_completed,
                      _task_outcome=reply.outcome)
        await self._speak_response(reply.response)
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return True


@dataclass
class VoiceServices:
    """Keep the graph's shared output and the reminder task it actually starts."""

    responses: ResponseHandler
    features: FeatureManager
    timer_commands: TimerVoiceCommands
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


def shared_output_zone(config: dict) -> str | None:
    """Fixed zone for shared notices that have no origin room.

    ``reminders.output_zone`` (user decision 2026-11-16: the living until
    SCHEMA-V2), else ``tts.streaming.default_output_zone``.

    Args:
        config: Central configuration.

    Returns:
        The configured zone id, or None when neither is set.
    """
    zone = config.get("reminders", {}).get("output_zone")
    if zone is not None:
        return zone
    return config.get("tts", {}).get("streaming", {}).get("default_output_zone")


def timer_fallback_zone(config: dict, voice_delivery) -> str | None:
    """Ring zone of a voice timer whose turn has no origin room (T030-U1 R1).

    With zone routes (a ZoneManager, so the delivery requires a zone) it is
    the shared-notice zone of :func:`shared_output_zone`, the same one shared
    reminders use; without zone routes it is "default", the one local output.

    Args:
        config: Central configuration.
        voice_delivery: The feature manager's ``LocalVoiceDelivery``, or None.

    Returns:
        A zone id the timer's ring can be delivered to, or None when there is
        none (the timer is then refused instead of confirmed).
    """
    if voice_delivery is None:
        return None
    if not getattr(voice_delivery, "require_zone", False):
        return "default"
    return shared_output_zone(config)


def build_voice_services(config: dict, *, tts, shutdown: AsyncExitStack, zone_manager=None,
                         llm=None, hooks=None, before_handler_warn_ms: float = 5.0,
                         user_manager=None, ha_client=None, home_actions=None,
                         reminder_store=None, presence_detector=None,
                         reminder_wake_event: asyncio.Event | None = None,
                         register_output: Callable[[Callable[[], asyncio.Task]], None] | None = None,
                         household_clock: HouseholdClock | None = None,
                         alert_muted: Callable[[], bool] | None = None) -> VoiceServices:
    """Build the same local delivery graph used by main and register cleanup.

    Producers close before the shared output owner, including construction or
    warmup failures before the application starts listening. Alerts and reminders
    share the store and household clock. Alerts remain muted until an owner's
    mute check is bound here or after the capture control is built.
    """
    streaming = config.get("tts", {}).get("streaming", {})
    buffering = config.get("llm", {}).get("buffering", {})
    response = ResponseHandler(
        tts, zone_manager, llm,
        streaming_enabled=streaming.get("enabled", True),
        streaming_buffer_ms=streaming.get("buffer_ms", 150),
        streaming_prebuffer_ms=streaming.get("prebuffer_ms", 80),
        llm_buffer_preset=buffering.get("preset", "balanced"),
        llm_use_filler=buffering.get("use_filler", True),
        llm_filler_phrases=buffering.get("filler_phrases"),
        sentence_queue_capacity=streaming.get("sentence_queue_capacity", 2),
        sentence_max_chars=streaming.get("sentence_max_chars", 400),
        max_pending_responses=streaming.get("max_pending_responses", 16),
        playback_stall_timeout_s=streaming.get("stall_timeout_s", 1.0),
        playback_drain_timeout_s=streaming.get("drain_timeout_s"),
        shutdown_timeout_s=streaming.get("shutdown_timeout_s", 1.0),
        response_budgets=ResponseBudgets(**streaming.get("response_timeouts", {})),
        hooks=hooks, before_handler_warn_ms=before_handler_warn_ms,
    )
    shutdown.push_async_callback(response.wait_closed)
    if register_output is not None:
        register_output(response.begin_close)
    features = build_feature_manager(
        config, response_handler=response, zone_manager=zone_manager,
        user_manager=user_manager, ha_client=ha_client,
        reminder_store=reminder_store, household_clock=household_clock,
    )
    if alert_muted is not None:
        features.bind_alert_mute(alert_muted)
    shutdown.push_async_callback(features.aclose)
    timer_commands = TimerVoiceCommands(features.timer_manager)
    if reminder_store is not None and features.timer_manager is not None:
        features.timer_manager.bind_store(TimerStore(reminder_store))
        timer_commands = DurableTimerVoiceCommands(features.timer_manager)
    services = VoiceServices(response, features, timer_commands)
    if config.get("rooms", {}).get("enabled", False) and reminder_store is not None:
        reminder_config = dict(config.get("reminders", {}))
        reminder_config["output_zone"] = shared_output_zone(config)
        services.reminders = ReminderScheduler(
            reminder_store, features.voice_delivery, presence_detector, ha_client,
            reminder_config, home_actions=home_actions, wake_event=reminder_wake_event,
            household_clock=household_clock,
        )
        # Q31: a held reminder notice gets a new round when other activity frees the output.
        response.add_output_free_listener(services.reminders.notify_output_available)
        shutdown.push_async_callback(services.stop_reminders)
    return services
