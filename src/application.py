"""Application graph and operational lifecycle shared by main and acceptance.

The caller owns the AsyncExitStack passed to the factory. Construction may
initialize private storage and its retention workers; household schedulers,
listeners, capture and model warmup start only through this lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.dashboard.api import DashboardAPI
    from src.dashboard.authority import DashboardAuthority
    from src.home_assistant.ha_client import HomeAssistantClient
    from src.home_assistant.home_actions import HomeActionService
    from src.lists.list_manager import ListManager
    from src.memory.privacy_services import PrivacyServices
    from src.pipeline.voice_pipeline import VoicePipeline
    from src.pipeline.voice_services import VoiceServices
    from src.reminders.reminder_manager import ReminderManager
    from src.routines.routine_scheduler import RoutineScheduler
    from src.users.voice_registry import VoiceRegistry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationFactories:
    """Optional leaf constructors; omitted entries resolve main's current globals.

    These adapters replace hardware/model/transport boundaries for offline
    acceptance. Policies, managers, routing and privacy use the real assembly.
    """

    ha_client: Callable[..., Any] | None = None
    stt: Callable[[dict], Any] | None = None
    tts: Callable[[dict], Any] | None = None
    chroma: Callable[..., Any] | None = None
    wake_detector: Callable[..., Any] | None = None
    amplifier: Callable[..., Any] | None = None
    preflight: Callable[[], None] | None = None


@dataclass(frozen=True)
class ApplicationModels:
    """The exact model instances used by command processing and warmup."""

    stt: Any
    tts: Any
    chroma: Any
    speaker_identifier: Any = None
    emotion_detector: Any = None


@dataclass
class ApplicationGraph:
    """Real service identities and deferred operational startup.

    Resource cleanup remains on the factory caller's stack, including failures
    before warmup/run. A graph runs once; rebuilding requires closing that stack
    and constructing a new graph, rather than reusing retired native resources.
    """

    config: dict = field(repr=False)
    pipeline: VoicePipeline
    ha_client: HomeAssistantClient
    home_actions: HomeActionService
    privacy: PrivacyServices
    voice_registry: VoiceRegistry
    voice_services: VoiceServices
    home_services: Any
    home_capabilities: Any
    list_manager: ListManager
    reminder_manager: ReminderManager
    routine_scheduler: RoutineScheduler | None
    dashboard: DashboardAPI | None
    models: ApplicationModels
    warmup_models: Callable[..., Awaitable[None]] = field(repr=False)
    dashboard_authority: DashboardAuthority | None = None
    presence_detector: Any = None
    ambient_path: Any = None
    nightly_trainer: Any = None
    _run_entered: bool = field(default=False, init=False, repr=False)

    async def warmup(self) -> None:
        """Warm the shared models and prepare the response cache, when enabled."""
        models = self.models
        responses = self.voice_services.responses
        if self.config.get("warmup", {}).get("enabled", True):
            await self.warmup_models(
                models.stt, models.tts, models.speaker_identifier,
                models.emotion_detector, models.chroma, voice_output=responses,
                command_processor=self.pipeline.command_processor,
            )
        cache_config = self.config.get("tts", {}).get("response_cache", {}) or {}
        if cache_config.get("enabled", False):
            from src.tts.response_cache import ResponseCache

            cache = ResponseCache(models.tts, synthesize_callback=responses.asynthesize)
            try:
                await cache.build()
                responses._response_cache = cache
                logger.info("Application response cache ready: %s phrases", cache.size())
            except Exception:
                logger.warning("Application response cache unavailable; continuing without cache")

    async def _start_services(self) -> None:
        state_config = self.config.get("home_assistant", {}).get("state_prefetch", {})
        if state_config.get("enabled", True):
            try:
                await self.ha_client.start_state_sync(
                    full_refresh_interval_s=state_config.get("full_refresh_interval_s", 300),
                )
            except RuntimeError:
                logger.warning("Application HA state sync unavailable; continuing with REST")

        if self.routine_scheduler is not None:
            try:
                await self.routine_scheduler.start()
            except Exception:
                # Keep the same degraded contract as construction used to have.
                # The store remains owned by the outer resource stack.
                logger.error("Routine subsystem unavailable after startup failure")
                await self.routine_scheduler.stop()
                self.routine_scheduler = None
                if self.dashboard is not None:
                    self.dashboard.scheduler = None
        if self.presence_detector is not None:
            await self.presence_detector.start()
        self.voice_services.start_reminders()
        if self.nightly_trainer is not None:
            if self.nightly_trainer.start_scheduler():
                logger.info("Application nightly trainer scheduled")

    async def run(self) -> None:
        """Start operational producers, run voice and drain admission on exit."""
        if self._run_entered:
            raise RuntimeError("application_graph_already_run")
        self._run_entered = True
        dashboard_task = None
        ambient_task = None
        try:
            await self._start_services()
            if self.dashboard is not None:
                dashboard_task = asyncio.create_task(self.dashboard.start(), name="application-dashboard")
            if self.ambient_path is not None:
                ambient_task = asyncio.create_task(self.ambient_path.start(), name="application-ambient")
                ambient_task.add_done_callback(_report_ambient_failure)
            await self.pipeline.run()
        except KeyboardInterrupt:
            logger.info("Application stopping")
        finally:
            # Retain every operational owner/handle through repeated caller
            # cancellation. One failing stop must not skip later cleanup.
            closing = asyncio.create_task(self._stop_services(dashboard_task, ambient_task))
            cancelled = False
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            closing.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _stop_services(self, dashboard_task: asyncio.Task | None,
                             ambient_task: asyncio.Task | None) -> None:
        # Resources are still open. Drain admission and scheduled producers
        # before retiring command processing; stack callbacks back this up.
        async with AsyncExitStack() as stopping:
            stopping.push_async_callback(self.pipeline.stop)
            if self.ambient_path is not None:
                stopping.push_async_callback(_stop_task, self.ambient_path, ambient_task)
            stopping.push_async_callback(self.voice_services.stop_reminders)
            if self.presence_detector is not None:
                stopping.push_async_callback(self.presence_detector.stop)
            if self.nightly_trainer is not None:
                stopping.push_async_callback(asyncio.to_thread, self.nightly_trainer.stop_scheduler)
            if self.routine_scheduler is not None:
                stopping.push_async_callback(self.routine_scheduler.stop)
            if dashboard_task is not None:
                stopping.push_async_callback(_stop_task, self.dashboard, dashboard_task)


async def _stop_task(service: Any, task: asyncio.Task | None) -> None:
    try:
        await service.stop()
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _report_ambient_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("Application ambient path failed during startup",
                     exc_info=(type(error), error, error.__traceback__))
