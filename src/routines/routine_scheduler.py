"""Schedule normalized KZA routines using durable, single-occurrence claims."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone as utc_timezone
from enum import StrEnum
import json
import logging
import uuid
from zoneinfo import ZoneInfo

from src.routines.models import (
    ClaimResult,
    RoutineConflict,
    RoutineNotFound,
    RoutineRun,
    RoutineSpec,
    RunStatus,
    ScheduledRoutine,
    StateCondition,
    parse_routine_spec,
    run_to_dict,
    spec_to_dict,
)
from src.routines.routine_store import RoutineStore
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionRequest
from src.routines.models import RoutineValidationError

logger = logging.getLogger(__name__)

__all__ = ['RoutineScheduler', 'ScheduledRoutine', 'TriggerType']


class TriggerType(StrEnum):
    """Legacy import names; supported web schedules use TimeSchedule only."""

    TIME = 'time'
    TIME_RANGE = 'time_range'
    SUNRISE = 'sunrise'
    SUNSET = 'sunset'
    PRESENCE_ENTER = 'presence_enter'
    PRESENCE_LEAVE = 'presence_leave'
    PRESENCE_HOME = 'presence_home'
    PRESENCE_AWAY = 'presence_away'
    PRESENCE_GUEST = 'presence_guest'
    DEVICE_STATE = 'device_state'
    VOICE_COMMAND = 'voice_command'
    VOICE_USER = 'voice_user'
    VOICE_GUEST = 'voice_guest'
    WEBHOOK = 'webhook'
    CRON = 'cron'


class RoutineScheduler:
    """Own scheduler tasks and dispatch only newly committed routine claims.

    Args:
        routine_executor: Executor implementing async execute_run(RoutineRun).
        presence_detector: Retained constructor argument for legacy callers.
        ha_client: Client used for fresh, awaited condition reads.
        store: Durable routine and receipt storage owned until stop.
        timezone: Supported local calendar timezone, America/Montevideo.
        clock: A timezone-aware clock; defaults to current UTC time.
    """

    def __init__(self, routine_executor, presence_detector=None, ha_client=None, *,
                 store: RoutineStore, timezone: str = 'America/Montevideo',
                 clock: Callable[[], datetime] | None = None) -> None:
        if timezone != 'America/Montevideo':
            raise ValueError('Routine timezone must be America/Montevideo')
        self.executor = routine_executor
        self.presence = presence_detector
        self.ha_client = ha_client
        self.store = store
        self.timezone = timezone
        self._zone = ZoneInfo(timezone)
        self._clock = clock or (lambda: datetime.now(utc_timezone.utc))
        self._boot_at = self._utc(self._clock())
        self._running = False
        self._stopping = False
        self._scheduler_task: asyncio.Task | None = None
        self._workers: dict[str, asyncio.Task] = {}
        self._started_workers: set[str] = set()
        self._lifecycle_lock = asyncio.Lock()

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('Scheduler clock must have a timezone')
        return value.astimezone(utc_timezone.utc)

    async def register_routine(self, spec: RoutineSpec, *, actor: str) -> ScheduledRoutine:
        """Persist a new normalized definition with server-owned identity."""
        await self._validate_control_selection(spec)
        return await self.store.save(spec, actor=actor)

    async def update_routine(self, routine_id: str, spec: RoutineSpec, *,
                             expected_revision: int, actor: str) -> ScheduledRoutine:
        """Persist an optimistic update while retaining receipt snapshots."""
        await self._validate_control_selection(spec)
        return await self.store.save(spec, routine_id=routine_id,
                                     expected_revision=expected_revision, actor=actor)

    async def _validate_control_selection(self, spec: RoutineSpec) -> None:
        try:
            for action in spec.actions:
                await self.executor.home_actions.validate_selection(HomeActionRequest(
                    action.entity_id, str(action.operation), "admission", action.brightness_pct, ActionOrigin.ROUTINE,
                ), actor=HomeActor.household())
        except ValueError as error:
            raise RoutineValidationError(str(error)) from error

    async def unregister_routine(self, routine_id: str, *, expected_revision: int) -> None:
        """Delete an idle definition without deleting its receipts."""
        await self.store.delete(routine_id, expected_revision=expected_revision)

    async def get_routine(self, routine_id: str) -> ScheduledRoutine | None:
        """Load one persisted definition, or None if absent."""
        return await self.store.get(routine_id)

    async def get_all_routines(self) -> list[ScheduledRoutine]:
        """Load all KZA definitions without importing HA automations."""
        return await self.store.list_all()

    async def enable_routine(self, routine_id: str, *, expected_revision: int,
                             actor: str) -> ScheduledRoutine:
        """Enable an idle routine with optimistic revision checking."""
        return await self._set_enabled(routine_id, True, expected_revision, actor)

    async def disable_routine(self, routine_id: str, *, expected_revision: int,
                              actor: str) -> ScheduledRoutine:
        """Disable an idle routine with optimistic revision checking."""
        return await self._set_enabled(routine_id, False, expected_revision, actor)

    async def _set_enabled(self, routine_id: str, enabled: bool,
                           expected_revision: int, actor: str) -> ScheduledRoutine:
        routine = await self.store.get(routine_id)
        if routine is None:
            raise RoutineNotFound(routine_id)
        if not enabled:
            # Revoking control permission must never prevent disabling a job.
            return await self.store.save(replace(routine.spec, enabled=False), routine_id=routine_id,
                                         expected_revision=expected_revision, actor=actor)
        return await self.update_routine(routine_id, replace(routine.spec, enabled=enabled),
                                         expected_revision=expected_revision, actor=actor)

    def preview(self, spec: RoutineSpec) -> dict:
        """Return a normalized definition and ordered actions without I/O."""
        return spec_to_dict(parse_routine_spec(spec_to_dict(spec)))

    async def execute_by_id(self, routine_id: str, *, actor: str, request_id: str) -> RoutineRun:
        """Claim an actor-scoped manual attempt before starting any effects."""
        async with self._lifecycle_lock:
            if self._stopping:
                raise RoutineConflict('scheduler_stopping')
            occurrence_key = json.dumps(['manual', actor, request_id], separators=(',', ':'))
            claim = await self.store.claim_run(routine_id, occurrence_key,
                                               actor=actor, source='manual')
            return self._start_claimed_run(claim)

    async def get_run(self, routine_id: str, run_id: str) -> RoutineRun:
        """Load a receipt only when it belongs to the requested routine."""
        run = await self.store.get_run(run_id)
        if run is None or run.routine_id != routine_id:
            raise RoutineNotFound(run_id)
        return run

    async def get_manual_attempt(self, routine_id: str, *, actor: str, request_id: str) -> RoutineRun:
        """Read an actor's exact manual occurrence without admission or effects."""
        occurrence_key = json.dumps(['manual', actor, request_id], separators=(',', ':'))
        run = await self.store.get_run_by_occurrence(routine_id, occurrence_key)
        if run is None:
            raise RoutineNotFound(request_id)
        return run

    async def get_latest_run(self, routine_id: str) -> RoutineRun | None:
        """Read the latest persisted receipt without scheduling any effects."""
        return await self.store.get_latest_run(routine_id)

    async def execute_by_name(self, name: str, context: dict | None = None) -> dict | None:
        """Retain legacy name lookup through durable claims, without voice wiring.

        Context may supply actor and request_id. Without a request_id each legacy
        invocation is a new attempt; web callers must use execute_by_id instead.
        """
        routines = await self.get_all_routines()
        name_lower = name.lower()
        routine = next((item for item in routines if item.spec.name.lower() == name_lower), None)
        if routine is None:
            routine = next((item for item in routines if name_lower in item.spec.name.lower()), None)
        if routine is None:
            return None
        context = context or {}
        run = await self.execute_by_id(routine.routine_id,
                                      actor=context.get('actor', 'system:legacy'),
                                      request_id=context.get('request_id') or uuid.uuid4().hex)
        return run_to_dict(run)

    async def start(self) -> None:
        """Recover interrupted receipts before starting the UTC-clock loop."""
        async with self._lifecycle_lock:
            if self._running:
                return
            if self._workers:
                raise RoutineConflict('routine_busy')
            self._boot_at = self._utc(self._clock())
            await self.store.initialize()
            await self.store.recover_interrupted()
            self._stopping = False
            self._running = True
            self._scheduler_task = asyncio.create_task(self._time_loop(), name='routine-scheduler')

    async def stop(self) -> None:
        """Cancel and await every owned task before closing the durable store."""
        async with self._lifecycle_lock:
            self._stopping = True
            self._running = False
            loop_task = self._scheduler_task
            self._scheduler_task = None
            if loop_task is not None:
                loop_task.cancel()
                await asyncio.gather(loop_task, return_exceptions=True)
            workers = dict(self._workers)
            unstarted = {run_id for run_id, task in workers.items()
                         if not task.done() and run_id not in self._started_workers}
            for task in workers.values():
                task.cancel()
            await asyncio.gather(*workers.values(), return_exceptions=True)
            # A task cancelled before its first instruction cannot run its handler.
            for run_id in unstarted:
                try:
                    run = await self.store.get_run(run_id)
                    if run is not None and run.status == RunStatus.RUNNING:
                        await self.store.finish_run(run_id, RunStatus.INTERRUPTED)
                except Exception:
                    logger.exception('Routine interruption persistence failed: %s', run_id)
            self._workers.clear()
            await self.store.close()

    async def _time_loop(self) -> None:
        while self._running:
            try:
                await self._check_time_triggers(self._clock())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Routine scheduler tick failed')
            await asyncio.sleep(1)

    async def _check_time_triggers(self, now: datetime) -> None:
        """Claim only the current local minute, never previous due occurrences."""
        now = self._utc(now)
        async with self._lifecycle_lock:
            if self._stopping:
                return
            local_now = now.astimezone(self._zone)
            for routine in await self.store.list_all():
                schedule = routine.spec.schedule
                if not routine.spec.enabled or schedule is None:
                    continue
                if local_now.weekday() not in schedule.weekdays:
                    continue
                if local_now.strftime('%H:%M') != schedule.at:
                    continue
                due = local_now.replace(second=0, microsecond=0).astimezone(utc_timezone.utc)
                if due < self._boot_at or due > now:
                    continue
                try:
                    claim = await self.store.claim_run(
                        routine.routine_id, 'scheduled:' + due.isoformat(),
                        actor='system:scheduler', source='scheduled', scheduled_for=due,
                        expected_revision=routine.revision,
                    )
                except (RoutineNotFound, RoutineConflict):
                    continue
                self._start_claimed_run(claim)

    def _start_claimed_run(self, claim: ClaimResult) -> RoutineRun:
        if claim.created and claim.run.status == RunStatus.RUNNING:
            run_id = claim.run.run_id
            task = asyncio.create_task(self._run_claimed(claim.run), name='routine:' + run_id)
            self._workers[run_id] = task
            task.add_done_callback(lambda completed: self._observe_worker(run_id, completed))
        return claim.run

    def _observe_worker(self, run_id: str, task: asyncio.Task) -> None:
        self._workers.pop(run_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error('Routine worker failed: %s', run_id,
                             exc_info=(type(error), error, error.__traceback__))

    async def _run_claimed(self, run: RoutineRun) -> RoutineRun:
        """Evaluate fresh conditions and execute the immutable claimed snapshot."""
        self._started_workers.add(run.run_id)
        try:
            if not await self._check_conditions(run.snapshot.spec.conditions):
                return await self.store.finish_run(run.run_id, RunStatus.SKIPPED)
            return await self.executor.execute_run(run)
        except asyncio.CancelledError:
            try:
                await self.store.finish_run(run.run_id, RunStatus.INTERRUPTED)
            except Exception:
                # Preserve running/sending evidence for transactional recovery.
                logger.exception('Routine cancellation persistence failed: %s', run.run_id)
            raise
        finally:
            self._started_workers.discard(run.run_id)

    async def _check_conditions(self, conditions: tuple[StateCondition, ...]) -> bool:
        for condition in conditions:
            if self.ha_client is None:
                return False
            state = await self.ha_client.get_entity_state_fresh(condition.entity_id)
            if not isinstance(state, dict):
                return False
            actual = state.get('state')
            if actual is None or actual in ('unknown', 'unavailable') or actual != condition.state:
                return False
        return True
