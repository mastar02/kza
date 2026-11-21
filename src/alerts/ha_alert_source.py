"""Read explicitly selected Home Assistant states and maintain alert episodes."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, TYPE_CHECKING

from src.alerts.alert_manager import AlertCondition, AlertManager, AlertPriority, AlertProducerStatus, AlertType
from src.alerts.device_alerts import BATTERY_LOW, OFFLINE
from src.alerts.security_alerts import DOOR_OPEN_UNEXPECTED
from src.core.observation import Observation, ObservationReason, ObservationSource, ObservationState

if TYPE_CHECKING:
    from src.conversation.command_turns import CommandTurns
    from src.home_assistant.ha_client import HAStatesSnapshot
    from src.home_assistant.state_projection import EntityObservation

logger = logging.getLogger(__name__)


class _HAStateReader(Protocol):
    def register_state_callback(self, callback: Callable[[str, EntityObservation], None],
                                *, with_observation: bool) -> None: ...
    def unregister_state_callback(self, callback: Callable[[str, EntityObservation], None],
                                  *, with_observation: bool) -> None: ...
    def current_state_stream_generation(self) -> int | None: ...
    def get_entity_observation(self, entity_id: str) -> EntityObservation: ...
    async def get_state_snapshot(self) -> HAStatesSnapshot: ...


class _RuleType(StrEnum):
    OPEN = DOOR_OPEN_UNEXPECTED
    BATTERY = BATTERY_LOW
    UNAVAILABLE = OFFLINE


def _instant(epoch: float | None) -> datetime | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC)


@dataclass(frozen=True)
class _Rule:
    kind: _RuleType
    name: str
    threshold: float

    @property
    def alert_type(self) -> AlertType:
        return AlertType.SECURITY if self.kind is _RuleType.OPEN else AlertType.DEVICE


@dataclass(frozen=True)
class _Evidence:
    observed_at: float
    expires_at: float
    stream_generation: int | None
    revision: int | None = None

    def current(self, now: float, stream_generation: int | None) -> bool:
        return (self.observed_at <= now < self.expires_at
                and (self.stream_generation is None or self.stream_generation == stream_generation))

    def predates(self, previous: _Evidence) -> bool:
        if self.stream_generation is not None and previous.stream_generation is not None:
            if self.stream_generation != previous.stream_generation:
                return self.stream_generation < previous.stream_generation
            if self.revision is not None and previous.revision is not None:
                return self.revision < previous.revision
        return self.observed_at < previous.observed_at


def _configured_rules(config: dict, enabled_types: dict) -> dict[str, list[_Rule]]:
    groups = (
        ("door_window", _RuleType.OPEN, "open_seconds", 120.0, "binary_sensor"),
        ("battery", _RuleType.BATTERY, "below_percent", 20.0, "sensor"),
        ("unavailable", _RuleType.UNAVAILABLE, "grace_seconds", 300.0, None),
    )
    rules: dict[str, list[_Rule]] = {}
    for group, kind, threshold_key, default, domain in groups:
        alert_type = AlertType.SECURITY if kind is _RuleType.OPEN else AlertType.DEVICE
        if not enabled_types.get(alert_type.value, True):
            continue
        for entity_id, spec in config.get(group, {}).items():
            if (not isinstance(entity_id, str) or not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", entity_id)
                    or (domain is not None and entity_id.split(".", 1)[0] != domain)):
                raise ValueError(f"Invalid entity for alerts.ha.{group}")
            name = spec.get("name", entity_id)
            threshold = spec.get(threshold_key, default)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid name for alerts.ha.{group}")
            if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0
                    or (kind is _RuleType.BATTERY and threshold > 100)):
                raise ValueError(f"Invalid {threshold_key} for alerts.ha.{group}")
            rules.setdefault(entity_id, []).append(_Rule(kind, name, threshold))
    return rules


def _state_changed_at(state: dict, observed_at: float) -> float | None:
    value = state.get("last_changed")
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
        changed_at = instant.timestamp() if instant.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        return None
    return changed_at if changed_at is not None and changed_at <= observed_at else None


class HAAlertSource:
    """Own a bounded HA delta consumer and reconcile from dated read-only evidence."""

    def __init__(self, ha_client: _HAStateReader, alert_manager: AlertManager, config: dict,
                 *, enabled_types: dict | None = None, clock: Callable[[], float] = time.time) -> None:
        self._ha = ha_client
        self._alerts = alert_manager
        self._rules = _configured_rules(config, enabled_types or {})
        self._clock = clock
        maxsize = config.get("queue_maxsize", 128)
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("alerts.ha.queue_maxsize must be a positive integer")
        self._queue: asyncio.Queue[tuple[str, EntityObservation]] = asyncio.Queue(maxsize=maxsize)
        self._evaluate = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._notice_task: asyncio.Task | None = None
        self._notice_wake = asyncio.Event()
        self._delta_in_flight = False
        self._turns: CommandTurns | None = None
        self._overflow = self._reconciled_overflow = 0
        self._started_at: float | None = None
        self._latest: dict[str, _Evidence] = {}
        self._evidence: dict[tuple[str, _RuleType], _Evidence] = {}
        self._last_known: dict[tuple[str, _RuleType], _Evidence] = {}
        self._last_check: dict[_RuleType, float] = {}
        self._last_rest_expiry: float | None = None
        self._rest_failed = False
        self._producer_rules = {
            kind: tuple((entity, kind) for entity, rules in self._rules.items() if any(rule.kind is kind for rule in rules))
            for kind in _RuleType
        }
        self._notice_ready: set[tuple[str, _RuleType]] = set()
        self._baseline: set[tuple[str, _RuleType]] = set()

    @property
    def configured(self) -> bool:
        """Whether at least one HA condition was explicitly selected."""
        return bool(self._rules)

    @property
    def running(self) -> bool:
        """Whether the owned consumer can still receive deltas."""
        return self._task is not None and not self._task.done()

    @property
    def uncertain(self) -> bool:
        """Whether a dropped delta still needs a full snapshot to reconcile it."""
        return self._overflow != self._reconciled_overflow

    def get_status(self) -> dict:
        """Report current evidence coverage without exposing payloads or entity names."""
        count = sum(len(rules) for rules in self._rules.values())
        now = self._clock()
        generation = self._ha.current_state_stream_generation()
        known = (sum(evidence.current(now, generation) for evidence in self._evidence.values())
                 if self.running and not self.uncertain else 0)
        return {"running": self.running, "uncertain": self.uncertain, "overflow": self._overflow,
                "known": known, "unknown": count - known}

    def get_producer_status(self) -> tuple[AlertProducerStatus, ...]:
        """Project each configured check family from its real HA observations."""
        now = self._clock()
        generation = self._ha.current_state_stream_generation()
        running = self.running
        connected = running and (generation is not None or (
            self._last_rest_expiry is not None and now < self._last_rest_expiry
        ))
        producers = []
        for kind, keys in self._producer_rules.items():
            if not keys:
                continue
            current = [self._evidence[key] for key in keys if key in self._evidence
                       and self._evidence[key].current(now, generation) and running and not self.uncertain]
            previous = [self._last_known[key] for key in keys if key in self._last_known]
            known, unknown = len(current), len(keys) - len(current)
            state, reason = ObservationState.UNKNOWN, ObservationReason.UNKNOWN_TIME
            if not running:
                state = ObservationState.UNAVAILABLE
                reason = ObservationReason.CLOSED if self._started_at is not None else ObservationReason.UNCONFIGURED
            elif self.uncertain:
                reason = ObservationReason.CHANGED
            elif known:
                state, reason = (ObservationState.PARTIAL if unknown else ObservationState.FRESH), None
            elif not connected and (self._rest_failed or any(item.stream_generation is not None for item in previous)):
                state, reason = ObservationState.UNAVAILABLE, ObservationReason.UNAVAILABLE
            elif previous and all(item.expires_at <= now for item in previous):
                state, reason = ObservationState.STALE, ObservationReason.EXPIRED
            evidence = current or previous
            observation = Observation(
                ObservationSource.HA_ALERTS, state,
                _instant(min(item.observed_at for item in evidence)) if evidence else None,
                _instant(min(item.expires_at for item in evidence)) if evidence else None,
                _instant(self._last_check.get(kind)), reason,
            )
            producers.append(AlertProducerStatus(kind.value, observation, connected, known, unknown,
                                                 self._overflow, self.uncertain or unknown > 0))
        return tuple(producers)

    async def start(self, command_turns: CommandTurns | None = None) -> None:
        """Subscribe and start the worker without waiting on HA or voice output."""
        if self._task is not None or not self.configured:
            return
        if self._started_at is None:
            self._started_at = self._clock()
        self._turns = command_turns
        self._ha.register_state_callback(self._enqueue, with_observation=True)
        self._task = asyncio.create_task(self._consume(), name="ha-alert-deltas")
        self._notice_task = asyncio.create_task(self._deliver_notices(), name="ha-alert-voice")

    async def stop(self) -> None:
        """Unsubscribe and retire the exact consumer before HA and voice close."""
        task = self._task
        if task is None:
            return
        self._ha.unregister_state_callback(self._enqueue, with_observation=True)
        workers = (task, self._notice_task)
        workers = tuple(worker for worker in workers if worker is not None)
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._task = self._notice_task = None
        self._turns = None
        self._clear_queue()
        self._evidence.clear()
        self._notice_ready.clear()
        self._notice_wake.clear()

    def _enqueue(self, entity_id: str, observation: EntityObservation) -> None:
        # Called on the HA event loop while it owns its state projection lock.
        if entity_id in self._rules:
            try:
                self._queue.put_nowait((entity_id, observation))
            except asyncio.QueueFull:
                self._overflow += 1

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    async def _consume(self) -> None:
        while True:
            entity_id, observation = await self._queue.get()
            reconciled = self._reconciled_overflow
            self._delta_in_flight = True
            try:
                async with self._evaluate:
                    if self.uncertain or reconciled != self._reconciled_overflow:
                        continue
                    if (observation.source != "ha-ws-entities"
                            or observation.generation != self._ha.current_state_stream_generation()):
                        continue
                    now = self._clock()
                    for rule in self._rules[entity_id]:
                        self._last_check[rule.kind] = now
                    if not observation.current(now):
                        self._mark_unknown(entity_id)
                    else:
                        await self._apply(entity_id, observation.to_state(now), _Evidence(
                            observation.observed_at, observation.expires_at,
                            observation.generation, observation.revision,
                        ))
            except Exception as error:
                self._mark_unknown(entity_id)
                logger.warning("HA alert delta evaluation failed: %s", type(error).__name__)
            finally:
                self._delta_in_flight = False
                self._queue.task_done()
                if self._queue.empty():
                    self._notice_wake.set()

    async def reconcile(self) -> None:
        """Read one REST snapshot and reconcile it with fresher WS evidence."""
        if not self.running:
            return
        task, overflow = self._task, self._overflow
        read_started = self._clock()
        for kind, keys in self._producer_rules.items():
            if keys:
                self._last_check[kind] = read_started
        snapshot = await self._ha.get_state_snapshot()
        if self._task is not task:
            return
        self._rest_failed = not snapshot.available
        self._last_rest_expiry = snapshot.expires_at.timestamp() if snapshot.available and snapshot.expires_at else None
        if not self.uncertain:
            await self._queue.join()
        async with self._evaluate:
            if self._task is not task or self._overflow != overflow:
                return
            observed_at = snapshot.observed_at.timestamp() if snapshot.observed_at else None
            expires_at = snapshot.expires_at.timestamp() if snapshot.expires_at else None
            rest = {state["entity_id"]: state for state in snapshot.states
                    if state["entity_id"] in self._rules} if snapshot.available else {}
            for entity_id in self._rules:
                if (self._task is not task or self._overflow != overflow
                        or not self.uncertain and (self._delta_in_flight or not self._queue.empty())):
                    return
                now = self._clock()
                rest_current = (snapshot.available and observed_at is not None and expires_at is not None
                                and observed_at <= now < expires_at)
                if self.uncertain:
                    if not rest_current:
                        return
                    self._clear_queue()
                    self._evidence.clear()
                    self._notice_ready.clear()
                    self._reconciled_overflow = overflow
                live = self._ha.get_entity_observation(entity_id)
                previous = self._latest.get(entity_id)
                # The REST body may represent an earlier instant than the time
                # it finished downloading. Prefer WS changes seen during the read.
                if live.current(now) and (not rest_current or live.observed_at >= read_started):
                    generation = live.generation if live.source == "ha-ws-entities" else None
                    await self._apply(entity_id, live.to_state(now), _Evidence(
                        live.observed_at, live.expires_at, generation, live.revision,
                    ))
                elif rest_current and (previous is None or previous.observed_at <= read_started):
                    await self._apply(entity_id, rest.get(entity_id), _Evidence(observed_at, expires_at, None))
                else:
                    self._mark_unknown(entity_id)

    def _mark_unknown(self, entity_id: str) -> None:
        for rule in self._rules[entity_id]:
            self._forget_evidence((entity_id, rule.kind))

    def _forget_evidence(self, key: tuple[str, _RuleType]) -> None:
        self._evidence.pop(key, None)
        self._notice_ready.discard(key)

    async def _apply(self, entity_id: str, state: dict | None, evidence: _Evidence) -> None:
        previous = self._latest.get(entity_id)
        if previous is not None and evidence.predates(previous):
            return  # An overlapping REST read must not undo a newer delta.
        self._latest[entity_id] = evidence
        value = state.get("state") if state is not None else None
        if not value or value == "unknown":
            self._mark_unknown(entity_id)
            return
        changed_at = _state_changed_at(state, evidence.observed_at)
        for rule in self._rules[entity_id]:
            now = self._clock()
            if self.uncertain or not evidence.current(now, self._ha.current_state_stream_generation()):
                self._mark_unknown(entity_id)
                return
            await self._check_rule(entity_id, rule, value, changed_at, now, evidence)

    async def _check_rule(self, entity_id: str, rule: _Rule, value: str,
                          changed_at: float | None, now: float, evidence: _Evidence) -> None:
        key = (entity_id, rule.kind)
        self._notice_ready.discard(key)
        priority = AlertPriority.HIGH
        if rule.kind is _RuleType.BATTERY:
            try:
                percent = float(value)
            except ValueError:
                percent = math.nan
            if not 0 <= percent <= 100:
                self._forget_evidence(key)
                return
            active = percent < rule.threshold
            due_at = changed_at
            message = f"Batería baja en {rule.name}: {percent:g}%"
            if percent >= 10:
                priority = AlertPriority.MEDIUM
        else:
            if rule.kind is _RuleType.OPEN:
                if value not in ("on", "off"):
                    self._forget_evidence(key)
                    return
                active = value == "on"
                message = f"{rule.name} sigue abierta"
            else:
                active = value == "unavailable"
                message = f"{rule.name} no está disponible"
            if active and changed_at is None:
                self._forget_evidence(key)
                return
            due_at = changed_at + rule.threshold if active else None

        self._evidence[key] = evidence
        self._last_known[key] = evidence
        at_start = active and key not in self._baseline and (
            due_at is None or due_at <= self._started_at)
        self._baseline.add(key)
        condition = AlertCondition(entity_id, rule.kind.value, reannounce=not at_start, retain_if_history_full=True)
        if not active:
            await self._alerts.resolve_condition(rule.alert_type, condition)
        elif due_at is None or now >= due_at:
            if at_start:
                message = f"{message} (presente al iniciar)"
            await self._alerts.create_alert(rule.alert_type, priority, message, condition=condition, defer_voice=True)
            self._notice_ready.add(key)
            self._notice_wake.set()

    def _notice_current(self, key: tuple[str, _RuleType]) -> bool:
        evidence = self._evidence.get(key)
        return (not self.uncertain and not self._delta_in_flight and self._queue.empty()
                and key in self._notice_ready and evidence is not None
                and evidence.current(self._clock(), self._ha.current_state_stream_generation())
                and (self._turns is None or not self._turns.active))

    async def _deliver_notices(self) -> None:
        while True:
            await self._notice_wake.wait()
            self._notice_wake.clear()
            for entity_id, rules in self._rules.items():
                for rule in rules:
                    key = (entity_id, rule.kind)
                    if key not in self._notice_ready:
                        continue
                    if self._turns is not None:
                        await self._turns.wait_idle()
                    await self._alerts.deliver_deferred_notice(
                        rule.alert_type, AlertCondition(entity_id, rule.kind.value),
                        is_current=lambda: self._notice_current(key),
                    )
