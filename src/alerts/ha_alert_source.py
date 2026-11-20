"""Read explicitly selected Home Assistant states and maintain alert episodes."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, TYPE_CHECKING

from src.alerts.alert_manager import AlertCondition, AlertManager, AlertPriority, AlertType
from src.alerts.device_alerts import BATTERY_LOW, OFFLINE
from src.alerts.security_alerts import DOOR_OPEN_UNEXPECTED

if TYPE_CHECKING:
    from src.conversation.command_turns import CommandTurns
    from src.home_assistant.ha_client import HAStatesSnapshot
    from src.home_assistant.state_projection import EntityObservation

logger = logging.getLogger(__name__)


class _HAStateReader(Protocol):
    def register_state_callback(self, callback: Callable[[str, dict | None], None]) -> None: ...
    def unregister_state_callback(self, callback: Callable[[str, dict | None], None]) -> None: ...
    def get_entity_observation(self, entity_id: str) -> EntityObservation: ...
    async def get_state_snapshot(self) -> HAStatesSnapshot: ...


class _RuleType(StrEnum):
    OPEN = DOOR_OPEN_UNEXPECTED
    BATTERY = BATTERY_LOW
    UNAVAILABLE = OFFLINE


@dataclass(frozen=True)
class _Rule:
    kind: _RuleType
    name: str
    threshold: float

    @property
    def alert_type(self) -> AlertType:
        return AlertType.SECURITY if self.kind is _RuleType.OPEN else AlertType.DEVICE


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
        self._queue: asyncio.Queue[tuple[str, dict | None]] = asyncio.Queue(maxsize=maxsize)
        self._evaluate = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._turns: CommandTurns | None = None
        self._overflow = self._reconciled_overflow = 0
        self._started_at: float | None = None
        self._latest: dict[str, float] = {}
        self._fresh_until: dict[tuple[str, _RuleType], float] = {}
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
        known = (sum(now < expires_at for expires_at in self._fresh_until.values())
                 if self.running and not self.uncertain else 0)
        return {"running": self.running, "uncertain": self.uncertain, "overflow": self._overflow,
                "known": known, "unknown": count - known}

    async def start(self, command_turns: CommandTurns | None = None) -> None:
        """Subscribe and start the worker without waiting on HA or voice output."""
        if self._task is not None or not self.configured:
            return
        if self._started_at is None:
            self._started_at = self._clock()
        self._turns = command_turns
        self._ha.register_state_callback(self._enqueue)
        self._task = asyncio.create_task(self._consume(), name="ha-alert-deltas")

    async def stop(self) -> None:
        """Unsubscribe and retire the exact consumer before HA and voice close."""
        task = self._task
        if task is None:
            return
        self._ha.unregister_state_callback(self._enqueue)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None
        self._turns = None
        self._clear_queue()
        self._fresh_until.clear()

    def _enqueue(self, entity_id: str, state: dict | None) -> None:
        # Called on the HA event loop while it owns its state projection lock.
        if entity_id in self._rules:
            try:
                self._queue.put_nowait((entity_id, state))
            except asyncio.QueueFull:
                self._overflow += 1

    def _clear_queue(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    async def _consume(self) -> None:
        while True:
            entity_id, state = await self._queue.get()
            reconciled = self._reconciled_overflow
            try:
                async with self._evaluate:
                    if self._turns is not None:
                        await self._turns.wait_idle()
                    if self.uncertain or reconciled != self._reconciled_overflow:
                        continue
                    now = self._clock()
                    observation = self._ha.get_entity_observation(entity_id)
                    if not observation.current(now):
                        self._mark_unknown(entity_id)
                    elif state is None or (observation.exists and state == observation.state):
                        await self._apply(entity_id, state, observation.observed_at,
                                          observation.expires_at)
            except Exception as error:
                self._mark_unknown(entity_id)
                logger.warning("HA alert delta evaluation failed: %s", type(error).__name__)
            finally:
                self._queue.task_done()

    async def reconcile(self) -> None:
        """Read one REST snapshot and reconcile it with fresher WS evidence."""
        if not self.running:
            return
        task, overflow = self._task, self._overflow
        read_started = self._clock()
        snapshot = await self._ha.get_state_snapshot()
        async with self._evaluate:
            if self._task is not task or self._overflow != overflow:
                return
            observed_at = snapshot.observed_at.timestamp() if snapshot.observed_at else None
            expires_at = snapshot.expires_at.timestamp() if snapshot.expires_at else None
            rest = {state["entity_id"]: state for state in snapshot.states
                    if state["entity_id"] in self._rules} if snapshot.available else {}
            for entity_id in self._rules:
                if self._turns is not None:
                    await self._turns.wait_idle()
                if self._task is not task or self._overflow != overflow:
                    return
                now = self._clock()
                rest_current = (snapshot.available and observed_at is not None and expires_at is not None
                                and observed_at <= now < expires_at)
                if self.uncertain:
                    if not rest_current:
                        return
                    self._clear_queue()
                    self._fresh_until.clear()
                    self._reconciled_overflow = overflow
                live = self._ha.get_entity_observation(entity_id)
                # The REST body may represent an earlier instant than the time
                # it finished downloading. Prefer WS changes seen during the read.
                if live.current(now) and (not rest_current or live.observed_at >= read_started):
                    await self._apply(entity_id, live.state if live.exists else None,
                                      live.observed_at, live.expires_at)
                elif rest_current and self._latest.get(entity_id, 0) <= read_started:
                    await self._apply(entity_id, rest.get(entity_id), observed_at, expires_at)
                else:
                    self._mark_unknown(entity_id)

    def _mark_unknown(self, entity_id: str) -> None:
        for rule in self._rules[entity_id]:
            self._fresh_until.pop((entity_id, rule.kind), None)

    async def _apply(self, entity_id: str, state: dict | None,
                     observed_at: float, expires_at: float) -> None:
        if observed_at < self._latest.get(entity_id, 0):
            return  # An overlapping REST read must not undo a newer delta.
        self._latest[entity_id] = observed_at
        value = state.get("state") if state is not None else None
        if not value or value == "unknown":
            self._mark_unknown(entity_id)
            return
        changed_at = _state_changed_at(state, observed_at)
        for rule in self._rules[entity_id]:
            if self._turns is not None:
                await self._turns.wait_idle()
            now = self._clock()
            if now >= expires_at or self.uncertain:
                self._mark_unknown(entity_id)
                return
            await self._check_rule(entity_id, rule, value, changed_at, now, expires_at)

    async def _check_rule(self, entity_id: str, rule: _Rule, value: str,
                          changed_at: float | None, now: float, expires_at: float) -> None:
        key = (entity_id, rule.kind)
        priority = AlertPriority.HIGH
        if rule.kind is _RuleType.BATTERY:
            try:
                percent = float(value)
            except ValueError:
                percent = math.nan
            if not 0 <= percent <= 100:
                self._fresh_until.pop(key, None)
                return
            active = percent < rule.threshold
            due_at = changed_at
            message = f"Batería baja en {rule.name}: {percent:g}%"
            if percent >= 10:
                priority = AlertPriority.MEDIUM
        else:
            if rule.kind is _RuleType.OPEN:
                if value not in ("on", "off"):
                    self._fresh_until.pop(key, None)
                    return
                active = value == "on"
                message = f"{rule.name} sigue abierta"
            else:
                active = value == "unavailable"
                message = f"{rule.name} no está disponible"
            if active and changed_at is None:
                self._fresh_until.pop(key, None)
                return
            due_at = changed_at + rule.threshold if active else None

        self._fresh_until[key] = expires_at
        at_start = active and key not in self._baseline and (
            due_at is None or due_at <= self._started_at)
        self._baseline.add(key)
        condition = AlertCondition(entity_id, rule.kind.value, reannounce=not at_start)
        if not active:
            await self._alerts.resolve_condition(rule.alert_type, condition)
        elif due_at is None or now >= due_at:
            if at_start:
                message = f"{message} (presente al iniciar)"
            await self._alerts.create_alert(rule.alert_type, priority, message, condition=condition)
