"""Join V4 family lifecycles to the production composition and shutdown stack."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.process_lifecycle import close_logging
from src.memory.output_registration import _require
from src.memory.output_registration_audio import (
    AudioOwner, AudioPublisherBinding, bind_audio_family_publishers, selects_multi_room_audio,
)
from src.memory.output_registration_features import (
    FeaturesFamilyPublisher, bind_features_family_publishers, dashboard_transport_plan,
    features_family_selection, home_integrations_plan, household_features_plan, latency_plan,
)
from src.memory.output_registration_memory import (
    MemoryFamilyPublisher, bind_memory_family_publishers, memory_family_selection, resolve_memory_family_plans,
)
from src.memory.output_registration_offline import (
    ForensicDeclarations, OfflineFamilyPlan, OfflineFamilyPublishers, bind_offline_family_publishers,
    observe_build_forensic_outputs, observe_offline_copies,
    resolve_offline_family_plans,
)

if TYPE_CHECKING:
    from src.memory.output_registration_v2 import OutputRegistrationV2


def _audio_selection(config: dict) -> dict[str, bool]:
    rooms = config.get("rooms", {})
    has_rooms = bool(rooms.get("enabled", False))
    return {
        "audio_input": True,
        "audio_health": selects_multi_room_audio(config),
        "audio_output": True,
        "presence_hardware": has_rooms or bool(config.get("zones", {}).get("enabled", False)),
        "capture": True,
        "collectors": True,
        "voices": True,
    }


def _feature_plans(config: dict, root: Path) -> dict[str, dict]:
    household = ["feature_manager", "timers_runtime", "smart_notifications", "intercom_delivery"]
    if config.get("alerts", {}).get("enabled", True):
        household.append("alerts_runtime")
    ha = config.get("home_assistant", {})
    home = ["ha_client", "home_actions", "home_capabilities", "energy_monitor", "camera_health",
            "calendar_reader", "media_control"]
    if config.get("lists", {}).get("ha_sync_enabled", False):
        home.append("ha_todo_sync")
    dashboard = ["live_event_bus", "live_receipt_store", "browser_delivery"]
    latency = []
    if config.get("dashboard", {}).get("enabled", True):
        dashboard.append("dashboard_http")
        latency.append("health_aggregator")
    monitoring = config.get("monitoring", {})
    if monitoring.get("enabled", True):
        latency.append("latency_sqlite")
    if config.get("llm", {}).get("failover") and config.get("router", {}).get("enabled", True):
        latency.append("llm_metrics")
    db_path = root / monitoring.get("db_path", "./data/latency.db") if "latency_sqlite" in latency else None
    return {
        "household_features": household_features_plan(selected_roles=household),
        "home_integrations": home_integrations_plan(
            selected_roles=home, ha_url=ha.get("url"), selection_configured=bool(ha.get("control_selection_path")),
        ),
        "dashboard_transport": dashboard_transport_plan(selected_roles=dashboard),
        "latency": latency_plan(selected_roles=latency, db_path=db_path),
    }


@dataclass(slots=True)
class CloseObservation:
    """Retain the task that executes an existing async cleanup callback."""

    callback: Callable[[], Awaitable[None]]
    task: asyncio.Task | None = field(default=None, init=False)

    def begin(self) -> asyncio.Task:
        if self.task is None:
            self.task = asyncio.create_task(self.callback())
        return self.task

    async def close(self) -> None:
        await asyncio.shield(self.begin())


OwnerSlot = Callable[[], object]
ReceiptSlot = Callable[[], asyncio.Task | None]


@dataclass(slots=True)
class RuntimeOutputPublishers:
    """Bind the four families using the same configuration passed to the graph."""

    config: dict
    root: Path
    admission_root: Path
    runtime_root: Path | None = None
    _audio_selection: dict[str, bool] = field(default_factory=dict, init=False)
    _features_plans: dict[str, dict] = field(default_factory=dict, init=False)
    _memory_plans: dict[str, dict] = field(default_factory=dict, init=False)
    _offline_plans: dict[str, OfflineFamilyPlan] = field(default_factory=dict, init=False)
    audio: dict[str, AudioPublisherBinding] = field(default_factory=dict, init=False)
    _memory: dict[str, MemoryFamilyPublisher] = field(default_factory=dict, init=False)
    _features: dict[str, FeaturesFamilyPublisher] = field(default_factory=dict, init=False)
    offline: OfflineFamilyPublishers | None = field(default=None, init=False)
    _slots: dict[str, dict[str, OwnerSlot]] = field(default_factory=dict, init=False)
    _receipts: dict[tuple[str, str], ReceiptSlot] = field(default_factory=dict, init=False)
    _offline_slots: dict[str, Callable[[], tuple[object, ...]]] = field(default_factory=dict, init=False)
    _logging_close: CloseObservation | None = field(default=None, init=False)

    def select(self, cloud_withdrawn: bool) -> dict[str, bool]:
        """Resolve epoch selection after V4 has recovered the boot consent marker."""
        self._audio_selection = _audio_selection(self.config)
        self._features_plans = _feature_plans(self.config, self.root)
        self._memory_plans = resolve_memory_family_plans(self.config, self.root, cloud_withdrawn=cloud_withdrawn)
        manifests = (self.runtime_root / "release.json",) if self.runtime_root is not None else ()
        self._offline_plans = resolve_offline_family_plans(
            self.config, self.root, admission_root=self.admission_root,
            forensic=ForensicDeclarations(code_manifests=manifests),
        )
        return (self._audio_selection | features_family_selection(self._features_plans)
                | memory_family_selection(self._memory_plans)
                | {name: plan.selected for name, plan in self._offline_plans.items()})

    async def bind(self, coordinator: OutputRegistrationV2, shutdown: AsyncExitStack) -> None:
        """Write all intents before the application constructs the selected owners."""
        self.audio = await bind_audio_family_publishers(coordinator, self.config, selection=self._audio_selection)
        self._memory = await bind_memory_family_publishers(coordinator, self._memory_plans)
        self._features = await bind_features_family_publishers(coordinator, self._features_plans)
        self.offline = await bind_offline_family_publishers(coordinator, self._offline_plans)
        shutdown.push_async_exit(self._after_close)
        self._logging_close = CloseObservation(
            lambda: asyncio.to_thread(coordinator.admission._prepare_receipt, fallback=close_logging),
        )
        await self.handoff("logging", "bootstrap_logging", logging.getLogger,
                           receipt=lambda: self._logging_close.task)
        copies = await observe_offline_copies(self._offline_plans["offline_copies"], coordinator)
        await self.bind_offline("offline_copies", lambda: (copies,))
        forensic = self._offline_plans["build_forensic_outputs"]
        if forensic.selected:
            inventory = await observe_build_forensic_outputs(forensic)
            await self.bind_offline("build_forensic_outputs", lambda: (inventory,))

    def _role(self, producer: str, role: str) -> tuple[MemoryFamilyPublisher | FeaturesFamilyPublisher | None, str]:
        if producer in self._memory:
            return self._memory[producer], self._memory_plans[producer]["role_modes"][role]
        if producer in self._features and self._features_plans[producer]["enabled"][role]:
            return self._features[producer], "active"
        return None, "disabled"

    async def opening(self, producer: str, role: str, slot: OwnerSlot, *,
                      receipt: ReceiptSlot | None = None, lazy: bool = False) -> None:
        """Attach a live slot and record opening before its original work."""
        binder, mode = self._role(producer, role)
        if binder is None or mode == "disabled":
            return
        _require(not lazy or mode == "lazy", "publisher_lazy_role")
        owner = slot()
        if owner is None:
            return  # Composition still checks the original selected role set.
        await binder.attach(role, owner)
        self._slots.setdefault(producer, {})[role] = slot
        if receipt is not None:
            self._receipts[producer, role] = receipt
        if not lazy:
            await binder.opening(role, owner)

    async def opened(self, producer: str, role: str) -> None:
        """Confirm that the selected owner finished opening."""
        binder, mode = self._role(producer, role)
        slot = self._slots.get(producer, {}).get(role)
        if binder is not None and mode != "disabled" and slot is not None:
            await binder.opened(role, slot())

    async def handoff(self, producer: str, role: str, slot: OwnerSlot, *,
                      receipt: ReceiptSlot | None = None, lazy: bool = False) -> None:
        """Register an owner whose native work is deferred to graph execution."""
        await self.opening(producer, role, slot, receipt=receipt, lazy=lazy)
        if not lazy:
            await self.opened(producer, role)

    async def handoff_many(
        self, producer: str, slots: Mapping[str, OwnerSlot], *, receipt: ReceiptSlot | None = None,
    ) -> None:
        """Associate the roles reached by the same original shutdown owner."""
        for role, slot in slots.items():
            await self.handoff(producer, role, slot, receipt=receipt)

    async def bind_audio(self, producer: str, read_graph: Callable[[], Mapping[str, tuple[AudioOwner, ...]]]) -> None:
        if producer in self.audio:
            binder = self.audio[producer]
            await binder.attached(read_graph)
            await binder.opening()

    async def bind_offline(self, producer: str, owners: Callable[[], tuple[object, ...]], *,
                           receipt: ReceiptSlot | None = None) -> None:
        if self._offline_plans[producer].selected:
            await self.offline.attach(producer, *owners())
            await self.offline.opening(producer, *owners())
            self._offline_slots[producer] = owners
            if receipt is not None:
                self._receipts[producer, "owner"] = receipt

    def _owners(self, producer: str) -> dict[str, object]:
        return {role: slot() for role, slot in self._slots.get(producer, {}).items()}

    def require_composed(self) -> None:
        """Reconcile every selected role with its nominal graph projection."""
        for producer, binder in (self._memory | self._features).items():
            _require(binder.matches(self._owners(producer)), f"publisher_owner_graph:{producer}")
        for binder in self.audio.values():
            binder.require_current()
        for producer, owners in self._offline_slots.items():
            _require(self.offline.matches_attached_owners(producer, *owners()), f"offline_owner_graph:{producer}")

    async def _after_close(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            return
        await self._logging_close.close()
        for slot in self._receipts.values():
            task = slot()
            _require(isinstance(task, asyncio.Task) and task.done() and not task.cancelled()
                     and task.exception() is None, "publisher_owner_drain_unconfirmed")
        self.require_composed()
        for binder in self.audio.values():
            await binder.drained()
        for producer, binder in (self._memory | self._features).items():
            await binder.drained(self._owners(producer))
        for producer, owners in self._offline_slots.items():
            await self.offline.drained(producer, *owners())
